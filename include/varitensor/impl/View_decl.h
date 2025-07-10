/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * Licence, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef VARITENSOR_VIEW_H
#define VARITENSOR_VIEW_H

#include <map>
#include <vector>

#include "common.h"
#include "ExpressionIteratorBase.h"

namespace varitensor {
namespace impl {

struct WidthInfo {
    /** Data struct to store the memory width of a single increment along a dimension and the total
     * memory width of the entire dimension (i.e. width * size).
     */
    size_t width{0};
    size_t total{0};
    WidthInfo& operator+=(WidthInfo&& other);
};

template<bool is_const=false>
class ViewIterator: public ExpressionIteratorBase {
public:
// =================================================================================================
//                                                                       ctor / dtor / copy / move |
// =================================================================================================

    ViewIterator() = default;

    ViewIterator(const Tensor* target, double* data_ptr, const Dimensions& dimensions):
        m_target{target},
        m_data_ptr{data_ptr}
    {
        std::unordered_map<int, int> counts;
        for (auto& dimension: dimensions) counts[dimension.index.id()] += 1;

        for (auto& dimension: dimensions) {
            deny(counts[dimension.index.id()] > 2,
                       "Indices in contraction expressions cannot appear more than twice!");
            if (counts[dimension.index.id()] == 2) {
                m_repeated.push_back(dimension);
                m_repeated_positions.push_back(0);
                counts[dimension.index.id()] = -1;
            }
            else if (counts[dimension.index.id()] == 1) {
                m_dimensions.push_back(dimension);
                m_size *= dimension.size();
                m_positions.push_back(0);
            }

            m_widths[dimension.index.id()] += {
                dimension.width,
                dimension.width * (dimension.size() -1)
            };
        }

        if (!m_dimensions.empty()) {
            m_cached_id = m_dimensions.front().index.id();
            m_cached_info = m_widths[m_cached_id];
        }
    }

    ViewIterator(const ViewIterator& other) = default;
    ViewIterator& operator=(const ViewIterator& other) = default;
    ViewIterator(ViewIterator&& other) noexcept = default;
    ViewIterator& operator=(ViewIterator&& other) = default;

// =================================================================================================
//                                                                              forwards iteration |
// =================================================================================================

    ViewIterator& operator++() {
        if (!increment_positions(m_positions, m_dimensions, *this)) {
            m_data_ptr = nullptr;
        }
        return *this;
    }

    ViewIterator operator++(int) {
        ViewIterator copy = *this;
        ++*this;
        return copy;
    }

    double& operator*() const requires(!is_const) {
        strict_deny(is_contracted(), "Cannot dereference non-const contracted iterator! Use std::as_const or cbegin/cend for const iteration.");
        return *m_data_ptr;
    }

    double operator*() const requires(is_const) {
        return deref();
    }

    template<typename T>
    requires std::is_same_v<T, ViewIterator<>> || std::is_same_v<T, ViewIterator<true>>
    bool operator==(const T& other) const {
        return m_data_ptr == other.m_data_ptr;
    }

// =================================================================================================
//                                                                                     information |
// =================================================================================================

    [[nodiscard]] bool is_contracted() const { // used to enforce the fact that contractions can only be r-values
        return !m_repeated.empty();
    }

    bool is_metric() const;

    [[nodiscard]] bool finished() const {
        return m_data_ptr == nullptr;
    }

// =================================================================================================
//                                                                                 index iteration |
// =================================================================================================
/* Note that deref has to be const for this class to maintain a sensible interface, but this
 * requires increment and reset to pretend they are const, even though they're not. This is fine in
 * deref, because it always returns the class to the state it was in before, but these functions are
 * not really const if called on their own.
 */

    void increment(const int index_id) const {
        if (index_id == m_cached_id) m_data_ptr += m_cached_info.width;
        else m_data_ptr += m_widths[index_id].width;
    }

    void reset(const int index_id) const {
        if (index_id == m_cached_id) m_data_ptr -= m_cached_info.total;
        else m_data_ptr -= m_widths[index_id].total;
    }

    [[nodiscard]] double deref() const { // only pseudo-const
        double sum = 0;

        do sum += *m_data_ptr;
        while (increment_positions(m_repeated_positions, m_repeated, *this));

        return sum;
    }

private:
    friend class ViewIterator<!is_const>;

// =================================================================================================
//                                                                                    data members |
// =================================================================================================

    const Tensor* m_target{nullptr};
    mutable double* m_data_ptr{nullptr};

    mutable std::map<int, WidthInfo> m_widths;

    // profiling has indicated that map lookup can be a bottleneck: these caches are used to reduce this
    int m_cached_id{-1};
    WidthInfo m_cached_info;

    // for index contraction
    Dimensions m_repeated;
    mutable std::vector<int> m_repeated_positions;
};

} // namespace impl



class View {
/**
 * Represents a view on a tensor using provided indices. Repeated indices
 * are summed over (contraction). Can be used as a forwards iterator.
 */
public:
    explicit View(const Tensor& target); // a plain view on a tensor
    View(const Tensor& target, double* data_ptr, impl::Dimensions dimensions); // an offset view

    void operator=(const Tensor& other) &&; // NOLINT - allows T[i, 2] = U
//    void operator=(const View&& other) &&; // NOLINT - allows T[i, 2] = U[i, 2]

    bool operator==(const View& other) const;

    using iterator = impl::ViewIterator<>;
    using const_iterator = impl::ViewIterator<true>;

    [[nodiscard]] iterator begin();
    [[nodiscard]] iterator end();
    [[nodiscard]] const_iterator begin() const;
    [[nodiscard]] const_iterator end() const;
    [[nodiscard]] const_iterator cbegin() const;
    [[nodiscard]] const_iterator cend() const;
    [[nodiscard]] impl::ExpressionIterator vbegin() const;


private:
    const Tensor& m_target;
    double* m_data_ptr; // allows us to store an offset
    impl::Dimensions m_dimensions;
};

} // namespace varitensor

#endif
