/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * Licence, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "varitensor/impl/View.h"
#include "varitensor/Tensor.h"
#include "varitensor/impl/deny.h"

namespace varitensor {

namespace impl {

inline WidthInfo& WidthInfo::operator+=(WidthInfo&& other) {
    width += other.width;
    total += other.total;
    return *this;
}

template<>
bool ViewIterator<>::is_metric() const {
    return m_target->is_metric();
}

template<>
bool ViewIterator<true>::is_metric() const {
    return m_target->is_metric();
}

} // namespace impl

// =================================================================================================
//                                                                                      View ctors |
// =================================================================================================

View::View(const Tensor& target):
    m_target{target},
    m_data_ptr{target.m_data.get()},
    m_dimensions{m_target.m_dimensions}
{
    for (size_t i = 0; i < m_dimensions.size(); ++i) {
        impl::deny(m_dimensions[i].size() > m_target.m_dimensions[i].index.size(),
                        "Index size mismatch");
        impl::deny(m_dimensions[i].variance != m_target.m_dimensions[i].variance,
                        "Index variance mismatch");
    }
}

View::View(const Tensor& target, double* const data_ptr, impl::Dimensions dimensions):
    m_target{target},
    m_data_ptr{data_ptr},
    m_dimensions{std::move(dimensions)}
{} // we only call this from Tensor::operator[], so no need to validate the dimensions

// =================================================================================================
//                                                                                  View operators |
// =================================================================================================

void View::operator=(const Tensor& other) && { // NOLINT - allows T[i, 2] = U
    const View other_view{other};

    impl::deny(m_dimensions.size() != other.m_dimensions.size(),
                    "Cannot assign to View with different dimensions");
    for (size_t i = 0; i < m_dimensions.size(); ++i) {
        impl::deny(m_dimensions[i].index.size() != other.m_dimensions[i].index.size(),
                        "Index size mismatch when assigning view");
        impl::deny(m_dimensions[i].variance != other.m_dimensions[i].variance,
                        "Variance mismatch when assigning view");
    }

    auto iter1 = begin();
    auto iter2 = other_view.begin();
    impl::deny(iter1.is_contracted(), "Cannot assign to a contraction expression");

    for (auto end=this->end(); iter1 != end; ++iter1, ++iter2) *iter1 = *iter2;
}

bool View::operator==(const View& other) const {
    auto iter1 = begin();
    auto iter2 = other.begin();
    for (auto end = this->end(); iter1 != end;  ++iter1, ++iter2) {
        if (*iter1 != *iter2) return false;
    }
    return iter2.finished();
}

// =================================================================================================
//                                                                                  View iteration |
// =================================================================================================

View::iterator View::begin() {
    return iterator{&m_target, m_data_ptr, m_dimensions};
}

View::iterator View::end() {
    return iterator{&m_target, nullptr, {}};
}

View::const_iterator View::begin() const {
    return const_iterator{&m_target, m_data_ptr, m_dimensions};
}

 View::const_iterator View::end() const {
    return const_iterator{&m_target, nullptr, {}};
}

View::const_iterator View::cbegin() const {
    return const_iterator{&m_target, m_data_ptr, m_dimensions};
}

View::const_iterator View::cend() const {
    return const_iterator{&m_target, nullptr, {}};
}

impl::ExpressionIterator View::vbegin() const {
    return begin();
}

[[nodiscard]] double* View::data() const {
    return m_data_ptr;
}

[[nodiscard]] bool View::is_scalar() const {
    return m_dimensions.empty();
}

[[nodiscard]] double View::get_scalar() const {
    return *m_data_ptr;
}

void View::populate(Tensor& tensor, const bool allocate/* = true */) const {
    auto iter = cbegin();

    if (allocate) {
        tensor.m_dimensions = iter.dimensions();
        tensor.m_size = iter.size();
        tensor.m_data = impl::allocate(tensor.m_size);
    }

    if (!iter.is_contracted() && iter.is_contiguous()) {
        impl::copy(tensor.m_data.get(), m_data_ptr, tensor.m_size);
    }
    else {
        double* running_ptr = tensor.m_data.get();
        for (const auto end=cend(); iter != end; ++iter, ++running_ptr) {
            *running_ptr = *iter;
        }
    }
}

} // namespace varitensor