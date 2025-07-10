/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * Licence, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef VARITENSOR_VIEW_IMPL_H
#define VARITENSOR_VIEW_IMPL_H

#include "deny.h"
#include "View_decl.h"
#include "varitensor/Tensor.h"

namespace varitensor {

namespace impl {
inline WidthInfo& WidthInfo::operator+=(WidthInfo&& other) {
    width += other.width;
    total += other.total;
    return *this;
}

// =================================================================================================
//                                                                       ViewIterator::is_metric() |
// =================================================================================================

template<>
inline bool ViewIterator<>::is_metric() const {
    return m_target->is_metric();
}

template<>
inline bool ViewIterator<true>::is_metric() const {
    return m_target->is_metric();
}

} // namespace impl

// =================================================================================================
//                                                                                      View ctors |
// =================================================================================================

inline View::View(const Tensor& target):
    m_target{target},
    m_data_ptr{target.m_data},
    m_dimensions{m_target.m_dimensions}
{
    for (size_t i = 0; i < m_dimensions.size(); ++i) {
        impl::deny(m_dimensions[i].size() > m_target.m_dimensions[i].index.size(),
                   "Index size mismatch!");
        impl::deny(m_dimensions[i].variance != m_target.m_dimensions[i].variance,
                   "Index variance mismatch!");
    }
}

inline View::View(const Tensor& target, double* data_ptr, impl::Dimensions dimensions):
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
               "Cannot assign to View with different dimensions!");
    for (size_t i = 0; i < m_dimensions.size(); ++i) {
        impl::deny(m_dimensions[i].index.size() != other.m_dimensions[i].index.size(),
                   "Index size mismatch when assigning view!");
        impl::deny(m_dimensions[i].variance != other.m_dimensions[i].variance,
                   "Variance mismatch when assigning view!");
    }

    auto iter1 = begin();
    auto iter2 = other_view.begin();
    impl::deny(iter1.is_contracted(), "Cannot assign to a tensor contraction!");

    for (auto end=this->end(); iter1 != end; ++iter1, ++iter2) *iter1 = *iter2;
}

inline bool View::operator==(const View& other) const {
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

inline View::iterator View::begin() {
    return iterator{&m_target, m_data_ptr, m_dimensions};
}

inline View::iterator View::end() {
    return iterator{&m_target, nullptr, {}};
}

inline View::const_iterator View::begin() const {
    return const_iterator{&m_target, m_data_ptr, m_dimensions};
}

inline View::const_iterator View::end() const {
    return const_iterator{&m_target, nullptr, {}};
}

inline View::const_iterator View::cbegin() const {
    return const_iterator{&m_target, m_data_ptr, m_dimensions};
}

inline View::const_iterator View::cend() const {
    return const_iterator{&m_target, nullptr, {}};
}

inline impl::ExpressionIterator View::vbegin() const {
    return begin();
}

} // namespace varitensor

#endif
