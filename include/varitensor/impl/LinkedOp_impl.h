/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * Licence, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef VARITENSOR_LINKEDOP_IMPL_H
#define VARITENSOR_LINKEDOP_IMPL_H

#include "increment_positions.h"
#include "LinkedOp_decl.h"
#include "visitors.h"

namespace varitensor::impl {

// =================================================================================================
//                                                                                   iterator ctor |
// =================================================================================================

inline LinkedOpIterator::LinkedOpIterator(
    const double modifier,
    const Expressions& sub_expressions,
    const std::vector<Operation>* signs,
    const bool end /* = false */
):
    m_modifier {modifier},
    m_signs{signs},
    m_end{end}
{
    if (sub_expressions.empty() || m_end) return;

    for (auto& expression: sub_expressions) m_sub_iterators.emplace_back(std::visit(VBegin, expression));
    m_dimensions = std::visit(GetDimensions, m_sub_iterators[0]);
    m_size = std::visit(GetSize, m_sub_iterators[0]);

    if constexpr (VARITENSOR_VALIDATION_ON) {
        for (size_t i = 1; i < m_sub_iterators.size(); ++i) { // start at 1 as we've already done 0
            deny(std::visit(GetSize, m_sub_iterators[i]) != m_size,
                 "Attempt to add or subtract tensors of different sizes!");
            for (auto& dimension1: std::visit(GetDimensions, m_sub_iterators[i])) {
                for (auto& dimension2: m_dimensions) {
                    if (dimension1.index == dimension2.index) {
                        goto INDEX_MATCH_FOUND; // for-else is a valid use of goto
                    }
                }
                throw std::logic_error("Attempt to add or subtract tensors with non-matching indices!");
                INDEX_MATCH_FOUND:;
            }
        }
    }

    m_positions = std::vector(m_dimensions.size(), 0);
}

// =================================================================================================
//                                                                              forwards iteration |
// =================================================================================================

inline LinkedOpIterator& LinkedOpIterator::operator++() {
    m_end = !increment_positions(m_positions, m_dimensions, *this);
    return *this;
}

inline LinkedOpIterator LinkedOpIterator::operator++(int) {
    LinkedOpIterator copy = *this;
    ++*this;
    return copy;
}

inline double LinkedOpIterator::operator*() const {
    return deref();
}

inline bool LinkedOpIterator::operator==(const LinkedOpIterator& other) const {
    if (m_end != other.m_end) return false;
    if (m_end) return true;
    return m_modifier == other.m_modifier && m_positions == other.m_positions; // good enough for normal iteration
}

// =================================================================================================
//                                                                                 index iteration |
// =================================================================================================

inline void LinkedOpIterator::increment(const int index_id) const {
    for (auto& iterator: m_sub_iterators) std::visit(Increment(index_id), iterator);
}

inline void LinkedOpIterator::reset(const int index_id) const {
    for (auto& iterator: m_sub_iterators) std::visit(Reset(index_id), iterator);
}

inline double LinkedOpIterator::deref() const {
    double sum = 0;
    for (size_t i=0; i < m_sub_iterators.size(); ++i) {
        sum += std::visit(Deref, m_sub_iterators[i]) * static_cast<double> ((*m_signs)[i]);
    }
    return sum + m_modifier;
}

// =================================================================================================
//                                                                                     information |
// =================================================================================================

inline bool LinkedOpIterator::is_metric() {
    return false;
}

inline LinkedOpIterator LinkedOp::begin() const {
    return iterator{m_modifier, m_sub_expressions, &m_signs};
}

inline LinkedOpIterator LinkedOp::end() const {
    return iterator{m_modifier, m_sub_expressions, &m_signs, true};
}

inline ExpressionIterator LinkedOp::vbegin() const {
    return begin();
}

// =================================================================================================
//                                                                                         helpers |
// =================================================================================================

inline void LinkedOp::add_element(const Tensor& tensor, const Operation sign) {
    m_sub_expressions.emplace_back(View{tensor});
    m_signs.push_back(sign);
}

inline void LinkedOp::add_element(const LinkedOp& summation, const Operation sign) {
    m_sub_expressions.emplace_back(summation);
    m_signs.push_back(sign);
}

inline void LinkedOp::add_element(const SummedOp& product, const Operation sign) {
    m_sub_expressions.emplace_back(product);
    m_signs.push_back(sign);
}

inline void LinkedOp::add_element(const View& view, const Operation sign) {
    m_sub_expressions.emplace_back(view);
    m_signs.push_back(sign);
}

inline void LinkedOp::add_element(const double value, const Operation sign) {
    m_modifier += sign == ADD ? value : -value;
}

} // namespace varitensor::impl

#endif
