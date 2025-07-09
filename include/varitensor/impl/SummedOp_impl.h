/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * Licence, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef VARITENSOR_SUMMED_OP_IMPL_H
#define VARITENSOR_SUMMED_OP_IMPL_H

#include <map>
#include <ranges>
#include <vector>

#include "deny.h"
#include "SummedOp_decl.h"
#include "View_decl.h"

namespace varitensor::impl {

// =================================================================================================
//                                                                                   Couple struct |
// =================================================================================================

inline void Couple::add(const int dimension, const bool is_metric) {
    soft_deny(dims[0] != -1 && dims[1] != -1,
              "Indices in multiplication expression cannot appear more than twice!");
    const auto target = dims[0] == -1 ? 0 : 1;
    dims[target] = dimension;
    if (!is_metric) other_index = dimension;
}

inline void Couple::clear() {
    dims[0] = -1;
    dims[1] = -1;
    metric_index = -1;
}

inline bool Couple::has() const {
    return dims[0] != -1;
}

inline bool Couple::is_repeated() const {
    return dims[1] != -1;
}

inline bool Couple::is_metric() const {
    return metric_index != -1;
}

// =================================================================================================
//                                                                                   iterator ctor |
// =================================================================================================

inline SummedOpIterator::SummedOpIterator(const double modifier, const Expressions& sub_expressions, const bool end/* = false */):
    m_modifier{modifier},
    m_end{end}
{
    if (m_end || sub_expressions.empty()) return;

    for (auto& expression: sub_expressions) m_sub_iterators.emplace_back(std::visit(VBegin, expression));

    // get the naive dimensions whilst recording metric and repetition information that will be useful later
    Dimensions total_dimensions;
    std::map<int, Couple> partners;
    int i{0};

    for (auto & iterator : m_sub_iterators) {
        auto dimensions = std::visit(GetDimensions, iterator);

        for (size_t k = 0; k < dimensions.size(); ++k) {
            total_dimensions.emplace_back(dimensions[k]);

            if (std::visit([](auto& expression) {return expression.is_metric();}, iterator)) {
                partners[dimensions[k].index.id()].add(i, true);
                const int other = k == 0 ? i+1 : i-1; // metric tensors only have 2 indices
                partners[dimensions[k].index.id()].metric_index = other;
            }
            else {
                partners[dimensions[k].index.id()].add(i, false);
            }

            ++i;
        }
    }

    // handle any metric indices
    for (auto& couple: partners | std::views::values) {
        if (couple.is_repeated() && couple.is_metric()) {
            std::swap(total_dimensions[couple.metric_index], total_dimensions[couple.other_index]);
        }
    }

    for (auto& dimension: total_dimensions) {
        if (auto& couple = partners[dimension.index.id()]; couple.is_repeated()) { // repeated index
            m_repeated.emplace_back(dimension);
            couple.clear();
        }
        else if (couple.has()) { // non-repeated index
            m_dimensions.push_back(dimension);
            m_dimensions.back().width = m_size;
            m_size *= dimension.size();
        }
    }

    m_positions = std::vector(m_dimensions.size(), 0);
    m_repeated_positions = std::vector(m_repeated.size(), 0);
}

// =================================================================================================
//                                                                              forwards iteration |
// =================================================================================================

inline SummedOpIterator& SummedOpIterator::operator++() {
    m_end = !increment_positions(m_positions, m_dimensions, *this);
    return *this;
}

inline SummedOpIterator SummedOpIterator::operator++(int) {
    SummedOpIterator copy = *this;
    ++*this;
    return copy;
}

inline double SummedOpIterator::operator*() const {
    return deref();
}

inline bool SummedOpIterator::operator==(const SummedOpIterator& other) const {
    if (m_end != other.m_end) return false;
    if (m_end) return true;
    return m_modifier == other.m_modifier && m_positions == other.m_positions; // good enough for normal iteration
}

// =================================================================================================
//                                                                                 index iteration |
// =================================================================================================

inline void SummedOpIterator::increment(const int index_id) const {
    for (auto& iterator: m_sub_iterators) std::visit(Increment(index_id), iterator);
}

inline void SummedOpIterator::reset(const int index_id) const {
    for (auto& iterator: m_sub_iterators) std::visit(Reset(index_id), iterator);
}

inline double SummedOpIterator::deref() const {
    double sum = 0;

    do {
        double product = 1;
        for (auto& iterator: m_sub_iterators) product *= std::visit(Deref, iterator);
        sum += product;
    }
    while (increment_positions(m_repeated_positions, m_repeated, *this));

    return sum * m_modifier;
}

// =================================================================================================
//                                                                                     information |
// =================================================================================================

inline bool SummedOpIterator::is_metric() {
    return false;
}

// =================================================================================================
//                                                                              SummedOp iteration |
// =================================================================================================

inline SummedOp::iterator SummedOp::begin() const {
    return iterator{m_modifier, m_sub_expressions};
}

inline SummedOp::iterator SummedOp::end() const {
    return iterator{m_modifier, m_sub_expressions, true};
}

inline ExpressionIterator SummedOp::vbegin() const {
    return begin();
}

// =================================================================================================
//                                                                                SummedOp helpers |
// =================================================================================================

inline void SummedOp::add_element(const Tensor& tensor, Operation) {
    m_sub_expressions.emplace_back(View{tensor});
}

inline void SummedOp::add_element(const LinkedOp& linked_op, Operation) {
    m_sub_expressions.emplace_back(linked_op);
}

inline void SummedOp::add_element(const SummedOp& summed_op, Operation) {
    m_modifier *= summed_op.m_modifier;
    for (auto& expression: summed_op.m_sub_expressions) m_sub_expressions.emplace_back(expression);
}

inline void SummedOp::add_element(const View& view, Operation) {
    m_sub_expressions.emplace_back(view);
}

inline void SummedOp::add_element(const double& value, const Operation operation) {
    m_modifier *= operation == MUL ? value : 1/value;
}

} // namespace varitensor::impl

#include "View_impl.h"

#endif
