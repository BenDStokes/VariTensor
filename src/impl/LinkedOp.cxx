/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * Licence, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "varitensor/impl/LinkedOp.h"

#include "varitensor/Tensor.h"
#include "varitensor/impl/bulk.h"
#include "varitensor/impl/deny.h"
#include "varitensor/impl/Preparatory.h"
#include "varitensor/impl/ProductOp.h"
#include "varitensor/impl/visitors.h"

namespace varitensor::impl {

[[nodiscard]] double* LinkedOpIterator::data() {
    throw VariTensorInternalError("Function should never be called");
}

// =================================================================================================
//                                                                                   iterator ctor |
// =================================================================================================

LinkedOpIterator::LinkedOpIterator(
    const double modifier,
    Preparatory& preparatory,
    const std::vector<Operation>* signs,
    const bool end/* = false */
):
    ExpressionIteratorBase{preparatory},
    m_modifier{modifier},
    m_sub_iterators{std::move(preparatory.sub_iterators)},
    m_signs{signs},
    m_end{end}
{}

// =================================================================================================
//                                                                              forwards iteration |
// =================================================================================================

LinkedOpIterator& LinkedOpIterator::operator++() {
    m_end = !increment_positions(m_positions, m_dimensions, *this);
    return *this;
}

LinkedOpIterator LinkedOpIterator::operator++(int) {
    LinkedOpIterator copy = *this;
    ++*this;
    return copy;
}

double LinkedOpIterator::operator*() const {
    return deref();
}

bool LinkedOpIterator::operator==(const LinkedOpIterator& other) const {
    if (m_end != other.m_end) [[likely]] return false;
    if (m_end) return true;
    return m_modifier == other.m_modifier && m_positions == other.m_positions; // good enough for normal iteration
}

// =================================================================================================
//                                                                                 index iteration |
// =================================================================================================

void LinkedOpIterator::increment(const int index_id) const {
    for (auto& iterator: m_sub_iterators) std::visit(Increment(index_id), iterator);
}

void LinkedOpIterator::reset(const int index_id) const {
    for (auto& iterator: m_sub_iterators) std::visit(Reset(index_id), iterator);
}

double LinkedOpIterator::deref() const {
    double sum = 0;
    for (size_t i=0; i < m_sub_iterators.size(); ++i) {
        sum += std::visit(Deref, m_sub_iterators[i]) * static_cast<double> ((*m_signs)[i]);
    }
    return sum + m_modifier;
}

// =================================================================================================
//                                                                                     information |
// =================================================================================================

bool LinkedOpIterator::is_metric() {
    return false;
}

bool LinkedOpIterator::is_contiguous() {
    return false;
}

// =================================================================================================
//                                                                              LinkedOp iteration |
// =================================================================================================

LinkedOpIterator LinkedOp::begin() const {
    Preparatory preparatory{m_sub_expressions, LINKED};
    return iterator{m_modifier, preparatory, &m_signs};
}

LinkedOpIterator LinkedOp::end() const {
    Preparatory preparatory{};
    return iterator{m_modifier, preparatory, &m_signs, true};
}

ExpressionIterator LinkedOp::vbegin() const {
    return begin();
}

// =================================================================================================
//                                                                               LinkedOp populate |
// =================================================================================================

[[nodiscard]] bool LinkedOp::is_scalar() const {
    return m_sub_expressions.empty();
}

[[nodiscard]] double LinkedOp::get_scalar() const {
    return m_modifier;
}

void LinkedOp::add_element(const double value, const Operation sign) {
    deny(!is_scalar(), "Cannot add/subtract scalar with non-scalar expression");
    sign == ADD ? m_modifier += value : m_modifier -= value;
}

void LinkedOp::add_element(const Tensor& tensor, const Operation sign) {
    m_sub_expressions.emplace_back(View{tensor});
    sign == ADD ? m_signs.push_back(ADD) : m_signs.push_back(SUB);
}

void LinkedOp::add_element(const LinkedOp& linked_op, const Operation sign) {
    if (sign == ADD) {
        m_modifier += linked_op.m_modifier;
        m_signs.append_range(linked_op.m_signs);
    }
    else {
        m_modifier -= linked_op.m_modifier;
        for (auto& sub_sign: linked_op.m_signs) m_signs.push_back(static_cast<Operation>(sub_sign * -1));
    }
    m_sub_expressions.append_range(linked_op.m_sub_expressions);
}

void LinkedOp::add_element(const ProductOp& product_op, const Operation sign) {
    m_sub_expressions.emplace_back(product_op);
    m_signs.push_back(sign);
}

void LinkedOp::add_element(const View& view, const Operation sign) {
    m_sub_expressions.emplace_back(view);
    m_signs.push_back(sign);
}

void LinkedOp::populate(Tensor& tensor, const bool allocate/* = true */) {
    auto populate_aligned_indices = [&](Preparatory& preparatory) {
        if (allocate) {
            tensor.m_dimensions = preparatory.dimensions;
            tensor.m_size = preparatory.size;
            DoublePtr new_data = allocate_copy(
                std::visit(IsContiguous, preparatory.sub_iterators[0]) ?
                    std::get<View>(m_sub_expressions[0]).data() :
                    std::visit(GetTensor{}, m_sub_expressions[0]).m_data.get(),
                tensor.m_size
            );
            tensor.m_data.swap(new_data);
        }

        // for +=/-=, we start with the 0th in the right place, for +/- we handled with 0th above, so start i at 1
        for (unsigned i=1; i<preparatory.sub_iterators.size(); ++i) {
            piecewise(
                tensor.m_data.get(),
                std::visit(IsContiguous, preparatory.sub_iterators[i]) ?
                    std::get<View>(m_sub_expressions[i]).data() :
                    std::visit(GetTensor{}, m_sub_expressions[i]).m_data.get(),
                tensor.m_data.get(),
                tensor.m_size,
                m_signs[i]
            );
        }
    };

    switch (Preparatory preparatory{m_sub_expressions, LINKED}; preparatory.state) {
    case SCALAR:
        tensor.populate_scalar(m_modifier, allocate);
        break;
    case ALIGNED_INDICES:
        populate_aligned_indices(preparatory);
        break;
    case GENERAL: {
        iterator iter{m_modifier, preparatory, &m_signs};
        tensor.populate_general(iter, end(), allocate);
        break;
    }
    case FREE_MULTIPLICATION: // to stop the compiler warning
    default:
        throw VariTensorInternalError("Invalid expression state!");
    }
}

} // namespace varitensor::impl
