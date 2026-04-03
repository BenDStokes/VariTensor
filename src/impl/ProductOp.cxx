/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * Licence, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include <vector>

#include "varitensor/impl/ProductOp.h"
#include "varitensor/Tensor.h"
#include "varitensor/impl/bulk.h"
#include "varitensor/impl/deny.h"
#include "varitensor/impl/Preparatory.h"
#include "varitensor/impl/View.h"
#include "varitensor/impl/visitors.h"

namespace varitensor::impl {

[[nodiscard]] double* ProductOpIterator::data() {
    throw VariTensorInternalError("Function should never be called");
}

// =================================================================================================
//                                                                                   iterator ctor |
// =================================================================================================

ProductOpIterator::ProductOpIterator(
    const double modifier,
    Preparatory& preparatory,
    const bool end/* = false */
):
    ExpressionIteratorBase{preparatory},
    m_modifier{modifier},
    m_sub_iterators{std::move(preparatory.sub_iterators)},
    m_repeated{std::move(preparatory.repeated)},
    m_repeated_positions(m_repeated.size(), 0),
    m_end{end}
{}

// =================================================================================================
//                                                                              forwards iteration |
// =================================================================================================

ProductOpIterator& ProductOpIterator::operator++() {
    m_end = !increment_positions(m_positions, m_dimensions, *this);
    return *this;
}

ProductOpIterator ProductOpIterator::operator++(int) {
    ProductOpIterator copy = *this;
    ++*this;
    return copy;
}

double ProductOpIterator::operator*() const {
    return deref();
}

bool ProductOpIterator::operator==(const ProductOpIterator& other) const {
    if (m_end != other.m_end) [[likely]] return false;
    if (m_end) return true;
    return m_modifier == other.m_modifier && m_positions == other.m_positions; // good enough for normal iteration
}

// =================================================================================================
//                                                                                 index iteration |
// =================================================================================================

void ProductOpIterator::increment(const int index_id) const {
    for (auto& iterator: m_sub_iterators) std::visit(Increment(index_id), iterator);
}

void ProductOpIterator::reset(const int index_id) const {
    for (auto& iterator: m_sub_iterators) std::visit(Reset(index_id), iterator);
}

double ProductOpIterator::deref() const {
    /* Despite modifying m_repeated_positions, this function is const as it always returns the positions to 0 */
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

bool ProductOpIterator::is_metric() {
    return false;
}

bool ProductOpIterator::is_contiguous() {
    return false;
}

// =================================================================================================
//                                                                              ProductOp populate |
// =================================================================================================

void ProductOp::populate(Tensor& tensor) {
    auto populate_free_multiplication = [&](Preparatory& preparatory) {
        DoublePtr new_data = allocate(preparatory.size, 0);

        if (std::visit(IsContiguous, preparatory.sub_iterators[0])) {
            broadcast_vec(
                std::visit(GetData, preparatory.sub_iterators[0]),
                new_data.get(),
                std::visit(GetSize, preparatory.sub_iterators[0]),
                preparatory.size
            );
        }
        else {
            const auto tmp = std::visit(GetTensor{}, m_sub_expressions[0]);
            broadcast_vec(tmp.m_data.get(), new_data.get(), tmp.size(), preparatory.size);
        }

        size_t dim_i = std::visit(GetDimensions, preparatory.sub_iterators[0]).size() - 1;
        size_t width = std::visit(GetSize, preparatory.sub_iterators[0]);
        for (unsigned i=1; i < preparatory.sub_iterators.size(); ++i) {
            const size_t dim_i_previous = dim_i;
            dim_i += std::visit(GetDimensions, preparatory.sub_iterators[i]).size();

            if (std::visit(IsContiguous, preparatory.sub_iterators[i])) {
                broadcast_chunks(
                    std::visit(GetData, preparatory.sub_iterators[i]),
                    new_data.get(),
                    std::visit(GetSize, preparatory.sub_iterators[i]),
                    width
                );
            }
            else {
                const auto tmp = std::visit(GetTensor{}, m_sub_expressions[i]);
                broadcast_chunks(tmp.m_data.get(), new_data.get(), tmp.size(), width);
            }

            for (size_t j=dim_i_previous; j < dim_i; ++j) width *= preparatory.dimensions[j].size();
        }

        broadcast(m_modifier, new_data.get(), preparatory.size);

        // Set up the tensor at the end in case we're doing an assignment
        tensor.m_dimensions = preparatory.dimensions;
        tensor.m_size = preparatory.size;
        tensor.m_data.swap(new_data);
    };

    switch (Preparatory preparatory{m_sub_expressions, PRODUCT}; preparatory.state) {
    case SCALAR:
        tensor.populate_scalar(m_modifier, true);
        break;
    case FREE_MULTIPLICATION:
        populate_free_multiplication(preparatory);
        break;
    case GENERAL: {
        iterator iter{m_modifier, preparatory};
        tensor.populate_general(iter, end(), true);
        break;
    }
    default:
        throw VariTensorInternalError("Invalid expression state!");
    }
}

// =================================================================================================
//                                                                             ProductOp iteration |
// =================================================================================================

ProductOp::iterator ProductOp::begin() const {
    Preparatory preparatory{m_sub_expressions, PRODUCT};
    return iterator{m_modifier, preparatory};
}

ProductOp::iterator ProductOp::end() const {
    Preparatory preparatory{};
    return iterator{m_modifier, preparatory, true};
}

ExpressionIterator ProductOp::vbegin() const {
    return begin();
}

// =================================================================================================
//                                                                               ProductOp helpers |
// =================================================================================================


[[nodiscard]] bool ProductOp::is_scalar() const {
    return m_sub_expressions.empty();
}

[[nodiscard]] double ProductOp::get_scalar() const {
    return m_modifier;
}

void ProductOp::add_element(const double& value) {
    m_modifier *= value;
}

void ProductOp::add_element(const Tensor& tensor) {
    m_sub_expressions.emplace_back(View{tensor});
}

void ProductOp::add_element(const LinkedOp& linked_op) {
    m_sub_expressions.emplace_back(linked_op);
}

void ProductOp::add_element(const ProductOp& product_op) {
    m_modifier *= product_op.m_modifier;
    for (auto& expression: product_op.m_sub_expressions) m_sub_expressions.emplace_back(expression);
}

void ProductOp::add_element(const View& view) {
    m_sub_expressions.emplace_back(view);
}

} // namespace varitensor::impl
