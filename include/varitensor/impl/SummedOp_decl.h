/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * Licence, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef VARITENSOR_SUMMEDOP_H
#define VARITENSOR_SUMMEDOP_H

#include "common.h"
#include "ExpressionIteratorBase.h"
#include "LinkedOp_decl.h"
#include "View_decl.h"

namespace varitensor::impl {

struct Couple {
/**
 * Helper struct used to track repeated indices and metric counterparts.
 */
    int dims[2]{-1, -1};
    int metric_index{-1};
    int other_index{-1};

    void add(int dimension, bool is_metric);
    void clear();

    [[nodiscard]] bool has() const;
    [[nodiscard]] bool is_repeated() const;
    [[nodiscard]] bool is_metric() const;
};

class SummedOpIterator: public ExpressionIteratorBase {
public:
    SummedOpIterator() = default;
    SummedOpIterator(double modifier, const Expressions& sub_expressions, bool end=false);

    SummedOpIterator(const SummedOpIterator& other) = default;
    SummedOpIterator& operator=(const SummedOpIterator& other) = default;
    SummedOpIterator(SummedOpIterator&& other) noexcept = default;
    SummedOpIterator& operator=(SummedOpIterator&& other) = default;

    SummedOpIterator& operator++();
    SummedOpIterator operator++(int);
    double operator*() const;
    bool operator==(const SummedOpIterator& other) const;

    [[nodiscard]] static bool is_metric();

    void increment(int index_id) const;
    void reset(int index_id) const;
    [[nodiscard]] double deref() const;

private:
    double m_modifier{1};
    ExpressionIterators m_sub_iterators{};

    // for index summation
    Dimensions m_repeated;
    mutable std::vector<int> m_repeated_positions; // mutable for deref() which, though it modifies the positions, always returns them to 0

    bool m_end{false};
};

class SummedOp {
/**
 * Represents an operation between tensor expressions where repeated indices
 * are summed over (einstein summation convention). Supports multiplication
 * and division (though division by non-scalar tensor is not allowed). Will
 * also correctly swap in indices when summing over metric tensors.
 */
public:
    template <ExpressionOperand_c T, ExpressionOperand_c U>
    SummedOp(const T& first, const U& second, Operation sign):
        m_modifier{1}
    {
        // order matters - indices need to be added in the order they arrive
        add_element(first, MUL);
        add_element(second, sign);
    }

    using iterator = SummedOpIterator;
    [[nodiscard]] iterator begin() const;
    [[nodiscard]] iterator end() const;
    [[nodiscard]] ExpressionIterator vbegin() const;

private:
    double m_modifier; // used to collate double/scalar operands
    Expressions m_sub_expressions;

    void add_element(const Tensor& tensor, Operation);
    void add_element(const LinkedOp& summation, Operation);
    void add_element(const SummedOp& product, Operation);
    void add_element(const View& view, Operation);
    void add_element(const double& value, Operation operation);
};

} // namespace::impl

template<varitensor::impl::ExpressionOperand_c T, varitensor::impl::ExpressionOperand_c U>
varitensor::impl::SummedOp operator*(const T& first, const U& second) {
    return {first, second, varitensor::impl::MUL};
}
template<varitensor::impl::ExpressionOperand_c T, varitensor::impl::ExpressionOperand_c U>
varitensor::impl::SummedOp operator/(const T& first, const U& second) {
    return {first, second, varitensor::impl::DIV};
}

#endif
