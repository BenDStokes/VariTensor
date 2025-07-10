/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * Licence, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef VARITENSOR_LINKEDOP_H
#define VARITENSOR_LINKEDOP_H

#include "common.h"
#include "View_decl.h"

namespace varitensor::impl {

class LinkedOpIterator: public ExpressionIteratorBase {
public:
    LinkedOpIterator() = default;

    LinkedOpIterator(
        double modifier,
        const Expressions& sub_expressions,
        const std::vector<Operation>* signs,
        bool end = false
    );

    LinkedOpIterator(const LinkedOpIterator& other) = default;
    LinkedOpIterator& operator=(const LinkedOpIterator& other) = default;
    LinkedOpIterator(LinkedOpIterator&& other) noexcept = default;
    LinkedOpIterator& operator=(LinkedOpIterator&& other) = default;

    LinkedOpIterator& operator++();
    LinkedOpIterator operator++(int);
    double operator*() const;
    bool operator==(const LinkedOpIterator& other) const;

    [[nodiscard]] static bool is_metric();

    void increment(int index_id) const;
    void reset(int index_id) const;
    [[nodiscard]] double deref() const;

private:
    double m_modifier{1};
    ExpressionIterators m_sub_iterators{};
    const std::vector<Operation>* m_signs{nullptr};

    bool m_end{false};
};

class LinkedOp {
/**
 * Represents an operation between tensor expressions where repeated indices
 * are "linked". Supports addition and subtraction.
 */
public:
    template <ExpressionOperand_c T, ExpressionOperand_c U>
    LinkedOp(const T& first, const U& second, Operation sign):
        m_modifier{0}
    {
        add_element(first, ADD);
        add_element(second, sign);
    }

    using iterator = LinkedOpIterator;
    [[nodiscard]] iterator begin() const;
    [[nodiscard]] iterator end() const;
    [[nodiscard]] ExpressionIterator vbegin() const;

private:
    double m_modifier; // used to collate double/scalar operands
    Expressions m_sub_expressions;

    // used to track the sign of each sub-expression
    std::vector<Operation> m_signs;

    void add_element(const Tensor& tensor, Operation sign);
    void add_element(const LinkedOp& summation, Operation sign);
    void add_element(const SummedOp& product, Operation sign);
    void add_element(const View& view, Operation sign);
    void add_element(double value, Operation sign);
};

} // namespace varitensor::impl

template<varitensor::impl::ExpressionOperand_c T, varitensor::impl::ExpressionOperand_c U>
varitensor::impl::LinkedOp operator+(const T& first, const U& second) {
    return {first, second, varitensor::impl::ADD};
}
template<varitensor::impl::ExpressionOperand_c T, varitensor::impl::ExpressionOperand_c U>
varitensor::impl::LinkedOp operator-(const T& first, const U& second) {
    return {first, second, varitensor::impl::SUB};
}

#endif
