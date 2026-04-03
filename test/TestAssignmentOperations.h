#ifndef TEST_ASSIGNMENT_OPERATIONS_H
#define TEST_ASSIGNMENT_OPERATIONS_H

#include <algorithm>
#include <memory>

#include "TestSet.h"
#include "varitensor/Tensor.h"

struct AdditionAssignment final : TestSet::Test {
    explicit AdditionAssignment() : Test("Addition Assignment") {}
    bool run_test() override {
        // given
        const Index index{LATIN};

        Tensor t1{index};
        std::ranges::iota(t1, 0);

        Tensor t2{index};
        double n = 0;
        std::ranges::generate(t2, [&n]() mutable {return 2 * n++;});

        n = 0;
        Tensor expected{index};
        std::ranges::generate(expected, [&n]() mutable {return 3 * n++;});

        // when
        t1 += t2;

        // then
        return expected == t1;
    }
};

struct SubtractionAssignment final : TestSet::Test {
    explicit SubtractionAssignment() : Test("Subtraction Assignment") {}
    bool run_test() override {
        // given
        const Index index{LATIN};

        Tensor t1{index};
        double n=0;
        std::ranges::generate(t1, [&n]() mutable {return 3 * n++;});

        Tensor t2{index};
        n=0;
        std::ranges::generate(t2, [&n]() mutable {return 2 * n++;});

        Tensor expected{index };
        std::ranges::iota(expected, 0);

        // when
        t1 -= t2;

        // then
        return expected == t1;
    }
};

struct MultiplicationAssignment final : TestSet::Test {
    explicit MultiplicationAssignment() : Test("Multiplication Assignment") {}
    bool run_test() override {
        // given
        const Index index{LATIN};

        Tensor t1{index};
        std::ranges::iota(t1, 0);

        Tensor t2{index};
        double n=0;
        std::ranges::generate(t2, [&n]() mutable {return 2 * n++;});

        // when
        t1 *= t2;

        // then
        return static_cast<double>(t1) == 10;
    }
};

struct DivisionAssignment final : TestSet::Test {
    explicit DivisionAssignment() : Test("Division Assignment") {}
    bool run_test() override {
        // given
        const Index index{LATIN};

        Tensor t1{index};
        std::ranges::iota(t1, 0);

        const Tensor t2{2};

        auto expected = Tensor{index};
        double n=0;
        std::ranges::generate(expected, [&n]() mutable {return 0.5 * n++;});

        // when
        t1 /= t2;

        // then
        return expected == t1;
    }
};

struct ExpressionAssignment final : TestSet::Test {
    explicit ExpressionAssignment() : Test("Expression Assignment") {}
    bool run_test() override {
        // given
        const Index index{LATIN};

        Tensor t1{index}, t2{index}, t3{index}, expected{index};
        std::ranges::iota(t1, 0);
        std::ranges::iota(t2, 0);
        std::ranges::iota(t3, 0);

        double n = 0;
        std::ranges::generate(expected, [&n]() mutable {return 3 * n++;});

        // when
        t1 += t2 + t3;

        // then
        return expected == t1;
    }
};

struct DoubleScalarAssignment final : TestSet::Test {
    explicit DoubleScalarAssignment() : Test("Double-Scalar") {}
    bool run_test() override {
        // given
        double scalar = 1;
        const Tensor t1{2};

        // when
        scalar += t1;

        // then
        return scalar == 3;
    }
};

struct DoubleTensorAssignmentThrows final : TestSet::Test {
    explicit DoubleTensorAssignmentThrows() : Test("Double-Tensor Throws") {}
    bool run_test() override {
        // given
        const Index index{LATIN};
        const Tensor t1{index};

        // when
        try {
            double scalar = 1;
            scalar += t1; // NOLINT - the value is used to check for the exception
        }
        catch (const std::logic_error&) {
            // then
            return true;
        }
        return false;
    }
};

struct TestAssignmentOperations final : TestSet {
    explicit TestAssignmentOperations() : TestSet("Test Assignment Operations") {
        add_sub_test(std::make_unique<AdditionAssignment>());
        add_sub_test(std::make_unique<SubtractionAssignment>());
        add_sub_test(std::make_unique<MultiplicationAssignment>());
        add_sub_test(std::make_unique<DivisionAssignment>());
        add_sub_test(std::make_unique<ExpressionAssignment>());
        add_sub_test(std::make_unique<DoubleScalarAssignment>());
#if VARITENSOR_VALIDATION_ON
        add_sub_test(std::make_unique<DoubleTensorAssignmentThrows>());
#endif
    }
};

#endif
