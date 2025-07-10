#ifndef TEST_LINEAR_OPERATIONS_H
#define TEST_LINEAR_OPERATIONS_H

#include <memory>

#include "TestSet.h"
#include "varitensor/Tensor.h"

struct DisagreementThrows final: TestSet::Test
{
    explicit DisagreementThrows() : Test("Disagreement Throws") {}
    bool run_test() override {
        Index index1{LATIN}, index2{2}, index3{LATIN};

        Tensor t1{{index1, COVARIANT}};
        Tensor t2{{index1, COVARIANT}, {index2, COVARIANT}};
        Tensor t3{{index3, COVARIANT}};
        Tensor t4{{index1, CONTRAVARIANT}};

        try
        { // rank mismatch
            Tensor test = t1 + t2;
            return false;
        }
        catch(std::length_error&) {}

        try
        { // index mismatch
            Tensor test = t1 + t3;
            return false;
        }
        catch(std::domain_error&) {}

        try
        { // variance mismatch
            Tensor test = t1 + t4;
            return false;
        }
        catch(std::domain_error&) {}

        return true;
    }
};

struct VectorAddition final: TestSet::Test
{
    explicit VectorAddition(): Test("Addition") {}
    bool run_test() override
    {
        // given
        const Index index(LATIN);

        Tensor t1{index};
        std::ranges::iota(t1, 0);

        Tensor t2{index};
        double n = 0;
        std::ranges::generate(t2, [&n]() mutable {return 2 * n++;});

        Tensor expected{index};
        n = 0;
        std::ranges::generate(expected, [&n]() mutable {return 3 * n++;});

        // when
        const Tensor actual = t1 + t2;

        // then
        return expected == actual;
    }
};

struct VectorSubtraction final : TestSet::Test
{
    explicit VectorSubtraction() : Test("Subtraction") {}
    bool run_test() override
    {
        // given
        const Index index{LATIN};

        Tensor t1{index};
        double n = 0;
        std::ranges::generate(t1, [&n]() mutable {return 2 * n++;});

        Tensor t2{index};
        std::ranges::iota(t2, 0);

        // when
        const Tensor actual = t1 - t2;

        // then
        return actual == t2;
    }
};

struct LinearInversion final : TestSet::Test
{
    explicit LinearInversion() : Test("Linear Inversion") {}
    bool run_test() override
    {
        // given
        const Index index(LATIN);

        const Tensor test{index};
        Tensor expected{index};

        std::ranges::fill(test, 1);
        std::ranges::fill(expected, -1);

        // when
        const Tensor actual = -test;

        // then
        return expected == actual;
    }
};


struct TestLinearOperations final : TestSet {
    explicit TestLinearOperations() : TestSet("Test Linear Operations") {
        add_sub_test(std::make_unique<DisagreementThrows>());
        add_sub_test(std::make_unique<VectorAddition>());
        add_sub_test(std::make_unique<VectorSubtraction>());
    }
};

#endif
