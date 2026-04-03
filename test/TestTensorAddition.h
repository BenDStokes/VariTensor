#ifndef TEST_TENSOR_ADDITION_H
#define TEST_TENSOR_ADDITION_H

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
        catch(TensorLogicError&) {}

        try
        { // index mismatch
            Tensor test = t1 + t3;
            return false;
        }
        catch(TensorLogicError&) {}

        try
        { // variance mismatch
            Tensor test = t1 + t4;
            return false;
        }
        catch(TensorLogicError&) {}

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

struct TripleAdd final: TestSet::Test {
    explicit TripleAdd(): Test("Ai + Bi + Ci") {}
    bool run_test() override {
        // given
        const Index i{LATIN};

        Tensor a{i}, b {i}, c{i};

        a[0] =  1; b[0] = -1; c[0] =  3;
        a[1] =  2; b[1] =  4; c[1] =  2;
        a[2] = -3; b[2] = -1; c[2] = -1;

        Tensor expected{i};
        expected[0] = 3; expected[1] = 8; expected[2] = -5;

        // when
        const Tensor result = a[i] + b[i] + c[i];

        return result == expected;
    }
};

struct MatrixAdd final: TestSet::Test {
    explicit MatrixAdd(): Test("Matrix Add") {}
    bool run_test() override
    {
        // given
        const Index i(LATIN), j(LATIN);

        Tensor t1{i, j};
        std::ranges::fill(t1, 1);

        const Tensor t2{i, j};
        std::ranges::fill(t2, 1);

        const Tensor expected{i, j};
        std::ranges::fill(expected, 2);

        // when
        const Tensor actual = t1 + t2;

        // then
        return expected == actual;
    }
};

struct TransposedAdd final: TestSet::Test {
    explicit TransposedAdd(): Test("Transposed Add") {}
    bool run_test() override
    {
        // given
        const Index i(LATIN), j(LATIN);

        Tensor t1{i, j};
        std::ranges::iota(t1, 0);

        const Tensor t2{j, i};
        std::ranges::iota(t2, 0);

        const Tensor expected{i, j};
        int n = 0;
        std::ranges::generate(expected, [&n]() mutable {
            const int value = 4*n - 8*(n/3);
            ++n;
            return value;
        });

        // when
        const Tensor actual = t1 + t2;

        // then
        return expected == actual;
    }
};


struct TestTensorAddition final : TestSet {
    explicit TestTensorAddition() : TestSet("Test Addition Operations") {
#if VARITENSOR_VALIDATION_ON
        add_sub_test(std::make_unique<DisagreementThrows>());
#endif
        add_sub_test(std::make_unique<VectorAddition>());
        add_sub_test(std::make_unique<VectorSubtraction>());
        add_sub_test(std::make_unique<LinearInversion>());
        add_sub_test(std::make_unique<TripleAdd>());
        add_sub_test(std::make_unique<MatrixAdd>());
        add_sub_test(std::make_unique<TransposedAdd>());
    }
};

#endif
