#ifndef TEST_SCALAR_H
#define TEST_SCALAR_H

#include "TestSet.h"
#include "varitensor/Tensor.h"

struct StaticCast final : TestSet::Test
{
    explicit StaticCast() : Test("Static Cast") {}
    bool run_test() override
    {
        const Tensor test_tensor{1};
        return static_cast<double>(test_tensor) == 1;
    }
};

struct NonScalarCastException final : TestSet::Test
{
    explicit NonScalarCastException() : Test("Non-Scalar Cast Exception") {}
    bool run_test() override
    {
        const Index index{LATIN};
        const Tensor test_tensor {index};

        try
        {
            static_cast<double>(test_tensor);
            return false;
        }
        catch(std::logic_error&)
        {
            return true;
        }
    }
};

struct DoubleAddition final : TestSet::Test
{
    explicit DoubleAddition() : Test("Addition (Double)") {}
    bool run_test() override
    {
        const Tensor test_tensor{1};

        const Tensor result = test_tensor + 1;
        return static_cast<double>(result) == 2;
    }
};

struct TensorAddition final : TestSet::Test
{
    explicit TensorAddition() : Test("Addition (Tensor)") {}
    bool run_test() override
    {
        const Tensor test_tensor1{1}, test_tensor2{1};

        const Tensor result = test_tensor1 + test_tensor2;
        return static_cast<double>(result) == 2;
    }
};

struct DoubleSubtraction final : TestSet::Test
{
    explicit DoubleSubtraction() : Test("Subtraction (Double)") {}
    bool run_test() override
    {
        const Tensor test_tensor{1};
        const Tensor result = test_tensor - 1;
        return static_cast<double>(result) == 0;
    }
};

struct TensorSubtraction final : TestSet::Test
{
    explicit TensorSubtraction() : Test("Subtraction (Tensor)") {}
    bool run_test() override
    {
        const Tensor test_tensor1{1}, test_tensor2{1};

        const Tensor result = test_tensor1 - test_tensor2;
        return static_cast<double>(result) == 0;
    }
};

struct DoubleMultiplication final : TestSet::Test
{
    explicit DoubleMultiplication() : Test("Multiplication (Double)") {}
    bool run_test() override
    {
        const Tensor test_tensor{1};
        const Tensor result = test_tensor * 2;
        return static_cast<double>(result) == 2;
    }
};

struct TensorMultiplication final : TestSet::Test
{
    explicit TensorMultiplication() : Test("Multiplication (Tensor)") {}
    bool run_test() override
    {
        const Tensor test_tensor1{2}, test_tensor2{3};

        const Tensor result = test_tensor1 * test_tensor2;
        return static_cast<double>(result) == 6;
    }
};

struct DoubleDivision final : TestSet::Test
{
    explicit DoubleDivision() : Test("Division (Double)") {}
    bool run_test() override
    {
        const Tensor test_tensor{2};
        const Tensor result = test_tensor / 2;
        return static_cast<double>(result) == 1;
    }
};

struct TensorDivision final : TestSet::Test
{
    explicit TensorDivision() : Test("Division (Tensor)") {}
    bool run_test() override
    {
        const Tensor test_tensor1{6}, test_tensor2{3};

        const Tensor result = test_tensor1 / test_tensor2;
        return static_cast<double>(result) == 2;
    }
};

struct TestScalar final : TestSet {
    explicit TestScalar() : TestSet("Test Scalars") {
        add_sub_test(std::make_unique<StaticCast>());
        add_sub_test(std::make_unique<NonScalarCastException>());
        add_sub_test(std::make_unique<DoubleAddition>());
        add_sub_test(std::make_unique<TensorAddition>());
        add_sub_test(std::make_unique<DoubleSubtraction>());
        add_sub_test(std::make_unique<TensorSubtraction>());
        add_sub_test(std::make_unique<DoubleMultiplication>());
        add_sub_test(std::make_unique<TensorMultiplication>());
        add_sub_test(std::make_unique<DoubleDivision>());
        add_sub_test(std::make_unique<TensorDivision>());
    }
};

#endif
