#ifndef TEST_INDEX_MANIPULATION_H
#define TEST_INDEX_MANIPULATION_H

#include <memory>

#include "TestSet.h"
#include "varitensor/Tensor.h"

namespace local {

struct BasicIndexLowering final: TestSet::Test {
    explicit BasicIndexLowering() : Test("Basic Index Lowering") {}

    bool run_test() override {
        //given
        const Index i{LATIN}, j{LATIN};

        const Tensor T{
            {i, CONTRAVARIANT}
        };
        const Tensor g = metric_tensor({
            {i, COVARIANT},
            {j, COVARIANT}
        });

        // when
        const Tensor result = T[i] * g[i, j];

        // then
        return
            !result.has_index(i) &&
            result.has_index(j) &&
            result.variance(j) == COVARIANT;
    }
};

struct CorrectOrdering final: TestSet::Test {
    explicit CorrectOrdering() : Test("Correct Index Ordering") {}

    bool run_test() override {
        //given
        const Index i{LATIN}, j{LATIN}, k{LATIN};

        const auto T = Tensor{
            {i, COVARIANT},
            {j, COVARIANT}
        };
        const auto g = metric_tensor({
            {i, CONTRAVARIANT},
            {k, CONTRAVARIANT}
        });

        // when
        const Tensor result = T[i, j] * g[i, k];

        // then
        return !result.has_index(i) &&
               result.has_index(k) &&
               result.index_position(k) == 0 &&
               result.variance(k) == CONTRAVARIANT;
    }
};

struct RaiseMiddle final: TestSet::Test {
    explicit RaiseMiddle() : Test("Raise Middle Index") {}

    bool run_test() override {
        //given
        const Index i{LATIN}, j{LATIN}, k{LATIN},
                    l{LATIN}, m{LATIN}, n{LATIN};

        const auto T = Tensor{
            {i, CONTRAVARIANT},
            {j, CONTRAVARIANT},
            {k, CONTRAVARIANT},
            {l, CONTRAVARIANT},
            {m, CONTRAVARIANT},
        };
        const auto g = metric_tensor({
            {k, COVARIANT},
            {n, COVARIANT}
        });

        // when
        const Tensor result = g * T;

        // then
        return result.has_index(i) &&
               result.has_index(j) &&
               result.has_index(n) &&
               result.has_index(l) &&
               result.has_index(m) &&
               !result.has_index(k) &&
               result.index_position(n) == 2 &&
               result.variance(n) == COVARIANT;
    }
};

struct BothRepeated final: TestSet::Test {
    explicit BothRepeated() : Test("Both Indices Repeated") {}

    bool run_test() override {
        //given
        const Index i{LATIN}, j{LATIN}, k{LATIN}, l{LATIN};

        const auto T = Tensor{
            {i, CONTRAVARIANT},
            {j, CONTRAVARIANT},
        };
        const auto U = Tensor{
            {k, CONTRAVARIANT},
            {l, CONTRAVARIANT},
        };
        const auto g = metric_tensor({
        {j, COVARIANT},
        {k, COVARIANT}
        });

        // when
        const Tensor result = T * g * U;

        // then
        return result.has_index(i) &&
               result.has_index(l) &&
               !result.has_index(j) &&
               !result.has_index(k) &&
               result.variance(i) == CONTRAVARIANT &&
               result.variance(l) == CONTRAVARIANT;
    }
};

} // namespace advanced_tests

struct TestIndexManip final : TestSet {
    explicit TestIndexManip() : TestSet("Test Index Manipulation") {
        add_sub_test(std::make_unique<local::BasicIndexLowering>());
        add_sub_test(std::make_unique<local::CorrectOrdering>());
        add_sub_test(std::make_unique<local::RaiseMiddle>());
        add_sub_test(std::make_unique<local::BothRepeated>());
    }
};

#endif
