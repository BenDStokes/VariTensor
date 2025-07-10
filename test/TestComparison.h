#ifndef TEST_COMPARISON_H
#define TEST_COMPARISON_H

#include <memory>
#include <numeric>

#include "TestSet.h"
#include "varitensor/Tensor.h"

struct TensorEquality final: TestSet::Test {
    explicit TensorEquality() : Test("Equality (Tensor)") {}

    bool run_test() override {
        // given
        const auto i = Index(LATIN);

        const auto first = Tensor{i};
        const auto second = Tensor{i};

        // when
        std::ranges::fill(first, 1);
        std::ranges::fill(second, 1);

        // then
        return first == second;
    }
};

struct NonEqualData final: TestSet::Test {
    explicit NonEqualData() : Test("Non-Equal Data (Tensor)") {}

    bool run_test() override {
        // given
        const auto i = Index(LATIN);
        const auto j = Index(LATIN);

        const auto first = Tensor{{i, COVARIANT}};
        const auto second = Tensor{{j, COVARIANT}};

        // when
        for (auto& value: first) value = 1;
        for (auto& value: second) value = 2;

        // then
        return first != second;
    }
};

struct DoubleEquality final: TestSet::Test {
    explicit DoubleEquality() : Test("Equality (Double)") {}

    bool run_test() override
    {
        const Tensor test{1};
        return test == 1;
    }
};

struct ViewEquality final: TestSet::Test {
    explicit ViewEquality() : Test("Equality (View)") {}

    bool run_test() override
    {
        // given
        const Index mu{GREEK};
        const Index j{LATIN};

        const Tensor t1{mu};
        const Tensor t2{mu};

        // when
        std::ranges::iota(t1, 0);
        std::ranges::iota(t2, 0);

        // then
        return t1[j] == t2[j];
    }
};

struct TestComparison final: TestSet {
    explicit TestComparison() : TestSet("Test Comparison") {
        add_sub_test(std::make_unique<TensorEquality>());
        add_sub_test(std::make_unique<DoubleEquality>());
        add_sub_test(std::make_unique<ViewEquality>());
        add_sub_test(std::make_unique<NonEqualData>());
    }
};

#endif
