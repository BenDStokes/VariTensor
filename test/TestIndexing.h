#ifndef TEST_INDEXING_H
#define TEST_INDEXING_H

#include <memory>

#include "TestSet.h"
#include "varitensor/pretty_print.h"
#include "varitensor/Tensor.h"

namespace local {

struct DisagreementThrows final: TestSet::Test {
    explicit DisagreementThrows(): Test("Disagreement Throws") {}

    bool run_test() override {
        const Index mu{GREEK}, nu{GREEK}, xi{GREEK};
        const Index i{5};
        const Tensor T{mu, nu};

        try {
            auto result = T[mu, nu, xi];
            return false;
        }
        catch(std::logic_error&) {}

        try {
            auto result = T[mu];
            return false;
        }
        catch(std::logic_error&) {}

        try {
            auto result = T[mu, i];
            return false;
        }
        catch(std::logic_error&) {}

        return true;
    }
};

} // namespace local

struct AllStatic final: TestSet::Test {
    explicit AllStatic(): Test("All Static") {}

    bool run_test() override {
        const Index mu{GREEK}, nu{GREEK};
        const Tensor T{mu, nu};
        std::ranges::iota(T, 0);

        const auto result = T[2, 3];

        return static_cast<double>(result) == 14;
    }
};

struct AllFree final: TestSet::Test {
    explicit AllFree(): Test("All Free") {}

    bool run_test() override {
        //given
        const Index mu{GREEK}, nu{GREEK};

        Tensor T{mu, nu};
        Tensor expected{mu, nu};

        std::ranges::iota(T, 0);
        std::ranges::iota(expected, 0);

        // when
        const Tensor actual = T[mu, nu];

        // then
        return expected == actual;
    }
};

struct BasicContraction final: TestSet::Test {
    explicit BasicContraction(): Test("Basic Contraction") {}

    bool run_test() override {
        const Index mu{GREEK}, nu{GREEK};
        const Tensor T{mu, nu};
        std::ranges::iota(T, 0);

        const Tensor result = T[mu, mu];

        return static_cast<double>(result) == 30;
    }
};

struct IndexReduction final: TestSet::Test {
    explicit IndexReduction(): Test("Index Reduction") {}

    bool run_test() override {
        const Index mu{GREEK}, i{LATIN};

        Tensor T{{mu, COVARIANT}};
        Tensor expected{{i, COVARIANT}};

        std::ranges::iota(T, 0);
        std::ranges::iota(expected, 0);

        const Tensor actual = T[i];

        return expected == actual;
    }
};

struct ComplexExpression final: TestSet::Test {
    explicit ComplexExpression(): Test("Complex Expression") {}

    bool run_test() override {
        // given
        const Index mu{GREEK}, nu{GREEK}, xi{GREEK}, omicron{GREEK};
        const Index i{LATIN};

        Tensor T{mu, nu, xi, omicron};
        std::ranges::iota(T, 0);

        const Tensor expected{i};
        const std::vector<double> values = {286, 350, 414};
        std::ranges::copy(values, expected.begin());

        // when
        const Tensor actual = T[mu, mu, i, 1];

        // then
        return expected == actual;
    }
};

struct DoubleContraction final: TestSet::Test {
    explicit DoubleContraction(): Test("Double Contraction") {}

    bool run_test() override {
        // given
        const Index mu{GREEK}, nu{GREEK}, xi{GREEK}, omicron{GREEK};

        Tensor T{mu, nu, xi, omicron};

        std::ranges::iota(T, 0);
        const Tensor expected{2040};

        // when
        const Tensor actual = T[mu, mu, nu, nu];

        // then
        return expected == actual;
    }
};

struct ExcessiveContractionThrows final: TestSet::Test {
    explicit ExcessiveContractionThrows(): Test("Excessive Contraction") {}

    bool run_test() override {
        // given
        const Index mu{GREEK}, nu{GREEK}, xi{GREEK}, omicron{GREEK};
        const Tensor T{mu, nu, xi, omicron};

        // when
        try {
            Tensor actual = T[mu, mu, mu, nu];
            return false;
        }
        catch(std::logic_error&) {}

        // then
        return true;
    }
};

struct ReducingContraction final: TestSet::Test {
    explicit ReducingContraction(): Test("Reducing Contraction") {}

    bool run_test() override {
        const Index mu{GREEK}, nu{GREEK};
        const Index i{LATIN};

        const Tensor T{mu, nu};
        std::ranges::iota(T, 0);

        const Tensor result = T[i, i];

        return static_cast<double>(result) == 15;
    }
};

struct SplitContraction final: TestSet::Test {
    explicit SplitContraction(): Test("Split Contraction") {}

    bool run_test() override {
        // given
        const Index i{LATIN}, j{LATIN}, k{LATIN};

        Tensor T{i, j, k};
        std::ranges::iota(T, 0);

        const Tensor expected{j};
        std::vector<double> values = {30, 39, 48};
        std::ranges::copy(values, expected.begin());

        // when
        const Tensor actual = T[i, j, i];

        // then
        return expected == actual;
    }
};

struct ScalarShortCircuit final: TestSet::Test {
    explicit ScalarShortCircuit(): Test("Scalar Short Circuit") {}

    bool run_test() override {
        const auto T = Tensor{2};
        const auto result = T[];

        return result == 2;
    }
};

struct SliceAssignment final: TestSet::Test {
    explicit SliceAssignment(): Test("Slice Assignment") {}

    bool run_test() override {
        // given
        const Index i{LATIN}, j{LATIN}, k{LATIN};

        const Tensor T{i, j, k};
        const Tensor U{i, j};

        std::ranges::fill(U, 1);

        // when
        T[i, j, 0] = U;

        // then
        return std::ranges::all_of(T[i, j, 0], [](const auto& value) { return value == 1; })
            && std::ranges::all_of(T[i, j, k(1)], [](const auto& value) { return value == 0; });
    }
};

struct SimpleInterval final: TestSet::Test {
    explicit SimpleInterval(): Test("Simple Interval") {}

    bool run_test() override {
        // given
        const Index mu{GREEK}, i{LATIN};

        const Tensor T{mu};
        Tensor expected{i};

        std::ranges::iota(T, 0);
        std::ranges::iota(expected, 1);

        // when
        const Tensor actual = T[mu(1)];

        // then
        return expected == actual[i];
    }
};

struct MultipleIntervals final: TestSet::Test {
    explicit MultipleIntervals(): Test("Multiple Intervals") {}

    bool run_test() override {
        // given
        const Index i{3}, j{2}, mu{5}, nu{5};

        const Tensor T{mu, nu};
        std::ranges::iota(T, 0);

        const Tensor expected{i, j};
        expected[0, 0] = 11;
        expected[1, 0] = 12;
        expected[2, 0] = 13;
        expected[0, 1] = 16;
        expected[1, 1] = 17;
        expected[2, 1] = 18;

        // when
        const Tensor actual = T[mu(1, 3), nu(2, 3)];

        // then
        return expected == actual[i, j];
    }
};

struct BadInterval final: TestSet::Test {
    explicit BadInterval(): Test("Bad Intervals") {}

    bool run_test() override {
        const Index i{LATIN};

        int interval_sum = 0; // pointless sum to keep the compiler from complaining
        try {
            const auto interval = i(1, 3);
            interval_sum += interval.first; // NOLINT - the linter will correctly flag this as unused
            return false;
        }
        catch(std::logic_error&) {}

        try {
            const auto interval = i(-1, 3);
            interval_sum += interval.first; // NOLINT - the linter will correctly flag this as unused
            return false;
        }
        catch(std::logic_error&) {}

        try {
            const auto interval = i(3, 1);
            interval_sum += interval.first; // NOLINT - the linter will correctly flag this as unused
            return false;
        }
        catch(std::logic_error&) {}

        return true;
    }
};

struct TestIndexing final: TestSet {
    explicit TestIndexing(): TestSet("Test Indexing") {
        add_sub_test(std::make_unique<local::DisagreementThrows>());
        add_sub_test(std::make_unique<AllStatic>());
        add_sub_test(std::make_unique<AllFree>());
        add_sub_test(std::make_unique<IndexReduction>());
        add_sub_test(std::make_unique<BasicContraction>());
        add_sub_test(std::make_unique<ReducingContraction>());
        add_sub_test(std::make_unique<DoubleContraction>());
        add_sub_test(std::make_unique<ComplexExpression>());
        add_sub_test(std::make_unique<ExcessiveContractionThrows>());
        add_sub_test(std::make_unique<SplitContraction>());
        add_sub_test(std::make_unique<ScalarShortCircuit>());
        add_sub_test(std::make_unique<SliceAssignment>());
        add_sub_test(std::make_unique<SimpleInterval>());
        add_sub_test(std::make_unique<MultipleIntervals>());
        add_sub_test(std::make_unique<BadInterval>());
    }
};

#endif
