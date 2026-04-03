#ifndef TEST_INFORMATION_H
#define TEST_INFORMATION_H

#include <memory>
#include <sstream>
#include <utility>

#include "TestSet.h"
#include "varitensor/pretty_print.h"
#include "varitensor/Tensor.h"

struct GetName final: TestSet::Test {
    explicit GetName() : Test("Name Set and Get") {}

    bool run_test() override
    {
        const Tensor test{"test", 1};
        return test.name() == "test";
    }
};

struct GetSize final: TestSet::Test {
    explicit GetSize() : Test("Get Size") {}

    bool run_test() override {
        const Tensor size_1{1};
        return size_1.size() == 1; // NOLINT - this is a test
    }
};

struct GetDimensionSize final: TestSet::Test {
    explicit GetDimensionSize() : Test("Get Dimension Size") {}

    bool run_test() override {
        const Index i{LATIN};

        const Tensor i_3{{i, CONTRAVARIANT}};

        return i_3.size(0) == 3;
    }
};

struct GetRank final: TestSet::Test {
    explicit GetRank() : Test("Get Rank") {}

    bool run_test() override {
        const Index i{LATIN};

        const Tensor rank_1{{i, CONTRAVARIANT}};

        return rank_1.rank() == 1;
    }
};

struct IsScalar final: TestSet::Test {
    explicit IsScalar() : Test("Is Scalar") {}

    bool run_test() override {
        const Index i{LATIN};

        const Tensor scalar{1};
        const Tensor non_scalar{{i, CONTRAVARIANT}};

        return scalar.is_scalar() && !non_scalar.is_scalar(); // NOLINT - this is a test
    }
};

struct IsMetric final: TestSet::Test {
    explicit IsMetric() : Test("Is Metric") {}

    bool run_test() override {
        const Index i{LATIN};
        const Index j{LATIN};

        const auto metric = metric_tensor({
            {i, COVARIANT},
            {j, COVARIANT}
        });
        const Tensor non_metric{{i, CONTRAVARIANT}};

        return metric.is_metric() && !non_metric.is_metric();
    }
};

struct GetIndices final: TestSet::Test {
    explicit GetIndices() : Test("Get Indices") {}

    bool run_test() override {
        const Index i{LATIN};
        const std::vector indices = {i};

        const Tensor test{{i, CONTRAVARIANT}};

        return test.indices() == indices;
    }
};

struct GetVQIIndices final: TestSet::Test {
    explicit GetVQIIndices() : Test("Get Qualified Indices") {}

    bool run_test() override {
        const Index i{LATIN};
        const std::vector<VarianceQualifiedIndex> qualified_indices = {
            {i, CONTRAVARIANT}
        };

        const Tensor test{{i, CONTRAVARIANT}};

        return test.qualified_indices() == qualified_indices;
    }
};

struct GetVariance final: TestSet::Test {
    explicit GetVariance() : Test("Get Variance") {}

    bool run_test() override {
        const Index i{LATIN};

        const Tensor i_covariant{{i, COVARIANT}};
        const Tensor i_contravariant{{i, CONTRAVARIANT}};

        return i_covariant.variance(i) == COVARIANT && i_contravariant.variance(i) == CONTRAVARIANT;
    }
};

struct HasIndex final: TestSet::Test {
    explicit HasIndex() : Test("Has Index") {}

    bool run_test() override {
        const Index i{LATIN};

        const Tensor with_i{{i}};
        const Tensor without_i{1};

        return with_i.has_index(i) && !without_i.has_index(i);
    }
};

struct IndexPosition final: TestSet::Test {
    explicit IndexPosition() : Test("Index Position") {}

    bool run_test() override {
        const Index i{LATIN};
        const Index j{LATIN};

        const Tensor j_1{{i, j}};

        return j_1.index_position(j) == 1;
    }
};

struct IntervalNaming final: TestSet::Test {
    explicit IntervalNaming() : Test("Interval Naming") {}

    bool run_test() override {
        const Index i{"i", LATIN};
        const Index i_interval = i(1);

        return i_interval.name() == "i(1-2)";
    }
};

struct ContractionIterationThrows final: TestSet::Test {
    explicit ContractionIterationThrows() : Test("Contraction Iteration") {}

    bool run_test() override {
        const Index i{LATIN}, j{LATIN}, k{LATIN};
        const Tensor t1{i, j, k};

        try {
            double sum = 0;
            for (const auto view = t1[i, j, j]; const auto value: std::as_const(view)) sum += value;
        }
        catch (...) {
            return false;
        }

        try {
            // assign to contraction should fail as the value does not exist
            for (auto& value: t1[i, j, j]) value = 1;
        }
        catch (...) {
            return true;
        }

        return false;
    }
};

struct PrettyPrint final: TestSet::Test {
    explicit PrettyPrint() : Test("PrettyPrint") {}

    bool run_test() override {
        const Index i{LATIN}, j{LATIN}, k{LATIN};
        const Tensor t1{i, j, k};
        std::stringstream stream;
        stream << t1;
        return true;
    }
};

struct TestInformation final: TestSet {
    explicit TestInformation() : TestSet("Test Information Retrieval") {
        add_sub_test(std::make_unique<GetName>());
        add_sub_test(std::make_unique<GetSize>());
        add_sub_test(std::make_unique<GetDimensionSize>());
        add_sub_test(std::make_unique<GetRank>());
        add_sub_test(std::make_unique<IsScalar>());
        add_sub_test(std::make_unique<IsMetric>());
        add_sub_test(std::make_unique<GetIndices>());
        add_sub_test(std::make_unique<GetVariance>());
        add_sub_test(std::make_unique<HasIndex>());
        add_sub_test(std::make_unique<IndexPosition>());
        add_sub_test(std::make_unique<IntervalNaming>());
#if VARITENSOR_VALIDATION_ON
        add_sub_test(std::make_unique<ContractionIterationThrows>());
#endif
        add_sub_test(std::make_unique<PrettyPrint>());
    }
};

#endif
