#ifndef TEST_MANIPULATION_H
#define TEST_MANIPULATION_H

#include <memory>

#include "TestSet.h"
#include "varitensor/Tensor.h"

struct SetName final : TestSet::Test {
    explicit SetName() : Test("Set Name") {}
    bool run_test() override {
        Tensor test_tensor(1);
        test_tensor.set_name("test");
        return test_tensor.name() == "test";
    }
};

struct Transposition final : TestSet::Test {
    explicit Transposition() : Test("Index Transposition") {}
    bool run_test() override
    {
        const Index i{LATIN}, j{LATIN};
        Tensor test_tensor{i, j};

        test_tensor.transpose(i, j);

        return test_tensor.index_position(i) == 1 && test_tensor.index_position(j) == 0;
    }
};

struct Relabelling final : TestSet::Test {
    explicit Relabelling() : Test("Index Relabelling") {}
    bool run_test() override {
        const Index i{LATIN},j{LATIN};
        Tensor test_tensor{i};

        test_tensor.relabel(i, j);

        return test_tensor.has_index(j) && !test_tensor.has_index(i);
    }
};

struct SetVariance final : TestSet::Test {
    explicit SetVariance() : Test("Set Variance") {}
    bool run_test() override {
        const Index i{LATIN};
        Tensor test_tensor1{{i, COVARIANT}};
        Tensor test_tensor2{{i, CONTRAVARIANT}};

        test_tensor1.set_variance(i, CONTRAVARIANT);
        test_tensor2.set_variance(i, COVARIANT);

        return test_tensor1.variance(i) == CONTRAVARIANT && test_tensor2.variance(i) == COVARIANT;
    }
};

struct Raising final : TestSet::Test {
    explicit Raising() : Test("Index Raising") {}
    bool run_test() override {
        const Index i{LATIN};
        Tensor test_tensor{{i, COVARIANT}};

        test_tensor.raise(i);
        return test_tensor.variance(i) == CONTRAVARIANT;
    }
};

struct Lowering final : TestSet::Test {
    explicit Lowering() : Test("Index Lowering") {}
    bool run_test() override {
        const Index i{LATIN};
        Tensor test_tensor{{i, CONTRAVARIANT}};

        test_tensor.lower(i);
        return test_tensor.variance(i) == COVARIANT;
    }
};

struct TestManipulation final : TestSet {
    explicit TestManipulation() : TestSet("Test Manipulation Functions") {
        add_sub_test(std::make_unique<SetName>());
        add_sub_test(std::make_unique<Transposition>());
        add_sub_test(std::make_unique<Relabelling>());
        add_sub_test(std::make_unique<SetVariance>());
        add_sub_test(std::make_unique<Raising>());
        add_sub_test(std::make_unique<Lowering>());
    }
};

#endif
