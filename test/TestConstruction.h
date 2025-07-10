#ifndef TEST_CONSTRUCTION_H
#define TEST_CONSTRUCTION_H

#include "TestSet.h"
#include "varitensor/Tensor.h"

struct VqiList final: TestSet::Test {
    explicit VqiList() : Test("VQI Init-List Ctor") {}

    bool run_test() override {
        const Index i{LATIN}, j{LATIN};

        const Tensor test_tensor{
            {
                {i, CONTRAVARIANT},
                {j, COVARIANT},
            }
        };

        return true;
    }
};

struct VqiVector final: TestSet::Test {
    explicit VqiVector() : Test("VQI Vector Ctor") {}

    bool run_test() override {
        const Index i{LATIN}, j{LATIN};
        const auto indices = std::vector<VarianceQualifiedIndex>{
            {i, CONTRAVARIANT},
            {j, COVARIANT},
        };

        const Tensor test_tensor{indices};

        return true;
    }
};

struct IndexList final: TestSet::Test {
    explicit IndexList() : Test("Index Init-List Ctor") {}

    bool run_test() override {
        Index i{LATIN}, j{LATIN};
        const Tensor test_tensor{{i, j}};

        return true;
    }
};

struct IndexVector final: TestSet::Test {
    explicit IndexVector() : Test("Index Vector Ctor") {}

    bool run_test() override {
        const Index i{LATIN}, j{LATIN};
        const std::vector indices{i, j};

        const Tensor test_tensor{indices};

        return true;
    }
};

struct Copy final: TestSet::Test {
    explicit Copy() : Test("Copying") {}

    bool run_test() override {
        const Tensor first(1);
        Tensor second(first); // NOLINT - clang tidy will correctly attempt to avoid the copy
        return first == second;
    }
};

struct CopyAssign final: TestSet::Test {
    explicit CopyAssign() : Test("Copy Assignment") {}

    bool run_test() override {
        const Tensor first(1);
        auto second = first; // NOLINT - clang tidy will correctly attempt to avoid the copy
        return first == second;
    }
};

struct Move final: TestSet::Test {
    explicit Move() : Test("Moving") {}

    bool run_test() override {
        Tensor first(1);
        const Tensor second(1);
        const Tensor third(std::move(first));
        return second == third;
    }
};

struct MoveAssign final: TestSet::Test {
    explicit MoveAssign() : Test("Move Assignment") {}

    bool run_test() override {
        Tensor first(1);
        auto second = std::move(first);
        return true;
    }
};

struct TestConstruction final: TestSet {
    explicit TestConstruction() : TestSet("Test Construction") {
        add_sub_test(std::make_unique<VqiList>());
        add_sub_test(std::make_unique<VqiVector>());
        add_sub_test(std::make_unique<IndexList>());
        add_sub_test(std::make_unique<IndexVector>());
        add_sub_test(std::make_unique<Copy>());
        add_sub_test(std::make_unique<CopyAssign>());
        add_sub_test(std::make_unique<Move>());
        add_sub_test(std::make_unique<MoveAssign>());
    }
};

#endif
