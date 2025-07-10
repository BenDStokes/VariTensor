#ifndef TEST_TENSOR_MULTIPLICATION_H
#define TEST_TENSOR_MULTIPLICATION_H

#include <memory>

#include "TestSet.h"
#include "varitensor/Tensor.h"

namespace tm_tests {

struct TwoByTwoTimesTwoByOneFree final: TestSet::Test {
    explicit TwoByTwoTimesTwoByOneFree() : Test("2x2*2x1 Free") {}

    bool run_test() override {
        //given
        const Index i{LATIN}, j{LATIN}, k{LATIN};

        // these functions were chosen because they give all integer components
        auto f1 = [](const double x) -> double {return (1-x) * (x-3) * (x-9);};
        auto f2 = [](const double x) -> double {return (1-x) * (x-3) * (x-9) - x;};

        const Tensor first{i, j};
        double n = 0;
        std::ranges::generate(first, [&]() mutable {return f1(n++);});

        const Tensor second{k};
        n = 0;
        std::ranges::generate(second, [&]() mutable {return f2(n++);});

        const Tensor expected{i, j, k};
        for (auto iter=expected.begin(); iter!=expected.end(); ++iter) {
            int ii = iter.positions(0);
            int jj = iter.positions(1);
            int kk = iter.positions(2);

            *iter = f1(ii + 3 * jj) * f2(kk);
        }

        //when
        Tensor result = first * second;
        result.set_name("result");

        //then
        if (result.rank() != 3) return false;
        return expected == result;
    }
};

struct TwoByTwoTimesTwoByTwoFree final: TestSet::Test {
    explicit TwoByTwoTimesTwoByTwoFree() : Test("2x2*2x2 Free") {}

    bool run_test() override {
        //given
        const Index i{LATIN}, j{LATIN}, k{LATIN}, l{LATIN};

        // these functions were chosen because they give all integer components
        auto f1 = [](const double x) -> double {return (1-x) * (x-3) * (x-9);};
        auto f2 = [](const double x) -> double {return (1-x) * (x-3) * (x-9) - x;};

        const Tensor first{i, j};
        double n = 0;
        std::ranges::generate(first, [&]() mutable {return f1(n++);});

        const Tensor second{k, l};
        n = 0;
        std::ranges::generate(second, [&]() mutable {return f2(n++);});

        const Tensor expected{i, j, k, l};
        for (auto iter=expected.begin(); iter!=expected.end(); ++iter) {
            int ii = iter.positions(0);
            int jj = iter.positions(1);
            int kk = iter.positions(2);
            int ll = iter.positions(3);

            *iter = f1(ii + 3 * jj) * f2(kk + 3 * ll);
        }

        //when
        const Tensor result = first * second;

        //then
        if (result.rank() != 4) return false;
        return expected == result;
    }
};

struct SingleRepeated final: TestSet::Test {
    explicit SingleRepeated() : Test("Single Repeated") {}

    bool run_test() override {
        //given
        const Index i = Index{LATIN};

        const Tensor first{i};
        const Tensor second{i};

        std::ranges::fill(first, 2);
        std::ranges::fill(second, 3);

        //when
        const Tensor result = first * second;

        //then
        return static_cast<double> (result) == 18;
    }
};

struct Mixed final: TestSet::Test {
    explicit Mixed() : Test("Mixed") {}

    bool run_test() override {
        //given
        const Index i{LATIN}, j{LATIN}, k{LATIN};

        const Tensor first = Tensor{i, k};
        const Tensor second = Tensor{j, k};
        const Tensor expected = Tensor{i, j};

        std::ranges::fill(first, 2);
        std::ranges::fill(second, 3);
        std::ranges::fill(expected, 18);

        //when
        const Tensor result = first * second;

        //then
        return expected == result;
    }
};

struct DivThrows final: TestSet::Test {
    explicit DivThrows() : Test("Tensor Division Throws") {}

    bool run_test() override {
        //given
        const Index i{LATIN};

        const Tensor first = Tensor{i};
        const Tensor second = Tensor{i};

        std::ranges::fill(first, 6);
        std::ranges::fill(second, 3);

        //when
        try {
            auto result = first / second;
        }
        catch (const std::logic_error&) {
            return true;
        }
        return false;
    }
};

} // namespace tm_tests

struct TestTensorMultiplication final: TestSet {
    explicit TestTensorMultiplication() : TestSet("Test Tensor Multiplication") {
        add_sub_test(std::make_unique<tm_tests::TwoByTwoTimesTwoByOneFree>());
        add_sub_test(std::make_unique<tm_tests::TwoByTwoTimesTwoByTwoFree>());
        add_sub_test(std::make_unique<tm_tests::SingleRepeated>());
        add_sub_test(std::make_unique<tm_tests::Mixed>());
        add_sub_test(std::make_unique<tm_tests::DivThrows>());
    }
};

#endif
