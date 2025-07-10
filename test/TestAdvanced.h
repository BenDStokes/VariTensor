#ifndef TEST_ADVANCED_H
#define TEST_ADVANCED_H

#include <memory>

#include "TestSet.h"
#include "varitensor/Tensor.h"
#include "varitensor/pretty_print.h"

namespace advanced_tests {

struct EpsilonMultiplication final: TestSet::Test {
    explicit EpsilonMultiplication(): Test("Epsilon Multiplication") {}

    bool run_test() override {
        const Index i{LATIN}, j{LATIN}, k{LATIN};

        const auto epsilon = levi_civita_symbol({i, j, k});
        return epsilon*epsilon == Tensor{6};
    }
};

struct EpsilonDeltaOneFree final: TestSet::Test {
    explicit EpsilonDeltaOneFree(): Test("Epsilon-Delta (1 free)") {}

    bool run_test() override {
        // given
        const Index i{LATIN}, j{LATIN}, k{LATIN}, l{LATIN};

        const Tensor epsilon = levi_civita_symbol({i, j, k});
        const Tensor delta = kronecker_delta({i, j});

        // then
        const Tensor lhs = epsilon[i,j,k] * epsilon[l,j,k];
        const Tensor rhs = delta[i,l] * 2.0;

        return lhs == rhs;
    }
};

struct EpsilonDeltaTwoFree final: TestSet::Test {
    explicit EpsilonDeltaTwoFree(): Test("Epsilon-Delta (2 free)") {}

    bool run_test() override {
        // given
         const Index i{LATIN}, j{LATIN}, k{LATIN}, l{LATIN}, m{LATIN};

         const Tensor epsilon = levi_civita_symbol({i, j, k});
         const Tensor delta = kronecker_delta({i, j});

         // when
         const Tensor lhs = epsilon[i,j,k] * epsilon[l,m,k];
         const Tensor rhs = delta[i,l] * delta[j,m] - delta[i,m] * delta[j,l];

         // then
         return lhs == rhs;
     }
};

struct CrossProductIdentity final: TestSet::Test {
     explicit CrossProductIdentity() : Test("Cross Product Identity") {}

     bool run_test() override {
         // given
         const Index i{LATIN}, j{LATIN}, k{LATIN}, l{LATIN}, m{LATIN};

         const Tensor epsilon = levi_civita_symbol({i, j, k});
         Tensor a{i}, b {i}, c{i};

         std::ranges::fill(a, 1);
         std::ranges::fill(b, 2);
         std::ranges::fill(c, 4);

         Tensor expected{i};
         std::ranges::fill(expected, 1);

         // when
         // A x B x C === B (A . C) - C (A . B)
         // E_ijk * E_klm * a_j * b_l * c_m === b_i (a_i * c_i) - c_i (a_i * b_i)
         Tensor lhs = epsilon[i, j, k] * epsilon[k, l, m] * a[j] * b[l] * c[m];
         Tensor rhs = b[i] * (a[i] * c[i]) - c[i] * (a[i] * b[i]);

         // then
         return lhs == rhs && rhs == expected;
     }
};

struct Rank9Operation final: TestSet::Test {
    explicit Rank9Operation(): Test("Rank 9 Operation") {}

    bool run_test() override {
        // given
        const Index a{LATIN}, b{LATIN}, c{LATIN},
                    d{LATIN}, e{LATIN}, f{LATIN},
                    g{LATIN}, h{LATIN}, i{LATIN};

        const auto T = Tensor{
            {a, COVARIANT},
            {b, COVARIANT},
            {c, CONTRAVARIANT},
            {d, COVARIANT},
            {e, CONTRAVARIANT}
        };
        const auto U = Tensor{
            {a, COVARIANT},
            {b, CONTRAVARIANT},
            {c, CONTRAVARIANT},
            {d, CONTRAVARIANT},
            {e, COVARIANT}
        };

        const auto V = Tensor{
            {a, COVARIANT},
            {b, COVARIANT},
            {c, CONTRAVARIANT},
            {d, COVARIANT},
            {e, COVARIANT},
            {f, CONTRAVARIANT},
            {g, CONTRAVARIANT},
            {h, COVARIANT},
            {i, COVARIANT}
        };

        std::ranges::fill(T, 2);
        std::ranges::fill(U, 3);
        std::ranges::fill(V, 18);

        // when
        const Tensor result = T[a, b, c, d, i] * U[e, f, i, g, h];

        // then
        return V[a, b, c, d, e, f, g, h, 0] == result;
    }
};

} // namespace advanced_tests

struct TestAdvanced final: TestSet {
    explicit TestAdvanced() : TestSet("Advanced Tests") {
        add_sub_test(std::make_unique<advanced_tests::EpsilonMultiplication>());
        add_sub_test(std::make_unique<advanced_tests::EpsilonDeltaOneFree>());
        add_sub_test(std::make_unique<advanced_tests::EpsilonDeltaTwoFree>());
        add_sub_test(std::make_unique<advanced_tests::CrossProductIdentity>());
        add_sub_test(std::make_unique<advanced_tests::Rank9Operation>());
    }
};

#endif
