#ifndef TEST_ADVANCED_H
#define TEST_ADVANCED_H

#include <memory>

#include "TestSet.h"
#include "varitensor/Tensor.h"
#include "varitensor/pre_defined.h"

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
        const Index i{LATIN}, j{LATIN}, k{LATIN},
                    l{LATIN}, m{LATIN};

        const Tensor epsilon = levi_civita_symbol({i, j, k});
        Tensor a{i}, b {i}, c{i};

        a[0] =  1; b[0] = -1; c[0] =  3;
        a[1] =  2; b[1] =  4; c[1] =  2;
        a[2] = -3; b[2] = -1; c[2] = -1;

        Tensor expected{i};
        expected[0] = -40; expected[1] = 20; expected[2] = 0;

        // when
        // A x B x C === B (A . C) - C (A . B)
        // E_ijk * E_klm * a_j * b_l * c_m === b_j (a_i * c_i) - c_j (a_i * b_i)
        Tensor lhs = epsilon[i, j, k] * epsilon[k, l, m] * a[j] * b[l] * c[m];
        Tensor rhs = b[i] * (a[j] * c[j]) - c[i] * (a[j] * b[j]);

        // then
        return lhs == expected && rhs == expected && lhs == rhs;
    }
};

struct ComplexTree final: TestSet::Test {
    explicit ComplexTree() : Test("Complex Tree") {}

    bool run_test() override {
        // given
        const Index i{LATIN}, j{LATIN}, k{LATIN};

        const Tensor epsilon = levi_civita_symbol({i, j, k});
        Tensor a{i}, b {i}, c{i}, d{i};

        a[0] =  1; b[0] = -1; c[0] =  3; d[0] =  1;
        a[1] =  2; b[1] =  4; c[1] =  2; d[1] = -5;
        a[2] = -3; b[2] = -1; c[2] = -1; d[2] =  5;

        Tensor expected{k, i};
        expected[0, 0] = -51;
        expected[1, 0] = -102;
        expected[2, 0] = 153;
        expected[0, 1] = -141;
        expected[1, 1] = -282;
        expected[2, 1] = 423;
        expected[0, 2] = 229;
        expected[1, 2] = 458;
        expected[2, 2] = -687;

        // when
        Tensor result =
            a[k] * (
                c[i] * (a[j] * b[j]) +
                a[i] * (d[j] * c[j]) -
                d[i]
            ) -
            a[k] * (
                b[i] -
                a[i] * (
                    b[j] * c[j] +
                    d[j] * (a[j] - d[j])
                )
            );

        // then
        return result == expected;
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
        add_sub_test(std::make_unique<advanced_tests::ComplexTree>());
        add_sub_test(std::make_unique<advanced_tests::Rank9Operation>());
    }
};

#endif
