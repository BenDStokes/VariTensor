/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * Licence, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef VARITENSOR_TENSOR_H
#define VARITENSOR_TENSOR_H

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "impl/bulk.h"
#include "impl/common.h"
#include "impl/deny.h"
#include "impl/Index.h"
#include "impl/LinkedOp.h"
#include "impl/ProductOp.h"
#include "impl/View.h"

// Not used here, but we want users who include this file to get the pre-defined tensors as well
#include "impl/pre_defined.h" // NOLINT

namespace varitensor {
namespace impl {

template<typename... Types>
concept AllInt_c = (... && std::is_same_v<Types, int>);

template<typename T>
concept Indexable_c = std::is_integral_v<T> || std::is_same_v<T, Index> || std::is_same_v<T, Interval>;

template<typename T>
concept Indices_c =
    std::is_same_v<std::decay_t<T>, std::vector<Index>> ||
    std::is_same_v<std::decay_t<T>, std::initializer_list<Index>> ||
    std::is_same_v<std::decay_t<T>, std::vector<VarianceQualifiedIndex>> ||
    std::is_same_v<std::decay_t<T>, std::initializer_list<VarianceQualifiedIndex>>;

template<typename T>
concept VarianceQualifiedIndices_c =
    std::is_same_v<std::decay_t<T>, std::vector<VarianceQualifiedIndex>> ||
    std::is_same_v<std::decay_t<T>, std::initializer_list<VarianceQualifiedIndex>>;

const std::string TENSOR_DEFAULT_NAME = "Vari-Tensor";

enum TensorClass {
    TENSOR,
    METRIC_TENSOR,
    KRONECKER_DELTA,
    LEVI_CIVITA_SYMBOL,
    CHRISTOFFEL_SYMBOL
};

} // namespace impl

using Indexable = std::variant<int, Index>;
using Indexables = std::vector<Indexable>;

class Tensor {
public:
// =================================================================================================
//                                                                                         c/dtors |
// =================================================================================================
    // General use ctors; each has 4 versions: unaccompanied, w/ initial value, w/ name, and w/ name and initial value

    // std::initializer_list<VarianceQualifiedIndex>
    Tensor(std::initializer_list<VarianceQualifiedIndex> vq_indices);
    Tensor(std::initializer_list<VarianceQualifiedIndex> vq_indices, double initial_value);
    Tensor(std::string name, std::initializer_list<VarianceQualifiedIndex> vq_indices);
    Tensor(std::string name, std::initializer_list<VarianceQualifiedIndex> vq_indices, double initial_value);

    // std::initializer_list<Index>
    Tensor(std::initializer_list<Index> indices);
    Tensor(std::initializer_list<Index> indices, double initial_value);
    Tensor(std::string name, std::initializer_list<Index> indices);
    Tensor(std::string name, std::initializer_list<Index> indices, double initial_value);

    // std::vector<VarianceQualifiedIndex>&
    explicit Tensor(const std::vector<VarianceQualifiedIndex>& vq_indices);
    Tensor(const std::vector<VarianceQualifiedIndex>& vq_indices, double initial_value);
    Tensor(std::string name, const std::vector<VarianceQualifiedIndex>& vq_indices);
    Tensor(std::string name, const std::vector<VarianceQualifiedIndex>& vq_indices, double initial_value);

    // std::vector<Index>&
    explicit Tensor(const std::vector<Index>& indices);
    Tensor(const std::vector<Index>& indices, double initial_value);
    Tensor(std::string name, const std::vector<Index>& indices);
    Tensor(std::string name, const std::vector<Index>& indices, double initial_value);

// -------------------------------------------------------------------------------------------------

    // Scalar
    explicit Tensor(double initial_value);
    Tensor(std::string name, double initial_value);

    template<typename E>
    requires impl::Expression_c<std::remove_cvref_t<E>> // "requires impl::Expression_c" to keep the forwarding reference
    Tensor(E&& expression) { // NOLINT - we want implicit conversion
        expression.populate(*this);

        if constexpr (!std::is_same_v<std::decay_t<E>, impl::LinkedOp>) {
            // for a linked operation, the widths are already correct
            size_t width = 1;
            for (auto& dimension: m_dimensions) {
                dimension.width = width;
                width *= dimension.size();
            }
        }
    }

    ~Tensor() noexcept = default;

// =================================================================================================
//                                                                                     copy / move |
// =================================================================================================

    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);

    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

// =================================================================================================
//                                                                                      Conversion |
// =================================================================================================

    explicit operator double() const;

// =================================================================================================
//                                                                                        Indexing |
// =================================================================================================

    // short-circuit for scalar
    template<impl::Indexable_c... Indices>
    double& operator[](Indices...) requires (sizeof...(Indices) == 0) {
        return *m_data;
    }

    // short-circuit for all int indices
    template<impl::Indexable_c... Indices>
    requires (impl::AllInt_c<Indices...> && sizeof...(Indices) > 0)
    double& operator[](Indices... indices) {
        auto data = m_data.get();

        size_t n = 0;
        for(const auto index : {indices...}) {
            impl::deny(n >= m_dimensions.size(), "Indexing dimension mismatch");
            impl::deny(index < 0, "Indices cannot be less than zero");
            impl::deny(index >= m_dimensions[n].index.size(), "Index size mismatch");

            data += m_dimensions[n].width * index;
            ++n;
        }

        return *data;
    }

    template<impl::Indexable_c... Indices>
    requires (!impl::AllInt_c<Indices...>)
    [[nodiscard]] View operator[](Indices... indices) const {
        impl::deny(sizeof...(indices) != m_dimensions.size(),
                        "Indexing dimension mismatch");

        size_t n{0};
        size_t offset{0};
        impl::Dimensions passed_indices;
        construct_passed_indices(n, offset, passed_indices, indices...);

        return View{*this, m_data.get() + offset, passed_indices};
    }

    View operator[](Indexables indices) const;
    double& operator[](const std::vector<int>& indices) const;

    // =================================================================================================
    //                                                                                     information |
    // =================================================================================================

    [[nodiscard]] std::string name() const;
    [[nodiscard]] size_t size() const;
    [[nodiscard]] size_t size(int dimension) const;
    [[nodiscard]] int rank() const;
    [[nodiscard]] bool is_scalar() const;
    [[nodiscard]] double get_scalar() const;
    [[nodiscard]] bool is_metric() const;
    [[nodiscard]] bool has_index(const Index& index) const;
    [[nodiscard]] size_t index_position(const Index& index) const;
    [[nodiscard]] std::vector<Index> indices() const;
    [[nodiscard]] const Index& indices(int index) const;
    [[nodiscard]] std::vector<VarianceQualifiedIndex> qualified_indices() const;
    [[nodiscard]] Variance variance(const Index& index) const;
    [[nodiscard]] Variance variance(int index) const;

    // ---------------------------------------------------------------------------------------- printing

    template<typename S>
    requires requires(S stream) {
        stream << std::string{""};
    }
    std::ostream& dump(S& ostream) const {
        // outputs a comma-separated dump of every value in the tensor
        const double* data = m_data.get();
        ostream << std::to_string(*data);
        for(size_t i=1; i<m_size; ++i) {
            ostream << std::string{", "} << std::to_string(*(data + i));
        }
        ostream << "\n";

        return ostream;
    }

    friend void write_data(std::ostream& stream, const Tensor& tensor);
    friend std::ostream& pretty_print(std::ostream& ostream, const Tensor& tensor);

// =================================================================================================
//                                                                                    manipulation |
// =================================================================================================

    Tensor& set_name(const std::string& name);
    Tensor& transpose(const Index& first, const Index& second);
    Tensor& relabel(const Index& old_index, const Index& new_index);
    Tensor& set_variance(const Index& index, Variance variance);
    Tensor& raise(const Index& index);
    Tensor& lower(const Index& index);

// =================================================================================================
//                                                                                       iteration |
// =================================================================================================

    [[nodiscard]] View::iterator begin() const;
    [[nodiscard]] View::iterator end() const;
    [[nodiscard]] View::const_iterator cbegin() const;
    [[nodiscard]] View::const_iterator cend() const;

// =================================================================================================
//                                                                                      arithmetic |
// =================================================================================================

    impl::ProductOp operator-() const;

// ---------------------------------------------------------------------------------------- addition

    friend impl::LinkedOp operator+(const Tensor& first, const Tensor& second);
    friend impl::LinkedOp operator+(const Tensor& first, const double& second);
    friend impl::LinkedOp operator+(const double& first, const Tensor& second);

// ----------------------------------------------------------------------------- addition assignment

    friend Tensor& operator+=(Tensor& first, const Tensor& second);
    friend double& operator+=(double& first, const Tensor& second);
    friend Tensor& operator+=(Tensor& first, const double& second);

    template<impl::Expression_c Expression>
    friend Tensor& operator+=(Tensor& first, Expression&& second) {
        (first + second).populate(first, false);
        return first;
    }

// ------------------------------------------------------------------------------------- subtraction

    friend impl::LinkedOp operator-(const Tensor& first, const Tensor& second);
    friend impl::LinkedOp operator-(const Tensor& first, const double& second);
    friend impl::LinkedOp operator-(const double& first, const Tensor& second);

// -------------------------------------------------------------------------- subtraction assignment

    friend Tensor& operator-=(Tensor& first, const Tensor& second);
    friend double& operator-=(double& first, const Tensor& second);
    friend Tensor& operator-=(Tensor& first, const double& second);

    template<impl::Expression_c Expression>
    friend Tensor& operator-=(Tensor& first, Expression&& expression) {
        (first - expression).populate(first, false);
        return first;
    }

// ---------------------------------------------------------------------------------- multiplication

    friend impl::ProductOp operator*(const Tensor& first, const Tensor& second);
    friend impl::ProductOp operator*(const Tensor& first, const double& second);
    friend impl::ProductOp operator*(const double& first, const Tensor& second);

// ----------------------------------------------------------------------- multiplication assignment

    friend Tensor& operator*=(Tensor& first, const Tensor& second);
    friend double& operator*=(double& first, const Tensor& second);
    friend Tensor& operator*=(Tensor& first, const double& second);

    template<impl::Expression_c Expression>
    friend Tensor& operator*=(Tensor& first, Expression&& second) {
        (first * second).populate(first);
        return first;
    }

// ---------------------------------------------------------------------------------------- division

    friend impl::ProductOp operator/(const Tensor& first, const Tensor& second);
    friend impl::ProductOp operator/(const Tensor& first, const double& second);
    friend impl::ProductOp operator/(const double& first, const Tensor& second);

// ----------------------------------------------------------------------------- division assignment

    friend Tensor& operator/=(Tensor& first, const Tensor& second);
    friend double& operator/=(double& first, const Tensor& second);
    friend Tensor& operator/=(Tensor& first, const double& second);

    template<impl::Expression_c Expression>
    friend Tensor& operator/=(Tensor& first, Expression&& second) {
        (first / second).populate(first);
        return first;
    }

// =================================================================================================
//                                                                                           logic |
// =================================================================================================

    friend bool operator==(const Tensor& first, const Tensor& second);
    friend bool operator==(const Tensor& first, const double& second);
    friend bool operator==(const double& first, const Tensor& second);

private:
// =================================================================================================
//                                                                                         friends |
// =================================================================================================

    friend class View;
    friend class impl::LinkedOp;
    friend class impl::ProductOp;

// =================================================================================================
//                                                                                    data members |
// =================================================================================================

    impl::Dimensions m_dimensions;
    size_t m_size{1};
    impl::DoublePtr m_data{nullptr, [](double*){}};
    std::string m_name{impl::TENSOR_DEFAULT_NAME};
    impl::TensorClass m_tensor_class{impl::TENSOR};

// =================================================================================================
//                                                                       pre-defined tensor makers |
// =================================================================================================

    friend Tensor levi_civita_symbol(std::initializer_list<Index> indices);
    friend Tensor levi_civita_symbol(std::initializer_list<VarianceQualifiedIndex> indices);
    friend Tensor antisymmetric_symbol(std::initializer_list<Index> indices);
    friend Tensor antisymmetric_symbol(std::initializer_list<VarianceQualifiedIndex> indices);
    friend Tensor kronecker_delta(std::initializer_list<Index> indices);
    friend Tensor kronecker_delta(std::initializer_list<VarianceQualifiedIndex> indices);
    friend Tensor metric_tensor(std::initializer_list<VarianceQualifiedIndex>, const MetricFunction&);
    // unqualified indices don't make sense for a metric tensor

    template<impl::Indices_c I>
    static Tensor make_levi_civita_symbol(const I& indices) {
        impl::deny(indices.size() <= 1,
                        "Levi-Civita Symbol must have at least 2 indices");

        const auto expected_size = indices.begin()->index.size();
        for (const auto& [index, _]: indices) {
            impl::deny(index.size() != expected_size,
                            "Indices to Levi-Civita Symbol must all the the same size");
        }

        auto epsilon = Tensor{"Epsilon",  indices};
        epsilon.set_tensor_class(impl::LEVI_CIVITA_SYMBOL);

        std::vector<int> permutation(indices.size());
        std::ranges::iota(permutation, 0);
        do {
            int inversions = 0;
            for (size_t i=0; i<permutation.size(); ++i) {
                for (size_t j=0; j<permutation.size(); ++j) {
                    if (
                        (j < i && permutation[i] < permutation[j]) ||
                        (j > i && permutation[i] > permutation[j])
                    ) {
                        ++inversions;
                    }
                }
            }
            if (inversions / 2 % 2) epsilon[permutation] = -1;
            else epsilon[permutation] = 1;
        }
        while (std::ranges::next_permutation(permutation).found);

        return epsilon;
    }

    template<impl::Indices_c I>
    static Tensor make_kronecker_delta(const I& indices) {
        impl::deny(indices.size() <= 1,
                        "Kronecker Delta must have at least 2 indices");

        const auto expected_size = indices.begin()->index.size();
        for (const auto& [index, _]: indices) {
            impl::deny(index.size() != expected_size,
                            "Indices to Kronecker Delta must all the the same size");
        }

        auto delta = Tensor{"delta", indices};
        delta.set_tensor_class(impl::KRONECKER_DELTA);

        // The diagonal memory locations are multiples of the sum of the index widths; for a
        // contiguous, symmetric tensor, this sum is given by the geometric series:
        // sum[n=0 to rank](index_length^n).
        const auto index_length_sum = static_cast<size_t>(
            (pow(expected_size, delta.rank()) - 1) / (expected_size - 1)
        );

        double* data = delta.m_data.get();
        for (int i=0; i<expected_size; ++i) {
            data[i * index_length_sum] = 1;
        }

        return delta;
    }

// =================================================================================================
//                                                                                         helpers |
// =================================================================================================

    void set_tensor_class(impl::TensorClass tensor_class);

    template<impl::VarianceQualifiedIndices_c V>
    void from_variance_qualified_indices(const V& variance_qualified_indices) {
        for(auto& vqi: variance_qualified_indices) {
            for (auto& dimension: m_dimensions) {
                impl::deny(dimension.index.id() == vqi.index.id(),
                                "Cannot initialize tensor with repeated indices");
            }
            m_dimensions.emplace_back(vqi.index, vqi.variance, m_size);
            m_size *= vqi.index.size();
        }
    }

// -------------------------------------------------------------------------------------- population

    template<typename IteratorType>
    void populate_general(IteratorType& iter, IteratorType end, const bool allocate) {
        if (allocate) {
            const size_t new_size = iter.size();
            impl::DoublePtr new_data = impl::allocate(new_size);

            double* running_ptr = new_data.get();
            for (; iter != end; ++iter) *running_ptr++ = iter.deref();

            m_dimensions = iter.dimensions();
            m_size = new_size;
            m_data.swap(new_data);
        }
        else {
            double* running_ptr = m_data.get();
            for (; iter != end; ++iter) *running_ptr++ = iter.deref();
        }
    }

    void populate_scalar(double scalar, bool allocate);

// =================================================================================================
//                                                               variadic overloads for operator[] |
// =================================================================================================

    template<impl::Indexable_c... Indices>
    void construct_passed_indices(
        size_t& n,
        size_t& offset,
        impl::Dimensions& passed_indices,
        const Index& index,
        Indices... indices
    ) const {
        impl::deny(n >= m_dimensions.size(), "Indexing dimension mismatch!");
        impl::deny(index.size() > m_dimensions[n].index.size(), "Index size mismatch");

        passed_indices.emplace_back(index, m_dimensions[n].variance, m_dimensions[n].width);
        construct_passed_indices(++n, offset, passed_indices, indices...);
    }

    template<impl::Indexable_c... Indices>
    void construct_passed_indices(
        size_t& n,
        size_t& offset,
        impl::Dimensions& passed_indices,
        const Interval& index,
        Indices... indices
    ) const {
        impl::deny(n >= m_dimensions.size(), "Indexing dimension mismatch!");
        impl::deny(index.last >= m_dimensions[n].index.size(), "Index size mismatch");

        offset += m_dimensions[n].width * index.first;
        passed_indices.emplace_back(index, m_dimensions[n].variance, m_dimensions[n].width);
        construct_passed_indices(++n, offset, passed_indices, indices...);
    }

    template<impl::Indexable_c... Indices>
    void construct_passed_indices(
        size_t& n,
        size_t& offset,
        impl::Dimensions& passed_indices,
        const int index,
        Indices... indices
    ) const {
        impl::deny(n >= m_dimensions.size(), "Indexing dimension mismatch!");
        impl::deny(index+1 > m_dimensions[n].index.size(), "Index size mismatch");

        offset += index * m_dimensions[n].width;
        construct_passed_indices(++n, offset, passed_indices, indices...);
    }

    static void construct_passed_indices(size_t&, size_t&, impl::Dimensions&) {} // terminating overload
};

}  // namespace varitensor

#endif // VARITENSOR_TENSOR_H
