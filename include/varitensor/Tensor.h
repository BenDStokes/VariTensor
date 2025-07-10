/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * Licence, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef VARITENSOR_TENSOR_H
#define VARITENSOR_TENSOR_H

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "impl/common.h"
#include "impl/deny.h"
#include "impl/Index.h"
#include "impl/LinkedOp_decl.h"
#include "impl/SummedOp_decl.h"
#include "impl/View_decl.h"

namespace varitensor {
namespace impl {

template<typename... Types>
concept AllInt_c = (... && std::is_same_v<Types, int>);

template<typename T>
concept Indexable_c = std::is_integral_v<T> || std::is_same_v<T, Index> || std::is_same_v<T, Interval>;

const std::string TENSOR_DEFAULT_NAME = "VariTensor";

enum TensorClass {
    TENSOR,
    METRIC_TENSOR,
    KRONECKER_DELTA,
    LEVI_CIVITA_SYMBOL,
    CHRISTOFFEL_SYMBOL
};

} // namespace impl

using Indexables = std::vector<std::variant<int, Index>>;

class Tensor {
public:
// =================================================================================================
//                                                                                         c/dtors |
// =================================================================================================

    Tensor(const std::initializer_list<VarianceQualifiedIndex> vq_indices):
        Tensor{impl::TENSOR_DEFAULT_NAME, vq_indices}
    {}

    Tensor(std::string name, const std::initializer_list<VarianceQualifiedIndex> vq_indices):
        m_size{1}, m_name{std::move(name)}
    {
        from_container(vq_indices);
    }

    Tensor(const std::initializer_list<Index> indices):
        Tensor{impl::TENSOR_DEFAULT_NAME, indices}
    {}

    Tensor(std::string name, const std::initializer_list<Index> indices):
        m_size{1}, m_name{std::move(name)}
    {
        std::vector<VarianceQualifiedIndex> vq_indices;
        for (const auto& index: indices) vq_indices.emplace_back(index, COVARIANT);
        from_container(vq_indices);
    }

    explicit Tensor(const std::vector<VarianceQualifiedIndex>& vq_indices):
        Tensor{impl::TENSOR_DEFAULT_NAME, vq_indices}
    {}

    Tensor(std::string name, const std::vector<VarianceQualifiedIndex>& vq_indices):
        m_size{1}, m_name{std::move(name)}
    {
        from_container(vq_indices);
    }

    explicit Tensor(const std::vector<Index>& indices):
        Tensor{impl::TENSOR_DEFAULT_NAME, indices}
    {}

    Tensor(std::string name, const std::vector<Index>& indices):
        m_size{1}, m_name{std::move(name)}
    {
        std::vector<VarianceQualifiedIndex> vq_indices;
        for (const auto& index: indices) vq_indices.emplace_back(index, COVARIANT);
        from_container(vq_indices);
    }

    explicit Tensor(const double value):
        Tensor{impl::TENSOR_DEFAULT_NAME, value}
    {}

    Tensor(std::string name, const double value):
        m_size{1}, m_name{std::move(name)}
    {
        m_data = new double[1]; // still use array-new so that delete[] works in the dtor
        *m_data = value;
    }

    ~Tensor() noexcept {
        delete[] m_data;
    }

    template<impl::Expression_c E>
    Tensor(E&& expression): // NOLINT - we want implicit conversion
        m_size{1}, m_name{impl::TENSOR_DEFAULT_NAME}
    {
        auto iterator = expression.begin();

        m_dimensions = iterator.dimensions();
        size_t width = 1;
        for (auto& dimension: m_dimensions) {
            dimension.width = width;
            width *= dimension.size();
        }

        m_size = iterator.size();
        m_data = new double[m_size];

        fill(m_data, expression, iterator);
    }

// =================================================================================================
//                                                                                     copy / move |
// =================================================================================================

    Tensor(const Tensor& other):
        m_dimensions{other.m_dimensions},
        m_size{other.m_size},
        m_name{other.m_name},
        m_tensor_class{other.m_tensor_class}
    {
        m_data = new double[m_size];
        std::memcpy(m_data, other.m_data, m_size*sizeof(double));
    }

    Tensor& operator=(const Tensor& other){ // NOLINT - self assigment is handled fine
        m_name = other.m_name;
        m_dimensions = other.m_dimensions;
        m_size = other.m_size;
        m_tensor_class = other.m_tensor_class;

        const auto new_data = new double[m_size];
        std::memcpy(new_data, other.m_data, m_size*sizeof(double));
        delete[] m_data;
        m_data = new_data;

        return *this;
    }

    Tensor(Tensor&& other) noexcept:
        m_dimensions{std::move(other.m_dimensions)},
        m_size{other.m_size},
        m_data{other.m_data},
        m_name{std::move(other.m_name)},
        m_tensor_class{other.m_tensor_class}
    {
        other.m_data = nullptr;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        m_name = std::move(other.m_name);
        m_dimensions = std::move(other.m_dimensions);
        m_size = other.m_size;
        m_tensor_class = other.m_tensor_class;

        m_data = other.m_data;
        other.m_data = nullptr;

        return *this;
    }

// =================================================================================================
//                                                                                       Conversion |
// =================================================================================================

    explicit operator double() const {
        impl::deny(m_size > 1, "Attempt to convert non-scalar tensor to double");
        return *m_data;
    }

// =================================================================================================
//                                                                                        Indexing |
// =================================================================================================

    // short-circuit for scalar
    template<impl::Indexable_c... Indices>
    double& operator[](Indices...) const requires (sizeof...(Indices) == 0) {
        return *m_data;
    }

    // short-circuit for all int indices
    template<impl::Indexable_c... Indices>
    requires (impl::AllInt_c<Indices...> && sizeof...(Indices) > 0)
    double& operator[](Indices... indices) const {
        auto data = m_data;

        size_t n = 0;
        for(const auto index : {indices...}) {
            impl::deny(n >= m_dimensions.size(), "Indexing dimension mismatch!");
            impl::deny(index < 0, "Indices cannot be less than zero!");
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
                   "Indexing dimension mismatch!");

        size_t n{0};
        size_t offset{0};
        impl::Dimensions passed_indices;
        construct_passed_indices(n, offset, passed_indices, indices...);

        return View{*this, m_data + offset, passed_indices};
    }

    View operator[](Indexables indices) const {
        impl::deny(indices.size() != m_dimensions.size(), "Indexing dimension mismatch!");

        size_t offset{0};
        impl::Dimensions passed_indices;
        for (size_t i = 0; i < indices.size(); ++i) {
            if (std::holds_alternative<Index>(indices[i])) {
                impl::deny(std::get<Index>(indices[i]).size() > m_dimensions[i].index.size(),
                           "Index size mismatch");
                passed_indices.emplace_back(
                    std::get<Index>(indices[i]),
                    m_dimensions[i].variance,
                    m_dimensions[i].width
                );
            }
            else {
                impl::deny(std::get<int>(indices[i]) >= m_dimensions[i].index.size(),
                           "Index size mismatch");
                offset += std::get<int>(indices[i]) * m_dimensions[i].width;
            }
        }

        return View{*this, m_data + offset, passed_indices};
    }

    double& operator[](const std::vector<int>& indices) const {
        impl::deny(indices.size() > m_dimensions.size(), "Indexing dimension mismatch!");
        auto data = m_data;

        int n = 0;
        for(const auto index : indices) {
            impl::deny(index < 0, "Indices cannot be less than zero!");
            impl::deny(index >= m_dimensions[n].index.size(), "Index size mismatch!");

            data += m_dimensions[n].width * index;
            ++n;
        }

        return *data;
    }

// =================================================================================================
//                                                                                     information |
// =================================================================================================

    [[nodiscard]] std::string name() const {return m_name;}
    [[nodiscard]] size_t size() const {return m_size;}
    [[nodiscard]] size_t size(const int dimension) const {return m_dimensions[dimension].index.size();}
    [[nodiscard]] int rank() const {return static_cast<int> (m_dimensions.size());}
    [[nodiscard]] bool is_scalar() const {return m_size == 1;}
    [[nodiscard]] bool is_metric() const {return m_tensor_class == impl::METRIC_TENSOR;}

    [[nodiscard]] std::vector<Index> indices() const {
        std::vector<Index> indices;
        for (const auto& dimension: m_dimensions) indices.emplace_back(dimension.index);
        return indices;
    }

    [[nodiscard]] std::vector<VarianceQualifiedIndex> qualified_indices() const {
        std::vector<VarianceQualifiedIndex> qualified_indices;
        for (const auto& dimension: m_dimensions) qualified_indices.emplace_back(dimension.index, dimension.variance);
        return qualified_indices;
    }

    [[nodiscard]] Variance variance(const Index& index) const {
        for (const auto& dimension: m_dimensions) {
            if (dimension.index == index){
                return dimension.variance;
            }
        }
        throw std::logic_error("Missing index when finding variance!");
    }

    [[nodiscard]] Variance variance(const int index) const {
        impl::deny(static_cast<size_t> (index) >= m_dimensions.size(), "Index out of bounds!");
        return m_dimensions[index].variance;
    }

    [[nodiscard]] bool has_index(const Index& index) const {
        return std::ranges::any_of(m_dimensions, [&](const auto& dimension) {return dimension.index == index;});
    }

    [[nodiscard]] size_t index_position(const Index& index) const {
        for (size_t i=0; i<m_dimensions.size(); ++i) {
            if (m_dimensions[i].index == index){
                return i;
            }
        }
        throw std::logic_error("Missing index when finding index position");
    }

    [[nodiscard]] const Index& indices(const int index) const {
        impl::deny(static_cast<size_t>(index) >= m_dimensions.size(), "Index out of bounds!");
        return m_dimensions[index].index;
    }

// ---------------------------------------------------------------------------------------- printing

    std::ostream& dump(std::ostream& ostream) const {
        // outputs a comma-separated dump of every value in the tensor
        ostream << std::to_string(*m_data);
        for(size_t i=1; i<m_size; ++i) {
            ostream << std::string{", "} << std::to_string(*(m_data + i));
        }
        ostream << "\n";

        return ostream;
    }

    friend void write_data(std::ostream& stream, const Tensor& tensor);
    friend std::ostream& pretty_print(std::ostream& ostream, const Tensor& tensor);

// =================================================================================================
//                                                                                    manipulation |
// =================================================================================================

    Tensor& set_name(const std::string& name) {
        m_name = name;
        return *this;
    }

    Tensor& transpose(const Index& index1, const Index& index2) {
        impl::deny(index1.size() != index2.size(), "Cannot transpose indices of different lengths");

        int pos1{-1}, pos2{-1};
        for (size_t i=0; i<m_dimensions.size(); ++i) {
            if (m_dimensions[i].index == index1) pos1 = static_cast<int> (i);
            else if (m_dimensions[i].index == index2) pos2 = static_cast<int> (i);
        }

        impl::deny(pos1 == -1 || pos2 == -1, "No such index to transpose");

        const auto temp_dim = m_dimensions[pos1];
        m_dimensions[pos1] = m_dimensions[pos2];
        m_dimensions[pos2] = temp_dim;

        return *this;
    }

    Tensor& relabel(const Index& first, const Index& second) {
        for (auto& dimension: m_dimensions) {
            if (dimension.index == first) {
                impl::deny(dimension.size() != second.size(), "Cannot relabel to an index with different size!");
                dimension.index = second;
                return *this;
            }
        }
        throw std::logic_error("Cannot find index to relabel!");
    }

    Tensor& set_variance(const Index& index, const Variance variance) {
        for (auto& dimension: m_dimensions) {
            if (dimension.index == index) {
                dimension.variance = variance;
                return *this;
            }
        }
        throw std::logic_error("Cannot find index!");
    }

    Tensor& raise(const Index& index) {
        return set_variance(index, CONTRAVARIANT);
    }

    Tensor& lower(const Index& index) {
        return set_variance(index, COVARIANT);
    }

// =================================================================================================
//                                                                                       iteration |
// =================================================================================================

    [[nodiscard]] View::iterator begin() const {
        return View{*this}.begin();
    }

    [[nodiscard]] View::iterator end() const {
        return View{*this}.end();
    }

    [[nodiscard]] View::const_iterator cbegin() const {
        return View{*this}.cbegin();
    }

    [[nodiscard]] View::const_iterator cend() const {
        return View{*this}.cend();
    }

// =================================================================================================
//                                                                                      arithmetic |
// =================================================================================================

    impl::SummedOp operator-() const {
        return {*this, -1.0, impl::MUL};
    }

// ---------------------------------------------------------------------------------------- addition

    friend impl::LinkedOp operator+(const Tensor& first, const Tensor& second) {
        return operation_tt<impl::LinkedOp>(first, second, impl::ADD);
    }

    friend impl::LinkedOp operator+(const Tensor& first, const double& second) {
        return operation_td<impl::LinkedOp>(first, second, impl::ADD);
    }

    friend impl::LinkedOp operator+(const double& first, const Tensor& second) {
        return operation_dt<impl::LinkedOp>(first, second, impl::ADD);
    }

// ----------------------------------------------------------------------------- addition assignment

    friend Tensor& operator+=(Tensor& first, const Tensor& second) {
        resolve_assignment(
            operation_tt<impl::LinkedOp>(first, second, impl::ADD),
            first,
            impl::ADD
        );
        return first;
    }

    friend double& operator+=(double& first, const Tensor& second) {
        first += static_cast<double>(second);
        return first;
    }

    friend Tensor& operator+=(Tensor& first, const double& second) {
        resolve_assignment(
            operation_td<impl::LinkedOp>(first, second, impl::ADD),
            first,
            impl::ADD
        );
        return first;
    }

    template<impl::Expression_c E>
    friend Tensor& operator+=(Tensor& first, E&& expression) {
        resolve_assignment(first + expression, first, impl::ADD);
        return first;
    }

// ------------------------------------------------------------------------------------- subtraction

    friend impl::LinkedOp operator-(const Tensor& first, const Tensor& second) {
        return operation_tt<impl::LinkedOp>(first, second, impl::SUB);
    }

    friend impl::LinkedOp operator-(const Tensor& first, const double& second) {
        return operation_td<impl::LinkedOp>(first, second, impl::SUB);
    }

    friend impl::LinkedOp operator-(const double& first, const Tensor& second) {
        return operation_dt<impl::LinkedOp>(first, second, impl::SUB);
    }

// -------------------------------------------------------------------------- subtraction assignment

    friend Tensor& operator-=(Tensor& first, const Tensor& second) {
        resolve_assignment(
            operation_tt<impl::LinkedOp>(first, second, impl::SUB),
            first,
            impl::SUB
        );
        return first;
    }

    friend double& operator-=(double& first, const Tensor& second) {
        first += static_cast<double>(second);
        return first;
    }

    friend Tensor& operator-=(Tensor& first, const double& second) {
        resolve_assignment(
            operation_td<impl::LinkedOp>(first, second, impl::SUB),
            first,
            impl::SUB
        );
        return first;
    }

    template<impl::Expression_c E>
    friend Tensor& operator-=(Tensor& first, E&& expression) {
        resolve_assignment(first - expression, first, impl::SUB);
        return first;
    }

// ---------------------------------------------------------------------------------- multiplication

    friend impl::SummedOp operator*(const Tensor& first, const Tensor& second) {
        return operation_tt<impl::SummedOp>(first, second, impl::MUL);
    }

    friend impl::SummedOp operator*(const Tensor& first, const double& second) {
        return operation_td<impl::SummedOp>(first, second, impl::MUL);
    }

    friend impl::SummedOp operator*(const double& first, const Tensor& second) {
        return operation_dt<impl::SummedOp>(first, second, impl::MUL);
    }

// ----------------------------------------------------------------------- multiplication assignment

    friend Tensor& operator*=(Tensor& first, const Tensor& second) {
        first = first * second; // in general, multiplying by a tensor can change the result completely
        return first;
    }

    friend double& operator*=(double& first, const Tensor& second) {
        impl::deny(second.m_size > 1, "Cannot multiply-assign a non-scalar tensor to a double!");
        first *= static_cast<double>(second);
        return first;
    }

    friend Tensor& operator*=(Tensor& first, const double& second) {
        resolve_assignment(
            operation_td<impl::SummedOp>(first, second, impl::MUL),
            first,
            impl::MUL
        );
        return first;
    }

    template<impl::Expression_c E>
    friend Tensor& operator*=(Tensor& first, E&& expression) {
        resolve_assignment(first * expression, first, impl::MUL);
        return first;
    }

// ---------------------------------------------------------------------------------------- division

    friend impl::SummedOp operator/(const Tensor& first, const Tensor& second) {
        impl::deny(second.m_size != 1, "Cannot divide by non-scalar tensor!");
        return {first, *second.m_data, impl::DIV};
    }

    friend impl::SummedOp operator/(const Tensor& first, const double& second) {
        return operation_td<impl::SummedOp>(first, second, impl::DIV);
    }

    friend impl::SummedOp operator/(const double& first, const Tensor& second) {
        impl::deny(second.m_size != 1, "Cannot divide by non-scalar tensor!");
        return {first, second, impl::DIV};
    }

// ----------------------------------------------------------------------------- division assignment

    friend Tensor& operator/=(Tensor& first, const Tensor& second) {
        impl::deny(second.m_size != 1, "Cannot divide by non-scalar tensor!");
        return first /= *second.m_data;
    }

    friend double& operator/=(double& first, const Tensor& second) {
        impl::deny(second.m_size != 1, "Cannot divide by non-scalar tensor!");
        return first;
    }

    friend Tensor& operator/=(Tensor& first, const double& second) {
        resolve_assignment(
            operation_td<impl::SummedOp>(first, second, impl::DIV),
            first,
            impl::DIV
        );
        return first;
    }

    template<impl::Expression_c E>
    friend Tensor& operator/=(Tensor& first, E&& expression) {
        impl::deny(!expression.is_scalar(), "Cannot divide by non-scalar tensor!");
        return first += expression.deref();
    }

// =================================================================================================
//                                                                                           logic |
// =================================================================================================

    friend bool operator==(const Tensor& first, const Tensor& second) {
        try {
            // if the validation is turned off, operator-() will not throw on disagreement, but we still need to check here
            if constexpr (!VARITENSOR_VALIDATION_ON) assert_agreement(first.m_dimensions, second.m_dimensions); // NOLINT

            for (const auto& value: Tensor{first - second}) {
                if (value != 0) return false;
            }
        }
        catch (std::logic_error&) {
            return false;
        }

        return true;
    }

    friend bool operator==(const Tensor& first, const double& second) {
        if (first.size() != 1) return false;
        return *first.m_data == second;
    }

    friend bool operator==(const double& first, const Tensor& second) {
        if (second.size() != 1) return false;
        return first == *second.m_data;
    }

private:
    friend class View;
    friend class impl::LinkedOp;
    friend class impl::SummedOp;

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

    template<typename T>
    static Tensor make_levi_civita_symbol(const T& indices) {
        impl::deny(indices.size() <= 1,
                   "Levi-Civita Symbol must have at least 2 indices!");

        const auto expected_size = indices.begin()->index.size();
        for (const auto& [index, _]: indices) {
            impl::deny(index.size() != expected_size,
                       "Indices to Levi-Civita Symbol must all the the same size!");
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
            if ((inversions / 2) % 2) epsilon[permutation] = -1;
            else epsilon[permutation] = 1;
        }
        while (std::ranges::next_permutation(permutation).found);

        return epsilon;
    }

    template<typename T>
    static Tensor make_kronecker_delta(const T& indices) {
        impl::deny(indices.size() <= 1,
                   "Kronecker Delta must have at least 2 indices!");

        const auto expected_size = indices.begin()->index.size();
        for (const auto& [index, _]: indices) {
            impl::deny(index.size() != expected_size,
                       "Indices to Kronecker Delta must all the the same size!");
        }

        auto delta = Tensor{"delta", indices};
        delta.set_tensor_class(impl::KRONECKER_DELTA);

        // The diagonal memory locations are multiples of the sum of the index widths; for a
        // contiguous, symmetric tensor, this sum is given by the geometric series:
        // sum[n=0 to rank](index_length^n)
        const auto index_length_sum = static_cast<size_t>(
            (pow(expected_size, delta.rank()) - 1) / (expected_size - 1)
        );

        for (int i=0; i<expected_size; ++i) {
            delta.m_data[i * index_length_sum] = 1;
        }

        return delta;
    }

// =================================================================================================
//                                                                                    data members |
// =================================================================================================

    impl::Dimensions m_dimensions;
    size_t m_size;
    double* m_data{nullptr};
    std::string m_name;
    impl::TensorClass m_tensor_class{impl::TENSOR};

// =================================================================================================
//                                                                                         helpers |
// =================================================================================================

    void set_tensor_class(const impl::TensorClass tensor_class) {
        m_tensor_class = tensor_class;
    }

    static void assert_agreement(const impl::Dimensions& first, const impl::Dimensions& second) {
        if (first.size() != second.size()) throw std::length_error("Tensor rank does not agree");
        for (auto& dim1: first) {
            for (auto& dim2: second) {
                if (dim1.index == dim2.index) {
                    if (dim1.variance == dim2.variance) goto agreement_loop_exit;
                    throw std::domain_error("Cannot add with mismatched index variance");
                }
            }
            throw std::domain_error("Cannot add tensors with different indices");
            agreement_loop_exit:; // goto is the best way to do for-else in c++
        }
    }

    static void assert_agreement_static(const impl::Dimensions& first, const impl::Dimensions& second) {
        if constexpr (VARITENSOR_VALIDATION_ON) {
            assert_agreement(first, second);
        }
    }

    template<typename T>
    void from_container(const T& variance_qualified_indices) {
        for(auto& vqi: variance_qualified_indices) {
            for (auto& dimension: m_dimensions) {
                impl::deny(dimension.index.id() == vqi.index.id(),
                     "Cannot initialize tensor with repeated indices"
                );
            }
            m_dimensions.emplace_back(vqi.index, vqi.variance, m_size);
            m_size *= vqi.index.size();
        }
        m_data = new double[m_size];
        for (size_t i = 0; i<m_size; ++i) m_data[i] = 0;
    }

    template<impl::Expression_c E, impl::ExpressionIterator_c I>
    static void fill(double* running_ptr, const E& expression, I& iterator) {
        for (auto end = expression.end(); iterator != end; ++iterator) {
            *running_ptr++ = iterator.deref();
        }
    }

// ---------------------------------------------------------------------------- arithmetic operation

    template<typename T>
    static T operation_tt(const Tensor& first, const Tensor& second, impl::Operation operation) {
        if (first.m_size == 1 && second.m_size == 1) return {*first.m_data, *second.m_data, operation};
        if (first.m_size == 1) return {*first.m_data, second, operation};
        if (second.m_size == 1) return {first, *second.m_data, operation};

        if (operation == impl::ADD || operation == impl::SUB) {
            assert_agreement_static(first.m_dimensions, second.m_dimensions);
        }
        return {first, second, operation};
    }

    template<typename T>
    static T operation_td(const Tensor& first, const double& second, impl::Operation operation) {
        if (first.m_size == 1) return {*first.m_data, second, operation};
        return {first, second, operation};
    }

    template<typename T>
    static T operation_dt(const double& first, const Tensor& second, impl::Operation operation) {
        if (second.m_size == 1) return {*second.m_data, second, operation};
        return {first, second, operation};
    }

    template<impl::Expression_c E>
    static void resolve_assignment(E&& expression, Tensor& tensor, const impl::Operation operation) {
        auto iterator = expression.begin();
        if (operation == impl::SUB || operation == impl::ADD) assert_agreement_static(tensor.m_dimensions, iterator.dimensions());
        fill(tensor.m_data, expression, iterator);
    }

// =================================================================================================
//                                                             variadic overloads (for operator[]) |
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

#include "impl/increment_positions.h"
#include "impl/LinkedOp_impl.h"
#include "impl/pre_defined.h"
#include "impl/View_impl.h"
#include "impl/SummedOp_impl.h"

#endif
