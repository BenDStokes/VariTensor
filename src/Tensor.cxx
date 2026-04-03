/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * Licence, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "varitensor/Tensor.h"

namespace varitensor {

// =================================================================================================
//                                                                                         c/dtors |
// =================================================================================================

// ---------------------------------------------------------------------- variance-qualified indices

Tensor::Tensor(const std::initializer_list<VarianceQualifiedIndex> vq_indices):
    Tensor{impl::TENSOR_DEFAULT_NAME, vq_indices}
{}

Tensor::Tensor(const std::initializer_list<VarianceQualifiedIndex> vq_indices, const double initial_value):
    Tensor{impl::TENSOR_DEFAULT_NAME, vq_indices, initial_value}
{}

Tensor::Tensor(std::string name, const std::initializer_list<VarianceQualifiedIndex> vq_indices):
    m_name{std::move(name)}
{
    from_variance_qualified_indices(vq_indices);
    m_data = impl::allocate_zeroed(m_size);
}

Tensor::Tensor(std::string name, const std::initializer_list<VarianceQualifiedIndex> vq_indices, const double initial_value):
    m_name{std::move(name)}
{
    from_variance_qualified_indices(vq_indices);
    m_data = impl::allocate(m_size, initial_value);
}

// ----------------------------------------------------------------------------- unqualified indices

Tensor::Tensor(const std::initializer_list<Index> indices):
    Tensor{impl::TENSOR_DEFAULT_NAME, indices}
{}

Tensor::Tensor(const std::initializer_list<Index> indices, const double initial_value):
    Tensor{impl::TENSOR_DEFAULT_NAME, indices, initial_value}
{}

Tensor::Tensor(std::string name, const std::initializer_list<Index> indices):
    m_name{std::move(name)}
{
    std::vector<VarianceQualifiedIndex> vq_indices;
    for (const auto& index: indices) vq_indices.emplace_back(index, COVARIANT);
    from_variance_qualified_indices(vq_indices);
    m_data = impl::allocate_zeroed(m_size);

}

Tensor::Tensor(std::string name, const std::initializer_list<Index> indices, const double initial_value):
    m_name{std::move(name)}
{
    std::vector<VarianceQualifiedIndex> vq_indices;
    for (const auto& index: indices) vq_indices.emplace_back(index, COVARIANT);
    from_variance_qualified_indices(vq_indices);
    m_data = impl::allocate(m_size, initial_value);
}

// ------------------------------------------------------------------- variance-qualified index list

Tensor::Tensor(const std::vector<VarianceQualifiedIndex>& vq_indices):
    Tensor{impl::TENSOR_DEFAULT_NAME, vq_indices}
{}

Tensor::Tensor(const std::vector<VarianceQualifiedIndex>& vq_indices, const double initial_value):
    Tensor{impl::TENSOR_DEFAULT_NAME, vq_indices, initial_value}
{}

Tensor::Tensor(std::string name, const std::vector<VarianceQualifiedIndex>& vq_indices):
    m_name{std::move(name)}
{
    from_variance_qualified_indices(vq_indices);
    m_data = impl::allocate_zeroed(m_size);
}

Tensor::Tensor(std::string name, const std::vector<VarianceQualifiedIndex>& vq_indices, const double initial_value):
    m_name{std::move(name)}
{
    from_variance_qualified_indices(vq_indices);
    m_data = impl::allocate(m_size, initial_value);
}

// -------------------------------------------------------------------------- unqualified index list

Tensor::Tensor(const std::vector<Index>& indices):
    Tensor{impl::TENSOR_DEFAULT_NAME, indices}
{}

Tensor::Tensor(const std::vector<Index>& indices, const double initial_value):
    Tensor{impl::TENSOR_DEFAULT_NAME, indices, initial_value}
{}

Tensor::Tensor(std::string name, const std::vector<Index>& indices):
    m_name{std::move(name)}
{
    std::vector<VarianceQualifiedIndex> vq_indices;
    for (const auto& index: indices) vq_indices.emplace_back(index, COVARIANT);
    from_variance_qualified_indices(vq_indices);
    m_data = impl::allocate_zeroed(m_size);
}

Tensor::Tensor(std::string name, const std::vector<Index>& indices, const double initial_value):
    m_name{std::move(name)}
{
    std::vector<VarianceQualifiedIndex> vq_indices;
    for (const auto& index: indices) vq_indices.emplace_back(index, COVARIANT);
    from_variance_qualified_indices(vq_indices);
    m_data = impl::allocate(m_size, initial_value);
}

// ------------------------------------------------------------------------------------------ scalar

Tensor::Tensor(const double initial_value):
    Tensor{impl::TENSOR_DEFAULT_NAME, initial_value}
{}

Tensor::Tensor(std::string name, const double initial_value):
    m_name{std::move(name)}
{
    m_data = impl::allocate(1);
    *m_data = initial_value;
}

// =================================================================================================
//                                                                                     copy / move |
// =================================================================================================

Tensor::Tensor(const Tensor& other):
    m_dimensions{other.m_dimensions},
    m_size{other.m_size},
    m_name{other.m_name},
    m_tensor_class{other.m_tensor_class}
{
    m_data = impl::allocate(m_size);
    impl::copy(m_data.get(), other.m_data.get(), m_size);
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (other.m_data != m_data) { // if this isn't copy-to-self
        m_name = other.m_name;
        m_dimensions = other.m_dimensions;
        m_size = other.m_size;
        m_tensor_class = other.m_tensor_class;

        impl::DoublePtr new_data = impl::allocate(m_size);
        impl::copy(new_data.get(), other.m_data.get(), m_size);
        m_data.swap(new_data);
    }

    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept:
    m_dimensions{std::move(other.m_dimensions)},
    m_size{other.m_size},
    m_data{std::move(other.m_data)},
    m_name{std::move(other.m_name)},
    m_tensor_class{other.m_tensor_class}
{
    other.m_data = nullptr;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (other.m_data != m_data) { // if this isn't move-to-self
        m_name = std::move(other.m_name);
        m_dimensions = std::move(other.m_dimensions);
        m_size = other.m_size;
        m_tensor_class = other.m_tensor_class;

        m_data = std::move(other.m_data);
        other.m_data = nullptr;
    }

    return *this;
}

// =================================================================================================
//                                                                                       Conversion |
// =================================================================================================

Tensor::operator double() const {
    impl::deny(m_size > 1, "Attempt to convert non-scalar tensor to double");
    return *m_data;
}

// =================================================================================================
//                                                                                        Indexing |
// =================================================================================================

View Tensor::operator[](Indexables indices) const {
    impl::deny(indices.size() != m_dimensions.size(), "Indexing dimension mismatch");

    size_t offset{0};
    size_t width{1};
    impl::Dimensions passed_indices;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (std::holds_alternative<Index>(indices[i])) {
            impl::deny(std::get<Index>(indices[i]).size() > m_dimensions[i].index.size(),
                            "Index size mismatch");
            passed_indices.emplace_back(
                std::get<Index>(indices[i]),
                m_dimensions[i].variance,
                width
            );
            width *= std::get<Index>(indices[i]).size();
        }
        else {
            impl::deny(std::get<int>(indices[i]) >= m_dimensions[i].index.size(),
                            "Index size mismatch");
            offset += std::get<int>(indices[i]) * width;
        }
    }

    return View{*this, m_data.get() + offset, passed_indices};
}

double& Tensor::operator[](const std::vector<int>& indices) const {
    impl::deny(indices.size() > m_dimensions.size(), "Indexing dimension mismatch");
    auto data = m_data.get();

    int n = 0;
    for(const auto index : indices) {
        impl::deny(index < 0, "Indices cannot be less than zero");
        impl::deny(index >= m_dimensions[n].index.size(), "Index size mismatch");

        data += m_dimensions[n].width * index;
        ++n;
    }

    return *data;
}

// =================================================================================================
//                                                                                     information |
// =================================================================================================

[[nodiscard]] std::string Tensor::name() const {return m_name;}
[[nodiscard]] size_t Tensor::size() const {return m_size;}
[[nodiscard]] size_t Tensor::size(const int dimension) const {return m_dimensions[dimension].index.size();}
[[nodiscard]] int Tensor::rank() const {return static_cast<int> (m_dimensions.size());}
[[nodiscard]] bool Tensor::is_scalar() const {return m_size == 1;}
[[nodiscard]] double Tensor::get_scalar() const {return *m_data;}
[[nodiscard]] bool Tensor::is_metric() const {return m_tensor_class == impl::METRIC_TENSOR;}

[[nodiscard]] std::vector<Index> Tensor::indices() const {
    std::vector<Index> indices;
    for (const auto& dimension: m_dimensions) indices.emplace_back(dimension.index);
    return indices;
}

[[nodiscard]] std::vector<VarianceQualifiedIndex> Tensor::qualified_indices() const {
    std::vector<VarianceQualifiedIndex> qualified_indices;
    for (const auto& dimension: m_dimensions) qualified_indices.emplace_back(dimension.index, dimension.variance);
    return qualified_indices;
}

[[nodiscard]] Variance Tensor::variance(const Index& index) const {
    for (const auto& dimension: m_dimensions) {
        if (dimension.index == index){
            return dimension.variance;
        }
    }
    throw TensorLogicError("Missing index when finding variance!");
}

[[nodiscard]] Variance Tensor::variance(const int index) const {
    impl::deny(static_cast<size_t> (index) >= m_dimensions.size(), "Index out of bounds");
    return m_dimensions[index].variance;
}

[[nodiscard]] bool Tensor::has_index(const Index& index) const {
    return std::ranges::any_of(m_dimensions, [&](const auto& dimension) {return dimension.index == index;});
}

[[nodiscard]] size_t Tensor::index_position(const Index& index) const {
    for (size_t i=0; i<m_dimensions.size(); ++i) {
        if (m_dimensions[i].index == index){
            return i;
        }
    }
    throw TensorLogicError("Missing index when finding index position");
}

[[nodiscard]] const Index& Tensor::indices(const int index) const {
    impl::deny(static_cast<size_t>(index) >= m_dimensions.size(), "Index out of bounds");
    return m_dimensions[index].index;
}

// =================================================================================================
//                                                                                    manipulation |
// =================================================================================================

Tensor& Tensor::set_name(const std::string& name) {
    m_name = name;
    return *this;
}

Tensor& Tensor::transpose(const Index& first, const Index& second) {
    /* Note that we can't just swap the indices as this would mess up the SIMD memory
     * alignment; we have to perform a true transposition.
     */

    impl::deny(rank() < 2, "Tensor has too few indices for transposition");
    impl::deny(first == second, "Cannot transpose identical indices");
    impl::deny(first.size() != second.size(), "Cannot transpose indices of different lengths");

    // differentiate between the transposition and static dimensions
    impl::Dimension* dim1{nullptr};
    impl::Dimension* dim2{nullptr};
    impl::Dimensions static_dims;
    for (auto & m_dimension : m_dimensions) {
        if      (m_dimension.index == first)  dim1 = &m_dimension;
        else if (m_dimension.index == second) dim2 = &m_dimension;
        else static_dims.emplace_back(m_dimension);
    }

    impl::deny(!(dim1 && dim2), "Missing index when transposing");

    const auto transpose_size = dim1->index.size(); // NOLINT - we check for nullptr in the line above
    auto transpose_slice = [&transpose_size, &dim1, &dim2](double* data) {
        for (int j = 0; j < transpose_size - 1; ++j) {
            for (int i = j + 1; i < transpose_size; ++i) {
                std::swap(
                    *(data + i*dim1->width + j*dim2->width),
                    *(data + i*dim2->width + j*dim1->width)
                );
            }
        }
    };

    if (static_dims.empty()) transpose_slice(m_data.get());
    else {
        auto static_iter = begin();
        std::vector<int> static_positions(static_dims.size());
        do transpose_slice(static_iter.data());
        while (impl::increment_positions(static_positions, static_dims, static_iter));
    }

    std::swap(dim1->index, dim2->index);
    std::swap(dim1->variance, dim2->variance);
    // the width is intentionally NOT swapped to maintain memory alignment

    return *this;
}

Tensor& Tensor::relabel(const Index& old_index, const Index& new_index) {
    for (auto& dimension: m_dimensions) {
        if (dimension.index == old_index) {
            impl::deny(dimension.size() != new_index.size(),
                 "Cannot relabel to an index with different size");
            dimension.index = new_index;
            return *this;
        }
    }
    throw TensorLogicError("Cannot find index to relabel!");
}

Tensor& Tensor::set_variance(const Index& index, const Variance variance) {
    for (auto& dimension: m_dimensions) {
        if (dimension.index == index) {
            dimension.variance = variance;
            return *this;
        }
    }
    throw TensorLogicError("Cannot find index!");
}

Tensor& Tensor::raise(const Index& index) {
    return set_variance(index, CONTRAVARIANT);
}

Tensor& Tensor::lower(const Index& index) {
    return set_variance(index, COVARIANT);
}

// =================================================================================================
//                                                                                       iteration |
// =================================================================================================

[[nodiscard]] View::iterator Tensor::begin() const {
    return View{*this}.begin();
}

[[nodiscard]] View::iterator Tensor::end() const {
    return View{*this}.end();
}

[[nodiscard]] View::const_iterator Tensor::cbegin() const {
    return View{*this}.cbegin();
}

[[nodiscard]] View::const_iterator Tensor::cend() const {
    return View{*this}.cend();
}

// =================================================================================================
//                                                                                      arithmetic |
// =================================================================================================

impl::ProductOp Tensor::operator-() const {
    return {*this, -1.0, impl::MUL};
}

// ---------------------------------------------------------------------------------------- addition

impl::LinkedOp operator+(const Tensor& first, const Tensor& second) {
    return {first, second, impl::ADD};
}

impl::LinkedOp operator+(const Tensor& first, const double& second) {
    return {first, second, impl::ADD};
}

impl::LinkedOp operator+(const double& first, const Tensor& second) {
    return {first, second, impl::ADD};
}

// ----------------------------------------------------------------------------- addition assignment

Tensor& operator+=(Tensor& first, const Tensor& second) {
    impl::LinkedOp{first, second, impl::ADD}.populate(first, false);
    return first;
}

double& operator+=(double& first, const Tensor& second) {
    impl::deny(second.m_size != 1, "Cannot add double and non-scalar tensor");
    return first += *second.m_data;
}

Tensor& operator+=(Tensor& first, const double& second) {
    impl::deny(first.m_size != 1, "Cannot add non-scalar tensor and double");
    *first.m_data += second;
    return first;
}

// ------------------------------------------------------------------------------------- subtraction

impl::LinkedOp operator-(const Tensor& first, const Tensor& second) {
    return {first, second, impl::SUB};
}

impl::LinkedOp operator-(const Tensor& first, const double& second) {
    return {first, second, impl::SUB};
}

impl::LinkedOp operator-(const double& first, const Tensor& second) {
    return {first, second, impl::SUB};
}

// -------------------------------------------------------------------------- subtraction assignment

Tensor& operator-=(Tensor& first, const Tensor& second) {
    impl::LinkedOp{first, second, impl::SUB}.populate(first, false);
    return first;
}

double& operator-=(double& first, const Tensor& second) {
    impl::deny(second.m_size != 1, "Cannot subtract double and non-scalar tensor");
    return first -= *second.m_data;
}

Tensor& operator-=(Tensor& first, const double& second) {
    impl::deny(first.m_size != 1, "Cannot subtract non-scalar tensor and double");
    *first.m_data -= second;
    return first;
}

// ---------------------------------------------------------------------------------- multiplication

impl::ProductOp operator*(const Tensor& first, const Tensor& second) {
    return {first, second, impl::MUL};
}

impl::ProductOp operator*(const Tensor& first, const double& second) {
    return {first, second, impl::MUL};
}

impl::ProductOp operator*(const double& first, const Tensor& second) {
    return {first, second, impl::MUL};
}

// ----------------------------------------------------------------------- multiplication assignment

Tensor& operator*=(Tensor& first, const Tensor& second) {
    impl::ProductOp{first, second, impl::MUL}.populate(first);
    return first;
}

double& operator*=(double& first, const Tensor& second) {
    impl::deny(second.m_size != 1, "Cannot assign non-scalar tensor to double!");
    return first *= *second.m_data;
}

Tensor& operator*=(Tensor& first, const double& second) {
    impl::ProductOp{first, second, impl::MUL}.populate(first);
    return first;
}

// ---------------------------------------------------------------------------------------- division

impl::ProductOp operator/(const Tensor& first, const Tensor& second) {
    return impl::ProductOp{first, second, impl::DIV};
}

impl::ProductOp operator/(const Tensor& first, const double& second) {
    return {first, second, impl::DIV};
}

impl::ProductOp operator/(const double& first, const Tensor& second) {
    return {first, second, impl::DIV};
}

// ----------------------------------------------------------------------------- division assignment

Tensor& operator/=(Tensor& first, const Tensor& second) {
    impl::deny(second.m_size != 1, "Cannot divide by non-scalar tensor!");
    impl::ProductOp{first, *second.m_data, impl::DIV}.populate(first);
    return first;
}

double& operator/=(double& first, const Tensor& second) {
    impl::deny(second.m_size != 1, "Cannot divide by non-scalar tensor!");
    return first /= *second.m_data;
}

Tensor& operator/=(Tensor& first, const double& second) {
    impl::ProductOp{first, second, impl::DIV}.populate(first);
    return first;
}

// =================================================================================================
//                                                                                           logic |
// =================================================================================================

bool operator==(const Tensor& first, const Tensor& second) {
    // this operator ignores the name and tensor_class attributes
    try {
        for (const auto& value: Tensor{first - second}) {
            if (value != 0) return false; // indices MUST be linked when comparing
        }
    }
    catch (std::logic_error&) {
        return false;
    }

    return true;
}

bool operator==(const Tensor& first, const double& second) {
    if (first.size() != 1) return false;
    return *first.m_data == second;
}

bool operator==(const double& first, const Tensor& second) {
    if (second.size() != 1) return false;
    return first == *second.m_data;
}

// =================================================================================================
//                                                                                         helpers |
// =================================================================================================

void Tensor::set_tensor_class(const impl::TensorClass tensor_class) {
    m_tensor_class = tensor_class;
}

void Tensor::populate_scalar(const double scalar, const bool allocate) {
    if (allocate) { // for LinkedOp assignment, the original tensor is already set up correctly
        m_data = impl::allocate(1);
        m_size = 1;
    }

    *m_data = scalar;
}

}  // namespace varitensor
