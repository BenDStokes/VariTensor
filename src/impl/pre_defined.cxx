/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * Licence, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "varitensor/impl/deny.h"
#include "../../include/varitensor/pre_defined.h"
#include "varitensor/Tensor.h"

namespace varitensor {

Tensor metric_tensor(
    const std::initializer_list<VarianceQualifiedIndex> indices,
    const MetricFunction& metric_function/* = EUCLIDEAN_METRIC */
) {
    impl::deny(indices.size() != 2,
                    "Metric tensor must have exactly 2 indices");
    impl::deny(indices.begin()->index == (indices.begin() + 1)->index,
                    "Metric tensor indices must be different");
    impl::deny(indices.begin()->variance != (indices.begin() + 1)->variance,
                    "Metric tensor indices must have the same variance");

    Tensor metric{"g", indices};
    metric.set_tensor_class(impl::METRIC_TENSOR);

    const auto index_size = indices.begin()->index.size();
    for (int i = 0; i < index_size; ++i) {
        for (int j = 0; j < index_size; ++j) {
            metric[i, j] = metric_function(i, j);
        }
    }

    return metric;
}

Tensor levi_civita_symbol(const std::initializer_list<Index> indices) {
    std::vector<VarianceQualifiedIndex> vq_indices;
    for (const auto& index: indices) vq_indices.emplace_back(index, COVARIANT);
    return Tensor::make_levi_civita_symbol(vq_indices);
}

Tensor levi_civita_symbol(const std::initializer_list<VarianceQualifiedIndex> indices) {
    return Tensor::make_levi_civita_symbol(indices);
}

Tensor antisymmetric_symbol(const std::initializer_list<Index> indices) {
    std::vector<VarianceQualifiedIndex> vq_indices;
    for (const auto& index: indices) vq_indices.emplace_back(index, COVARIANT);
    return Tensor::make_levi_civita_symbol(vq_indices);
}

Tensor antisymmetric_symbol(const std::initializer_list<VarianceQualifiedIndex> indices) {
    return Tensor::make_levi_civita_symbol(indices);
}

Tensor kronecker_delta(const std::initializer_list<Index> indices) {
    std::vector<VarianceQualifiedIndex> vq_indices;
    for (const auto& index: indices) vq_indices.emplace_back(index, COVARIANT);
    return Tensor::make_kronecker_delta(vq_indices);
}

Tensor kronecker_delta(const std::initializer_list<VarianceQualifiedIndex> indices) {
    return Tensor::make_kronecker_delta(indices);
}

} // namespace varitensor
