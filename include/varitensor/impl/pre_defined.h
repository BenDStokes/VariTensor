/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * Licence, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef VARITENSOR_PRE_DEFINED_H
#define VARITENSOR_PRE_DEFINED_H

#include "varitensor/Tensor.h"

namespace varitensor {

const MetricFunction EUCLIDEAN_METRIC = [](const int i, const int j) {
    return i == j ? 1 : 0;
};

inline Tensor metric_tensor(
    const std::initializer_list<VarianceQualifiedIndex> indices,
    const MetricFunction& metric_function = EUCLIDEAN_METRIC
) {
    impl::soft_deny(indices.size() != 2,
                    "Metric tensor must have exactly 2 indices!");
    impl::soft_deny(indices.begin()->index == (indices.begin() + 1)->index,
                    "Metric tensor indices must be different!");
    impl::soft_deny(indices.begin()->variance != (indices.begin() + 1)->variance,
                    "Metric tensor indices must have the same variance!");

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

inline Tensor levi_civita_symbol(const std::initializer_list<Index> indices) {
    std::vector<VarianceQualifiedIndex> vq_indices;
    for (const auto& index: indices) vq_indices.emplace_back(index, COVARIANT);
    return Tensor::make_levi_civita_symbol(vq_indices);
}

inline Tensor levi_civita_symbol(const std::initializer_list<VarianceQualifiedIndex> indices) {
    return Tensor::make_levi_civita_symbol(indices);
}

inline Tensor antisymmetric_symbol(const std::initializer_list<Index> indices) {
    std::vector<VarianceQualifiedIndex> vq_indices;
    for (const auto& index: indices) vq_indices.emplace_back(index, COVARIANT);
    return Tensor::make_levi_civita_symbol(vq_indices);
}

inline Tensor antisymmetric_symbol(const std::initializer_list<VarianceQualifiedIndex> indices) {
    return Tensor::make_levi_civita_symbol(indices);
}

inline Tensor kronecker_delta(const std::initializer_list<Index> indices) {
    std::vector<VarianceQualifiedIndex> vq_indices;
    for (const auto& index: indices) vq_indices.emplace_back(index, COVARIANT);
    return Tensor::make_kronecker_delta(vq_indices);
}

inline Tensor kronecker_delta(const std::initializer_list<VarianceQualifiedIndex> indices) {
    return Tensor::make_kronecker_delta(indices);
}

} // namespace varitensor

#endif
