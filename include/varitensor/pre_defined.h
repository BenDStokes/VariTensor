/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * Licence, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef VARITENSOR_PRE_DEFINED_H
#define VARITENSOR_PRE_DEFINED_H

#include "impl/common.h"

namespace varitensor {

const MetricFunction EUCLIDEAN_METRIC = [](const int i, const int j) {
    return i == j ? 1 : 0;
};

Tensor metric_tensor(
    std::initializer_list<VarianceQualifiedIndex> indices,
    const MetricFunction& metric_function = EUCLIDEAN_METRIC
);

Tensor levi_civita_symbol(std::initializer_list<Index> indices);
Tensor levi_civita_symbol(std::initializer_list<VarianceQualifiedIndex> indices);

Tensor antisymmetric_symbol(std::initializer_list<Index> indices);
Tensor antisymmetric_symbol(std::initializer_list<VarianceQualifiedIndex> indices);

Tensor kronecker_delta(std::initializer_list<Index> indices);
Tensor kronecker_delta(std::initializer_list<VarianceQualifiedIndex> indices);

} // namespace varitensor

#endif // VARITENSOR_PRE_DEFINED_H
