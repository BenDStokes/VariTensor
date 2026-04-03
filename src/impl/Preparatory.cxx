/*
* This Source Code Form is subject to the terms of the Mozilla Public
 * Licence, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include <map>
#include <ranges>

#include "varitensor/impl/Preparatory.h"
#include "varitensor/impl/deny.h"
#include "varitensor/impl/visitors.h"

namespace varitensor::impl {

namespace {

struct Couple {
    /**
     * Helper struct used to track repeated indices and metric counterparts.
     */
    int dims[2]{-1, -1};
    int metric_index{-1};
    int other_index{-1};

    void add(const int dimension, const bool is_metric) {
        deny(dims[0] != -1 && dims[1] != -1,
               "Indices in multiplication expression cannot appear more than twice");
        const auto target = dims[0] == -1 ? 0 : 1;
        dims[target] = dimension;
        if (!is_metric) other_index = dimension;
    }

    void clear() {
        dims[0] = -1;
        dims[1] = -1;
        metric_index = -1;
    }

    [[nodiscard]] bool has() const {
        return dims[0] != -1;
    }

    [[nodiscard]] bool is_repeated() const {
        return dims[1] != -1;
    }

    [[nodiscard]] bool is_metric() const {
        return metric_index != -1;
    }
};

} // anonymous namespace

Preparatory::Preparatory(const Expressions& sub_expressions, const PreparatoryType type) {
    if (sub_expressions.empty()) return; // scalar case

    if (type == LINKED) prepare_linked_operation(sub_expressions);
    else prepare_product_operation(sub_expressions);
}

void Preparatory::prepare_linked_operation(const Expressions& sub_expressions) {
    state = ALIGNED_INDICES;

    // construct the first iterator so that we have something to compare against
    sub_iterators.emplace_back(std::visit(VBegin, sub_expressions[0]));
    dimensions = std::visit(GetDimensions, sub_iterators[0]);
    size = std::visit(GetSize, sub_iterators[0]);

    std::function<void(unsigned&, const Dimensions&)> index_matching_function;

    auto find_index = [this] (
        const unsigned& j, const Dimensions& current_dimensions, unsigned k=0
    ) {
        for (; k<current_dimensions.size(); ++k) {
            if (dimensions[j].index == current_dimensions[k].index) {
                deny(dimensions[j].variance != current_dimensions[k].variance,
                    "Cannot add or subtract tensors with indices of disagreeing variance");
                return;
            }
        }
        deny(true, // this is always an error, but go through deny() for consistency
            "Cannot add or subtract tensors with un-pairable indices");
    };

    auto assume_aligned = [this, &find_index, &index_matching_function](
        const unsigned& j, const Dimensions& current_dimensions
    ) {
        deny(dimensions[j].variance != current_dimensions[j].variance,
                "Cannot add or subtract tensors with indices of disagreeing variance");
        if (dimensions[j].index != current_dimensions[j].index) {
            state = GENERAL;
            find_index(j, current_dimensions, j);
            index_matching_function = find_index; // switch to the general case
        }
    };

    // assume that we have only views with aligned indices, e.g. (T_ab + U_ab) NOT (T_ab + U_ba)
    index_matching_function = assume_aligned;
    for (unsigned i=1; i<sub_expressions.size(); ++i) { // start i at 1 as we just did the first iterator
        sub_iterators.emplace_back(std::visit(VBegin, sub_expressions[i]));
        deny(std::visit(GetDimensions, sub_iterators[i]).size() != dimensions.size(),
            "Cannot add or subtract tensors with different numbers of indices");

        const Dimensions& current_dimensions = std::visit(GetDimensions, sub_iterators[i]);
        for (unsigned j=0; j<dimensions.size(); ++j) {
            index_matching_function(j, current_dimensions);
        }
    }
}

void Preparatory::prepare_product_operation(const Expressions& sub_expressions) {
    state = FREE_MULTIPLICATION;

    for (auto& expression: sub_expressions) sub_iterators.emplace_back(std::visit(VBegin, expression));

    // get the naive dimensions whilst recording metric and repetition information that will be useful later
    Dimensions total_dimensions;
    std::map<int, Couple> partners;
    int i{0};

    for (auto& iterator: sub_iterators) {
        auto sub_dimensions = std::visit(GetDimensions, iterator);

        for (size_t k = 0; k < sub_dimensions.size(); ++k) {
            total_dimensions.emplace_back(sub_dimensions[k]);

            if (std::visit([](auto& expression) {return expression.is_metric();}, iterator)) {
                partners[sub_dimensions[k].index.id()].add(i, true);
                const int other = k == 0 ? i+1 : i-1; // metric tensors only have 2 indices
                partners[sub_dimensions[k].index.id()].metric_index = other;
            }
            else {
                partners[sub_dimensions[k].index.id()].add(i, false);
            }

            ++i;
        }
    }

    // handle any metric indices
    for (auto& couple: partners | std::views::values) {
        if (couple.is_repeated() && couple.is_metric()) {
            std::swap(total_dimensions[couple.metric_index], total_dimensions[couple.other_index]);
        }
    }

    for (auto& dimension: total_dimensions) {
        if (auto& couple = partners[dimension.index.id()]; couple.is_repeated()) { // repeated index
            repeated.emplace_back(dimension);
            couple.clear();
            state = GENERAL;
        }
        else if (couple.has()) { // non-repeated index
            dimensions.push_back(dimension);
            dimensions.back().width = size;
            size *= dimension.size();
        }
    }
}

} // namespace varitensor::impl