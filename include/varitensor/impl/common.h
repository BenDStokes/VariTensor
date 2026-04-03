/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * Licence, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef VARITENSOR_COMMON_H
#define VARITENSOR_COMMON_H

#include <functional>
#include <memory>
#include <variant>

#include "Index.h"

namespace varitensor {

class Tensor;
class View;

enum Variance {COVARIANT = 0, LOWER = 0, CONTRAVARIANT = 1, UPPER = 1};
struct VarianceQualifiedIndex {
    Index index;
    Variance variance = COVARIANT;
    bool operator==(const VarianceQualifiedIndex& other) const {
        return index == other.index && variance == other.variance;
    }
};

using MetricFunction = std::function<double(int, int)>;

namespace impl {

// NB: we occasionally use the fact that ADD == -SUB
enum Operation {SUB=-1, ADD=1, MUL=2, DIV=3};

enum ExprState {
    SCALAR,
    ALIGNED_INDICES,
    FREE_MULTIPLICATION,
    GENERAL
};

struct Dimension {
    Index index;
    Variance variance{};
    size_t width{}; // the memory width of a single increment along this dimension
    [[nodiscard]] int size() const {return index.size();}
};
using Dimensions = std::vector<Dimension>;

using DoublePtr = std::unique_ptr<double, void(*)(double*)>;

template <bool is_const>
class ViewIterator;

class LinkedOp;
class LinkedOpIterator;

class ProductOp;
class ProductOpIterator;

template<typename T>
concept Expression_c = std::same_as<T, View> || std::same_as<T, LinkedOp> || std::same_as<T, ProductOp>;
using Expression = std::variant<View, LinkedOp, ProductOp>;
using Expressions = std::vector<Expression>;

template<typename T>
concept ExpressionIterator_c =
    std::same_as<T, ViewIterator<true>>  ||
    std::same_as<T, ViewIterator<false>> ||
    std::same_as<T, LinkedOpIterator>    ||
    std::same_as<T, ProductOpIterator>;
using ExpressionIterator = std::variant<ViewIterator<true>, ViewIterator<false>, LinkedOpIterator, ProductOpIterator>;
using ExpressionIterators = std::vector<ExpressionIterator>;

template <typename T>
concept ExpressionOperand_c =
    std::is_same_v<T, Tensor>   ||
    std::is_same_v<T, View>     ||
    std::is_same_v<T, LinkedOp> ||
    std::is_same_v<T, ProductOp> ||
    std::is_same_v<T, double>;

} // namespace impl
} // namespace varitensor

#endif // VARITENSOR_COMMON_H
