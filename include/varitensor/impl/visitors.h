/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * Licence, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef VARITENSOR_VISITORS_H
#define VARITENSOR_VISITORS_H

#include "common.h"

namespace varitensor::impl {

inline auto Deref = [](const auto& expression) {return expression.deref();};
inline auto VBegin = [](auto& expression) {return expression.vbegin();};
inline auto GetDimensions = [](auto& expression) {return expression.dimensions();};
inline auto GetSize = [](auto& expression) {return expression.size();};

struct Reset {
    explicit Reset(const int index_id_): index_id{index_id_} {}

    template <ExpressionIterator_c E>
    void operator()(const E& expression) const {expression.reset(index_id);}

    int index_id;
};

struct Increment {
    explicit Increment(const int index_id_): index_id{index_id_} {}

    template <ExpressionIterator_c E>
    void operator()(const E& expression) const {expression.increment(index_id);}

    int index_id;
};

} // namespace varitensor::impl

#endif
