/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * Licence, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef VARITENSOR_DENY_H
#define VARITENSOR_DENY_H

#ifndef VARITENSOR_VALIDATION_ON
#define VARITENSOR_VALIDATION_ON 1
#endif

#include <stdexcept>

namespace varitensor {

struct TensorLogicError final: std::logic_error {
    explicit TensorLogicError(const std::string& message):
        std::logic_error("<Vari-Tensor> Tensor logic error: " + message)
    {}
};

namespace impl {

struct VariTensorInternalError final: std::logic_error {
    explicit VariTensorInternalError(const std::string& message):
        std::logic_error("<Vari-Tensor> Tensor logic error: " + message)
    {}
};

inline void deny(const bool condition, const std::string& message) {
    if constexpr (VARITENSOR_VALIDATION_ON) {
        if (condition) [[unlikely]] {
            throw TensorLogicError(message);
        }
    }
}

} // namespace impl
} // namespace varitensor

#endif // VARITENSOR_DENY_H
