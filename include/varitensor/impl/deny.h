/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * Licence, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef VARITENSOR_DENY_H
#define VARITENSOR_DENY_H

#ifndef VARITENSOR_VALIDATION_ON
#define VARITENSOR_VALIDATION_ON 1
#include <stdexcept>
#endif

namespace varitensor::impl {

inline void strict_deny(const bool condition, const std::string& message) {
    if (condition) {
        throw std::logic_error(message);
    }
}

inline void deny(const bool condition, const std::string& message) {
    if constexpr (VARITENSOR_VALIDATION_ON) {
        strict_deny(condition, message);
    }
}

} // namespace varitensor::impl

#endif
