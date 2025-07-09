/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * Licence, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef VARITENSOR_EXPRESSIONITERATOR_H
#define VARITENSOR_EXPRESSIONITERATOR_H

#include "common.h"

namespace varitensor::impl {

class ExpressionIteratorBase {
/** Class to reduce code duplication in expression iterators
 *
 * Virtual functions have been omitted for performance
 */
public:
    using iterator_category = std::forward_iterator_tag;
    using difference_type   = std::ptrdiff_t;
    using value_type        = double;

    [[nodiscard]] const std::vector<int>& positions() {
        return m_positions;
    }

    [[nodiscard]] int positions(const int index) const {
        return m_positions[index];
    }

    [[nodiscard]] int positions(const Index& index) const {
        for (size_t i = 0; i < m_dimensions.size(); ++i) {
            if (m_dimensions[i].index == index) return m_positions[i];
        }
        throw std::runtime_error("Unable to find index!");
    }

    [[nodiscard]] const Dimensions& dimensions() const {
        return m_dimensions;
    }

    [[nodiscard]] size_t size() const {
        return m_size;
    }

    [[nodiscard]] bool is_scalar() const {
        return m_size == 1;
    }


protected:
    size_t m_size{1};
    Dimensions m_dimensions;
    std::vector<int> m_positions;
};

} // namespace varitensor::impl

#endif
