/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * Licence, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef VARITENSOR_INCREMENT_POSITIONS_H
#define VARITENSOR_INCREMENT_POSITIONS_H

#include <vector>

namespace varitensor::impl {

template <ExpressionIterator_c E>
bool increment_positions(
    std::vector<int>& positions,
    const std::vector<Dimension>& dimensions,
    const E& iterator // NB: not really const but we have to pretend to maintain the constness of the * operators
) {
    /**
     * Increments the positions vector in accordance with the dimensions vector. The iterator
     * provided will be told which index to increment, or reset if a dimension is overflown. Returns
     * true if the increment was successful, false if the end of the dimensions has been reached.
     */
    for (size_t i=0; i<positions.size(); ++i) {
        // check if we're about to overflow the next index
        if (positions[i] + 1 == dimensions[i].index.size()) {
            // if not, reset it and move on to the next one
            positions[i] = 0;
            iterator.reset(dimensions[i].index.id());
            continue;
        }

        // once we've found an index we can increment, do so
        ++positions[i];
        iterator.increment(dimensions[i].index.id());

        return true;
    }

    // note that if we reach the end, all the indices will have been reset to 0
    return false;
}

} // namespace varitensor::impl

#endif
