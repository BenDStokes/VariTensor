/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * Licence, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "varitensor/impl/ExpressionIteratorBase.h"
#include "varitensor/Tensor.h"
#include "varitensor/impl/Preparatory.h"

namespace varitensor::impl {

ExpressionIteratorBase::ExpressionIteratorBase(Preparatory& preparatory):
        m_size{preparatory.size},
        m_dimensions{std::move(preparatory.dimensions)},
        m_positions(m_dimensions.size(), 0)
{}

std::vector<int>& ExpressionIteratorBase::positions() {
    return m_positions;
}

int ExpressionIteratorBase::positions(const int index) const {
    return m_positions[index];
}

int ExpressionIteratorBase::positions(const Index& index) const {
    for (size_t i = 0; i < m_dimensions.size(); ++i) {
        if (m_dimensions[i].index == index) return m_positions[i];
    }
    throw TensorLogicError(
        "Unable to find index!"
    ); // can't use deny() as there is nothing to return
}

const Dimensions& ExpressionIteratorBase::dimensions() const {
    return m_dimensions;
}

size_t ExpressionIteratorBase::size() const {
    return m_size;
}

bool ExpressionIteratorBase::is_scalar() const {
    return m_size == 1;
}

} // namespace varitensor::impl
