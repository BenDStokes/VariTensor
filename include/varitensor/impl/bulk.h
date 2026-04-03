/*
* This Source Code Form is subject to the terms of the Mozilla Public
 * Licence, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef VARITENSOR_BULK_H
#define VARITENSOR_BULK_H

#include "common.h"

namespace varitensor::impl {

void deallocate(double* data);

DoublePtr allocate(size_t size);
DoublePtr allocate(size_t size, const double& initial_value);
DoublePtr allocate_zeroed(size_t size);
DoublePtr allocate_copy(const double* data, size_t size);

void copy(double* data1, const double* data2, size_t size);

/* Broadcast data1 along the length of data2, e.g. 123 with size 9 data2 gives 123 123 123 */
void broadcast_vec(const double* data1, double* data2, size_t size1, size_t size2);

/* Broadcast each element of data1 as "chunks" into data2, e.g. 123 with size 9 data2 gives 111 222 333 */
void broadcast_chunks(const double* data1, double* data2, size_t size1, size_t interval);

/* Multiply every element of data by value */
void broadcast(double value, double* data, size_t size);

/* data3[i] = operation(data1[i], data2[i]) */
void piecewise(const double* data1, const double* data2,  double* data3, size_t size, Operation operation);

} // namespace varitensor::impl

#endif // VARITENSOR_BULK_H
