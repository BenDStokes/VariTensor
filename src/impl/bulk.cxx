/*
* This Source Code Form is subject to the terms of the Mozilla Public
 * Licence, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "varitensor/impl/bulk.h"

#include <cstring>

namespace varitensor::impl {

#if defined (__AVX__)

// =================================================================================================
//                                                                                AVX256 Functions |
// =================================================================================================

#include <immintrin.h>

constexpr size_t REG_WIDTH_256 = 4;
constexpr std::align_val_t M256_ALIGN{alignof(__m256d)};

void deallocate(double* data) {
    operator delete[](data, M256_ALIGN);
}

DoublePtr allocate(const size_t size) {
    /**
     * To make SIMD operations more efficient, we allocate aligned memory and pad
     * all tensors to be multiples of the packing size.
     */
    const size_t remainder = size % REG_WIDTH_256;
    const size_t padded_size = remainder ? size - remainder + REG_WIDTH_256 : size;

    return DoublePtr{
        static_cast<double*> (
            operator new[](sizeof(double) * padded_size, M256_ALIGN)
        ),
        deallocate
    };
}

DoublePtr allocate_zeroed(const size_t size) {
    DoublePtr data = allocate(size);
    std::memset(data.get(), 0, size*sizeof(double));
    return data;
}

void copy(double* data1, const double* data2, const size_t size) {
    size_t i{0};
    for (; i<size - size%REG_WIDTH_256; i+=REG_WIDTH_256) {
        _mm256_store_pd(
            data1 + i,
            _mm256_loadu_pd(data2 + i)
        );
    }
    for (; i<size; ++i) data1[i] = data2[i];
}

void broadcast_vec(const double* data1, double* data2, const size_t size1, const size_t size2) {
    /* Whilst there is no data in the target, it's hard to play any tricks here, so just do it the
     * easy way.
     */

    size_t j_start{0};
    for (int i=0; i < static_cast<int>(size1 - REG_WIDTH_256 + 1); i += REG_WIDTH_256) {
        const __m256d values = _mm256_load_pd(data1 + i);
        for (size_t j=j_start; j<size2; j+=size1) {
            _mm256_storeu_pd(data2 + j, values);
        }
        j_start += REG_WIDTH_256;
    }

    for (size_t i=size1 - size1 % REG_WIDTH_256; i<size1; ++i) {
        const double value = data1[i];
        for (size_t j = j_start; j<size2; j+=size1) {
            data2[j] = value;
        }
        ++j_start;
    }
}

void broadcast_chunks(const double* data1, double* data2, const size_t size1, const size_t interval) {
    /* For this function, there will generally already be data in the target Tensor. Therefore, we
     * can't play any tricks with the memory padding as this would overwrite meaningful data.
     */
    double* running_ptr = data2;
    for (size_t i=0; i < size1; ++i) {
        const __m256d values = _mm256_set1_pd(data1[i]);
        for (int j=0; j < static_cast<int>(interval - REG_WIDTH_256 + 1); j += REG_WIDTH_256) {
            _mm256_storeu_pd(
                running_ptr,
                _mm256_mul_pd(
                    values,
                    _mm256_loadu_pd(running_ptr)
                )
            );

            running_ptr += REG_WIDTH_256;
        }
        for (size_t j=0; j < interval % REG_WIDTH_256; ++j) {
            *running_ptr *= data1[i];
            ++running_ptr;
        }
    }
}

void broadcast(const double value, double* data, const size_t size) {
    const __m256d scalar_vec = _mm256_set1_pd(value);
    for (size_t i=0; i<size; i+=REG_WIDTH_256) {
        _mm256_store_pd(
            data + i,
            _mm256_mul_pd(
                _mm256_load_pd(data + i),
                scalar_vec
            )
        );
    }
}

void piecewise(const double* data1, const double* data2,  double* data3, const size_t size, const Operation operation) {
    switch (operation) {
    case ADD:
        for (size_t i=0; i<size; i+=REG_WIDTH_256) {
            _mm256_store_pd(data3 + i,  _mm256_add_pd(_mm256_load_pd(data1 + i), _mm256_load_pd(data2 + i)));
        }
        break;
    case SUB:
        for (size_t i=0; i<size; i+=REG_WIDTH_256) {
            _mm256_store_pd(data3 + i,  _mm256_sub_pd(_mm256_load_pd(data1 + i), _mm256_load_pd(data2 + i)));
        }
        break;
    case MUL:
        for (size_t i=0; i<size; i+=REG_WIDTH_256) {
            _mm256_store_pd(data3 + i,  _mm256_mul_pd(_mm256_load_pd(data1 + i), _mm256_load_pd(data2 + i)));
        }
        break;
    case DIV:
        for (size_t i=0; i<size; i+=REG_WIDTH_256) {
            _mm256_store_pd(data3 + i,  _mm256_div_pd(_mm256_load_pd(data1 + i), _mm256_load_pd(data2 + i)));
        }
        break;
    }
}

# else

// =================================================================================================
//                                                                              Non-SIMD Functions |
// =================================================================================================

void deallocate(double* data) {
    std::free(data);
}

DoublePtr allocate(const size_t size) {
    auto data = DoublePtr{
        static_cast<double*>(malloc(size * sizeof(double))),
        deallocate
    };
    if (!data) throw std::bad_alloc{};
    return data;
}

DoublePtr allocate_zeroed(const size_t size) {
    return DoublePtr{
        static_cast<double*>(calloc(size, sizeof(double))),
        deallocate
    };
}

void copy(double* data1, const double* data2, const size_t size) {
    for (size_t i=0; i<size; ++i) data1[i] = data2[i];
}

void broadcast_vec(const double* data1, double* data2, const size_t size1, const size_t size2) {
    for (size_t i=0; i < size1; i++) {
        const double current_value = data1[i];
        for (size_t j=i; j < size2; j += size1) {
            data2[j] = current_value;
        }
    }
}

void broadcast_chunks(const double* data1, double* data2, const size_t size1, const size_t interval) {
    for (size_t i=0; i < size1; ++i) {
        const double current_value = data1[i];
        for (size_t j=0; j < interval; j++) {
            *data2 *= current_value;
            ++data2;
        }
    }
}

void broadcast(const double value, double* data, const size_t size) {
    for (size_t i=0; i<size; ++i) data[i] *= value;
}

void piecewise(const double* data1, const double* data2,  double* data3, const size_t size, const Operation operation) {
    switch (operation) {
        case ADD: for (size_t i=0; i<size; ++i) data3[i] = data1[i] + data2[i]; break;
        case SUB: for (size_t i=0; i<size; ++i) data3[i] = data1[i] - data2[i]; break;
        case MUL: for (size_t i=0; i<size; ++i) data3[i] = data1[i] * data2[i]; break;
        case DIV: for (size_t i=0; i<size; ++i) data3[i] = data1[i] / data2[i]; break;
    }
}

#endif

DoublePtr allocate(const size_t size, const double& initial_value) {
    DoublePtr data = allocate(size);
    std::fill_n(data.get(), size, initial_value);
    return data;
}

DoublePtr allocate_copy(const double* data, const size_t size) {
    DoublePtr copy = allocate(size);
    std::memcpy(copy.get(), data, size * sizeof(double));
    return copy;
}

} // namespace varitensor::impl
