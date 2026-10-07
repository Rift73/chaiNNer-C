#ifndef CHAINNER_SEPARABLE_H
#define CHAINNER_SEPARABLE_H
#include "chainner.h"
/* Symmetric, odd-length float32 kernels, with replicate/reflect/reflect101
 * border modes (OpenCV codes 1, 2, 4). Kernels are owned by the caller. */
cn_status cn_symmetric_filter_f32(const float *src, float *dst, size_t height,
    size_t width, size_t channels, const float *kx, size_t nx,
    const float *ky, size_t ny, int border, size_t lanes);
#endif
