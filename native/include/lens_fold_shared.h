/* Lens Blur's tap folds: one list of taps over one row, the same sequence in the baseline unit lens_complete_ops.c
 * and the ISA units lens_avx2.c and lens_avx512.c. A static helper compiled into each including unit, never inline,
 * so the linker cannot fold a VEX copy into baseline callers. Every including unit uses it (C4505). */
#ifndef CHAINNER_LENS_FOLD_SHARED_H
#define CHAINNER_LENS_FOLD_SHARED_H
#include "fused_shared.h"
#include <stddef.h>

/* A convolution kernel's nonzero coefficients, in kernel order, each with its source's distance from the output. */
#define CN_LENS_MAX_TAPS 129
typedef struct lens_taps {
    size_t count;
    ptrdiff_t offsets[CN_LENS_MAX_TAPS];
    float coefficients[CN_LENS_MAX_TAPS];
} lens_taps;

/* Outputs [first, width) of a row: the taps in order over src[x + offset], each one fused below fused, separately
 * rounded from fused on (what cn_convolution_spatial's output does at the same column). */
static void lens_fold_scalar(const float *src, const lens_taps *taps, size_t first, size_t fused, size_t width,
    float *out)
{
    for (size_t x = first; x < width; ++x) {
        float sum = 0;
        if (x < fused)
            for (size_t k = 0; k < taps->count; ++k)
                sum = fused_tap(src[(ptrdiff_t)x + taps->offsets[k]], taps->coefficients[k], sum);
        else
            for (size_t k = 0; k < taps->count; ++k)
                sum += src[(ptrdiff_t)x + taps->offsets[k]] * taps->coefficients[k];
        out[x] = sum;
    }
}

/* lens_avx2.c and lens_avx512.c: the same outputs from column 0, for ISA level avx2 or avx512 and fused > 0, the ones
 * below fused in groups of 8 or 16. */
void cn_lens_fold_avx2(const float *src, const lens_taps *taps, size_t fused, size_t width, float *out);
void cn_lens_fold_avx512(const float *src, const lens_taps *taps, size_t fused, size_t width, float *out);
#endif
