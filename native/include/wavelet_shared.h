/* Wavelet Color Fix's blur job (wavelet_ops.c) and its AVX2 unit (wavelet_avx2.c), reached only through blur_range's
 * dispatch on the call's ISA level. */
#ifndef CHAINNER_WAVELET_SHARED_H
#define CHAINNER_WAVELET_SHARED_H
#include "isa.h"

#include <stddef.h>

/* One blur pass over planes of height x width: low = the nine-tap blur of source at radius, and with high,
 * high = (first ? 0 : high) + (source - low). fused: the taps accumulate as fmaf (oneDNN), else as separate products.
 * isa: the level the C entry read once for the call (cn_isa_current). */
typedef struct wavelet_job {
    const float *source;
    float *low, *high;
    size_t height, width, radius;
    int first, fused;
    cn_isa_level isa;
} wavelet_job;

/* Outputs [from, to) of one row, all inside the row's interior (x - radius >= 0 and x + radius < width), eight at a
 * time: each lane runs blur_range's sequence (value = 0, then the nine taps in row-major order as vfmadd when fused,
 * else vmulps then vaddps; then the high update), the last 1-7 the same sequence in scalar code. Where two different
 * NaNs meet in one fused tap, vfmadd's operand order may keep the other one (ARCHITECTURE's NaN-payload row); every
 * other bit is the scalar loop's. */
void cn_wavelet_interior_avx2(const wavelet_job *job, size_t from, size_t to);
#endif
