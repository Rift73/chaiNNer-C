/* Gamma's AVX2 unit (alpha_gamma_avx2.c), reached only through gamma_range's dispatch on the call's ISA level
 * (alpha_gamma_ops.c). */
#ifndef CHAINNER_GAMMA_SHARED_H
#define CHAINNER_GAMMA_SHARED_H
#include <stddef.h>

/* out[i] = gamma_approx(src[i], gamma) for i in [0, count), count a multiple of 8: each lane runs gamma_approx's
 * operations in its order (chainner-rs's AVX lanes, which the scalar function mirrors), so every output has the
 * scalar function's bits. */
void cn_gamma_approx_avx2(const float *src, float *out, size_t count, float gamma);
#endif
