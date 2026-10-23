/* Unsharp Mask's AVX2 unit (unsharp_avx2.c), reached only through unsharp_range's dispatch on the call's ISA level
 * (unsharp_complete_ops.c). */
#ifndef CHAINNER_UNSHARP_SHARED_H
#define CHAINNER_UNSHARP_SHARED_H
#include <stddef.h>

/* output[i] = fmaf(source[i], alpha, fmaf(blurred[i], beta, 0)) for i in [0, count), eight at a time as two
 * vfmadd (single roundings, fmaf's bits), the last 1-7 through the CRT's fmaf. */
void cn_unsharp_fused_avx2(const float *source, const float *blurred, float *output, size_t count, float alpha,
                           float beta);
#endif
