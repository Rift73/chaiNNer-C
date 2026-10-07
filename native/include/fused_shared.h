/* The fused tap of the separable filters (separable_shared.h) and the spatial
 * convolution (convolution_shared.h): a static helper compiled into each including
 * unit, never inline, so the linker cannot fold a VEX copy into baseline callers.
 * Every unit including either family header uses it (C4505). */
#ifndef CHAINNER_FUSED_SHARED_H
#define CHAINNER_FUSED_SHARED_H
#if defined(__AVX2__)
#include <immintrin.h>
#else
#include <math.h>
#endif

/* One fused tap, value * coefficient + sum rounded once. In the baseline unit the
 * CRT's fmaf (B3's call); in an ISA unit (/arch:AVX2 defines __AVX2__) the FMA3
 * instruction, since fmaf stays a CRT call there (SP4a compiler check). */
static float fused_tap(float value, float coefficient, float sum)
{
#if defined(__AVX2__)
    return _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(value), _mm_set_ss(coefficient), _mm_set_ss(sum)));
#else
    return fmaf(value, coefficient, sum);
#endif
}
#endif
