#if !defined(_M_X64) && !(defined(__linux__) && defined(__x86_64__))
#error "lens_avx512.c is an x86-64 /arch:AVX512 unit"
#endif
/* Lens Blur's tap fold, AVX-512 unit (/arch:AVX512): no export, reached only through the fold's dispatch on
 * the call's ISA level. Every lane runs its output's sequence, the taps in order, each one vfmadd. */
#include "lens_fold_shared.h"
#include <immintrin.h>

void cn_lens_fold_avx512(const float *src, const lens_taps *taps, size_t fused, size_t width, float *out)
{
    size_t x = 0, k, count = taps->count;
    /* Eight groups at a time keep eight independent sums (a 4-cycle FMA at 2 per cycle needs 8 in flight). */
    for (; fused - x >= 128; x += 128) {
        __m512 sum0 = _mm512_setzero_ps(), sum1 = sum0, sum2 = sum0, sum3 = sum0;
        __m512 sum4 = sum0, sum5 = sum0, sum6 = sum0, sum7 = sum0;
        for (k = 0; k < count; ++k) {
            const float *values = src + (ptrdiff_t)x + taps->offsets[k];
            __m512 coefficient = _mm512_set1_ps(taps->coefficients[k]);
            sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(values), coefficient, sum0);
            sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(values + 16), coefficient, sum1);
            sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(values + 32), coefficient, sum2);
            sum3 = _mm512_fmadd_ps(_mm512_loadu_ps(values + 48), coefficient, sum3);
            sum4 = _mm512_fmadd_ps(_mm512_loadu_ps(values + 64), coefficient, sum4);
            sum5 = _mm512_fmadd_ps(_mm512_loadu_ps(values + 80), coefficient, sum5);
            sum6 = _mm512_fmadd_ps(_mm512_loadu_ps(values + 96), coefficient, sum6);
            sum7 = _mm512_fmadd_ps(_mm512_loadu_ps(values + 112), coefficient, sum7);
        }
        _mm512_storeu_ps(out + x, sum0);
        _mm512_storeu_ps(out + x + 16, sum1);
        _mm512_storeu_ps(out + x + 32, sum2);
        _mm512_storeu_ps(out + x + 48, sum3);
        _mm512_storeu_ps(out + x + 64, sum4);
        _mm512_storeu_ps(out + x + 80, sum5);
        _mm512_storeu_ps(out + x + 96, sum6);
        _mm512_storeu_ps(out + x + 112, sum7);
    }
    for (; fused - x >= 16; x += 16) {
        __m512 sum = _mm512_setzero_ps();
        for (k = 0; k < count; ++k)
            sum = _mm512_fmadd_ps(_mm512_loadu_ps(src + (ptrdiff_t)x + taps->offsets[k]),
                _mm512_set1_ps(taps->coefficients[k]), sum);
        _mm512_storeu_ps(out + x, sum);
    }
    if (fused > x) {
        __mmask16 mask = (__mmask16)((1u << (fused - x)) - 1u);
        __m512 sum = _mm512_setzero_ps();
        for (k = 0; k < count; ++k)
            sum = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(mask, src + (ptrdiff_t)x + taps->offsets[k]),
                _mm512_set1_ps(taps->coefficients[k]), sum);
        _mm512_mask_storeu_ps(out + x, mask, sum);
    }
    lens_fold_scalar(src, taps, fused, fused, width, out);
}
