#if !defined(_M_X64) && !(defined(__linux__) && defined(__x86_64__))
#error "lens_avx2.c is an x86-64 /arch:AVX2 unit"
#endif
/* Lens Blur's component composition and tap fold, AVX2 unit (/arch:AVX2, spec 4.3): no
 * export, reached only through compose_range's and the fold's dispatch on the call's ISA
 * level. In the composition each lane runs compose_one's operations in its order, the
 * zero products and sums included: they are significant for infinities and signed zero.
 * In the fold each lane runs its output's sequence, the taps in order, each one vfmadd. */
#include "lens_fold_shared.h"
#include "lens_shared.h"
#include <immintrin.h>

void cn_lens_compose_avx2(const compose_context *ctx, size_t begin, size_t end)
{
    const __m256 zero = _mm256_setzero_ps(), a = _mm256_set1_ps(ctx->a), b = _mm256_set1_ps(ctx->b);
    size_t i = begin;
    for (; end - i >= 8; i += 8) {
        __m256 real = _mm256_sub_ps(_mm256_loadu_ps(ctx->f1 + i), _mm256_loadu_ps(ctx->f4 + i));
        __m256 imag = _mm256_add_ps(_mm256_loadu_ps(ctx->f2 + i), _mm256_loadu_ps(ctx->f3 + i));
        __m256 complex_real = _mm256_sub_ps(_mm256_mul_ps(zero, imag), zero);
        __m256 complex_imag = _mm256_add_ps(zero, imag);
        __m256 value;
        real = _mm256_add_ps(real, complex_real);
        imag = _mm256_add_ps(zero, complex_imag);
        value = _mm256_add_ps(_mm256_mul_ps(real, a), _mm256_mul_ps(imag, b));
        if (ctx->accumulate) value = _mm256_add_ps(_mm256_loadu_ps(ctx->out + i), value);
        _mm256_storeu_ps(ctx->out + i, value);
    }
    for (; i < end; ++i) ctx->out[i] = compose_one(ctx, i);
}

void cn_lens_fold_avx2(const float *src, const lens_taps *taps, size_t fused, size_t width, float *out)
{
    size_t x = 0, k, count = taps->count;
    /* Eight groups at a time keep eight independent sums (a 4-cycle FMA at 2 per cycle needs 8 in flight). */
    for (; fused - x >= 64; x += 64) {
        __m256 sum0 = _mm256_setzero_ps(), sum1 = sum0, sum2 = sum0, sum3 = sum0;
        __m256 sum4 = sum0, sum5 = sum0, sum6 = sum0, sum7 = sum0;
        for (k = 0; k < count; ++k) {
            const float *values = src + (ptrdiff_t)x + taps->offsets[k];
            __m256 coefficient = _mm256_set1_ps(taps->coefficients[k]);
            sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(values), coefficient, sum0);
            sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(values + 8), coefficient, sum1);
            sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(values + 16), coefficient, sum2);
            sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(values + 24), coefficient, sum3);
            sum4 = _mm256_fmadd_ps(_mm256_loadu_ps(values + 32), coefficient, sum4);
            sum5 = _mm256_fmadd_ps(_mm256_loadu_ps(values + 40), coefficient, sum5);
            sum6 = _mm256_fmadd_ps(_mm256_loadu_ps(values + 48), coefficient, sum6);
            sum7 = _mm256_fmadd_ps(_mm256_loadu_ps(values + 56), coefficient, sum7);
        }
        _mm256_storeu_ps(out + x, sum0);
        _mm256_storeu_ps(out + x + 8, sum1);
        _mm256_storeu_ps(out + x + 16, sum2);
        _mm256_storeu_ps(out + x + 24, sum3);
        _mm256_storeu_ps(out + x + 32, sum4);
        _mm256_storeu_ps(out + x + 40, sum5);
        _mm256_storeu_ps(out + x + 48, sum6);
        _mm256_storeu_ps(out + x + 56, sum7);
    }
    for (; fused - x >= 8; x += 8) {
        __m256 sum = _mm256_setzero_ps();
        for (k = 0; k < count; ++k)
            sum = _mm256_fmadd_ps(_mm256_loadu_ps(src + (ptrdiff_t)x + taps->offsets[k]),
                _mm256_set1_ps(taps->coefficients[k]), sum);
        _mm256_storeu_ps(out + x, sum);
    }
    lens_fold_scalar(src, taps, x, fused, width, out);
}
