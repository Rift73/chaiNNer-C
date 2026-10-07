#if !defined(_M_X64)
#error "lens_avx2.c is an x86-64 /arch:AVX2 unit"
#endif
/* Lens Blur's component composition, AVX2 unit (/arch:AVX2, spec 4.3): no export,
 * reached only through compose_range's dispatch on the call's ISA level. Each lane
 * runs compose_one's operations in its order, the zero products and sums
 * included: they are significant for infinities and signed zero. */
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
