#if !defined(_M_X64) && !(defined(__linux__) && defined(__x86_64__))
#error "unsharp_avx2.c is an x86-64 /arch:AVX2 unit"
#endif
/* Unsharp Mask's fused weighted sum, AVX2 unit (/arch:AVX2, spec 4.3): no export, reached only through
 * unsharp_range's dispatch on the call's ISA level (unsharp_shared.h). */
#include "unsharp_shared.h"

#include <immintrin.h>
#include <math.h>

void cn_unsharp_fused_avx2(const float *source, const float *blurred, float *output, size_t count, float alpha,
                           float beta)
{
    const __m256 a = _mm256_set1_ps(alpha), b = _mm256_set1_ps(beta), zero = _mm256_setzero_ps();
    size_t i = 0;
    for (; count - i >= 8; i += 8) {
        const __m256 low = _mm256_fmadd_ps(_mm256_loadu_ps(blurred + i), b, zero);
        _mm256_storeu_ps(output + i, _mm256_fmadd_ps(_mm256_loadu_ps(source + i), a, low));
    }
    for (; i < count; ++i) output[i] = fmaf(source[i], alpha, fmaf(blurred[i], beta, 0.0f));
}
