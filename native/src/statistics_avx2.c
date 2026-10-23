#if !defined(_M_X64)
#error "statistics_avx2.c is an x86-64 /arch:AVX2 unit"
#endif
/* np.min/np.max's 8-lane reduce (statistics_ops.c reduce_contiguous), AVX2 unit (/arch:AVX2, spec 4.3): no export,
 * reached only through reduce_contiguous's dispatch on the ISA level. Each lane runs the scalar extreme's sequence:
 * vmaxps/vminps (the second operand on ties and on a NaN second operand), then a NaN first operand put back by a
 * blend, so every lane holds the scalar accumulator's bits. */
#include "numeric.h"

#include <immintrin.h>

static __m256 extreme8(__m256 a, __m256 b, int maximum)
{
    const __m256 picked = maximum ? _mm256_max_ps(a, b) : _mm256_min_ps(a, b);
    return _mm256_blendv_ps(picked, a, _mm256_cmp_ps(a, a, _CMP_UNORD_Q));
}

size_t cn_numpy_extreme8_avx2(float *accumulator, const float *x, size_t len, int maximum)
{
    __m256 acc = _mm256_loadu_ps(accumulator);
    size_t done = 0;
    for (; len - done >= 64; done += 64) {
        const float *p = x + done;
        const __m256 r01 = extreme8(_mm256_loadu_ps(p), _mm256_loadu_ps(p + 8), maximum);
        const __m256 r23 = extreme8(_mm256_loadu_ps(p + 16), _mm256_loadu_ps(p + 24), maximum);
        const __m256 r45 = extreme8(_mm256_loadu_ps(p + 32), _mm256_loadu_ps(p + 40), maximum);
        const __m256 r67 = extreme8(_mm256_loadu_ps(p + 48), _mm256_loadu_ps(p + 56), maximum);
        acc = extreme8(acc, extreme8(extreme8(r01, r23, maximum), extreme8(r45, r67, maximum), maximum), maximum);
    }
    for (; len - done >= 8; done += 8) acc = extreme8(acc, _mm256_loadu_ps(x + done), maximum);
    _mm256_storeu_ps(accumulator, acc);
    return done;
}
