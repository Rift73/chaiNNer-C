#if !defined(_M_X64)
#error "alpha_gamma_avx2.c is an x86-64 /arch:AVX2 unit"
#endif
/* Gamma's approximation, AVX2 unit (/arch:AVX2, spec 4.3): no export, reached only through gamma_range's dispatch on
 * the call's ISA level (alpha_gamma_ops.c). Eight elements at a time run gamma_approx's operations in its order, one
 * instruction for each scalar operation: vmulps and vaddps or vsubps where it multiplies and adds (no fused
 * multiply-add), vmaxps and vminps for its ?: clamps (each selects its second operand on a NaN, as those forms do),
 * vroundps for floorf, compare masks for its conditional terms, and integer shifts and masks for its bit casts. Its
 * notices (chainner-rs, avx2_mathfun, sse_mathfun) are in alpha_gamma_ops.c. */
#include "gamma_shared.h"

#include <immintrin.h>

static __m256 gamma_log(__m256 x)
{
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 invalid = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LE_OQ);
    x = _mm256_max_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(0x00800000)));
    const __m256i bits = _mm256_castps_si256(x);
    const __m256i exponent = _mm256_sub_epi32(_mm256_srli_epi32(bits, 23), _mm256_set1_epi32(127));
    x = _mm256_castsi256_ps(_mm256_or_si256(_mm256_andnot_si256(_mm256_set1_epi32(0x7f800000), bits),
        _mm256_set1_epi32(0x3f000000)));
    __m256 e = _mm256_add_ps(_mm256_cvtepi32_ps(exponent), one);
    const __m256 below = _mm256_cmp_ps(x, _mm256_set1_ps(0.707106781186547524f), _CMP_LT_OQ);
    const __m256 temporary = _mm256_and_ps(below, x);
    e = _mm256_sub_ps(e, _mm256_and_ps(below, one));
    x = _mm256_sub_ps(x, one);
    x = _mm256_add_ps(x, temporary);
    const __m256 z = _mm256_mul_ps(x, x);
    __m256 y = _mm256_mul_ps(_mm256_set1_ps(7.0376836292E-2f), x);
    y = _mm256_add_ps(y, _mm256_set1_ps(-1.1514610310E-1f));
    y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(1.1676998740E-1f));
    y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(-1.2420140846E-1f));
    y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(1.4249322787E-1f));
    y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(-1.6668057665E-1f));
    y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(2.0000714765E-1f));
    y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(-2.4999993993E-1f));
    y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(3.3333331174E-1f));
    y = _mm256_mul_ps(y, x);
    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, _mm256_mul_ps(e, _mm256_set1_ps(-2.12194440e-4f)));
    y = _mm256_sub_ps(y, _mm256_mul_ps(z, _mm256_set1_ps(0.5f)));
    x = _mm256_add_ps(x, y);
    x = _mm256_add_ps(x, _mm256_mul_ps(e, _mm256_set1_ps(0.693359375f)));
    return _mm256_blendv_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(-1)), invalid);
}

static __m256 gamma_exp(__m256 x)
{
    const __m256 one = _mm256_set1_ps(1.0f);
    x = _mm256_min_ps(x, _mm256_set1_ps(88.3762626647949f));
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.3762626647949f));
    __m256 fx = _mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f));
    fx = _mm256_add_ps(fx, _mm256_set1_ps(0.5f));
    const __m256 temporary = _mm256_floor_ps(fx);
    fx = _mm256_sub_ps(temporary, _mm256_and_ps(_mm256_cmp_ps(temporary, fx, _CMP_GT_OQ), one));
    __m256 z = _mm256_mul_ps(fx, _mm256_set1_ps(-2.12194440e-4f));
    x = _mm256_sub_ps(x, _mm256_mul_ps(fx, _mm256_set1_ps(0.693359375f)));
    x = _mm256_sub_ps(x, z);
    z = _mm256_mul_ps(x, x);
    __m256 y = _mm256_mul_ps(_mm256_set1_ps(1.9875691500E-4f), x);
    y = _mm256_add_ps(y, _mm256_set1_ps(1.3981999507E-3f));
    y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(8.3334519073E-3f));
    y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(4.1665795894E-2f));
    y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(1.6666665459E-1f));
    y = _mm256_add_ps(_mm256_mul_ps(y, x), _mm256_set1_ps(5.0000001201E-1f));
    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, x);
    y = _mm256_add_ps(y, one);
    const __m256i exponent =
        _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(fx), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(exponent));
}

void cn_gamma_approx_avx2(const float *src, float *out, size_t count, float gamma)
{
    const __m256 power = _mm256_set1_ps(gamma), zero = _mm256_setzero_ps(), one = _mm256_set1_ps(1.0f);
    for (size_t i = 0; i < count; i += 8) {
        const __m256 x = _mm256_loadu_ps(src + i);
        __m256 result = gamma_exp(_mm256_mul_ps(gamma_log(x), power));
        result = _mm256_blendv_ps(result, zero, _mm256_cmp_ps(x, zero, _CMP_EQ_OQ));
        result = _mm256_max_ps(result, zero);
        _mm256_storeu_ps(out + i, _mm256_min_ps(result, one));
    }
}
