#if !defined(_M_X64)
#error "linear_light_avx2.c is an x86-64 /arch:AVX2 unit"
#endif
/* Linear Light, AVX2 unit (/arch:AVX2, spec 4.3): no export, reached only through cn_linear_light's dispatch on the
 * call's ISA level (linear_light_ops.c). Four values at a time, widened to double, run light_pow's and
 * linear_light.h's operations in their order, one instruction for each scalar operation: vmulpd, vdivpd, vaddpd and
 * vsubpd (no fused multiply-add), compare masks and blends for the ?: choices, and integer shifts, adds and masks for
 * the bit casts; fabs and copysign are the sign bit's and/andnot/or. */
#include "linear_light.h"
#include "linear_light_shared.h"

#include <immintrin.h>

/* c[0] + c[1] x + c[2] x^2 + c[3] x^3 */
static __m256d light_pair(const double *c, __m256d x, __m256d x2)
{
    const __m256d low = _mm256_add_pd(_mm256_set1_pd(c[0]), _mm256_mul_pd(_mm256_set1_pd(c[1]), x));
    const __m256d high = _mm256_add_pd(_mm256_set1_pd(c[2]), _mm256_mul_pd(_mm256_set1_pd(c[3]), x));
    return _mm256_add_pd(low, _mm256_mul_pd(x2, high));
}

static __m256d light_poly8(const double *c, __m256d x)
{
    const __m256d x2 = _mm256_mul_pd(x, x), x4 = _mm256_mul_pd(x2, x2);
    return _mm256_add_pd(light_pair(c, x, x2), _mm256_mul_pd(x4, light_pair(c + 4, x, x2)));
}

static __m256d light_poly12(const double *c, __m256d x)
{
    const __m256d x2 = _mm256_mul_pd(x, x), x4 = _mm256_mul_pd(x2, x2);
    return _mm256_add_pd(light_poly8(c, x), _mm256_mul_pd(_mm256_mul_pd(x4, x4), light_pair(c + 8, x, x2)));
}

static __m256d light_pow(__m256d a, double power)
{
    const __m256d one = _mm256_set1_pd(1.0), g = _mm256_set1_pd(power);
    const __m256i bits = _mm256_castpd_si256(a);
    __m256d e = _mm256_sub_pd(
        _mm256_castsi256_pd(_mm256_or_si256(_mm256_set1_epi64x(0x4330000000000000), _mm256_srli_epi64(bits, 52))),
        _mm256_set1_pd(LIGHT_EXPONENT_BIAS));
    __m256d m = _mm256_castsi256_pd(_mm256_or_si256(_mm256_and_si256(bits, _mm256_set1_epi64x(0x000fffffffffffff)),
                                                    _mm256_set1_epi64x(0x3ff0000000000000)));
    const __m256d fold = _mm256_cmp_pd(m, _mm256_set1_pd(LIGHT_SQRT2), _CMP_GT_OQ);
    m = _mm256_blendv_pd(m, _mm256_mul_pd(m, _mm256_set1_pd(0.5)), fold);
    e = _mm256_add_pd(e, _mm256_and_pd(fold, one));
    const __m256d s = _mm256_div_pd(_mm256_sub_pd(m, one), _mm256_add_pd(m, one));
    const __m256d p = light_poly8(light_log_terms, _mm256_mul_pd(s, s));
    const __m256d log2a = _mm256_add_pd(
        e, _mm256_mul_pd(_mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(2.0), s), p), _mm256_set1_pd(LIGHT_INV_LN2)));
    const __m256d t = _mm256_mul_pd(g, log2a);
    const __m256d round = _mm256_set1_pd(LIGHT_ROUND);
    const __m256d shifted = _mm256_add_pd(t, round);
    const __m256d f = _mm256_mul_pd(_mm256_sub_pd(t, _mm256_sub_pd(shifted, round)), _mm256_set1_pd(LIGHT_LN2));
    const __m256i scale =
        _mm256_slli_epi64(_mm256_add_epi64(_mm256_castpd_si256(shifted), _mm256_set1_epi64x(1023)), 52);
    __m256d r = _mm256_mul_pd(light_poly12(light_exp_terms, f), _mm256_castsi256_pd(scale));
    r = _mm256_blendv_pd(a, r, _mm256_cmp_pd(a, _mm256_set1_pd(INFINITY), _CMP_LT_OQ));
    return _mm256_blendv_pd(r, _mm256_setzero_pd(), _mm256_cmp_pd(a, _mm256_setzero_pd(), _CMP_EQ_OQ));
}

/* cn_light_linearize or cn_light_encode of |value| (a), before the sign. */
static __m256d light_curve(__m256d a, int transfer, int encode)
{
    const __m256d zero = _mm256_setzero_pd();
    __m256d low = zero, high = zero, knee = zero;
    int inclusive = 0;
    switch (transfer * 2 + encode) {
    case CN_LIGHT_GAMMA24 * 2: return light_pow(a, 2.4);
    case CN_LIGHT_GAMMA24 * 2 + 1: return light_pow(a, 1 / 2.4);
    case CN_LIGHT_GAMMA22 * 2: return light_pow(a, 2.2);
    case CN_LIGHT_GAMMA22 * 2 + 1: return light_pow(a, 1 / 2.2);
    case CN_LIGHT_SRGB * 2:
        low = _mm256_div_pd(a, _mm256_set1_pd(12.92));
        high = light_pow(_mm256_div_pd(_mm256_add_pd(a, _mm256_set1_pd(.055)), _mm256_set1_pd(1.055)), 2.4);
        knee = _mm256_set1_pd(.04045), inclusive = 1;
        break;
    case CN_LIGHT_SRGB * 2 + 1:
        low = _mm256_mul_pd(_mm256_set1_pd(12.92), a);
        high = _mm256_sub_pd(_mm256_mul_pd(_mm256_set1_pd(1.055), light_pow(a, 1 / 2.4)), _mm256_set1_pd(.055));
        knee = _mm256_set1_pd(.0031308), inclusive = 1;
        break;
    case CN_LIGHT_BT709 * 2:
        low = _mm256_div_pd(a, _mm256_set1_pd(4.5));
        high = light_pow(_mm256_div_pd(_mm256_add_pd(a, _mm256_set1_pd(.099)), _mm256_set1_pd(1.099)), 1 / .45);
        knee = _mm256_set1_pd(.081);
        break;
    default:
        low = _mm256_mul_pd(_mm256_set1_pd(4.5), a);
        high = _mm256_sub_pd(_mm256_mul_pd(_mm256_set1_pd(1.099), light_pow(a, .45)), _mm256_set1_pd(.099));
        knee = _mm256_set1_pd(.018);
        break;
    }
    const __m256d below = inclusive ? _mm256_cmp_pd(a, knee, _CMP_LE_OQ) : _mm256_cmp_pd(a, knee, _CMP_LT_OQ);
    return _mm256_blendv_pd(high, low, below);
}

void cn_light_values_avx2(float *values, size_t count, size_t channels, int skip_alpha, int transfer, int encode)
{
    const __m256d sign = _mm256_set1_pd(-0.0);
    /* Lanes left as they are: a pixel's last channel when alpha is skipped. */
    const __m128 keep = _mm_castsi128_ps(!skip_alpha      ? _mm_setzero_si128()
                                         : channels == 4 ? _mm_set_epi32(-1, 0, 0, 0)
                                                         : _mm_set_epi32(-1, 0, -1, 0));
    for (size_t i = 0; i < count; i += 4) {
        const __m128 x = _mm_loadu_ps(values + i);
        const __m256d value = _mm256_cvtps_pd(x);
        const __m256d v = light_curve(_mm256_andnot_pd(sign, value), transfer, encode);
        const __m128 y = _mm256_cvtpd_ps(_mm256_or_pd(v, _mm256_and_pd(sign, value)));
        _mm_storeu_ps(values + i, _mm_blendv_ps(y, x, keep));
    }
}
