#if !defined(_M_X64)
#error "color_avx2.c is an x86-64 /arch:AVX2 unit"
#endif
/* Normal map output, AVX2 unit (/arch:AVX2, spec 4.3): no export, reached only through
 * normal_output_range's dispatch on the call's ISA level (SP4b Task 8). Whole groups of
 * 8 pixels run normal_pixels' operations in its order, lane by lane: vsqrtps for B3's
 * CRT sqrtf (sqrtss on every input the sum can be: non-negative, +inf or a NaN, which
 * it quiets), vdivps, the inversions as sign xors and fabsf as a sign clear (B3's
 * andps on the widened double, whose NaN payload survives the round trip); the
 * channels are interleaved by permutes, which move bits. The chunk's last pixels run
 * normal_pixels. A pixel with a non-finite dx or dy then takes its NaN outputs' bits
 * from normal_nan_outputs (color_shared.h), never from the vector arithmetic. */
#include "color_shared.h"

#include <immintrin.h>

/* 8 pixels of (z, y, x): lane j of the k-th vector takes pixel index[k][j]'s channel
 * j + k * 8 mod 3, picked by the blends' masks (bit j set: the y or the x lane). */
static void store_three(float *pixel, __m256 z, __m256 y, __m256 x)
{
    const __m256i first = _mm256_setr_epi32(0, 0, 0, 1, 1, 1, 2, 2);
    const __m256i second = _mm256_setr_epi32(2, 3, 3, 3, 4, 4, 4, 5);
    const __m256i third = _mm256_setr_epi32(5, 5, 6, 6, 6, 7, 7, 7);
    __m256 out = _mm256_blend_ps(_mm256_permutevar8x32_ps(z, first),
        _mm256_permutevar8x32_ps(y, first), 0x92);
    _mm256_storeu_ps(pixel, _mm256_blend_ps(out, _mm256_permutevar8x32_ps(x, first), 0x24));
    out = _mm256_blend_ps(_mm256_permutevar8x32_ps(z, second),
        _mm256_permutevar8x32_ps(y, second), 0x24);
    _mm256_storeu_ps(pixel + 8, _mm256_blend_ps(out, _mm256_permutevar8x32_ps(x, second), 0x49));
    out = _mm256_blend_ps(_mm256_permutevar8x32_ps(z, third),
        _mm256_permutevar8x32_ps(y, third), 0x49);
    _mm256_storeu_ps(pixel + 16, _mm256_blend_ps(out, _mm256_permutevar8x32_ps(x, third), 0x92));
}

/* 8 pixels of (z, y, x, a): a 4 x 8 transpose. */
static void store_four(float *pixel, __m256 z, __m256 y, __m256 x, __m256 a)
{
    __m256 zy_low = _mm256_unpacklo_ps(z, y), zy_high = _mm256_unpackhi_ps(z, y);
    __m256 xa_low = _mm256_unpacklo_ps(x, a), xa_high = _mm256_unpackhi_ps(x, a);
    __m256 p0 = _mm256_shuffle_ps(zy_low, xa_low, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 p1 = _mm256_shuffle_ps(zy_low, xa_low, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 p2 = _mm256_shuffle_ps(zy_high, xa_high, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 p3 = _mm256_shuffle_ps(zy_high, xa_high, _MM_SHUFFLE(3, 2, 3, 2));
    _mm256_storeu_ps(pixel, _mm256_permute2f128_ps(p0, p1, 0x20));
    _mm256_storeu_ps(pixel + 8, _mm256_permute2f128_ps(p2, p3, 0x20));
    _mm256_storeu_ps(pixel + 16, _mm256_permute2f128_ps(p0, p1, 0x31));
    _mm256_storeu_ps(pixel + 24, _mm256_permute2f128_ps(p2, p3, 0x31));
}

void cn_normal_output_avx2(const normal_output_job *job, size_t begin, size_t end)
{
    const float *dx = job->dx, *dy = job->dy, *alpha = job->alpha;
    size_t channels = (size_t)job->channels, i = begin;
    float *pixel = job->dst + begin * channels;
    const __m256 sign = _mm256_set1_ps(-0.0f), one = _mm256_set1_ps(1.0f);
    const __m256 two = _mm256_set1_ps(2.0f), four = _mm256_set1_ps(4.0f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256i exponent = _mm256_set1_epi32(0x7F800000);
    /* -x is a sign xor; without the inversion the xor with +0.0 keeps every bit. */
    const __m256 flip_r = job->invert_r ? sign : _mm256_setzero_ps();
    const __m256 flip_g = job->invert_g ? sign : _mm256_setzero_ps();
    for (; end - i >= 8; i += 8, pixel += 8 * channels) {
        __m256 x = _mm256_loadu_ps(dx + i), y = _mm256_loadu_ps(dy + i);
        __m256 square = _mm256_mul_ps(x, x);
        __m256 yy = _mm256_mul_ps(y, y);
        square = _mm256_add_ps(square, yy);
        square = _mm256_add_ps(square, four);
        __m256 length = _mm256_sqrt_ps(square);
        x = _mm256_div_ps(x, length);
        y = _mm256_div_ps(y, length);
        __m256 z = _mm256_div_ps(two, length);
        x = _mm256_add_ps(_mm256_xor_ps(x, flip_r), one);
        y = _mm256_add_ps(_mm256_xor_ps(y, flip_g), one);
        z = _mm256_andnot_ps(sign, z);
        y = _mm256_mul_ps(y, half);
        x = _mm256_mul_ps(x, half);
        if (channels == 4)
            store_four(pixel, z, y, x, alpha ? _mm256_loadu_ps(alpha + i) : one);
        else
            store_three(pixel, z, y, x);
        /* Pixels with a non-finite dx or dy take their NaN outputs from
           normal_nan_outputs, as in normal_pixels. */
        __m256i dx_bits = _mm256_loadu_si256((const __m256i *)(dx + i));
        __m256i dy_bits = _mm256_loadu_si256((const __m256i *)(dy + i));
        __m256i special = _mm256_or_si256(
            _mm256_cmpeq_epi32(_mm256_and_si256(dx_bits, exponent), exponent),
            _mm256_cmpeq_epi32(_mm256_and_si256(dy_bits, exponent), exponent));
        int lanes = _mm256_movemask_ps(_mm256_castsi256_ps(special));
        for (size_t lane = 0; lanes; ++lane, lanes >>= 1)
            if (lanes & 1) normal_nan_outputs(job, i + lane, pixel + lane * channels);
    }
    normal_pixels(job, i, end);
}
