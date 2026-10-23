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
 * from normal_nan_outputs (color_shared.h), never from the vector arithmetic.
 * Also the float colour conversions' BGR to HSV and HLS (hue_shared.h), reached through color_range. */
#include "color_shared.h"
#include "hue_shared.h"

#include <float.h>
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

void cn_rgb_hue_avx2(const float *src, float *dst, size_t count, size_t scn, int blue, int hls)
{
    const __m256i index = _mm256_mullo_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), _mm256_set1_epi32((int)scn));
    const __m256 epsilon = _mm256_set1_ps(FLT_EPSILON), half = _mm256_set1_ps(0.5f);
    const __m256 sixty = _mm256_set1_ps(60.0f), magnitude = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    for (size_t i = 0; i < count; i += 8, src += 8 * scn, dst += 24) {
        const __m256 b = _mm256_i32gather_ps(src + blue, index, 4), g = _mm256_i32gather_ps(src + 1, index, 4);
        const __m256 r = _mm256_i32gather_ps(src + (blue ^ 2), index, 4);
        const __m256 low = _mm256_min_ps(_mm256_min_ps(r, g), b), high = _mm256_max_ps(_mm256_max_ps(r, g), b);
        const __m256 diff = _mm256_sub_ps(high, low), sum = _mm256_add_ps(high, low);
        const __m256 l = _mm256_mul_ps(sum, half);
        __m256 s, reciprocal;
        if (hls) {
            s = _mm256_blendv_ps(_mm256_div_ps(diff, _mm256_sub_ps(_mm256_set1_ps(2.0f), sum)),
                _mm256_div_ps(diff, sum), _mm256_cmp_ps(l, half, _CMP_LT_OQ));
            reciprocal = _mm256_div_ps(sixty, diff);
        } else {
            s = _mm256_div_ps(diff, _mm256_add_ps(_mm256_and_ps(high, magnitude), epsilon));
            reciprocal = _mm256_div_ps(sixty, _mm256_add_ps(diff, epsilon));
        }
        const __m256 is_r = _mm256_cmp_ps(high, r, _CMP_EQ_OQ), is_g = _mm256_cmp_ps(high, g, _CMP_EQ_OQ);
        const __m256 delta = _mm256_blendv_ps(
            _mm256_blendv_ps(_mm256_sub_ps(r, g), _mm256_sub_ps(b, r), is_g), _mm256_sub_ps(g, b), is_r);
        const __m256 base = _mm256_blendv_ps(
            _mm256_blendv_ps(_mm256_set1_ps(240.0f), _mm256_set1_ps(120.0f), is_g),
            _mm256_and_ps(_mm256_cmp_ps(g, b, _CMP_LT_OQ), _mm256_set1_ps(360.0f)), is_r);
        __m256 h = _mm256_fmadd_ps(delta, reciprocal, base);
        if (hls) {
            /* Without a difference above FLT_EPSILON (a NaN one included) h and s stay +0. */
            const __m256 valid = _mm256_cmp_ps(diff, epsilon, _CMP_GT_OQ);
            h = _mm256_and_ps(valid, h);
            s = _mm256_and_ps(valid, s);
            store_three(dst, h, l, s);
        } else {
            store_three(dst, h, s, high);
        }
    }
}

void cn_hue_rgb_avx2(const float *src, float *dst, size_t count, size_t dcn, int blue, int hls)
{
    const __m256i index = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
    const __m256 one = _mm256_set1_ps(1.0f), two = _mm256_set1_ps(2.0f), three = _mm256_set1_ps(3.0f);
    const __m256 four = _mm256_set1_ps(4.0f), six = _mm256_set1_ps(6.0f);
    for (size_t i = 0; i < count; i += 8, src += 24, dst += 8 * dcn) {
        __m256 h = _mm256_i32gather_ps(src, index, 4);
        const __m256 s = _mm256_i32gather_ps(src + (hls ? 2 : 1), index, 4);
        const __m256 v = _mm256_i32gather_ps(src + (hls ? 1 : 2), index, 4);
        h = _mm256_mul_ps(h, _mm256_set1_ps(6.0f / 360.0f));
        const __m256 pre = _mm256_cvtepi32_ps(_mm256_cvttps_epi32(h)), original = h;
        h = _mm256_sub_ps(h, pre);
        const __m256 sixth = _mm256_mul_ps(hls ? original : pre, _mm256_set1_ps(1.0f / 6.0f));
        const __m256 sector = _mm256_sub_ps(pre, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvttps_epi32(sixth)), six));
        const __m256 below1 = _mm256_cmp_ps(sector, one, _CMP_LT_OQ), below2 = _mm256_cmp_ps(sector, two, _CMP_LT_OQ);
        const __m256 below4 = _mm256_cmp_ps(sector, four, _CMP_LT_OQ);
        __m256 b, g, r;
        if (hls) {
            const __m256 ls = _mm256_mul_ps(v, s);
            const __m256 element =
                _mm256_blendv_ps(_mm256_sub_ps(s, ls), ls, _mm256_cmp_ps(v, _mm256_set1_ps(0.5f), _CMP_LE_OQ));
            const __m256 twice = _mm256_add_ps(h, h);
            const __m256 t0 = _mm256_add_ps(v, element), t1 = _mm256_sub_ps(v, element);
            const __m256 t2 = _mm256_sub_ps(_mm256_add_ps(v, element), _mm256_mul_ps(element, twice));
            const __m256 t3 = _mm256_add_ps(_mm256_sub_ps(v, element), _mm256_mul_ps(element, twice));
            const __m256 upto2 = _mm256_cmp_ps(sector, two, _CMP_LE_OQ);
            const __m256 upto4 = _mm256_cmp_ps(sector, four, _CMP_LE_OQ);
            b = _mm256_blendv_ps(_mm256_blendv_ps(_mm256_blendv_ps(t2, t0, upto4), t3, upto2), t1, below2);
            g = _mm256_blendv_ps(_mm256_blendv_ps(_mm256_blendv_ps(t1, t2, below4), t0, upto2), t3, below1);
            r = _mm256_blendv_ps(_mm256_blendv_ps(t0, t3, upto4), t1, below4);
            r = _mm256_blendv_ps(_mm256_blendv_ps(r, t2, below2), t0, below1);
        } else {
            const __m256 t1 = _mm256_mul_ps(v, _mm256_sub_ps(one, s));
            const __m256 t2 = _mm256_mul_ps(v, _mm256_sub_ps(one, _mm256_mul_ps(s, h)));
            const __m256 t3 = _mm256_mul_ps(v, _mm256_sub_ps(one, _mm256_mul_ps(s, _mm256_sub_ps(one, h))));
            const __m256 at1 = _mm256_cmp_ps(sector, one, _CMP_EQ_OQ), at2 = _mm256_cmp_ps(sector, two, _CMP_EQ_OQ);
            const __m256 at3 = _mm256_cmp_ps(sector, three, _CMP_EQ_OQ), at4 = _mm256_cmp_ps(sector, four, _CMP_EQ_OQ);
            b = _mm256_and_ps(_mm256_cmp_ps(sector, four, _CMP_GT_OQ), t2);
            b = _mm256_blendv_ps(_mm256_blendv_ps(_mm256_blendv_ps(b, v, _mm256_or_ps(at3, at4)), t3, at2), t1, below2);
            g = _mm256_blendv_ps(s, t1, _mm256_cmp_ps(sector, three, _CMP_GT_OQ));
            g = _mm256_blendv_ps(_mm256_blendv_ps(_mm256_blendv_ps(g, t2, at3), v, _mm256_or_ps(at1, at2)), t3, below1);
            r = _mm256_blendv_ps(_mm256_blendv_ps(v, t3, at4), t1, _mm256_or_ps(at2, at3));
            r = _mm256_blendv_ps(_mm256_blendv_ps(r, t2, at1), v, below1);
        }
        const __m256 first = blue ? r : b, last = blue ? b : r;
        if (dcn == 4) store_four(dst, first, g, last, one);
        else store_three(dst, first, g, last);
    }
}

void cn_bgr_gray_avx2(const float *src, float *dst, size_t count, size_t scn, float cb, float cr, int ipp)
{
    const __m256i index = _mm256_mullo_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), _mm256_set1_epi32((int)scn));
    const __m256 blue = _mm256_set1_ps(cb), red = _mm256_set1_ps(cr), green = _mm256_set1_ps(0.587f);
    for (size_t i = 0; i < count; i += 8, src += 8 * scn) {
        const __m256 s0 = _mm256_i32gather_ps(src, index, 4), s1 = _mm256_i32gather_ps(src + 1, index, 4);
        const __m256 s2 = _mm256_i32gather_ps(src + 2, index, 4);
        const __m256 inner = ipp ? _mm256_fmadd_ps(s0, blue, _mm256_mul_ps(s1, green))
                                 : _mm256_fmadd_ps(s1, green, _mm256_mul_ps(s0, blue));
        _mm256_storeu_ps(dst + i, _mm256_fmadd_ps(s2, red, inner));
    }
}

void cn_bgr_yuv_avx2(const float *src, float *dst, size_t count, size_t scn, int blue, int ycrcb)
{
    const __m256i index = _mm256_mullo_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), _mm256_set1_epi32((int)scn));
    const __m256 c0 = _mm256_set1_ps(blue == 0 ? 0.114f : 0.299f), c2 = _mm256_set1_ps(blue == 0 ? 0.299f : 0.114f);
    const __m256 green = _mm256_set1_ps(0.587f), half = _mm256_set1_ps(0.5f);
    const __m256 kr = _mm256_set1_ps(ycrcb ? 0.713f : 0.877f), kb = _mm256_set1_ps(ycrcb ? 0.564f : 0.492f);
    for (size_t i = 0; i < count; i += 8, src += 8 * scn, dst += 24) {
        const __m256 s0 = _mm256_i32gather_ps(src, index, 4), s1 = _mm256_i32gather_ps(src + 1, index, 4);
        const __m256 s2 = _mm256_i32gather_ps(src + 2, index, 4);
        const __m256 y = _mm256_fmadd_ps(s0, c0, _mm256_fmadd_ps(s1, green, _mm256_mul_ps(s2, c2)));
        const __m256 cr = _mm256_fmadd_ps(_mm256_sub_ps(blue ? s0 : s2, y), kr, half);
        const __m256 cb = _mm256_fmadd_ps(_mm256_sub_ps(blue ? s2 : s0, y), kb, half);
        store_three(dst, y, ycrcb ? cr : cb, ycrcb ? cb : cr);
    }
}

void cn_yuv_bgr_avx2(const float *src, float *dst, size_t count, size_t dcn, int blue, int ycrcb)
{
    const __m256i index = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
    const __m256 half = _mm256_set1_ps(0.5f), one = _mm256_set1_ps(1.0f);
    const __m256 c0 = _mm256_set1_ps(ycrcb ? 1.403f : 1.140f), c1 = _mm256_set1_ps(ycrcb ? -0.714f : -0.581f);
    const __m256 c2 = _mm256_set1_ps(ycrcb ? -0.344f : -0.395f), c3 = _mm256_set1_ps(ycrcb ? 1.773f : 2.032f);
    for (size_t i = 0; i < count; i += 8, src += 24, dst += 8 * dcn) {
        const __m256 y = _mm256_i32gather_ps(src, index, 4);
        const __m256 cr = _mm256_sub_ps(_mm256_i32gather_ps(src + (ycrcb ? 1 : 2), index, 4), half);
        const __m256 cb = _mm256_sub_ps(_mm256_i32gather_ps(src + (ycrcb ? 2 : 1), index, 4), half);
        const __m256 b = _mm256_fmadd_ps(cb, c3, y), g = _mm256_fmadd_ps(cr, c1, _mm256_fmadd_ps(cb, c2, y));
        const __m256 r = _mm256_fmadd_ps(cr, c0, y);
        const __m256 first = blue ? r : b, last = blue ? b : r;
        if (dcn == 4) store_four(dst, first, g, last, one);
        else store_three(dst, first, g, last);
    }
}

/* One lattice entry's three int16 values for eight indices (in int16s): entries 0-1 from one 32-bit gather, 1-2 from
 * one at the next int16, so no gather reads past the entry's last value. */
static void lattice_entries(const int16_t *lattice, __m256i index, __m256i *e0, __m256i *e1, __m256i *e2)
{
    const __m256i low = _mm256_i32gather_epi32((const int *)lattice, index, 2);
    const __m256i high = _mm256_i32gather_epi32((const int *)lattice, _mm256_add_epi32(index, _mm256_set1_epi32(1)), 2);
    *e0 = _mm256_srai_epi32(_mm256_slli_epi32(low, 16), 16);
    *e1 = _mm256_srai_epi32(low, 16);
    *e2 = _mm256_srai_epi32(high, 16);
}

void cn_bgr_lab_avx2(const float *src, float *dst, size_t count, size_t scn, int blue, const int16_t *lattice)
{
    const __m256i index = _mm256_mullo_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7), _mm256_set1_epi32((int)scn));
    const __m256 one = _mm256_set1_ps(1.0f), zero = _mm256_setzero_ps();
    const __m256i sixteen = _mm256_set1_epi32(16), fifteen = _mm256_set1_epi32(15), top = _mm256_set1_epi32(32);
    for (size_t i = 0; i < count; i += 8, src += 8 * scn, dst += 24) {
        __m256i origin[3], fraction[3];
        for (int k = 0; k < 3; ++k) {
            const __m256 value = _mm256_i32gather_ps(src + (k == 1 ? 1 : k == 0 ? blue : blue ^ 2), index, 4);
            const __m256 clamped = _mm256_min_ps(_mm256_max_ps(value, zero), one);
            const __m256i coordinate = _mm256_cvtps_epi32(_mm256_mul_ps(clamped, _mm256_set1_ps(16384.0f)));
            origin[k] = _mm256_srai_epi32(coordinate, 9);
            fraction[k] = _mm256_and_si256(_mm256_srai_epi32(coordinate, 5), fifteen);
        }
        __m256i accum0 = _mm256_setzero_si256(), accum1 = accum0, accum2 = accum0;
        for (int corner = 0; corner < 8; ++corner) {
            const __m256i p = _mm256_min_epi32(_mm256_add_epi32(origin[0], _mm256_set1_epi32((corner >> 2) & 1)), top);
            const __m256i q = _mm256_min_epi32(_mm256_add_epi32(origin[1], _mm256_set1_epi32((corner >> 1) & 1)), top);
            const __m256i r = _mm256_min_epi32(_mm256_add_epi32(origin[2], _mm256_set1_epi32(corner & 1)), top);
            const __m256i w0 = corner & 4 ? fraction[0] : _mm256_sub_epi32(sixteen, fraction[0]);
            const __m256i w1 = corner & 2 ? fraction[1] : _mm256_sub_epi32(sixteen, fraction[1]);
            const __m256i w2 = corner & 1 ? fraction[2] : _mm256_sub_epi32(sixteen, fraction[2]);
            const __m256i weight = _mm256_mullo_epi32(_mm256_mullo_epi32(w0, w1), w2);
            const __m256i cell = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_add_epi32(
                _mm256_mullo_epi32(r, _mm256_set1_epi32(33)), q), _mm256_set1_epi32(33)), p);
            __m256i e0, e1, e2;
            lattice_entries(lattice, _mm256_mullo_epi32(cell, _mm256_set1_epi32(3)), &e0, &e1, &e2);
            accum0 = _mm256_add_epi32(accum0, _mm256_mullo_epi32(e0, weight));
            accum1 = _mm256_add_epi32(accum1, _mm256_mullo_epi32(e1, weight));
            accum2 = _mm256_add_epi32(accum2, _mm256_mullo_epi32(e2, weight));
        }
        const __m256i round = _mm256_set1_epi32(2048);
        const __m256 l = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_srai_epi32(_mm256_add_epi32(accum0, round), 12)),
                                       _mm256_set1_ps(100.0f / 16384.0f));
        const __m256 scale = _mm256_set1_ps(256.0f / 16384.0f), offset = _mm256_set1_ps(128.0f);
        const __m256 a = _mm256_sub_ps(
            _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_srai_epi32(_mm256_add_epi32(accum1, round), 12)), scale), offset);
        const __m256 b = _mm256_sub_ps(
            _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_srai_epi32(_mm256_add_epi32(accum2, round), 12)), scale), offset);
        store_three(dst, l, a, b);
    }
}

/* lab_rgb's x and z segment: v <= 6/29 ? (v - 16/116) / 7.787 (as a product) : v cubed; NaN takes the cube. */
static __m256 lab_segment(__m256 v)
{
    const __m256 low = _mm256_mul_ps(_mm256_sub_ps(v, _mm256_set1_ps(16.0f / 116.0f)), _mm256_set1_ps(1.0f / 7.787f));
    const __m256 cube = _mm256_mul_ps(_mm256_mul_ps(v, v), v);
    return _mm256_blendv_ps(cube, low, _mm256_cmp_ps(v, _mm256_set1_ps(6.0f / 29.0f), _CMP_LE_OQ));
}

/* inverse_gamma unfused for clipped values (0 to 1, or -0): truncated index 0-1023, the cubic's entries gathered. */
static __m256 lab_inverse_gamma8(__m256 value, const float *table)
{
    value = _mm256_mul_ps(value, _mm256_set1_ps(1024.0f));
    __m256i index = _mm256_cvttps_epi32(value);
    index = _mm256_min_epi32(_mm256_max_epi32(index, _mm256_setzero_si256()), _mm256_set1_epi32(1023));
    value = _mm256_sub_ps(value, _mm256_cvtepi32_ps(index));
    const __m256i entry = _mm256_slli_epi32(index, 2);
    __m256 result = _mm256_add_ps(_mm256_mul_ps(_mm256_i32gather_ps(table + 3, entry, 4), value),
                                  _mm256_i32gather_ps(table + 2, entry, 4));
    result = _mm256_add_ps(_mm256_mul_ps(result, value), _mm256_i32gather_ps(table + 1, entry, 4));
    return _mm256_add_ps(_mm256_mul_ps(result, value), _mm256_i32gather_ps(table, entry, 4));
}

void cn_lab_bgr_avx2(const float *src, float *dst, size_t count, size_t dcn, int blue, const float *xyz,
                     const float *gamma_table)
{
    const __m256i index = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
    const __m256 one = _mm256_set1_ps(1.0f), zero = _mm256_setzero_ps();
    for (size_t i = 0; i < count; i += 8, src += 24, dst += 8 * dcn) {
        const __m256 l = _mm256_i32gather_ps(src, index, 4), a = _mm256_i32gather_ps(src + 1, index, 4);
        const __m256 b = _mm256_i32gather_ps(src + 2, index, 4);
        const __m256 dark = _mm256_cmp_ps(l, _mm256_set1_ps(8.0f), _CMP_LE_OQ);
        const __m256 y_dark = _mm256_mul_ps(l, _mm256_set1_ps(1.0f / 903.3f));
        const __m256 fy_dark = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(7.787f), y_dark),
                                             _mm256_set1_ps(16.0f / 116.0f));
        const __m256 fy_light = _mm256_mul_ps(_mm256_add_ps(l, _mm256_set1_ps(16.0f)), _mm256_set1_ps(1.0f / 116.0f));
        const __m256 fy = _mm256_blendv_ps(fy_light, fy_dark, dark);
        const __m256 y = _mm256_blendv_ps(_mm256_mul_ps(_mm256_mul_ps(fy_light, fy_light), fy_light), y_dark, dark);
        const __m256 x = lab_segment(_mm256_add_ps(_mm256_mul_ps(a, _mm256_set1_ps(1.0f / 500.0f)), fy));
        const __m256 z = lab_segment(_mm256_add_ps(_mm256_mul_ps(b, _mm256_set1_ps(-1.0f / 200.0f)), fy));
        __m256 out[3];
        for (int k = 0; k < 3; ++k) {
            const float *c = xyz + 3 * k;
            const __m256 result = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(c[0]), x),
                _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(c[1]), y), _mm256_mul_ps(_mm256_set1_ps(c[2]), z)));
            out[k] = lab_inverse_gamma8(_mm256_max_ps(zero, _mm256_min_ps(result, one)), gamma_table);
        }
        /* out[0] is channel blue ^ 2, out[2] channel blue. */
        const __m256 first = blue ? out[0] : out[2], last = blue ? out[2] : out[0];
        if (dcn == 4) store_four(dst, first, out[1], last, one);
        else store_three(dst, first, out[1], last);
    }
}
