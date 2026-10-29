#if !defined(_M_X64) && !(defined(__linux__) && defined(__x86_64__))
#error "yuv_planes_avx2.c is an x86-64 /arch:AVX2 unit"
#endif
/* YUV plane codes and values, AVX2 unit (/arch:AVX2, spec 4.3): no export,
 * reached only through the entries' dispatch on the call's ISA level. Each lane
 * runs its element's sequence of yuv_planes_shared.h; groups of 8 start at the
 * segment's first column, and the last (last - first) % 8 elements run the scalar
 * segment. */
#include "yuv_planes_shared.h"
#include <immintrin.h>

/* Codes widened (vpmovzx), converted exactly (vcvtdq2ps), then vsubps, vdivps and,
 * for chroma, vaddps: code_value's operations. */
static __m256 code_lanes(__m256i codes, __m256 offset, __m256 scale, int chroma)
{
    __m256 value = _mm256_div_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(codes), offset), scale);
    return chroma ? _mm256_add_ps(value, _mm256_set1_ps(0.5f)) : value;
}

void cn_codes_to_f32_segment_avx2(const plane_codes_job *job, size_t y, size_t first,
    size_t last, size_t start)
{
    const __m256 offset = _mm256_set1_ps(job->offset), scale = _mm256_set1_ps(job->scale);
    float *out = (float *)job->out + start - first;
    size_t x = first;
    if (job->wide) {
        const uint16_t *row = (const uint16_t *)job->src + y * job->row_stride;
        for (; last - x >= 8; x += 8) {
            __m256i codes = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(row + x)));
            _mm256_storeu_ps(out + x, code_lanes(codes, offset, scale, job->chroma));
        }
    } else {
        const uint8_t *row = (const uint8_t *)job->src + y * job->row_stride;
        for (; last - x >= 8; x += 8) {
            __m256i codes = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(row + x)));
            _mm256_storeu_ps(out + x, code_lanes(codes, offset, scale, job->chroma));
        }
    }
    codes_to_f32_segment(job, y, x, last, start + (x - first));
}

/* value_code per lane: vsubps for chroma, vmulps, vaddps; vmaxps(scaled, 0), which
 * returns its second operand for NaN, and vminps(maximum); vroundps to nearest
 * even; vcvttps2dq, exact on those integers; then unsigned saturating packs, which
 * leave codes in [0, 65535] as they are. */
static __m128i code_words(__m256 value, const plane_codes_job *job)
{
    if (job->chroma) value = _mm256_sub_ps(value, _mm256_set1_ps(0.5f));
    __m256 scaled = _mm256_add_ps(_mm256_mul_ps(value, _mm256_set1_ps(job->scale)),
        _mm256_set1_ps(job->offset));
    scaled = _mm256_max_ps(scaled, _mm256_setzero_ps());
    scaled = _mm256_min_ps(scaled, _mm256_set1_ps(job->maximum));
    __m256i codes = _mm256_cvttps_epi32(
        _mm256_round_ps(scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    return _mm_packus_epi32(_mm256_castsi256_si128(codes), _mm256_extracti128_si256(codes, 1));
}

void cn_f32_to_codes_segment_avx2(const plane_codes_job *job, size_t y, size_t first,
    size_t last, size_t start)
{
    const float *row = (const float *)job->src + y * job->row_stride;
    size_t x = first;
    if (job->wide) {
        uint16_t *out = (uint16_t *)job->out + start - first;
        for (; last - x >= 8; x += 8)
            _mm_storeu_si128((__m128i *)(out + x), code_words(_mm256_loadu_ps(row + x), job));
    } else {
        uint8_t *out = (uint8_t *)job->out + start - first;
        for (; last - x >= 8; x += 8) {
            __m128i words = code_words(_mm256_loadu_ps(row + x), job);
            _mm_storel_epi64((__m128i *)(out + x), _mm_packus_epi16(words, words));
        }
    }
    f32_to_codes_segment(job, y, x, last, start + (x - first));
}

/* (m[0] * a + m[1] * b) + m[2] * c per lane. */
static __m256 matrix_lanes(const float *m, __m256 a, __m256 b, __m256 c)
{
    __m256 sum = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(m[0]), a),
        _mm256_mul_ps(_mm256_set1_ps(m[1]), b));
    return _mm256_add_ps(sum, _mm256_mul_ps(_mm256_set1_ps(m[2]), c));
}

/* Eight BGR pixels as three vectors of 8 values: with pb, pg and pr the B, G and R
 * lanes permuted by (0 3 6 1 4 7 2 5), (5 0 3 6 1 4 7 2) and (2 5 0 3 6 1 4 7), each
 * output vector takes lanes 0, 3, 6 from one of them, 1, 4, 7 from the next and 2,
 * 5 from the third (blend masks 0x49 and 0x92 over the third). */
#define CN_LANES_036 0x49
#define CN_LANES_147 0x92

void cn_yuv_to_bgr_row_avx2(const yuv_matrix_job *job, size_t row)
{
    const float *y = job->y + row * job->y_stride, *u = job->u + row * job->u_stride;
    const float *v = job->v + row * job->v_stride, *m = job->matrix;
    float *out = job->bgr_out + row * job->width * 3;
    const __m256 centre = _mm256_set1_ps(job->centre);
    const __m256i order_b = _mm256_setr_epi32(0, 3, 6, 1, 4, 7, 2, 5);
    const __m256i order_g = _mm256_setr_epi32(5, 0, 3, 6, 1, 4, 7, 2);
    const __m256i order_r = _mm256_setr_epi32(2, 5, 0, 3, 6, 1, 4, 7);
    int centred = job->centre != 0.0f;
    size_t x = 0;
    for (; job->width - x >= 8; x += 8) {
        __m256 luma = _mm256_loadu_ps(y + x), cb = _mm256_loadu_ps(u + x);
        __m256 cr = _mm256_loadu_ps(v + x);
        if (centred) {
            cb = _mm256_sub_ps(cb, centre);
            cr = _mm256_sub_ps(cr, centre);
        }
        __m256 pr = _mm256_permutevar8x32_ps(matrix_lanes(m, luma, cb, cr), order_r);
        __m256 pg = _mm256_permutevar8x32_ps(matrix_lanes(m + 3, luma, cb, cr), order_g);
        __m256 pb = _mm256_permutevar8x32_ps(matrix_lanes(m + 6, luma, cb, cr), order_b);
        float *p = out + 3 * x;
        _mm256_storeu_ps(p, _mm256_blend_ps(_mm256_blend_ps(pr, pb, CN_LANES_036), pg, CN_LANES_147));
        _mm256_storeu_ps(p + 8, _mm256_blend_ps(_mm256_blend_ps(pg, pr, CN_LANES_036), pb, CN_LANES_147));
        _mm256_storeu_ps(p + 16, _mm256_blend_ps(_mm256_blend_ps(pb, pg, CN_LANES_036), pr, CN_LANES_147));
    }
    yuv_to_bgr_row(job, row, x);
}

/* The inverse: each of B, G and R is gathered from the three loaded vectors by the
 * same blends, then put in order by the inverse permutations (B's and R's are
 * their own inverses; G's is (1 4 7 2 5 0 3 6)). */
void cn_bgr_to_yuv_row_avx2(const yuv_matrix_job *job, size_t row)
{
    const float *bgr = job->bgr + row * job->bgr_stride, *m = job->matrix;
    size_t offset = row * job->width;
    const __m256 centre = _mm256_set1_ps(job->centre);
    const __m256i order_b = _mm256_setr_epi32(0, 3, 6, 1, 4, 7, 2, 5);
    const __m256i order_g = _mm256_setr_epi32(1, 4, 7, 2, 5, 0, 3, 6);
    const __m256i order_r = _mm256_setr_epi32(2, 5, 0, 3, 6, 1, 4, 7);
    int centred = job->centre != 0.0f;
    size_t x = 0;
    for (; job->width - x >= 8; x += 8) {
        const float *p = bgr + 3 * x;
        __m256 in0 = _mm256_loadu_ps(p), in1 = _mm256_loadu_ps(p + 8);
        __m256 in2 = _mm256_loadu_ps(p + 16);
        __m256 b = _mm256_permutevar8x32_ps(_mm256_blend_ps(_mm256_blend_ps(in2, in0,
            CN_LANES_036), in1, CN_LANES_147), order_b);
        __m256 g = _mm256_permutevar8x32_ps(_mm256_blend_ps(_mm256_blend_ps(in1, in2,
            CN_LANES_036), in0, CN_LANES_147), order_g);
        __m256 r = _mm256_permutevar8x32_ps(_mm256_blend_ps(_mm256_blend_ps(in0, in1,
            CN_LANES_036), in2, CN_LANES_147), order_r);
        __m256 cb = matrix_lanes(m + 3, r, g, b), cr = matrix_lanes(m + 6, r, g, b);
        if (centred) {
            cb = _mm256_add_ps(cb, centre);
            cr = _mm256_add_ps(cr, centre);
        }
        _mm256_storeu_ps(job->y_out + offset + x, matrix_lanes(m, r, g, b));
        _mm256_storeu_ps(job->u_out + offset + x, cb);
        _mm256_storeu_ps(job->v_out + offset + x, cr);
    }
    bgr_to_yuv_row(job, row, x);
}
