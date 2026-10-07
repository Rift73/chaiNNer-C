#if !defined(_M_X64)
#error "convert_avx2.c is an x86-64 /arch:AVX2 unit"
#endif
/* The pixel conversions' AVX2 unit (/arch:AVX2, spec 4.3): no export, reached only
 * through convert_range's dispatch on the call's ISA level (SP4b D7) and
 * cn_pixels_normalized_f32's (Task 3b). Each lane runs its element's sequence in
 * convert_shared.h (B3's; the clamp is np.clip's: clamp_lanes); groups of 8 start at
 * indices = 0 (mod 8) inside [begin, end) and never cross the chunk's ends, and the
 * head and tail run convert_scalar. The normalized scan is one serial any-match
 * pass over integer words. */
#include "convert_shared.h"
#include <immintrin.h>

/* u8 -> f32: vpmovzxbd, vcvtdq2ps (exact), vdivps by 255.0f: rounded in the
 * caller's mode like B3's divss (u8_quotients in round to nearest; convert_scalar
 * divides otherwise). The clamp is the identity on them. */
static void u8_f32(const uint8_t *src, float *out, size_t i, size_t end)
{
    const __m256 scale = _mm256_set1_ps(255.0f);
    for (; end - i >= 8; i += 8) {
        __m256i bytes = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(src + i)));
        _mm256_storeu_ps(out + i, _mm256_div_ps(_mm256_cvtepi32_ps(bytes), scale));
    }
}

/* clamp_unit per lane (np.clip(x, 0, 1)): vmaxps(+0, v) then vminps(1, v), NumPy
 * 2.5.3's own form, so -0, a NaN's payload and the DAZ handling are clamp_unit's. */
static __m256 clamp_lanes(__m256 v)
{
    return _mm256_min_ps(_mm256_set1_ps(1.0f), _mm256_max_ps(_mm256_setzero_ps(), v));
}

/* f32 -> f32: clamp_lanes, or a copy of the bits when normalize is 0. */
static void f32_f32(const float *src, float *out, int normalize, size_t i, size_t end)
{
    if (!normalize) {
        for (; end - i >= 8; i += 8)
            _mm256_storeu_si256((__m256i *)(out + i), _mm256_loadu_si256((const __m256i *)(src + i)));
        return;
    }
    for (; end - i >= 8; i += 8)
        _mm256_storeu_ps(out + i, clamp_lanes(_mm256_loadu_ps(src + i)));
}

/* f32 -> u8: clamp_lanes when normalize; scaled = 255 * value (vmulps); events from
 * bit masks: 1, a finite value with an infinite scaled value; 4, a signalling NaN
 * source; 2, scaled NaN or outside [-2^31, 2^31), which stores B3's INT32_MIN.
 * Otherwise vroundps to nearest even (independent of MXCSR) and vcvtps2dq (exact on
 * integers); the low byte of each int32 is stored (vpshufb, vpermd: truncation,
 * never a saturating pack). */
static long f32_u8(const float *src, uint8_t *out, int normalize, size_t i, size_t end)
{
    const __m256 scale = _mm256_set1_ps(255.0f);
    const __m256 low = _mm256_set1_ps(-2147483648.0f), high = _mm256_set1_ps(2147483648.0f);
    const __m256i magnitude = _mm256_set1_epi32(0x7fffffff);
    const __m256i infinity = _mm256_set1_epi32(0x7f800000);
    const __m256i largest = _mm256_set1_epi32(0x7f7fffff);
    const __m256i quiet = _mm256_set1_epi32(0x00400000);
    const __m256i sentinel = _mm256_set1_epi32(INT32_MIN);
    const __m256i none = _mm256_setzero_si256();
    /* Byte 0 of each int32 to the low 4 bytes of its 128-bit lane, then both lanes'
     * low dwords together. */
    const __m256i bytes = _mm256_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                           0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    const __m256i gather = _mm256_setr_epi32(0, 4, 0, 0, 0, 0, 0, 0);
    __m256i overflow = none, invalid = none, signalling = none;
    long events = 0;
    for (; end - i >= 8; i += 8) {
        __m256 source = _mm256_loadu_ps(src + i);
        __m256 value = normalize ? clamp_lanes(source) : source;
        __m256 scaled = _mm256_mul_ps(scale, value);
        __m256i value_bits = _mm256_and_si256(_mm256_castps_si256(value), magnitude);
        __m256i scaled_bits = _mm256_and_si256(_mm256_castps_si256(scaled), magnitude);
        __m256i source_bits = _mm256_castps_si256(source);
        __m256i nonfinite = _mm256_cmpgt_epi32(value_bits, largest);
        __m256i infinite = _mm256_cmpeq_epi32(scaled_bits, infinity);
        overflow = _mm256_or_si256(overflow, _mm256_andnot_si256(nonfinite, infinite));
        __m256i nan = _mm256_cmpgt_epi32(_mm256_and_si256(source_bits, magnitude), infinity);
        __m256i loud = _mm256_cmpeq_epi32(_mm256_and_si256(source_bits, quiet), none);
        signalling = _mm256_or_si256(signalling, _mm256_and_si256(nan, loud));
        __m256 inside = _mm256_and_ps(_mm256_cmp_ps(scaled, low, _CMP_GE_OQ),
                                      _mm256_cmp_ps(scaled, high, _CMP_LT_OQ));
        __m256i outside = _mm256_cmpeq_epi32(_mm256_castps_si256(inside), none);
        invalid = _mm256_or_si256(invalid, outside);
        __m256i integers = _mm256_cvtps_epi32(
            _mm256_round_ps(scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        integers = _mm256_blendv_epi8(integers, sentinel, outside);
        __m256i packed = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(integers, bytes), gather);
        _mm_storel_epi64((__m128i *)(out + i), _mm256_castsi256_si128(packed));
    }
    if (!_mm256_testz_si256(overflow, overflow)) events |= 1;
    if (!_mm256_testz_si256(invalid, invalid)) events |= 2;
    if (!_mm256_testz_si256(signalling, signalling)) events |= 4;
    return events;
}

/* cn_pixels_normalized_f32's bit tests per lane: all ones where the word is +0 or in
 * [0x00800000, 0x3f800000] (a normal value in (0, 1]), the latter as the unsigned
 * bits - 0x00800000 <= 0x3f000000 (vpmaxud). Every other word (-0, a denormal, a
 * value above 1, a negative value, inf, NaN) gives zero lanes. */
static __m256i normalized_lanes(__m256i bits)
{
    const __m256i span = _mm256_set1_epi32(0x3f000000);
    __m256i shifted = _mm256_sub_epi32(bits, _mm256_set1_epi32(0x00800000));
    __m256i normal = _mm256_cmpeq_epi32(_mm256_max_epu32(shifted, span), span);
    return _mm256_or_si256(normal, _mm256_cmpeq_epi32(bits, _mm256_setzero_si256()));
}

/* Groups of 32 words, each tested once (vptest), so a refusal stops at the first
 * group holding a failing word; then groups of 8, and the last 1-7 words through a
 * masked load, whose lanes past count read 0, which passes. */
int cn_pixels_normalized_avx2(const float *src, size_t count)
{
    const __m256i every = _mm256_set1_epi32(-1);
    const __m256i *words = (const __m256i *)src;
    size_t i = 0;
    for (; count - i >= 32; i += 32, words += 4) {
        __m256i lanes = normalized_lanes(_mm256_loadu_si256(words));
        lanes = _mm256_and_si256(lanes, normalized_lanes(_mm256_loadu_si256(words + 1)));
        lanes = _mm256_and_si256(lanes, normalized_lanes(_mm256_loadu_si256(words + 2)));
        lanes = _mm256_and_si256(lanes, normalized_lanes(_mm256_loadu_si256(words + 3)));
        if (!_mm256_testc_si256(lanes, every)) return 0;
    }
    for (; count - i >= 8; i += 8, ++words)
        if (!_mm256_testc_si256(normalized_lanes(_mm256_loadu_si256(words)), every))
            return 0;
    if (i < count) {
        __m256i inside = _mm256_cmpgt_epi32(_mm256_set1_epi32((int)(count - i)),
                                            _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
        __m256i tail = _mm256_maskload_epi32((const int *)(src + i), inside);
        if (!_mm256_testc_si256(normalized_lanes(tail), every)) return 0;
    }
    return 1;
}

long cn_convert_avx2(const convert_job *j, size_t begin, size_t end)
{
    size_t first = (begin + 7) & ~(size_t)7, last;
    long events;
    if (first > end) first = end;
    last = first + (end - first) / 8 * 8;
    events = convert_scalar(j, begin, first);
    if (j->type == 2)
        u8_f32((const uint8_t *)j->src, (float *)j->out, first, last);
    else if (j->output == 0)
        f32_f32((const float *)j->src, (float *)j->out, j->normalize, first, last);
    else
        events |= f32_u8((const float *)j->src, (uint8_t *)j->out, j->normalize, first, last);
    return events | convert_scalar(j, last, end);
}
