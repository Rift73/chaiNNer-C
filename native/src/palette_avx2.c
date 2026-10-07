#if !defined(_M_X64)
#error "palette_avx2.c is an x86-64 /arch:AVX2 unit"
#endif
/* Median cut's AVX2 unit (/arch:AVX2, spec 4.3; SP4b D12, Task 5b): no export, reached
 * only through cn_palette_median_cut's dispatch on the ISA level it read once for the
 * call: describe's extrema, the NaN scan of the selected channel's copy, the median's
 * radix select passes and the count above the median. The resolution and the tails run
 * the shared scalar helpers, here with VEX encodings. */
#include "palette_shared.h"
#include <immintrin.h>

/* A block is three vectors, 24 floats, a multiple of every channel count 1-4: lane l of
 * the block's vector k holds channel (8k + l) % channels in every block. */
#define EXTREMA_BLOCK 24
/* Groups of 8 counted in int32 lanes before the lanes are added up (no lane wraps). */
#define COUNT_FLUSH ((size_t)1 << 24)
/* For each 4-bit mask of kept lanes, a vpermilps control moving those lanes, in order,
 * to the front (the other lanes don't matter); and the masks' bit counts, 4 bits each. */
static const int PACK_LANES[16][4] = {
    {0, 0, 0, 0}, {0, 0, 0, 0}, {1, 0, 0, 0}, {0, 1, 0, 0},
    {2, 0, 0, 0}, {0, 2, 0, 0}, {1, 2, 0, 0}, {0, 1, 2, 0},
    {3, 0, 0, 0}, {0, 3, 0, 0}, {1, 3, 0, 0}, {0, 1, 3, 0},
    {2, 3, 0, 0}, {0, 2, 3, 0}, {1, 2, 3, 0}, {0, 1, 2, 3},
};
#define PACK_COUNTS UINT64_C(0x4332322132212110)

/* A candidate folds into a channel's running pair with strict compares (the first of
 * equal values stays), and marks the channel when it is a NaN. */
static void fold_candidate(float value, float *low, float *high, int *nan)
{
    if (value < *low) *low = value;
    if (value > *high) *high = value;
    *nan |= palette_nan(value);
}

/* One vector into its lanes' running low and high (vcmpps _CMP_LT_OQ / _CMP_GT_OQ and
 * vblendvps: original bits, the first of equal values stays) and NaN marks. */
static void fold_vector(__m256 v, __m256 *low, __m256 *high, __m256 *nan)
{
    *low = _mm256_blendv_ps(*low, v, _mm256_cmp_ps(v, *low, _CMP_LT_OQ));
    *high = _mm256_blendv_ps(*high, v, _mm256_cmp_ps(v, *high, _CMP_GT_OQ));
    *nan = _mm256_or_ps(*nan, _mm256_cmp_ps(v, v, _CMP_UNORD_Q));
}

/* Vector k's lanes into their channels' pairs (fold_candidate) and NaN marks. */
static void fold_lanes(__m256 low, __m256 high, __m256 nan, size_t k, size_t channels,
                       float *lo, float *hi, int *nans)
{
    float lanes_low[8], lanes_high[8];
    _mm256_storeu_ps(lanes_low, low);
    _mm256_storeu_ps(lanes_high, high);
    int marks = _mm256_movemask_ps(nan);
    for (size_t l = 0; l < 8; ++l) {
        size_t c = (8 * k + l) % channels;
        fold_candidate(lanes_low[l], lo + c, hi + c, nans + c);
        fold_candidate(lanes_high[l], lo + c, hi + c, nans + c);
        nans[c] |= marks >> l & 1;
    }
}

/* palette_shared.h. Each lane of the three block vectors keeps its own low, high and NaN
 * mark (fold_vector); the lanes, then the floats after the last block, fold into each
 * channel's pair with strict compares. A channel holding a NaN is redone by
 * channel_extrema (its last NaN). */
void cn_palette_extrema_avx2(const float *data, size_t start, size_t count, size_t channels,
                             float *low, float *high)
{
    const float *pixels = data + start * channels;
    const size_t floats = count * channels;
    float lo[4], hi[4];
    int nans[4];
    for (size_t c = 0; c < channels; ++c) {
        lo[c] = hi[c] = pixels[c];
        nans[c] = palette_nan(pixels[c]);
    }
    size_t i = channels;
    if (floats >= EXTREMA_BLOCK) {
        __m256 low0 = _mm256_loadu_ps(pixels), low1 = _mm256_loadu_ps(pixels + 8);
        __m256 low2 = _mm256_loadu_ps(pixels + 16);
        __m256 high0 = low0, high1 = low1, high2 = low2;
        __m256 nan0 = _mm256_cmp_ps(low0, low0, _CMP_UNORD_Q);
        __m256 nan1 = _mm256_cmp_ps(low1, low1, _CMP_UNORD_Q);
        __m256 nan2 = _mm256_cmp_ps(low2, low2, _CMP_UNORD_Q);
        for (i = EXTREMA_BLOCK; floats - i >= EXTREMA_BLOCK; i += EXTREMA_BLOCK) {
            fold_vector(_mm256_loadu_ps(pixels + i), &low0, &high0, &nan0);
            fold_vector(_mm256_loadu_ps(pixels + i + 8), &low1, &high1, &nan1);
            fold_vector(_mm256_loadu_ps(pixels + i + 16), &low2, &high2, &nan2);
        }
        fold_lanes(low0, high0, nan0, 0, channels, lo, hi, nans);
        fold_lanes(low1, high1, nan1, 1, channels, lo, hi, nans);
        fold_lanes(low2, high2, nan2, 2, channels, lo, hi, nans);
    }
    for (; i < floats; ++i) {
        size_t c = i % channels;
        fold_candidate(pixels[i], lo + c, hi + c, nans + c);
    }
    for (size_t c = 0; c < channels; ++c) {
        if (nans[c]) channel_extrema(data, start, count, channels, c, lo + c, hi + c);
        low[c] = lo[c];
        high[c] = hi[c];
    }
}

/* palette_shared.h: per lane, the compare mask (-1) subtracted from an int32 count,
 * flushed into the total every COUNT_FLUSH groups; the last 1-7 values compare one by
 * one. */
size_t cn_palette_count_above_avx2(const float *values, size_t count, float median)
{
    const __m256 threshold = _mm256_set1_ps(median);
    size_t above = 0, i = 0;
    while (count - i >= 8) {
        size_t groups = (count - i) / 8;
        if (groups > COUNT_FLUSH) groups = COUNT_FLUSH;
        __m256i lanes = _mm256_setzero_si256();
        for (size_t g = 0; g < groups; ++g, i += 8) {
            __m256 above_lanes = _mm256_cmp_ps(_mm256_loadu_ps(values + i), threshold, _CMP_GT_OQ);
            lanes = _mm256_sub_epi32(lanes, _mm256_castps_si256(above_lanes));
        }
        uint32_t counts[8];
        _mm256_storeu_si256((__m256i *)counts, lanes);
        for (size_t l = 0; l < 8; ++l) above += counts[l];
    }
    for (; i < count; ++i) above += values[i] > median;
    return above;
}

/* palette_shared.h: an unordered compare per lane (a NaN whatever the MXCSR state), the
 * marks OR-ed and tested once; the last 1-7 values through palette_nan. */
int cn_palette_any_nan_avx2(const float *values, size_t count)
{
    __m256 marks = _mm256_setzero_ps();
    size_t i = 0;
    for (; count - i >= 8; i += 8) {
        __m256 v = _mm256_loadu_ps(values + i);
        marks = _mm256_or_ps(marks, _mm256_cmp_ps(v, v, _CMP_UNORD_Q));
    }
    if (_mm256_movemask_ps(marks)) return 1;
    for (; i < count; ++i)
        if (palette_nan(values[i])) return 1;
    return 0;
}

/* Eight keys (median_key per lane): the bits xor-ed with their sign spread over every
 * bit, or with 0x80000000 alone for a clear sign. */
static __m256i median_keys(__m256i bits)
{
    const __m256i sign = _mm256_set1_epi32(INT32_MIN);
    return _mm256_xor_si256(bits, _mm256_or_si256(_mm256_srai_epi32(bits, 31), sign));
}

/* palette_shared.h: per group of 8 the digits are computed in lanes and stored, then
 * counted one by one, lanes l and l + 4 into way l; the last 1-7 values into way 0. */
void cn_palette_median_count_avx2(const float *values, size_t begin, size_t end, unsigned shift,
                                  uint32_t mask, uint32_t (*ways)[MEDIAN_BINS])
{
    const __m128i bits_shift = _mm_cvtsi32_si128((int)shift);
    const __m256i digits = _mm256_set1_epi32((int)mask);
    uint32_t digit[8];
    size_t i = begin;
    for (; end - i >= 8; i += 8) {
        __m256i keys = median_keys(_mm256_loadu_si256((const __m256i *)(values + i)));
        _mm256_storeu_si256((__m256i *)digit,
                            _mm256_and_si256(_mm256_srl_epi32(keys, bits_shift), digits));
        ++ways[0][digit[0]];
        ++ways[1][digit[1]];
        ++ways[2][digit[2]];
        ++ways[3][digit[3]];
        ++ways[0][digit[4]];
        ++ways[1][digit[5]];
        ++ways[2][digit[6]];
        ++ways[3][digit[7]];
    }
    for (; i < end; ++i) ++ways[0][median_key(values[i]) >> shift & mask];
}

/* palette_shared.h: per group of 8 the kept lanes are a compare mask; each half moves
 * its kept lanes to the front (vpermilps, PACK_LANES) and is stored whole at narrowed
 * + kept, which then advances by their count: a store ends at most at the group's last
 * index, whose values are loaded already. The last 1-7 values one by one. */
size_t cn_palette_median_narrow_avx2(const float *values, size_t count, unsigned shift,
                                     uint32_t prefix, float *narrowed)
{
    const __m128i bits_shift = _mm_cvtsi32_si128((int)shift);
    const __m256i target = _mm256_set1_epi32((int)prefix);
    size_t i = 0, kept = 0;
    for (; count - i >= 8; i += 8) {
        __m256 group = _mm256_loadu_ps(values + i);
        __m256i keys = median_keys(_mm256_castps_si256(group));
        __m256i in = _mm256_cmpeq_epi32(_mm256_srl_epi32(keys, bits_shift), target);
        unsigned lanes = (unsigned)_mm256_movemask_ps(_mm256_castsi256_ps(in));
        unsigned low = lanes & 15, high = lanes >> 4;
        __m128i order = _mm_loadu_si128((const __m128i *)PACK_LANES[low]);
        _mm_storeu_ps(narrowed + kept, _mm_permutevar_ps(_mm256_castps256_ps128(group), order));
        kept += (size_t)(PACK_COUNTS >> 4 * low & 15);
        order = _mm_loadu_si128((const __m128i *)PACK_LANES[high]);
        _mm_storeu_ps(narrowed + kept, _mm_permutevar_ps(_mm256_extractf128_ps(group, 1), order));
        kept += (size_t)(PACK_COUNTS >> 4 * high & 15);
    }
    for (; i < count; ++i) {
        float value = values[i];
        narrowed[kept] = value;
        kept += median_key(value) >> shift == prefix;
    }
    return kept;
}
