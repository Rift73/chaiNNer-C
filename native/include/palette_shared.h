/* Median cut's per-channel extrema (palette_ops.c's describe) and its median's radix
 * select (SP4b Task 5b). Shared by that baseline unit and the /arch:AVX2 unit
 * palette_avx2.c: static helpers compiled into each including unit, never inline, used
 * by both (C4505). */
#ifndef CHAINNER_PALETTE_SHARED_H
#define CHAINNER_PALETTE_SHARED_H
#include "chainner.h"
#include <string.h>

/* isnan as a bit test: an exponent of all ones and a nonzero fraction, whatever the
 * MXCSR state (B3 called the CRT's _fdclass). */
static int palette_nan(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT32_C(0x7fffffff)) > UINT32_C(0x7f800000);
}

/* Channel c of data's pixels [start, start + count), count >= 1: low and high start at
 * the first value; a later value replaces low when it compares below it or is a NaN,
 * and high when above it or a NaN. That is B3's describe loop verbatim, with the bit
 * test for isnan: a strict compare (honouring DAZ) keeps the first of equal values, so
 * a zero-class extremum keeps the first one seen, and a NaN the last. */
static void channel_extrema(const float *data, size_t start, size_t count, size_t channels,
                            size_t c, float *low, float *high)
{
    const float *value = data + start * channels + c;
    float lo = *value, hi = lo;
    for (size_t p = 1; p < count; ++p) {
        value += channels;
        float v = *value;
        if (v < lo || palette_nan(v)) lo = v;
        if (v > hi || palette_nan(v)) hi = v;
    }
    *low = lo;
    *high = hi;
}

/* A non-NaN float's order-preserving key, bits ^ (bits >> 31 ? 0xFFFFFFFF : 0x80000000)
 * written without a branch (Rust's f32::total_cmp order): unsigned key order is
 * -inf < ... < -0 < +0 < ... < +inf, and equal keys are equal bits. */
static uint32_t median_key(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits ^ ((UINT32_C(0) - (bits >> 31)) | UINT32_C(0x80000000));
}

/* The radix select's counting arrays: MEDIAN_WAYS arrays of MEDIAN_BINS uint32 counts
 * (an 11-bit digit), so neighbouring values count into separate arrays and a run of
 * equal digits (a smooth image's) is not one chain of increments of one counter. */
#define MEDIAN_BINS 2048
#define MEDIAN_WAYS 4

/* palette_avx2.c, for ISA level avx2 or above. */
/* channels 1-4: every channel's low and high, a vector pass folding with strict
 * compares; a channel holding a NaN is redone by channel_extrema. Without a NaN these
 * are channel_extrema's values: equal compares mean equal bits except within the zero
 * class (+-0, and under DAZ the denormals), where the representative may differ (SP4b
 * D12 as built). The extrema reach nothing but range = high - low, whose bits that
 * choice leaves unchanged, except that two zero-class extrema give +0 or -0, a range
 * that never compares above the +0 a bucket starts from, so it is never stored; the
 * stored range is only ever compared. */
void cn_palette_extrema_avx2(const float *data, size_t start, size_t count, size_t channels,
                             float *low, float *high);
/* The values[0, count) that compare above median (_CMP_GT_OQ, as the baseline's >).
 * palette_ops.c's split_pixels starts its low slots at this count and does not bound
 * them: the predicate here must stay exactly the split's (a > compare honouring DAZ,
 * false for NaN), or the split writes past the bucket's span (count too high) or leaves
 * slots of it unwritten, which the copy-back then reads as stale pooled bytes (count too
 * low). SP4b Task 5b kept this coupling unbounded: the bound cost 9-12 % of the split. */
size_t cn_palette_count_above_avx2(const float *values, size_t count, float median);
/* 1 when values[0, count) holds a NaN (an unordered compare per lane), else 0. */
int cn_palette_any_nan_avx2(const float *values, size_t count);
/* The radix select's passes (palette_ops.c's median_count_range and median_narrow, the
 * same counts and copies): the ways count the digits (median_key >> shift) & mask of
 * values[begin, end), at most a quarter of them plus 7 per way; and the values[0, count)
 * whose median_key >> shift is prefix are copied in order to narrowed (room for count
 * values; narrowed may be values itself), returning how many. */
void cn_palette_median_count_avx2(const float *values, size_t begin, size_t end, unsigned shift,
                                  uint32_t mask, uint32_t (*ways)[MEDIAN_BINS]);
size_t cn_palette_median_narrow_avx2(const float *values, size_t count, unsigned shift,
                                     uint32_t prefix, float *narrowed);
#endif
