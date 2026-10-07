/* Stable median-cut buckets, preserving NumPy's split and reduction order. */
#include "parallel.h"
#include "numeric.h"
#include "isa.h"
#include "palette_shared.h"
#include "scratch.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct { size_t start, count, channel; float range; } bucket;

/* Channel c's extrema widen the bucket as np.argmax does, in channel order: the first
   widest channel, and the first NaN before every finite one (then 1: stop). low and
   high reach nothing but range = high - low and its compares (SP4b D12 as built): the
   AVX2 extrema may return another zero-class value than B3's first-seen one, which
   leaves the stored range's bits unchanged (a zero-class pair's +-0 range is never
   stored; palette_shared.h). A change that lets their bits reach
   anything else must first restore the first-seen rule at every level. */
static int widen(bucket *result, size_t c, float low, float high) {
    float range = high - low;
    if (range > result->range || palette_nan(range)) {
        result->range = range; result->channel = c;
        return palette_nan(range);
    }
    return 0;
}

/* B3's describe: channel_extrema per channel, stopping at the first NaN range; at ISA
   level avx2 (channels 1-4) every channel's extrema from the AVX2 unit (the same
   ranges: palette_shared.h), combined in the same order. The extrema go only to widen
   (see there). isa: the level the C entry read once for the call. */
static bucket describe(const float *data, size_t start, size_t count, size_t channels,
                       cn_isa_level isa) {
    bucket result = {start, count, 0, 0};
#if defined(_MSC_VER) && defined(_M_X64)
    if (isa >= CN_ISA_AVX2 && channels <= 4) {
        float low[4], high[4];
        cn_palette_extrema_avx2(data, start, count, channels, low, high);
        for (size_t c = 0; c < channels; ++c)
            if (widen(&result, c, low[c], high[c])) break;
        return result;
    }
#else
    (void)isa;
#endif
    for (size_t c = 0; c < channels; ++c) {
        float low, high;
        channel_extrema(data, start, count, channels, c, &low, &high);
        if (widen(&result, c, low, high)) break;
    }
    return result;
}

/* The selected channel of a bucket's pixels copied into values (the median's
   selection and counts read the copy), and whether it holds a NaN. */
static int gather_channel(float *values, const float *source, size_t count, size_t channels,
                          size_t channel, cn_isa_level isa) {
    source += channel;
#if defined(_MSC_VER) && defined(_M_X64)
    if (isa >= CN_ISA_AVX2) {
        for (size_t p = 0; p < count; ++p) values[p] = source[p * channels];
        return cn_palette_any_nan_avx2(values, count);
    }
#else
    (void)isa;
#endif
    int has_nan = 0;
    for (size_t p = 0; p < count; ++p) {
        values[p] = source[p * channels];
        has_nan |= palette_nan(values[p]);
    }
    return has_nan;
}

/* How many of the bucket's selected-channel values compare above median. The copy in
   values holds the same floats as the bucket, and a count does not depend on their
   order. */
static size_t count_above(const float *values, size_t count, float median, cn_isa_level isa) {
#if defined(_MSC_VER) && defined(_M_X64)
    if (isa >= CN_ISA_AVX2) return cn_palette_count_above_avx2(values, count, median);
#else
    (void)isa;
#endif
    size_t above = 0;
    for (size_t p = 0; p < count; ++p) above += values[p] > median;
    return above;
}

/* The median's selection (SP4b Task 5b): an exact radix select over order-preserving
   keys. A non-NaN float's key is bits ^ (bits >> 31 ? 0xFFFFFFFF : 0x80000000), Rust's
   f32::total_cmp order: unsigned key order is -inf < ... < -0 < +0 < ... < +inf, and
   equal keys are equal bits. Key order refines the compare order in both MXCSR states:
   by default it orders only -0 before +0, which compare equal; under DAZ the denormals
   and +-0, which all compare equal, are one interval of keys (0x7F800000-0x807FFFFF).
   So the value of rank r in key order compares equal to the value of rank r in compare
   order, which B3's quickselect (rank middle) and lower-half max (rank middle - 1)
   returned. The median reaches only > compares and the midpoint, and compare-equal
   inputs give a compare-equal midpoint (equal up to +-0, or as DAZ reads them), so the
   counts, splits and outputs are B3's. The selection reads values and its narrowed copy
   only, runs no float operation and does not depend on the MXCSR state. The keys
   (median_key, palette_shared.h) have three digits, high to low: bits 31-21 and 20-10
   (MEDIAN_BINS values each), then bits 9-0. */
#define MEDIAN_LOW_BINS 1024
/* Values per counting chunk: one way counts at most a quarter of them plus 7, so no
   uint32 counter wraps; the chunk's counts are then added into bins. */
#define MEDIAN_CHUNK ((size_t)UINT32_MAX)

static float median_value(uint32_t key) {
    uint32_t bits = key >> 31 ? key ^ UINT32_C(0x80000000) : ~key;
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

typedef struct {
    uint32_t ways[MEDIAN_WAYS][MEDIAN_BINS];
    size_t bins[MEDIAN_BINS];
} median_counts;

/* The ways (value i - begin into way (i - begin) % 4) count the digits
   (median_key >> shift) & mask of values[begin, end); at ISA level avx2 the AVX2 unit's
   (the same counts). */
static void median_count_range(const float *values, size_t begin, size_t end, unsigned shift,
                               uint32_t mask, uint32_t (*ways)[MEDIAN_BINS], cn_isa_level isa) {
#if defined(_MSC_VER) && defined(_M_X64)
    if (isa >= CN_ISA_AVX2) {
        cn_palette_median_count_avx2(values, begin, end, shift, mask, ways);
        return;
    }
#else
    (void)isa;
#endif
    size_t i = begin;
    for (; end - i >= MEDIAN_WAYS; i += MEDIAN_WAYS)
        for (size_t w = 0; w < MEDIAN_WAYS; ++w)
            ++ways[w][median_key(values[i + w]) >> shift & mask];
    for (; i < end; ++i) ++ways[0][median_key(values[i]) >> shift & mask];
}

/* counts->bins[d]: how many values[0, count) have the digit d = (median_key >> shift) &
   mask, counted in chunks. */
static void median_count(const float *values, size_t count, unsigned shift, uint32_t mask,
                         median_counts *counts, cn_isa_level isa) {
    const size_t bins = (size_t)mask + 1;
    memset(counts->bins, 0, bins * sizeof(size_t));
    for (size_t begin = 0; begin < count;) {
        size_t end = count - begin > MEDIAN_CHUNK ? begin + MEDIAN_CHUNK : count;
        for (size_t w = 0; w < MEDIAN_WAYS; ++w) memset(counts->ways[w], 0, bins * sizeof(uint32_t));
        median_count_range(values, begin, end, shift, mask, counts->ways, isa);
        for (size_t d = 0; d < bins; ++d)
            counts->bins[d] += (size_t)counts->ways[0][d] + counts->ways[1][d] +
                               counts->ways[2][d] + counts->ways[3][d];
        begin = end;
    }
}

/* Copies the values[0, count) whose median_key >> shift is prefix, in order, to
   narrowed (room for count values; narrowed may be values itself: value i is read
   before slot kept <= i is written) and returns how many. The store is unconditional,
   the advance is not. At ISA level avx2 the AVX2 unit's (the same copy). */
static size_t median_narrow(const float *values, size_t count, unsigned shift, uint32_t prefix,
                            float *narrowed, cn_isa_level isa) {
#if defined(_MSC_VER) && defined(_M_X64)
    if (isa >= CN_ISA_AVX2) return cn_palette_median_narrow_avx2(values, count, shift, prefix, narrowed);
#else
    (void)isa;
#endif
    size_t kept = 0;
    for (size_t i = 0; i < count; ++i) {
        float value = values[i];
        narrowed[kept] = value;
        kept += median_key(value) >> shift == prefix;
    }
    return kept;
}

/* The bins holding ranks rank[0] <= rank[1] of counts->bins; each rank is then made
   relative to the first value of its bin. */
static void median_digits(const median_counts *counts, size_t rank[2], uint32_t digit[2]) {
    size_t below = 0, d = 0;
    for (size_t t = 0; t < 2; ++t) {
        while (below + counts->bins[d] <= rank[t]) below += counts->bins[d++];
        digit[t] = (uint32_t)d;
        rank[t] -= below;
    }
}

/* Of the keys whose bits from shift up are low_prefix, the largest; of those whose bits
   are high_prefix, the smallest (each set nonempty). */
static void median_bounds(const float *values, size_t count, unsigned shift,
                          uint32_t low_prefix, uint32_t high_prefix,
                          uint32_t *low_key, uint32_t *high_key) {
    uint32_t low = low_prefix << shift;
    uint32_t high = high_prefix << shift | ((UINT32_C(1) << shift) - 1);
    for (size_t i = 0; i < count; ++i) {
        uint32_t key = median_key(values[i]), top = key >> shift;
        uint32_t in_low = UINT32_C(0) - (top == low_prefix);
        uint32_t in_high = UINT32_C(0) - (top == high_prefix);
        uint32_t candidate = key & in_low;
        low = candidate > low ? candidate : low;
        candidate = key | ~in_high;
        high = candidate < high ? candidate : high;
    }
    *low_key = low;
    *high_key = high;
}

/* The values of ranks first and first + pair (pair 0 or 1) of values[0, count) in key
   order (an MSD radix select). Level 0 counts every top digit; while both ranks fall in
   one bin, that bin's values are copied to narrowed (room for count values) and level
   1 counts their middle digits; while both ranks again fall in one bin, narrowed keeps
   that bin's values and level 2 counts their low digits. Ranks in two bins are the
   last key of the first bin and the first key of the next nonempty one: above level 2,
   one more pass finds them (median_bounds). */
static void median_select(const float *values, size_t count, size_t first, size_t pair,
                          float *narrowed, median_counts *counts, cn_isa_level isa,
                          float *lower, float *upper) {
    size_t rank[2] = {first, first + pair};
    uint32_t digit[2], low, high;
    median_count(values, count, 21, MEDIAN_BINS - 1, counts, isa);
    median_digits(counts, rank, digit);
    if (digit[0] != digit[1]) {
        median_bounds(values, count, 21, digit[0], digit[1], &low, &high);
    } else {
        uint32_t prefix = digit[0];
        size_t kept = median_narrow(values, count, 21, prefix, narrowed, isa);
        median_count(narrowed, kept, 10, MEDIAN_BINS - 1, counts, isa);
        median_digits(counts, rank, digit);
        prefix <<= 11;
        if (digit[0] != digit[1]) {
            median_bounds(narrowed, kept, 10, prefix | digit[0], prefix | digit[1], &low, &high);
        } else {
            prefix |= digit[0];
            kept = median_narrow(narrowed, kept, 10, prefix, narrowed, isa);
            median_count(narrowed, kept, 0, MEDIAN_LOW_BINS - 1, counts, isa);
            median_digits(counts, rank, digit);
            low = prefix << 10 | digit[0];
            high = prefix << 10 | digit[1];
        }
    }
    *lower = median_value(low);
    *upper = median_value(high);
}

/* The stable split: pixel p goes to the next high slot (from 0) when its selected value
   compares above median, else to the next low slot (from high_count). The slot is
   chosen without a branch; pixels are fixed-size copies for 1, 3 and 4 channels.
   high_count must be exactly the number of pixels this predicate sends high: too high
   and the low slots run past the span; too low and high and low slots collide, leaving
   slots of the span unwritten for the copy-back to read as stale pooled bytes. At ISA
   level avx2 it comes from another unit
   (cn_palette_count_above_avx2, whose prototype states the coupling). A per-pixel
   bound on the low slot cost 9-12 % of the split (SP4b Task 5b). */
static size_t next_slot(size_t above, size_t *high, size_t *low) {
    size_t mask = (size_t)0 - above;
    size_t slot = (*high & mask) | (*low & ~mask);
    *high += above;
    *low += 1 - above;
    return slot;
}

static void split_pixels(float *scratch, const float *source, size_t count, size_t channels,
                         size_t channel, float median, size_t high_count) {
    size_t high = 0, low = high_count;
    switch (channels) {
    case 1:
        for (size_t p = 0; p < count; ++p)
            scratch[next_slot(source[p] > median, &high, &low)] = source[p];
        break;
    case 3:
        for (size_t p = 0; p < count; ++p) {
            const float *pixel = source + p * 3;
            size_t slot = next_slot(pixel[channel] > median, &high, &low);
            memcpy(scratch + slot * 3, pixel, 3 * sizeof(float));
        }
        break;
    case 4:
        for (size_t p = 0; p < count; ++p) {
            const float *pixel = source + p * 4;
            size_t slot = next_slot(pixel[channel] > median, &high, &low);
            memcpy(scratch + slot * 4, pixel, 4 * sizeof(float));
        }
        break;
    default:
        for (size_t p = 0; p < count; ++p) {
            const float *pixel = source + p * channels;
            size_t slot = next_slot(pixel[channel] > median, &high, &low);
            memcpy(scratch + slot * channels, pixel, channels * sizeof(float));
        }
    }
}

typedef struct {
    const float *data;
    const bucket *buckets;
    float *out;
    size_t channels, block, period;
} average_job;

/* Axis zero is contiguous only for one-channel buckets (pairwise sums, in blocks of
   job->block and job->period: the root's until a split, then the children's). Other channels use
   ordered strided sums in NumPy's reduction loop: each channel's sum starts at 0 and
   adds its values in pixel order; for 3 and 4 channels the sums run side by side in one
   pass, each in that order. */
static void averages(void *opaque, size_t begin, size_t end) {
    average_job *j = opaque;
    const size_t channels = j->channels;
    for (size_t i = begin; i < end; ++i) {
        const bucket *b = j->buckets + i;
        const float *source = j->data + b->start * channels;
        float *out = j->out + i * channels;
        float sums[4] = {0, 0, 0, 0};
        if (channels == 1) {
            out[0] = numpy_mean_f32(cn_numpy_sum_f32(source, b->count, 1, j->block, j->period), b->count);
            continue;
        }
        if (channels == 3) {
            for (size_t p = 0; p < b->count; ++p) {
                sums[0] += source[p * 3];
                sums[1] += source[p * 3 + 1];
                sums[2] += source[p * 3 + 2];
            }
        } else if (channels == 4) {
            for (size_t p = 0; p < b->count; ++p) {
                sums[0] += source[p * 4];
                sums[1] += source[p * 4 + 1];
                sums[2] += source[p * 4 + 2];
                sums[3] += source[p * 4 + 3];
            }
        } else {
            for (size_t c = 0; c < channels; ++c) {
                float sum = 0;
                for (size_t p = 0; p < b->count; ++p) sum += source[p * channels + c];
                out[c] = numpy_mean_f32(sum, b->count);
            }
            continue;
        }
        for (size_t c = 0; c < channels; ++c) out[c] = numpy_mean_f32(sums[c], b->count);
    }
}

/* The pairwise sum blocks of np.mean (cn_numpy_sum_f32): root_block and root_period for
   the root bucket, the caller's array (numpy_sum_block of its column on NumPy 2.5.3), and
   child_block (period 0) for the fresh aligned arrays each split makes (0, one tree, on NumPy
   2.5.3; the B3 manifests pin NumPy 1.24's 8192 for both). */
CN_EXPORT cn_status cn_palette_median_cut(const float *src, float *out,
                                         size_t pixels, size_t channels, size_t capacity,
                                         size_t root_block, size_t root_period, size_t child_block,
                                         size_t *written) {
    if (!src || !out || !written || !pixels || !channels || !capacity)
        return CN_INVALID_ARGUMENT;
    if (channels > SIZE_MAX / sizeof(float) ||
        pixels > SIZE_MAX / sizeof(float) / channels ||
        capacity > SIZE_MAX / sizeof(bucket) ||
        capacity > SIZE_MAX / sizeof(float) / channels) return CN_SIZE_OVERFLOW;
    /* A split produces nonempty children, so at most pixels colors exist. */
    size_t limit = capacity < pixels ? capacity : pixels;
    size_t count = pixels * channels;
    /* data, scratch and the selection's narrowed copy and counts from one pooled lease
       (SP4b D10): data is written whole (src) before any read; each split writes the
       bucket's span of scratch (every slot, as its count of values above the median is
       the split's own predicate) before the copy back reads it. The selected channel's
       copy lives at the start of scratch: it is read (selection, counts) before the
       split overwrites it. median_narrow writes the narrowed values it keeps before
       they are read, and median_count zeroes the counts it reads. */
    size_t total = 0, data_at = 0, scratch_at = 0, narrowed_at = 0, counts_at = 0;
    if (!cn_scratch_carve(&total, count * sizeof(float), &data_at) ||
        !cn_scratch_carve(&total, count * sizeof(float), &scratch_at) ||
        !cn_scratch_carve(&total, pixels * sizeof(float), &narrowed_at) ||
        !cn_scratch_carve(&total, sizeof(median_counts), &counts_at))
        return CN_ALLOCATION_FAILED;
    bucket *buckets = malloc(limit * sizeof(bucket));
    if (!buckets) return CN_ALLOCATION_FAILED;
    cn_scratch lease;
    if (cn_scratch_lease(total, &lease, NULL) != CN_OK) {
        free(buckets);
        return CN_ALLOCATION_FAILED;
    }
    float *data = (float *)((char *)lease.data + data_at);
    float *scratch = (float *)((char *)lease.data + scratch_at);
    float *values = scratch;
    float *narrowed = (float *)((char *)lease.data + narrowed_at);
    median_counts *counts = (median_counts *)((char *)lease.data + counts_at);
    memcpy(data, src, count * sizeof(float));
    cn_isa_level isa = cn_isa_current();
    buckets[0] = describe(data, 0, pixels, channels, isa);
    size_t used = 1, block = root_block, period = root_period;
    cn_status status = CN_OK;
    while (used < capacity) {
        size_t selected = 0;
        for (size_t i = 1; i < used; ++i)
            if (buckets[i].range > buckets[selected].range) selected = i;
        bucket b = buckets[selected];
        if (b.range == 0) break;
        const float *source = data + b.start * channels;
        int has_nan = gather_channel(values, source, b.count, channels, b.channel, isa);
        size_t middle = b.count / 2;
        float median;
        if (has_nan) median = cn_f32_from_bits(CN_NPY_NANF_BITS); /* compared only */
        else {
            /* upper: rank middle; lower: rank middle - 1 for an even count. */
            size_t pair = 1 - b.count % 2;
            float lower, upper;
            median_select(values, b.count, middle - pair, pair, narrowed, counts, isa, &lower,
                          &upper);
            if (b.count % 2) median = upper;
            else median = (lower + upper) / 2.0f;
        }
        size_t high_count = count_above(values, b.count, median, isa);
        if (!high_count) {
            /* np.mean on the strided selected channel still uses pairwise sums. */
            median = numpy_mean_f32(cn_numpy_sum_f32(source + b.channel, b.count, channels, block, period), b.count);
            high_count = count_above(values, b.count, median, isa);
        }
        if (!high_count || high_count == b.count) {
            /* Original construction of an empty child fails at np.min. */
            status = 6;
            break;
        }
        split_pixels(scratch, source, b.count, channels, b.channel, median, high_count);
        memcpy(data + b.start * channels, scratch, b.count * channels * sizeof(float));
        memmove(buckets + selected, buckets + selected + 1, (used - selected - 1) * sizeof(bucket));
        --used;
        buckets[used++] = describe(data, b.start, high_count, channels, isa);
        buckets[used++] = describe(data, b.start + high_count, b.count - high_count, channels, isa);
        block = child_block;
        period = 0;
    }
    if (status == CN_OK) {
        average_job job = {data, buckets, out, channels, block, period};
        status = cn_parallel_for(used, 32, averages, &job);
        *written = used;
    }
    cn_scratch_release(&lease);
    free(buckets);
    return status;
}

typedef struct {
    const float *source;
    float *out;
    size_t colors, channels;
} repeat_palette_job;

static void repeat_palette(void *opaque, size_t begin, size_t end) {
    repeat_palette_job *job = opaque;
    for (size_t i = begin; i < end; ++i) {
        size_t source = i < job->colors ? i : job->colors - 1;
        memcpy(job->out + i * job->channels, job->source + source * job->channels,
               job->channels * sizeof(float));
    }
}

CN_EXPORT cn_status cn_palette_pad_edge(const float *source, float *out,
                                        size_t colors, size_t channels, size_t capacity) {
    if (!source || !out || !colors || !channels || capacity < colors)
        return CN_INVALID_ARGUMENT;
    if (channels > SIZE_MAX / sizeof(float) || capacity > SIZE_MAX / sizeof(float) / channels)
        return CN_SIZE_OVERFLOW;
    repeat_palette_job job = {source, out, colors, channels};
    return cn_parallel_for(capacity, 4096, repeat_palette, &job);
}
