#if !defined(_M_X64) && !(defined(__linux__) && defined(__x86_64__))
#error "neighborhood_avx2.c is an x86-64 /arch:AVX2 unit"
#endif
/* Median Blur's 3x3 and 5x5 interiors, for images without NaN and -0, and its histogram rows, AVX2 unit (/arch:AVX2,
 * spec 4.3): no export, reached only through small_median_range's and histogram_range's dispatch on the call's ISA level
 * (median_shared.h). */
#include "median_shared.h"

#include <immintrin.h>
#include <string.h>

/* Groups of eight outputs of row y in [first, last), each lane's window of side x side samples loaded from rows
 * (row starts, clamped) and sorted by the pairs; the last group overlaps back. */
static inline void median_groups(const float *const *rows, float *out, size_t c, size_t radius, size_t first,
                                 size_t last, const uint8_t (*pairs)[2], size_t length, size_t side)
{
    for (size_t at = first; at < last; at += 8) {
        if (last - at < 8) at = last - 8;
        __m256 v[25];
        size_t n = 0;
        for (size_t dy = 0; dy < side; ++dy)
            for (size_t dx = 0; dx < side; ++dx) v[n++] = _mm256_loadu_ps(rows[dy] + at - radius * c + dx * c);
        for (size_t i = 0; i < length; ++i) {
            const __m256 a = v[pairs[i][0]], b = v[pairs[i][1]];
            v[pairs[i][0]] = _mm256_min_ps(a, b);
            v[pairs[i][1]] = _mm256_max_ps(a, b);
        }
        _mm256_storeu_ps(out + at, v[side * side / 2]);
    }
}

void cn_median_interior_avx2(const float *src, float *out, size_t h, size_t w, size_t c, size_t radius, size_t y,
                             size_t first, size_t last)
{
    const size_t row = w * c;
    const float *rows[5];
    for (size_t dy = 0; dy <= 2 * radius; ++dy)
        rows[dy] = src + (y + dy < radius ? 0 : y + dy - radius >= h ? h - 1 : y + dy - radius) * row;
    if (radius == 1)
        median_groups(rows, out + y * row, c, 1, first, last, cn_median_square3,
                      sizeof cn_median_square3 / sizeof cn_median_square3[0], 3);
    else
        median_groups(rows, out + y * row, c, 2, first, last, cn_median_square5,
                      sizeof cn_median_square5 / sizeof cn_median_square5[0], 5);
}

int cn_median_special_avx2(const float *src, size_t count)
{
    const __m256i magnitude = _mm256_set1_epi32(0x7fffffff), infinity = _mm256_set1_epi32(0x7f800000);
    const __m256i negative_zero = _mm256_set1_epi32((int)0x80000000u);
    size_t i = 0;
    for (; count - i >= 8; i += 8) {
        const __m256i words = _mm256_loadu_si256((const __m256i *)(src + i));
        const __m256i special = _mm256_or_si256(
            _mm256_cmpgt_epi32(_mm256_and_si256(words, magnitude), infinity), _mm256_cmpeq_epi32(words, negative_zero));
        if (!_mm256_testz_si256(special, special)) return 1;
    }
    for (; i < count; ++i) {
        uint32_t word;
        memcpy(&word, src + i, sizeof word);
        if ((word & 0x7fffffffu) > 0x7f800000u || word == 0x80000000u) return 1;
    }
    return 0;
}

/* The running sums of 16 counts: each 128-bit lane's own, then the low lane's total on every sum of the high one. */
static inline __m256i count_sums(__m256i counts)
{
    counts = _mm256_add_epi16(counts, _mm256_slli_si256(counts, 2));
    counts = _mm256_add_epi16(counts, _mm256_slli_si256(counts, 4));
    counts = _mm256_add_epi16(counts, _mm256_slli_si256(counts, 8));
    const __m256i low = _mm256_shufflehi_epi16(_mm256_permute2x128_si256(counts, counts, 0x08), 0xFF);
    return _mm256_add_epi16(counts, _mm256_unpackhi_epi64(low, low));
}

/* The first of 16 running sums above limit; the last one is. */
static inline size_t first_above(__m256i sums, uint32_t limit)
{
    const __m256i over = _mm256_subs_epu16(sums, _mm256_set1_epi16((short)limit));
    const uint32_t within = (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi16(over, _mm256_setzero_si256()));
    return _tzcnt_u32(~within) / 2;
}

void cn_median_row_avx2(const uint16_t *coarse, const uint16_t *fine, size_t w, size_t radius, uint8_t *out,
                        size_t step)
{
    const size_t r = radius, kernel = 2 * r + 1;
    const uint32_t middle = (uint32_t)(kernel * kernel / 2);
    __m256i window = _mm256_setzero_si256(), buckets[16];
    size_t updated[16];
    for (size_t b = 0; b < 16; ++b) updated[b] = SIZE_MAX;
    for (size_t x = 0; x <= r && x < w; ++x) {
        size_t weight = x == 0 ? r + 1 : 1;
        if (x == w - 1 && r >= w) weight += r - w + 1;
        const __m256i counts = _mm256_loadu_si256((const __m256i *)(coarse + x * 16));
        window = _mm256_add_epi16(window, _mm256_mullo_epi16(counts, _mm256_set1_epi16((short)weight)));
    }
    for (size_t x = 0; x < w; ++x) {
        const __m256i sums = count_sums(window);
        const size_t bucket = first_above(sums, middle);
        uint16_t totals[16];
        _mm256_storeu_si256((__m256i *)totals, sums);
        const uint32_t below = bucket ? totals[bucket - 1] : 0;
        const uint16_t *bins = fine + bucket * 16;
        __m256i counts;
        if (updated[bucket] == SIZE_MAX || x - updated[bucket] >= kernel) {
            const size_t left = x < r ? 0 : x - r, right = x + r < w ? x + r : w - 1;
            counts = _mm256_setzero_si256();
            for (size_t xx = left; xx <= right; ++xx) {
                size_t weight = 1;
                if (!xx && x < r) weight += r - x;
                if (xx == w - 1 && r >= w - x) weight += r - (w - x - 1);
                counts = _mm256_add_epi16(counts,
                    _mm256_mullo_epi16(_mm256_loadu_si256((const __m256i *)(bins + xx * 256)),
                                       _mm256_set1_epi16((short)weight)));
            }
        } else {
            counts = buckets[bucket];
            for (size_t xx = updated[bucket]; xx < x; ++xx) {
                const size_t leave = xx < r ? 0 : xx - r, enter = xx + r + 1 < w ? xx + r + 1 : w - 1;
                counts = _mm256_sub_epi16(counts, _mm256_loadu_si256((const __m256i *)(bins + leave * 256)));
                counts = _mm256_add_epi16(counts, _mm256_loadu_si256((const __m256i *)(bins + enter * 256)));
            }
        }
        buckets[bucket] = counts;
        updated[bucket] = x;
        out[x * step] = (uint8_t)(bucket * 16 + first_above(count_sums(counts), middle - below));
        const size_t leave = x < r ? 0 : x - r, enter = x + r + 1 < w ? x + r + 1 : w - 1;
        window = _mm256_sub_epi16(window, _mm256_loadu_si256((const __m256i *)(coarse + leave * 16)));
        window = _mm256_add_epi16(window, _mm256_loadu_si256((const __m256i *)(coarse + enter * 16)));
    }
}
