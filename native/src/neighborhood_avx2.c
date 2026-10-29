#if !defined(_M_X64) && !(defined(__linux__) && defined(__x86_64__))
#error "neighborhood_avx2.c is an x86-64 /arch:AVX2 unit"
#endif
/* Median Blur's 3x3 and 5x5 interiors, AVX2 unit (/arch:AVX2, spec 4.3): no export, reached only through
 * small_median_range's dispatch on the call's ISA level, for images without NaN and -0 (median_shared.h). */
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
