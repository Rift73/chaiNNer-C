#if !defined(_M_X64)
#error "png_encode_avx2.c is an x86-64 AVX2 unit"
#endif
/* PNG row filters, AVX2 unit (spec 4.3): no export, reached only through png_encode.c's dispatch on the call's ISA
 * level. 32 bytes at a time from byte bpp on, the row's first bpp bytes and its tail by the scalar rules
 * (png_encode_shared.h); every filtered byte and cost is an exact integer, so the choice and the bytes are the
 * scalar unit's. */
#include "png_encode_shared.h"
#include <immintrin.h>

static __m256i load(const uint8_t *p)
{
    return _mm256_loadu_si256((const __m256i *)p);
}

static __m256i absolute_difference(__m256i x, __m256i y)
{
    return _mm256_or_si256(_mm256_subs_epu8(x, y), _mm256_subs_epu8(y, x));
}

/* 0xFF where x <= y (unsigned). */
static __m256i less_equal(__m256i x, __m256i y)
{
    return _mm256_cmpeq_epi8(_mm256_min_epu8(x, y), x);
}

/* (a + b) >> 1 without overflow: the rounded-up average less the dropped half. */
static __m256i average(__m256i a, __m256i b)
{
    return _mm256_sub_epi8(_mm256_avg_epu8(a, b), _mm256_and_si256(_mm256_xor_si256(a, b), _mm256_set1_epi8(1)));
}

/* png_paeth in bytes: pa = |b - c| and pb = |a - c| fit in a byte; pc = |(a - c) + (b - c)| is pa + pb when a - c
 * and b - c have the same sign (either may be 0), else |pa - pb|. The sum saturates at 255, which keeps pa <= pc and
 * pb <= pc as they are, since pa and pb are at most 255. */
static __m256i paeth(__m256i a, __m256i b, __m256i c)
{
    const __m256i zero = _mm256_setzero_si256();
    const __m256i cb = _mm256_subs_epu8(c, b), ca = _mm256_subs_epu8(c, a);
    const __m256i pa = _mm256_or_si256(_mm256_subs_epu8(b, c), cb);
    const __m256i pb = _mm256_or_si256(_mm256_subs_epu8(a, c), ca);
    const __m256i same = _mm256_cmpeq_epi8(_mm256_cmpeq_epi8(ca, zero), _mm256_cmpeq_epi8(cb, zero));
    const __m256i pc = _mm256_blendv_epi8(absolute_difference(pa, pb), _mm256_adds_epu8(pa, pb), same);
    const __m256i use_a = _mm256_and_si256(less_equal(pa, pb), less_equal(pa, pc));
    return _mm256_blendv_epi8(_mm256_blendv_epi8(c, b, less_equal(pb, pc)), a, use_a);
}

/* Four 64-bit partial sums of the 32 filtered bytes' costs (png_cost: the signed byte's magnitude). */
static __m256i costs(__m256i filtered)
{
    return _mm256_sad_epu8(_mm256_abs_epi8(filtered), _mm256_setzero_si256());
}

static uint64_t total(__m256i sums)
{
    const __m128i half = _mm_add_epi64(_mm256_castsi256_si128(sums), _mm256_extracti128_si256(sums, 1));
    return (uint64_t)_mm_cvtsi128_si64(half) + (uint64_t)_mm_extract_epi64(half, 1);
}

void cn_png_filter_row_avx2(const uint8_t *row, const uint8_t *up, size_t length, size_t bpp, uint8_t *out)
{
    const size_t head = length < bpp ? length : bpp;
    uint64_t sums[5] = {0, 0, 0, 0, 0};
    png_costs(row, up, 0, head, bpp, sums);
    __m256i s0 = _mm256_setzero_si256(), s1 = s0, s2 = s0, s3 = s0, s4 = s0;
    size_t i = head;
    for (; i + 32 <= length; i += 32) {
        const __m256i x = load(row + i), a = load(row + i - bpp), b = load(up + i), c = load(up + i - bpp);
        s0 = _mm256_add_epi64(s0, costs(x));
        s1 = _mm256_add_epi64(s1, costs(_mm256_sub_epi8(x, a)));
        s2 = _mm256_add_epi64(s2, costs(_mm256_sub_epi8(x, b)));
        s3 = _mm256_add_epi64(s3, costs(_mm256_sub_epi8(x, average(a, b))));
        s4 = _mm256_add_epi64(s4, costs(_mm256_sub_epi8(x, paeth(a, b, c))));
    }
    sums[0] += total(s0);
    sums[1] += total(s1);
    sums[2] += total(s2);
    sums[3] += total(s3);
    sums[4] += total(s4);
    png_costs(row, up, i, length, bpp, sums);
    const int type = png_choice(sums);
    uint8_t *filtered = out + 1;
    out[0] = (uint8_t)type;
    if (type == 0) {
        memcpy(filtered, row, length);
        return;
    }
    png_filter_bytes(type, row, up, 0, head, bpp, filtered);
    for (i = head; i + 32 <= length; i += 32) {
        const __m256i x = load(row + i), b = load(up + i);
        __m256i prediction;
        switch (type) {
        case 1: prediction = load(row + i - bpp); break;
        case 2: prediction = b; break;
        case 3: prediction = average(load(row + i - bpp), b); break;
        default: prediction = paeth(load(row + i - bpp), b, load(up + i - bpp)); break;
        }
        _mm256_storeu_si256((__m256i *)(filtered + i), _mm256_sub_epi8(x, prediction));
    }
    png_filter_bytes(type, row, up, i, length, bpp, filtered);
}

void cn_png_swap_row_avx2(const uint8_t *source, size_t width, size_t channels, uint8_t *out)
{
    const size_t length = width * channels;
    size_t i = 0;
    if (channels == 4) {
        const __m256i order = _mm256_setr_epi8(2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15, 2, 1, 0, 3, 6, 5,
                                               4, 7, 10, 9, 8, 11, 14, 13, 12, 15);
        for (; i + 32 <= length; i += 32)
            _mm256_storeu_si256((__m256i *)(out + i), _mm256_shuffle_epi8(load(source + i), order));
    } else {
        /* Four pixels per step: 16 bytes in, the first 12 swapped and the last 4 (the next pixels' bytes, unswapped)
         * written over by the next step or the scalar tail. */
        const __m128i order = _mm_setr_epi8(2, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9, 12, 13, 14, 15);
        for (; i + 16 <= length; i += 12)
            _mm_storeu_si128((__m128i *)(out + i),
                             _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(source + i)), order));
    }
    png_swap_pixels(source, i / channels, width, channels, out);
}
