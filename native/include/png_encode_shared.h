/* The default PNG encoder's row filters (cn_png_encode_u8, png_encode.c): the scalar rules, compiled into each
 * including unit (png_encode.c, the baseline entry, and the AVX2 unit png_encode_avx2.c, which runs them for a row's
 * first bpp bytes and its tail), and the AVX2 unit's declarations. No AVX-512 unit: it filtered 1.4-1.7x faster than
 * AVX2 but left the encode's time unchanged (the filters take about 1 % of it at AVX2; measured 2026-10-21).
 *
 * A row is filtered as libpng 1.6's png_write_find_filter does with every filter enabled and no weighting (MinSum):
 * each of the five filters (PNG specification 9.2: None, Sub, Up, Average, Paeth) gives a filtered row, whose cost is
 * the sum over its bytes of the byte read as a signed value and made positive (0x80 costs 128); the cheapest wins, a
 * tie going to the lower filter type. Before byte bpp a byte has no left neighbours (a = c = 0); the row above the
 * first is all zeros. Every value is an exact integer, so the units give the same choice and the same bytes. */
#ifndef CHAINNER_PNG_ENCODE_SHARED_H
#define CHAINNER_PNG_ENCODE_SHARED_H
#include "chainner.h"
#include <stdlib.h>
#include <string.h>

/* Paeth's predictor (PNG specification 9.4): of a (left), b (up) and c (up-left), the one nearest to a + b - c, ties
 * in the order a, b, c. */
CN_FORCE_INLINE int png_paeth(int a, int b, int c)
{
    const int pa = abs(b - c), pb = abs(a - c), pc = abs(a + b - 2 * c);
    return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
}

/* A filtered byte's cost. */
CN_FORCE_INLINE uint32_t png_cost(uint8_t v)
{
    return v < 128 ? v : 256u - v;
}

/* Byte x under filter type 0-4, with a = left, b = up, c = up-left. */
CN_FORCE_INLINE uint8_t png_filtered(int type, int x, int a, int b, int c)
{
    switch (type) {
    case 1: return (uint8_t)(x - a);
    case 2: return (uint8_t)(x - b);
    case 3: return (uint8_t)(x - ((a + b) >> 1));
    case 4: return (uint8_t)(x - png_paeth(a, b, c));
    default: return (uint8_t)x;
    }
}

/* Adds to sums[f] the costs of the row's bytes [begin, end) under filter f. */
static inline void png_costs(const uint8_t *row, const uint8_t *up, size_t begin, size_t end, size_t bpp,
                             uint64_t sums[5])
{
    uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0;
    size_t i = begin;
    for (; i < end && i < bpp; ++i) {
        const int x = row[i], b = up[i];
        s0 += png_cost((uint8_t)x);
        s1 += png_cost((uint8_t)x);
        s2 += png_cost((uint8_t)(x - b));
        s3 += png_cost((uint8_t)(x - (b >> 1)));
        s4 += png_cost((uint8_t)(x - b)); /* Paeth's predictor of (0, b, 0) is b */
    }
    for (; i < end; ++i) {
        const int x = row[i], a = row[i - bpp], b = up[i], c = up[i - bpp];
        s0 += png_cost((uint8_t)x);
        s1 += png_cost((uint8_t)(x - a));
        s2 += png_cost((uint8_t)(x - b));
        s3 += png_cost((uint8_t)(x - ((a + b) >> 1)));
        s4 += png_cost((uint8_t)(x - png_paeth(a, b, c)));
    }
    sums[0] += s0;
    sums[1] += s1;
    sums[2] += s2;
    sums[3] += s3;
    sums[4] += s4;
}

/* The cheapest filter type of sums, ties to the lower type. */
CN_FORCE_INLINE int png_choice(const uint64_t sums[5])
{
    int best = 0;
    for (int f = 1; f < 5; ++f)
        if (sums[f] < sums[best]) best = f;
    return best;
}

/* Writes the row's bytes [begin, end) under filter type `type` to out[begin, end). */
static inline void png_filter_bytes(int type, const uint8_t *row, const uint8_t *up, size_t begin, size_t end,
                                    size_t bpp, uint8_t *out)
{
    size_t i = begin;
    if (type == 0) {
        if (end > begin) memcpy(out + begin, row + begin, end - begin);
        return;
    }
    for (; i < end && i < bpp; ++i) out[i] = png_filtered(type, row[i], 0, up[i], 0);
    switch (type) {
    case 1:
        for (; i < end; ++i) out[i] = (uint8_t)(row[i] - row[i - bpp]);
        break;
    case 2:
        for (; i < end; ++i) out[i] = (uint8_t)(row[i] - up[i]);
        break;
    case 3:
        for (; i < end; ++i) out[i] = (uint8_t)(row[i] - ((row[i - bpp] + up[i]) >> 1));
        break;
    default:
        for (; i < end; ++i) out[i] = (uint8_t)(row[i] - png_paeth(row[i - bpp], up[i], up[i - bpp]));
        break;
    }
}

/* Pixels [begin, end) of a packed BGR or BGRA row (channels 3 or 4) as RGB or RGBA. */
static inline void png_swap_pixels(const uint8_t *source, size_t begin, size_t end, size_t channels, uint8_t *out)
{
    for (size_t x = begin; x < end; ++x) {
        const uint8_t *s = source + x * channels;
        uint8_t *o = out + x * channels;
        o[0] = s[2];
        o[1] = s[1];
        o[2] = s[0];
        if (channels == 4) o[3] = s[3];
    }
}

#if CN_ISA_UNITS()
/* png_encode_avx2.c: one row of `length` bytes filtered as above (out[0] the filter type, out[1..length] the
 * filtered bytes; up is the row above, length bytes), and png_swap_pixels of a whole row of `width` pixels. */
void cn_png_filter_row_avx2(const uint8_t *row, const uint8_t *up, size_t length, size_t bpp, uint8_t *out);
void cn_png_swap_row_avx2(const uint8_t *source, size_t width, size_t channels, uint8_t *out);
#endif

#endif
