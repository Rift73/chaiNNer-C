#if !defined(_M_X64) && !(defined(__linux__) && defined(__x86_64__))
#error "png_unfilter_avx2.c is an x86-64 AVX2 unit"
#endif
/* Undoing PNG's Paeth filter, AVX2 unit: no export, reached only through png_codec.c's dispatch on the decode's ISA
 * level. A row's channels are independent, so each pixel's run together, 128 bits wide whatever bpp; the left
 * neighbour is carried in a register, and it is what limits the row: every byte waits for the predictor of the pixel
 * before it. Every value is an exact integer, so the bytes are the scalar code's. */
#include <immintrin.h>
#include <string.h>

#include "chainner.h"

CN_FORCE_INLINE __m128i load_pixel(const uint8_t *p, size_t bpp)
{
    if (bpp == 1) return _mm_cvtsi32_si128(*p);
    if (bpp == 2) {
        uint16_t pixel;
        memcpy(&pixel, p, 2);
        return _mm_cvtsi32_si128(pixel);
    }
    if (bpp <= 4) {
        int pixel;
        memcpy(&pixel, p, 4);
        return _mm_cvtsi32_si128(pixel);
    }
    return _mm_loadl_epi64((const __m128i *)p);
}

CN_FORCE_INLINE void store_pixel(uint8_t *p, __m128i x, size_t bpp)
{
    const uint64_t pixel = (uint64_t)_mm_cvtsi128_si64(x);
    memcpy(p, &pixel, bpp);
}

/* The Paeth predictor (PNG specification 9.4) of a, b and c as thresholds on a that b and c alone give, so that only
 * the comparisons, the blends between their results and the row's addition wait for a. With d = |b - c|:
 *   b >= c: a if a < c + 1 - 2d or a > b - 1; else c if a < c - d/2; else b.
 *   b <  c: a if a < b + 1 or a > c + 2d - 1; else c if a > c + d/2; else b.
 * The bounds saturate to bytes, which keeps these true for every a, b and c. a and the result are offset by 0x80, for
 * the signed comparisons; b1 and c1 are b + 1 and c + 1, saturating. */
CN_FORCE_INLINE __m128i paeth(__m128i a, __m128i b, __m128i c, __m128i b1, __m128i c1)
{
    const __m128i offset = _mm_set1_epi8((char)0x80), one = _mm_set1_epi8(1);
    const __m128i b_offset = _mm_xor_si128(b, offset), c_offset = _mm_xor_si128(c, offset);
    const __m128i b_below_c = _mm_cmpgt_epi8(c_offset, b_offset);
    const __m128i d = _mm_or_si128(_mm_subs_epu8(b, c), _mm_subs_epu8(c, b));
    const __m128i twice = _mm_adds_epu8(d, d), half = _mm_and_si128(_mm_srli_epi16(d, 1), _mm_set1_epi8(0x7F));
    const __m128i low = _mm_blendv_epi8(_mm_subs_epu8(c1, twice), b1, b_below_c);
    const __m128i high = _mm_blendv_epi8(_mm_subs_epu8(b, one), _mm_adds_epu8(c, _mm_subs_epu8(twice, one)), b_below_c);
    const __m128i keep_a =
        _mm_or_si128(_mm_cmpgt_epi8(_mm_xor_si128(low, offset), a), _mm_cmpgt_epi8(a, _mm_xor_si128(high, offset)));
    const __m128i take_c = _mm_blendv_epi8(_mm_cmpgt_epi8(_mm_xor_si128(_mm_subs_epu8(c, half), offset), a),
                                           _mm_cmpgt_epi8(a, _mm_xor_si128(_mm_adds_epu8(c, half), offset)), b_below_c);
    return _mm_blendv_epi8(_mm_blendv_epi8(b_offset, c_offset, take_c), a, keep_a);
}

/* A load of bpp 3 or 6 reads past its pixel; it stops at to, where the row above may still be changing. */
CN_FORCE_INLINE size_t unpaeth(uint8_t *row, const uint8_t *prior, size_t from, size_t to, size_t bpp)
{
    const __m128i offset = _mm_set1_epi8((char)0x80), one = _mm_set1_epi8(1);
    const size_t reach = bpp == 3 ? 4 : bpp == 6 ? 8 : bpp;
    size_t i = from;
    if (i + reach > to) return i;
    __m128i a = _mm_xor_si128(load_pixel(row + i - bpp, bpp), offset), c = load_pixel(prior + i - bpp, bpp);
    __m128i c1 = _mm_adds_epu8(c, one);
    for (; i + reach <= to; i += bpp) {
        const __m128i b = load_pixel(prior + i, bpp), b1 = _mm_adds_epu8(b, one);
        a = _mm_add_epi8(load_pixel(row + i, bpp), paeth(a, b, c, b1, c1));
        store_pixel(row + i, _mm_xor_si128(a, offset), bpp);
        c = b;
        c1 = b1;
    }
    return i;
}

size_t cn_png_unpaeth_avx2(uint8_t *row, const uint8_t *prior, size_t from, size_t to, size_t bpp);

size_t cn_png_unpaeth_avx2(uint8_t *row, const uint8_t *prior, size_t from, size_t to, size_t bpp)
{
    switch (bpp) {
    case 1: return unpaeth(row, prior, from, to, 1);
    case 2: return unpaeth(row, prior, from, to, 2);
    case 3: return unpaeth(row, prior, from, to, 3);
    case 4: return unpaeth(row, prior, from, to, 4);
    case 6: return unpaeth(row, prior, from, to, 6);
    default: return unpaeth(row, prior, from, to, 8);
    }
}
