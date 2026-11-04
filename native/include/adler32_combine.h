/* The stream's Adler-32 from the Adler-32 values of its pieces, for PNG's bands: png_encode.c writes it, the decoder
 * (codecs/png_codec.c, codecs/png_inflate.c) checks it. */
#ifndef CHAINNER_ADLER32_COMBINE_H
#define CHAINNER_ADLER32_COMBINE_H
#include <stdint.h>

/* The Adler-32 of A followed by B from adler(A), adler(B) and B's length (RFC 1950's sums s1 and s2 mod 65521):
 * s1(AB) = s1(A) + s1(B) - 1 and s2(AB) = s2(A) + s2(B) + length(B) * (s1(A) - 1). */
static inline uint32_t adler32_combine(uint32_t a, uint32_t b, uint64_t b_length)
{
    const uint64_t base = 65521, a1 = a & 0xFFFF, b1 = b & 0xFFFF;
    const uint64_t s1 = (a1 + b1 + base - 1) % base;
    const uint64_t s2 = ((a >> 16) + (b >> 16) + (b_length % base) * ((a1 + base - 1) % base)) % base;
    return (uint32_t)(s2 << 16 | s1);
}

#endif
