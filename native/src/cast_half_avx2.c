#if !defined(_M_X64) && !(defined(__linux__) && defined(__x86_64__))
#error "cast_half_avx2.c is an x86-64 /arch:AVX2 unit"
#endif
/* float32 -> float16, AVX2 unit (/arch:AVX2, spec 4.3): no export, reached only
 * through cn_tensor_cast_typed's dispatch on the call's ISA level. Integer lanes
 * round the float's 23-bit significand to 10 bits, half to even, and rebias the
 * exponent: tensor_complete_ops.cpp's half_bits for the values it takes. No F16C: it
 * quiets a signalling NaN, which the cast keeps, and is not in the level's sets. */
#include "cast_half.h"
#include <immintrin.h>
#include <stdint.h>
#include <string.h>

size_t cn_float_to_half_avx2(const char *src, ptrdiff_t from, char *out, ptrdiff_t to,
                             size_t n)
{
    /* A gather's byte offsets 0, from, ..., 7 * from are int32 lanes. */
    if (from != 4 && (from > INT32_MAX / 7 || from < -(INT32_MAX / 7))) return 0;
    const __m256i offsets = _mm256_mullo_epi32(_mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                                               _mm256_set1_epi32((int)from));
    const __m256i magnitude = _mm256_set1_epi32(0x7fffffff);
    /* 2^-14, the least normal half, and 65520, the least value rounding to Inf */
    const __m256i below_normal = _mm256_set1_epi32(0x387fffff);
    const __m256i overflowing = _mm256_set1_epi32(0x477ff000);
    const __m256i rebias = _mm256_set1_epi32(112 << 10);
    const __m256i round = _mm256_set1_epi32(0xfff);
    const __m256i one = _mm256_set1_epi32(1);
    const __m256i sign_bit = _mm256_set1_epi32(0x8000);
    size_t done = 0;
    for (; n - done >= 8; done += 8, src += 8 * from, out += 8 * to) {
        __m256i bits = from == 4 ? _mm256_loadu_si256((const __m256i *)src)
                                 : _mm256_i32gather_epi32((const int *)src, offsets, 1);
        __m256i abs = _mm256_and_si256(bits, magnitude);
        __m256i zero = _mm256_cmpeq_epi32(abs, _mm256_setzero_si256());
        __m256i normal = _mm256_and_si256(_mm256_cmpgt_epi32(abs, below_normal),
                                          _mm256_cmpgt_epi32(overflowing, abs));
        if (_mm256_movemask_epi8(_mm256_or_si256(zero, normal)) != -1) break;
        /* Half to even at bit 13: add 0xfff and the kept significand's lowest bit. */
        __m256i odd = _mm256_and_si256(_mm256_srli_epi32(abs, 13), one);
        __m256i rounded = _mm256_srli_epi32(
            _mm256_add_epi32(abs, _mm256_add_epi32(round, odd)), 13);
        __m256i half = _mm256_andnot_si256(zero, _mm256_sub_epi32(rounded, rebias));
        half = _mm256_or_si256(half, _mm256_and_si256(_mm256_srli_epi32(bits, 16), sign_bit));
        /* Lanes 0-7 as 16-bit words in the low 128 bits */
        __m256i packed = _mm256_permute4x64_epi64(_mm256_packus_epi32(half, half), 0xd8);
        if (to == 2) {
            _mm_storeu_si128((__m128i *)out, _mm256_castsi256_si128(packed));
        } else {
            uint16_t words[8];
            _mm_storeu_si128((__m128i *)words, _mm256_castsi256_si128(packed));
            for (size_t k = 0; k < 8; ++k) memcpy(out + (ptrdiff_t)k * to, &words[k], 2);
        }
    }
    return done;
}
