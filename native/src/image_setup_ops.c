/* Image preparation predicates. No image arithmetic or global image storage. */
#include "chainner.h"
#include "exceptional.h"
#include "exceptional_shared.h"
#include "isa.h"
#include <fenv.h>
#include <stdlib.h>
#include <string.h>
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#endif
/* SSE2 is part of the x86-64 baseline (no /arch is needed and no level is read). */
#if defined(_M_X64) || defined(__x86_64__)
#include <emmintrin.h>
#define CN_SETUP_SSE2 1
#endif

CN_EXPORT uint64_t cn_image_fp_state(void) {
    uint64_t result = (uint64_t)(unsigned int)fegetround() << 32;
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
    result |= _mm_getcsr() & 0xe040u;
#endif
    return result;
}

#ifdef CN_SETUP_SSE2
/* exceptional_bits on 4 words as SSE2 compares on the words, the x86 baseline's form of
 * the AVX2 unit's lanes: the exponent all ones (pand, pcmpeqd), or, as signed int32,
 * below bound (pcmpgtd): with bound 0x80000001 that is -0 alone, with 0x80800000 (DAZ)
 * -0 and the negative denormals. */
static __m128i exceptional_words(__m128i words, __m128i bound) {
    const __m128i exponent = _mm_set1_epi32(0x7f800000);
    __m128i special = _mm_cmpeq_epi32(_mm_and_si128(words, exponent), exponent);
    return _mm_or_si128(special, _mm_cmpgt_epi32(bound, words));
}
#endif

/* exceptional.h: the caller's DAZ read once; at ISA level avx2 or above the AVX2
 * unit's scan. Otherwise, on x86, groups of 16 words through exceptional_words, tested
 * once (pmovmskb), so the scan returns at the first group holding a match (B3's loop
 * tested every word with three branches); then the remaining words, and every word
 * elsewhere, through exceptional_bits in order. */
int cn_exceptional_scan(const float *src, size_t count, cn_isa_level isa) {
    int daz = (cn_image_fp_state() & 0x40u) != 0;
    size_t i = 0;
#if defined(_MSC_VER) && defined(_M_X64)
    if (isa >= CN_ISA_AVX2) return cn_exceptional_scan_avx2(src, count, daz);
#else
    (void)isa;
#endif
#ifdef CN_SETUP_SSE2
    {
        const __m128i bound = _mm_set1_epi32(daz ? (int)0x80800000u : (int)0x80000001u);
        const __m128i *words = (const __m128i *)src;
        for (; count - i >= 16; i += 16, words += 4) {
            __m128i lanes = exceptional_words(_mm_loadu_si128(words), bound);
            lanes = _mm_or_si128(lanes, exceptional_words(_mm_loadu_si128(words + 1), bound));
            lanes = _mm_or_si128(lanes, exceptional_words(_mm_loadu_si128(words + 2), bound));
            lanes = _mm_or_si128(lanes, exceptional_words(_mm_loadu_si128(words + 3), bound));
            if (_mm_movemask_epi8(lanes)) return 1;
        }
    }
#endif
    for (; i < count; ++i) {
        uint32_t bits; memcpy(&bits, src + i, sizeof(bits));
        if (exceptional_bits(bits, daz)) return 1;
    }
    return 0;
}

CN_EXPORT cn_status cn_image_exceptional(const float *src, size_t count, int *result) {
    if (!src || !result || (uintptr_t)src % _Alignof(float)) return CN_INVALID_ARGUMENT;
    if (count > SIZE_MAX / sizeof(float) ||
        (uintptr_t)src > UINTPTR_MAX - count * sizeof(float)) return CN_SIZE_OVERFLOW;
    *result = cn_exceptional_scan(src, count, cn_isa_current());
    return CN_OK;
}

/* At most limit+1 distinct rows are needed to choose the extraction path.
 * Hash equality deliberately agrees with the existing sorted distinct pass:
 * signed zeros compare equal, while every NaN-containing row is distinct. */
CN_EXPORT cn_status cn_image_distinct_exceeds(const float *src, size_t pixels,
        size_t channels, size_t limit, int *result) {
    if (!src || !result || !pixels || !channels || limit > 4096 ||
        (uintptr_t)src % _Alignof(float)) return CN_INVALID_ARGUMENT;
    if (channels > SIZE_MAX / sizeof(float) ||
        pixels > SIZE_MAX / sizeof(float) / channels ||
        (uintptr_t)src > UINTPTR_MAX - pixels * channels * sizeof(float))
        return CN_SIZE_OVERFLOW;
    *result = 0;
    if (pixels <= limit) return CN_OK;
    size_t capacity = 2;
    while (capacity < (limit + 1) * 2) capacity *= 2;
    size_t *table = calloc(capacity, sizeof(size_t));
    if (!table) return CN_ALLOCATION_FAILED;
    int daz = (cn_image_fp_state() & 0x40u) != 0;
    size_t unique = 0;
    for (size_t p = 0; p < pixels; ++p) {
        uint64_t hash = UINT64_C(14695981039346656037);
        int nan = 0;
        for (size_t c = 0; c < channels; ++c) {
            uint32_t bits; memcpy(&bits, src + p * channels + c, sizeof(bits));
            if ((bits & UINT32_C(0x7fffffff)) == 0 ||
                (daz && (bits & UINT32_C(0x7fffffff)) < UINT32_C(0x00800000))) bits = 0;
            if ((bits & UINT32_C(0x7fffffff)) > UINT32_C(0x7f800000)) nan = 1;
            hash = (hash ^ bits) * UINT64_C(1099511628211);
        }
        size_t slot = (size_t)hash & (capacity - 1);
        if (!nan) {
            while (table[slot]) {
                size_t old = table[slot] - 1, c = 0;
                while (c < channels && src[old * channels + c] == src[p * channels + c]) ++c;
                if (c == channels) break;
                slot = (slot + 1) & (capacity - 1);
            }
            if (table[slot]) continue;
            table[slot] = p + 1;
        }
        if (++unique > limit) { *result = 1; break; }
    }
    free(table);
    return CN_OK;
}
