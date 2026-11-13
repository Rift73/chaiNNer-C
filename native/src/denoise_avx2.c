#if !defined(_M_X64) && !(defined(__linux__) && defined(__x86_64__))
#error "denoise_avx2.c is an x86-64 /arch:AVX2 unit"
#endif
/* Non-local means rows, AVX2 unit (/arch:AVX2): no export, reached only through nlm's dispatch on the call's
 * ISA level, for one and two channels. denoise_shared.h has the rounding rule. An independent implementation of the
 * integer contract in denoise_ops.cpp, no OpenCV source code incorporated. */
#include "denoise_shared.h"
#include <immintrin.h>

/* Two 16-byte runs' differences as 16-bit lanes: |d| <= 255. */
CN_FORCE_INLINE __m256i differences(const uint8_t *a, ptrdiff_t displacement)
{
    return _mm256_sub_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)a)),
        _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(a + displacement))));
}

static void nlm_column_add(uint32_t *columns, const uint8_t *row, ptrdiff_t displacement, size_t count, size_t channels)
{
    if (channels == 1) {
        for (size_t i = 0; i < count; i += 16) {
            __m256i d = differences(row + i, displacement), squares = _mm256_mullo_epi16(d, d);
            __m256i *low = (__m256i *)(columns + i), *high = (__m256i *)(columns + i + 8);
            _mm256_storeu_si256(low, _mm256_add_epi32(_mm256_loadu_si256(low),
                _mm256_cvtepu16_epi32(_mm256_castsi256_si128(squares))));
            _mm256_storeu_si256(high, _mm256_add_epi32(_mm256_loadu_si256(high),
                _mm256_cvtepu16_epi32(_mm256_extracti128_si256(squares, 1))));
        }
    } else {
        for (size_t i = 0; i < count; i += 8) {
            __m256i d = differences(row + i * 2, displacement);
            __m256i *lanes = (__m256i *)(columns + i);
            _mm256_storeu_si256(lanes, _mm256_add_epi32(_mm256_loadu_si256(lanes), _mm256_madd_epi16(d, d)));
        }
    }
}

static void nlm_column_slide(uint32_t *columns, const uint8_t *down, const uint8_t *up, ptrdiff_t displacement,
    size_t count, size_t channels)
{
    if (channels == 1) {
        for (size_t i = 0; i < count; i += 16) {
            __m256i in = differences(down + i, displacement), out = differences(up + i, displacement);
            __m256i added = _mm256_mullo_epi16(in, in), removed = _mm256_mullo_epi16(out, out);
            __m256i *low = (__m256i *)(columns + i), *high = (__m256i *)(columns + i + 8);
            _mm256_storeu_si256(low, _mm256_sub_epi32(_mm256_add_epi32(_mm256_loadu_si256(low),
                _mm256_cvtepu16_epi32(_mm256_castsi256_si128(added))),
                _mm256_cvtepu16_epi32(_mm256_castsi256_si128(removed))));
            _mm256_storeu_si256(high, _mm256_sub_epi32(_mm256_add_epi32(_mm256_loadu_si256(high),
                _mm256_cvtepu16_epi32(_mm256_extracti128_si256(added, 1))),
                _mm256_cvtepu16_epi32(_mm256_extracti128_si256(removed, 1))));
        }
    } else {
        for (size_t i = 0; i < count; i += 8) {
            __m256i in = differences(down + i * 2, displacement), out = differences(up + i * 2, displacement);
            __m256i *lanes = (__m256i *)(columns + i);
            _mm256_storeu_si256(lanes, _mm256_sub_epi32(_mm256_add_epi32(_mm256_loadu_si256(lanes),
                _mm256_madd_epi16(in, in)), _mm256_madd_epi16(out, out)));
        }
    }
}

static void nlm_window_pass(uint32_t *out, const uint32_t *in, const uint32_t *columns, size_t count, size_t span, int odd)
{
    for (size_t i = 0; i < count; i += 8) {
        __m256i sum = _mm256_add_epi32(_mm256_loadu_si256((const __m256i *)(in + i)),
            _mm256_loadu_si256((const __m256i *)(in + i + span)));
        if (odd) sum = _mm256_add_epi32(sum, _mm256_loadu_si256((const __m256i *)(columns + i + 2 * span)));
        _mm256_storeu_si256((__m256i *)(out + i), sum);
    }
}

CN_FORCE_INLINE void add_weighted(const nlm_block *b, __m256i weight, uint32_t *sums, const uint8_t *pixels)
{
    size_t plane = b->rows * b->row_lanes;
    uint32_t *first = sums + plane;
    _mm256_storeu_si256((__m256i *)sums, _mm256_add_epi32(_mm256_loadu_si256((const __m256i *)sums), weight));
    if (b->channels == 1) {
        __m256i value = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)pixels));
        _mm256_storeu_si256((__m256i *)first, _mm256_add_epi32(_mm256_loadu_si256((const __m256i *)first),
            _mm256_mullo_epi32(weight, value)));
    } else {
        uint32_t *second = first + plane;
        __m256i pair = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)pixels));
        _mm256_storeu_si256((__m256i *)first, _mm256_add_epi32(_mm256_loadu_si256((const __m256i *)first),
            _mm256_mullo_epi32(weight, _mm256_and_si256(pair, _mm256_set1_epi32(0xffff)))));
        _mm256_storeu_si256((__m256i *)second, _mm256_add_epi32(_mm256_loadu_si256((const __m256i *)second),
            _mm256_mullo_epi32(weight, _mm256_srli_epi32(pair, 16))));
    }
}

static void nlm_accumulate(const nlm_block *b, size_t count, uint32_t *a_sums, const uint8_t *a_pixels, uint32_t *b_sums,
    const uint8_t *b_pixels)
{
    const __m256i last = _mm256_set1_epi32((int)b->limit - 1);
    const __m128i shift = _mm_cvtsi32_si128(b->shift);
    size_t c = b->channels;
    for (size_t i = 0; i < count; i += 8) {
        __m256i distance = _mm256_loadu_si256((const __m256i *)(b->distances + i));
        __m256i weighed = _mm256_cmpeq_epi32(_mm256_min_epu32(distance, last), distance);
        if (_mm256_testz_si256(weighed, weighed)) continue;
        __m256i weight = _mm256_mask_i32gather_epi32(_mm256_setzero_si256(), b->weights,
            _mm256_srl_epi32(distance, shift), weighed, 4);
        if (a_sums) add_weighted(b, weight, a_sums + i, a_pixels + i * c);
        if (b_sums) add_weighted(b, weight, b_sums + i, b_pixels + i * c);
    }
}

#include "denoise_block_shared.h"

void cn_nlm_block_avx2(const nlm_block *block) { nlm_block_sums(block); }
