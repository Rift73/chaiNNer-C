#if !defined(_M_X64) && !(defined(__linux__) && defined(__x86_64__))
#error "denoise_avx512.c is an x86-64 /arch:AVX512 unit"
#endif
/* Non-local means rows, AVX-512 unit (/arch:AVX512, spec 4.3): denoise_avx2.c at twice the lanes. */
#include "denoise_shared.h"
#include <immintrin.h>

/* Two 32-byte runs' differences as 16-bit lanes: |d| <= 255. */
CN_FORCE_INLINE __m512i differences(const uint8_t *a, ptrdiff_t displacement)
{
    return _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i *)a)),
        _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i *)(a + displacement))));
}

static void nlm_column_add(uint32_t *columns, const uint8_t *row, ptrdiff_t displacement, size_t count, size_t channels)
{
    if (channels == 1) {
        for (size_t i = 0; i < count; i += 32) {
            __m512i d = differences(row + i, displacement), squares = _mm512_mullo_epi16(d, d);
            _mm512_storeu_si512(columns + i, _mm512_add_epi32(_mm512_loadu_si512(columns + i),
                _mm512_cvtepu16_epi32(_mm512_castsi512_si256(squares))));
            _mm512_storeu_si512(columns + i + 16, _mm512_add_epi32(_mm512_loadu_si512(columns + i + 16),
                _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(squares, 1))));
        }
    } else {
        for (size_t i = 0; i < count; i += 16) {
            __m512i d = differences(row + i * 2, displacement);
            _mm512_storeu_si512(columns + i, _mm512_add_epi32(_mm512_loadu_si512(columns + i), _mm512_madd_epi16(d, d)));
        }
    }
}

static void nlm_column_slide(uint32_t *columns, const uint8_t *down, const uint8_t *up, ptrdiff_t displacement,
    size_t count, size_t channels)
{
    if (channels == 1) {
        for (size_t i = 0; i < count; i += 32) {
            __m512i in = differences(down + i, displacement), out = differences(up + i, displacement);
            __m512i added = _mm512_mullo_epi16(in, in), removed = _mm512_mullo_epi16(out, out);
            _mm512_storeu_si512(columns + i, _mm512_sub_epi32(_mm512_add_epi32(_mm512_loadu_si512(columns + i),
                _mm512_cvtepu16_epi32(_mm512_castsi512_si256(added))),
                _mm512_cvtepu16_epi32(_mm512_castsi512_si256(removed))));
            _mm512_storeu_si512(columns + i + 16, _mm512_sub_epi32(_mm512_add_epi32(_mm512_loadu_si512(columns + i + 16),
                _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(added, 1))),
                _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(removed, 1))));
        }
    } else {
        for (size_t i = 0; i < count; i += 16) {
            __m512i in = differences(down + i * 2, displacement), out = differences(up + i * 2, displacement);
            _mm512_storeu_si512(columns + i, _mm512_sub_epi32(_mm512_add_epi32(_mm512_loadu_si512(columns + i),
                _mm512_madd_epi16(in, in)), _mm512_madd_epi16(out, out)));
        }
    }
}

static void nlm_window_pass(uint32_t *out, const uint32_t *in, const uint32_t *columns, size_t count, size_t span, int odd)
{
    for (size_t i = 0; i < count; i += 16) {
        __m512i sum = _mm512_add_epi32(_mm512_loadu_si512(in + i), _mm512_loadu_si512(in + i + span));
        if (odd) sum = _mm512_add_epi32(sum, _mm512_loadu_si512(columns + i + 2 * span));
        _mm512_storeu_si512(out + i, sum);
    }
}

CN_FORCE_INLINE void add_weighted(const nlm_block *b, __m512i weight, uint32_t *sums, const uint8_t *pixels)
{
    size_t plane = b->rows * b->row_lanes;
    uint32_t *first = sums + plane;
    _mm512_storeu_si512(sums, _mm512_add_epi32(_mm512_loadu_si512(sums), weight));
    if (b->channels == 1) {
        __m512i value = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)pixels));
        _mm512_storeu_si512(first, _mm512_add_epi32(_mm512_loadu_si512(first), _mm512_mullo_epi32(weight, value)));
    } else {
        uint32_t *second = first + plane;
        __m512i pair = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i *)pixels));
        _mm512_storeu_si512(first, _mm512_add_epi32(_mm512_loadu_si512(first),
            _mm512_mullo_epi32(weight, _mm512_and_si512(pair, _mm512_set1_epi32(0xffff)))));
        _mm512_storeu_si512(second, _mm512_add_epi32(_mm512_loadu_si512(second),
            _mm512_mullo_epi32(weight, _mm512_srli_epi32(pair, 16))));
    }
}

static void nlm_accumulate(const nlm_block *b, size_t count, uint32_t *a_sums, const uint8_t *a_pixels, uint32_t *b_sums,
    const uint8_t *b_pixels)
{
    const __m512i limit = _mm512_set1_epi32((int)b->limit);
    const __m128i shift = _mm_cvtsi32_si128(b->shift);
    size_t c = b->channels;
    for (size_t i = 0; i < count; i += 16) {
        __m512i distance = _mm512_loadu_si512(b->distances + i);
        __mmask16 weighed = _mm512_cmplt_epu32_mask(distance, limit);
        if (!weighed) continue;
        __m512i weight = _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), weighed,
            _mm512_srl_epi32(distance, shift), b->weights, 4);
        if (a_sums) add_weighted(b, weight, a_sums + i, a_pixels + i * c);
        if (b_sums) add_weighted(b, weight, b_sums + i, b_pixels + i * c);
    }
}

#include "denoise_block_shared.h"

void cn_nlm_block_avx512(const nlm_block *block) { nlm_block_sums(block); }
