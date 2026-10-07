#if !defined(_M_X64)
#error "morphology_avx2.c is an x86-64 /arch:AVX2 unit"
#endif
/* Morphology's AVX2 unit (/arch:AVX2, spec 4.3; SP4b D11): no export, reached only
 * through the C entries' dispatch on the call's ISA level: the exceptional scan of
 * cn_image_exceptional and the cn_morphology_complete pre-scan, the ellipse combine
 * and the cross merge of cn_filter_morphology. Groups stay inside the caller's range
 * (a chunk, or the whole serial scan); the rest runs the shared scalar helpers, here
 * with VEX encodings. */
#include "exceptional.h"
#include "exceptional_shared.h"
#include "merge_shared.h"
#include "morphology_shared.h"
#include <immintrin.h>
#include <string.h>

/* exceptional_bits per lane, as compares on the words: the exponent all ones
 * (vpand, vpcmpeqd), or, as signed int32, below bound (vpcmpgtd): with bound
 * 0x80000001 that is -0 alone, with 0x80800000 (DAZ) -0 and the negative denormals. */
static __m256i exceptional_lanes(__m256i words, __m256i bound)
{
    const __m256i exponent = _mm256_set1_epi32(0x7f800000);
    __m256i special = _mm256_cmpeq_epi32(_mm256_and_si256(words, exponent), exponent);
    return _mm256_or_si256(special, _mm256_cmpgt_epi32(bound, words));
}

/* Groups of 32 words, tested once (vptest), so the scan returns at the first group
 * holding a match; then groups of 8, and the last 1-7 words through exceptional_bits. */
int cn_exceptional_scan_avx2(const float *src, size_t count, int daz)
{
    const __m256i bound = _mm256_set1_epi32(daz ? (int)0x80800000u : (int)0x80000001u);
    const __m256i *words = (const __m256i *)src;
    size_t i = 0;
    for (; count - i >= 32; i += 32, words += 4) {
        __m256i lanes = exceptional_lanes(_mm256_loadu_si256(words), bound);
        lanes = _mm256_or_si256(lanes, exceptional_lanes(_mm256_loadu_si256(words + 1), bound));
        lanes = _mm256_or_si256(lanes, exceptional_lanes(_mm256_loadu_si256(words + 2), bound));
        lanes = _mm256_or_si256(lanes, exceptional_lanes(_mm256_loadu_si256(words + 3), bound));
        if (!_mm256_testz_si256(lanes, lanes)) return 1;
    }
    for (; count - i >= 8; i += 8, ++words) {
        __m256i lanes = exceptional_lanes(_mm256_loadu_si256(words), bound);
        if (!_mm256_testz_si256(lanes, lanes)) return 1;
    }
    for (; i < count; ++i) {
        uint32_t bits;
        memcpy(&bits, src + i, sizeof(bits));
        if (exceptional_bits(bits, daz)) return 1;
    }
    return 0;
}

/* update's vector form per lane: vmaxps/vminps(value, src), B3's maxss/minss
 * value,src (0x180024CAE/0x180024CB4), whose lanes return src on equal or unordered
 * operands and, under DAZ, a returned denormal as the zero of its sign, as maxss/minss
 * do. */
static __m256 update_lanes(__m256 value, __m256 src, int maximum)
{
    return maximum ? _mm256_max_ps(value, src) : _mm256_min_ps(value, src);
}

/* Per row segment (combine_segment), groups of 8 run combine_one's sequence over the
 * offsets; the segment's last 1-7 outputs run combine_one. */
void cn_morphology_combine_avx2(const morphology_context *ctx, size_t begin, size_t end)
{
    const size_t row_width = ctx->width * ctx->channels;
    const __m256 border = _mm256_set1_ps(ctx->border);
    float *out = ctx->out;
    size_t i = begin;
    while (i < end) {
        size_t y, stop = combine_segment(ctx, i, end, &y);
        for (; stop - i >= 8; i += 8) {
            __m256 value = ctx->offset ? _mm256_loadu_ps(out + i) : border;
            for (size_t dy = ctx->offset; dy < ctx->offset_end; ++dy) {
                const float *src = dy < ctx->zero_from ? ctx->src : ctx->current;
                size_t step = dy * row_width;
                if (y >= dy)
                    value = update_lanes(value, _mm256_loadu_ps(src + i - step), ctx->maximum);
                if (dy && dy < ctx->height - y)
                    value = update_lanes(value, _mm256_loadu_ps(src + i + step), ctx->maximum);
            }
            _mm256_storeu_ps(out + i, value);
        }
        for (; i < stop; ++i) out[i] = combine_one(ctx, i, y);
    }
}

/* Groups of 8 as merge_one: vcmpps(a, b, _CMP_GT_OQ) or (_CMP_LT_OQ), ordered like
 * B3's comiss + seta (an unordered pair keeps b), then vblendvps(b, a, mask), which
 * moves bits (no DAZ flush); the last 1-7 elements through merge_one. */
void cn_morphology_merge_avx2(const float *a, float *b, size_t begin, size_t end, int maximum)
{
    size_t i = begin;
    if (maximum) {
        for (; end - i >= 8; i += 8) {
            __m256 x = _mm256_loadu_ps(a + i), y = _mm256_loadu_ps(b + i);
            _mm256_storeu_ps(b + i, _mm256_blendv_ps(y, x, _mm256_cmp_ps(x, y, _CMP_GT_OQ)));
        }
    } else {
        for (; end - i >= 8; i += 8) {
            __m256 x = _mm256_loadu_ps(a + i), y = _mm256_loadu_ps(b + i);
            _mm256_storeu_ps(b + i, _mm256_blendv_ps(y, x, _mm256_cmp_ps(x, y, _CMP_LT_OQ)));
        }
    }
    for (; i < end; ++i) merge_one(a, b, i, maximum);
}
