#if !defined(_M_X64)
#error "planar_avx2.c is an x86-64 /arch:AVX2 unit"
#endif
/* Planar to interleaved float32, AVX2 unit (/arch:AVX2, spec 4.3): no export,
 * reached only through planar_range's dispatch on the call's ISA level. A lane's
 * value is half_bits_to_f32's (integer rebiasing, the payload-keeping Inf/NaN
 * form, and the exact mantissa * 2^-24 for subnormals) or the float32 sample's
 * bits; permutes and blends only move bits. No F16C: its conversion quiets a
 * signalling NaN, which NumPy's cast keeps. */
#include "planar_shared.h"
#include <immintrin.h>

static __m256 halves(const uint16_t *p)
{
    __m256i h = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)p));
    __m256i sign = _mm256_slli_epi32(_mm256_and_si256(h, _mm256_set1_epi32(0x8000)), 16);
    __m256i exponent = _mm256_and_si256(h, _mm256_set1_epi32(0x7c00));
    __m256i rest = _mm256_slli_epi32(_mm256_and_si256(h, _mm256_set1_epi32(0x7fff)), 13);
    __m256i normal = _mm256_add_epi32(rest, _mm256_set1_epi32(112 << 23));
    __m256i special = _mm256_or_si256(rest, _mm256_set1_epi32(0x7f800000));
    __m256 subnormal = _mm256_mul_ps(
        _mm256_cvtepi32_ps(_mm256_and_si256(h, _mm256_set1_epi32(0x3ff))),
        _mm256_set1_ps(0x1p-24f));
    __m256i bits = _mm256_blendv_epi8(normal, special,
        _mm256_cmpeq_epi32(exponent, _mm256_set1_epi32(0x7c00)));
    bits = _mm256_blendv_epi8(bits, _mm256_castps_si256(subnormal),
        _mm256_cmpeq_epi32(exponent, _mm256_setzero_si256()));
    return _mm256_castsi256_ps(_mm256_or_si256(bits, sign));
}

static __m256 load8(const planar_job *job, size_t offset)
{
    if (job->half) return halves((const uint16_t *)job->src + offset);
    return _mm256_loadu_ps((const float *)job->src + offset);
}

/* a, b, c: 8 pixels of output channels 0, 1, 2; stores a0 b0 c0 a1 b1 c1 ... */
static void store3(float *out, __m256 a, __m256 b, __m256 c)
{
    const __m256i a0 = _mm256_setr_epi32(0, 0, 0, 1, 0, 0, 2, 0);
    const __m256i b0 = _mm256_setr_epi32(0, 0, 0, 0, 1, 0, 0, 2);
    const __m256i c0 = _mm256_setr_epi32(0, 0, 0, 0, 0, 1, 0, 0);
    const __m256i a1 = _mm256_setr_epi32(0, 3, 0, 0, 4, 0, 0, 5);
    const __m256i b1 = _mm256_setr_epi32(0, 0, 3, 0, 0, 4, 0, 0);
    const __m256i c1 = _mm256_setr_epi32(2, 0, 0, 3, 0, 0, 4, 0);
    const __m256i a2 = _mm256_setr_epi32(0, 0, 6, 0, 0, 7, 0, 0);
    const __m256i b2 = _mm256_setr_epi32(5, 0, 0, 6, 0, 0, 7, 0);
    const __m256i c2 = _mm256_setr_epi32(0, 5, 0, 0, 6, 0, 0, 7);
    __m256 o0 = _mm256_blend_ps(_mm256_permutevar8x32_ps(a, a0), _mm256_permutevar8x32_ps(b, b0), 0x92);
    __m256 o1 = _mm256_blend_ps(_mm256_permutevar8x32_ps(a, a1), _mm256_permutevar8x32_ps(b, b1), 0x24);
    __m256 o2 = _mm256_blend_ps(_mm256_permutevar8x32_ps(a, a2), _mm256_permutevar8x32_ps(b, b2), 0x49);
    _mm256_storeu_ps(out, _mm256_blend_ps(o0, _mm256_permutevar8x32_ps(c, c0), 0x24));
    _mm256_storeu_ps(out + 8, _mm256_blend_ps(o1, _mm256_permutevar8x32_ps(c, c1), 0x49));
    _mm256_storeu_ps(out + 16, _mm256_blend_ps(o2, _mm256_permutevar8x32_ps(c, c2), 0x92));
}

void cn_planar_rows_avx2(const planar_job *job, size_t begin, size_t end)
{
    size_t channels = job->channels, width = job->width, body = width - width % 8;
    if (channels != 1 && channels != 3) {
        planar_rows(job, begin, end, 0);
        return;
    }
    size_t first = job->reverse && channels == 3 ? 2 * job->plane_stride : 0;
    size_t last = job->reverse && channels == 3 ? 0 : 2 * job->plane_stride;
    for (size_t y = begin; y < end; ++y) {
        float *row = job->out + y * width * channels;
        size_t base = y * job->row_stride;
        for (size_t x = 0; x < body; x += 8) {
            if (channels == 1) {
                _mm256_storeu_ps(row + x, load8(job, base + x));
                continue;
            }
            store3(row + 3 * x, load8(job, first + base + x),
                load8(job, job->plane_stride + base + x), load8(job, last + base + x));
        }
    }
    planar_rows(job, begin, end, body);
}
