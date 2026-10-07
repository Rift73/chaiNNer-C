#if !defined(_M_X64)
#error "dither_avx2.c is an x86-64 /arch:AVX2 unit"
#endif
/* The palette dither's AVX2 unit (/arch:AVX2, spec 4.3; SP4b D12): no export, reached
 * only through nearest_palette's linear branch when cn_neighborhood_palette_apply read
 * ISA level avx2 or above for the call and built the channel-major palette (job->soa). */
#include "dither_shared.h"
#include "numeric.h"
#include <immintrin.h>
#include <intrin.h>
#include <math.h>

#define DITHER_GROUPS (DITHER_SOA_ENTRIES / 8)

/* Entries [8g, 8g + 8): per lane palette_distance's sequence, 0 + d0 d0, then + dk dk in
 * channel order (vsubps palette - color, vmulps, vaddps; no fused multiply-add), for
 * 1, 3 or 4 channels. */
static __m256 group_distance(const float *entries, size_t stride, size_t channels,
                             __m256 t0, __m256 t1, __m256 t2, __m256 t3)
{
    __m256 delta = _mm256_sub_ps(_mm256_loadu_ps(entries), t0);
    __m256 d = _mm256_add_ps(_mm256_setzero_ps(), _mm256_mul_ps(delta, delta));
    if (channels == 1) return d;
    delta = _mm256_sub_ps(_mm256_loadu_ps(entries + stride), t1);
    d = _mm256_add_ps(d, _mm256_mul_ps(delta, delta));
    delta = _mm256_sub_ps(_mm256_loadu_ps(entries + 2 * stride), t2);
    d = _mm256_add_ps(d, _mm256_mul_ps(delta, delta));
    if (channels == 3) return d;
    delta = _mm256_sub_ps(_mm256_loadu_ps(entries + 3 * stride), t3);
    return _mm256_add_ps(d, _mm256_mul_ps(delta, delta));
}

/* The lanes of group g below count. */
static __m256 group_lanes(size_t count, size_t g)
{
    const __m256i lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256i left = _mm256_set1_epi32((int)(count - 8 * g));
    return _mm256_castsi256_ps(_mm256_cmpgt_epi32(left, lane));
}

/* dither_shared.h. Lanes below palette_count with an ordered distance enter the minimum
 * (vminps over +inf); the others never do. Index 0 when the first distance is NaN; else
 * the first lane, in entry order, whose distance compares equal (_CMP_EQ_OQ, honouring
 * DAZ as the scan's < does) to that minimum. channels is 1, 3 or 4 (palette creation and
 * apply refuse any other count); color holds that many floats. */
size_t cn_dither_nearest_avx2(const dither_job *job, const float *color)
{
    const size_t count = job->palette_count, channels = job->channels;
    const size_t stride = job->soa_stride, groups = (count + 7) / 8;
    const __m256 infinity = _mm256_set1_ps(CN_INFINITY_F);
    const __m256 t0 = _mm256_set1_ps(color[0]);
    const __m256 t1 = channels > 1 ? _mm256_set1_ps(color[1]) : t0;
    const __m256 t2 = channels > 2 ? _mm256_set1_ps(color[2]) : t0;
    const __m256 t3 = channels > 3 ? _mm256_set1_ps(color[3]) : t0;
    __m256 distance[DITHER_GROUPS];
    __m256 minimum = infinity;
    for (size_t g = 0; g < groups; ++g) {
        __m256 d = group_distance(job->soa + 8 * g, stride, channels, t0, t1, t2, t3);
        distance[g] = d;
        __m256 ordered = _mm256_and_ps(group_lanes(count, g), _mm256_cmp_ps(d, d, _CMP_ORD_Q));
        minimum = _mm256_min_ps(minimum, _mm256_blendv_ps(infinity, d, ordered));
    }
    if (_mm256_movemask_ps(_mm256_cmp_ps(distance[0], distance[0], _CMP_UNORD_Q)) & 1)
        return 0;
    minimum = _mm256_min_ps(minimum, _mm256_permute2f128_ps(minimum, minimum, 1));
    minimum = _mm256_min_ps(minimum, _mm256_permute_ps(minimum, _MM_SHUFFLE(1, 0, 3, 2)));
    minimum = _mm256_min_ps(minimum, _mm256_permute_ps(minimum, _MM_SHUFFLE(2, 3, 0, 1)));
    for (size_t g = 0; g < groups; ++g) {
        __m256 equal = _mm256_cmp_ps(distance[g], minimum, _CMP_EQ_OQ);
        int lanes = _mm256_movemask_ps(_mm256_and_ps(group_lanes(count, g), equal));
        if (lanes) {
            unsigned long first;
            _BitScanForward(&first, (unsigned long)lanes);
            return 8 * g + first;
        }
    }
    return 0;
}
