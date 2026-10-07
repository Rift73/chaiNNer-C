#if !defined(_M_X64)
#error "resample_avx2.c is an x86-64 /arch:AVX2 unit"
#endif
/* Resample's vertical pass, AVX2 unit (/arch:AVX2, spec 4.3): no export, reached only
 * through vertical_range's dispatch on the call's ISA level (SP4b D14). Each lane runs
 * its output's vertical_one sequence: acc = 0, then acc = acc + in * w per tap in
 * order (a vmulps, then a vaddps; no fused multiply-add), then B3's clip instructions. Blocks
 * of 64 outputs keep 8 independent sums (a vaddps chain per group: 8 in flight cover
 * its latency at 2 per cycle), then single groups of 8; the row's last outputs run
 * vertical_one. Copyright (c) 2015 PistonDevelopers; (c) 2023 Michael Schmidt. MIT
 * notice in resample_shared.h. */
#include "resample_shared.h"
#include <immintrin.h>

/* One tap of a group: acc + in * weight, two roundings. */
static __m256 tap(__m256 acc, const float *in, __m256 weight)
{
    return _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(in), weight));
}

/* B3's clips (vertical_range, B3 0x180020472-0x18002048B): clip 1 is
 * minss(1, maxss(0, acc)), the constants first, so a NaN acc passes; clip 2 is
 * minss(maxss(acc, 0), 1), so a NaN acc gives +0. */
static __m256 clipped(__m256 sum, int clip)
{
    __m256 zero = _mm256_setzero_ps(), one = _mm256_set1_ps(1.0f);
    if (clip == 1) return _mm256_min_ps(one, _mm256_max_ps(zero, sum));
    if (clip == 2) return _mm256_min_ps(_mm256_max_ps(sum, zero), one);
    return sum;
}

/* Whether a lane of a or b holds a NaN. */
static int unordered(__m256 a, __m256 b)
{
    return _mm256_movemask_ps(_mm256_cmp_ps(a, b, _CMP_UNORD_Q));
}

/* Outputs [x, x + n) of the row through vertical_one: the row's last outputs, and a
 * group redone in B3's order. Where two NaNs meet, B3's addss keeps the accumulated
 * one (its first source) and the compiler may commute vaddps's sources, so a group
 * whose sum holds a NaN is redone unless clip 2 has made every NaN +0. */
static void redo(const resample_context *ctx, const coefficient_line *line, float *out, size_t x, size_t n)
{
    for (size_t k = x; k < x + n; ++k) out[k] = vertical_one(ctx, line, k);
}

void cn_resample_vertical_avx2(const resample_context *ctx, size_t y, size_t first, size_t last)
{
    const coefficient_line *line = &ctx->vertical[y];
    size_t stride = ctx->target_width * ctx->channels, count = line->count, x = first, i;
    const float *rows = ctx->intermediate + line->start * stride, *weights = line->weights;
    float *out = ctx->out + y * stride;
    int clip = ctx->clip;
    for (; last - x >= 64; x += 64) {
        const float *in = rows + x;
        __m256 sum0 = _mm256_setzero_ps(), sum1 = sum0, sum2 = sum0, sum3 = sum0;
        __m256 sum4 = sum0, sum5 = sum0, sum6 = sum0, sum7 = sum0;
        for (i = 0; i < count; ++i, in += stride) {
            __m256 weight = _mm256_set1_ps(weights[i]);
            sum0 = tap(sum0, in, weight);
            sum1 = tap(sum1, in + 8, weight);
            sum2 = tap(sum2, in + 16, weight);
            sum3 = tap(sum3, in + 24, weight);
            sum4 = tap(sum4, in + 32, weight);
            sum5 = tap(sum5, in + 40, weight);
            sum6 = tap(sum6, in + 48, weight);
            sum7 = tap(sum7, in + 56, weight);
        }
        _mm256_storeu_ps(out + x, clipped(sum0, clip));
        _mm256_storeu_ps(out + x + 8, clipped(sum1, clip));
        _mm256_storeu_ps(out + x + 16, clipped(sum2, clip));
        _mm256_storeu_ps(out + x + 24, clipped(sum3, clip));
        _mm256_storeu_ps(out + x + 32, clipped(sum4, clip));
        _mm256_storeu_ps(out + x + 40, clipped(sum5, clip));
        _mm256_storeu_ps(out + x + 48, clipped(sum6, clip));
        _mm256_storeu_ps(out + x + 56, clipped(sum7, clip));
        if (clip != 2 && (unordered(sum0, sum1) | unordered(sum2, sum3) | unordered(sum4, sum5) |
                unordered(sum6, sum7)))
            redo(ctx, line, out, x, 64);
    }
    for (; last - x >= 8; x += 8) {
        const float *in = rows + x;
        __m256 sum = _mm256_setzero_ps();
        for (i = 0; i < count; ++i, in += stride) sum = tap(sum, in, _mm256_set1_ps(weights[i]));
        _mm256_storeu_ps(out + x, clipped(sum, clip));
        if (clip != 2 && unordered(sum, sum)) redo(ctx, line, out, x, 8);
    }
    redo(ctx, line, out, x, last - x);
}
