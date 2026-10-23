#if !defined(_M_X64)
#error "resample_avx2.c is an x86-64 /arch:AVX2 unit"
#endif
/* Resample's two passes, AVX2 unit (/arch:AVX2, spec 4.3): no export, reached only
 * through horizontal_range's and vertical_range's dispatch on the call's ISA level
 * (SP4b D14). The horizontal pass runs eight source rows side by side, one per lane, on
 * columns transposed from them (cn_resample_horizontal_avx2 in resample_shared.h).
 * In the vertical pass each lane runs
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

/* r[0..7] transposed in place: lane j of r[i] becomes lane i of r[j]. */
static void transpose8(__m256 r[8])
{
    __m256 t0 = _mm256_unpacklo_ps(r[0], r[1]), t1 = _mm256_unpackhi_ps(r[0], r[1]);
    __m256 t2 = _mm256_unpacklo_ps(r[2], r[3]), t3 = _mm256_unpackhi_ps(r[2], r[3]);
    __m256 t4 = _mm256_unpacklo_ps(r[4], r[5]), t5 = _mm256_unpackhi_ps(r[4], r[5]);
    __m256 t6 = _mm256_unpacklo_ps(r[6], r[7]), t7 = _mm256_unpackhi_ps(r[6], r[7]);
    __m256 u0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 u1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 u2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 u3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 u4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 u5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 u6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 u7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));
    r[0] = _mm256_permute2f128_ps(u0, u4, 0x20);
    r[1] = _mm256_permute2f128_ps(u1, u5, 0x20);
    r[2] = _mm256_permute2f128_ps(u2, u6, 0x20);
    r[3] = _mm256_permute2f128_ps(u3, u7, 0x20);
    r[4] = _mm256_permute2f128_ps(u0, u4, 0x31);
    r[5] = _mm256_permute2f128_ps(u1, u5, 0x31);
    r[6] = _mm256_permute2f128_ps(u2, u6, 0x31);
    r[7] = _mm256_permute2f128_ps(u3, u7, 0x31);
}

/* A tile's transposed source columns and its sums, in groups of 8 floats (32 KiB on the stack), and its widest
 * run of outputs. */
#define TILE_GROUPS 1024
#define TILE_OUTPUTS 64

/* count samples from each of the 8 rows, starting at sample at, into count groups of 8 (one lane per row). */
static void transpose_in(const float *const row[8], size_t at, size_t count, float *groups)
{
    size_t j = 0, r;
    for (; count - j >= 8; j += 8) {
        __m256 v[8];
        for (r = 0; r < 8; ++r) v[r] = _mm256_loadu_ps(row[r] + at + j);
        transpose8(v);
        for (r = 0; r < 8; ++r) _mm256_storeu_ps(groups + (j + r) * 8, v[r]);
    }
    for (; j < count; ++j)
        for (r = 0; r < 8; ++r) groups[j * 8 + r] = row[r][at + j];
}

/* count groups of 8 back into the first rows of 8 rows, from sample at of each. */
static void transpose_out(const float *groups, size_t count, float *target, size_t stride, size_t at, size_t rows)
{
    size_t j = 0, r;
    for (; count - j >= 8; j += 8) {
        __m256 v[8];
        for (r = 0; r < 8; ++r) v[r] = _mm256_loadu_ps(groups + (j + r) * 8);
        transpose8(v);
        for (r = 0; r < rows; ++r) _mm256_storeu_ps(target + r * stride + at + j, v[r]);
    }
    for (; j < count; ++j)
        for (r = 0; r < rows; ++r) target[r * stride + at + j] = groups[j * 8 + r];
}

int cn_resample_horizontal_avx2(const resample_context *ctx, size_t y, size_t rows)
{
    const size_t channels = ctx->channels, width = ctx->target_width, out_count = width * channels;
    const coefficient_line *lines = ctx->horizontal;
    float groups[TILE_GROUPS * 8];
    const float *row[8];
    for (size_t r = 0; r < 8; ++r)
        row[r] = ctx->source + (y + (r < rows ? r : rows - 1)) * ctx->source_width * channels;
    __m256 nan = _mm256_setzero_ps();
    for (size_t x = 0; x < width;) {
        /* The tile: outputs [x, last) whose source columns [first, end) and sums fit the groups. */
        const size_t first = lines[x].start;
        size_t last = x, end = first;
        while (last < width && last - x < TILE_OUTPUTS) {
            const size_t reach = lines[last].start + lines[last].count;
            const size_t next_end = reach > end ? reach : end;
            if ((next_end - first + last + 1 - x) * channels > TILE_GROUPS) break;
            end = next_end;
            ++last;
        }
        if (last == x) return 0;
        float *columns = groups, *sums = groups + (end - first) * channels * 8;
        transpose_in(row, first * channels, (end - first) * channels, columns);
        for (size_t k = x; k < last; ++k) {
            const float *in = columns + (lines[k].start - first) * channels * 8, *weights = lines[k].weights;
            __m256 sum0 = _mm256_setzero_ps(), sum1 = sum0, sum2 = sum0, sum3 = sum0;
            for (size_t i = 0; i < lines[k].count; ++i, in += channels * 8) {
                __m256 weight = _mm256_set1_ps(weights[i]);
                sum0 = tap(sum0, in, weight);
                if (channels > 1) sum1 = tap(sum1, in + 8, weight);
                if (channels > 2) sum2 = tap(sum2, in + 16, weight);
                if (channels > 3) sum3 = tap(sum3, in + 24, weight);
            }
            float *out = sums + (k - x) * channels * 8;
            _mm256_storeu_ps(out, sum0);
            if (channels > 1) _mm256_storeu_ps(out + 8, sum1);
            if (channels > 2) _mm256_storeu_ps(out + 16, sum2);
            if (channels > 3) _mm256_storeu_ps(out + 24, sum3);
            nan = _mm256_or_ps(nan, _mm256_or_ps(_mm256_cmp_ps(sum0, sum1, _CMP_UNORD_Q),
                                        _mm256_cmp_ps(sum2, sum3, _CMP_UNORD_Q)));
        }
        transpose_out(sums, (last - x) * channels, ctx->intermediate + y * out_count, out_count, x * channels, rows);
        x = last;
    }
    return !_mm256_movemask_ps(nan);
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
