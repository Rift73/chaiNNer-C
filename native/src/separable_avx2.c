#if !defined(_M_X64)
#error "separable_avx2.c is an x86-64 /arch:AVX2 unit"
#endif
/* The separable filter's AVX2 unit (/arch:AVX2, spec 4.3): no export, reached
 * only through box_complete_ops.c's row dispatch on the call's ISA level, for
 * lanes 8 (OpenCV's AVX2/FMA dispatch) with the coordinate tables. Groups of 8
 * sit at absolute columns = 0 (mod 8) below row_width / 8 * 8, inside the
 * caller's [first, last); each lane runs its output's fused sequence
 * (separable_horizontal_one, separable_vertical_one), one vfmadd per fused tap.
 * Every other output goes through the spans. Copyright (C) 2000-2008 Intel
 * Corporation; (C) 2009 Willow Garage Inc. BSD notice in separable_shared.h. */
#include "border_runs_shared.h"
#include "separable_isa_shared.h"
#include "separable_shared.h"
#include <immintrin.h>

/* The 8 lanes of horizontal tap `offset` for the group at `column` of line (src's
 * row): one load when every lane's pixel + offset lies in the row, else each lane
 * through the column table. pixel and channel are the lanes'. */
static __m256 horizontal_lanes(const separable_context *ctx, const float *line, size_t column,
    const int64_t *pixel, const size_t *channel, int64_t offset)
{
    size_t c = ctx->channels, lane;
    float values[8];
    if (pixel[0] + offset >= 0 && pixel[7] + offset < (int64_t)ctx->width)
        return _mm256_loadu_ps(line + ((ptrdiff_t)column + (ptrdiff_t)offset * (ptrdiff_t)c));
    for (lane = 0; lane < 8; ++lane)
        values[lane] = line[(size_t)ctx->columns[pixel[lane] + offset] * c + channel[lane]];
    return _mm256_loadu_ps(values);
}

/* The horizontal group at `column` of line, any lane's tap possibly reflected. */
static __m256 horizontal_group(const separable_context *ctx, const float *line, size_t column)
{
    size_t c = ctx->channels, radius = ctx->rx, part = column % c, channel[8], lane, k;
    int64_t pixel[8], x = (int64_t)(column / c);
    const float *kernel = ctx->kx;
    __m256 sum, pair;
    for (lane = 0; lane < 8; ++lane) {
        pixel[lane] = x;
        channel[lane] = part;
        if (++part == c) {
            part = 0;
            ++x;
        }
    }
    if (radius == 0) return horizontal_lanes(ctx, line, column, pixel, channel, 0);
    if (radius <= 2) {
        pair = _mm256_add_ps(horizontal_lanes(ctx, line, column, pixel, channel, -1),
            horizontal_lanes(ctx, line, column, pixel, channel, 1));
        sum = _mm256_fmadd_ps(horizontal_lanes(ctx, line, column, pixel, channel, 0),
            _mm256_set1_ps(kernel[radius]), _mm256_mul_ps(pair, _mm256_set1_ps(kernel[radius + 1])));
        if (radius == 2) {
            pair = _mm256_add_ps(horizontal_lanes(ctx, line, column, pixel, channel, 2),
                horizontal_lanes(ctx, line, column, pixel, channel, -2));
            sum = _mm256_fmadd_ps(pair, _mm256_set1_ps(kernel[4]), sum);
        }
        return sum;
    }
    sum = _mm256_setzero_ps();
    for (k = 0; k <= radius * 2; ++k)
        sum = _mm256_fmadd_ps(horizontal_lanes(ctx, line, column, pixel, channel, (int64_t)k - (int64_t)radius),
            _mm256_set1_ps(kernel[k]), sum);
    return sum;
}

/* count horizontal outputs (a multiple of 8) into out, from the elements of a row
 * in which each one's every tap lies (the interior, or a border run's reflected
 * copy): center is the first output's element, and tap offset o of the group at j
 * is one load at center + j + o * c. Eight groups at a time keep eight independent
 * sums (a 4-cycle FMA at 2 per cycle needs 8 in flight); a tail of four or more
 * groups keeps four, then one group at a time. */
static void horizontal_interior(const separable_context *ctx, const float *center, size_t count, float *out)
{
    ptrdiff_t c = (ptrdiff_t)ctx->channels, radius = (ptrdiff_t)ctx->rx, k;
    const float *kernel = ctx->kx;
    if (radius <= 2) {
        for (; count; count -= 8, center += 8, out += 8) {
            __m256 sum = _mm256_loadu_ps(center);
            if (radius > 0) {
                __m256 pair = _mm256_add_ps(_mm256_loadu_ps(center - c), _mm256_loadu_ps(center + c));
                sum = _mm256_fmadd_ps(sum, _mm256_set1_ps(kernel[radius]),
                    _mm256_mul_ps(pair, _mm256_set1_ps(kernel[radius + 1])));
                if (radius == 2) {
                    pair = _mm256_add_ps(_mm256_loadu_ps(center + 2 * c), _mm256_loadu_ps(center - 2 * c));
                    sum = _mm256_fmadd_ps(pair, _mm256_set1_ps(kernel[4]), sum);
                }
            }
            _mm256_storeu_ps(out, sum);
        }
        return;
    }
    for (; count >= 64; count -= 64, center += 64, out += 64) {
        __m256 sum0 = _mm256_setzero_ps(), sum1 = sum0, sum2 = sum0, sum3 = sum0;
        __m256 sum4 = sum0, sum5 = sum0, sum6 = sum0, sum7 = sum0;
        for (k = 0; k <= radius * 2; ++k) {
            const float *taps = center + (k - radius) * c;
            __m256 coefficient = _mm256_set1_ps(kernel[k]);
            sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(taps), coefficient, sum0);
            sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(taps + 8), coefficient, sum1);
            sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(taps + 16), coefficient, sum2);
            sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(taps + 24), coefficient, sum3);
            sum4 = _mm256_fmadd_ps(_mm256_loadu_ps(taps + 32), coefficient, sum4);
            sum5 = _mm256_fmadd_ps(_mm256_loadu_ps(taps + 40), coefficient, sum5);
            sum6 = _mm256_fmadd_ps(_mm256_loadu_ps(taps + 48), coefficient, sum6);
            sum7 = _mm256_fmadd_ps(_mm256_loadu_ps(taps + 56), coefficient, sum7);
        }
        _mm256_storeu_ps(out, sum0);
        _mm256_storeu_ps(out + 8, sum1);
        _mm256_storeu_ps(out + 16, sum2);
        _mm256_storeu_ps(out + 24, sum3);
        _mm256_storeu_ps(out + 32, sum4);
        _mm256_storeu_ps(out + 40, sum5);
        _mm256_storeu_ps(out + 48, sum6);
        _mm256_storeu_ps(out + 56, sum7);
    }
    if (count >= 32) {
        __m256 sum0 = _mm256_setzero_ps(), sum1 = sum0, sum2 = sum0, sum3 = sum0;
        for (k = 0; k <= radius * 2; ++k) {
            const float *taps = center + (k - radius) * c;
            __m256 coefficient = _mm256_set1_ps(kernel[k]);
            sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(taps), coefficient, sum0);
            sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(taps + 8), coefficient, sum1);
            sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(taps + 16), coefficient, sum2);
            sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(taps + 24), coefficient, sum3);
        }
        _mm256_storeu_ps(out, sum0);
        _mm256_storeu_ps(out + 8, sum1);
        _mm256_storeu_ps(out + 16, sum2);
        _mm256_storeu_ps(out + 24, sum3);
        count -= 32;
        center += 32;
        out += 32;
    }
    for (; count; count -= 8, center += 8, out += 8) {
        __m256 sum = _mm256_setzero_ps();
        for (k = 0; k <= radius * 2; ++k)
            sum = _mm256_fmadd_ps(_mm256_loadu_ps(center + (k - radius) * c), _mm256_set1_ps(kernel[k]), sum);
        _mm256_storeu_ps(out, sum);
    }
}

/* Horizontal groups [column, stop) of line, some lane's tap possibly reflected (a
 * border run): their taps read the row's virtual elements [column - rx c, stop +
 * rx c), copied once into edge (reflected_copy), and horizontal_interior runs on
 * the copy. A copy above SEPARABLE_EDGE_FLOATS takes horizontal_group per group. */
static void horizontal_run(const separable_context *ctx, const float *line, size_t column, size_t stop, float *out)
{
    float edge[SEPARABLE_EDGE_FLOATS];
    size_t reach = ctx->rx * ctx->channels;
    if (column >= stop) return;
    if (stop - column + 2 * reach > SEPARABLE_EDGE_FLOATS) {
        for (; column < stop; column += 8, out += 8) _mm256_storeu_ps(out, horizontal_group(ctx, line, column));
        return;
    }
    reflected_copy(ctx, line, column, stop - column + 2 * reach, edge);
    horizontal_interior(ctx, edge + reach, stop - column, out);
}

void cn_separable_horizontal_avx2(const separable_context *ctx, size_t y, size_t first, size_t last, float *out)
{
    size_t c = ctx->channels, row_width = ctx->width * c, fused = row_width / 8 * 8;
    size_t column = (first + 7) / 8 * 8, stop = (last < fused ? last : fused) / 8 * 8, head, tail;
    const float *line = ctx->src + y * row_width;
    if (column >= stop) {
        separable_horizontal_span(ctx, y, first, last, out);
        return;
    }
    /* Interior groups: column >= rx * c and column + 8 <= (width - rx) * c; the border
     * runs [column, head) and [tail, stop) hold the others, each widened into the
     * interior to a block of eight groups where the segment has one. */
    border_runs(column, stop, ctx->rx * c, ctx->width > ctx->rx ? (ctx->width - ctx->rx) * c : 0, 8, 64, &head,
        &tail);
    separable_horizontal_span(ctx, y, first, column, out);
    horizontal_run(ctx, line, column, head, out + (column - first));
    horizontal_interior(ctx, line + head, tail - head, out + (head - first));
    horizontal_run(ctx, line, tail, stop, out + (tail - first));
    separable_horizontal_span(ctx, y, stop, last, out + (stop - first));
}

void cn_separable_vertical_avx2(const separable_context *ctx, const float *buffer, size_t stride,
    size_t first_row, size_t y, size_t first, size_t last, float *out)
{
    size_t row_width = ctx->width * ctx->channels, fused = row_width / 8 * 8, at;
    size_t column = (first + 7) / 8 * 8, stop = (last < fused ? last : fused) / 8 * 8;
    ptrdiff_t radius = (ptrdiff_t)ctx->ry, k;
    const float *kernel = ctx->ky, *center = vertical_line(ctx, buffer, stride, first_row, y, 0);
    __m256 middle = _mm256_set1_ps(kernel[radius]);
    if (column >= stop) {
        separable_vertical_span(ctx, buffer, stride, first_row, y, first, last, out);
        return;
    }
    separable_vertical_span(ctx, buffer, stride, first_row, y, first, column, out);
    /* Eight groups at a time (eight independent sums), then a tail of four or more
     * groups with four, then one group at a time, as horizontal_interior. */
    for (; stop - column >= 64; column += 64) {
        __m256 sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7;
        at = column - first;
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(center + at), middle, _mm256_setzero_ps());
        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(center + at + 8), middle, _mm256_setzero_ps());
        sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(center + at + 16), middle, _mm256_setzero_ps());
        sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(center + at + 24), middle, _mm256_setzero_ps());
        sum4 = _mm256_fmadd_ps(_mm256_loadu_ps(center + at + 32), middle, _mm256_setzero_ps());
        sum5 = _mm256_fmadd_ps(_mm256_loadu_ps(center + at + 40), middle, _mm256_setzero_ps());
        sum6 = _mm256_fmadd_ps(_mm256_loadu_ps(center + at + 48), middle, _mm256_setzero_ps());
        sum7 = _mm256_fmadd_ps(_mm256_loadu_ps(center + at + 56), middle, _mm256_setzero_ps());
        for (k = 1; k <= radius; ++k) {
            const float *below = vertical_line(ctx, buffer, stride, first_row, y, k) + at;
            const float *above = vertical_line(ctx, buffer, stride, first_row, y, -k) + at;
            __m256 coefficient = _mm256_set1_ps(kernel[radius + k]);
            sum0 = _mm256_fmadd_ps(_mm256_add_ps(_mm256_loadu_ps(below), _mm256_loadu_ps(above)), coefficient, sum0);
            sum1 = _mm256_fmadd_ps(_mm256_add_ps(_mm256_loadu_ps(below + 8), _mm256_loadu_ps(above + 8)),
                coefficient, sum1);
            sum2 = _mm256_fmadd_ps(_mm256_add_ps(_mm256_loadu_ps(below + 16), _mm256_loadu_ps(above + 16)),
                coefficient, sum2);
            sum3 = _mm256_fmadd_ps(_mm256_add_ps(_mm256_loadu_ps(below + 24), _mm256_loadu_ps(above + 24)),
                coefficient, sum3);
            sum4 = _mm256_fmadd_ps(_mm256_add_ps(_mm256_loadu_ps(below + 32), _mm256_loadu_ps(above + 32)),
                coefficient, sum4);
            sum5 = _mm256_fmadd_ps(_mm256_add_ps(_mm256_loadu_ps(below + 40), _mm256_loadu_ps(above + 40)),
                coefficient, sum5);
            sum6 = _mm256_fmadd_ps(_mm256_add_ps(_mm256_loadu_ps(below + 48), _mm256_loadu_ps(above + 48)),
                coefficient, sum6);
            sum7 = _mm256_fmadd_ps(_mm256_add_ps(_mm256_loadu_ps(below + 56), _mm256_loadu_ps(above + 56)),
                coefficient, sum7);
        }
        _mm256_storeu_ps(out + at, sum0);
        _mm256_storeu_ps(out + at + 8, sum1);
        _mm256_storeu_ps(out + at + 16, sum2);
        _mm256_storeu_ps(out + at + 24, sum3);
        _mm256_storeu_ps(out + at + 32, sum4);
        _mm256_storeu_ps(out + at + 40, sum5);
        _mm256_storeu_ps(out + at + 48, sum6);
        _mm256_storeu_ps(out + at + 56, sum7);
    }
    if (stop - column >= 32) {
        __m256 sum0, sum1, sum2, sum3;
        at = column - first;
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(center + at), middle, _mm256_setzero_ps());
        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(center + at + 8), middle, _mm256_setzero_ps());
        sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(center + at + 16), middle, _mm256_setzero_ps());
        sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(center + at + 24), middle, _mm256_setzero_ps());
        for (k = 1; k <= radius; ++k) {
            const float *below = vertical_line(ctx, buffer, stride, first_row, y, k) + at;
            const float *above = vertical_line(ctx, buffer, stride, first_row, y, -k) + at;
            __m256 coefficient = _mm256_set1_ps(kernel[radius + k]);
            sum0 = _mm256_fmadd_ps(_mm256_add_ps(_mm256_loadu_ps(below), _mm256_loadu_ps(above)), coefficient, sum0);
            sum1 = _mm256_fmadd_ps(_mm256_add_ps(_mm256_loadu_ps(below + 8), _mm256_loadu_ps(above + 8)),
                coefficient, sum1);
            sum2 = _mm256_fmadd_ps(_mm256_add_ps(_mm256_loadu_ps(below + 16), _mm256_loadu_ps(above + 16)),
                coefficient, sum2);
            sum3 = _mm256_fmadd_ps(_mm256_add_ps(_mm256_loadu_ps(below + 24), _mm256_loadu_ps(above + 24)),
                coefficient, sum3);
        }
        _mm256_storeu_ps(out + at, sum0);
        _mm256_storeu_ps(out + at + 8, sum1);
        _mm256_storeu_ps(out + at + 16, sum2);
        _mm256_storeu_ps(out + at + 24, sum3);
        column += 32;
    }
    for (; column < stop; column += 8) {
        __m256 sum;
        at = column - first;
        sum = _mm256_fmadd_ps(_mm256_loadu_ps(center + at), middle, _mm256_setzero_ps());
        for (k = 1; k <= radius; ++k) {
            __m256 pair = _mm256_add_ps(_mm256_loadu_ps(vertical_line(ctx, buffer, stride, first_row, y, k) + at),
                _mm256_loadu_ps(vertical_line(ctx, buffer, stride, first_row, y, -k) + at));
            sum = _mm256_fmadd_ps(pair, _mm256_set1_ps(kernel[radius + k]), sum);
        }
        _mm256_storeu_ps(out + at, sum);
    }
    separable_vertical_span(ctx, buffer + (stop - first), stride, first_row, y, stop, last, out + (stop - first));
}
