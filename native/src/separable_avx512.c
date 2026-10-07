#if !defined(_M_X64)
#error "separable_avx512.c is an x86-64 /arch:AVX512 unit"
#endif
/* The separable filter's AVX-512 unit (/arch:AVX512, spec 4.3, D15): no export,
 * reached only through box_complete_ops.c's row dispatch on the call's ISA level,
 * for lanes 8 (OpenCV's AVX2/FMA dispatch) with the coordinate tables.
 * separable_avx2.c's layout in groups of 16 at absolute columns = 0 (mod 16). The
 * mirror (lanes 8) fixes each output's sequence by its column alone: fused below
 * row_width / 8 * 8 (separable_horizontal_one, separable_vertical_one; one vfmadd per
 * fused tap), unfused from there on. The vector outputs of a row segment [first,
 * last) are [first, end), end = min(last, row_width / 8 * 8); the groups cover them
 * from first rounded down to end rounded up, and a group reaching past either end (a
 * chunk's first group; the group holding end, which straddles the fused boundary or
 * the segment's end) loads and stores only its lanes inside [first, end) through a
 * lane mask. The outputs from end on, the straddling group's other lanes among them,
 * go through the spans. Copyright (C) 2000-2008 Intel Corporation; (C) 2009 Willow
 * Garage Inc. BSD notice in separable_shared.h. */
#include "border_runs_shared.h"
#include "group_masks_shared.h"
#include "separable_isa_shared.h"
#include "separable_shared.h"
#include <immintrin.h>

/* The 16 lanes of horizontal tap `offset` for the group at `column` of line (src's
 * row), only the lanes in mask: one load when every lane's pixel + offset lies in the
 * row, else each lane in mask through the column table (the others 0). pixel and
 * channel are the lanes'. */
static __m512 horizontal_lanes(const separable_context *ctx, const float *line, size_t column,
    const int64_t *pixel, const size_t *channel, int64_t offset, __mmask16 mask)
{
    size_t c = ctx->channels, lane;
    float values[16];
    if (pixel[0] + offset >= 0 && pixel[15] + offset < (int64_t)ctx->width)
        return _mm512_maskz_loadu_ps(mask, line + ((ptrdiff_t)column + (ptrdiff_t)offset * (ptrdiff_t)c));
    for (lane = 0; lane < 16; ++lane)
        values[lane] =
            mask >> lane & 1 ? line[(size_t)ctx->columns[pixel[lane] + offset] * c + channel[lane]] : 0.0f;
    return _mm512_loadu_ps(values);
}

/* The horizontal group at `column` of line, any lane's tap possibly reflected, only
 * the lanes in mask. */
static __m512 horizontal_group(const separable_context *ctx, const float *line, size_t column, __mmask16 mask)
{
    size_t c = ctx->channels, radius = ctx->rx, part = column % c, channel[16], lane, k;
    int64_t pixel[16], x = (int64_t)(column / c);
    const float *kernel = ctx->kx;
    __m512 sum, pair;
    for (lane = 0; lane < 16; ++lane) {
        pixel[lane] = x;
        channel[lane] = part;
        if (++part == c) {
            part = 0;
            ++x;
        }
    }
    if (radius == 0) return horizontal_lanes(ctx, line, column, pixel, channel, 0, mask);
    if (radius <= 2) {
        pair = _mm512_add_ps(horizontal_lanes(ctx, line, column, pixel, channel, -1, mask),
            horizontal_lanes(ctx, line, column, pixel, channel, 1, mask));
        sum = _mm512_fmadd_ps(horizontal_lanes(ctx, line, column, pixel, channel, 0, mask),
            _mm512_set1_ps(kernel[radius]), _mm512_mul_ps(pair, _mm512_set1_ps(kernel[radius + 1])));
        if (radius == 2) {
            pair = _mm512_add_ps(horizontal_lanes(ctx, line, column, pixel, channel, 2, mask),
                horizontal_lanes(ctx, line, column, pixel, channel, -2, mask));
            sum = _mm512_fmadd_ps(pair, _mm512_set1_ps(kernel[4]), sum);
        }
        return sum;
    }
    sum = _mm512_setzero_ps();
    for (k = 0; k <= radius * 2; ++k)
        sum = _mm512_fmadd_ps(
            horizontal_lanes(ctx, line, column, pixel, channel, (int64_t)k - (int64_t)radius, mask),
            _mm512_set1_ps(kernel[k]), sum);
    return sum;
}

/* The group of horizontal outputs whose first element is center, in a row (or a
 * border run's copy) holding each one's every tap, only the lanes in mask loaded:
 * OpenCV's small symmetric row (radius <= 2) or its ordinary row, each lane's fused
 * sequence. */
static __m512 horizontal_masked(const separable_context *ctx, const float *center, __mmask16 mask)
{
    ptrdiff_t c = (ptrdiff_t)ctx->channels, radius = (ptrdiff_t)ctx->rx, k;
    const float *kernel = ctx->kx;
    __m512 sum = _mm512_maskz_loadu_ps(mask, center), pair;
    if (radius == 0) return sum;
    if (radius <= 2) {
        pair = _mm512_add_ps(_mm512_maskz_loadu_ps(mask, center - c), _mm512_maskz_loadu_ps(mask, center + c));
        sum = _mm512_fmadd_ps(sum, _mm512_set1_ps(kernel[radius]),
            _mm512_mul_ps(pair, _mm512_set1_ps(kernel[radius + 1])));
        if (radius == 2) {
            pair = _mm512_add_ps(_mm512_maskz_loadu_ps(mask, center + 2 * c),
                _mm512_maskz_loadu_ps(mask, center - 2 * c));
            sum = _mm512_fmadd_ps(pair, _mm512_set1_ps(kernel[4]), sum);
        }
        return sum;
    }
    sum = _mm512_setzero_ps();
    for (k = 0; k <= radius * 2; ++k)
        sum = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(mask, center + (k - radius) * c), _mm512_set1_ps(kernel[k]), sum);
    return sum;
}

/* count horizontal outputs (a multiple of 16) into out, from the elements of a row
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
        for (; count; count -= 16, center += 16, out += 16) {
            __m512 sum = _mm512_loadu_ps(center);
            if (radius > 0) {
                __m512 pair = _mm512_add_ps(_mm512_loadu_ps(center - c), _mm512_loadu_ps(center + c));
                sum = _mm512_fmadd_ps(sum, _mm512_set1_ps(kernel[radius]),
                    _mm512_mul_ps(pair, _mm512_set1_ps(kernel[radius + 1])));
                if (radius == 2) {
                    pair = _mm512_add_ps(_mm512_loadu_ps(center + 2 * c), _mm512_loadu_ps(center - 2 * c));
                    sum = _mm512_fmadd_ps(pair, _mm512_set1_ps(kernel[4]), sum);
                }
            }
            _mm512_storeu_ps(out, sum);
        }
        return;
    }
    for (; count >= 128; count -= 128, center += 128, out += 128) {
        __m512 sum0 = _mm512_setzero_ps(), sum1 = sum0, sum2 = sum0, sum3 = sum0;
        __m512 sum4 = sum0, sum5 = sum0, sum6 = sum0, sum7 = sum0;
        for (k = 0; k <= radius * 2; ++k) {
            const float *taps = center + (k - radius) * c;
            __m512 coefficient = _mm512_set1_ps(kernel[k]);
            sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(taps), coefficient, sum0);
            sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(taps + 16), coefficient, sum1);
            sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(taps + 32), coefficient, sum2);
            sum3 = _mm512_fmadd_ps(_mm512_loadu_ps(taps + 48), coefficient, sum3);
            sum4 = _mm512_fmadd_ps(_mm512_loadu_ps(taps + 64), coefficient, sum4);
            sum5 = _mm512_fmadd_ps(_mm512_loadu_ps(taps + 80), coefficient, sum5);
            sum6 = _mm512_fmadd_ps(_mm512_loadu_ps(taps + 96), coefficient, sum6);
            sum7 = _mm512_fmadd_ps(_mm512_loadu_ps(taps + 112), coefficient, sum7);
        }
        _mm512_storeu_ps(out, sum0);
        _mm512_storeu_ps(out + 16, sum1);
        _mm512_storeu_ps(out + 32, sum2);
        _mm512_storeu_ps(out + 48, sum3);
        _mm512_storeu_ps(out + 64, sum4);
        _mm512_storeu_ps(out + 80, sum5);
        _mm512_storeu_ps(out + 96, sum6);
        _mm512_storeu_ps(out + 112, sum7);
    }
    if (count >= 64) {
        __m512 sum0 = _mm512_setzero_ps(), sum1 = sum0, sum2 = sum0, sum3 = sum0;
        for (k = 0; k <= radius * 2; ++k) {
            const float *taps = center + (k - radius) * c;
            __m512 coefficient = _mm512_set1_ps(kernel[k]);
            sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(taps), coefficient, sum0);
            sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(taps + 16), coefficient, sum1);
            sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(taps + 32), coefficient, sum2);
            sum3 = _mm512_fmadd_ps(_mm512_loadu_ps(taps + 48), coefficient, sum3);
        }
        _mm512_storeu_ps(out, sum0);
        _mm512_storeu_ps(out + 16, sum1);
        _mm512_storeu_ps(out + 32, sum2);
        _mm512_storeu_ps(out + 48, sum3);
        count -= 64;
        center += 64;
        out += 64;
    }
    for (; count; count -= 16, center += 16, out += 16) {
        __m512 sum = _mm512_setzero_ps();
        for (k = 0; k <= radius * 2; ++k)
            sum = _mm512_fmadd_ps(_mm512_loadu_ps(center + (k - radius) * c), _mm512_set1_ps(kernel[k]), sum);
        _mm512_storeu_ps(out, sum);
    }
}

/* Horizontal groups [column, stop) (16-lane grid) of a row (or a border run's copy)
 * holding each output's every tap, into out (column's output): center is column's
 * element, and the lanes outside [first, end) are masked off (group_masks). */
static void horizontal_groups(const separable_context *ctx, const float *center, size_t column, size_t stop,
    size_t first, size_t end, float *out)
{
    __mmask16 head, tail;
    size_t whole, whole_end;
    if (column >= stop) return;
    group_masks(column, stop, first, end, &head, &tail, &whole, &whole_end);
    if (head) _mm512_mask_storeu_ps(out, head, horizontal_masked(ctx, center, head));
    horizontal_interior(ctx, center + whole, whole_end - whole, out + whole);
    if (tail) _mm512_mask_storeu_ps(out + whole_end, tail, horizontal_masked(ctx, center + whole_end, tail));
}

/* Horizontal groups [column, stop) of line, some lane's tap possibly reflected (a
 * border run), into out (column's output), the lanes outside [first, end) masked off:
 * their taps read the row's virtual elements [column - rx c, min(stop, end) + rx c),
 * copied once into edge (reflected_copy), and horizontal_groups runs on the copy. A
 * copy above SEPARABLE_EDGE_FLOATS takes horizontal_group per group. */
static void horizontal_run(const separable_context *ctx, const float *line, size_t column, size_t stop,
    size_t first, size_t end, float *out)
{
    float edge[SEPARABLE_EDGE_FLOATS];
    size_t reach = ctx->rx * ctx->channels, total;
    if (column >= stop) return;
    total = (end < stop ? end : stop) - column + 2 * reach;
    if (total > SEPARABLE_EDGE_FLOATS) {
        for (; column < stop; column += 16, out += 16) {
            __mmask16 mask = lane_mask(first > column ? first - column : 0, end < column + 16 ? end - column : 16);
            _mm512_mask_storeu_ps(out, mask, horizontal_group(ctx, line, column, mask));
        }
        return;
    }
    reflected_copy(ctx, line, column, total, edge);
    horizontal_groups(ctx, edge + reach, column, stop, first, end, out);
}

void cn_separable_horizontal_avx512(const separable_context *ctx, size_t y, size_t first, size_t last, float *out)
{
    size_t c = ctx->channels, row_width = ctx->width * c, fused = row_width / 8 * 8;
    size_t end = last < fused ? last : fused, column = first / 16 * 16, stop = (end + 15) / 16 * 16, head, tail;
    const float *line = ctx->src + y * row_width;
    float *row;
    if (first >= end) {
        separable_horizontal_span(ctx, y, first, last, out);
        return;
    }
    row = out - (first - column); /* column's output (the contract in separable_shared.h) */
    /* Interior groups: column >= rx * c and column + 16 <= (width - rx) * c; the
     * border runs [column, head) and [tail, stop) hold the others, each widened into
     * the interior to a block of eight groups where the segment has one. */
    border_runs(column, stop, ctx->rx * c, ctx->width > ctx->rx ? (ctx->width - ctx->rx) * c : 0, 16, 128, &head,
        &tail);
    horizontal_run(ctx, line, column, head, first, end, row);
    horizontal_groups(ctx, line + head, head, tail, first, end, row + (head - column));
    horizontal_run(ctx, line, tail, stop, first, end, row + (tail - column));
    separable_horizontal_span(ctx, y, end, last, out + (end - first));
}

/* The vertical group at `at` (relative to column first's buffer and output
 * elements) of row y, only the lanes in mask loaded: each lane's fused sequence. */
static __m512 vertical_masked(const separable_context *ctx, const float *buffer, size_t stride,
    size_t first_row, size_t y, ptrdiff_t at, __mmask16 mask)
{
    ptrdiff_t radius = (ptrdiff_t)ctx->ry, k;
    const float *kernel = ctx->ky;
    __m512 sum = _mm512_fmadd_ps(
        _mm512_maskz_loadu_ps(mask, vertical_line(ctx, buffer, stride, first_row, y, 0) + at),
        _mm512_set1_ps(kernel[radius]), _mm512_setzero_ps());
    for (k = 1; k <= radius; ++k) {
        __m512 pair = _mm512_add_ps(
            _mm512_maskz_loadu_ps(mask, vertical_line(ctx, buffer, stride, first_row, y, k) + at),
            _mm512_maskz_loadu_ps(mask, vertical_line(ctx, buffer, stride, first_row, y, -k) + at));
        sum = _mm512_fmadd_ps(pair, _mm512_set1_ps(kernel[radius + k]), sum);
    }
    return sum;
}

void cn_separable_vertical_avx512(const separable_context *ctx, const float *buffer, size_t stride,
    size_t first_row, size_t y, size_t first, size_t last, float *out)
{
    size_t row_width = ctx->width * ctx->channels, fused = row_width / 8 * 8, whole, whole_end, at, j;
    size_t end = last < fused ? last : fused, column = first / 16 * 16, stop = (end + 15) / 16 * 16;
    ptrdiff_t radius = (ptrdiff_t)ctx->ry, k, shift = (ptrdiff_t)column - (ptrdiff_t)first;
    const float *kernel = ctx->ky, *center = vertical_line(ctx, buffer, stride, first_row, y, 0);
    __m512 middle = _mm512_set1_ps(kernel[radius]);
    __mmask16 head, tail;
    if (first >= end) {
        separable_vertical_span(ctx, buffer, stride, first_row, y, first, last, out);
        return;
    }
    group_masks(column, stop, first, end, &head, &tail, &whole, &whole_end);
    if (head)
        _mm512_mask_storeu_ps(out + shift, head, vertical_masked(ctx, buffer, stride, first_row, y, shift, head));
    /* Whole groups from column + whole on (at or after first): eight groups at a time
     * (eight independent sums), then a tail of four or more groups with four, then
     * one group at a time, as horizontal_interior. */
    for (j = whole; whole_end - j >= 128; j += 128) {
        __m512 sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7;
        at = column + j - first;
        sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(center + at), middle, _mm512_setzero_ps());
        sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(center + at + 16), middle, _mm512_setzero_ps());
        sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(center + at + 32), middle, _mm512_setzero_ps());
        sum3 = _mm512_fmadd_ps(_mm512_loadu_ps(center + at + 48), middle, _mm512_setzero_ps());
        sum4 = _mm512_fmadd_ps(_mm512_loadu_ps(center + at + 64), middle, _mm512_setzero_ps());
        sum5 = _mm512_fmadd_ps(_mm512_loadu_ps(center + at + 80), middle, _mm512_setzero_ps());
        sum6 = _mm512_fmadd_ps(_mm512_loadu_ps(center + at + 96), middle, _mm512_setzero_ps());
        sum7 = _mm512_fmadd_ps(_mm512_loadu_ps(center + at + 112), middle, _mm512_setzero_ps());
        for (k = 1; k <= radius; ++k) {
            const float *below = vertical_line(ctx, buffer, stride, first_row, y, k) + at;
            const float *above = vertical_line(ctx, buffer, stride, first_row, y, -k) + at;
            __m512 coefficient = _mm512_set1_ps(kernel[radius + k]);
            sum0 = _mm512_fmadd_ps(_mm512_add_ps(_mm512_loadu_ps(below), _mm512_loadu_ps(above)), coefficient, sum0);
            sum1 = _mm512_fmadd_ps(_mm512_add_ps(_mm512_loadu_ps(below + 16), _mm512_loadu_ps(above + 16)),
                coefficient, sum1);
            sum2 = _mm512_fmadd_ps(_mm512_add_ps(_mm512_loadu_ps(below + 32), _mm512_loadu_ps(above + 32)),
                coefficient, sum2);
            sum3 = _mm512_fmadd_ps(_mm512_add_ps(_mm512_loadu_ps(below + 48), _mm512_loadu_ps(above + 48)),
                coefficient, sum3);
            sum4 = _mm512_fmadd_ps(_mm512_add_ps(_mm512_loadu_ps(below + 64), _mm512_loadu_ps(above + 64)),
                coefficient, sum4);
            sum5 = _mm512_fmadd_ps(_mm512_add_ps(_mm512_loadu_ps(below + 80), _mm512_loadu_ps(above + 80)),
                coefficient, sum5);
            sum6 = _mm512_fmadd_ps(_mm512_add_ps(_mm512_loadu_ps(below + 96), _mm512_loadu_ps(above + 96)),
                coefficient, sum6);
            sum7 = _mm512_fmadd_ps(_mm512_add_ps(_mm512_loadu_ps(below + 112), _mm512_loadu_ps(above + 112)),
                coefficient, sum7);
        }
        _mm512_storeu_ps(out + at, sum0);
        _mm512_storeu_ps(out + at + 16, sum1);
        _mm512_storeu_ps(out + at + 32, sum2);
        _mm512_storeu_ps(out + at + 48, sum3);
        _mm512_storeu_ps(out + at + 64, sum4);
        _mm512_storeu_ps(out + at + 80, sum5);
        _mm512_storeu_ps(out + at + 96, sum6);
        _mm512_storeu_ps(out + at + 112, sum7);
    }
    if (whole_end - j >= 64) {
        __m512 sum0, sum1, sum2, sum3;
        at = column + j - first;
        sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(center + at), middle, _mm512_setzero_ps());
        sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(center + at + 16), middle, _mm512_setzero_ps());
        sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(center + at + 32), middle, _mm512_setzero_ps());
        sum3 = _mm512_fmadd_ps(_mm512_loadu_ps(center + at + 48), middle, _mm512_setzero_ps());
        for (k = 1; k <= radius; ++k) {
            const float *below = vertical_line(ctx, buffer, stride, first_row, y, k) + at;
            const float *above = vertical_line(ctx, buffer, stride, first_row, y, -k) + at;
            __m512 coefficient = _mm512_set1_ps(kernel[radius + k]);
            sum0 = _mm512_fmadd_ps(_mm512_add_ps(_mm512_loadu_ps(below), _mm512_loadu_ps(above)), coefficient, sum0);
            sum1 = _mm512_fmadd_ps(_mm512_add_ps(_mm512_loadu_ps(below + 16), _mm512_loadu_ps(above + 16)),
                coefficient, sum1);
            sum2 = _mm512_fmadd_ps(_mm512_add_ps(_mm512_loadu_ps(below + 32), _mm512_loadu_ps(above + 32)),
                coefficient, sum2);
            sum3 = _mm512_fmadd_ps(_mm512_add_ps(_mm512_loadu_ps(below + 48), _mm512_loadu_ps(above + 48)),
                coefficient, sum3);
        }
        _mm512_storeu_ps(out + at, sum0);
        _mm512_storeu_ps(out + at + 16, sum1);
        _mm512_storeu_ps(out + at + 32, sum2);
        _mm512_storeu_ps(out + at + 48, sum3);
        j += 64;
    }
    for (; j < whole_end; j += 16) {
        __m512 sum;
        at = column + j - first;
        sum = _mm512_fmadd_ps(_mm512_loadu_ps(center + at), middle, _mm512_setzero_ps());
        for (k = 1; k <= radius; ++k) {
            __m512 pair = _mm512_add_ps(_mm512_loadu_ps(vertical_line(ctx, buffer, stride, first_row, y, k) + at),
                _mm512_loadu_ps(vertical_line(ctx, buffer, stride, first_row, y, -k) + at));
            sum = _mm512_fmadd_ps(pair, _mm512_set1_ps(kernel[radius + k]), sum);
        }
        _mm512_storeu_ps(out + at, sum);
    }
    if (tail) {
        at = column + whole_end - first;
        _mm512_mask_storeu_ps(out + at, tail,
            vertical_masked(ctx, buffer, stride, first_row, y, (ptrdiff_t)at, tail));
    }
    separable_vertical_span(ctx, buffer + (end - first), stride, first_row, y, end, last, out + (end - first));
}
