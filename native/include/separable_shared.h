/* Float32 separable filters matching OpenCV 4.8.0
 * modules/imgproc/src/filter.simd.hpp (separable float32 row/column filters),
 * moved from box_complete_ops.c. Shared by that baseline unit and the ISA units
 * separable_avx2.c (/arch:AVX2) and separable_avx512.c (/arch:AVX512): the
 * context, and static helpers compiled into each including unit, never inline, so
 * the linker cannot fold a VEX copy into baseline callers. Every including unit
 * uses every helper (C4505).
 * https://github.com/opencv/opencv/tree/4.8.0
 *
 * License Agreement For Open Source Computer Vision Library
 * Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
 * Copyright (C) 2009, Willow Garage Inc., all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *  * The name of the copyright holders may not be used to endorse or promote
 *    products derived from this software without specific prior written permission.
 * This software is provided by the copyright holders and contributors "as is"
 * and any express or implied warranties, including, but not limited to, the
 * implied warranties of merchantability and fitness for a particular purpose
 * are disclaimed. In no event shall the Intel Corporation or contributors be
 * liable for any direct, indirect, incidental, special, exemplary, or
 * consequential damages (including, but not limited to, procurement of
 * substitute goods or services; loss of use, data, or profits; or business
 * interruption) however caused and on any theory of liability, whether in
 * contract, strict liability, or tort (including negligence or otherwise)
 * arising in any way out of the use of this software, even if advised of
 * the possibility of such damage.
 */
#ifndef CHAINNER_SEPARABLE_SHARED_H
#define CHAINNER_SEPARABLE_SHARED_H
#include "chainner.h"
#include "fused_shared.h"
#include "isa.h"

/* isa: the level the C entry read once for the call (cn_isa_current).
 * columns[v] for the virtual pixel column v in [-rx, width + rx), and rows[v] for
 * the virtual row v in [-ry, height + ry): the column (row) that v reflects to.
 * Both are NULL when a radius exceeds box_complete_ops.c's
 * SEPARABLE_TABLE_RADIUS; each tap then reflects its own coordinate.
 * Row tiles (D9): band rows per band, block columns per block, blocks per band. */
typedef struct separable_context {
    const float *src, *kx, *ky;
    float *dst;
    size_t height, width, channels, rx, ry, lanes;
    int vertical, border;
    cn_isa_level isa;
    const int32_t *columns, *rows;
    size_t band, block, blocks;
} separable_context;

static size_t reflected(int64_t coordinate, size_t length, int border)
{
    int64_t period;
    if (length == 1) return 0;
    if (border == 1)
        return coordinate < 0 ? 0 : (uint64_t)coordinate >= length ? length - 1 : (size_t)coordinate;
    if (border == 2) {
        period = (int64_t)length * 2;
        coordinate %= period;
        if (coordinate < 0) coordinate += period;
        return (size_t)(coordinate < (int64_t)length ? coordinate : period - 1 - coordinate);
    }
    period = (int64_t)(length - 1) * 2;
    coordinate %= period;
    if (coordinate < 0) coordinate += period;
    return (size_t)(coordinate < (int64_t)length ? coordinate : period - coordinate);
}

/* The column (row) virtual coordinate v of a length reflects to: the table's
 * entry (table points at v = 0), or reflected's without a table. */
static size_t source_index(const int32_t *table, int64_t v, size_t length, int border)
{
    return table ? (size_t)table[v] : reflected(v, length, border);
}

/* Tap `offset` of horizontal output pixel x: line is src's row with the output's
 * channel added. */
static float horizontal_tap(const separable_context *ctx, const float *line, int64_t x, int64_t offset)
{
    return line[source_index(ctx->columns, x + offset, ctx->width, ctx->border) * ctx->channels];
}

/* Horizontal output (y, column) of src: B3's separable_scalar horizontal forms
 * (OpenCV's small symmetric rows, then ordinary rows), each output's sequence
 * verbatim; fused (lanes 8) below the absolute column row_width / 8 * 8. */
static float separable_horizontal_one(const separable_context *ctx, size_t y, size_t column)
{
    size_t c = ctx->channels, radius = ctx->rx, k;
    int64_t x = (int64_t)(column / c);
    const float *line = ctx->src + y * ctx->width * c + column % c;
    const float *kernel = ctx->kx;
    int fused = ctx->lanes == 8 && column < ctx->width * c / 8 * 8;
    float result;
    if (radius == 0) return horizontal_tap(ctx, line, x, 0);
    if (radius <= 2) {
        float center = horizontal_tap(ctx, line, x, 0);
        float pair = horizontal_tap(ctx, line, x, -1) + horizontal_tap(ctx, line, x, 1);
        float side = pair * kernel[radius + 1];
        result = fused ? fused_tap(center, kernel[radius], side) : center * kernel[radius] + side;
        if (radius == 2) {
            pair = horizontal_tap(ctx, line, x, 2) + horizontal_tap(ctx, line, x, -2);
            result = fused ? fused_tap(pair, kernel[4], result) : result + pair * kernel[4];
        }
        return result;
    }
    if (fused) {
        result = 0.0f;
        for (k = 0; k <= radius * 2; ++k)
            result = fused_tap(horizontal_tap(ctx, line, x, (int64_t)k - (int64_t)radius), kernel[k], result);
    } else {
        result = horizontal_tap(ctx, line, x, -(int64_t)radius) * kernel[0];
        for (k = 1; k <= radius * 2; ++k)
            result += horizontal_tap(ctx, line, x, (int64_t)k - (int64_t)radius) * kernel[k];
    }
    return result;
}

/* Tap `offset` of vertical output row y: buffer holds horizontal outputs from row
 * first_row on, stride apart, starting in the output's column. */
static float vertical_tap(const separable_context *ctx, const float *buffer, size_t stride,
    size_t first_row, size_t y, int64_t offset)
{
    size_t row = source_index(ctx->rows, (int64_t)y + offset, ctx->height, ctx->border);
    return buffer[(row - first_row) * stride];
}

/* Vertical output (y, column) from the horizontal outputs in buffer (vertical_tap):
 * B3's separable_scalar vertical forms (OpenCV's symmetric columns, the radius-1
 * unfused form), each output's sequence verbatim; fused (lanes 8) below the
 * absolute column row_width / 8 * 8. */
static float separable_vertical_one(const separable_context *ctx, const float *buffer, size_t stride,
    size_t first_row, size_t y, size_t column)
{
    size_t radius = ctx->ry, k;
    const float *kernel = ctx->ky;
    int fused = ctx->lanes == 8 && column < ctx->width * ctx->channels / 8 * 8;
    float result;
    if (radius == 1 && !fused) {
        float pair = vertical_tap(ctx, buffer, stride, first_row, y, -1) +
            vertical_tap(ctx, buffer, stride, first_row, y, 1);
        result = pair * kernel[2] + vertical_tap(ctx, buffer, stride, first_row, y, 0) * kernel[1];
        return result + 0.0f;
    }
    result = fused ? fused_tap(vertical_tap(ctx, buffer, stride, first_row, y, 0), kernel[radius], 0.0f) :
        vertical_tap(ctx, buffer, stride, first_row, y, 0) * kernel[radius] + 0.0f;
    for (k = 1; k <= radius; ++k) {
        float pair = vertical_tap(ctx, buffer, stride, first_row, y, (int64_t)k) +
            vertical_tap(ctx, buffer, stride, first_row, y, -(int64_t)k);
        result = fused ? fused_tap(pair, kernel[radius + k], result) : result + pair * kernel[radius + k];
    }
    return result;
}

/* Horizontal outputs [first, last) of row y, one at a time, into out[0 ...]. */
static void separable_horizontal_span(const separable_context *ctx, size_t y, size_t first, size_t last,
    float *out)
{
    size_t column;
    for (column = first; column < last; ++column) out[column - first] = separable_horizontal_one(ctx, y, column);
}

/* Vertical outputs [first, last) of row y, one at a time, into out[0 ...]; buffer
 * holds row first_row's horizontal output of column first. */
static void separable_vertical_span(const separable_context *ctx, const float *buffer, size_t stride,
    size_t first_row, size_t y, size_t first, size_t last, float *out)
{
    size_t column;
    for (column = first; column < last; ++column)
        out[column - first] = separable_vertical_one(ctx, buffer + (column - first), stride, first_row, y, column);
}

/* separable_avx2.c, for ISA level avx2, lanes 8 and the coordinate tables:
 * outputs [first, last) of row y into out[0 ...], inside one chunk. Groups of 8 at
 * absolute columns = 0 (mod 8) below row_width / 8 * 8; every other output through
 * the spans above. The vertical buffer is as separable_vertical_span's. */
void cn_separable_horizontal_avx2(const separable_context *ctx, size_t y, size_t first, size_t last, float *out);
void cn_separable_vertical_avx2(const separable_context *ctx, const float *buffer, size_t stride,
    size_t first_row, size_t y, size_t first, size_t last, float *out);
/* separable_avx512.c: the same for ISA level avx512, in groups of 16 at absolute
 * columns = 0 (mod 16) over the outputs of [first, last) below row_width / 8 * 8; a
 * group reaching past them loads and stores only its lanes inside (a lane mask). A
 * chunk's first group can start before first: out, and the vertical buffer's rows,
 * then lie in arrays that also hold the row's columns from first / 16 * 16 on, as
 * box_complete_ops.c's callers' do (a tile's columns start on the 16-lane grid; the
 * fallback's out and buffer are rows of whole-image arrays). */
void cn_separable_horizontal_avx512(const separable_context *ctx, size_t y, size_t first, size_t last, float *out);
void cn_separable_vertical_avx512(const separable_context *ctx, const float *buffer, size_t stride,
    size_t first_row, size_t y, size_t first, size_t last, float *out);
#endif
