/* Helpers of the separable filter's ISA units (separable_avx2.c, separable_avx512.c):
 * the horizontal border runs' reflected copy and the vertical rows, static helpers
 * compiled into each including unit, never inline. Every including unit uses every
 * helper (C4505). Copyright (C) 2000-2008 Intel Corporation; (C) 2009 Willow Garage
 * Inc. BSD notice in separable_shared.h. */
#ifndef CHAINNER_SEPARABLE_ISA_SHARED_H
#define CHAINNER_SEPARABLE_ISA_SHARED_H
#include "separable_shared.h"
#include <string.h>

/* The most floats a horizontal border run's reflected copy holds (16 KiB of the run
 * function's stack); a larger run takes its unit's per-group path. */
#define SEPARABLE_EDGE_FLOATS 4096

/* edge[i] = line's virtual element column - rx c + i, i < total: the elements
 * inside the row are the row's own (one copy, edge[before ... beyond)), each one
 * outside it the same channel of the pixel the column table reflects its pixel to. */
static void reflected_copy(const separable_context *ctx, const float *line, size_t column, size_t total,
    float *edge)
{
    size_t c = ctx->channels, reach = ctx->rx * c, part = column % c, i;
    size_t before = column < reach ? reach - column : 0, beyond = ctx->width * c + reach - column;
    int64_t x = (int64_t)(column / c) - (int64_t)ctx->rx;
    if (beyond > total) beyond = total;
    for (i = 0; i < before; ++i) {
        edge[i] = line[(size_t)ctx->columns[x] * c + part];
        if (++part == c) {
            part = 0;
            ++x;
        }
    }
    memcpy(edge + before, line + (column + before - reach), (beyond - before) * sizeof(float));
    for (i = beyond, x = (int64_t)ctx->width, part = 0; i < total; ++i) {
        edge[i] = line[(size_t)ctx->columns[x] * c + part];
        if (++part == c) {
            part = 0;
            ++x;
        }
    }
}

/* buffer's row for vertical tap `offset` of row y, at column first. */
static const float *vertical_line(const separable_context *ctx, const float *buffer, size_t stride,
    size_t first_row, size_t y, ptrdiff_t offset)
{
    return buffer + ((size_t)ctx->rows[(ptrdiff_t)y + offset] - first_row) * stride;
}
#endif
