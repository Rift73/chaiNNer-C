/* C box filters matching OpenCV 5.0.0.
 * The fixed-point divisor and SIMD/scalar rounding contracts are derived from
 * modules/imgproc/src/box_filter.simd.hpp (ColumnSum specializations) and
 * modules/imgproc/src/filter.simd.hpp (separable float32 row/column filters).
 * https://github.com/opencv/opencv/tree/5.0.0
 * Changed implementation: bounded uint64 row sums and independent columns,
 * with multiplicities for replicate borders rather than padded ring buffers.
 * C ABI/validation/threadpool integration are GPL-3.0-only.
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
#include "chainner.h"
#include "parallel.h"
#include "numeric.h"
#include "separable.h"
#include "separable_shared.h"
#include <limits.h>
#include <math.h>
#include <stdlib.h>

typedef struct box_kernel_context {
    float *output;
    size_t height, width;
    float value, edge_x, edge_y;
    int fractional_x, fractional_y;
} box_kernel_context;

static void box_kernel_range(void *opaque, size_t begin, size_t end)
{
    const box_kernel_context *ctx = opaque;
    for (size_t i = begin; i < end; ++i) {
        size_t x = i % ctx->width, y = i / ctx->width;
        float value = ctx->value;
        /* Preserve the original ndarray operation order: normalize, vertical
         * edge scaling, then horizontal edge scaling (including corners). */
        if (ctx->fractional_y) {
            if (y == 0) value *= ctx->edge_y;
            if (y == ctx->height - 1) value *= ctx->edge_y;
        }
        if (ctx->fractional_x) {
            if (x == 0) value *= ctx->edge_x;
            if (x == ctx->width - 1) value *= ctx->edge_x;
        }
        ctx->output[i] = value;
    }
}

/* chaiNNer's GPL-3.0-only get_kernel_2d arithmetic, expressed as one C pass. */
CN_EXPORT cn_status cn_box_kernel_2d(float *output, size_t count,
    double radius_x, double radius_y)
{
    if (!output || (uintptr_t)output % _Alignof(float) ||
        !(radius_x >= 0 && radius_x <= 1000) ||
        !(radius_y >= 0 && radius_y <= 1000)) return CN_INVALID_ARGUMENT;
    size_t width = 2 * (size_t)ceil(radius_x) + 1;
    size_t height = 2 * (size_t)ceil(radius_y) + 1;
    if (count != width * height) return CN_INVALID_ARGUMENT;
    if ((uintptr_t)output > UINTPTR_MAX - count * sizeof(float)) return CN_SIZE_OVERFLOW;
    double dx = radius_x - floor(radius_x), dy = radius_y - floor(radius_y);
    float denominator = (float)((2.0 * radius_y + 1.0) * (2.0 * radius_x + 1.0));
    box_kernel_context ctx = {output, height, width, 1.0f / denominator,
        (float)dx, (float)dy, dx != 0, dy != 0};
    return cn_parallel_for(count, 16384, box_kernel_range, &ctx);
}

typedef struct mean_context {
    const uint8_t *src;
    uint8_t *dst;
    uint64_t *rows;
    size_t h, w, rx, ry;
    uint64_t area;
    uint64_t divisor, delta;
    double scale;
} mean_context;

/* Box areas above INT_MAX overflow the original kernel-area expression.
 * Keep mathematically defined means there, with exact integer numerators up
 * to 255*(INT_MAX^2), without a compiler-specific 128-bit integer type. */
typedef struct wide_sum { uint64_t low, high; } wide_sum;

static wide_sum multiply_small(uint64_t value, uint32_t factor)
{
    uint64_t a = (value & UINT32_MAX) * factor;
    uint64_t b = (value >> 32) * factor;
    wide_sum result;
    result.low = a + (b << 32);
    result.high = (b >> 32) + (result.low < a);
    return result;
}

static void add_word(wide_sum *sum, uint64_t value)
{
    uint64_t old = sum->low;
    sum->low += value;
    sum->high += sum->low < old;
}

static void subtract_word(wide_sum *sum, uint64_t value)
{
    sum->high -= sum->low < value;
    sum->low -= value;
}

static void add_wide(wide_sum *sum, wide_sum value)
{
    sum->high += value.high;
    add_word(sum, value.low);
}

static uint8_t rounded_mean(wide_sum total, uint64_t area)
{
    unsigned int low = 0, high = 256;
    /* Both effective dimensions are odd. The denominator is odd, so exact
     * half-integer ties cannot occur for an integer input sum. */
    add_word(&total, area / 2);
    while (high - low > 1) {
        unsigned int mid = low + (high - low) / 2;
        wide_sum candidate = multiply_small(area, mid);
        if (candidate.high < total.high ||
            (candidate.high == total.high && candidate.low <= total.low)) low = mid;
        else high = mid;
    }
    return (uint8_t)low;
}

static void mean_wide_columns(void *raw, size_t begin, size_t end)
{
    const mean_context *ctx = (const mean_context *)raw;
    size_t x;
    for (x = begin; x < end; ++x) {
        size_t y, before = ctx->ry < ctx->h ? ctx->ry : ctx->h;
        wide_sum sum = multiply_small(ctx->rows[x], (uint32_t)ctx->ry);
        for (y = 0; y < before; ++y) add_word(&sum, ctx->rows[y * ctx->w + x]);
        if (ctx->ry > before) add_wide(&sum,
            multiply_small(ctx->rows[(ctx->h - 1) * ctx->w + x], (uint32_t)(ctx->ry - before)));
        for (y = 0; y < ctx->h; ++y) {
            size_t old = y >= ctx->ry ? y - ctx->ry : 0;
            size_t next = ctx->ry >= ctx->h - 1 - y ? ctx->h - 1 : y + ctx->ry;
            add_word(&sum, ctx->rows[next * ctx->w + x]);
            ctx->dst[y * ctx->w + x] = rounded_mean(sum, ctx->area);
            subtract_word(&sum, ctx->rows[old * ctx->w + x]);
        }
    }
}

static void mean_rows(void *raw, size_t begin, size_t end)
{
    const mean_context *ctx = (const mean_context *)raw;
    size_t y;
    for (y = begin; y < end; ++y) {
        const uint8_t *src = ctx->src + y * ctx->w;
        uint64_t *dst = ctx->rows + y * ctx->w;
        size_t x, last = ctx->rx < ctx->w - 1 ? ctx->rx : ctx->w - 1;
        uint64_t sum = (uint64_t)ctx->rx * src[0];
        for (x = 0; x <= last; ++x) sum += src[x];
        if (ctx->rx > last) sum += (uint64_t)(ctx->rx - last) * src[ctx->w - 1];
        dst[0] = sum;
        for (x = 1; x < ctx->w; ++x) {
            size_t old = x > ctx->rx ? x - ctx->rx - 1 : 0;
            size_t next = ctx->rx >= ctx->w - 1 - x ? ctx->w - 1 : x + ctx->rx;
            sum += src[next]; sum -= src[old];
            dst[x] = sum;
        }
    }
}

static void mean_columns(void *raw, size_t begin, size_t end)
{
    const mean_context *ctx = (const mean_context *)raw;
    size_t x;
    for (x = begin; x < end; ++x) {
        size_t y, before = ctx->ry < ctx->h ? ctx->ry : ctx->h;
        uint64_t sum = (uint64_t)ctx->ry * ctx->rows[x];
        for (y = 0; y < before; ++y) sum += ctx->rows[y * ctx->w + x];
        if (ctx->ry > before) sum += (uint64_t)(ctx->ry - before) * ctx->rows[(ctx->h - 1) * ctx->w + x];
        for (y = 0; y < ctx->h; ++y) {
            size_t old = y >= ctx->ry ? y - ctx->ry : 0;
            size_t next = ctx->ry >= ctx->h - 1 - y ? ctx->h - 1 : y + ctx->ry;
            uint64_t total = sum + ctx->rows[next * ctx->w + x];
            unsigned int value;
            if (ctx->area <= 256) value = (unsigned int)((total + ctx->delta) * ctx->divisor >> 23);
            else if (ctx->area <= (UINT32_C(1) << 23) && x < ctx->w / 8 * 8) {
                float mean = (float)total * (float)ctx->scale;
                value = (unsigned int)nearbyintf(mean);
            } else value = (unsigned int)nearbyint((double)total * ctx->scale);
            ctx->dst[y * ctx->w + x] = (uint8_t)(value > 255 ? 255 : value);
            sum = total - ctx->rows[old * ctx->w + x];
        }
    }
}

CN_EXPORT cn_status cn_box_mean_u8(const uint8_t *src, uint8_t *dst,
    size_t height, size_t width, size_t radius)
{
    size_t count, rx, ry, kw, kh;
    uint64_t area;
    uintptr_t a = (uintptr_t)src, b = (uintptr_t)dst;
    mean_context ctx;
    cn_status status;
    if (!src || !dst || !height || !width || height > INT_MAX || width > INT_MAX ||
        radius > (INT_MAX - 1) / 2)
        return CN_INVALID_ARGUMENT;
    if (width > SIZE_MAX / height || width * height > SIZE_MAX / sizeof(uint64_t))
        return CN_SIZE_OVERFLOW;
    count = width * height;
    if (a > UINTPTR_MAX - count || b > UINTPTR_MAX - count) return CN_SIZE_OVERFLOW;
    if (a < b + count && b < a + count) return CN_INVALID_ARGUMENT;
    rx = width == 1 ? 0 : radius; ry = height == 1 ? 0 : radius;
    kw = rx * 2 + 1; kh = ry * 2 + 1;
    area = (uint64_t)kw * (uint64_t)kh;
    ctx.src = src; ctx.dst = dst; ctx.h = height; ctx.w = width;
    ctx.rx = rx; ctx.ry = ry; ctx.area = area; ctx.scale = 1.0 / (double)area;
    ctx.divisor = UINT64_C(1) << 23; ctx.delta = 0;
    if (area <= 256 && area != 1) {
        double factor = (double)(UINT64_C(1) << 23) / (double)area;
        ctx.divisor = (uint64_t)floor(factor); ctx.delta = area / 2;
        if (factor - (double)ctx.divisor < 0.5) ++ctx.delta;
        else ++ctx.divisor;
    }
    ctx.rows = (uint64_t *)malloc(count * sizeof(uint64_t));
    if (!ctx.rows) return CN_ALLOCATION_FAILED;
    status = cn_parallel_for(height, 1 + 16384 / width, mean_rows, &ctx);
    if (status == CN_OK) status = cn_parallel_for(width, 1 + 16384 / height,
        area > INT_MAX ? mean_wide_columns : mean_columns, &ctx);
    free(ctx.rows);
    return status;
}

/* Float32 separable filters (OpenCV 5.0.0 filter.simd.hpp: small symmetric rows,
 * ordinary larger rows, and symmetric columns; each output's sequence in
 * separable_shared.h) in row tiles (SP4b D9). A tile is a band of output rows by a
 * block of row columns: it writes the horizontal outputs of the rows its band's
 * vertical taps read into a stack buffer of SEPARABLE_TILE_FLOATS floats, then
 * its vertical outputs from the buffer. Constants from Task 3's geometry sweep
 * (gaussian and parallel-branches at K = 1); they never depend on the machine or
 * the ISA level. */
#define SEPARABLE_TILE_FLOATS 32768 /* 128 KiB of the tile function's stack */
#define SEPARABLE_MIN_BAND 32 /* B0: a band holds B = max(B0, 4 ry) rows */
#define SEPARABLE_TABLE_RADIUS 4096 /* larger radii: no coordinate tables */
#define SEPARABLE_GRAIN 1024 /* the fallback's two passes (B3's partition) */

static void horizontal_row(const separable_context *ctx, size_t y, size_t first, size_t last, float *out)
{
#if defined(_MSC_VER) && defined(_M_X64)
    if (ctx->isa >= CN_ISA_AVX512 && ctx->lanes == 8 && ctx->columns) {
        cn_separable_horizontal_avx512(ctx, y, first, last, out);
        return;
    }
    if (ctx->isa >= CN_ISA_AVX2 && ctx->lanes == 8 && ctx->columns) {
        cn_separable_horizontal_avx2(ctx, y, first, last, out);
        return;
    }
#endif
    separable_horizontal_span(ctx, y, first, last, out);
}

static void vertical_row(const separable_context *ctx, const float *buffer, size_t stride, size_t first_row,
    size_t y, size_t first, size_t last, float *out)
{
#if defined(_MSC_VER) && defined(_M_X64)
    if (ctx->isa >= CN_ISA_AVX512 && ctx->lanes == 8 && ctx->rows) {
        cn_separable_vertical_avx512(ctx, buffer, stride, first_row, y, first, last, out);
        return;
    }
    if (ctx->isa >= CN_ISA_AVX2 && ctx->lanes == 8 && ctx->rows) {
        cn_separable_vertical_avx2(ctx, buffer, stride, first_row, y, first, last, out);
        return;
    }
#endif
    separable_vertical_span(ctx, buffer, stride, first_row, y, first, last, out);
}

/* Tiles [begin, end): tile t is band t / blocks, block t % blocks. The band's
 * vertical taps read rows [lo, hi), at most R = min(height, B + 2 ry) of them, so
 * the buffer holds R x W <= SEPARABLE_TILE_FLOATS. */
static void separable_tiles_range(void *raw, size_t begin, size_t end)
{
    const separable_context *ctx = (const separable_context *)raw;
    /* 64-byte aligned: a block starts on the 16-float grid and is a multiple of 16
     * floats wide (all but the last block), so the AVX-512 rows' whole-group loads and
     * stores of the buffer each stay inside one cache line. */
    _Alignas(64) float buffer[SEPARABLE_TILE_FLOATS];
    size_t row_width = ctx->width * ctx->channels, tile, y;
    for (tile = begin; tile < end; ++tile) {
        size_t first_row = tile / ctx->blocks * ctx->band, first = tile % ctx->blocks * ctx->block;
        size_t last_row = ctx->height - first_row < ctx->band ? ctx->height : first_row + ctx->band;
        size_t last = row_width - first < ctx->block ? row_width : first + ctx->block;
        size_t lo = first_row < ctx->ry ? 0 : first_row - ctx->ry;
        size_t hi = ctx->height - last_row < ctx->ry ? ctx->height : last_row + ctx->ry;
        size_t stride = last - first;
        for (y = lo; y < hi; ++y) horizontal_row(ctx, y, first, last, buffer + (y - lo) * stride);
        for (y = first_row; y < last_row; ++y)
            vertical_row(ctx, buffer, stride, lo, y, first, last, ctx->dst + y * row_width + first);
    }
}

/* The fallback: B3's two passes over an image-size temporary, the horizontal one
 * from src into dst, then the vertical one (vertical set) from src = the temporary,
 * walked row segment by row segment. */
static void separable_range(void *raw, size_t begin, size_t end)
{
    const separable_context *ctx = (const separable_context *)raw;
    size_t row_width = ctx->width * ctx->channels;
    while (begin < end) {
        size_t y = begin / row_width, first = begin % row_width;
        size_t last = row_width - first < end - begin ? row_width : first + (end - begin);
        float *out = ctx->dst + y * row_width + first;
        if (ctx->vertical) vertical_row(ctx, ctx->src + first, row_width, 0, y, first, last, out);
        else horizontal_row(ctx, y, first, last, out);
        begin += last - first;
    }
}

/* table[i] = reflected(i - radius) for the virtual coordinates [-radius,
 * length + radius); reflected(v) is v inside [0, length). */
static void reflected_table(int32_t *table, size_t length, size_t radius, int border)
{
    size_t i;
    for (i = 0; i < radius; ++i) {
        table[i] = (int32_t)reflected((int64_t)i - (int64_t)radius, length, border);
        table[radius + length + i] = (int32_t)reflected((int64_t)(length + i), length, border);
    }
    for (i = 0; i < length; ++i) table[radius + i] = (int32_t)i;
}

/* D9's geometry: band rows B = min(height, max(B0, 4 ry)), buffer rows
 * R = min(height, B + 2 ry), block columns W = min(16 ceil(row_width / 16),
 * 16 floor(SEPARABLE_TILE_FLOATS / (16 R))). Sets band, block and blocks and
 * returns the tile count, or 0 when R > SEPARABLE_TILE_FLOATS / 16 (the
 * fallback). */
static size_t separable_tiles(separable_context *ctx)
{
    size_t row_width = ctx->width * ctx->channels, band = 4 * ctx->ry, rows, whole, most;
    if (band < SEPARABLE_MIN_BAND) band = SEPARABLE_MIN_BAND;
    if (band > ctx->height) band = ctx->height;
    rows = ctx->height - band < 2 * ctx->ry ? ctx->height : band + 2 * ctx->ry;
    if (rows > SEPARABLE_TILE_FLOATS / 16) return 0;
    whole = (row_width + 15) / 16 * 16;
    most = SEPARABLE_TILE_FLOATS / (16 * rows) * 16;
    ctx->band = band;
    ctx->block = whole < most ? whole : most;
    ctx->blocks = (row_width + ctx->block - 1) / ctx->block;
    return (ctx->height + band - 1) / band * ctx->blocks;
}

/* Both entries, after their checks: src, dst, kernels, sizes, radii, lanes, border
 * and isa set. Coordinate tables (one malloc) unless a radius exceeds
 * SEPARABLE_TABLE_RADIUS; then the tiles, or the fallback (no tables, or
 * R > SEPARABLE_TILE_FLOATS / 16) with its temporary. */
static cn_status separable_run(separable_context *ctx)
{
    size_t count = ctx->height * ctx->width * ctx->channels, tiles = 0;
    int32_t *tables = NULL;
    float *dst = ctx->dst, *temporary;
    cn_status status;
    ctx->vertical = 0;
    ctx->columns = ctx->rows = NULL;
    if (ctx->rx <= SEPARABLE_TABLE_RADIUS && ctx->ry <= SEPARABLE_TABLE_RADIUS) {
        size_t span = ctx->width + 2 * ctx->rx;
        tables = (int32_t *)malloc((span + ctx->height + 2 * ctx->ry) * sizeof(int32_t));
        if (!tables) return CN_ALLOCATION_FAILED;
        reflected_table(tables, ctx->width, ctx->rx, ctx->border);
        reflected_table(tables + span, ctx->height, ctx->ry, ctx->border);
        ctx->columns = tables + ctx->rx;
        ctx->rows = tables + span + ctx->ry;
        tiles = separable_tiles(ctx);
    }
    if (tiles) {
        status = cn_parallel_for(tiles, 1, separable_tiles_range, ctx);
        free(tables);
        return status;
    }
    temporary = (float *)malloc(count * sizeof(float));
    if (!temporary) {
        free(tables);
        return CN_ALLOCATION_FAILED;
    }
    ctx->dst = temporary;
    status = cn_parallel_for(count, SEPARABLE_GRAIN, separable_range, ctx);
    if (status == CN_OK) {
        ctx->src = temporary; ctx->dst = dst; ctx->vertical = 1;
        status = cn_parallel_for(count, SEPARABLE_GRAIN, separable_range, ctx);
    }
    free(temporary);
    free(tables);
    return status;
}

static void coefficients(double radius, float *kernel, size_t count)
{
    size_t i;
    float sum, edge = (float)(radius - floor(radius));
    for (i = 0; i < count; ++i) kernel[i] = 1.0f;
    if (radius != floor(radius)) kernel[0] = kernel[count - 1] = edge;
    /* np.sum of the fresh contiguous kernel row: one unbuffered call (block 0). */
    sum = cn_numpy_sum_f32(kernel, count, 1, 0, 0);
    for (i = 0; i < count; ++i) kernel[i] /= sum;
}

CN_EXPORT cn_status cn_box_separable(const float *src, float *dst,
    size_t height, size_t width, size_t channels, double radius_x, double radius_y, size_t lanes)
{
    size_t count, bytes, rx, ry;
    uintptr_t a = (uintptr_t)src, b = (uintptr_t)dst;
    float kx[2001], ky[2001];
    separable_context ctx;
    if (!src || !dst || !height || !width || !channels ||
        height > INT_MAX || width > INT_MAX || (lanes != 4 && lanes != 8) ||
        !isfinite(radius_x) || !isfinite(radius_y) || radius_x < 0 || radius_y < 0 ||
        radius_x > 1000 || radius_y > 1000 || a % _Alignof(float) || b % _Alignof(float))
        return CN_INVALID_ARGUMENT;
    if (height > SIZE_MAX / width || height * width > SIZE_MAX / sizeof(float) / channels)
        return CN_SIZE_OVERFLOW;
    count = height * width * channels; bytes = count * sizeof(float);
    if (a > UINTPTR_MAX - bytes || b > UINTPTR_MAX - bytes) return CN_SIZE_OVERFLOW;
    if (a < b + bytes && b < a + bytes) return CN_INVALID_ARGUMENT;
    rx = (size_t)ceil(radius_x); ry = (size_t)ceil(radius_y);
    coefficients(radius_x, kx, rx * 2 + 1); coefficients(radius_y, ky, ry * 2 + 1);
    ctx.src = src; ctx.dst = dst; ctx.kx = kx; ctx.ky = ky;
    ctx.height = height; ctx.width = width; ctx.channels = channels;
    ctx.rx = rx; ctx.ry = ry; ctx.lanes = lanes; ctx.border = 4;
    ctx.isa = cn_isa_current();
    return separable_run(&ctx);
}

cn_status cn_symmetric_filter_f32(const float *src, float *dst, size_t height,
    size_t width, size_t channels, const float *kx, size_t nx,
    const float *ky, size_t ny, int border, size_t lanes)
{
    uintptr_t a = (uintptr_t)src, b = (uintptr_t)dst;
    if (!src || !dst || !kx || !ky || !height || !width || !channels
        || height > INT_MAX || width > INT_MAX || (lanes != 4 && lanes != 8)
        || (border != 1 && border != 2 && border != 4)
        || !nx || !ny || !(nx & 1) || !(ny & 1) || nx > INT_MAX || ny > INT_MAX
        || a % _Alignof(float) || b % _Alignof(float)
        || (uintptr_t)kx % _Alignof(float) || (uintptr_t)ky % _Alignof(float))
        return CN_INVALID_ARGUMENT;
    if (height > SIZE_MAX / width || height * width > SIZE_MAX / sizeof(float) / channels)
        return CN_SIZE_OVERFLOW;
    size_t count = height * width * channels, bytes = count * sizeof(float);
    if (a > UINTPTR_MAX - bytes || b > UINTPTR_MAX - bytes) return CN_SIZE_OVERFLOW;
    if (a < b + bytes && b < a + bytes) return CN_INVALID_ARGUMENT;
    separable_context ctx = {src, kx, ky, dst, height, width, channels,
                            nx / 2, ny / 2, lanes, 0, border, cn_isa_current(), NULL, NULL, 0, 0, 0};
    return separable_run(&ctx);
}
