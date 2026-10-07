/* Altered standalone C adaptation of Pillow 9.2.0:
 * src/libImaging/Geometry.c affine transforms and byte samplers;
 * src/libImaging/Convert.c and ImagingUtils.h alpha conversion.
 * Source: https://github.com/python-pillow/Pillow/tree/9.2.0
 * Interleaved buffers and the chaiNNer shared CPU pool replace Imaging objects.
 *
 * The Python Imaging Library (PIL) is
 * Copyright (c) 1997-2011 by Secret Labs AB
 * Copyright (c) 1995-2011 by Fredrik Lundh
 * Pillow is the friendly PIL fork. It is
 * Copyright (c) 2010-2022 by Alex Clark and contributors
 *
 * Like PIL, Pillow is licensed under the open source HPND License:
 * By obtaining, using, and/or copying this software and/or its associated
 * documentation, you agree that you have read, understood, and will comply
 * with the following terms and conditions:
 * Permission to use, copy, modify, and distribute this software and its
 * associated documentation for any purpose and without fee is hereby granted,
 * provided that the above copyright notice appears in all copies, and that
 * both that copyright notice and this permission notice appear in supporting
 * documentation, and that the name of Secret Labs AB or the author not be
 * used in advertising or publicity pertaining to distribution of the software
 * without specific, written prior permission.
 * SECRET LABS AB AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
 * SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS.
 * IN NO EVENT SHALL SECRET LABS AB OR THE AUTHOR BE LIABLE FOR ANY SPECIAL,
 * INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
 * LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
 * PERFORMANCE OF THIS SOFTWARE.
 */
#include "chainner.h"
#include "parallel.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct rotate_context {
    const uint8_t *source;
    float *out;
    uint8_t *premultiplied;
    size_t height, width, channels, out_width;
    double matrix[6];
    uint8_t fill[4];
    int interpolation, quarter, fixed, undo_alpha;
    uint32_t fixed_matrix[6];
    const double *row_starts;
} rotate_context;

static void premultiply_range(void *opaque, size_t begin, size_t end)
{
    const rotate_context *ctx = opaque;
    size_t channels = ctx->channels;
    for (size_t p = begin; p < end; ++p) {
        const uint8_t *src = ctx->source + p * channels;
        uint8_t *out = ctx->premultiplied + p * channels;
        unsigned int alpha = src[channels - 1];
        for (size_t c = 0; c + 1 < channels; ++c) {
            unsigned int temporary = src[c] * alpha + 128;
            out[c] = (uint8_t)(((temporary >> 8) + temporary) >> 8);
        }
        out[channels - 1] = (uint8_t)alpha;
    }
}

static size_t clip_index(int64_t index, size_t count)
{
    return index < 0 ? 0 : (uint64_t)index >= count ? count - 1 : (size_t)index;
}

static double cubic(double v1, double v2, double v3, double v4, double d)
{
    double p1 = v2;
    double p2 = -v1 + v3;
    double p3 = 2.0 * (v1 - v2) + v3 - v4;
    double p4 = -v1 + v2 - v3 + v4;
    return p1 + d * (p2 + d * (p3 + d * p4));
}

static void interpolate(const rotate_context *ctx, double x, double y, uint8_t *pixel)
{
    if (!(x >= 0.0 && x < (double)ctx->width && y >= 0.0 && y < (double)ctx->height))
        return;
    x -= 0.5;
    y -= 0.5;
    int64_t ix = (int64_t)floor(x), iy = (int64_t)floor(y);
    double dx = x - (double)ix, dy = y - (double)iy;
    size_t ccount = ctx->channels;
    if (ctx->interpolation == 2) {
        size_t x0 = clip_index(ix, ctx->width), x1 = clip_index(ix + 1, ctx->width);
        size_t y0 = clip_index(iy, ctx->height);
        size_t y1 = iy + 1 >= 0 && (uint64_t)(iy + 1) < ctx->height ? (size_t)(iy + 1) : y0;
        for (size_t c = 0; c < ccount; ++c) {
            const uint8_t *r0 = ctx->source + y0 * ctx->width * ccount;
            const uint8_t *r1 = ctx->source + y1 * ctx->width * ccount;
            double v1 = r0[x0 * ccount + c] + (r0[x1 * ccount + c] - r0[x0 * ccount + c]) * dx;
            double v2 = r1[x0 * ccount + c] + (r1[x1 * ccount + c] - r1[x0 * ccount + c]) * dx;
            pixel[c] = (uint8_t)(v1 + (v2 - v1) * dy);
        }
    } else {
        --ix;
        --iy;
        size_t xx[4], yy[4];
        for (size_t i = 0; i < 4; ++i) xx[i] = clip_index(ix + (int64_t)i, ctx->width);
        yy[0] = clip_index(iy, ctx->height);
        for (size_t i = 1; i < 4; ++i) {
            int64_t yi = iy + (int64_t)i;
            yy[i] = yi >= 0 && (uint64_t)yi < ctx->height ? (size_t)yi : yy[i - 1];
        }
        for (size_t c = 0; c < ccount; ++c) {
            double values[4];
            for (size_t i = 0; i < 4; ++i) {
                const uint8_t *row = ctx->source + yy[i] * ctx->width * ccount;
                values[i] = cubic(row[xx[0] * ccount + c], row[xx[1] * ccount + c],
                    row[xx[2] * ccount + c], row[xx[3] * ccount + c], dx);
            }
            double result = cubic(values[0], values[1], values[2], values[3], dy);
            pixel[c] = result <= 0.0 ? 0 : result >= 255.0 ? 255 : (uint8_t)result;
        }
    }
}

static void write_pixel(const rotate_context *ctx, size_t pixel_index, const uint8_t *pixel)
{
    unsigned int alpha = pixel[ctx->channels - 1];
    for (size_t c = 0; c < ctx->channels; ++c) {
        unsigned int value = pixel[c];
        if (ctx->undo_alpha && c + 1 < ctx->channels && alpha != 0 && alpha != 255) {
            value = 255 * value / alpha;
            if (value > 255) value = 255;
        }
        ctx->out[pixel_index * ctx->channels + c] = (float)value / 255.0f;
    }
}

static void rotate_range(void *opaque, size_t begin, size_t end)
{
    const rotate_context *ctx = opaque;
    const double *a = ctx->matrix;
    for (size_t y = begin; y < end; ++y) {
        uint32_t fixed_x = ctx->fixed_matrix[2] + (uint32_t)y * ctx->fixed_matrix[1];
        uint32_t fixed_y = ctx->fixed_matrix[5] + (uint32_t)y * ctx->fixed_matrix[4];
        double xx = ctx->row_starts ? ctx->row_starts[y * 2] : 0.0;
        double yy = ctx->row_starts ? ctx->row_starts[y * 2 + 1] : 0.0;
        for (size_t x = 0; x < ctx->out_width; ++x) {
            uint8_t pixel[4];
            memcpy(pixel, ctx->fill, ctx->channels);
            if (ctx->quarter >= 0) {
                size_t sy = y, sx = x;
                if (ctx->quarter == 1) { sy = x; sx = ctx->width - 1 - y; }
                else if (ctx->quarter == 2) { sy = ctx->height - 1 - y; sx = ctx->width - 1 - x; }
                else if (ctx->quarter == 3) { sy = ctx->height - 1 - x; sx = y; }
                memcpy(pixel, ctx->source + (sy * ctx->width + sx) * ctx->channels, ctx->channels);
            } else if (ctx->interpolation == 0) {
                if (ctx->fixed) {
                    /* Unsigned arithmetic defines the original Windows 16.16
                     * wrap behavior without signed integer overflow. */
                    uint32_t sx = fixed_x >> 16, sy = fixed_y >> 16;
                    if (!(fixed_x & UINT32_C(0x80000000)) && !(fixed_y & UINT32_C(0x80000000)) &&
                        sx < ctx->width && sy < ctx->height)
                        memcpy(pixel, ctx->source + ((size_t)sy * ctx->width + sx) * ctx->channels, ctx->channels);
                    fixed_x += ctx->fixed_matrix[0];
                    fixed_y += ctx->fixed_matrix[3];
                } else {
                    if (xx >= 0.0 && xx < (double)ctx->width && yy >= 0.0 && yy < (double)ctx->height)
                        memcpy(pixel, ctx->source + ((size_t)yy * ctx->width + (size_t)xx) * ctx->channels, ctx->channels);
                    xx += a[0];
                    yy += a[3];
                }
            } else {
                double xin = (double)x + 0.5, yin = (double)y + 0.5;
                interpolate(ctx, a[0] * xin + a[1] * yin + a[2],
                    a[3] * xin + a[4] * yin + a[5], pixel);
            }
            write_pixel(ctx, y * ctx->out_width + x, pixel);
        }
    }
}

static int check_fixed(const double *a, size_t x, size_t y)
{
    return fabs((double)x * a[0] + (double)y * a[1] + a[2]) < 32768.0 &&
        fabs((double)x * a[3] + (double)y * a[4] + a[5]) < 32768.0;
}

CN_EXPORT cn_status cn_rotate_affine_u8(const uint8_t *source, float *out,
    size_t height, size_t width, size_t channels, size_t out_height, size_t out_width,
    const double *matrix, int interpolation, const uint8_t *fill, int quarter)
{
    if (!source || !out || !matrix || !fill || !height || !width || !out_height || !out_width ||
        channels < 1 || channels > 4 || (interpolation != 0 && interpolation != 2 && interpolation != 3) ||
        quarter < -1 || quarter > 3) return CN_INVALID_ARGUMENT;
    if (height > INT32_MAX || width > INT32_MAX || out_height > INT32_MAX || out_width > INT32_MAX ||
        width > SIZE_MAX / channels / height ||
        out_width > SIZE_MAX / sizeof(float) / channels / out_height ||
        out_height > SIZE_MAX / 2 / sizeof(double)) return CN_SIZE_OVERFLOW;
    if (quarter >= 0 && (out_height != (quarter % 2 ? width : height) ||
        out_width != (quarter % 2 ? height : width))) return CN_INVALID_ARGUMENT;
    for (size_t i = 0; i < 6; ++i) {
        if (!isfinite(matrix[i])) return CN_INVALID_ARGUMENT;
        if (i % 3 != 2 && fabs(matrix[i]) > 1.0) return CN_INVALID_ARGUMENT;
    }
    rotate_context ctx = {0};
    ctx.source = source;
    ctx.out = out;
    ctx.height = height;
    ctx.width = width;
    ctx.channels = channels;
    ctx.out_width = out_width;
    ctx.interpolation = interpolation;
    ctx.quarter = quarter;
    ctx.undo_alpha = quarter < 0 && interpolation != 0 && (channels == 2 || channels == 4);
    memcpy(ctx.matrix, matrix, sizeof(ctx.matrix));
    memcpy(ctx.fill, fill, channels);
    double *starts = NULL;
    cn_status status = CN_OK;
    if (ctx.undo_alpha) {
        ctx.premultiplied = malloc(height * width * channels);
        if (!ctx.premultiplied) return CN_ALLOCATION_FAILED;
        status = cn_parallel_for(height * width, 65536, premultiply_range, &ctx);
        if (status != CN_OK) goto cleanup;
        ctx.source = ctx.premultiplied;
    }
    if (quarter < 0 && interpolation == 0) {
        ctx.fixed = (matrix[1] != 0.0 || matrix[3] != 0.0) &&
            check_fixed(matrix, 0, 0) && check_fixed(matrix, out_width, out_height) &&
            check_fixed(matrix, 0, out_height) && check_fixed(matrix, out_width, 0);
        if (ctx.fixed) {
            for (size_t i = 0; i < 6; ++i) {
                double value = matrix[i];
                if (i == 2) { value += matrix[0] * 0.5; value += matrix[1] * 0.5; }
                if (i == 5) { value += matrix[3] * 0.5; value += matrix[4] * 0.5; }
                ctx.fixed_matrix[i] = (uint32_t)(int64_t)floor(value * 65536.0 + 0.5);
            }
        } else {
            starts = malloc(out_height * 2 * sizeof(double));
            if (!starts) { status = CN_ALLOCATION_FAILED; goto cleanup; }
            double xo = matrix[2] + matrix[1] * 0.5 + matrix[0] * 0.5;
            double yo = matrix[5] + matrix[4] * 0.5 + matrix[3] * 0.5;
            for (size_t y = 0; y < out_height; ++y) {
                starts[y * 2] = xo;
                starts[y * 2 + 1] = yo;
                xo += matrix[1];
                yo += matrix[4];
            }
            ctx.row_starts = starts;
        }
    }
    size_t grain = 65536 / out_width;
    status = cn_parallel_for(out_height, grain ? grain : 1, rotate_range, &ctx);
cleanup:
    free(starts);
    free(ctx.premultiplied);
    return status;
}
