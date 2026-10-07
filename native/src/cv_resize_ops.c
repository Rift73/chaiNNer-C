/* Altered C extraction of OpenCV 4.8.0 modules/imgproc/src/resize.cpp:
 * float32 area decimation and baseline linear/cubic interpolation paths.
 * Copyright (C) 2000-2008, 2017, Intel Corporation, all rights reserved.
 * Copyright (C) 2009, Willow Garage Inc., all rights reserved.
 * Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * * Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * The name of the copyright holders may not be used to endorse or promote
 *   products derived from this software without specific prior written permission.
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
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct area_tap { size_t index; float weight; } area_tap;
typedef struct area_line { size_t start, count; } area_line;
typedef struct resize_context {
    const float *src;
    float *out;
    size_t h, w, channels, dh, dw, ix, iy;
    const area_tap *xt, *yt;
    const area_line *xl, *yl;
} resize_context;

static cn_status validate(const float *src, float *out, size_t h, size_t w,
    size_t channels, size_t dh, size_t dw)
{
    if (!src || !out || !h || !w || !dh || !dw || !channels || channels > 4 ||
        (uintptr_t)src % _Alignof(float) || (uintptr_t)out % _Alignof(float))
        return CN_INVALID_ARGUMENT;
    if (h > INT32_MAX || w > INT32_MAX || dh > INT32_MAX || dw > INT32_MAX ||
        h > SIZE_MAX / w || h * w > SIZE_MAX / sizeof(float) / channels ||
        dh > SIZE_MAX / dw || dh * dw > SIZE_MAX / sizeof(float) / channels)
        return CN_SIZE_OVERFLOW;
    size_t in_bytes = h * w * channels * sizeof(float);
    size_t out_bytes = dh * dw * channels * sizeof(float);
    uintptr_t a = (uintptr_t)src, b = (uintptr_t)out;
    if (a > UINTPTR_MAX - in_bytes || b > UINTPTR_MAX - out_bytes)
        return CN_SIZE_OVERFLOW;
    if (a < b + out_bytes && b < a + in_bytes) return CN_INVALID_ARGUMENT;
    return CN_OK;
}

static void integer_area(void *opaque, size_t begin, size_t end)
{
    const resize_context *ctx = opaque;
    size_t row = ctx->dw * ctx->channels, area = ctx->ix * ctx->iy;
    float scale = 1.0f / (float)area;
    for (size_t i = begin; i < end; ++i) {
        size_t y = i / row, x = (i % row) / ctx->channels, ch = i % ctx->channels;
        size_t origin = (y * ctx->iy * ctx->w + x * ctx->ix) * ctx->channels + ch;
        float sum = 0.0f;
        /* OpenCV's baseline SSE2 2x2 path uses paired row sums for C1/C4.
         * The scalar tail instead uses the four-tap left-associated sum. */
        int vector = ctx->ix == 2 && ctx->iy == 2 &&
            (ctx->channels == 4 || (ctx->channels == 1 && i % row < row / 4 * 4));
        if (vector) {
            const float *p = ctx->src + origin;
            size_t c = ctx->channels, step = ctx->w * c;
            sum = (p[0] + p[c]) + (p[step] + p[step + c]);
        } else {
            size_t k = 0;
            for (; k + 4 <= area; k += 4) {
                float values[4];
                for (size_t t = 0; t < 4; ++t) {
                    size_t offset = ((k + t) / ctx->ix * ctx->w + (k + t) % ctx->ix) * ctx->channels;
                    values[t] = ctx->src[origin + offset];
                }
                sum += ((values[0] + values[1]) + values[2]) + values[3];
            }
            for (; k < area; ++k)
                sum += ctx->src[origin + (k / ctx->ix * ctx->w + k % ctx->ix) * ctx->channels];
        }
        ctx->out[i] = sum * scale;
    }
}

static void make_area_table(size_t source, size_t dest, double scale,
    area_tap *taps, area_line *lines)
{
    size_t count = 0;
    for (size_t d = 0; d < dest; ++d) {
        double left = (double)d * scale, right = left + scale;
        double cell = fmin(scale, (double)source - left);
        size_t first = (size_t)ceil(left), last = (size_t)floor(right);
        if (last >= source) last = source - 1;
        if (first > last) first = last;
        lines[d].start = count;
        if ((double)first - left > 1e-3) {
            taps[count].index = first - 1;
            taps[count++].weight = (float)(((double)first - left) / cell);
        }
        for (size_t s = first; s < last; ++s) {
            taps[count].index = s;
            taps[count++].weight = (float)(1.0 / cell);
        }
        if (right - (double)last > 1e-3) {
            taps[count].index = last;
            taps[count++].weight = (float)(fmin(fmin(right - (double)last, 1.0), cell) / cell);
        }
        lines[d].count = count - lines[d].start;
    }
}

static void fractional_area(void *opaque, size_t begin, size_t end)
{
    const resize_context *ctx = opaque;
    size_t row = ctx->dw * ctx->channels;
    for (size_t i = begin; i < end; ++i) {
        size_t y = i / row, x = (i % row) / ctx->channels, ch = i % ctx->channels;
        area_line xl = ctx->xl[x], yl = ctx->yl[y];
        float sum = 0.0f;
        for (size_t ky = 0; ky < yl.count; ++ky) {
            area_tap yt = ctx->yt[yl.start + ky];
            const float *src = ctx->src + yt.index * ctx->w * ctx->channels + ch;
            float horizontal = 0.0f;
            for (size_t kx = 0; kx < xl.count; ++kx) {
                area_tap xt = ctx->xt[xl.start + kx];
                horizontal += src[xt.index * ctx->channels] * xt.weight;
            }
            /* Canonical OpenCV single-stripe order: only image row zero
             * starts by adding to +0. Later rows start with the first product.
             * OpenCV repeats the +0 initialization at scheduler-dependent
             * stripe boundaries, changing underflow zero signs. Keep output
             * deterministic across worker counts and overlapping calls. */
            float product = yt.weight * horizontal;
            if (ky == 0 && y != 0) sum = product;
            else sum += product;
        }
        ctx->out[i] = sum;
    }
}

CN_EXPORT cn_status cn_cv_resize_area(const float *src, float *out,
    size_t h, size_t w, size_t channels, size_t dh, size_t dw)
{
    cn_status status = validate(src, out, h, w, channels, dh, dw);
    if (status != CN_OK) return status;
    if (dh > h || dw > w) return CN_INVALID_ARGUMENT;
    if (dh == h && dw == w) {
        memcpy(out, src, h * w * channels * sizeof(float));
        return CN_OK;
    }
    double sx = 1.0 / ((double)dw / (double)w);
    double sy = 1.0 / ((double)dh / (double)h);
    size_t ix = (size_t)round(sx), iy = (size_t)round(sy);
    resize_context ctx = {src, out, h, w, channels, dh, dw, ix, iy, NULL, NULL, NULL, NULL};
    if (fabs(sx - (double)ix) < DBL_EPSILON && fabs(sy - (double)iy) < DBL_EPSILON)
        return cn_parallel_for(dh * dw * channels, 16384, integer_area, &ctx);
    if (h > SIZE_MAX - w || h + w > SIZE_MAX / 2 / sizeof(area_tap) ||
        dh > SIZE_MAX - dw || dh + dw > SIZE_MAX / sizeof(area_line))
        return CN_SIZE_OVERFLOW;
    area_tap *taps = malloc((h + w) * 2 * sizeof(*taps));
    area_line *lines = malloc((dh + dw) * sizeof(*lines));
    if (!taps || !lines) { free(taps); free(lines); return CN_ALLOCATION_FAILED; }
    ctx.xt = taps; ctx.yt = taps + w * 2;
    ctx.xl = lines; ctx.yl = lines + dw;
    make_area_table(w, dw, sx, taps, lines);
    make_area_table(h, dh, sy, taps + w * 2, lines + dw);
    status = cn_parallel_for(dh * dw * channels, 16384, fractional_area, &ctx);
    free(lines); free(taps);
    return status;
}

static void linear_range(void *opaque, size_t begin, size_t end)
{
    const resize_context *ctx = opaque;
    double sx = 1.0 / ((double)ctx->dw / (double)ctx->w);
    double sy = 1.0 / ((double)ctx->dh / (double)ctx->h);
    size_t row = ctx->dw * ctx->channels;
    for (size_t i = begin; i < end; ++i) {
        size_t y = i / row, x = (i % row) / ctx->channels, ch = i % ctx->channels;
        float fx = (float)(((double)x + 0.5) * sx - 0.5);
        float fy = (float)(((double)y + 0.5) * sy - 0.5);
        int64_t ix = (int64_t)floorf(fx), iy = (int64_t)floorf(fy);
        fx -= (float)ix; fy -= (float)iy;
        if (ix < 0) { ix = 0; fx = 0.0f; }
        if (ix >= (int64_t)ctx->w - 1) { ix = (int64_t)ctx->w - 1; fx = 0.0f; }
        int64_t y0 = iy < 0 ? 0 : iy >= (int64_t)ctx->h ? (int64_t)ctx->h - 1 : iy;
        int64_t y1 = iy + 1 < 0 ? 0 : iy + 1 >= (int64_t)ctx->h ? (int64_t)ctx->h - 1 : iy + 1;
        size_t o0 = ((size_t)y0 * ctx->w + (size_t)ix) * ctx->channels + ch;
        size_t o1 = ((size_t)y1 * ctx->w + (size_t)ix) * ctx->channels + ch;
        float a0 = 1.0f - fx, b0 = 1.0f - fy;
        float r0 = ctx->src[o0], r1 = ctx->src[o1];
        if ((size_t)ix + 1 < ctx->w) {
            r0 = r0 * a0 + ctx->src[o0 + ctx->channels] * fx;
            r1 = r1 * a0 + ctx->src[o1 + ctx->channels] * fx;
        }
        ctx->out[i] = r0 * b0 + r1 * fy;
    }
}

/* The public bridge only uses this source-visible baseline when IPP is off.
 * This is not a substitute for IPP's different linear interpolation arithmetic. */
CN_EXPORT cn_status cn_cv_resize_linear(const float *src, float *out,
    size_t h, size_t w, size_t channels, size_t dh, size_t dw)
{
    cn_status status = validate(src, out, h, w, channels, dh, dw);
    if (status != CN_OK) return status;
    if (dh == h && dw == w) {
        memcpy(out, src, h * w * channels * sizeof(float));
        return CN_OK;
    }
    if (h == dh * 2 && w == dw * 2)
        return cn_cv_resize_area(src, out, h, w, channels, dh, dw);
    resize_context ctx = {src, out, h, w, channels, dh, dw, 0, 0, NULL, NULL, NULL, NULL};
    return cn_parallel_for(dh * dw * channels, 16384, linear_range, &ctx);
}

static void cubic_coefficients(float x, float *coefficients)
{
    const float a = -0.75f;
    coefficients[0] = ((a * (x + 1) - 5 * a) * (x + 1) + 8 * a) * (x + 1) - 4 * a;
    coefficients[1] = ((a + 2) * x - (a + 3)) * x * x + 1;
    coefficients[2] = ((a + 2) * (1 - x) - (a + 3)) * (1 - x) * (1 - x) + 1;
    coefficients[3] = 1.0f - coefficients[0] - coefficients[1] - coefficients[2];
}

static size_t clamp_index(int64_t i, size_t count)
{
    return i < 0 ? 0 : i >= (int64_t)count ? count - 1 : (size_t)i;
}

static void cubic_range(void *opaque, size_t begin, size_t end)
{
    const resize_context *ctx = opaque;
    double sx = 1.0 / ((double)ctx->dw / (double)ctx->w);
    double sy = 1.0 / ((double)ctx->dh / (double)ctx->h);
    size_t row = ctx->dw * ctx->channels;
    for (size_t i = begin; i < end; ++i) {
        size_t y = i / row, x = (i % row) / ctx->channels, ch = i % ctx->channels;
        float fx = (float)(((double)x + 0.5) * sx - 0.5);
        float fy = (float)(((double)y + 0.5) * sy - 0.5);
        int64_t ix = (int64_t)floorf(fx), iy = (int64_t)floorf(fy);
        fx -= (float)ix; fy -= (float)iy;
        float alpha[4], beta[4], horizontal[4];
        cubic_coefficients(fx, alpha);
        cubic_coefficients(fy, beta);
        int border = ix < 1 || ix + 2 >= (int64_t)ctx->w;
        for (size_t ky = 0; ky < 4; ++ky) {
            const float *src = ctx->src + clamp_index(iy + (int64_t)ky - 1, ctx->h) * ctx->w * ctx->channels + ch;
            float sum = src[clamp_index(ix - 1, ctx->w) * ctx->channels] * alpha[0];
            if (border) sum = 0.0f + sum;
            for (size_t kx = 1; kx < 4; ++kx)
                sum += src[clamp_index(ix + (int64_t)kx - 1, ctx->w) * ctx->channels] * alpha[kx];
            horizontal[ky] = sum;
        }
        /* Baseline SSE2 uses nested multiply/add; only its final scalar tail
         * has the left-associated C expression. No FMA is used by that build. */
        if (i % row < row / 4 * 4)
            ctx->out[i] = horizontal[0] * beta[0] + (horizontal[1] * beta[1] +
                (horizontal[2] * beta[2] + horizontal[3] * beta[3]));
        else
            ctx->out[i] = ((horizontal[0] * beta[0] + horizontal[1] * beta[1]) +
                horizontal[2] * beta[2]) + horizontal[3] * beta[3];
    }
}

CN_EXPORT cn_status cn_cv_resize_cubic(const float *src, float *out,
    size_t h, size_t w, size_t channels, size_t dh, size_t dw)
{
    cn_status status = validate(src, out, h, w, channels, dh, dw);
    if (status != CN_OK) return status;
    if (dh == h && dw == w) {
        memcpy(out, src, h * w * channels * sizeof(float));
        return CN_OK;
    }
    resize_context ctx = {src, out, h, w, channels, dh, dw, 0, 0, NULL, NULL, NULL, NULL};
    return cn_parallel_for(dh * dw * channels, 16384, cubic_range, &ctx);
}
