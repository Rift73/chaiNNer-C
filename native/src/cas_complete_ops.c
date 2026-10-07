/* CAS luminance and double-precision morphology.
 * Morphology is an altered C extraction of OpenCV 4.8.0 morph.simd.hpp's
 * MorphRowFilter, MorphColumnFilter, and MorphFilter double/NoVec paths.
 *
 * Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
 * Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

typedef struct luminance_context {
    const float *src;
    double *out;
    size_t channels;
    int fused;
} luminance_context;

static void luminance_range(void *opaque, size_t begin, size_t end)
{
    const luminance_context *ctx = opaque;
    for (size_t i = begin; i < end; ++i) {
        const float *pixel = ctx->src + i * ctx->channels;
        double sum = 0.0;
        if (ctx->fused) {
            sum = fma((double)pixel[0], 0.2126, sum);
            sum = fma((double)pixel[1], 0.7152, sum);
            sum = fma((double)pixel[2], 0.0722, sum);
        } else {
            sum += (double)pixel[0] * 0.2126;
            sum += (double)pixel[1] * 0.7152;
            sum += (double)pixel[2] * 0.0722;
        }
        ctx->out[i] = sum;
    }
}

CN_EXPORT cn_status cn_cas_luminance(const float *src, double *out,
    size_t count, size_t channels, int fused)
{
    if (!src || !out || channels < 3 || (fused != 0 && fused != 1))
        return CN_INVALID_ARGUMENT;
    if (channels > SIZE_MAX / sizeof(float) || count > SIZE_MAX / sizeof(double) ||
        count > SIZE_MAX / sizeof(float) / channels) return CN_SIZE_OVERFLOW;
    luminance_context ctx = {src, out, channels, fused};
    return cn_parallel_for(count, 65536, luminance_range, &ctx);
}

typedef struct extrema_context {
    const double *src, *row_low, *row_high;
    double *low, *high;
    const uint8_t *kernel;
    size_t h, w, kh, kw;
} extrema_context;

static double minimum(double a, double b) { return b < a ? b : a; }
static double maximum(double a, double b) { return a < b ? b : a; }

static double sample(const double *src, size_t h, size_t w,
    int64_t y, int64_t x, int high)
{
    if (y < 0 || x < 0 || y >= (int64_t)h || x >= (int64_t)w)
        return high ? -DBL_MAX : DBL_MAX;
    return src[(size_t)y * w + (size_t)x];
}

static void rectangle_rows(void *opaque, size_t begin, size_t end)
{
    const extrema_context *ctx = opaque;
    for (size_t i = begin; i < end; ++i) {
        size_t y = i / ctx->w, x = i % ctx->w;
        int64_t left = (int64_t)x - (int64_t)(ctx->kw / 2);
        /* The first output of each scalar pair folds the common taps first,
         * then the leftmost tap. Do not regroup comparisons at NaN or zero. */
        int paired = ctx->kw > 1 && !(x & 1) && x + 1 < ctx->w;
        size_t first = paired ? 1 : 0;
        double low = sample(ctx->src, ctx->h, ctx->w, (int64_t)y, left + (int64_t)first, 0);
        double high = sample(ctx->src, ctx->h, ctx->w, (int64_t)y, left + (int64_t)first, 1);
        for (size_t k = first + 1; k < ctx->kw; ++k) {
            low = minimum(low, sample(ctx->src, ctx->h, ctx->w, (int64_t)y, left + (int64_t)k, 0));
            high = maximum(high, sample(ctx->src, ctx->h, ctx->w, (int64_t)y, left + (int64_t)k, 1));
        }
        if (paired) {
            low = minimum(low, sample(ctx->src, ctx->h, ctx->w, (int64_t)y, left, 0));
            high = maximum(high, sample(ctx->src, ctx->h, ctx->w, (int64_t)y, left, 1));
        }
        ctx->low[i] = low;
        ctx->high[i] = high;
    }
}

static void rectangle_columns(void *opaque, size_t begin, size_t end)
{
    const extrema_context *ctx = opaque;
    for (size_t i = begin; i < end; ++i) {
        size_t y = i / ctx->w, x = i % ctx->w;
        int64_t top = (int64_t)y - (int64_t)(ctx->kh / 2);
        int paired = ctx->kh > 1 && !(y & 1) && y + 1 < ctx->h;
        size_t first = paired ? 1 : 0;
        double low = sample(ctx->row_low, ctx->h, ctx->w, top + (int64_t)first, (int64_t)x, 0);
        double high = sample(ctx->row_high, ctx->h, ctx->w, top + (int64_t)first, (int64_t)x, 1);
        for (size_t k = first + 1; k < ctx->kh; ++k) {
            low = minimum(low, sample(ctx->row_low, ctx->h, ctx->w, top + (int64_t)k, (int64_t)x, 0));
            high = maximum(high, sample(ctx->row_high, ctx->h, ctx->w, top + (int64_t)k, (int64_t)x, 1));
        }
        if (paired) {
            low = minimum(low, sample(ctx->row_low, ctx->h, ctx->w, top, (int64_t)x, 0));
            high = maximum(high, sample(ctx->row_high, ctx->h, ctx->w, top, (int64_t)x, 1));
        }
        ctx->low[i] = low;
        ctx->high[i] = high;
    }
}

static void masked_extrema(void *opaque, size_t begin, size_t end)
{
    const extrema_context *ctx = opaque;
    for (size_t i = begin; i < end; ++i) {
        size_t y = i / ctx->w, x = i % ctx->w;
        double low = 0.0, high = 0.0;
        int initialized = 0;
        for (size_t ky = 0; ky < ctx->kh; ++ky) {
            for (size_t kx = 0; kx < ctx->kw; ++kx) {
                if (!ctx->kernel[ky * ctx->kw + kx]) continue;
                int64_t sy = (int64_t)y + (int64_t)ky - (int64_t)(ctx->kh / 2);
                int64_t sx = (int64_t)x + (int64_t)kx - (int64_t)(ctx->kw / 2);
                double a = sample(ctx->src, ctx->h, ctx->w, sy, sx, 0);
                double b = sample(ctx->src, ctx->h, ctx->w, sy, sx, 1);
                if (!initialized) { low = a; high = b; initialized = 1; }
                else { low = minimum(low, a); high = maximum(high, b); }
            }
        }
        ctx->low[i] = low;
        ctx->high[i] = high;
    }
}

CN_EXPORT cn_status cn_cas_extrema_f64(const double *src, double *low, double *high,
    size_t h, size_t w, const uint8_t *kernel, size_t kh, size_t kw)
{
    if (!src || !low || !high || !kernel || !h || !w || !kh || !kw)
        return CN_INVALID_ARGUMENT;
    if (h > INT32_MAX || w > INT32_MAX || kh > INT32_MAX || kw > INT32_MAX ||
        h > SIZE_MAX / w || h * w > SIZE_MAX / (2 * sizeof(double)) || kh > SIZE_MAX / kw)
        return CN_SIZE_OVERFLOW;
    size_t active = 0;
    for (size_t k = 0; k < kh * kw; ++k) active += kernel[k] != 0;
    if (!active) return CN_INVALID_ARGUMENT;
    extrema_context ctx = {src, NULL, NULL, low, high, kernel, h, w, kh, kw};
    if (active != kh * kw) return cn_parallel_for(h * w, 16384, masked_extrema, &ctx);
    double *scratch = malloc(h * w * 2 * sizeof(double));
    if (!scratch) return CN_ALLOCATION_FAILED;
    ctx.low = scratch;
    ctx.high = scratch + h * w;
    cn_status status = cn_parallel_for(h * w, 16384, rectangle_rows, &ctx);
    if (status == CN_OK) {
        ctx.row_low = ctx.low;
        ctx.row_high = ctx.high;
        ctx.low = low;
        ctx.high = high;
        status = cn_parallel_for(h * w, 16384, rectangle_columns, &ctx);
    }
    free(scratch);
    return status;
}
