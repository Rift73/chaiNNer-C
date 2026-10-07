/* C adaptation of chaiNNer-rs threshold.rs and util/bilinear.rs at
 * 6f6ead6064f81b4049d3deb803c9279c7950736b. The analytic quadrant-area branches
 * and float32 operation order are preserved. Alteration: each output detects
 * edges from its own 3x3 neighborhood, avoiding a mutable edge bitmap and
 * enabling independent, bounded-memory CPU work items.
 * https://github.com/chaiNNer-org/chaiNNer-rs
 *
 * MIT License
 * Copyright (c) 2023 Michael Schmidt
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 * C ABI, ownership checks and parallel integration are GPL-3.0-only.
 */
#include "chainner.h"
#include "chainner_ext_kernels.h"
#include "parallel.h"
#include "numeric.h"
#include <math.h>
#include <string.h>

typedef struct quadrant { float p00, p10, p01, p11; } quadrant;
typedef struct area_coefficients { float a, b, c, d; } area_coefficients;

static float rust_clamp(float value, float low, float high)
{
    value = value < low ? low : value;
    return value > high ? high : value;
}

static float sample(quadrant q, float x, float y)
{
    float y0 = q.p00 + (q.p10 - q.p00) * x;
    float y1 = q.p01 + (q.p11 - q.p01) * x;
    return y0 + (y1 - y0) * y;
}

static quadrant corner(quadrant q, float size)
{
    quadrant result = {sample(q, 0.0f, 0.0f), sample(q, size, 0.0f),
        sample(q, 0.0f, size), sample(q, size, size)};
    return result;
}

static int inside(float value)
{
    return value >= 0.0001f && value <= 1.0f - 0.0001f;
}

static int middle(area_coefficients q)
{
    return 0.25f * q.a + 0.5f * q.b + 0.5f * q.c + q.d > 0.0f;
}

static float segment_area(area_coefficients q, float x1, float x2)
{
    float xmid = (x1 + x2) * 0.5f;
    float y0 = xmid * q.b + q.d;
    float y1 = xmid * q.a + xmid * q.b + q.c + q.d;
    int zero0 = fabsf(y0) < 0.0001f, zero1 = fabsf(y1) < 0.0001f;
    int above0, above1;
    if (zero0 && zero1) {
        if (y0 == 0.0f && y1 == 0.0f) above0 = above1 = 0;
        else if (y0 == 0.0f) above0 = above1 = y1 > 0.0f;
        else if (y1 == 0.0f) above0 = above1 = y0 > 0.0f;
        else { above0 = y0 > 0.0f; above1 = y1 > 0.0f; }
    } else if (zero0) above0 = above1 = y1 > 0.0f;
    else if (zero1) above0 = above1 = y0 > 0.0f;
    else { above0 = y0 > 0.0f; above1 = y1 > 0.0f; }
    float total = x2 - x1;
    float area = above0 == above1 ? total :
        ((q.b * q.c - q.a * q.d) * logf((q.a * x2 + q.c) / (q.a * x1 + q.c)) -
            q.a * q.b * (x2 - x1)) / (q.a * q.a);
    return above0 ? area : total - area;
}

static float quadrant_area(quadrant values, float threshold)
{
    area_coefficients q = {values.p11 - values.p10 - values.p01 + values.p00,
        values.p10 - values.p00, values.p01 - values.p00, values.p00 - threshold};
    if (fabsf(q.a) < 0.0001f) {
        if (fabsf(q.c) < 0.0001f) {
            if (fabsf(q.b) < 0.0001f) return q.d > 0.0f ? 1.0f : 0.0f;
            float x = -q.d / q.b;
            float side = (x - 1.0f) * q.b + q.d;
            float area = rust_clamp(x, 0.0f, 1.0f);
            return side > 0.0f ? area : 1.0f - area;
        }
        if (fabsf(q.b) < 0.0001f) {
            float y = -q.d / q.c;
            float side = (y - 1.0f) * q.c + q.d;
            float area = rust_clamp(y, 0.0f, 1.0f);
            return side > 0.0f ? area : 1.0f - area;
        }
        float x0 = rust_clamp(-q.d / q.b, 0.0f, 1.0f);
        float x1 = rust_clamp(-(q.c + q.d) / q.b, 0.0f, 1.0f);
        if (fabsf(x0 - x1) < 0.0001f) return middle(q) ? 1.0f : 0.0f;
        float low = fminf(x0, x1), high = fmaxf(x0, x1);
        float line = (0.5f * q.b * (low * low - high * high) + q.d * (low - high)) / q.c;
        float area = 0.0f;
        area += x1 * q.b + q.d > 0.0f ? line : (high - low) - line;
        if (q.d > 0.0f) area += low;
        if (q.b + q.d > 0.0f) area += 1.0f - high;
        return area;
    }
    float i1, i2 = 0.0f;
    int count;
    if (fabsf(q.b) < 0.0001f) { count = 1; i1 = -(q.c + q.d) / (q.a + q.b); }
    else if (fabsf(q.a + q.b) < 0.0001f) { count = 1; i1 = -q.d / q.b; }
    else { count = 2; i1 = -(q.c + q.d) / (q.a + q.b); i2 = -q.d / q.b; }
    if (count == 2 && fabsf(i1 - i2) < 0.0001f) {
        float cx = (i1 + i2) * 0.5f, cy = -q.b / q.a;
        if (!inside(cx)) {
            if (!inside(cy)) return middle(q) ? 1.0f : 0.0f;
            return 0.5f * q.b + q.d > 0.0f ? cy : 1.0f - cy;
        }
        if (!inside(cy)) return 0.5f * q.c + q.d > 0.0f ? cx : 1.0f - cx;
        float area = cx * cy + (1.0f - cx) * (1.0f - cy);
        return q.d > 0.0f ? area : 1.0f - area;
    }
    if (count == 1) { if (!inside(i1)) count = 0; }
    else if (!inside(i1)) {
        if (!inside(i2)) count = 0;
        else { count = 1; i1 = i2; }
    } else if (!inside(i2)) count = 1;
    if (!count) return segment_area(q, 0.0f, 1.0f);
    if (count == 1) return segment_area(q, 0.0f, i1) + segment_area(q, i1, 1.0f);
    float low = fminf(i1, i2), high = fmaxf(i1, i2);
    return segment_area(q, 0.0f, low) + segment_area(q, low, high) + segment_area(q, high, 1.0f);
}

typedef struct threshold_context {
    const float *src;
    float *dst;
    size_t height, width, channels;
    float threshold, corner_size;
} threshold_context;

static void aa_range(void *raw, size_t begin, size_t end)
{
    const threshold_context *ctx = raw;
    for (size_t i = begin; i < end; ++i) {
        size_t pixel = i / ctx->channels, channel = i % ctx->channels;
        size_t x = pixel % ctx->width, y = pixel / ctx->width;
        size_t xs[3] = {x ? x - 1 : 0, x, x + 1 < ctx->width ? x + 1 : x};
        size_t ys[3] = {y ? y - 1 : 0, y, y + 1 < ctx->height ? y + 1 : y};
        float p[9];
        int edge = 0, center = ctx->src[i] > ctx->threshold;
        for (size_t yy = 0; yy < 3; ++yy) for (size_t xx = 0; xx < 3; ++xx) {
            float value = ctx->src[(ys[yy] * ctx->width + xs[xx]) * ctx->channels + channel];
            p[yy * 3 + xx] = value;
            if ((value > ctx->threshold) != center) edge = 1;
        }
        if (!edge) { ctx->dst[i] = center ? 1.0f : 0.0f; continue; }
        quadrant tl = {p[4], p[3], p[1], p[0]}, tr = {p[4], p[5], p[1], p[2]};
        quadrant bl = {p[4], p[3], p[7], p[6]}, br = {p[4], p[5], p[7], p[8]};
        float sum = quadrant_area(corner(tl, ctx->corner_size), ctx->threshold) +
            quadrant_area(corner(tr, ctx->corner_size), ctx->threshold);
        sum += quadrant_area(corner(bl, ctx->corner_size), ctx->threshold);
        sum += quadrant_area(corner(br, ctx->corner_size), ctx->threshold);
        ctx->dst[i] = sum * 0.25f;
    }
}

static cn_status valid_buffers(const float *src, float *dst, size_t input_count, size_t output_count)
{
    uintptr_t a = (uintptr_t)src, b = (uintptr_t)dst;
    if (!src || !dst || a % _Alignof(float) || b % _Alignof(float)) return CN_INVALID_ARGUMENT;
    if (input_count > SIZE_MAX / sizeof(float) || output_count > SIZE_MAX / sizeof(float))
        return CN_SIZE_OVERFLOW;
    size_t in_bytes = input_count * sizeof(float), out_bytes = output_count * sizeof(float);
    if (a > UINTPTR_MAX - in_bytes || b > UINTPTR_MAX - out_bytes) return CN_SIZE_OVERFLOW;
    if (a < b + out_bytes && b < a + in_bytes) return CN_INVALID_ARGUMENT;
    return CN_OK;
}

CN_EXPORT cn_status cn_threshold_aa(const float *src, float *dst,
    size_t height, size_t width, size_t channels, float threshold, float smoothness)
{
    if (!height || !width || !channels) return CN_INVALID_ARGUMENT;
    if (height > SIZE_MAX / width || height * width > SIZE_MAX / channels)
        return CN_SIZE_OVERFLOW;
    size_t count = height * width * channels;
    cn_status status = valid_buffers(src, dst, count, count);
    if (status != CN_OK) return status;
    threshold_context ctx = {src, dst, height, width, channels, threshold,
        rust_clamp(0.5f + smoothness / 2.0f, 0.5f, 1.0f)};
    return cn_parallel_for(count, 1024, aa_range, &ctx);
}

typedef struct channel_mean_context {
    const float *src;
    float *dst;
    size_t channels, block;
    int planes;
} channel_mean_context;

static void channel_mean_range(void *raw, size_t begin, size_t end)
{
    const channel_mean_context *ctx = raw;
    /* np.mean(axis=-1), divided as _methods._mean does: by an np.intp count, in float64,
       then rounded to float32. With the channel axis innermost in NumPy's iteration, each
       pixel's channels are summed pairwise: one inner call (block 0) for an aligned image,
       NPY_BUFSIZE calls (block 8192) for a buffered unaligned one. Otherwise (planes: a planar CHW view as HWC) the reduction adds whole
       channel planes in channel order onto add's identity, +0.0. */
    for (size_t i = begin; i < end; ++i) {
        const float *pixel = ctx->src + i * ctx->channels;
        float sum = 0.0f;
        if (ctx->planes)
            for (size_t k = 0; k < ctx->channels; ++k) sum += pixel[k];
        else
            sum = cn_numpy_sum_f32(pixel, ctx->channels, 1, ctx->block, 0);
        ctx->dst[i] = (float)((double)sum / (double)ctx->channels);
    }
}

CN_EXPORT cn_status cn_threshold_channel_mean(const float *src, float *dst, size_t pixels, size_t channels,
    size_t block, int planes)
{
    if (!pixels || !channels) return CN_INVALID_ARGUMENT;
    if (pixels > SIZE_MAX / channels) return CN_SIZE_OVERFLOW;
    cn_status status = valid_buffers(src, dst, pixels * channels, pixels);
    if (status != CN_OK) return status;
    channel_mean_context ctx = {src, dst, channels, block, planes};
    return cn_parallel_for(pixels, 1024, channel_mean_range, &ctx);
}

typedef struct hard_context {
    const float *src;
    float *dst;
    float threshold, maximum;
    int type, ipp;
} hard_context;

static float truncate_scalar(float value, float threshold)
{
    /* Deliberate correction: IPP's vector/scalar TRUNC paths disagreed on NaN
     * and equal signed zeros, depending on the allocated output's alignment.
     * Preserve the original scalar std::min contract consistently instead. */
    uint32_t bits, bound;
    memcpy(&bits, &value, sizeof(bits));
    memcpy(&bound, &threshold, sizeof(bound));
    uint32_t mask = UINT32_C(0) - (uint32_t)(value > threshold);
    bits = (bits & ~mask) | (bound & mask);
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static void hard_range(void *raw, size_t begin, size_t end)
{
    const hard_context *ctx = raw;
    /* IPP's strict greater-than filter uses nextafter for the inclusive cutoff.
     * Keeping that operation preserves its infinities and NaN threshold rules. */
    float next = nextafterf(ctx->threshold, CN_INFINITY_F);
    for (size_t i = begin; i < end; ++i) {
        float value = ctx->src[i], result;
        switch (ctx->type) {
        case 0: result = value > ctx->threshold ? ctx->maximum : 0.0f; break;
        case 1: result = value <= ctx->threshold ? ctx->maximum : 0.0f; break;
        case 2: result = truncate_scalar(value, ctx->threshold); break;
        case 3:
            result = ctx->ipp ? (value < next ? 0.0f : value) :
                (value > ctx->threshold ? value : 0.0f);
            break;
        default:
            result = ctx->ipp ? (value > ctx->threshold ? 0.0f : value) :
                (value <= ctx->threshold ? value : 0.0f);
            break;
        }
        ctx->dst[i] = result;
    }
}

CN_EXPORT cn_status cn_threshold_hard(const float *src, float *dst, size_t count,
    int type, float threshold, float maximum, int ipp)
{
    if (!count || type < 0 || type > 4 || (ipp != 0 && ipp != 1)) return CN_INVALID_ARGUMENT;
    cn_status status = valid_buffers(src, dst, count, count);
    if (status != CN_OK) return status;
    hard_context ctx = {src, dst, threshold, maximum, type, ipp};
    return cn_parallel_for(count, 65536, hard_range, &ctx);
}
