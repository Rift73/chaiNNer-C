/* DPID: rapid, detail-preserving image downscaling (Weber et al., SIGGRAPH Asia 2016).
 *
 * Altered C adaptation of https://github.com/mergian/dpid main.cu (kernelGuidance,
 * calcAverage and kernelDownsampling): float32 images of one to four channels in place of
 * 8-bit BGR, the guidance and the output kept in float (no 8-bit truncation), the distance
 * taken over the colour channels and divided by the square root of their count (for three
 * channels on 0..1, the reference's division by sqrt(3 * 255^2) on 0..255), an alpha
 * channel either carried with the colour weights (a premultiplied image) or downscaled as
 * its own one-channel DPID, and the native pool in place of CUDA warps. The surrounding C
 * ABI, validation and pool integration are GPL-3.0-only.
 *
 * Copyright (c) 2016, Nicolas Weber, Sandra C. Amend / GCC / TU-Darmstadt
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the <organization> nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include "chainner.h"
#include "chainner_ext_kernels.h"
#include "parallel.h"
#include "run_kernels.h"
#include "scratch.h"
#include <math.h>

/* One output column's (or row's) footprint along its axis: source samples [start, start +
 * count) and their coverage, main.cu's contribution() factors for that axis. */
typedef struct dpid_span {
    size_t start, count;
    const float *weights;
} dpid_span;

/* main.cu's Local along one axis, in double: output i covers [i * p, min((i + 1) * p,
 * source)) with p = source / target, and sample x weighs 1 - (s - x) where it starts before
 * s and 1 - (x + 1 - e) where it ends after e (both for a sample that holds the whole span,
 * as a growing side's do). The spans tile [0, source), so their counts sum to at most
 * source + target - 1: weights holds source + target values. */
static void axis_spans(size_t source, size_t target, dpid_span *spans, float *weights)
{
    double pitch = (double)source / (double)target;
    for (size_t i = 0; i < target; ++i) {
        double s = (double)i * pitch, e = fmin((double)(i + 1) * pitch, (double)source);
        size_t start = (size_t)floor(s), end = (size_t)ceil(e);
        spans[i].start = start;
        spans[i].count = end - start;
        spans[i].weights = weights;
        for (size_t x = start; x < end; ++x) {
            double f = 1.0;
            if ((double)x < s) f *= 1.0 - (s - (double)x);
            if ((double)x + 1.0 > e) f *= 1.0 - ((double)x + 1.0 - e);
            *weights++ = (float)f;
        }
    }
}

/* mode: the channel count, or 5 for RGBA whose alpha is its own one-channel DPID. power: 0
 * for lambda 0 (every distance weighs 1), 1 for lambda 1 (the distance itself), else 2
 * (powf of the squared distance with half of lambda). */
typedef struct dpid_context {
    const float *source;
    float *guidance, *out;
    const dpid_span *columns, *rows;
    size_t width, target_width, target_height;
    int mode, power;
    float half_lambda;
} dpid_context;

/* kernelGuidance for output rows [begin, end): each output pixel's coverage-weighted mean
 * of its footprint, every channel on its own. A footprint row sums in float, the rows in
 * double. */
CN_FORCE_INLINE void guidance_rows(const dpid_context *ctx, size_t begin, size_t end,
    size_t channels)
{
    size_t stride = ctx->width * channels;
    for (size_t oy = begin; oy < end; ++oy) {
        const dpid_span *row = &ctx->rows[oy];
        float *guidance = ctx->guidance + oy * ctx->target_width * channels;
        for (size_t ox = 0; ox < ctx->target_width; ++ox, guidance += channels) {
            const dpid_span *column = &ctx->columns[ox];
            double sum[4] = {0.0, 0.0, 0.0, 0.0}, total = 0.0;
            for (size_t j = 0; j < row->count; ++j) {
                const float *pixel = ctx->source + (row->start + j) * stride +
                    column->start * channels;
                float line[4] = {0.0f, 0.0f, 0.0f, 0.0f}, coverage = 0.0f;
                for (size_t i = 0; i < column->count; ++i, pixel += channels) {
                    float f = column->weights[i];
                    for (size_t c = 0; c < channels; ++c) line[c] += pixel[c] * f;
                    coverage += f;
                }
                double f = row->weights[j];
                for (size_t c = 0; c < channels; ++c) sum[c] += f * line[c];
                total += f * coverage;
            }
            for (size_t c = 0; c < channels; ++c) guidance[c] = (float)(sum[c] / total);
        }
    }
}

static void guidance_range(void *opaque, size_t begin, size_t end)
{
    const dpid_context *ctx = opaque;
    switch (ctx->mode) {
    case 1: guidance_rows(ctx, begin, end, 1); break;
    case 2: guidance_rows(ctx, begin, end, 2); break;
    case 3: guidance_rows(ctx, begin, end, 3); break;
    /* RGBA in both alpha modes: a channel's guidance does not depend on the others. */
    default: guidance_rows(ctx, begin, end, 4); break;
    }
}

/* calcAverage: the (1 2 1) x (1 2 1) tent of the guidance around (ox, oy), renormalised
 * where it leaves the image. */
CN_FORCE_INLINE void tent(const dpid_context *ctx, size_t ox, size_t oy, size_t channels,
    float *average)
{
    size_t width = ctx->target_width;
    double sum[4] = {0.0, 0.0, 0.0, 0.0}, total = 0.0;
    for (size_t y = oy ? oy - 1 : 0; y <= oy + 1 && y < ctx->target_height; ++y) {
        double ky = y == oy ? 2.0 : 1.0;
        for (size_t x = ox ? ox - 1 : 0; x <= ox + 1 && x < width; ++x) {
            double k = ky * (x == ox ? 2.0 : 1.0);
            const float *g = ctx->guidance + (y * width + x) * channels;
            for (size_t c = 0; c < channels; ++c) sum[c] += k * g[c];
            total += k;
        }
    }
    for (size_t c = 0; c < channels; ++c) average[c] = (float)(sum[c] / total);
}

/* main.cu's lambda() of a normalised distance, given its square: d^lambda is
 * (d^2)^(lambda / 2), so no square root unless lambda is 1. */
CN_FORCE_INLINE float distance_weight(const dpid_context *ctx, float squared)
{
    if (ctx->power == 0) return 1.0f;
    if (ctx->power == 1) return sqrtf(squared);
    return powf(squared, ctx->half_lambda);
}

/* kernelDownsampling for output rows [begin, end): per output pixel the tent average, then
 * the footprint's pixels weighted by coverage x (distance to the average)^lambda; a weight
 * sum of 0 gives the average. The distance runs over the colour channels (all but RGBA's
 * alpha) and is divided by the square root of their count. RGBA's alpha takes the colour
 * weights, or with separate its own weights from its own distance. A footprint row sums in
 * float, the rows in double. */
CN_FORCE_INLINE void weighted_rows(const dpid_context *ctx, size_t begin, size_t end,
    size_t channels, int separate)
{
    size_t colour = channels == 4 ? 3 : channels, stride = ctx->width * channels;
    float inverse = 1.0f / (float)colour;
    for (size_t oy = begin; oy < end; ++oy) {
        const dpid_span *row = &ctx->rows[oy];
        float *out = ctx->out + oy * ctx->target_width * channels;
        for (size_t ox = 0; ox < ctx->target_width; ++ox, out += channels) {
            const dpid_span *column = &ctx->columns[ox];
            float average[4];
            tent(ctx, ox, oy, channels, average);
            double sum[4] = {0.0, 0.0, 0.0, 0.0}, total = 0.0, alpha_total = 0.0;
            for (size_t j = 0; j < row->count; ++j) {
                const float *pixel = ctx->source + (row->start + j) * stride +
                    column->start * channels;
                float line[4] = {0.0f, 0.0f, 0.0f, 0.0f}, weights = 0.0f, alpha_weights = 0.0f;
                for (size_t i = 0; i < column->count; ++i, pixel += channels) {
                    float squared = 0.0f;
                    for (size_t c = 0; c < colour; ++c) {
                        float d = average[c] - pixel[c];
                        squared += d * d;
                    }
                    float w = distance_weight(ctx, squared * inverse) * column->weights[i];
                    for (size_t c = 0; c < colour; ++c) line[c] += pixel[c] * w;
                    weights += w;
                    if (channels == 4 && separate) {
                        float d = average[3] - pixel[3];
                        float a = distance_weight(ctx, d * d) * column->weights[i];
                        line[3] += pixel[3] * a;
                        alpha_weights += a;
                    } else if (channels == 4) {
                        line[3] += pixel[3] * w;
                    }
                }
                double f = row->weights[j];
                for (size_t c = 0; c < channels; ++c) sum[c] += f * line[c];
                total += f * weights;
                alpha_total += f * alpha_weights;
            }
            size_t shared = separate ? colour : channels;
            for (size_t c = 0; c < shared; ++c)
                out[c] = total == 0.0 ? average[c] : (float)(sum[c] / total);
            if (separate) out[3] = alpha_total == 0.0 ? average[3] : (float)(sum[3] / alpha_total);
        }
    }
}

static void weighted_range(void *opaque, size_t begin, size_t end)
{
    const dpid_context *ctx = opaque;
    switch (ctx->mode) {
    case 1: weighted_rows(ctx, begin, end, 1, 0); break;
    case 2: weighted_rows(ctx, begin, end, 2, 0); break;
    case 3: weighted_rows(ctx, begin, end, 3, 0); break;
    case 4: weighted_rows(ctx, begin, end, 4, 0); break;
    default: weighted_rows(ctx, begin, end, 4, 1); break;
    }
}

/* DPID of a C-contiguous (height, width, channels) float32 image into (target_height,
 * target_width, channels), with lambda >= 0. Two passes over the native pool: the guidance
 * of every output pixel, then per output row band the tent and the weighted footprints.
 * A growing side gives footprints within one or two samples: an area sample, as DPID is a
 * downscaler. Four channels are RGBA: separate_alpha 0 carries alpha with the colour
 * weights (the caller premultiplied), 1 downscales alpha as its own one-channel DPID. gamma
 * 1 works on cn_alpha_gamma(source, 2.2) and applies 1 / 2.2 to the result, as
 * cn_resample_filtered does; the result is not clipped (each output is a weighted mean of
 * its inputs). */
CN_EXPORT cn_status cn_resample_dpid(const float *source, float *out,
    size_t height, size_t width, size_t channels, size_t target_height,
    size_t target_width, float lambda, int separate_alpha, int gamma)
{
    if (!source || !out || !height || !width || !target_height || !target_width ||
        channels < 1 || channels > 4 || !isfinite(lambda) || lambda < 0.0f ||
        separate_alpha < 0 || separate_alpha > 1 || gamma < 0 || gamma > 1)
        return CN_INVALID_ARGUMENT;
    if (height > INT32_MAX || width > INT32_MAX ||
        target_height > UINT32_MAX || target_width > UINT32_MAX ||
        width > SIZE_MAX / sizeof(float) / channels / height ||
        target_width > SIZE_MAX / sizeof(float) / channels / target_height ||
        target_width > SIZE_MAX - target_height)
        return CN_SIZE_OVERFLOW;
    size_t span_count = target_width + target_height;
    if (span_count > SIZE_MAX / sizeof(dpid_span) || width > SIZE_MAX - span_count ||
        height > SIZE_MAX - span_count - width ||
        span_count + width + height > SIZE_MAX / sizeof(float))
        return CN_SIZE_OVERFLOW;
    size_t weight_count = span_count + width + height;
    /* One pooled lease: the guidance, the spans, their weights, then with gamma the linear
     * source, each at a 64-byte offset. */
    size_t total = 0, guidance_at = 0, spans_at = 0, weights_at = 0, linear_at = 0;
    if (!cn_scratch_carve(&total, target_height * target_width * channels * sizeof(float),
            &guidance_at) ||
        !cn_scratch_carve(&total, span_count * sizeof(dpid_span), &spans_at) ||
        !cn_scratch_carve(&total, weight_count * sizeof(float), &weights_at) ||
        (gamma && !cn_scratch_carve(&total, height * width * channels * sizeof(float),
            &linear_at)))
        return CN_ALLOCATION_FAILED;
    cn_scratch scratch = {NULL, 0, -1};
    cn_status status = cn_scratch_lease(total, &scratch, NULL);
    if (status != CN_OK) return status;
    char *base = scratch.data;
    dpid_span *columns = (dpid_span *)(base + spans_at), *rows = columns + target_width;
    float *weights = (float *)(base + weights_at);
    axis_spans(width, target_width, columns, weights);
    axis_spans(height, target_height, rows, weights + width + target_width);
    const float *pixels = source;
    if (gamma) {
        float *linear = (float *)(base + linear_at);
        status = cn_alpha_gamma(source, linear, height * width, channels, 2.2f);
        if (status != CN_OK) goto cleanup;
        pixels = linear;
    }
    dpid_context ctx = {pixels, (float *)(base + guidance_at), out, columns, rows, width,
        target_width, target_height, channels == 4 && separate_alpha ? 5 : (int)channels,
        lambda == 0.0f ? 0 : lambda == 1.0f ? 1 : 2, 0.5f * lambda};
    /* About 64 Ki source samples per chunk: an output row reads about height /
     * target_height + 2 footprint rows of up to width + target_width samples. */
    double row_work = ((double)height / (double)target_height + 2.0) *
        (double)(width + target_width) * (double)channels;
    size_t grain = row_work >= 65536.0 ? 1 : (size_t)(65536.0 / row_work);
    status = cn_parallel_for(target_height, grain, guidance_range, &ctx);
    if (status != CN_OK) goto cleanup;
    status = cn_parallel_for(target_height, grain, weighted_range, &ctx);
    if (status != CN_OK) goto cleanup;
    if (gamma)
        status = cn_alpha_gamma(out, out, target_height * target_width, channels, 1.0f / 2.2f);
cleanup:
    cn_scratch_release(&scratch);
    return status;
}
