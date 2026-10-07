/* Altered C implementation of OpenCV 4.8.0 morphology contracts.
 * Sources: modules/imgproc/src/morph.simd.hpp and morph.dispatch.cpp at
 * https://github.com/opencv/opencv/tree/4.8.0
 * Finite ellipses use running-window row extrema instead of full kernel scans.
 * Ordered exceptional-value paths preserve SIMD/scalar operand and tail order.
 * C ABI, checked ownership and threadpool integration are GPL-3.0-only.
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
#include "exceptional.h"
#include "morph_entry.h"
#include "morphology_shared.h"
#include "parallel.h"
#include "scratch.h"
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>

static void fold_value(float value, float *acc, int *initialized,
                       int maximum, int vector)
{
    if (*initialized) *acc = update(*acc, value, maximum, vector);
    else { *acc = value; *initialized = 1; }
}

/* Repeating a constant border tap is idempotent even for NaN/signed zero.
 * Compress only identical outside spans; retain the order of all image taps. */
static void fold_horizontal(const morphology_context *ctx, size_t y, size_t channel,
    int64_t left, int64_t right, float *acc, int *initialized, int vector)
{
    int64_t x, last;
    if (left < 0) fold_value(ctx->border, acc, initialized, ctx->maximum, vector);
    x = left < 0 ? 0 : left;
    last = right >= (int64_t)ctx->width ? (int64_t)ctx->width - 1 : right;
    for (; x <= last; ++x)
        fold_value(ctx->src[(y * ctx->width + (size_t)x) * ctx->channels + channel],
                   acc, initialized, ctx->maximum, vector);
    if (right >= (int64_t)ctx->width)
        fold_value(ctx->border, acc, initialized, ctx->maximum, vector);
}

static void fold_vertical(const morphology_context *ctx, size_t x, size_t channel,
    int64_t top, int64_t bottom, float *acc, int *initialized, int vector)
{
    int64_t y, last;
    if (top < 0) fold_value(ctx->border, acc, initialized, ctx->maximum, vector);
    y = top < 0 ? 0 : top;
    last = bottom >= (int64_t)ctx->height ? (int64_t)ctx->height - 1 : bottom;
    for (; y <= last; ++y)
        fold_value(ctx->src[((size_t)y * ctx->width + x) * ctx->channels + channel],
                   acc, initialized, ctx->maximum, vector);
    if (bottom >= (int64_t)ctx->height)
        fold_value(ctx->border, acc, initialized, ctx->maximum, vector);
}

static void ordered_rectangle(void *raw, size_t begin, size_t end)
{
    const morphology_context *ctx = (const morphology_context *)raw;
    size_t i, row_width = ctx->width * ctx->channels;
    size_t vector_end = ctx->lane_half ? row_width / ctx->lane_half * ctx->lane_half : 0;
    if (!ctx->stage) vector_end -= vector_end % ctx->channels;
    for (i = begin; i < end; ++i) {
        size_t y = i / row_width, column = i % row_width;
        size_t x = column / ctx->channels, channel = column % ctx->channels;
        int vector = column < vector_end, initialized = 0;
        float acc = ctx->border;
        if (!ctx->stage) {
            int64_t left = (int64_t)x - (int64_t)ctx->radius;
            int64_t right = (int64_t)x + (int64_t)ctx->radius;
            size_t scalar_start = vector_end / ctx->channels;
            if (!vector && x + 1 < ctx->width && (x - scalar_start) % 2 == 0) {
                fold_horizontal(ctx, y, channel, left + 1, right, &acc, &initialized, vector);
                fold_horizontal(ctx, y, channel, left, left, &acc, &initialized, vector);
            } else fold_horizontal(ctx, y, channel, left, right, &acc, &initialized, vector);
        } else {
            int64_t top = (int64_t)y - (int64_t)ctx->radius;
            int64_t bottom = (int64_t)y + (int64_t)ctx->radius;
            /* FilterEngine emits four rows per batch; each pair shares its
             * middle taps, then appends the first/last tap respectively. */
            if (y % 2 == 0 && y + 1 < ctx->height) {
                fold_vertical(ctx, x, channel, top + 1, bottom, &acc, &initialized, vector);
                fold_vertical(ctx, x, channel, top, top, &acc, &initialized, vector);
            } else fold_vertical(ctx, x, channel, top, bottom, &acc, &initialized, vector);
        }
        ctx->out[i] = acc;
    }
}

static void ordered_shape(void *raw, size_t begin, size_t end)
{
    const morphology_context *ctx = (const morphology_context *)raw;
    size_t i, row_width = ctx->width * ctx->channels;
    size_t vector_end = ctx->lane_half ? row_width / ctx->lane_half * ctx->lane_half : 0;
    for (i = begin; i < end; ++i) {
        size_t y = i / row_width, column = i % row_width;
        size_t x = column / ctx->channels, channel = column % ctx->channels;
        int vector = column < vector_end, initialized = 0;
        int64_t top = (int64_t)y - (int64_t)ctx->radius;
        int64_t bottom = (int64_t)y + (int64_t)ctx->radius;
        int64_t yy = top < 0 ? 0 : top;
        int64_t last = bottom >= (int64_t)ctx->height ? (int64_t)ctx->height - 1 : bottom;
        float acc = ctx->border;
        if (top < 0) fold_value(ctx->border, &acc, &initialized, ctx->maximum, vector);
        for (; yy <= last; ++yy) {
            int64_t dy = yy - (int64_t)y;
            size_t half = ctx->shape == 1 ? (dy == 0 ? ctx->radius : 0) :
                ctx->spans[(size_t)(dy + (int64_t)ctx->radius)];
            fold_horizontal(ctx, (size_t)yy, channel,
                (int64_t)x - (int64_t)half, (int64_t)x + (int64_t)half,
                &acc, &initialized, vector);
        }
        if (bottom >= (int64_t)ctx->height)
            fold_value(ctx->border, &acc, &initialized, ctx->maximum, vector);
        ctx->out[i] = acc;
    }
}

/* Each line's running extremum over [x - offset, x + offset] (clipped to the row):
 * the deque keeps a strictly monotonic run of values, a new value dropping every
 * entry it ties or beats, so its head is the window's last extremal element, whose
 * bits are stored. The entries carry their values: the compares are B3's, without
 * the indirect src loads. A chunk's lines run one after another, so they share the
 * deque of its first line (its slot of the queue, which no other chunk uses). */
static void ellipse_horizontal(void *raw, size_t begin, size_t end)
{
    const morphology_context *ctx = (const morphology_context *)raw;
    struct morph_entry *queue = ctx->queue + begin * ctx->width;
    size_t line;
    for (line = begin; line < end; ++line) {
        size_t y = line / ctx->channels, c = line % ctx->channels;
        const float *src = ctx->src + y * ctx->width * ctx->channels + c;
        float *out = ctx->out + y * ctx->width * ctx->channels + c;
        size_t head = 0, tail = 0, inserted = 0, x;
        for (x = 0; x < ctx->width; ++x) {
            size_t left = x > ctx->offset ? x - ctx->offset : 0;
            size_t right = ctx->offset >= ctx->width - 1 - x ? ctx->width - 1 : x + ctx->offset;
            while (head < tail && queue[head].index < left) ++head;
            while (inserted <= right) {
                float value = src[inserted * ctx->channels];
                while (head < tail) {
                    float previous = queue[tail - 1].value;
                    if (ctx->maximum ? previous > value : previous < value) break;
                    --tail;
                }
                queue[tail].value = value;
                queue[tail].index = (uint32_t)inserted;
                ++tail;
                ++inserted;
            }
            out[x * ctx->channels] = queue[head].value;
        }
    }
}

/* The combine over dy in [ctx->offset, ctx->offset_end) (morphology_shared.h): per row
 * segment of the chunk, every output through combine_one, or at ISA level avx2 or
 * above through cn_morphology_combine_avx2. */
static void ellipse_combine(void *raw, size_t begin, size_t end)
{
    const morphology_context *ctx = (const morphology_context *)raw;
    size_t i = begin;
#if defined(_MSC_VER) && defined(_M_X64)
    if (ctx->isa >= CN_ISA_AVX2) {
        cn_morphology_combine_avx2(ctx, begin, end);
        return;
    }
#endif
    while (i < end) {
        size_t y, stop = combine_segment(ctx, i, end, &y);
        for (; i < stop; ++i) ctx->out[i] = combine_one(ctx, i, y);
    }
}

/* The ellipse's half-width at offset dy, clipped to the row. */
static size_t ellipse_half(const morphology_context *ctx, size_t dy)
{
    size_t half = ctx->spans[ctx->radius + dy];
    return half < ctx->width ? half : ctx->width - 1;
}

/* Exposed separately to test every node radius against getStructuringElement,
 * without allocating a quadratic kernel or processing an image. */
CN_EXPORT cn_status cn_morphology_ellipse_spans(size_t radius, size_t *spans)
{
    size_t i;
    uintptr_t address = (uintptr_t)spans;
    double inv;
    if (!spans || address % _Alignof(size_t) || radius > 46340) return CN_INVALID_ARGUMENT;
    if (address > UINTPTR_MAX - (radius * 2 + 1) * sizeof(size_t)) return CN_SIZE_OVERFLOW;
    if (!radius) { spans[0] = 0; return CN_OK; }
    inv = 1.0 / ((double)radius * (double)radius);
    for (i = 0; i <= radius * 2; ++i) {
        int64_t dy = (int64_t)i - (int64_t)radius;
        double value = (double)((int64_t)(radius * radius) - dy * dy) * inv;
        spans[i] = (size_t)nearbyint((double)radius * sqrt(value));
    }
    return CN_OK;
}

/* The image-size scratch (horizontal, alternate, queue) is one lease of the shared
 * pool, every buffer written before it is read (D10); the ellipse spans stay a small
 * malloc. Iteration i writes out when (iterations - 1 - i) is even, else alternate,
 * so the last writes out (ping-pong, D11). */
CN_EXPORT cn_status cn_morphology_complete(const float *src, float *out,
    size_t height, size_t width, size_t channels, size_t radius, size_t iterations,
    int shape, int maximum, size_t lanes)
{
    size_t pixels, count, bytes, iteration, limit;
    size_t total = 0, horizontal_at = 0, alternate_at = 0, queue_at = 0;
    uintptr_t a = (uintptr_t)src, b = (uintptr_t)out;
    int exceptional, deques, sized = 1;
    float *horizontal = NULL, *alternate = NULL;
    size_t *spans = NULL;
    const float *current = src;
    cn_scratch scratch = {NULL, 0, -1};
    morphology_context ctx;
    cn_status status = CN_OK;
    if (!height || !width || !channels || !radius || !iterations ||
        shape < 0 || shape > 2 || (maximum != 0 && maximum != 1) ||
        (lanes != 0 && lanes != 4 && lanes != 8 && lanes != 16)) return CN_INVALID_ARGUMENT;
    if (width > SIZE_MAX / height || width * height > SIZE_MAX / channels)
        return CN_SIZE_OVERFLOW;
    pixels = width * height; count = pixels * channels;
    if (count > SIZE_MAX / sizeof(size_t) || radius > (INT_MAX - 1) / 2 ||
        height > INT_MAX || width > (size_t)INT_MAX / channels) return CN_SIZE_OVERFLOW;
    bytes = count * sizeof(float);
    if (!src || !out || a % _Alignof(float) || b % _Alignof(float)) return CN_INVALID_ARGUMENT;
    if (a > UINTPTR_MAX - bytes || b > UINTPTR_MAX - bytes) return CN_SIZE_OVERFLOW;
    if (a < b + bytes && b < a + bytes) return CN_INVALID_ARGUMENT;
    if (shape == 2 && radius > 46340) return CN_INVALID_ARGUMENT;
    if (shape == 0) {
        if (iterations > (size_t)(INT_MAX - 1) / 2 / radius) return CN_SIZE_OVERFLOW;
        radius *= iterations; iterations = 1;
    }
    ctx.isa = cn_isa_current();
    exceptional = cn_exceptional_scan(src, count, ctx.isa);
    deques = shape == 2 && !exceptional;
    if (shape == 0 || deques) sized = cn_scratch_carve(&total, bytes, &horizontal_at);
    if (iterations > 1) sized = sized && cn_scratch_carve(&total, bytes, &alternate_at);
    if (deques)
        sized = sized && count <= SIZE_MAX / sizeof(struct morph_entry) &&
            cn_scratch_carve(&total, count * sizeof(struct morph_entry), &queue_at);
    if (!sized) return CN_ALLOCATION_FAILED;
    if (shape == 2) spans = (size_t *)malloc((radius * 2 + 1) * sizeof(size_t));
    if (total) status = cn_scratch_lease(total, &scratch, NULL);
    if (status != CN_OK || (shape == 2 && !spans)) {
        status = CN_ALLOCATION_FAILED; goto cleanup;
    }
    if (spans) {
        status = cn_morphology_ellipse_spans(radius, spans);
        if (status != CN_OK) goto cleanup;
    }
    if (shape == 0 || deques) horizontal = (float *)((char *)scratch.data + horizontal_at);
    if (iterations > 1) alternate = (float *)((char *)scratch.data + alternate_at);
    ctx.queue = deques ? (struct morph_entry *)((char *)scratch.data + queue_at) : NULL;
    ctx.height = height; ctx.width = width; ctx.channels = channels; ctx.radius = radius;
    ctx.lane_half = lanes / 2; ctx.maximum = maximum; ctx.shape = shape;
    ctx.spans = spans; ctx.offset = ctx.offset_end = ctx.zero_from = 0; ctx.stage = 0;
    ctx.current = src;
    ctx.border = maximum ? -FLT_MAX : FLT_MAX;
    limit = radius < height - 1 ? radius : height - 1;
    for (iteration = 0; iteration < iterations; ++iteration) {
        float *destination = (iterations - 1 - iteration) % 2 ? alternate : out;
        ctx.src = current; ctx.out = destination;
        if (shape == 0) {
            ctx.out = horizontal; ctx.stage = 0;
            status = cn_parallel_for(count, 512, ordered_rectangle, &ctx);
            if (status != CN_OK) break;
            ctx.src = horizontal; ctx.out = destination; ctx.stage = 1;
            status = cn_parallel_for(count, 512, ordered_rectangle, &ctx);
        } else if (deques) {
            /* One horizontal pass per half-width and one combine per run of offsets dy
             * sharing it (B3 ran a combine per dy). The half-width falls with dy, so
             * the offsets of half-width 0, whose row extrema are current's own bits,
             * end the list and join the run before them. */
            size_t dy = 0;
            while (dy <= limit) {
                size_t half = ellipse_half(&ctx, dy), last = dy;
                while (last < limit && ellipse_half(&ctx, last + 1) == half) ++last;
                ctx.zero_from = half ? last + 1 : dy;
                while (last < limit && ellipse_half(&ctx, last + 1) == 0) ++last;
                if (half) {
                    ctx.src = current; ctx.out = horizontal; ctx.offset = half;
                    status = cn_parallel_for(height * channels, 1 + 16384 / width, ellipse_horizontal, &ctx);
                    if (status != CN_OK) break;
                }
                ctx.src = horizontal; ctx.current = current; ctx.out = destination;
                ctx.offset = dy; ctx.offset_end = last + 1;
                status = cn_parallel_for(count, 65536, ellipse_combine, &ctx);
                if (status != CN_OK) break;
                dy = last + 1;
            }
        } else status = cn_parallel_for(count, 128, ordered_shape, &ctx);
        if (status != CN_OK) break;
        current = destination;
    }
cleanup:
    cn_scratch_release(&scratch);
    free(spans);
    return status;
}
