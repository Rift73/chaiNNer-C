/* Resize buffer passes and content-aware border selection.
 * SPDX-License-Identifier: GPL-3.0-only */
#include "chainner.h"
#include "chainner_ext_kernels.h"
#include "parallel.h"
#include "numeric.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static cn_status image_size(size_t h, size_t w, size_t c)
{
    if (!h || !w || !c || c > 4) return CN_INVALID_ARGUMENT;
    if (h > SIZE_MAX / w || h * w > SIZE_MAX / c ||
        h * w * c > SIZE_MAX / sizeof(float)) return CN_SIZE_OVERFLOW;
    return CN_OK;
}

typedef struct nearest_context {
    const float *src;
    float *out;
    size_t w, c, out_w;
    uint64_t x_step, y_step;
} nearest_context;

static void nearest_range(void *opaque, size_t begin, size_t end)
{
    const nearest_context *ctx = opaque;
    for (size_t p = begin; p < end; ++p) {
        uint64_t y = (uint64_t)(p / ctx->out_w);
        uint64_t x = (uint64_t)(p % ctx->out_w);
        size_t sy = (size_t)((y * ctx->y_step + (ctx->y_step >> 1)) >> 32);
        size_t sx = (size_t)((x * ctx->x_step + (ctx->x_step >> 1)) >> 32);
        memcpy(ctx->out + p * ctx->c,
            ctx->src + (sy * ctx->w + sx) * ctx->c, ctx->c * sizeof(float));
    }
}

CN_EXPORT cn_status cn_resample_nearest(const float *src, float *out,
    size_t h, size_t w, size_t c, size_t out_h, size_t out_w)
{
    if (!src || !out) return CN_INVALID_ARGUMENT;
    cn_status status = image_size(h, w, c);
    if (status != CN_OK) return status;
    status = image_size(out_h, out_w, c);
    if (status != CN_OK) return status;
    if (h > INT32_MAX || w > INT32_MAX || out_h > UINT32_MAX || out_w > UINT32_MAX)
        return CN_SIZE_OVERFLOW;
    /* Match chainner_ext's 32-bit fractional center mapping, including its
     * truncation at exact half-pixel ties. A floating point formula differs.
     * https://github.com/chaiNNer-org/chaiNNer-rs/blob/main/crates/image-ops/src/scale/scale.rs
     */
    nearest_context ctx = {src, out, w, c, out_w,
        ((uint64_t)w << 32) / out_w, ((uint64_t)h << 32) / out_h};
    return cn_parallel_for(out_h * out_w, 65536, nearest_range, &ctx);
}

typedef struct alpha_context {
    const float *src;
    float *out;
    int undo;
} alpha_context;

static void alpha_range(void *opaque, size_t begin, size_t end)
{
    const alpha_context *ctx = opaque;
    for (size_t p = begin; p < end; ++p) {
        const float *src = ctx->src + p * 4;
        float *out = ctx->out + p * 4;
        float alpha = src[3];
        if (!ctx->undo) {
            for (size_t c = 0; c < 3; ++c) out[c] = src[c] * alpha;
            out[3] = alpha;
        } else {
            float denominator = alpha < 0.0001f ? 0.0001f : alpha;
            float reciprocal = 1.0f / denominator;
            for (size_t c = 0; c < 3; ++c) {
                float value = src[c] * reciprocal;
                out[c] = value > 1.0f ? 1.0f : value;
            }
            out[3] = alpha > 1.0f ? 1.0f : alpha;
        }
    }
}

CN_EXPORT cn_status cn_resample_alpha(const float *src, float *out,
    size_t pixels, int undo)
{
    if (!src || !out || !pixels || (undo != 0 && undo != 1)) return CN_INVALID_ARGUMENT;
    if (pixels > SIZE_MAX / (4 * sizeof(float))) return CN_SIZE_OVERFLOW;
    alpha_context ctx = {src, out, undo};
    return cn_parallel_for(pixels, 65536, alpha_range, &ctx);
}

typedef struct content_context {
    const float *src;
    unsigned char *mask;
    const float *color;
    float tolerance;
    size_t c;
    int pairwise;
    float *scratch; /* one row of differences past 8192 channels, else NULL */
} content_context;

static void content_range(void *opaque, size_t begin, size_t end)
{
    const content_context *ctx = opaque;
    float row[8192];
    float *differences = ctx->scratch ? ctx->scratch : row;
    for (size_t p = begin; p < end; ++p) {
        const float *pixel = ctx->src + p * ctx->c;
        float difference = fabsf(pixel[0] - ctx->color[0]);
        if (ctx->c > 1) {
            difference = 0.0f;
            if (ctx->pairwise) {
                /* NumPy 2.5.3 reduces each pixel's fresh contiguous channels in one
                   unbuffered inner loop: one pairwise tree, however many channels. */
                for (size_t c = 0; c < ctx->c; ++c)
                    differences[c] = fabsf(pixel[c] - ctx->color[c]);
                difference += cn_numpy_sum_f32(differences, ctx->c, 1, 0, 0);
            } else for (size_t c = 0; c < ctx->c; ++c)
                difference += fabsf(pixel[c] - ctx->color[c]);
            difference /= (float)ctx->c;
        }
        ctx->mask[p] = (unsigned char)(difference > ctx->tolerance);
    }
}

static int compare_float(const void *a, const void *b)
{
    float x = *(const float *)a, y = *(const float *)b;
    if (isnan(x)) return isnan(y) ? 0 : 1;
    if (isnan(y)) return -1;
    return (x > y) - (x < y);
}

typedef struct section { size_t start, end; } section;

static size_t section_distance(section item, size_t middle)
{
    if (middle < item.start) return item.start - middle;
    return middle >= item.end ? middle - item.end : 0;
}

static section select_section(const unsigned char *flags, size_t length, int mode)
{
    section selected = {0, length};
    int found = 0;
    size_t position = 0;
    while (position < length) {
        if (!flags[position]) { ++position; continue; }
        section current = {position, position};
        while (position < length && flags[position]) ++position;
        current.end = position;
        if (!found) { selected = current; found = 1; }
        else if (mode == 1) selected.end = current.end;
        else if (mode == 2 && section_distance(current, length / 2) <
            section_distance(selected, length / 2)) selected = current;
        else if (mode == 3 && current.end - current.start >
            selected.end - selected.start) selected = current;
    }
    return selected;
}

CN_EXPORT cn_status cn_resample_border(const float *src, size_t h, size_t w,
    size_t c, float tolerance, int mode, size_t *bounds, int pairwise)
{
    if (!src || !bounds || !h || !w || !c || mode < 1 || mode > 3 || (pairwise != 0 && pairwise != 1)) return CN_INVALID_ARGUMENT;
    if (h > SIZE_MAX / w || h * w > SIZE_MAX / c || h * w * c > SIZE_MAX / sizeof(float)) return CN_SIZE_OVERFLOW;
    cn_status status = CN_OK;
    if (h > SIZE_MAX / 2 || w > SIZE_MAX / 2 - h ||
        2 * (w + h) > SIZE_MAX / sizeof(float)) return CN_SIZE_OVERFLOW;
    size_t border_count = 2 * (h + w);
    float *border = malloc(border_count * sizeof(float));
    float *color = malloc(c * sizeof(float));
    unsigned char *mask = malloc(h * w);
    unsigned char *flags = malloc(h > w ? h : w);
    float *scratch = pairwise && c > 8192 ? malloc(c * sizeof(float)) : NULL;
    if (!border || !color || !mask || !flags || (pairwise && c > 8192 && !scratch)) {
        free(border); free(color); free(mask); free(flags); free(scratch);
        return CN_ALLOCATION_FAILED;
    }
    content_context ctx = {src, mask, color, tolerance, c, pairwise, scratch};
    for (size_t channel = 0; channel < c; ++channel) {
        for (size_t x = 0; x < w; ++x) {
            border[x] = src[x * c + channel];
            border[w + x] = src[((h - 1) * w + x) * c + channel];
        }
        for (size_t y = 0; y < h; ++y) {
            border[2 * w + y] = src[y * w * c + channel];
            border[2 * w + h + y] = src[(y * w + w - 1) * c + channel];
        }
        qsort(border, border_count, sizeof(float), compare_float);
        color[channel] = isnan(border[border_count - 1]) ? cn_f32_from_bits(CN_NPY_NANF_BITS) :
            ((0.0f + border[border_count / 2 - 1]) + border[border_count / 2]) / 2.0f;
    }
    free(border);
    /* Past 8192 channels the pixels share one scratch row, so they run as one call. */
    status = cn_parallel_for(h * w, scratch ? h * w : 65536, content_range, &ctx);
    free(scratch);
    free(color);
    if (status != CN_OK) { free(mask); free(flags); return status; }
    memset(flags, 0, w);
    for (size_t y = 0; y < h; ++y)
        for (size_t x = 0; x < w; ++x) flags[x] |= mask[y * w + x];
    section horizontal = select_section(flags, w, mode);
    memset(flags, 0, h);
    for (size_t y = 0; y < h; ++y)
        for (size_t x = horizontal.start; x < horizontal.end; ++x)
            flags[y] |= mask[y * w + x];
    section vertical = select_section(flags, h, mode);
    if (mode != 1) {
        size_t width = horizontal.end - horizontal.start;
        memset(flags, 0, width);
        for (size_t y = vertical.start; y < vertical.end; ++y)
            for (size_t x = horizontal.start; x < horizontal.end; ++x)
                flags[x - horizontal.start] |= mask[y * w + x];
        section inner = select_section(flags, width, 1);
        horizontal.end = horizontal.start + inner.end;
        horizontal.start += inner.start;
    }
    free(mask); free(flags);
    bounds[0] = horizontal.start;
    bounds[1] = vertical.start;
    bounds[2] = horizontal.end - horizontal.start;
    bounds[3] = vertical.end - vertical.start;
    return CN_OK;
}
