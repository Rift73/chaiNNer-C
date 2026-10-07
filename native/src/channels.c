/* Channel assembly and image layout kernels. SPDX-License-Identifier: GPL-3.0-only */
#include "chainner.h"
#include "parallel.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

/* A null source denotes a constant channel. The Python boundary owns every
 * source buffer until the synchronous parallel call returns. */
typedef struct cn_channel_source {
    const float *data;
    size_t channels, channel;
    float constant;
} cn_channel_source;

static cn_status image_size(size_t height, size_t width, size_t channels)
{
    if (height == 0 || width == 0 || channels == 0) return CN_INVALID_ARGUMENT;
    if (height > SIZE_MAX / width || height * width > SIZE_MAX / channels ||
        height * width * channels > SIZE_MAX / sizeof(float)) return CN_SIZE_OVERFLOW;
    return CN_OK;
}

typedef struct assemble_context {
    const cn_channel_source *sources;
    float *out;
    size_t channels;
} assemble_context;

static void assemble_range(void *opaque, size_t begin, size_t end)
{
    const assemble_context *ctx = opaque;
    for (size_t p = begin; p < end; ++p) {
        for (size_t c = 0; c < ctx->channels; ++c) {
            const cn_channel_source *src = &ctx->sources[c];
            ctx->out[p * ctx->channels + c] = src->data == NULL ? src->constant
                : src->data[p * src->channels + src->channel];
        }
    }
}

CN_EXPORT cn_status cn_channels_assemble(const cn_channel_source *sources,
    float *out, size_t pixels, size_t channels)
{
    if (sources == NULL || out == NULL ||
        (uintptr_t)sources % _Alignof(cn_channel_source) ||
        (uintptr_t)out % _Alignof(float)) return CN_INVALID_ARGUMENT;
    cn_status status = image_size(1, pixels, channels);
    if (status != CN_OK) return status;
    if (channels > SIZE_MAX / sizeof(cn_channel_source)) return CN_SIZE_OVERFLOW;
    size_t bytes = pixels * channels * sizeof(float);
    uintptr_t dst = (uintptr_t)out, descriptors = (uintptr_t)sources;
    if (dst > UINTPTR_MAX - bytes ||
        descriptors > UINTPTR_MAX - channels * sizeof(cn_channel_source)) return CN_SIZE_OVERFLOW;
    if (descriptors < dst + bytes && dst < descriptors + channels * sizeof(cn_channel_source)) return CN_INVALID_ARGUMENT;
    for (size_t c = 0; c < channels; ++c) {
        if (sources[c].data != NULL) {
            if (sources[c].channel >= sources[c].channels ||
                (uintptr_t)sources[c].data % _Alignof(float)) return CN_INVALID_ARGUMENT;
            status = image_size(1, pixels, sources[c].channels);
            if (status != CN_OK) return status;
            uintptr_t src = (uintptr_t)sources[c].data;
            size_t input_bytes = pixels * sources[c].channels * sizeof(float);
            if (src > UINTPTR_MAX - input_bytes) return CN_SIZE_OVERFLOW;
            if (src < dst + bytes && dst < src + input_bytes) return CN_INVALID_ARGUMENT;
        }
    }
    assemble_context ctx = {sources, out, channels};
    return cn_parallel_for(pixels, 65536, assemble_range, &ctx);
}

typedef struct flip_context {
    const float *src;
    float *out;
    size_t height, width, channels;
    int axis;
} flip_context;

static void flip_range(void *opaque, size_t begin, size_t end)
{
    const flip_context *ctx = opaque;
    for (size_t p = begin; p < end; ++p) {
        size_t y = p / ctx->width, x = p % ctx->width;
        if (ctx->axis <= 0) y = ctx->height - 1 - y;
        if (ctx->axis != 0) x = ctx->width - 1 - x;
        memcpy(ctx->out + p * ctx->channels,
            ctx->src + (y * ctx->width + x) * ctx->channels,
            ctx->channels * sizeof(float));
    }
}

CN_EXPORT cn_status cn_channels_flip(const float *src, float *out,
    size_t height, size_t width, size_t channels, int axis)
{
    if (src == NULL || out == NULL || axis < -1 || axis > 1) return CN_INVALID_ARGUMENT;
    cn_status status = image_size(height, width, channels);
    if (status != CN_OK) return status;
    flip_context ctx = {src, out, height, width, channels, axis};
    return cn_parallel_for(height * width, 65536, flip_range, &ctx);
}

typedef struct shift_context {
    const float *src;
    float *out;
    size_t height, width, channels, out_channels;
    int64_t dx, dy;
    int fill;
} shift_context;

static float shifted_sample(const shift_context *ctx, int64_t y, int64_t x, size_t c)
{
    if (x < 0 || y < 0 || x >= (int64_t)ctx->width || y >= (int64_t)ctx->height)
        return ctx->fill == 0 && c % 4 == 3 ? 1.0f : 0.0f;
    if (ctx->fill == 1 && ctx->channels < 4) {
        if (c == 3) return 1.0f;
        if (ctx->channels == 1) c = 0;
    }
    return ctx->src[((size_t)y * ctx->width + (size_t)x) * ctx->channels + c];
}

static void shift_range(void *opaque, size_t begin, size_t end)
{
    const shift_context *ctx = opaque;
    for (size_t p = begin; p < end; ++p) {
        int64_t y = (int64_t)(p / ctx->width) - ctx->dy;
        int64_t x = (int64_t)(p % ctx->width) - ctx->dx;
        if (ctx->fill == 2) {
            if (x < 0) x += (int64_t)ctx->width;
            if (y < 0) y += (int64_t)ctx->height;
            memcpy(ctx->out + p * ctx->channels,
                ctx->src + ((size_t)y * ctx->width + (size_t)x) * ctx->channels,
                ctx->channels * sizeof(float));
        } else {
            for (size_t c = 0; c < ctx->out_channels; ++c) {
                float value = shifted_sample(ctx, y, x, c);
                /* OpenCV 5.0.0's float warpAffine still interpolates an integer
                 * translation where the source cell starts in [-1, size - 1]
                 * (beyond, the border value is stored). For 1, 3 and 4 channels its
                 * warp_kernels invokers use lerps with zero fractions:
                 * v0 = p00 + 0(p01 - p00), v1 = p10 + 0(p11 - p10), v = v0 + 0(v1 - v0),
                 * so an infinite or NaN tap gives NaN and the zero products set a zero
                 * result's sign. Other channel counts keep the remap invoker's
                 * weighted taps, p00 + 0 p01 + 0 p10 + 0 p11. */
                if (x >= -1 && y >= -1 && x < (int64_t)ctx->width && y < (int64_t)ctx->height) {
                    const float fraction = 0.0f;
                    float p01 = shifted_sample(ctx, y, x + 1, c);
                    float p10 = shifted_sample(ctx, y + 1, x, c);
                    float p11 = shifted_sample(ctx, y + 1, x + 1, c);
                    if (ctx->out_channels == 1 || ctx->out_channels == 3 || ctx->out_channels == 4) {
                        float v0 = value + fraction * (p01 - value);
                        float v1 = p10 + fraction * (p11 - p10);
                        value = v0 + fraction * (v1 - v0);
                    } else {
                        value += p01 * fraction;
                        value += p10 * fraction;
                        value += p11 * fraction;
                    }
                }
                ctx->out[p * ctx->out_channels + c] = value;
            }
        }
    }
}

CN_EXPORT cn_status cn_channels_shift(const float *src, float *out,
    size_t height, size_t width, size_t channels, int64_t dx, int64_t dy, int fill)
{
    if (src == NULL || out == NULL || fill < -1 || fill > 2) return CN_INVALID_ARGUMENT;
    cn_status status = image_size(height, width, channels);
    if (status != CN_OK) return status;
    if (height > INT64_MAX / 2 || width > INT64_MAX / 2) return CN_SIZE_OVERFLOW;
    if (fill == 2) {
        if (dx < 0 || dy < 0 || dx >= (int64_t)width || dy >= (int64_t)height)
            return CN_INVALID_ARGUMENT;
    } else {
        /* 128: OpenCV 5.0.0's CV_CN_MAX (native_versions.CV_CN_MAX in the bridge). */
        if (height >= 32767 || width >= 32767 || channels > 128 || (fill == -1 && channels > 4) ||
            dx < -(int64_t)width - 1 || dx > (int64_t)width + 1 ||
            dy < -(int64_t)height - 1 || dy > (int64_t)height + 1)
            return CN_INVALID_ARGUMENT;
        if (fill == 1 && channels != 1 && channels != 3 && channels != 4)
            return CN_INVALID_ARGUMENT;
    }
    size_t out_channels = fill == 1 ? 4 : channels;
    status = image_size(height, width, out_channels);
    if (status != CN_OK) return status;
    shift_context ctx = {src, out, height, width, channels, out_channels, dx, dy, fill};
    return cn_parallel_for(height * width, 65536, shift_range, &ctx);
}

typedef struct bbox_context {
    const float *src;
    size_t *left, *right;
    size_t width, channels, channel;
    float threshold;
} bbox_context;

static void bbox_rows(void *opaque, size_t begin, size_t end)
{
    const bbox_context *ctx = opaque;
    for (size_t y = begin; y < end; ++y) {
        size_t left = ctx->width, right = 0;
        for (size_t x = 0; x < ctx->width; ++x) {
            if (ctx->src[(y * ctx->width + x) * ctx->channels + ctx->channel] > ctx->threshold) {
                if (left == ctx->width) left = x;
                right = x + 1;
            }
        }
        ctx->left[y] = left;
        ctx->right[y] = right;
    }
}

/* bounds = x, y, width, height; all zero if no sample exceeds threshold. */
CN_EXPORT cn_status cn_channels_bbox(const float *src, size_t height, size_t width,
    size_t channels, size_t channel, float threshold, size_t *bounds)
{
    if (src == NULL || bounds == NULL || channel >= channels) return CN_INVALID_ARGUMENT;
    cn_status status = image_size(height, width, channels);
    if (status != CN_OK) return status;
    if (height > SIZE_MAX / (2 * sizeof(size_t))) return CN_SIZE_OVERFLOW;
    size_t *rows = malloc(2 * height * sizeof(size_t));
    if (rows == NULL) return CN_ALLOCATION_FAILED;
    bbox_context ctx = {src, rows, rows + height, width, channels, channel, threshold};
    size_t grain = 65536 / width;
    if (grain == 0) grain = 1;
    status = cn_parallel_for(height, grain, bbox_rows, &ctx);
    if (status == CN_OK) {
        size_t left = width, right = 0, top = height, bottom = 0;
        for (size_t y = 0; y < height; ++y) {
            if (ctx.right[y] != 0) {
                if (ctx.left[y] < left) left = ctx.left[y];
                if (ctx.right[y] > right) right = ctx.right[y];
                if (top == height) top = y;
                bottom = y + 1;
            }
        }
        bounds[0] = right == 0 ? 0 : left;
        bounds[1] = right == 0 ? 0 : top;
        bounds[2] = right == 0 ? 0 : right - left;
        bounds[3] = right == 0 ? 0 : bottom - top;
    }
    free(rows);
    return status;
}

typedef struct crop_context {
    const float *src;
    float *out;
    size_t width, channels, x, y, crop_width, out_width;
} crop_context;

static void crop_rows(void *opaque, size_t begin, size_t end)
{
    const crop_context *ctx = opaque;
    size_t row_size = ctx->crop_width * ctx->channels;
    for (size_t y = begin; y < end; ++y)
        memcpy(ctx->out + y * ctx->out_width * ctx->channels,
            ctx->src + ((y + ctx->y) * ctx->width + ctx->x) * ctx->channels,
            row_size * sizeof(float));
}

/* out's rows are out_width pixels apart: crop_width for a compact crop, width
 * for the crop's place in a copy of the whole image (upstream's np.copy(img)[...]
 * view, Crop to Content). */
CN_EXPORT cn_status cn_channels_crop(const float *src, float *out, size_t height,
    size_t width, size_t channels, size_t x, size_t y, size_t crop_width, size_t crop_height,
    size_t out_width)
{
    if (src == NULL || out == NULL) return CN_INVALID_ARGUMENT;
    cn_status status = image_size(height, width, channels);
    if (status != CN_OK) return status;
    if (x >= width || y >= height || crop_width == 0 || crop_height == 0 ||
        crop_width > width - x || crop_height > height - y || out_width < crop_width)
        return CN_INVALID_ARGUMENT;
    status = image_size(crop_height, out_width, channels);
    if (status != CN_OK) return status;
    crop_context ctx = {src, out, width, channels, x, y, crop_width, out_width};
    size_t grain = 65536 / crop_width;
    if (grain == 0) grain = 1;
    return cn_parallel_for(crop_height, grain, crop_rows, &ctx);
}
