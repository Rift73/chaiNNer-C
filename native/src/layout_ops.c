/* Fused layout, padding, stacking and pixel-block kernels.
 * SPDX-License-Identifier: GPL-3.0-only */
#include "chainner.h"
#include "numeric.h"
#include "parallel.h"
#include <math.h>
#include <string.h>

static cn_status layout_size(size_t h, size_t w, size_t c)
{
    if (!h || !w || !c) return CN_INVALID_ARGUMENT;
    if (h > SIZE_MAX / w || h * w > SIZE_MAX / c ||
        h * w * c > SIZE_MAX / sizeof(float)) return CN_SIZE_OVERFLOW;
    return CN_OK;
}

static size_t reflected(size_t position, size_t length)
{
    if (length == 1) return 0;
    size_t period = (length - 1) * 2;
    position %= period;
    return position < length ? position : period - position;
}

static size_t border_coordinate(size_t position, size_t before, size_t length, int mode)
{
    if (position >= before && position - before < length) return position - before;
    if (mode == 1) return position < before ? 0 : length - 1;
    if (mode == 3) {
        if (position >= before) return (position - before) % length;
        size_t rem = (before - position) % length;
        return rem == 0 ? 0 : length - rem;
    }
    return reflected(position >= before ? position - before : before - position, length);
}

static float channel_value(const float *pixel, size_t src_c, size_t dst_channel, int expand_gray)
{
    if (src_c == 1 && expand_gray && dst_channel < 3) return pixel[0];
    return dst_channel < src_c ? pixel[dst_channel] : 1.0f;
}

typedef struct pad_context {
    const float *src, *color;
    float *out;
    size_t h, w, c, out_w, out_c, top, left;
    int mode;
} pad_context;

static void pad_range(void *opaque, size_t begin, size_t end)
{
    const pad_context *ctx = opaque;
    for (size_t p = begin; p < end; ++p) {
        size_t y = p / ctx->out_w, x = p % ctx->out_w;
        int inside = y >= ctx->top && y - ctx->top < ctx->h &&
            x >= ctx->left && x - ctx->left < ctx->w;
        if (!inside && ctx->mode == 0) {
            memcpy(ctx->out + p * ctx->out_c, ctx->color, ctx->out_c * sizeof(float));
        } else {
            size_t sy = border_coordinate(y, ctx->top, ctx->h, ctx->mode);
            size_t sx = border_coordinate(x, ctx->left, ctx->w, ctx->mode);
            const float *pixel = ctx->src + (sy * ctx->w + sx) * ctx->c;
            for (size_t c = 0; c < ctx->out_c; ++c)
                ctx->out[p * ctx->out_c + c] = channel_value(pixel, ctx->c, c, 1);
        }
    }
}

CN_EXPORT cn_status cn_layout_pad(const float *src, float *out, const float *color,
    size_t h, size_t w, size_t c, size_t out_h, size_t out_w, size_t out_c,
    size_t top, size_t left, int mode)
{
    if (!src || !out || !color || (mode != 0 && mode != 1 && mode != 3 && mode != 4) ||
        out_c < c) return CN_INVALID_ARGUMENT;
    cn_status status = layout_size(h, w, c);
    if (status != CN_OK) return status;
    status = layout_size(out_h, out_w, out_c);
    if (status != CN_OK) return status;
    if (h > SIZE_MAX / 2 || w > SIZE_MAX / 2) return CN_SIZE_OVERFLOW;
    pad_context ctx = {src, color, out, h, w, c, out_w, out_c, top, left, mode};
    return cn_parallel_for(out_h * out_w, 65536, pad_range, &ctx);
}

typedef struct paste_context {
    const float *src;
    float *out;
    size_t h, w, c, out_w, out_c, top, left, target_h, target_w;
    int expand_gray;
    double scale_y, scale_x;
} paste_context;

static void paste_range(void *opaque, size_t begin, size_t end)
{
    const paste_context *ctx = opaque;
    for (size_t p = begin; p < end; ++p) {
        size_t y = p / ctx->target_w, x = p % ctx->target_w;
        size_t sy = (size_t)((double)y * ctx->scale_y);
        size_t sx = (size_t)((double)x * ctx->scale_x);
        if (sy >= ctx->h) sy = ctx->h - 1;
        if (sx >= ctx->w) sx = ctx->w - 1;
        const float *pixel = ctx->src + (sy * ctx->w + sx) * ctx->c;
        float *dest = ctx->out + ((y + ctx->top) * ctx->out_w + x + ctx->left) * ctx->out_c;
        for (size_t c = 0; c < ctx->out_c; ++c)
            dest[c] = channel_value(pixel, ctx->c, c, ctx->expand_gray);
    }
}

/* Paste a nearest-resampled/promoted image into an already allocated canvas.
 * The checked Python caller creates non-overlapping rectangles covering it. */
CN_EXPORT cn_status cn_layout_paste(const float *src, float *out,
    size_t h, size_t w, size_t c, size_t out_h, size_t out_w, size_t out_c,
    size_t top, size_t left, size_t target_h, size_t target_w, int expand_gray)
{
    if (!src || !out || out_c < c || (expand_gray != 0 && expand_gray != 1))
        return CN_INVALID_ARGUMENT;
    cn_status status = layout_size(h, w, c);
    if (status != CN_OK) return status;
    status = layout_size(out_h, out_w, out_c);
    if (status != CN_OK) return status;
    if (!target_h || !target_w || top >= out_h || left >= out_w ||
        target_h > out_h - top || target_w > out_w - left) return CN_INVALID_ARGUMENT;
    paste_context ctx = {src, out, h, w, c, out_w, out_c, top, left,
        target_h, target_w, expand_gray, 1.0 / ((double)target_h / (double)h),
        1.0 / ((double)target_w / (double)w)};
    return cn_parallel_for(target_h * target_w, 65536, paste_range, &ctx);
}

/* NumPy's contiguous float32 reduction tree. The block sizes used here never
 * exceed 1024, below NumPy's separate 8192-element iterator buffering limit. */
static float pair_sum(const float *x, size_t n)
{
    if (n < 8) {
        float sum = -0.0f;
        for (size_t i = 0; i < n; ++i) sum += x[i];
        return sum;
    }
    if (n <= 128) {
        float r[8];
        for (size_t j = 0; j < 8; ++j) r[j] = x[j];
        size_t i = 8;
        for (; i < n - n % 8; i += 8)
            for (size_t j = 0; j < 8; ++j) r[j] += x[i + j];
        float sum = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) sum += x[i];
        return sum;
    }
    size_t half = (n / 2) & ~(size_t)7;
    return pair_sum(x, half) + pair_sum(x + half, n - half);
}

typedef struct zstack_context {
    const float *const *sources;
    float *out;
    size_t count, elements;
    int mode;
} zstack_context;

static void zstack_range(void *opaque, size_t begin, size_t end)
{
    const zstack_context *ctx = opaque;
    for (size_t p = begin; p < end; ++p) {
        float values[15], result = 0.0f;
        for (size_t i = 0; i < ctx->count; ++i) values[i] = ctx->sources[i][p];
        if (ctx->mode == 0) {
            if (ctx->elements == 1) result += pair_sum(values, ctx->count);
            else for (size_t i = 0; i < ctx->count; ++i) result += values[i];
            result /= (float)ctx->count;
        } else if (ctx->mode == 1) {
            int nan_seen = 0;
            for (size_t i = 0; i < ctx->count; ++i) nan_seen |= isnan(values[i]) != 0;
            if (nan_seen) {
                /* np.median partitions kth [mid] (or [mid - 1, mid]) and -1, NaN last;
                   _median_nancheck then returns part[-1], that NaN as stored. */
                size_t kth[3], k = 0, n = ctx->count;
                if (!(n % 2)) kth[k++] = n / 2 - 1;
                kth[k++] = n / 2;
                kth[k++] = n - 1;
                cn_numpy_partition_f32(values, n, kth, k);
                ctx->out[p] = values[n - 1];
                continue;
            }
            for (size_t i = 0; i < ctx->count; ++i) {
                float value = values[i];
                size_t j = i;
                while (j > 0 && values[j - 1] > value) { values[j] = values[j - 1]; --j; }
                values[j] = value;
            }
            if (ctx->count % 2) result = 0.0f + values[ctx->count / 2];
            else result = ((0.0f + values[ctx->count / 2 - 1]) + values[ctx->count / 2]) / 2.0f;
        } else {
            result = values[0];
            for (size_t i = 1; i < ctx->count; ++i) {
                if (isnan(result)) break;
                if (isnan(values[i]) || (ctx->mode == 2 ? values[i] <= result : values[i] >= result))
                    result = values[i];
            }
        }
        ctx->out[p] = result;
    }
}

CN_EXPORT cn_status cn_layout_zstack(const float *const *sources, float *out,
    size_t elements, size_t count, int mode)
{
    if (!sources || !out || count < 2 || count > 15 || mode < 0 || mode > 3)
        return CN_INVALID_ARGUMENT;
    cn_status status = layout_size(1, elements, 1);
    if (status != CN_OK) return status;
    for (size_t i = 0; i < count; ++i) if (!sources[i]) return CN_INVALID_ARGUMENT;
    zstack_context ctx = {sources, out, count, elements, mode};
    return cn_parallel_for(elements, 65536, zstack_range, &ctx);
}

typedef struct pixelate_context {
    const float *src;
    float *out;
    size_t h, w, c, sx, sy, blocks_x;
} pixelate_context;

static void pixelate_blocks(void *opaque, size_t begin, size_t end)
{
    const pixelate_context *ctx = opaque;
    float columns[1024], rows[1024];
    for (size_t block = begin; block < end; ++block) {
        size_t by = (block / ctx->blocks_x) * ctx->sy;
        size_t bx = (block % ctx->blocks_x) * ctx->sx;
        for (size_t c = 0; c < ctx->c; ++c) {
            for (size_t x = 0; x < ctx->sx; ++x) {
                size_t source_x = reflected(bx + x, ctx->w);
                float sum = 0.0f;
                for (size_t y = 0; y < ctx->sy; ++y) {
                    float value = ctx->src[(reflected(by + y, ctx->h) * ctx->w + source_x) * ctx->c + c];
                    rows[y] = value;
                    sum += value;
                }
                if (ctx->blocks_x == 1 && ctx->sx == 1 && ctx->c == 1)
                    sum = 0.0f + pair_sum(rows, ctx->sy);
                columns[x] = sum / (float)ctx->sy;
            }
            float average = 0.0f;
            if (ctx->c == 1) average += pair_sum(columns, ctx->sx);
            else for (size_t x = 0; x < ctx->sx; ++x) average += columns[x];
            average /= (float)ctx->sx;
            for (size_t y = by; y < by + ctx->sy && y < ctx->h; ++y)
                for (size_t x = bx; x < bx + ctx->sx && x < ctx->w; ++x)
                    ctx->out[(y * ctx->w + x) * ctx->c + c] = average;
        }
    }
}

CN_EXPORT cn_status cn_layout_pixelate(const float *src, float *out,
    size_t h, size_t w, size_t c, size_t sx, size_t sy)
{
    if (!src || !out || sx < 1 || sy < 1 || sx > 1024 || sy > 1024) return CN_INVALID_ARGUMENT;
    cn_status status = layout_size(h, w, c);
    if (status != CN_OK) return status;
    if (h > SIZE_MAX / 2 || w > SIZE_MAX / 2 || h > SIZE_MAX - sy || w > SIZE_MAX - sx)
        return CN_SIZE_OVERFLOW;
    size_t blocks_x = (w - 1) / sx + 1, blocks_y = (h - 1) / sy + 1;
    pixelate_context ctx = {src, out, h, w, c, sx, sy, blocks_x};
    size_t grain = 65536 / (sx * sy);
    if (grain == 0) grain = 1;
    return cn_parallel_for(blocks_x * blocks_y, grain, pixelate_blocks, &ctx);
}

typedef struct rotate_context {
    const uint8_t *src;
    float *out;
    size_t h, w, c, out_w, out_c;
    int quarter;
} rotate_context;

static void rotate_range(void *opaque, size_t begin, size_t end)
{
    const rotate_context *ctx = opaque;
    for (size_t p = begin; p < end; ++p) {
        size_t y = p / ctx->out_w, x = p % ctx->out_w, sy = y, sx = x;
        if (ctx->quarter == 1) { sy = x; sx = ctx->w - 1 - y; }
        else if (ctx->quarter == 2) { sy = ctx->h - 1 - y; sx = ctx->w - 1 - x; }
        else if (ctx->quarter == 3) { sy = ctx->h - 1 - x; sx = y; }
        const uint8_t *pixel = ctx->src + (sy * ctx->w + sx) * ctx->c;
        for (size_t c = 0; c < ctx->out_c; ++c) {
            uint8_t value;
            if (ctx->c == 1 && ctx->out_c == 4 && c < 3) value = pixel[0];
            else value = c < ctx->c ? pixel[c] : 255;
            ctx->out[p * ctx->out_c + c] = (float)value / 255.0f;
        }
    }
}

/* Pillow's 0/90/180/270 transpose cases: coordinate transform, optional alpha
 * expansion, and normalization in one pass. Generic resampling remains Pillow. */
CN_EXPORT cn_status cn_layout_rotate_u8(const uint8_t *src, float *out,
    size_t h, size_t w, size_t c, size_t out_c, int quarter)
{
    if (!src || !out || c > 4 || quarter < 0 || quarter > 3 ||
        !(out_c == c || (out_c == 4 && (c == 1 || c == 3)))) return CN_INVALID_ARGUMENT;
    cn_status status = layout_size(h, w, c);
    if (status != CN_OK) return status;
    status = layout_size(h, w, out_c);
    if (status != CN_OK) return status;
    rotate_context ctx = {src, out, h, w, c, quarter % 2 ? h : w, out_c, quarter};
    return cn_parallel_for(h * w, 65536, rotate_range, &ctx);
}
