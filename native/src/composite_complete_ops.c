/* Blend Images canvas construction around the existing exact C blend modes.
 * SP4b D13: a full overlap blends the layers straight into out (full_overlap); any
 * other geometry fills the canvas, gathers the base and overlay regions, blends them
 * and pastes the result, its three region buffers carved from one lease of the shared
 * scratch pool (D10). The fill and the gather walk row spans (layer_run). Every output
 * keeps B3's bits: every value is copied from a layer, or is 0.0f, 1.0f or a blend. */
#include "chainner.h"
#include "parallel.h"
#include "scratch.h"
#include <stdint.h>
#include <string.h>

cn_status cn_blend_images(const float *, const float *, float *, size_t, int, int, int);
typedef struct { const float *data; size_t height, width, channels; int constant; } canvas_layer;
typedef struct {
    size_t height, width, channels, base_top, base_left, paste_top, paste_left;
    size_t overlay_top, overlay_left, region_height, region_width;
    int padded;
} canvas_geometry;
typedef struct {
    const canvas_layer *base, *overlay;
    const canvas_geometry *g;
    float *out, *base_region, *overlay_region, *blended;
    size_t canvas_channels;
} canvas_job;

static int channels_valid(size_t channels) { return channels == 1 || channels == 3 || channels == 4; }
static float layer_value(const canvas_layer *layer, size_t pixel, size_t channel) {
    if (channel == 3 && layer->channels < 4) return 1.0f;
    return layer->data[(layer->constant ? 0 : pixel * layer->channels) + (layer->channels == 1 ? 0 : channel)];
}
/* count pixels of `channels` channels from the layer's pixel `pixel` on, as layer_value
   gives them: a constant layer's one pixel each time; an image's pixels as they are when
   it has `channels` channels, else expanded (a gray value in each colour channel, alpha
   1 where it has none). channels is at most 4. */
static void layer_run(const canvas_layer *layer, size_t pixel, size_t count, size_t channels, float *out) {
    if (layer->constant) {
        float value[4];
        for (size_t c = 0; c < channels; ++c) value[c] = layer_value(layer, 0, c);
        for (size_t p = 0; p < count; ++p, out += channels)
            for (size_t c = 0; c < channels; ++c) out[c] = value[c];
    } else if (layer->channels == channels) {
        memcpy(out, layer->data + pixel * channels, count * channels * sizeof(float));
    } else {
        for (size_t p = 0; p < count; ++p, out += channels)
            for (size_t c = 0; c < channels; ++c) out[c] = layer_value(layer, pixel + p, c);
    }
}
/* Canvas pixels [begin, end), row span by row span: zeros left and right of the base,
   the base's pixels inside it (0.0f is all zero bits). */
static void canvas_fill(void *opaque, size_t begin, size_t end) {
    const canvas_job *job = opaque;
    const canvas_geometry *g = job->g;
    const canvas_layer *base = job->base;
    size_t channels = g->channels;
    for (size_t p = begin; p < end;) {
        size_t y = p / g->width, x = p % g->width;
        size_t stop = x + (end - p < g->width - x ? end - p : g->width - x);
        float *row = job->out + y * g->width * channels;
        size_t lo = stop, hi = stop;
        if (y >= g->base_top && y - g->base_top < base->height) {
            size_t right = g->base_left + base->width;
            lo = x > g->base_left ? x : g->base_left;
            lo = lo < stop ? lo : stop;
            hi = right > lo ? right : lo;
            hi = hi < stop ? hi : stop;
        }
        if (lo > x) memset(row + x * channels, 0, (lo - x) * channels * sizeof(float));
        if (hi > lo)
            layer_run(base, (y - g->base_top) * base->width + lo - g->base_left, hi - lo, channels,
                row + lo * channels);
        if (stop > hi) memset(row + hi * channels, 0, (stop - hi) * channels * sizeof(float));
        p += stop - x;
    }
}
/* Region pixels [begin, end), row span by row span: the base region's canvas_channels
   of each canvas pixel under the paste, and the overlay region's pixels. */
static void canvas_gather(void *opaque, size_t begin, size_t end) {
    const canvas_job *job = opaque;
    const canvas_geometry *g = job->g;
    const canvas_layer *overlay = job->overlay;
    size_t canvas_channels = job->canvas_channels;
    for (size_t p = begin; p < end;) {
        size_t y = p / g->region_width, x = p % g->region_width;
        size_t count = end - p < g->region_width - x ? end - p : g->region_width - x;
        const float *source = job->out + ((y + g->paste_top) * g->width + x + g->paste_left) * g->channels;
        float *base_region = job->base_region + p * canvas_channels;
        if (canvas_channels == g->channels) {
            memcpy(base_region, source, count * canvas_channels * sizeof(float));
        } else {
            for (size_t i = 0; i < count; ++i, base_region += canvas_channels, source += g->channels)
                for (size_t c = 0; c < canvas_channels; ++c) base_region[c] = source[c];
        }
        layer_run(overlay, (y + g->overlay_top) * overlay->width + x + g->overlay_left, count,
            overlay->channels, job->overlay_region + p * overlay->channels);
        p += count;
    }
}
static void canvas_paste(void *opaque, size_t begin, size_t end) {
    const canvas_job *job = opaque;
    const canvas_geometry *g = job->g;
    for (size_t y = begin; y < end; ++y)
        memcpy(job->out + ((y + g->paste_top) * g->width + g->paste_left) * g->channels,
            job->blended + y * g->region_width * g->channels,
            g->region_width * g->channels * sizeof(float));
}
/* D13: the paste covers the whole canvas from (0, 0); the base fills the canvas from
   (0, 0), is not constant and has the canvas's channels; the overlay is not constant and
   its region is all of it, from (0, 0). Then the gathered base region is base->data and
   the overlay region is overlay->data value for value, and the paste copies the blend's
   rows into out unchanged, so blending the layers straight into out gives the same bits
   (blend's partition is unchanged; the fill, gather and paste are skipped). */
static int full_overlap(const canvas_layer *base, const canvas_layer *overlay,
        const canvas_geometry *g, size_t canvas_channels) {
    return g->paste_top == 0 && g->paste_left == 0 && g->region_height == g->height &&
        g->region_width == g->width && g->base_top == 0 && g->base_left == 0 &&
        base->height == g->height && base->width == g->width && !base->constant &&
        base->channels == canvas_channels && !overlay->constant && g->overlay_top == 0 &&
        g->overlay_left == 0 && overlay->height == g->region_height &&
        overlay->width == g->region_width;
}
static cn_status layer_validate(const canvas_layer *layer) {
    if (!layer || !layer->data || !layer->height || !layer->width || !channels_valid(layer->channels)
        || (layer->constant != 0 && layer->constant != 1) || (uintptr_t)layer->data % _Alignof(float)) return CN_INVALID_ARGUMENT;
    if (layer->height > SIZE_MAX / layer->width || layer->height * layer->width > SIZE_MAX / layer->channels / sizeof(float)) return CN_SIZE_OVERFLOW;
    size_t bytes = (layer->constant ? 1 : layer->height * layer->width) * layer->channels * sizeof(float);
    if ((uintptr_t)layer->data > UINTPTR_MAX - bytes) return CN_SIZE_OVERFLOW;
    return CN_OK;
}
CN_EXPORT cn_status cn_composite_canvas(const canvas_layer *base, const canvas_layer *overlay,
        const canvas_geometry *g, float *out, int mode) {
    cn_status status = layer_validate(base);
    if (status != CN_OK) return status;
    status = layer_validate(overlay);
    if (status != CN_OK) return status;
    if (!g || !out || !g->height || !g->width || !channels_valid(g->channels) || mode < 0 || mode > 22
        || (g->padded != 0 && g->padded != 1) || (uintptr_t)out % _Alignof(float)) return CN_INVALID_ARGUMENT;
    if (g->height > SIZE_MAX / g->width || g->height * g->width > SIZE_MAX / g->channels / sizeof(float)) return CN_SIZE_OVERFLOW;
    size_t out_bytes = g->height * g->width * g->channels * sizeof(float);
    if ((uintptr_t)out > UINTPTR_MAX - out_bytes) return CN_SIZE_OVERFLOW;
    if (g->base_top > g->height || base->height > g->height - g->base_top ||
        g->base_left > g->width || base->width > g->width - g->base_left ||
        g->paste_top > g->height || g->region_height > g->height - g->paste_top ||
        g->paste_left > g->width || g->region_width > g->width - g->paste_left ||
        g->overlay_top > overlay->height || g->region_height > overlay->height - g->overlay_top ||
        g->overlay_left > overlay->width || g->region_width > overlay->width - g->overlay_left) return CN_INVALID_ARGUMENT;
    size_t canvas_channels = g->padded ? 4 : base->channels;
    size_t pixels = g->region_height * g->region_width;
    size_t result_channels = pixels && overlay->channels > canvas_channels ? overlay->channels : canvas_channels;
    if (g->channels != result_channels) return CN_INVALID_ARGUMENT;
    const canvas_layer *layers[] = {base, overlay};
    for (size_t i = 0; i < 2; ++i) {
        uintptr_t address = (uintptr_t)layers[i]->data;
        size_t bytes = (layers[i]->constant ? 1 : layers[i]->height * layers[i]->width) * layers[i]->channels * sizeof(float);
        if (address < (uintptr_t)out + out_bytes && (uintptr_t)out < address + bytes) return CN_INVALID_ARGUMENT;
    }
    if (full_overlap(base, overlay, g, canvas_channels))
        return cn_blend_images(overlay->data, base->data, out, pixels, (int)overlay->channels,
            (int)canvas_channels, mode);
    /* The overlay region (a), base region (b) and blend (blended): one pooled lease at
       64-byte offsets, each written whole before it is read (the gather writes a and b,
       the blend writes blended). Each buffer is at most out_bytes (pixels is at most
       height * width, and every channel count at most g->channels), so only their sum
       can overflow, which cn_scratch_carve refuses: CN_ALLOCATION_FAILED, as B3's
       mallocs. */
    cn_scratch lease = {NULL, 0, -1};
    float *a = NULL, *b = NULL, *blended = NULL;
    if (pixels) {
        size_t total = 0, a_at = 0, b_at = 0, blended_at = 0;
        if (!cn_scratch_carve(&total, pixels * overlay->channels * sizeof(float), &a_at) ||
            !cn_scratch_carve(&total, pixels * canvas_channels * sizeof(float), &b_at) ||
            !cn_scratch_carve(&total, pixels * result_channels * sizeof(float), &blended_at))
            return CN_ALLOCATION_FAILED;
        status = cn_scratch_lease(total, &lease, NULL);
        if (status != CN_OK) return status;
        a = (float *)((char *)lease.data + a_at);
        b = (float *)((char *)lease.data + b_at);
        blended = (float *)((char *)lease.data + blended_at);
    }
    canvas_job job = {base, overlay, g, out, b, a, blended, canvas_channels};
    status = cn_parallel_for(g->height * g->width, 65536, canvas_fill, &job);
    if (status == CN_OK && pixels) status = cn_parallel_for(pixels, 65536, canvas_gather, &job);
    if (status == CN_OK && pixels) status = cn_blend_images(a, b, blended, pixels, (int)overlay->channels, (int)canvas_channels, mode);
    if (status == CN_OK && pixels) status = cn_parallel_for(g->region_height, 1 + 65536 / g->region_width, canvas_paste, &job);
    cn_scratch_release(&lease);
    return status;
}
