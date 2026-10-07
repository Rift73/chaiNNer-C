/* Transparency preparation and color representation arithmetic. Specialized
 * morphology, color transforms, transcendental ufuncs and matting remain in
 * their existing libraries. Inputs are read-only and outputs own their memory. */
#include "parallel.h"
#include <math.h>

/* Images are row-major pixel grids: pixel p's channel c at p * pixel + c * channel
 * elements, interleaved (channels, 1) or planar (1, plane) (D-16: upstream's
 * np.dstack keeps a planar input's layout). */
typedef struct {
    const float *image, *key;
    float *out;
    uint8_t *bg, *fg;
    float bg_threshold, fg_threshold;
    int masks;
    size_t pixel, channel, out_pixel, out_channel;
} chroma_job;

static inline void chroma_pixel(const chroma_job *j, size_t p, size_t pixel,
        size_t channel, size_t out_pixel, size_t out_channel) {
    const float *v = j->image + p * pixel;
    float diff = fabsf(v[0] - j->key[0]);
    diff += fabsf(v[channel] - j->key[1]);
    diff += fabsf(v[2 * channel] - j->key[2]);
    diff /= 3.0f;
    if (j->masks) {
        j->bg[p] = diff <= j->bg_threshold ? 255 : 0;
        j->fg[p] = diff > j->fg_threshold ? 255 : 0;
    } else {
        float *o = j->out + p * out_pixel;
        for (size_t c = 0; c < 3; ++c) o[c * out_channel] = v[c * channel];
        o[3 * out_channel] = diff > j->bg_threshold ? 1.0f : 0.0f;
    }
}

static void chroma_range(void *opaque, size_t begin, size_t end) {
    const chroma_job *j = opaque;
    if (j->pixel == 3 && j->channel == 1 && j->out_pixel == 4 && j->out_channel == 1)
        for (size_t p = begin; p < end; ++p) chroma_pixel(j, p, 3, 1, 4, 1);
    else
        for (size_t p = begin; p < end; ++p)
            chroma_pixel(j, p, j->pixel, j->channel, j->out_pixel, j->out_channel);
}

/* With masks, out and its strides are unused. */
CN_EXPORT cn_status cn_chroma_key(const float *image, const float *key, size_t pixels,
        size_t pixel_stride, size_t channel_stride, float bg_threshold, float fg_threshold,
        int masks, float *out, size_t out_pixel_stride, size_t out_channel_stride,
        uint8_t *bg, uint8_t *fg) {
    if (!image || !key || masks < 0 || masks > 1 || !pixel_stride || !channel_stride
        || (masks ? (!bg || !fg) : (!out || !out_pixel_stride || !out_channel_stride)))
        return CN_INVALID_ARGUMENT;
    if (masks) {
        out_pixel_stride = 4;
        out_channel_stride = 1;
    }
    if (pixels > SIZE_MAX / 4 / sizeof(float)) return CN_SIZE_OVERFLOW;
    chroma_job job = {image, key, out, bg, fg, bg_threshold, fg_threshold, masks,
        pixel_stride, channel_stride, out_pixel_stride, out_channel_stride};
    return cn_parallel_for(pixels, 65536, chroma_range, &job);
}

typedef struct {
    const uint8_t *bg, *fg;
    double *out;
} trimap_job;

static void trimap_range(void *opaque, size_t begin, size_t end) {
    trimap_job *j = opaque;
    for (size_t p = begin; p < end; ++p)
        j->out[p] = j->fg[p] > 128 ? 1.0 : j->bg[p] > 128 ? 0.0 : 0.5;
}

CN_EXPORT cn_status cn_chroma_trimap(const uint8_t *bg, const uint8_t *fg,
        double *out, size_t pixels) {
    if (!bg || !fg || !out) return CN_INVALID_ARGUMENT;
    if (pixels > SIZE_MAX / sizeof(double)) return CN_SIZE_OVERFLOW;
    trimap_job job = {bg, fg, out};
    return cn_parallel_for(pixels, 65536, trimap_range, &job);
}

typedef struct {
    const float *image;
    const void *trimap;
    double *rgb, *out;
    size_t channels;
    double fg, bg;
    int doubles, threshold;
} matting_job;

static void matting_range(void *opaque, size_t begin, size_t end) {
    matting_job *j = opaque;
    for (size_t p = begin; p < end; ++p) {
        for (size_t c = 0; c < 3; ++c)
            j->rgb[p * 3 + c] = j->image[p * j->channels + 2 - c];
        double value;
        if (j->doubles) {
            value = ((const double *)j->trimap)[p];
            if (j->threshold) {
                if (value > j->fg) value = 1.0;
                if (value < j->bg) value = 0.0;
            }
        } else {
            float v = ((const float *)j->trimap)[p];
            if (j->threshold) {
                if (v > (float)j->fg) v = 1.0f;
                if (v < (float)j->bg) v = 0.0f;
            }
            value = v;
        }
        j->out[p] = value;
    }
}

CN_EXPORT cn_status cn_matting_input(const float *image, size_t channels,
        const void *trimap, int doubles, size_t pixels, double fg, double bg,
        int threshold, double *rgb, double *out) {
    if (!image || !trimap || !rgb || !out || (channels != 3 && channels != 4)
        || doubles < 0 || doubles > 1 || threshold < 0 || threshold > 1)
        return CN_INVALID_ARGUMENT;
    if (pixels > SIZE_MAX / 3 / sizeof(double)) return CN_SIZE_OVERFLOW;
    matting_job job = {image, trimap, rgb, out, channels, fg, bg, doubles, threshold};
    return cn_parallel_for(pixels, 65536, matting_range, &job);
}

typedef struct {
    const float *image, *a, *b, *c, *d;
    float *out;
    int op;
} representation_job;

static float maximum(float a, float b) {
    return isnan(a) ? a : isnan(b) ? b : a > b ? a : b;
}

static void representation_range(void *opaque, size_t begin, size_t end) {
    representation_job *j = opaque;
    for (size_t p = begin; p < end; ++p) {
        const float *v = j->image + p * (j->op == 1 ? 4 : 3);
        float *o = j->out + p * (j->op == 0 ? 4 : 3);
        switch (j->op) {
            case 0: {
                float top = maximum(maximum(v[0], v[1]), v[2]);
                float soft = maximum(top, 0.001f);
                for (size_t c = 0; c < 3; ++c) o[c] = 1.0f - v[c] / soft;
                o[3] = 1.0f - top;
                break;
            }
            case 1: {
                float top = 1.0f - v[3];
                for (size_t c = 0; c < 3; ++c) o[c] = (1.0f - v[c]) * top;
                break;
            }
            case 2: o[0] = v[2]; o[1] = v[1]; o[2] = v[0] / 360.0f; break;
            case 3: o[0] = v[2] * 360.0f; o[1] = v[1]; o[2] = v[0]; break;
            case 4: o[0] = v[1]; o[1] = v[2]; o[2] = v[0] / 360.0f; break;
            case 5: o[0] = v[2] * 360.0f; o[1] = v[0]; o[2] = v[1]; break;
            case 6:
                o[0] = (v[2] + 127.0f) / 254.0f;
                o[1] = (v[1] + 127.0f) / 254.0f;
                o[2] = v[0] / 100.0f;
                break;
            case 7:
                o[0] = v[2] * 100.0f;
                o[1] = v[1] * 254.0f - 127.0f;
                o[2] = v[0] * 254.0f - 127.0f;
                break;
            case 8: {
                float cmax = 0.5f / maximum(fabsf(j->c[p]), fabsf(j->d[p]));
                o[0] = j->b[p] / (float)6.283185307179586 + 0.5f;
                o[1] = j->a[p] / cmax;
                o[2] = v[2];
                break;
            }
            case 9: {
                float cmax = 0.5f / maximum(fabsf(j->a[p]), fabsf(j->b[p]));
                float chroma = v[1] * cmax;
                o[0] = chroma * j->a[p] + 0.5f;
                o[1] = chroma * j->b[p] + 0.5f;
                o[2] = v[2];
                break;
            }
        }
    }
}

CN_EXPORT cn_status cn_color_representation(const float *image, float *out,
        size_t pixels, int operation, const float *a, const float *b,
        const float *c, const float *d) {
    if (!image || !out || operation < 0 || operation > 9
        || (operation == 8 && (!a || !b || !c || !d))
        || (operation == 9 && (!a || !b))) return CN_INVALID_ARGUMENT;
    if (pixels > SIZE_MAX / 4 / sizeof(float)) return CN_SIZE_OVERFLOW;
    representation_job job = {image, a, b, c, d, out, operation};
    return cn_parallel_for(pixels, 65536, representation_range, &job);
}
