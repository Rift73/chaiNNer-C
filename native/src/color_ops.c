#include "chainner.h"
#include "color_shared.h"
#include "numeric.h"
#include "numpy_clip_f32.h"
#include "parallel.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

static cn_status valid_floats(size_t count) {
    return count > SIZE_MAX / sizeof(float) ? CN_SIZE_OVERFLOW : CN_OK;
}

typedef struct linear_job {
    const float *src;
    float *dst;
    float factor, offset;
} linear_job;

static void linear_range(void *opaque, size_t begin, size_t end) {
    const linear_job *job = opaque;
    for (size_t i = begin; i < end; ++i) {
        float value = job->src[i] * job->factor;
        job->dst[i] = value + job->offset;
    }
}

CN_EXPORT cn_status cn_color_linear(const float *src, float *dst, size_t count,
                                    float factor, float offset) {
    if (valid_floats(count) != CN_OK) return CN_SIZE_OVERFLOW;
    if (count && (!src || !dst)) return CN_INVALID_ARGUMENT;
    linear_job job = {src, dst, factor, offset};
    return cn_parallel_for(count, 65536, linear_range, &job);
}

typedef struct hls_job {
    float *image;
    float hue, saturation;
} hls_job;

static void hls_range(void *opaque, size_t begin, size_t end) {
    const hls_job *job = opaque;
    for (size_t i = begin; i < end; ++i) {
        if (job->hue != 0.0f) {
            float hue = job->image[i * 3] + job->hue;
            if (hue >= 360.0f) hue -= 360.0f;
            if (hue < 0.0f) hue += 360.0f;
            job->image[i * 3] = hue;
        }
        if (job->saturation != 1.0f) {
            float sat = job->image[i * 3 + 2] * job->saturation;
            job->image[i * 3 + 2] = job->saturation > 1.0f ? cn_numpy_clip_f32(sat, 0.0f, 1.0f) : sat;
        }
    }
}

CN_EXPORT cn_status cn_hls_adjust(float *image, size_t pixels, float hue,
                                 float saturation_factor) {
    if (pixels > SIZE_MAX / sizeof(float) / 3) return CN_SIZE_OVERFLOW;
    if (pixels && !image) return CN_INVALID_ARGUMENT;
    hls_job job = {image, hue, saturation_factor};
    return cn_parallel_for(pixels, 65536, hls_range, &job);
}

typedef struct threshold_job {
    const float *src, *binary;
    float *dst;
    float threshold, maximum;
    int type;
} threshold_job;

static void threshold_range(void *opaque, size_t begin, size_t end) {
    const threshold_job *job = opaque;
    for (size_t i = begin; i < end; ++i) {
        float value = job->src[i];
        float result;
        if (job->binary) {
            float binary = job->binary[i];
            if (job->type == 1) binary = 1.0f - binary;
            switch (job->type) {
            case 0: case 1:
                result = job->maximum < 1.0f ? binary * job->maximum : binary;
                break;
            case 2: {
                float first = binary * job->threshold;
                float second = value * (1.0f - binary);
                result = first + second;
                break;
            }
            case 3: result = binary * value; break;
            default: result = (1.0f - binary) * value; break;
            }
        } else {
            switch (job->type) {
            case 0: result = value > job->threshold ? job->maximum : 0.0f; break;
            case 1: result = value <= job->threshold ? job->maximum : 0.0f; break;
            case 2: result = value < job->threshold ? value : job->threshold; break;
            case 3: result = value > job->threshold ? value : 0.0f; break;
            default: result = value <= job->threshold ? value : 0.0f; break;
            }
        }
        job->dst[i] = result;
    }
}

CN_EXPORT cn_status cn_threshold_f32(const float *src, const float *binary,
                                    float *dst, size_t count, int type,
                                    float threshold, float maximum) {
    if (valid_floats(count) != CN_OK) return CN_SIZE_OVERFLOW;
    if (type < 0 || type > 4 || (count && (!src || !dst))) return CN_INVALID_ARGUMENT;
    threshold_job job = {src, binary, dst, threshold, maximum, type};
    return cn_parallel_for(count, 65536, threshold_range, &job);
}

/* NumPy round(float32 * 255).astype(uint8): ties-to-even, wrap finite values,
 * and convert NaN/infinity to zero. Avoid undefined out-of-range C casts. */
static uint8_t quantize(float value) {
    float scaled = value * 255.0f;
    if (!isfinite(scaled)) return 0;
    float rounded = nearbyintf(scaled);
    if (rounded >= 0.0f && rounded <= 255.0f) return (uint8_t)rounded;
    float wrapped = fmodf(rounded, 256.0f);
    if (wrapped < 0.0f) wrapped += 256.0f;
    return (uint8_t)wrapped;
}

typedef struct quantize_job {
    const float *src;
    uint8_t *dst;
} quantize_job;

static void quantize_range(void *opaque, size_t begin, size_t end) {
    const quantize_job *job = opaque;
    for (size_t i = begin; i < end; ++i) job->dst[i] = quantize(job->src[i]);
}

CN_EXPORT cn_status cn_quantize_u8(const float *src, uint8_t *dst, size_t count) {
    if (valid_floats(count) != CN_OK) return CN_SIZE_OVERFLOW;
    if (count && (!src || !dst)) return CN_INVALID_ARGUMENT;
    quantize_job job = {src, dst};
    return cn_parallel_for(count, 65536, quantize_range, &job);
}

typedef struct histogram_job {
    const float *src;
    uint64_t *histograms;
    size_t pixels, channels;
} histogram_job;

static void histogram_range(void *opaque, size_t begin, size_t end) {
    const histogram_job *job = opaque;
    for (size_t block = begin; block < end; ++block) {
        size_t first = block * 65536;
        size_t last = job->pixels - first < 65536 ? job->pixels : first + 65536;
        uint64_t *histogram = job->histograms + block * 256;
        for (size_t pixel = first; pixel < last; ++pixel) {
            float value = job->src[pixel * job->channels];
            if (job->channels > 1) {
                for (size_t channel = 1; channel < job->channels; ++channel)
                    value += job->src[pixel * job->channels + channel];
                value /= (float)job->channels;
            }
            ++histogram[quantize(value)];
        }
    }
}

/* Otsu's between-class variance and Triangle's unnormalized line-distance
 * criterion. Tie direction and empty-class handling follow the OpenCV 4.8
 * behavior used by chaiNNer; histogram counts are reduced exactly as integers.
 * Reference: https://github.com/opencv/opencv/blob/4.8.0/modules/imgproc/src/thresh.cpp */
static int otsu(const uint64_t histogram[256], size_t pixels) {
    const double inverse_count = 1.0 / (double)pixels;
    double mean = 0.0;
    for (int bin = 0; bin < 256; ++bin) mean += (double)bin * (double)histogram[bin];
    mean *= inverse_count;
    double left_mean = 0.0, left_weight = 0.0, best_variance = 0.0;
    int selected = 0;
    for (int bin = 0; bin < 256; ++bin) {
        double probability = (double)histogram[bin] * inverse_count;
        left_mean *= left_weight;
        left_weight += probability;
        double right_weight = 1.0 - left_weight;
        if (left_weight < FLT_EPSILON || right_weight < FLT_EPSILON ||
            left_weight > 1.0 - FLT_EPSILON || right_weight > 1.0 - FLT_EPSILON)
            continue;
        left_mean = (left_mean + (double)bin * probability) / left_weight;
        double right_mean = (mean - left_weight * left_mean) / right_weight;
        double difference = left_mean - right_mean;
        double variance = left_weight * right_weight * difference * difference;
        if (variance > best_variance) {
            best_variance = variance;
            selected = bin;
        }
    }
    return selected;
}

static int triangle(uint64_t histogram[256]) {
    int left = 0, right = 255, peak = 0;
    while (left < 255 && !histogram[left]) ++left;
    while (right > 0 && !histogram[right]) --right;
    if (left > 0) --left;
    if (right < 255) ++right;
    for (int bin = 1; bin < 256; ++bin)
        if (histogram[bin] > histogram[peak]) peak = bin;
    double peak_count = (double)histogram[peak];
    int reversed = peak - left < right - peak;
    if (reversed) {
        for (int bin = 0; bin < 128; ++bin) {
            uint64_t swap = histogram[bin];
            histogram[bin] = histogram[255 - bin];
            histogram[255 - bin] = swap;
        }
        left = 255 - right;
        peak = 255 - peak;
    }
    int selected = left;
    double maximum_distance = 0.0;
    for (int bin = left + 1; bin <= peak; ++bin) {
        double distance = peak_count * (double)bin + (double)(left - peak) * (double)histogram[bin];
        if (distance > maximum_distance) {
            maximum_distance = distance;
            selected = bin;
        }
    }
    --selected;
    return reversed ? 255 - selected : selected;
}

CN_EXPORT cn_status cn_auto_threshold(const float *src, size_t pixels,
                                      size_t channels, int method, int *out) {
    if (!src || !out || !pixels || !channels || channels > 7 || (method != 0 && method != 1))
        return CN_INVALID_ARGUMENT;
    if (pixels > SIZE_MAX / sizeof(float) / channels) return CN_SIZE_OVERFLOW;
    size_t blocks = 1 + (pixels - 1) / 65536;
    if (blocks > SIZE_MAX / sizeof(uint64_t) / 256) return CN_SIZE_OVERFLOW;
    uint64_t *histograms = calloc(blocks * 256, sizeof(uint64_t));
    if (!histograms) return CN_ALLOCATION_FAILED;
    histogram_job job = {src, histograms, pixels, channels};
    cn_status status = cn_parallel_for(blocks, 1, histogram_range, &job);
    if (status == CN_OK) {
        for (size_t block = 1; block < blocks; ++block)
            for (size_t bin = 0; bin < 256; ++bin)
                histograms[bin] += histograms[block * 256 + bin];
        *out = method == 0 ? otsu(histograms, pixels) : triangle(histograms);
    }
    free(histograms);
    return status;
}

typedef struct adaptive_job {
    const uint8_t *src, *mean;
    uint8_t *dst;
    int delta, inverted;
    uint8_t maximum;
} adaptive_job;

static void adaptive_range(void *opaque, size_t begin, size_t end) {
    const adaptive_job *job = opaque;
    for (size_t i = begin; i < end; ++i) {
        int difference = (int)job->src[i] - (int)job->mean[i];
        int white = job->inverted ? difference <= -job->delta : difference > -job->delta;
        job->dst[i] = white ? job->maximum : 0;
    }
}

CN_EXPORT cn_status cn_adaptive_apply(const uint8_t *src, const uint8_t *mean,
                                      uint8_t *dst, size_t count, int delta,
                                      int inverted, int maximum) {
    if (count && (!src || !mean || !dst)) return CN_INVALID_ARGUMENT;
    if (delta < -1000000 || delta > 1000000 || (inverted != 0 && inverted != 1) ||
        maximum < 0 || maximum > 255) return CN_INVALID_ARGUMENT;
    uintptr_t a = (uintptr_t)src, b = (uintptr_t)mean, out = (uintptr_t)dst;
    if (a > UINTPTR_MAX - count || b > UINTPTR_MAX - count ||
        out > UINTPTR_MAX - count) return CN_SIZE_OVERFLOW;
    if (count && ((a < out + count && out < a + count) ||
                  (b < out + count && out < b + count))) return CN_INVALID_ARGUMENT;
    adaptive_job job = {src, mean, dst, delta, inverted, (uint8_t)maximum};
    return cn_parallel_for(count, 65536, adaptive_range, &job);
}

typedef struct material_job {
    const float *a, *b, *mask;
    float *dst;
    float minimum, span;
    int op;
} material_job;

static void material_range(void *opaque, size_t begin, size_t end) {
    const material_job *job = opaque;
    for (size_t i = begin; i < end; ++i) {
        if (job->op == 3) {
            float b = job->a[i * 3], g = job->a[i * 3 + 1], r = job->a[i * 3 + 2];
            float value = isnan(g) || g >= r ? g : r;
            value = isnan(b) || b >= value ? b : value;
            if (isnan(r) || isnan(g)) value = cn_f32_from_bits(CN_NPY_NANF_BITS);
            value = (value - job->minimum) / job->span;
            job->dst[i] = cn_numpy_clip_f32(value, 0.0f, 1.0f);
            continue;
        }
        for (size_t c = 0; c < 3; ++c) {
            size_t index = i * 3 + c;
            float result;
            if (job->op == 0) result = 1.0f - job->a[i];
            else if (job->op == 1) result = job->a[index] * job->b[index];
            else if (job->op == 2) {
                float left = job->mask[i] * job->a[index];
                float right = (1.0f - job->mask[i]) * 0.22f;
                result = left + right;
            } else {
                float left = job->mask[i] * job->a[index];
                float right = (1.0f - job->mask[i]) * job->b[index];
                result = left + right;
            }
            job->dst[index] = result;
        }
    }
}

CN_EXPORT cn_status cn_material_f32(const float *a, const float *b, const float *mask,
                                   float *dst, size_t pixels, int op,
                                   float minimum, float span) {
    if (op < 0 || op > 4 || (pixels && (!a || !dst))) return CN_INVALID_ARGUMENT;
    if (pixels && ((op == 1 || op == 4) && !b)) return CN_INVALID_ARGUMENT;
    if (pixels && ((op == 2 || op == 4) && !mask)) return CN_INVALID_ARGUMENT;
    if (pixels > SIZE_MAX / sizeof(float) / 3) return CN_SIZE_OVERFLOW;
    material_job job = {a, b, mask, dst, minimum, span, op};
    return cn_parallel_for(pixels, 65536, material_range, &job);
}

typedef struct height_job {
    const float *src;
    float *dst;
    size_t channels;
    int mode;
    float minimum, scale;
} height_job;

static void height_range(void *opaque, size_t begin, size_t end) {
    const height_job *job = opaque;
    for (size_t i = begin; i < end; ++i) {
        const float *p = job->src + i * job->channels;
        float value;
        if (job->mode == 6) value = job->channels == 4 ? p[3] : 1.0f;
        else if (job->channels == 1) {
            value = p[0];
            if (job->mode == 2) {
                float inverse = 1.0f - value;
                value = inverse * inverse;
                value *= inverse;
                value = 1.0f - value;
            }
        } else if (job->mode >= 3) value = p[5 - job->mode];
        else if (job->mode == 0) {
            value = p[2] + p[1];
            value += p[0];
            value /= 3.0f;
        } else if (job->mode == 1) {
            value = isnan(p[2]) || p[2] >= p[1] ? p[2] : p[1];
            value = isnan(value) || value >= p[0] ? value : p[0];
            if (isnan(p[1]) || isnan(p[0])) value = cn_f32_from_bits(CN_NPY_NANF_BITS);
        } else {
            value = (1.0f - p[2]) * (1.0f - p[1]);
            value *= 1.0f - p[0];
            value = 1.0f - value;
        }
        job->dst[i] = value;
    }
}

CN_EXPORT cn_status cn_height_map(const float *src, float *dst, size_t pixels,
                                 size_t channels, int mode) {
    if ((channels != 1 && channels != 3 && channels != 4) || mode < 0 || mode > 6 ||
        (pixels && (!src || !dst))) return CN_INVALID_ARGUMENT;
    if (pixels > SIZE_MAX / sizeof(float) / channels) return CN_SIZE_OVERFLOW;
    height_job job = {src, dst, channels, mode, 0.0f, 0.0f};
    return cn_parallel_for(pixels, 65536, height_range, &job);
}

static void prepare_range(void *opaque, size_t begin, size_t end) {
    const height_job *job = opaque;
    for (size_t i = begin; i < end; ++i) {
        float value = job->src[i];
        if (job->minimum > 0.0f && value < job->minimum) value = job->minimum;
        /* The original node skips multiplication when its Scale input is zero. */
        if (job->scale != 0.0f) value *= job->scale;
        job->dst[i] = value;
    }
}

CN_EXPORT cn_status cn_height_prepare(const float *src, float *dst, size_t count,
                                     float minimum, float scale) {
    if (valid_floats(count) != CN_OK) return CN_SIZE_OVERFLOW;
    if (count && (!src || !dst)) return CN_INVALID_ARGUMENT;
    height_job job = {src, dst, 1, 0, minimum, scale};
    return cn_parallel_for(count, 65536, prepare_range, &job);
}

/* The pixels of [begin, end): through the AVX2 unit at avx2, else normal_pixels
 * (color_shared.h). */
static void normal_output_range(void *opaque, size_t begin, size_t end) {
    const normal_output_job *job = opaque;
#if defined(_MSC_VER) && defined(_M_X64)
    if (job->isa >= CN_ISA_AVX2) {
        cn_normal_output_avx2(job, begin, end);
        return;
    }
#endif
    normal_pixels(job, begin, end);
}

CN_EXPORT cn_status cn_normal_output(const float *dx, const float *dy, const float *alpha,
                                     float *dst, size_t pixels, int invert_r,
                                     int invert_g, int channels) {
    if ((channels != 3 && channels != 4) || (invert_r != 0 && invert_r != 1) ||
        (invert_g != 0 && invert_g != 1) || (pixels && (!dx || !dy || !dst)))
        return CN_INVALID_ARGUMENT;
    if (pixels > SIZE_MAX / sizeof(float) / (size_t)channels) return CN_SIZE_OVERFLOW;
    normal_output_job job = {dx, dy, alpha, dst, invert_r, invert_g, channels, cn_isa_current()};
    return cn_parallel_for(pixels, 65536, normal_output_range, &job);
}
