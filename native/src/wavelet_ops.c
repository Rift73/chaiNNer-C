/* CPU wavelet decomposition from the existing Wavelet Color Fix algorithm.
 * Nine taps accumulate in row-major order. The default oneDNN CPU primitive
 * uses fused products; its explicit non-oneDNN normalized-image counterpart
 * uses separate products. The difference is observable for subnormal inputs.
 * Framework device/autograd/reduced-precision modes stay in the adapter. */
#include "parallel.h"
#include <math.h>
#include <stdlib.h>

static cn_status dimensions(size_t batches, size_t height, size_t width, size_t *count) {
    if (!batches || !height || !width) return CN_INVALID_ARGUMENT;
    if (height > SIZE_MAX / width) return CN_SIZE_OVERFLOW;
    size_t plane = height * width;
    if (batches > SIZE_MAX / 3 || plane > SIZE_MAX / (batches * 3) / sizeof(float))
        return CN_SIZE_OVERFLOW;
    *count = batches * 3 * plane;
    return CN_OK;
}

typedef struct wavelet_job {
    const float *source;
    float *low, *high;
    size_t height, width, radius;
    int first, fused;
} wavelet_job;

static cn_status buffers(const float *source, float *output, size_t count) {
    if (!source || !output || (uintptr_t)source % sizeof(float)
        || (uintptr_t)output % sizeof(float)) return CN_INVALID_ARGUMENT;
    size_t bytes = count * sizeof(float);
    uintptr_t a = (uintptr_t)source, b = (uintptr_t)output;
    if (a > UINTPTR_MAX - bytes || b > UINTPTR_MAX - bytes) return CN_SIZE_OVERFLOW;
    if (a < b + bytes && b < a + bytes) return CN_INVALID_ARGUMENT;
    return CN_OK;
}

static size_t sample_position(size_t position, size_t radius, size_t length, size_t tap) {
    if (tap == 0) return position < radius ? 0 : position - radius;
    if (tap == 2) return radius >= length - position ? length - 1 : position + radius;
    return position;
}

static void blur_range(void *opaque, size_t begin, size_t end) {
    wavelet_job *job = opaque;
    static const float weights[9] = {
        0.0625f, 0.125f, 0.0625f, 0.125f, 0.25f, 0.125f, 0.0625f, 0.125f, 0.0625f
    };
    size_t plane_size = job->height * job->width;
    for (size_t i = begin; i < end; ++i) {
        size_t x = i % job->width, y = i / job->width % job->height;
        size_t plane = (i / plane_size) * plane_size;
        float value = 0.0f;
        for (size_t dy = 0; dy < 3; ++dy) {
            size_t yy = sample_position(y, job->radius, job->height, dy);
            for (size_t dx = 0; dx < 3; ++dx) {
                size_t xx = sample_position(x, job->radius, job->width, dx);
                float sample = job->source[plane + yy * job->width + xx];
                float weight = weights[dy * 3 + dx];
                if (job->fused) value = fmaf(sample, weight, value);
                else {
                    float product = sample * weight;
                    value = value + product;
                }
            }
        }
        job->low[i] = value;
        if (job->high) {
            float difference = job->source[i] - value;
            float previous = job->first ? 0.0f : job->high[i];
            job->high[i] = previous + difference;
        }
    }
}

CN_EXPORT cn_status cn_wavelet_blur_f32(const float *source, float *output,
        size_t batches, size_t height, size_t width, size_t radius, int fused) {
    if (!radius || radius > 512 || (fused != 0 && fused != 1)) return CN_INVALID_ARGUMENT;
    size_t count;
    cn_status status = dimensions(batches, height, width, &count);
    if (status != CN_OK) return status;
    status = buffers(source, output, count);
    if (status != CN_OK) return status;
    wavelet_job job = {source, output, NULL, height, width, radius, 0, fused};
    return cn_parallel_for(count, 16384, blur_range, &job);
}

static cn_status decompose(const float *source, float *high, float *low, float *scratch,
        size_t count, size_t height, size_t width, int levels, int fused) {
    const float *current = source;
    for (int level = 0; level < levels; ++level) {
        float *next = (levels - level) % 2 ? low : scratch;
        wavelet_job job = {current, next, high, height, width, (size_t)1 << level, level == 0, fused};
        cn_status status = cn_parallel_for(count, 16384, blur_range, &job);
        if (status != CN_OK) return status;
        current = next;
    }
    return CN_OK;
}

CN_EXPORT cn_status cn_wavelet_decompose_f32(const float *source, float *high, float *low,
        size_t batches, size_t height, size_t width, int levels, int fused) {
    if (levels < 1 || levels > 10 || (fused != 0 && fused != 1)) return CN_INVALID_ARGUMENT;
    size_t count;
    cn_status status = dimensions(batches, height, width, &count);
    if (status != CN_OK) return status;
    status = buffers(source, high, count);
    if (status == CN_OK) status = buffers(source, low, count);
    if (status == CN_OK) status = buffers(high, low, count);
    if (status != CN_OK) return status;
    float *scratch = levels > 1 ? malloc(count * sizeof(float)) : NULL;
    if (levels > 1 && !scratch) return CN_ALLOCATION_FAILED;
    status = decompose(source, high, low, scratch, count, height, width, levels, fused);
    free(scratch);
    return status;
}

typedef struct addition_job { float *high; const float *low; } addition_job;

static void addition_range(void *opaque, size_t begin, size_t end) {
    addition_job *job = opaque;
    for (size_t i = begin; i < end; ++i) job->high[i] = job->high[i] + job->low[i];
}

CN_EXPORT cn_status cn_wavelet_reconstruct_f32(const float *content, const float *style,
        float *output, size_t batches, size_t height, size_t width, int levels, int fused) {
    if (levels < 1 || levels > 10 || (fused != 0 && fused != 1)) return CN_INVALID_ARGUMENT;
    size_t count;
    cn_status status = dimensions(batches, height, width, &count);
    if (status != CN_OK) return status;
    if (count > SIZE_MAX / sizeof(float) / 2) return CN_SIZE_OVERFLOW;
    status = buffers(content, output, count);
    if (status == CN_OK) status = buffers(style, output, count);
    if (status != CN_OK) return status;
    float *memory = malloc(count * sizeof(float) * 2);
    if (!memory) return CN_ALLOCATION_FAILED;
    float *low = memory, *scratch = memory + count;
    status = decompose(content, output, low, scratch, count, height, width, levels, fused);
    if (status == CN_OK)
        status = decompose(style, NULL, low, scratch, count, height, width, levels, fused);
    if (status == CN_OK) {
        addition_job job = {output, low};
        status = cn_parallel_for(count, 65536, addition_range, &job);
    }
    free(memory);
    return status;
}
