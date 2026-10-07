/* chaiNNer Create Noise: coordinate embeddings, seeded tables and fractal
 * composition. Arithmetic order follows the installed NumPy expressions.
 * Pixel workers share immutable tables; each invocation owns its RNG state.
 */
#include "cn_crt_math.h"
#include "procedural.h"
#include "parallel.h"
#include "noise_rng.h"
#include <math.h>
#include <string.h>

static const double pi = 3.14159265358979323846264338327950288;

typedef struct {
    procedural_job generator;
    size_t height, width;
    int horizontal, vertical, spherical, fractal;
    double spatial_scale, brightness;
    void *out;
} create_job;

static void coordinates(const create_job *j, size_t pixel, double *point) {
    double y = (double)(pixel / j->width), x = (double)(pixel % j->width);
    double h = (double)j->height, w = (double)j->width;
    if (j->spherical) {
        double theta = y * pi / h, alpha = (x * 2) * pi / w;
        double radius = h * cn_crt.sin(theta);
        point[0] = radius * cn_crt.cos(alpha) / pi / 2;
        point[1] = w * cn_crt.cos(theta) / pi / 2;
        point[2] = radius * cn_crt.sin(alpha) / pi / 2;
    } else if (j->horizontal && j->vertical) {
        double a = (x * 2) * pi / w, b = (y * 2) * pi / h;
        point[0] = w * cn_crt.cos(a) / pi / 2;
        point[1] = w * cn_crt.sin(a) / pi / 2;
        point[2] = h * cn_crt.cos(b) / pi / 2;
        point[3] = h * cn_crt.sin(b) / pi / 2;
    } else if (j->horizontal) {
        double a = (x * 2) * pi / w;
        point[0] = y;
        point[1] = w * cn_crt.cos(a) / pi / 2;
        point[2] = w * cn_crt.sin(a) / pi / 2;
    } else if (j->vertical) {
        double a = (y * 2) * pi / h;
        point[0] = x;
        point[1] = h * cn_crt.cos(a) / pi / 2;
        point[2] = h * cn_crt.sin(a) / pi / 2;
    } else {
        point[0] = y;
        point[1] = x;
    }
    for (int d = 0; d < j->generator.dimensions; ++d)
        point[d] /= j->spatial_scale;
}

static void create_range(void *opaque, size_t begin, size_t end) {
    create_job *j = opaque;
    for (size_t p = begin; p < end; ++p) {
        double point[4];
        coordinates(j, p, point);
        double value = cn_procedural_sample(&j->generator, point) * j->brightness;
        if (j->fractal) {
            float *out = j->out;
            out[p] = (float)((double)out[p] + value);
        } else ((double *)j->out)[p] = value;
    }
}

typedef struct { float *out; float divisor; } normalize_job;
static void normalize_range(void *opaque, size_t begin, size_t end) {
    normalize_job *j = opaque;
    for (size_t p = begin; p < end; ++p) j->out[p] /= j->divisor;
}

CN_EXPORT cn_status cn_create_procedural(void *out, size_t height, size_t width,
        uint64_t seed, int method, double scale, double brightness,
        int horizontal, int vertical, int spherical, int layers,
        double scale_ratio, double brightness_ratio, int increment_seed) {
    int fractal = layers != 0;
    size_t itemsize = fractal ? sizeof(float) : sizeof(double);
    if (!out || !height || !width || method < 0 || method > 2
        || horizontal < 0 || horizontal > 1 || vertical < 0 || vertical > 1
        || spherical < 0 || spherical > 1 || increment_seed < 0 || increment_seed > 1
        || (layers != 0 && (layers < 2 || layers > 20))
        || !isfinite(scale) || scale < 1 || !isfinite(brightness)
        || brightness < 0 || brightness > 1 || !isfinite(scale_ratio) || scale_ratio < 1
        || !isfinite(brightness_ratio) || brightness_ratio < 1
        || (uintptr_t)out % itemsize) return CN_INVALID_ARGUMENT;
    if (height > SIZE_MAX / width || height * width > SIZE_MAX / itemsize)
        return CN_SIZE_OVERFLOW;
    size_t count = height * width, bytes = count * itemsize;
    if ((uintptr_t)out > UINTPTR_MAX - bytes) return CN_SIZE_OVERFLOW;
    if (seed > UINT64_MAX - 20) return CN_INVALID_ARGUMENT;
    int dimensions = spherical ? 3 : 2 + horizontal + vertical;
    double scales[20], brightnesses[20], total_brightness = 0;
    int iterations = fractal ? layers : 1;
    for (int layer = 0; layer < iterations; ++layer) {
        double power = (double)layer;
        double relative = fractal ? 1 / cn_crt.pow(brightness_ratio, power) : 1;
        scales[layer] = fractal ? scale / cn_crt.pow(scale_ratio, power) : scale;
        brightnesses[layer] = brightness * relative;
        total_brightness += relative;
        if (!isfinite(scales[layer]) || !isfinite(relative)) return CN_INVALID_ARGUMENT;
    }
    double gradients[128] = {0};
    float values[16];
    int32_t table[512];
    int64_t permutation[512];
    size_t gradient_count = dimensions == 2 ? 16 : (size_t)dimensions << (dimensions - 1);
    size_t table_size = method == 2 ? gradient_count * 16 : 256;
    if (dimensions == 2) {
        for (size_t i = 0; i < 16; ++i) {
            double angle = (2 * pi) * (double)i / 16;
            gradients[2 * i] = cn_crt.cos(angle);
            gradients[2 * i + 1] = cn_crt.sin(angle);
        }
    } else {
        int patterns = 1 << (dimensions - 1);
        for (int zero = 0; zero < dimensions; ++zero)
            for (int pattern = 0; pattern < patterns; ++pattern) {
                int bit = dimensions - 2;
                for (int d = 0; d < dimensions; ++d)
                    if (d != zero)
                        gradients[(zero * patterns + pattern) * dimensions + d]
                            = ((pattern >> bit--) & 1) ? 1 : -1;
            }
    }
    for (int i = 0; i < 16; ++i) values[i] = (float)i / 15.0f;
    double root = sqrt((double)dimensions + 1);
    procedural_job generator = {NULL, gradients, values, table, NULL, table_size,
        gradient_count, 16, dimensions, method == 1, method == 2,
        (root - 1) / dimensions, (1 - 1 / root) / dimensions, 0.5,
        dimensions == 2 ? 50 : dimensions == 3 ? 39 : 32};
    create_job job = {generator, height, width, horizontal, vertical, spherical,
                     fractal, scale, brightness, out};
    if (fractal) memset(out, 0, bytes);
    for (int layer = 0; layer < iterations; ++layer) {
        if (layer == 0 || increment_seed) {
            uint64_t current_seed = seed + (increment_seed ? (uint64_t)layer : 0);
            uint32_t words[2] = {(uint32_t)current_seed, (uint32_t)(current_seed >> 32)};
            cn_rng rng;
            cn_rng_seed(&rng, words, words[1] ? 2 : 1);
            for (size_t i = 0; i < table_size; ++i) permutation[i] = (int64_t)i;
            cn_rng_shuffle_i64(&rng, permutation, table_size);
            for (size_t i = 0; i < table_size; ++i) table[i] = (int32_t)permutation[i];
        }
        job.spatial_scale = scales[layer];
        job.brightness = brightnesses[layer];
        cn_status status = cn_parallel_for(count, 8192, create_range, &job);
        if (status != CN_OK) return status;
    }
    if (fractal) {
        normalize_job norm = {out, (float)total_brightness};
        return cn_parallel_for(count, 65536, normalize_range, &norm);
    }
    return CN_OK;
}
