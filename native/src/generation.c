/* Fused image construction and deterministic procedural noise evaluation.
 * Tables and random samples are supplied by the existing seeded generators.
 * Per-pixel work is independent; no shared mutable RNG state crosses the ABI.
 */
#include "cn_crt_math.h"
#include "parallel.h"
#include "procedural.h"
#include <math.h>
#include <limits.h>

typedef struct {
    float *out;
    const float *a, *b, *gradient;
    size_t width, height, channels, square;
    int mode;
} fill_job;

static void fill_range(void *opaque, size_t begin, size_t end) {
    fill_job *j = opaque;
    for (size_t p = begin; p < end; ++p) {
        const float *color = j->a;
        if (j->mode == 1 && (((j->height - 1 - p / j->width) / j->square
                              + (p % j->width) / j->square) & 1)) color = j->b;
        for (size_t c = 0; c < j->channels; ++c) {
            float value = color[c];
            if (j->mode == 2) {
                float t = j->gradient[p];
                float left = j->a[c] * (1.0f - t);
                float right = j->b[c] * t;
                value = right + left;
            }
            j->out[p * j->channels + c] = value;
        }
    }
}

CN_EXPORT cn_status cn_image_fill(float *out, size_t height, size_t width,
        size_t channels, const float *a, const float *b, const float *gradient,
        int mode, size_t square) {
    if (!out || !a || !height || !width || !channels || channels > 4
        || mode < 0 || mode > 2 || (mode && !b)
        || (mode == 1 && !square) || (mode == 2 && !gradient)) return CN_INVALID_ARGUMENT;
    if (height > SIZE_MAX / width || height * width > SIZE_MAX / channels / sizeof(float))
        return CN_SIZE_OVERFLOW;
    fill_job job = {out, a, b, gradient, width, height, channels, square, mode};
    return cn_parallel_for(height * width, 65536, fill_range, &job);
}

typedef struct {
    const float *image;
    const void *noise, *salt;
    float *out;
    size_t channels, noise_channels, output_channels;
    int operation, bytes;
} noise_job;

static float sample_noise(const void *data, size_t index, int bytes) {
    return bytes ? (float)((const uint8_t *)data)[index] : ((const float *)data)[index];
}

static void noise_range(void *opaque, size_t begin, size_t end) {
    noise_job *j = opaque;
    for (size_t p = begin; p < end; ++p) {
        for (size_t c = 0; c < j->output_channels; ++c) {
            float value = j->image[p * j->channels + (j->channels == 1 ? 0 : c)];
            if (c < 3) {
                size_t ni = p * j->noise_channels + (j->noise_channels == 1 ? 0 : c);
                float n = sample_noise(j->noise, ni, j->bytes);
                if (j->operation == 0) value += n;
                else if (j->operation == 1) {
                    float product = value * n;
                    value += product;
                } else {
                    float salt = sample_noise(j->salt, ni, j->bytes);
                    value = salt == 1 ? 1 : n == 0 ? 0 : value;
                }
            }
            /* Comparisons preserve NaNs like numpy.clip. */
            j->out[p * j->output_channels + c] = value < 0 ? 0 : value > 1 ? 1 : value;
        }
    }
}

CN_EXPORT cn_status cn_noise_combine(const float *image, size_t pixels, size_t channels,
        const void *noise, const void *salt, size_t noise_channels, int bytes,
        int operation, float *out) {
    if (!image || !noise || !out || !channels || channels == 2
        || (noise_channels != 1 && noise_channels != 3) || bytes < 0 || bytes > 1
        || operation < 0 || operation > 2 || (operation == 2 && !salt)) return CN_INVALID_ARGUMENT;
    size_t output_channels = channels > noise_channels ? channels : noise_channels;
    if (pixels > SIZE_MAX / output_channels / sizeof(float)) return CN_SIZE_OVERFLOW;
    noise_job job = {image, noise, salt, out, channels, noise_channels, output_channels, operation, bytes};
    return cn_parallel_for(pixels, 65536, noise_range, &job);
}

static size_t modulo(int64_t value, size_t divisor) {
    int64_t r = value % (int64_t)divisor;
    return (size_t)(r < 0 ? r + (int64_t)divisor : r);
}

/* NumPy's contiguous reduction uses eight interleaved accumulators for these
 * short power-of-two arrays; retaining that order preserves value-noise bits.
 */
static double sum_corners(const double *v, int n) {
    if (n < 8) {
        double result = -0.0;
        for (int i = 0; i < n; ++i) result += v[i];
        return result;
    }
    double a[8];
    for (int k = 0; k < 8; ++k) a[k] = v[k];
    for (int i = 8; i < n; i += 8)
        for (int k = 0; k < 8; ++k) a[k] += v[i + k];
    return ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]));
}

static double value_at(const procedural_job *j, const double *point) {
    double base[6];
    double fraction[6], contributions[64];
    for (int d = 0; d < j->dimensions; ++d) {
        double block = floor(point[d]);
        base[d] = block;
        double t = point[d] - block;
        if (j->smooth) t = t * t * (3 - 2 * t);
        fraction[d] = t;
    }
    int count = 1 << j->dimensions;
    for (int corner = 0; corner < count; ++corner) {
        double weight = 1;
        int64_t hash = 0;
        for (int d = 0; d < j->dimensions; ++d) {
            int bit = (corner >> (j->dimensions - 1 - d)) & 1;
            /* Match NumPy's full expression, including signed zero. */
            weight *= fraction[d] * bit + (1 - fraction[d]) * (1 - bit);
            double coordinate = base[d] + bit;
            /* NumPy's float-to-int32 conversion yields INT32_MIN for invalid
             * or overflowing lattice coordinates, without C undefined casts. */
            int32_t corner_coordinate = isfinite(coordinate)
                && coordinate >= INT32_MIN && coordinate <= INT32_MAX
                ? (int32_t)coordinate : INT32_MIN;
            /* The index starts as int32 zeros (0 + coordinate cannot wrap); the
             * permutation table is np.arange's default int, int64 on Windows under
             * NumPy 2, so every later sum is int64. */
            hash = j->table[modulo(hash + corner_coordinate, j->table_size)];
        }
        contributions[corner] = j->values[modulo(hash, j->value_count)] * weight;
    }
    return sum_corners(contributions, count);
}

static double simplex_at(const procedural_job *j, const double *point) {
    int32_t vertex[7][6];
    double remainder[6];
    double total = -0.0;
    for (int d = 0; d < j->dimensions; ++d) total += point[d];
    double skew = total * j->f;
    for (int d = 0; d < j->dimensions; ++d) {
        double v = point[d] + skew;
        double base = floor(v);
        remainder[d] = v - base;
        int32_t cell = isfinite(base) && base >= INT32_MIN && base <= INT32_MAX
            ? (int32_t)base : INT32_MIN;
        for (int s = 0; s <= j->dimensions; ++s)
            vertex[s][d] = s == j->dimensions
                ? (cell == INT32_MAX ? INT32_MIN : cell + 1) : cell;
    }
    for (int i = 1; i < j->dimensions; ++i) {
        int largest = 0;
        for (int d = 1; d < j->dimensions; ++d)
            if (remainder[d] > remainder[largest]) largest = d;
        for (int s = i; s < j->dimensions; ++s)
            vertex[s][largest] = vertex[s][largest] == INT32_MAX
                ? INT32_MIN : vertex[s][largest] + 1;
        if (i != j->dimensions - 1) remainder[largest] = -1;
    }
    double result = -0.0;
    for (int s = 0; s <= j->dimensions; ++s) {
        int64_t hash = 0, sum = 0;
        for (int d = 0; d < j->dimensions; ++d) {
            /* int64 index sums, as in value_at; np.sum of the int32 vertices
             * accumulates in np.int_, int64 on Windows under NumPy 2. */
            hash = j->table[modulo(hash + vertex[s][d], j->table_size)];
            sum += vertex[s][d];
        }
        size_t gi = modulo(hash, j->gradient_count) * (size_t)j->dimensions;
        double unskew = (double)sum * j->g, displacement = -0.0, dot = -0.0;
        for (int d = 0; d < j->dimensions; ++d) {
            double delta = point[d] - ((double)vertex[s][d] - unskew);
            displacement += delta * delta;
            dot += delta * j->gradients[gi + (size_t)d];
        }
        double radius = j->r2 - displacement;
        if (radius < 0) radius = 0;
        result += cn_crt.pow(radius, 4.0) * dot;
    }
    return result * j->scale + 0.5;
}

double cn_procedural_sample(const procedural_job *j, const double *point) {
    return j->simplex ? simplex_at(j, point) : value_at(j, point);
}

static void procedural_range(void *opaque, size_t begin, size_t end) {
    procedural_job *j = opaque;
    for (size_t i = begin; i < end; ++i) {
        const double *point = j->points + i * (size_t)j->dimensions;
        j->out[i] = cn_procedural_sample(j, point);
    }
}

CN_EXPORT cn_status cn_procedural_noise(const double *points, size_t count, int dimensions,
        const int32_t *table, size_t table_size, const float *values, size_t value_count,
        const double *gradients, size_t gradient_count, int smooth, int simplex,
        double f, double g, double r2, double scale, double *out) {
    if (!points || !table || !out || dimensions < 1 || dimensions > 6 || !table_size
        || table_size > INT32_MAX || smooth < 0 || smooth > 1 || simplex < 0 || simplex > 1
        || (simplex && (!gradients || !gradient_count || dimensions < 2))
        || (!simplex && (!values || !value_count))
        || value_count > INT32_MAX || gradient_count > INT32_MAX
        || !isfinite(f) || !isfinite(g) || !isfinite(r2) || !isfinite(scale)) return CN_INVALID_ARGUMENT;
    if (count > SIZE_MAX / (size_t)dimensions / sizeof(double)) return CN_SIZE_OVERFLOW;
    /* Validate before workers start: integer conversion must never see NaN or
     * out-of-range coordinates, even from a direct non-Python ABI caller. */
    for (size_t i = 0; i < count * (size_t)dimensions; ++i)
        if (!isfinite(points[i]) || fabs(points[i]) > 100000000.0) return CN_INVALID_ARGUMENT;
    if (simplex && (fabs(f) > 1 || fabs(g) > 1)) return CN_INVALID_ARGUMENT;
    for (size_t i = 0; i < table_size; ++i)
        if (table[i] < 0 || (size_t)table[i] >= table_size) return CN_INVALID_ARGUMENT;
    procedural_job job = {points, gradients, values, table, out, table_size, gradient_count,
                          value_count, dimensions, smooth, simplex, f, g, r2, scale};
    return cn_parallel_for(count, 8192, procedural_range, &job);
}
