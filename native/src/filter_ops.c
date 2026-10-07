#include "chainner.h"
#include "cn_crt_math.h"
#include "isa.h"
#include "merge_shared.h"
#include "morph_entry.h"
#include "parallel.h"
#include "scratch.h"
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static float clipped(float x, float low, float high) {
    if (x < low) x = low;
    return x > high ? high : x;
}

typedef struct arithmetic_job {
    const float *a, *b, *c, *d;
    float *out;
    float scale, second;
    int operation, clip;
} arithmetic_job;

static void arithmetic_range(void *context, size_t begin, size_t end) {
    arithmetic_job *j = (arithmetic_job *)context;
    for (size_t i = begin; i < end; ++i) {
        float x = j->a[i], value;
        switch (j->operation) {
        case 0: value = cn_crt.hypotf(x, j->b[i]) * j->scale; break;
        case 1: value = x * j->scale; value += 0.5f; break;
        case 2: value = x * j->scale; break;
        case 3: value = x - j->b[i]; value *= j->scale; break;
        case 4: {
            float high = j->b[i] - x, low = x - j->c[i];
            value = isnan(high) || high > low ? high : low;
            value *= clipped(j->d[i] * 10.0f, -0.75f, 0.75f);
            break;
        }
        case 5: {
            float other = fabsf(j->b[i]);
            value = fabsf(x);
            value = isnan(value) || value > other ? value : other;
            break;
        }
        case 6: value = x; break;
        case 7: value = cn_crt.powf(x, j->scale); break;
        case 8:
            value = x < 0 ? 0 : x;
            value = cn_crt.powf(value, j->scale);
            break;
        default: {
            float real = x - j->d[i], imag = j->b[i] + j->c[i];
            value = real * j->scale + imag * j->second;
            if (j->operation == 10) value += j->out[i];
            break;
        }
        }
        j->out[i] = j->clip ? clipped(value, 0, 1) : value;
    }
}

CN_EXPORT cn_status cn_filter_arithmetic(const float *a, const float *b,
                                       const float *c, const float *d, float *out,
                                       size_t n, int operation, float scale,
                                       float second, int clip) {
    if (!a || !out || operation < 0 || operation > 10) return CN_INVALID_ARGUMENT;
    if ((operation == 0 || operation == 3 || operation == 4 || operation == 5 ||
         operation >= 9) && !b) return CN_INVALID_ARGUMENT;
    if ((operation == 4 || operation >= 9) && (!c || !d)) return CN_INVALID_ARGUMENT;
    if (n > SIZE_MAX / sizeof(float)) return CN_SIZE_OVERFLOW;
    arithmetic_job job = {a, b, c, d, out, scale, second, operation, clip};
    return cn_parallel_for(n, 65536, arithmetic_range, &job);
}

CN_EXPORT cn_status cn_filter_lens_normalize(const float *complex_values,
                                           const double *parameters, float *out,
                                           size_t components, size_t length) {
    if (!complex_values || !parameters || !out || !components || !length)
        return CN_INVALID_ARGUMENT;
    if (components > SIZE_MAX / sizeof(double) / 2 ||
        length > SIZE_MAX / sizeof(float) / 2 ||
        components > SIZE_MAX / sizeof(float) / 2 / length) return CN_SIZE_OVERFLOW;
    /* Upstream adds one term for every pair of float32 complex taps, in this
       order. Under NumPy 2.5's NEP 50 the Python-float coefficients are weak,
       so each term and the running total (Python int 0 first) are float32. */
    float total = 0.0f;
    for (size_t component = 0; component < components; ++component) {
        const float *kernel = complex_values + component * length * 2;
        float a = (float)parameters[component * 2], b = (float)parameters[component * 2 + 1];
        for (size_t i = 0; i < length; ++i) {
            for (size_t k = 0; k < length; ++k) {
                float real = kernel[i * 2] * kernel[k * 2] -
                             kernel[i * 2 + 1] * kernel[k * 2 + 1];
                float imag = kernel[i * 2] * kernel[k * 2 + 1] +
                             kernel[i * 2 + 1] * kernel[k * 2];
                total += a * real + b * imag;
            }
        }
    }
    if (!(total > 0)) return CN_INVALID_ARGUMENT;
    float scale = (float)(1.0 / sqrt((double)total));
    for (size_t i = 0; i < components * length * 2; ++i) out[i] = complex_values[i] * scale;
    return CN_OK;
}

typedef struct morph_job {
    const float *src;
    float *out;
    struct morph_entry *queue;
    size_t width, height, channels, radius;
    int vertical, maximum;
} morph_job;

/* Each line's running extremum over [center - radius, center + radius] (clipped):
 * the deque keeps a strictly monotonic run of values, a new value dropping every entry
 * it ties or beats, so its head is the window's last extremal element, whose bits are
 * stored. Lengths are at most INT_MAX, so an index fits uint32_t. A chunk's lines run
 * one after another, so they share the deque of its first line (its slot of the
 * queue, which no other chunk uses). */
static void morph_lines(void *context, size_t begin, size_t end) {
    morph_job *j = (morph_job *)context;
    size_t length = j->vertical ? j->height : j->width;
    size_t stride = j->vertical ? j->width * j->channels : j->channels;
    struct morph_entry *queue = j->queue + begin * length;
    for (size_t line = begin; line < end; ++line) {
        size_t offset = j->vertical ? line : (line / j->channels) * stride * length + line % j->channels;
        const float *src = j->src + offset;
        float *out = j->out + offset;
        size_t head = 0, tail = 0, inserted = 0;
        for (size_t center = 0; center < length; ++center) {
            size_t last = j->radius >= length - 1 - center ? length - 1 : center + j->radius;
            size_t first = center > j->radius ? center - j->radius : 0;
            while (head < tail && queue[head].index < first) ++head;
            while (inserted <= last) {
                float value = src[inserted * stride];
                while (head < tail) {
                    float previous = queue[tail - 1].value;
                    if (j->maximum ? previous > value : previous < value) break;
                    --tail;
                }
                queue[tail].value = value;
                queue[tail].index = (uint32_t)inserted;
                ++tail;
                ++inserted;
            }
            out[center * stride] = queue[head].value;
        }
    }
}

/* isa: the level the C entry read once for the call (cn_isa_current). */
typedef struct morph_merge_job {
    const float *a;
    float *b;
    int maximum;
    cn_isa_level isa;
} morph_merge_job;

static void morph_merge(void *context, size_t begin, size_t end) {
    morph_merge_job *j = (morph_merge_job *)context;
#if defined(_MSC_VER) && defined(_M_X64)
    if (j->isa >= CN_ISA_AVX2) {
        cn_morphology_merge_avx2(j->a, j->b, begin, end, j->maximum);
        return;
    }
#endif
    for (size_t i = begin; i < end; ++i) merge_one(j->a, j->b, i, j->maximum);
}

/* horizontal, alternate (cross, iterations > 1) and the queue are one lease of the
 * shared pool, every buffer written before it is read (D10). Iteration i writes out
 * when (iterations - 1 - i) is even, else alternate, so the last writes out
 * (ping-pong, D11). */
CN_EXPORT cn_status cn_filter_morphology(const float *src, float *out,
                                       size_t height, size_t width, size_t channels,
                                       size_t radius, size_t iterations,
                                       int cross, int maximum) {
    if (!src || !out || !height || !width || !channels || !radius || !iterations)
        return CN_INVALID_ARGUMENT;
    if (width > SIZE_MAX / height || width * height > SIZE_MAX / channels ||
        width * height * channels > SIZE_MAX / sizeof(size_t)) return CN_SIZE_OVERFLOW;
    if (!cross && radius > SIZE_MAX / iterations) return CN_SIZE_OVERFLOW;
    /* OpenCV's int dimensions (D11), before any allocation. */
    if (height > INT_MAX || width > INT_MAX) return CN_SIZE_OVERFLOW;
    size_t n = width * height * channels;
    size_t total = 0, horizontal_at = 0, alternate_at = 0, queue_at = 0;
    int alternates = cross && iterations > 1;
    if (!cn_scratch_carve(&total, n * sizeof(float), &horizontal_at) ||
        (alternates && !cn_scratch_carve(&total, n * sizeof(float), &alternate_at)) ||
        n > SIZE_MAX / sizeof(struct morph_entry) ||
        !cn_scratch_carve(&total, n * sizeof(struct morph_entry), &queue_at))
        return CN_ALLOCATION_FAILED;
    cn_scratch scratch;
    cn_status status = cn_scratch_lease(total, &scratch, NULL);
    if (status != CN_OK) return status;
    float *horizontal = (float *)((char *)scratch.data + horizontal_at);
    float *alternate = alternates ? (float *)((char *)scratch.data + alternate_at) : NULL;
    struct morph_entry *queue = (struct morph_entry *)((char *)scratch.data + queue_at);
    if (!cross) { radius *= iterations; iterations = 1; }
    const float *current = src;
    cn_isa_level isa = cn_isa_current();
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
        float *destination = (iterations - 1 - iteration) % 2 ? alternate : out;
        morph_job job = {current, horizontal, queue, width, height, channels, radius, 0, maximum};
        status = cn_parallel_for(height * channels, 1 + 65536 / width, morph_lines, &job);
        if (status != CN_OK) break;
        job.src = cross ? current : horizontal;
        job.out = destination; job.vertical = 1;
        status = cn_parallel_for(width * channels, 1 + 65536 / height, morph_lines, &job);
        if (status != CN_OK) break;
        if (cross) {
            morph_merge_job merge = {horizontal, destination, maximum, isa};
            status = cn_parallel_for(n, 65536, morph_merge, &merge);
            if (status != CN_OK) break;
        }
        current = destination;
    }
    cn_scratch_release(&scratch);
    return status;
}

static size_t reflect_index(int64_t index, size_t length) {
    if (length == 1) return 0;
    int64_t period = 2 * ((int64_t)length - 1);
    index %= period;
    if (index < 0) index += period;
    return (size_t)(index < (int64_t)length ? index : period - index);
}

static double coordinate(size_t index, size_t length, double scale) {
    if (length <= 1) return 0;
    if (index == length - 1) return scale;
    double step = 1.0 / (double)(length - 1);
    return ((double)index * step) * scale;
}

typedef struct reference_job {
    const float *src, *reference;
    float *out;
    size_t width, height, ref_width, ref_height, channels, radius, scale;
    double spatial;
} reference_job;

static int candidate_less(const reference_job *j, size_t a, size_t b) {
    for (size_t c = 0; c < j->channels; ++c) {
        float x = j->reference[a * j->channels + c], y = j->reference[b * j->channels + c];
        if (x < y || (!isnan(x) && isnan(y))) return 1;
        if (x > y || (isnan(x) && !isnan(y))) return 0;
    }
    double ax = coordinate(a % j->ref_width, j->ref_width, j->spatial);
    double bx = coordinate(b % j->ref_width, j->ref_width, j->spatial);
    if (ax != bx) return ax < bx;
    return coordinate(a / j->ref_width, j->ref_height, j->spatial) <
           coordinate(b / j->ref_width, j->ref_height, j->spatial);
}

static void reference_range(void *context, size_t begin, size_t end) {
    reference_job *j = (reference_job *)context;
    for (size_t p = begin; p < end; ++p) {
        size_t y = p / j->width, x = p % j->width;
        size_t block_y = y / j->scale, block_x = x / j->scale;
        /* Preserve upstream's height-derived scale in both directions, including
           untouched zero columns when width and height use different ratios. */
        if (block_x >= j->ref_width) {
            memset(j->out + p * j->channels, 0, j->channels * sizeof(float));
            continue;
        }
        size_t best = 0;
        double best_distance = 0;
        int initialized = 0;
        double target_x = coordinate(x, j->width, j->spatial);
        double target_y = coordinate(y, j->height, j->spatial);
        for (int64_t dy = -(int64_t)j->radius; dy <= (int64_t)j->radius; ++dy) {
            size_t ry = reflect_index((int64_t)block_y + dy, j->ref_height);
            for (int64_t dx = -(int64_t)j->radius; dx <= (int64_t)j->radius; ++dx) {
                size_t rx = reflect_index((int64_t)block_x + dx, j->ref_width);
                size_t candidate = ry * j->ref_width + rx;
                double sum = -0.0;
                for (size_t c = 0; c < j->channels; ++c) {
                    double difference = (double)j->src[p * j->channels + c] -
                                        j->reference[candidate * j->channels + c];
                    sum += difference * difference;
                }
                double difference = target_x - coordinate(rx, j->ref_width, j->spatial);
                sum += difference * difference;
                difference = target_y - coordinate(ry, j->ref_height, j->spatial);
                sum += difference * difference;
                double distance = sqrt(sum);
                int better = !initialized || (!isnan(best_distance) &&
                    (isnan(distance) || distance < best_distance));
                if (!better && ((isnan(distance) && isnan(best_distance)) || distance == best_distance))
                    better = candidate_less(j, candidate, best);
                if (better) { best = candidate; best_distance = distance; initialized = 1; }
            }
        }
        memcpy(j->out + p * j->channels, j->reference + best * j->channels,
               j->channels * sizeof(float));
    }
}

CN_EXPORT cn_status cn_filter_quantize_reference(const float *src, const float *reference,
                                                float *out, size_t height, size_t width,
                                                size_t ref_height, size_t ref_width,
                                                size_t channels, size_t radius,
                                                double spatial) {
    if (!src || !reference || !out || !height || !width || !ref_height || !ref_width ||
        (channels != 3 && channels != 4) || radius < 1 || radius > 5 ||
        height % ref_height || width % ref_width || height < ref_height || width < ref_width)
        return CN_INVALID_ARGUMENT;
    if (width > SIZE_MAX / height || width * height > SIZE_MAX / sizeof(float) / channels ||
        width > (size_t)INT64_MAX / 2 || height > (size_t)INT64_MAX / 2)
        return CN_SIZE_OVERFLOW;
    reference_job job = {src, reference, out, width, height, ref_width, ref_height,
                         channels, radius, height / ref_height, spatial};
    return cn_parallel_for(width * height, 4096, reference_range, &job);
}

/* The installed OpenCV 4.8 Intel IPP backend uses floating-point 5x5 chamfer
   costs (1, 1.4, 2.1969). Use a generic two-sweep neighborhood implementation;
   dependencies inside a sweep deliberately remain serial. The scalar OpenCV
   backend has different fixed-point rounding and stays a separate Python path. */
static void chamfer(const unsigned char *binary, float *distance,
                    size_t height, size_t width, int white_zero) {
    const int dy[8] = {-2, -2, -1, -1, -1, -1, -1, 0};
    const int dx[8] = {-1, 1, -2, -1, 0, 1, 2, -1};
    const float cost[8] = {2.1969f, 2.1969f, 2.1969f, 1.4f, 1.0f, 1.4f, 2.1969f, 1.0f};
    size_t pixels = width * height;
    for (size_t p = 0; p < pixels; ++p)
        distance[p] = ((binary[p] != 0) == white_zero) ? 0 : FLT_MAX;
    for (int pass = 0; pass < 2; ++pass) {
        int direction = pass == 0 ? 1 : -1;
        for (size_t index = 0; index < pixels; ++index) {
            size_t p = pass == 0 ? index : pixels - index - 1;
            if (distance[p] == 0) continue;
            int64_t y = (int64_t)(p / width), x = (int64_t)(p % width);
            float best = distance[p];
            for (size_t k = 0; k < 8; ++k) {
                int64_t ny = y + direction * dy[k], nx = x + direction * dx[k];
                if (ny < 0 || nx < 0 || (size_t)ny >= height || (size_t)nx >= width) continue;
                float alternative = distance[(size_t)ny * width + (size_t)nx] + cost[k];
                if (alternative < best) best = alternative;
            }
            distance[p] = best;
        }
    }
}

typedef struct distance_job {
    const unsigned char *binary;
    const float *black, *white;
    float *out;
    float spread;
} distance_job;

typedef struct chamfer_job {
    const unsigned char *binary;
    float *distances[2];
    size_t height, width;
} chamfer_job;

static void chamfer_range(void *context, size_t begin, size_t end) {
    chamfer_job *j = (chamfer_job *)context;
    for (size_t i = begin; i < end; ++i)
        chamfer(j->binary, j->distances[i], j->height, j->width, i == 1);
}

static void distance_finish(void *context, size_t begin, size_t end) {
    distance_job *j = (distance_job *)context;
    for (size_t p = begin; p < end; ++p) {
        float value = j->binary[p] ? j->black[p] : j->white[p];
        value /= j->spread; value /= 2;
        value = j->binary[p] ? value + 0.5f : 0.5f - value;
        j->out[p] = clipped(value, 0, 1);
    }
}

CN_EXPORT cn_status cn_filter_binary_sdf(const unsigned char *binary, float *out,
                                       size_t height, size_t width, float spread) {
    if (!binary || !out || !height || !width || !(spread > 0)) return CN_INVALID_ARGUMENT;
    if (width > SIZE_MAX / height || width * height > SIZE_MAX / sizeof(float) ||
        width > (size_t)INT64_MAX / 2 || height > (size_t)INT64_MAX / 2)
        return CN_SIZE_OVERFLOW;
    size_t n = width * height;
    float *black = (float *)malloc(n * sizeof(float)), *white = (float *)malloc(n * sizeof(float));
    if (!black || !white) { free(black); free(white); return CN_ALLOCATION_FAILED; }
    chamfer_job transforms = {binary, {black, white}, height, width};
    cn_status status;
    if (n >= 65536) status = cn_parallel_for(2, 1, chamfer_range, &transforms);
    else { chamfer_range(&transforms, 0, 2); status = CN_OK; }
    if (status != CN_OK) { free(black); free(white); return status; }
    distance_job job = {binary, black, white, out, spread};
    status = cn_parallel_for(n, 65536, distance_finish, &job);
    free(black); free(white);
    return status;
}
