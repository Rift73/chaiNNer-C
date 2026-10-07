/* C17 adaptation of OpenCV 5.0.0 K-means++ for chaiNNer's palette contract.
 * Original authors/license: see chainner_native.LICENSE.txt (OpenCV notices).
 * Reference: modules/core/src/kmeans.cpp, norm.dispatch.cpp (normL2Sqr_, the
 * baseline build) and sum.simd.hpp (sum64f), core/operations.hpp.
 * Unlike the original wrapper, each invocation owns its random state. Parallel
 * tasks only write independent distance/label rows; reductions retain order.
 */
#include "parallel.h"
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* A numerical failure has a distinct bridge mapping to the original cv2.error. */
#define CN_KMEANS_NO_CONVERGENCE 5

static uint32_t random_u32(uint64_t *state) {
    *state = (uint64_t)(uint32_t)*state * UINT32_C(4164903690) +
             (uint32_t)(*state >> 32);
    return (uint32_t)*state;
}

static double random_f64(uint64_t *state) {
    uint64_t high = random_u32(state);
    uint64_t bits = (high << 32) | random_u32(state);
    return (double)bits * 5.4210108624275221700372640043497e-20;
}

/* Match the baseline SSE2 norm of OpenCV's nondispatched norm.dispatch.cpp.
 * Image channels ordinarily number 1, 3 or 4. Wider helper arrays still retain
 * the four-vector accumulation/reduction order, including the scalar tail.
 */
static float distance(const float *a, const float *b, size_t channels) {
    float sums[4][4] = {{0}};
    size_t c = 0;
    for (; channels - c >= 16; c += 16) {
        for (size_t group = 0; group < 4; ++group)
            for (size_t lane = 0; lane < 4; ++lane) {
                float delta = a[c + group * 4 + lane] - b[c + group * 4 + lane];
                sums[group][lane] += delta * delta;
            }
    }
    float lanes[4];
    for (size_t lane = 0; lane < 4; ++lane)
        lanes[lane] = ((sums[0][lane] + sums[1][lane]) + sums[2][lane]) + sums[3][lane];
    float result = (lanes[0] + lanes[2]) + (lanes[1] + lanes[3]);
    for (; c < channels; ++c) {
        float delta = a[c] - b[c];
        result += delta * delta;
    }
    return result;
}

typedef struct {
    const float *data, *centers, *previous;
    float *next;
    double *distances;
    int *labels;
    size_t channels, colors, candidate;
    int only_distance;
} kmeans_job;

static void seed_distances(void *opaque, size_t begin, size_t end) {
    kmeans_job *j = opaque;
    const float *center = j->data + j->candidate * j->channels;
    for (size_t i = begin; i < end; ++i) {
        float d = distance(j->data + i * j->channels, center, j->channels);
        /* std::min(d, previous) keeps d for an unordered comparison. */
        j->next[i] = j->previous && j->previous[i] < d ? j->previous[i] : d;
    }
}

static void assign_distances(void *opaque, size_t begin, size_t end) {
    kmeans_job *j = opaque;
    for (size_t i = begin; i < end; ++i) {
        const float *point = j->data + i * j->channels;
        if (j->only_distance) {
            j->distances[i] = distance(point,
                j->centers + (size_t)j->labels[i] * j->channels, j->channels);
            continue;
        }
        double best = DBL_MAX;
        int label = 0;
        for (size_t k = 0; k < j->colors; ++k) {
            double d = distance(point, j->centers + k * j->channels, j->channels);
            if (best > d) { best = d; label = (int)k; }
        }
        j->distances[i] = best;
        j->labels[i] = label;
    }
}

static cn_status initialize_centers(kmeans_job *job, size_t pixels, uint64_t *rng,
                                    float *centers, float *scratch, size_t *seeds) {
    float *dist = scratch, *trial = scratch + pixels, *candidate = scratch + pixels * 2;
    seeds[0] = random_u32(rng) % pixels;
    job->candidate = seeds[0]; job->previous = NULL; job->next = dist;
    cn_status status = cn_parallel_for(pixels, 4096, seed_distances, job);
    if (status != CN_OK) return status;
    double total = 0;
    for (size_t i = 0; i < pixels; ++i) total += dist[i];
    for (size_t k = 1; k < job->colors; ++k) {
        double best_sum = DBL_MAX;
        size_t best_center = SIZE_MAX;
        for (int t = 0; t < 3; ++t) {
            double p = random_f64(rng) * total;
            size_t chosen = 0;
            for (; chosen < pixels - 1; ++chosen) {
                p -= dist[chosen];
                if (p <= 0) break;
            }
            job->candidate = chosen; job->previous = dist; job->next = candidate;
            status = cn_parallel_for(pixels, 4096, seed_distances, job);
            if (status != CN_OK) return status;
            /* CV_ENABLE_UNROLLED: eight float distances add in float, left to
               right (/fp:precise), before each widening into the double sum. */
            double sum = 0;
            size_t i = 0;
            for (; pixels - i >= 8; i += 8)
                sum += (double)(((((((candidate[i] + candidate[i + 1]) + candidate[i + 2]) +
                    candidate[i + 3]) + candidate[i + 4]) + candidate[i + 5]) +
                    candidate[i + 6]) + candidate[i + 7]);
            for (; i < pixels; ++i) sum += candidate[i];
            if (sum < best_sum) {
                best_sum = sum; best_center = chosen;
                float *swap = trial; trial = candidate; candidate = swap;
            }
        }
        if (best_center == SIZE_MAX) return CN_KMEANS_NO_CONVERGENCE;
        seeds[k] = best_center; total = best_sum;
        float *swap = dist; dist = trial; trial = swap;
    }
    for (size_t k = 0; k < job->colors; ++k)
        memcpy(centers + k * job->channels, job->data + seeds[k] * job->channels,
               job->channels * sizeof(float));
    return CN_OK;
}

CN_EXPORT cn_status cn_palette_kmeans(const float *data, float *out,
                                      size_t pixels, size_t channels, size_t colors) {
    if (!data || !out || !pixels || !channels || !colors || colors > pixels)
        return CN_INVALID_ARGUMENT;
    if (pixels > INT_MAX || channels > INT_MAX || colors > INT_MAX ||
        channels > SIZE_MAX / sizeof(float) ||
        pixels > SIZE_MAX / sizeof(float) / channels ||
        colors > SIZE_MAX / sizeof(float) / channels / 2 ||
        pixels > SIZE_MAX / sizeof(float) / 3 ||
        pixels > SIZE_MAX / sizeof(double) || pixels > SIZE_MAX / sizeof(int) ||
        colors > SIZE_MAX / sizeof(size_t)) return CN_SIZE_OVERFLOW;
    size_t center_values = colors * channels;
    float *storage = malloc(center_values * 2 * sizeof(float));
    float *scratch = malloc(pixels * 3 * sizeof(float));
    float *temp = malloc(channels * sizeof(float));
    double *distances = malloc(pixels * sizeof(double));
    int *labels = malloc(pixels * sizeof(int));
    int *counts = malloc(colors * sizeof(int));
    size_t *seeds = malloc(colors * sizeof(size_t));
    cn_status status = CN_ALLOCATION_FAILED;
    if (!storage || !scratch || !temp || !distances || !labels || !counts || !seeds)
        goto cleanup;
    float *centers = storage, *old = storage + center_values;
    uint64_t rng = UINT64_C(0xffffffff);
    double best_compactness = DBL_MAX;
    kmeans_job job = {0};
    job.data = data; job.channels = channels; job.colors = colors;
    job.labels = labels; job.distances = distances;
    int attempts = colors == 1 ? 1 : 10;
    int iterations = colors == 1 ? 2 : 10;
    for (int attempt = 0; attempt < attempts; ++attempt) {
        double compactness = 0;
        for (int iteration = 0;;) {
            double max_shift = iteration == 0 ? DBL_MAX : 0;
            float *swap = centers; centers = old; old = swap;
            if (!iteration) {
                status = initialize_centers(&job, pixels, &rng, centers, scratch, seeds);
                if (status != CN_OK) goto cleanup;
            } else {
                memset(centers, 0, center_values * sizeof(float));
                memset(counts, 0, colors * sizeof(int));
                /* Row order is part of the installed float32 centroid contract. */
                for (size_t i = 0; i < pixels; ++i) {
                    size_t k = (size_t)labels[i];
                    for (size_t c = 0; c < channels; ++c)
                        centers[k * channels + c] += data[i * channels + c];
                    ++counts[k];
                }
                for (size_t k = 0; k < colors; ++k) {
                    if (counts[k]) continue;
                    size_t largest = 0;
                    for (size_t l = 1; l < colors; ++l)
                        if (counts[largest] < counts[l]) largest = l;
                    float scale = 1.0f / (float)counts[largest];
                    for (size_t c = 0; c < channels; ++c)
                        temp[c] = centers[largest * channels + c] * scale;
                    double furthest_distance = 0;
                    size_t furthest = SIZE_MAX;
                    for (size_t i = 0; i < pixels; ++i) {
                        if ((size_t)labels[i] != largest) continue;
                        double d = distance(data + i * channels, temp, channels);
                        if (furthest_distance <= d) { furthest_distance = d; furthest = i; }
                    }
                    if (furthest == SIZE_MAX) { status = CN_KMEANS_NO_CONVERGENCE; goto cleanup; }
                    --counts[largest]; ++counts[k]; labels[furthest] = (int)k;
                    for (size_t c = 0; c < channels; ++c) {
                        float v = data[furthest * channels + c];
                        centers[largest * channels + c] -= v;
                        centers[k * channels + c] += v;
                    }
                }
                for (size_t k = 0; k < colors; ++k) {
                    float scale = 1.0f / (float)counts[k];
                    double shift = 0;
                    for (size_t c = 0; c < channels; ++c) {
                        size_t offset = k * channels + c;
                        centers[offset] *= scale;
                        /* The original subtraction rounds to float before widening. */
                        double delta = (double)(centers[offset] - old[offset]);
                        shift += delta * delta;
                    }
                    if (max_shift < shift) max_shift = shift;
                }
            }
            ++iteration;
            job.centers = centers;
            job.only_distance = iteration == iterations || max_shift <= 1.0;
            status = cn_parallel_for(pixels, colors > 16 ? 512 : 4096, assign_distances, &job);
            if (status != CN_OK) goto cleanup;
            if (job.only_distance) {
                size_t i = 0;
                /* cv::sum's double path groups four values before the outer sum. */
                for (; pixels - i >= 4; i += 4)
                    compactness += ((distances[i] + distances[i + 1]) + distances[i + 2]) + distances[i + 3];
                for (; i < pixels; ++i) compactness += distances[i];
                break;
            }
        }
        if (compactness < best_compactness) {
            best_compactness = compactness;
            memcpy(out, centers, center_values * sizeof(float));
        }
    }
    status = best_compactness < DBL_MAX ? CN_OK : CN_KMEANS_NO_CONVERGENCE;
cleanup:
    free(seeds); free(counts); free(labels); free(distances);
    free(temp); free(scratch); free(storage);
    return status;
}
