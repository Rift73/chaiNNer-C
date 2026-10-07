#include "chainner.h"
#include "chainner_ext_kernels.h"
#include "dither_shared.h"
#include "numeric.h"
#include "parallel.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Column counts fit uint16_t through public radius 1000 (height <=2001).
   Window accumulators use uint32_t: an entire window can hold 4,004,001
   samples, avoiding the original OpenCV uint16 saturation/overflow. */
static cn_status dimensions(size_t h, size_t w, size_t c) {
    if (!h || !w || (c != 1 && c != 3 && c != 4)) return CN_INVALID_ARGUMENT;
    if (h > SIZE_MAX / w || h * w > SIZE_MAX / sizeof(float) / c)
        return CN_SIZE_OVERFLOW;
    return CN_OK;
}

static size_t low_edge(size_t position, size_t radius) {
    return position < radius ? 0 : position - radius;
}

static size_t high_edge(size_t position, size_t radius, size_t length) {
    return radius >= length - position - 1 ? length - 1 : position + radius;
}

typedef struct small_median_job {
    const float *src;
    float *out;
    size_t h, w, c, radius, vector_lanes;
    int sorting_network;
} small_median_job;

/* Comparator networks adapted from OpenCV 5.0.0 median_blur.simd.hpp.
   Copyright (C) 2000-2008, 2018 Intel Corporation; (C) 2009 Willow Garage;
   (C) 2014-2015 Itseez. BSD notice is in chainner_native.LICENSE.txt.
   Scalar std::min/max and vector MINPS/MAXPS choose different operands for
   NaNs and equal signed zeros. Preserve those choices and the SIMD bounds:
   a row interior holding at least one vector is processed entirely by
   vectors (the last one overlaps back), and a shorter interior is scalar. */
static void median_minmax(float *a, float *b, int vector) {
    float first = *a, second = *b;
    if (vector) {
        *a = first < second ? first : second;
        *b = second > first ? second : first;
    } else {
        *a = second < first ? second : first;
        *b = second < first ? first : second;
    }
}

static float network_median(float *values, size_t count, int vector) {
    static const uint8_t thin3[][2] = {{0,1},{1,2},{0,1}};
    static const uint8_t thin5[][2] = {{0,1},{3,4},{2,3},{3,4},{0,2},{2,4},{1,3},{1,2}};
    static const uint8_t square3[][2] = {
        {1,2},{4,5},{7,8},{0,1},{3,4},{6,7},{1,2},{4,5},{7,8},
        {0,3},{5,8},{4,7},{3,6},{1,4},{2,5},{4,7},{4,2},{6,4},{4,2}
    };
    static const uint8_t square5[][2] = {
        {1,2},{0,1},{1,2},{4,5},{3,4},{4,5},{0,3},{2,5},{2,3},{1,4},
        {1,2},{3,4},{7,8},{6,7},{7,8},{10,11},{9,10},{10,11},{6,9},{8,11},
        {8,9},{7,10},{7,8},{9,10},{0,6},{4,10},{4,6},{2,8},{2,4},{6,8},
        {1,7},{5,11},{5,7},{3,9},{3,5},{7,9},{1,2},{3,4},{5,6},{7,8},
        {9,10},{13,14},{12,13},{13,14},{16,17},{15,16},{16,17},{12,15},{14,17},{14,15},
        {13,16},{13,14},{15,16},{19,20},{18,19},{19,20},{21,22},{23,24},{21,23},{22,24},
        {22,23},{18,21},{20,23},{20,21},{19,22},{22,24},{19,20},{21,22},{23,24},{12,18},
        {16,22},{16,18},{14,20},{20,24},{14,16},{18,20},{22,24},{13,19},{17,23},{17,19},
        {15,21},{15,17},{19,21},{13,14},{15,16},{17,18},{19,20},{21,22},{23,24},{0,12},
        {8,20},{8,12},{4,16},{16,24},{12,16},{2,14},{10,22},{10,14},{6,18},{6,10},
        {10,12},{1,13},{9,21},{9,13},{5,17},{13,17},{3,15},{11,23},{11,15},{7,19},
        {7,11},{11,13},{11,12}
    };
    const uint8_t (*pairs)[2];
    size_t length;
    if (count == 3) { pairs = thin3; length = sizeof(thin3) / sizeof(thin3[0]); }
    else if (count == 5) { pairs = thin5; length = sizeof(thin5) / sizeof(thin5[0]); }
    else if (count == 9) { pairs = square3; length = sizeof(square3) / sizeof(square3[0]); }
    else { pairs = square5; length = sizeof(square5) / sizeof(square5[0]); }
    for (size_t i = 0; i < length; ++i)
        median_minmax(values + pairs[i][0], values + pairs[i][1], vector);
    return values[count / 2];
}

/* Bounded selection over at most 25 samples, with no per-pixel allocation. */
static float select_middle(float *values, size_t count) {
    size_t lo = 0, hi = count, middle = count / 2;
    while (hi - lo > 1) {
        float pivot = values[middle];
        size_t less = lo, at = lo, greater = hi;
        while (at < greater) {
            if (values[at] < pivot) {
                float swap = values[at]; values[at++] = values[less]; values[less++] = swap;
            } else if (values[at] > pivot) {
                float swap = values[at]; values[at] = values[--greater]; values[greater] = swap;
            } else ++at;
        }
        if (middle < less) hi = less;
        else if (middle >= greater) lo = greater;
        else break;
    }
    return values[middle];
}

static void small_median_range(void *context, size_t begin, size_t end) {
    small_median_job *job = (small_median_job *)context;
    for (size_t i = begin; i < end; ++i) {
        size_t channel = i % job->c, x = i / job->c % job->w;
        size_t y = i / job->c / job->w, count = 0;
        float values[25];
        if (job->sorting_network && (job->w == 1 || job->h == 1)) {
            size_t position = job->w == 1 ? y : x;
            size_t length = job->w == 1 ? job->h : job->w;
            for (size_t d = 0; d <= 2 * job->radius; ++d) {
                size_t at = d < job->radius ? low_edge(position, job->radius - d) :
                    high_edge(position, d - job->radius, length);
                values[count++] = job->src[at * job->c + channel];
            }
            job->out[i] = network_median(values, count, 0);
            continue;
        }
        for (size_t dy = 0; dy <= 2 * job->radius; ++dy) {
            size_t yy = dy < job->radius ? low_edge(y, job->radius - dy) :
                high_edge(y, dy - job->radius, job->h);
            for (size_t dx = 0; dx <= 2 * job->radius; ++dx) {
                size_t xx = dx < job->radius ? low_edge(x, job->radius - dx) :
                    high_edge(x, dx - job->radius, job->w);
                values[count++] = job->src[(yy * job->w + xx) * job->c + channel];
            }
        }
        if (job->sorting_network) {
            size_t start = job->radius * job->c, flat_x = x * job->c + channel;
            size_t interior = job->w > 2 * job->radius ?
                (job->w - 2 * job->radius) * job->c : 0;
            size_t finish = interior >= job->vector_lanes ? start + interior : start;
            job->out[i] = network_median(values, count, flat_x >= start && flat_x < finish);
        } else job->out[i] = select_middle(values, count);
    }
}

CN_EXPORT cn_status cn_neighborhood_median_f32(const float *src, float *out,
        size_t h, size_t w, size_t c, size_t radius, size_t vector_lanes) {
    if (!src || !out || radius < 1 || radius > 2 || !h || !w || !c ||
        (vector_lanes != 4 && vector_lanes != 8 && vector_lanes != 16))
        return CN_INVALID_ARGUMENT;
    if (h > SIZE_MAX / w || h * w > SIZE_MAX / sizeof(float) / c) return CN_SIZE_OVERFLOW;
    small_median_job job = { src, out, h, w, c, radius, vector_lanes, 0 };
    for (size_t i = 0; i < h * w * c; ++i) {
        if (isnan(src[i]) || (src[i] == 0 && signbit(src[i]))) {
            job.sorting_network = 1;
            break;
        }
    }
    return cn_parallel_for(h * w * c, 8192, small_median_range, &job);
}

typedef struct column_histogram {
    uint16_t fine[256], coarse[16];
} column_histogram;

typedef struct histogram_job {
    const uint8_t *src;
    uint8_t *out;
    column_histogram *columns;
    size_t h, w, c, radius, original_width;
    int transpose;
} histogram_job;

static size_t image_index(const histogram_job *job, size_t y, size_t x, size_t c) {
    return (job->transpose ? x * job->original_width + y : y * job->w + x) * job->c + c;
}

static void add_column(column_histogram *hist, uint8_t value, size_t weight) {
    hist->fine[value] = (uint16_t)(hist->fine[value] + weight);
    hist->coarse[value >> 4] = (uint16_t)(hist->coarse[value >> 4] + weight);
}

/* A two-level, lazily updated sliding histogram. Fine bins are updated only
   for the coarse bucket containing the median. Scratch is O(min(H,W)*C),
   independent of radius and image area; the image is never padded/copied. */
static void histogram_range(void *context, size_t begin, size_t end) {
    histogram_job *job = (histogram_job *)context;
    size_t r = job->radius, kernel = 2 * r + 1;
    uint32_t middle = (uint32_t)(kernel * kernel / 2);
    for (size_t c = begin; c < end; ++c) {
        column_histogram *columns = job->columns + c * job->w;
        for (size_t x = 0; x < job->w; ++x) {
            for (size_t y = 0; y <= r && y < job->h; ++y) {
                size_t weight = y == 0 ? r + 1 : 1;
                if (y == job->h - 1 && r >= job->h) weight += r - job->h + 1;
                add_column(columns + x, job->src[image_index(job, y, x, c)], weight);
            }
        }
        for (size_t y = 0; y < job->h; ++y) {
            uint32_t coarse[16] = {0}, fine[16][16] = {{0}};
            size_t updated[16];
            for (size_t b = 0; b < 16; ++b) updated[b] = SIZE_MAX;
            for (size_t x = 0; x <= r && x < job->w; ++x) {
                uint32_t weight = (uint32_t)(x == 0 ? r + 1 : 1);
                if (x == job->w - 1 && r >= job->w) weight += (uint32_t)(r - job->w + 1);
                for (size_t b = 0; b < 16; ++b) coarse[b] += weight * columns[x].coarse[b];
            }
            for (size_t x = 0; x < job->w; ++x) {
                uint32_t below = 0;
                size_t bucket = 0;
                while (bucket < 15 && below + coarse[bucket] <= middle)
                    below += coarse[bucket++];
                if (updated[bucket] == SIZE_MAX || x - updated[bucket] >= kernel) {
                    memset(fine[bucket], 0, sizeof(fine[bucket]));
                    size_t left = low_edge(x, r), right = high_edge(x, r, job->w);
                    for (size_t xx = left; xx <= right; ++xx) {
                        uint32_t weight = 1;
                        if (!xx && x < r) weight += (uint32_t)(r - x);
                        if (xx == job->w - 1 && r >= job->w - x)
                            weight += (uint32_t)(r - (job->w - x - 1));
                        for (size_t k = 0; k < 16; ++k)
                            fine[bucket][k] += weight * columns[xx].fine[bucket * 16 + k];
                    }
                } else {
                    for (size_t xx = updated[bucket]; xx < x; ++xx) {
                        size_t leave = low_edge(xx, r), enter = high_edge(xx, r + 1, job->w);
                        for (size_t k = 0; k < 16; ++k) {
                            fine[bucket][k] -= columns[leave].fine[bucket * 16 + k];
                            fine[bucket][k] += columns[enter].fine[bucket * 16 + k];
                        }
                    }
                }
                updated[bucket] = x;
                size_t k = 0;
                while (k < 15 && below + fine[bucket][k] <= middle)
                    below += fine[bucket][k++];
                job->out[image_index(job, y, x, c)] = (uint8_t)(bucket * 16 + k);
                size_t leave = low_edge(x, r), enter = high_edge(x, r + 1, job->w);
                for (size_t b = 0; b < 16; ++b) {
                    coarse[b] -= columns[leave].coarse[b];
                    coarse[b] += columns[enter].coarse[b];
                }
            }
            if (y == job->h - 1) break;
            size_t leave = low_edge(y, r), enter = high_edge(y, r + 1, job->h);
            for (size_t x = 0; x < job->w; ++x) {
                uint8_t old = job->src[image_index(job, leave, x, c)];
                uint8_t next = job->src[image_index(job, enter, x, c)];
                --columns[x].fine[old]; --columns[x].coarse[old >> 4];
                ++columns[x].fine[next]; ++columns[x].coarse[next >> 4];
            }
        }
    }
}

CN_EXPORT cn_status cn_neighborhood_median_u8(const uint8_t *src, uint8_t *out,
        size_t h, size_t w, size_t c, size_t radius) {
    if (!src || !out || radius < 3 || radius > 1000) return CN_INVALID_ARGUMENT;
    cn_status status = dimensions(h, w, c);
    if (status != CN_OK) return status;
    size_t narrow = h < w ? h : w;
    if (narrow > SIZE_MAX / sizeof(column_histogram) / c) return CN_SIZE_OVERFLOW;
    column_histogram *columns = calloc(narrow * c, sizeof(column_histogram));
    if (!columns) return CN_ALLOCATION_FAILED;
    histogram_job job = { src, out, columns, h < w ? w : h, narrow, c, radius, w, h < w };
    status = cn_parallel_for(c, 1, histogram_range, &job);
    free(columns);
    return status;
}

/* These float32 operation sequences adapt chaiNNer-rs image-ops/dither
   (MIT option), as used by installed chainner_ext 0.3.10.
   Copyright (c) 2023 Michael Schmidt. Full notice is distributed in
   backend/src/nodes/impl/chainner_native.LICENSE.txt.
   https://github.com/chaiNNer-org/chaiNNer-rs/tree/6f6ead6064f81b4049d3deb803c9279c7950736b/crates/image-ops/src/dither
   Error diffusion deliberately keeps scan order: rows cannot run independently. */
typedef struct diffusion_tap { int y, x; float weight; } diffusion_tap;
static const diffusion_tap taps[8][12] = {
    {{0,1,0.4375f}, {1,-1,0.1875f}, {1,0,0.3125f}, {1,1,0.0625f}},
    {{0,1,0.145833328f}, {0,2,0.104166664f}, {1,-2,0.0625f}, {1,-1,0.104166664f}, {1,0,0.145833328f}, {1,1,0.104166664f}, {1,2,0.0625f}, {2,-2,0.020833334f}, {2,-1,0.0625f}, {2,0,0.104166664f}, {2,1,0.0625f}, {2,2,0.020833334f}},
    {{0,1,0.190476194f}, {0,2,0.095238097f}, {1,-2,0.0476190485f}, {1,-1,0.095238097f}, {1,0,0.190476194f}, {1,1,0.095238097f}, {1,2,0.0476190485f}, {2,-2,0.0238095243f}, {2,-1,0.0476190485f}, {2,0,0.095238097f}, {2,1,0.0476190485f}, {2,2,0.0238095243f}},
    {{0,1,0.125f}, {0,2,0.125f}, {1,-1,0.125f}, {1,0,0.125f}, {1,1,0.125f}, {2,0,0.125f}},
    {{0,1,0.25f}, {0,2,0.125f}, {1,-2,0.0625f}, {1,-1,0.125f}, {1,0,0.25f}, {1,1,0.125f}, {1,2,0.0625f}},
    {{0,1,0.15625f}, {0,2,0.09375f}, {1,-2,0.0625f}, {1,-1,0.125f}, {1,0,0.15625f}, {1,1,0.125f}, {1,2,0.0625f}, {2,-1,0.0625f}, {2,0,0.09375f}, {2,1,0.0625f}},
    {{0,1,0.25f}, {0,2,0.1875f}, {1,-2,0.0625f}, {1,-1,0.125f}, {1,0,0.1875f}, {1,1,0.125f}, {1,2,0.0625f}},
    {{0,1,0.5f}, {1,-1,0.25f}, {1,0,0.25f}}
};
static const size_t tap_counts[8] = {4,12,12,6,7,10,7,3};

static float clip(float value) {
    if (value < 0) return 0;
    return value > 1 ? 1 : value;
}

static float vector_round(float value) {
    /* glam's SIMD round uses nearest-even, unlike the scalar gray quantizer. */
    float lower = floorf(value), fraction = value - lower;
    return fraction > 0.5f || (fraction == 0.5f && fmodf(lower, 2) != 0) ? lower + 1 : lower;
}

/* isnan and isfinite as bit tests on the float's word, whatever the MXCSR state
   (B3 called the CRT's _fdclass per pixel). */
static uint32_t float_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static int nan_bits(float value) {
    return (float_bits(value) & UINT32_C(0x7fffffff)) > UINT32_C(0x7f800000);
}

static int finite_bits(float value) {
    return (float_bits(value) & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}

static uint32_t float_order(float value) {
    uint32_t bits = float_bits(value);
    /* Rust f32::total_cmp order, including signed zeros and NaN payloads. */
    return bits ^ ((bits >> 31) ? UINT32_MAX : UINT32_C(0x80000000));
}

static int palette_bits_compare(const void *a, const void *b) {
    const palette_entry *x = a, *y = b;
    int order = memcmp(x->color, y->color, sizeof(x->color));
    if (order) return order;
    return x->ordinal < y->ordinal ? -1 : x->ordinal > y->ordinal;
}

static int palette_compare(const void *a, const void *b) {
    const palette_entry *x = a, *y = b;
    uint32_t kx = float_order(x->key), ky = float_order(y->key);
    if (kx != ky) return kx < ky ? -1 : 1;
    /* Deliberate deterministic correction: upstream uses randomized hash-set
       order for equal luminance. Preserve first input order for these ties. */
    return x->ordinal < y->ordinal ? -1 : x->ordinal > y->ordinal;
}

static int coordinate_compare(const dither_job *job, size_t a, size_t b, size_t axis) {
    float x = job->palette[a].color[axis], y = job->palette[b].color[axis];
    if (x < y) return -1;
    if (x > y) return 1;
    return a < b ? -1 : a > b;
}

static void index_swap(size_t *a, size_t *b) { size_t value = *a; *a = *b; *b = value; }

static void select_coordinate(const dither_job *job, size_t *indices,
                              size_t begin, size_t end, size_t middle, size_t axis) {
    while (end - begin > 1) {
        size_t center = begin + (end - begin) / 2;
        if (coordinate_compare(job, indices[center], indices[begin], axis) < 0)
            index_swap(indices + center, indices + begin);
        if (coordinate_compare(job, indices[end - 1], indices[begin], axis) < 0)
            index_swap(indices + end - 1, indices + begin);
        if (coordinate_compare(job, indices[end - 1], indices[center], axis) < 0)
            index_swap(indices + end - 1, indices + center);
        size_t pivot = indices[center], low = begin, at = begin, high = end;
        while (at < high) {
            int order = coordinate_compare(job, indices[at], pivot, axis);
            if (order < 0) index_swap(indices + at++, indices + low++);
            else if (order > 0) index_swap(indices + at, indices + --high);
            else ++at;
        }
        if (middle < low) end = low;
        else if (middle >= high) begin = high;
        else return;
    }
}

/* Balanced kd tree with conservative bounding boxes. Distances are still the
   exact ordered float32 sums used by the baseline, never approximations.
   Equal-distance candidates use palette order regardless of traversal order. */
static size_t build_palette_tree(dither_job *job, size_t *indices, size_t begin, size_t end) {
    if (begin == end) return SIZE_MAX;
    size_t middle = begin + (end - begin) / 2, axis = 0;
    palette_tree *node = job->tree + middle;
    float widest = -1;
    for (size_t c = 0; c < job->channels; ++c) {
        float low = CN_INFINITY_F, high = -CN_INFINITY_F;
        for (size_t i = begin; i < end; ++i) {
            float value = job->palette[indices[i]].color[c];
            if (value < low) low = value;
            if (value > high) high = value;
        }
        node->low[c] = low; node->high[c] = high;
        if (high - low > widest) { widest = high - low; axis = c; }
    }
    select_coordinate(job, indices, begin, end, middle, axis);
    node->color = indices[middle];
    node->left = build_palette_tree(job, indices, begin, middle);
    node->right = build_palette_tree(job, indices, middle + 1, end);
    return middle;
}

/* ordered: the colors are unique and already in lookup order (chainner_ext's
   PaletteQuantization order, which keeps its own palette's channel count), so neither
   the bit-order deduplication nor the luminance sort runs.
   300 or more unique finite colors get the kd tree. A palette holding a nonfinite
   color gets none, so every pixel takes the linear scan used below 300 colors; with a
   tree, a pixel with a nonfinite channel takes it too (nearest_palette). Upstream's
   R-tree decides those queries by its traversal; Consult 8 D-5 applies the sub-300
   rule there instead of refusing the input. */
static cn_status prepare_palette(dither_job *job, const float *palette, size_t count,
                                 int ordered) {
    if (count > SIZE_MAX / sizeof(palette_entry) ||
        count > SIZE_MAX / sizeof(float) / job->channels) return CN_SIZE_OVERFLOW;
    job->palette = calloc(count, sizeof(palette_entry));
    if (!job->palette) return CN_ALLOCATION_FAILED;
    int finite = 1;
    for (size_t i = 0; i < count; ++i) {
        palette_entry *entry = job->palette + i;
        memcpy(entry->color, palette + i * job->channels, job->channels * sizeof(float));
        entry->ordinal = i;
        for (size_t c = 0; c < job->channels; ++c)
            if (!finite_bits(entry->color[c])) finite = 0;
    }
    size_t unique = ordered ? count : 0;
    if (!ordered) {
        qsort(job->palette, count, sizeof(palette_entry), palette_bits_compare);
        for (size_t i = 0; i < count; ++i) {
            if (unique && !memcmp(job->palette[i].color, job->palette[unique - 1].color, 4 * sizeof(float)))
                continue;
            job->palette[unique++] = job->palette[i];
        }
    }
    job->palette_count = unique;
    for (size_t i = 0; !ordered && i < unique; ++i)
        job->palette[i].key = cn_palette_key(job->palette[i].color, job->channels);
    if (!ordered) qsort(job->palette, unique, sizeof(palette_entry), palette_compare);
    if (unique >= 300 && finite) {
        if (unique > SIZE_MAX / sizeof(palette_tree) || unique > SIZE_MAX / sizeof(size_t))
            return CN_SIZE_OVERFLOW;
        job->tree = malloc(unique * sizeof(palette_tree));
        size_t *indices = malloc(unique * sizeof(size_t));
        if (!job->tree || !indices) { free(indices); return CN_ALLOCATION_FAILED; }
        for (size_t i = 0; i < unique; ++i) indices[i] = i;
        job->tree_root = build_palette_tree(job, indices, 0, unique);
        free(indices);
    }
    return CN_OK;
}

static float palette_distance(const dither_job *job, size_t index, const float *color) {
    float distance = 0;
    for (size_t c = 0; c < job->channels; ++c) {
        float delta = job->palette[index].color[c] - color[c];
        distance += delta * delta;
    }
    return distance;
}

static float palette_bound(const dither_job *job, size_t index, const float *color) {
    if (index == SIZE_MAX) return CN_INFINITY_F;
    const palette_tree *node = job->tree + index;
    float distance = 0;
    for (size_t c = 0; c < job->channels; ++c) {
        float delta = color[c] < node->low[c] ? node->low[c] - color[c] :
                      color[c] > node->high[c] ? node->high[c] - color[c] : 0;
        distance += delta * delta;
    }
    return distance;
}

static void search_palette(const dither_job *job, size_t index, const float *color,
                           size_t *best, float *best_distance) {
    const palette_tree *node = job->tree + index;
    float distance = palette_distance(job, node->color, color);
    if (distance < *best_distance || (distance == *best_distance && node->color < *best)) {
        *best = node->color; *best_distance = distance;
    }
    float left_bound = palette_bound(job, node->left, color);
    float right_bound = palette_bound(job, node->right, color);
    size_t first = node->left, second = node->right;
    if (right_bound < left_bound) {
        first = node->right; second = node->left;
        float temporary = left_bound; left_bound = right_bound; right_bound = temporary;
    }
    if (first != SIZE_MAX && left_bound <= *best_distance)
        search_palette(job, first, color, best, best_distance);
    if (second != SIZE_MAX && right_bound <= *best_distance)
        search_palette(job, second, color, best, best_distance);
}

static int finite_color(const dither_job *job, const float *color) {
    for (size_t c = 0; c < job->channels; ++c) if (!finite_bits(color[c])) return 0;
    return 1;
}

/* The linear scan: strict <, so the first index reaching the minimum distance wins, and
   index 0 when its distance is NaN (nothing compares below it). At ISA level avx2, with
   the channel-major palette, the AVX2 unit's brute-force search gives the same index. */
static size_t linear_nearest(const dither_job *job, const float *color) {
#if defined(_MSC_VER) && defined(_M_X64)
    if (job->isa >= CN_ISA_AVX2 && job->soa) return cn_dither_nearest_avx2(job, color);
#endif
    size_t best = 0;
    float best_distance = palette_distance(job, 0, color);
    for (size_t p = 1; p < job->palette_count; ++p) {
        float distance = palette_distance(job, p, color);
        if (distance < best_distance) { best = p; best_distance = distance; }
    }
    return best;
}

static void nearest_palette(const dither_job *job, const float *color, float *nearest) {
    size_t best = 0;
    if (job->tree && finite_color(job, color)) {
        float best_distance = palette_distance(job, 0, color);
        search_palette(job, job->tree_root, color, &best, &best_distance);
    } else {
        best = linear_nearest(job, color);
    }
    for (size_t c = 0; c < job->channels; ++c) nearest[c] = job->palette[best].color[c];
}

static void independent_dither_range(void *context, size_t begin, size_t end) {
    dither_job *job = (dither_job *)context;
    for (size_t p = begin; p < end; ++p) {
        size_t offset = p * job->channels;
        if (job->palette_count) {
            nearest_palette(job, job->src + offset, job->out + offset);
            continue;
        }
        float threshold = 0.5f;
        if (job->mode == 1) {
            size_t y = p / job->width % job->map_size, x = p % job->width % job->map_size;
            threshold = job->map[y * job->map_stride + x];
        }
        for (size_t c = 0; c < job->channels; ++c) {
            float value = job->src[offset + c];
            if (job->colors == 2)
                value = value >= (job->mode == 1 ? 1.0f - threshold : 0.5f) ? 1.0f : 0.0f;
            else {
                value = floorf(value * job->factor + threshold);
                value = job->mode == 1 ? value / job->factor : value * job->inverse;
            }
            job->out[offset + c] = value;
        }
    }
}

/* Uniform diffusion channels have independent histories and may run in
   parallel. Palette distances couple channels, so a palette scan stays ordered. */
static void diffuse_group(dither_job *job, size_t channel, size_t c) {
    size_t row_size = (job->width + 4) * c;
    float *memory = job->error_memory + channel * (job->width + 4) * 3;
    float *rows[3] = {memory, memory + row_size, memory + 2 * row_size};
    for (size_t y = 0; y < job->height; ++y) {
        float *old = rows[0]; rows[0] = rows[1]; rows[1] = rows[2]; rows[2] = old;
        memset(rows[2], 0, row_size * sizeof(float));
        for (size_t x = 0; x < job->width; ++x) {
            float color[4], nearest[4], error[4];
            size_t offset = (y * job->width + x) * job->channels + channel, error_x = (x + 2) * c;
            for (size_t k = 0; k < c; ++k) {
                color[k] = job->src[offset + k] + rows[0][error_x + k];
                if (job->palette_count) color[k] = job->channels > 1 && nan_bits(color[k]) ? 0 : clip(color[k]);
            }
            if (job->palette_count) nearest_palette(job, color, nearest);
            else for (size_t k = 0; k < c; ++k) {
                float value = color[k] * job->factor;
                value = job->channels == 1 ? floorf(value + 0.5f) : vector_round(value);
                value *= job->inverse;
                nearest[k] = job->channels > 1 && isnan(value) ? 0 : clip(value);
            }
            for (size_t k = 0; k < c; ++k) {
                job->out[offset + k] = nearest[k]; error[k] = color[k] - nearest[k];
            }
            for (size_t t = 0; t < tap_counts[job->algorithm]; ++t) {
                const diffusion_tap *tap = taps[job->algorithm] + t;
                size_t xx = tap->x < 0 ? x + 2 - (size_t)(-tap->x) : x + 2 + (size_t)tap->x;
                for (size_t k = 0; k < c; ++k)
                    rows[tap->y][xx * c + k] += error[k] * tap->weight;
            }
        }
    }
}

static void diffusion_range(void *context, size_t begin, size_t end) {
    dither_job *job = (dither_job *)context;
    for (size_t channel = begin; channel < end; ++channel)
        diffuse_group(job, channel, 1);
}

/* The rectangular traversal adapts zhang_hilbert 0.1.1, src/core.rs and
   src/arb.rs (MIT option), Copyright 2017 yvt. The full notice is included
   in chainner_native.LICENSE.txt. The state is local to each invocation;
   no image-sized coordinate list or recursion is required. */
typedef struct hilbert_level {
    size_t size[2];
    unsigned int curve, progress;
} hilbert_level;

typedef struct hilbert_scan {
    size_t size[2], levels, last, position[2], progress[2];
    hilbert_level level[64];
    unsigned int curve, end;
    int secondary_negative, helper, done;
} hilbert_scan;

static const unsigned int curve_address[8] = {180,120,75,135,30,45,225,210};
static const unsigned int curve_induction[8][4] = {
    {1,0,0,3}, {0,1,1,2}, {3,2,2,1}, {2,3,3,0},
    {7,4,4,5}, {6,5,5,4}, {5,6,6,7}, {4,7,7,6}
};

static size_t integer_log2(size_t value) {
    size_t result = 0;
    while (value >>= 1) ++result;
    return result;
}

static size_t hilbert_division(size_t size) {
    size_t mask = (size_t)1 << (integer_log2(size) - 1);
    return (size & mask) + mask;
}

static void hilbert_extra(const size_t size[2], unsigned int position,
                          unsigned int curve, size_t result[2]) {
    position ^= curve == 0 || curve == 5;
    for (size_t axis = 0; axis < 2; ++axis) {
        size_t larger = ((size[axis] + 3) >> 2) << 1;
        result[axis] = (position & (2u >> axis)) ? larger : size[axis] - larger;
    }
}

static void hilbert_block(hilbert_scan *scan, unsigned int curve, const size_t size[2]) {
    scan->secondary_negative = (curve & 2) != 0;
    scan->curve = curve; scan->end = curve_address[curve] >> 6;
    scan->progress[0] = size[curve & 1];
    scan->progress[1] = size[(curve & 1) ^ 1];
}

static void hilbert_init(hilbert_scan *scan, size_t width, size_t height) {
    memset(scan, 0, sizeof(*scan));
    scan->size[0] = width; scan->size[1] = height;
    scan->levels = 1;
    if (width == 1 || height == 1) {
        scan->progress[0] = 1; scan->progress[1] = width == 1 ? height : width;
        scan->curve = width == 1 ? 0 : 1;
        return;
    }
    scan->levels = integer_log2(width < height ? width : height) + 1;
    scan->level[0].size[0] = width; scan->level[0].size[1] = height;
    for (size_t i = 1; i <= scan->levels - 2; ++i) {
        for (size_t axis = 0; axis < 2; ++axis) {
            size_t previous = scan->level[i - 1].size[axis];
            scan->level[i].size[axis] = previous - hilbert_division(previous);
        }
        scan->level[i].curve = (unsigned int)(i % 2);
    }
    scan->last = scan->levels - 2;
    unsigned int curve;
    if (width & 1) { curve = 0; scan->helper = 1; }
    else if (height & 1) { curve = scan->levels == 2 ? 0 : 1; scan->helper = scan->levels != 2; }
    else curve = (unsigned int)(scan->last % 2);
    hilbert_level *last = scan->level + scan->last;
    if (scan->helper) --last->size[curve & 1];
    last->curve = curve;
    size_t size[2] = {last->size[0], last->size[1]};
    if (size[0] >= 3 && size[1] >= 3) {
        hilbert_extra(last->size, 0, curve, size);
        curve = curve_induction[curve][0];
        ++scan->last;
        memcpy(scan->level[scan->last].size, size, sizeof(size));
    }
    hilbert_block(scan, curve, size);
}

static void hilbert_primary_step(hilbert_scan *scan, size_t axis) {
    if ((scan->curve ^ (scan->curve >> 1)) & 2) --scan->position[axis];
    else ++scan->position[axis];
}

static int hilbert_next(hilbert_scan *scan, size_t position[2]) {
    if (scan->done) return 0;
    memcpy(position, scan->position, 2 * sizeof(size_t));
    size_t primary = scan->progress[0], secondary = scan->progress[1] - 1;
    size_t primary_axis = scan->curve & 1, secondary_axis = primary_axis ^ 1;
    if (secondary) {
        if (scan->secondary_negative) --scan->position[secondary_axis];
        else ++scan->position[secondary_axis];
        scan->progress[1] = secondary;
        return 1;
    }
    --primary;
    secondary = scan->level[scan->last].size[secondary_axis];
    scan->secondary_negative = !scan->secondary_negative;
    if (primary) {
        hilbert_primary_step(scan, primary_axis);
        scan->progress[0] = primary; scan->progress[1] = secondary;
        return 1;
    }
    if (scan->helper && (scan->last == scan->levels - 2 ||
                        scan->level[scan->levels - 2].progress == 3)) {
        hilbert_level *level = scan->level + scan->levels - 2;
        size_t axis = level->curve & 1;
        scan->end = 3; scan->curve = level->curve; scan->secondary_negative = 0;
        scan->progress[0] = 1; scan->progress[1] = level->size[axis ^ 1];
        scan->helper = 0; scan->last = scan->levels - 2;
        hilbert_primary_step(scan, axis);
        return 1;
    }
    if (!scan->last) { scan->done = 1; return 1; }
    size_t i = scan->last - 1;
    unsigned int enter;
    for (;;) {
        hilbert_level *level = scan->level + i;
        if (++level->progress == 4) {
            if (!i) { scan->done = 1; return 1; }
            --i;
        } else {
            unsigned int address = curve_address[level->curve] >> (level->progress * 2 - 2);
            unsigned int relative = address ^ (address >> 2);
            if ((relative >> secondary_axis) & 1) hilbert_primary_step(scan, primary_axis);
            else if (scan->secondary_negative) ++scan->position[secondary_axis];
            else --scan->position[secondary_axis];
            enter = scan->end ^ (relative & 3);
            break;
        }
    }
    if (i == scan->levels - 2) {
        hilbert_level *level = scan->level + i;
        unsigned int address = curve_address[level->curve] >> (level->progress * 2);
        unsigned int curve = curve_induction[level->curve][level->progress];
        hilbert_extra(level->size, address, level->curve, scan->level[i + 1].size);
        hilbert_block(scan, curve, scan->level[i + 1].size);
        return 1;
    }
    while (i < scan->levels - 2) {
        hilbert_level *level = scan->level + i, *next = level + 1;
        unsigned int address = curve_address[level->curve] >> (level->progress * 2);
        for (size_t axis = 0; axis < 2; ++axis) {
            size_t larger = hilbert_division(level->size[axis]);
            next->size[axis] = (address & (2u >> axis)) ? larger : level->size[axis] - larger;
        }
        next->curve = curve_induction[level->curve][level->progress]; next->progress = 0;
        ++i;
    }
    size_t size[2] = {scan->level[i].size[0], scan->level[i].size[1]};
    unsigned int parity = (unsigned int)((size[0] & 1) * 2 + (size[1] & 1));
    unsigned int curve;
    int helper = 0;
    if (!parity) {
        static const unsigned int scanning_type[2][4][2] = {
            {{0,1},{6,6},{7,7},{3,2}}, {{1,0},{5,5},{4,4},{2,3}}
        };
        unsigned int direction = 0, negative = 0;
        size_t parent = i - 1;
        for (;;) {
            hilbert_level *level = scan->level + parent;
            if (level->progress == 3) { if (!parent) break; --parent; }
            else {
                unsigned int address = curve_address[level->curve] >> (level->progress * 2);
                unsigned int relative = address ^ (address >> 2);
                direction = relative & 1; negative = (address & relative & 3) != 0;
                break;
            }
        }
        curve = scanning_type[negative][enter][direction];
    } else if (parity == 1) {
        helper = scan->position[0] + size[0] == scan->size[0] &&
                 scan->position[1] + 1 == size[1];
        curve = helper ? 5 : 6;
    } else curve = 7;
    if (helper) { --size[1]; memcpy(scan->level[i].size, size, sizeof(size)); }
    scan->level[i].curve = curve;
    if (size[0] >= 3 && size[1] >= 3) {
        scan->level[i].progress = 0;
        hilbert_extra(scan->level[i].size, enter, curve, size);
        curve = curve_induction[curve][0]; ++i;
        memcpy(scan->level[i].size, size, sizeof(size));
    }
    hilbert_block(scan, curve, size); scan->helper = helper; scan->last = i;
    return 1;
}

static size_t hilbert_part(size_t remaining, size_t minor) {
    size_t count = 1;
    if (remaining > minor) {
        size_t k = remaining / minor;
        size_t first = remaining / k - minor, second = minor - remaining / (k + 1);
        count = first < second ? k : k + 1;
    }
    if (count == 1) return remaining;
    size_t width = remaining / count;
    return width + (width & 1);
}

static cn_status riemersma(dither_job *job, uint32_t history_length, float decay_ratio) {
    float base = expf(logf(decay_ratio) / ((float)history_length - 1.0f));
    if (!(base > 0 && base < 1)) return CN_INVALID_ARGUMENT;
    size_t c = job->channels;
    if (history_length > SIZE_MAX / sizeof(float) / c) return CN_SIZE_OVERFLOW;
    float *history = calloc((size_t)history_length * c, sizeof(float));
    if (!history) return CN_ALLOCATION_FAILED;
    size_t history_index = 0;
    size_t major = job->width > job->height ? job->width : job->height;
    size_t minor = job->width > job->height ? job->height : job->width;
    int transpose = job->height > job->width;
    for (size_t start = 0; start < major;) {
        size_t length = hilbert_part(major - start, minor);
        hilbert_scan scan;
        hilbert_init(&scan, length, minor);
        size_t position[2];
        while (hilbert_next(&scan, position)) {
            size_t x = transpose ? position[1] : position[0] + start;
            size_t y = transpose ? position[0] + start : position[1];
            /* This guard also protects the foreign ABI if an impossible scan
               state is reached; no output can address beyond the image. */
            if (x >= job->width || y >= job->height) { free(history); return CN_INVALID_ARGUMENT; }
            size_t offset = (y * job->width + x) * c;
            float error[4] = {0}, color[4], nearest[4];
            for (size_t k = 0; k < history_length; ++k)
                for (size_t channel = 0; channel < c; ++channel)
                    error[channel] += history[k * c + channel];
            for (size_t k = 0; k < (size_t)history_length * c; ++k) history[k] *= base;
            for (size_t channel = 0; channel < c; ++channel) {
                color[channel] = job->src[offset + channel] + error[channel];
                if (job->palette_count) color[channel] = c > 1 && nan_bits(color[channel]) ? 0 : clip(color[channel]);
            }
            if (job->palette_count) nearest_palette(job, color, nearest);
            else for (size_t channel = 0; channel < c; ++channel) {
                float value = color[channel] * job->factor;
                value = c == 1 ? floorf(value + 0.5f) : vector_round(value);
                value *= job->inverse;
                nearest[channel] = c > 1 && isnan(value) ? 0 : clip(value);
            }
            for (size_t channel = 0; channel < c; ++channel) {
                job->out[offset + channel] = nearest[channel];
                /* Riemersma feeds original-minus-nearest into history, unlike
                   raster diffusion's error-adjusted-color-minus-nearest. */
                history[history_index * c + channel] = job->src[offset + channel] - nearest[channel];
            }
            history_index = (history_index + 1) % history_length;
        }
        start += length;
    }
    free(history);
    return CN_OK;
}

CN_EXPORT cn_status cn_neighborhood_uniform_riemersma(const float *src, float *out,
        size_t h, size_t w, size_t c, uint32_t colors, uint32_t history_length, float decay_ratio) {
    if (!src || !out || colors < 2 || history_length < 2) return CN_INVALID_ARGUMENT;
    cn_status status = dimensions(h, w, c);
    if (status != CN_OK) return status;
    dither_job job = {0};
    job.src = src; job.out = out; job.width = w; job.height = h; job.channels = c;
    job.colors = colors; job.factor = (float)(colors - 1); job.inverse = 1 / job.factor;
    return riemersma(&job, history_length, decay_ratio);
}

/* compatible, on the three palette entries: optional, and written 1 when given. Every
   input is dithered since Consult 8 D-5; the slot stays because the B3 golden manifests
   and the frozen reference_image_preparation/native_neighborhood.py record it, and its
   removal is a ruling. */
CN_EXPORT cn_status cn_neighborhood_palette_riemersma(const float *src, float *out,
        size_t h, size_t w, size_t c, const float *palette, size_t count,
        uint32_t history_length, float decay_ratio, int *compatible) {
    if (!src || !out || !palette || !count || history_length < 2)
        return CN_INVALID_ARGUMENT;
    if (compatible) *compatible = 1;
    cn_status status = dimensions(h, w, c);
    if (status != CN_OK) return status;
    dither_job job = {0};
    job.src = src; job.out = out; job.width = w; job.height = h; job.channels = c;
    status = prepare_palette(&job, palette, count, 0);
    if (status == CN_OK) status = riemersma(&job, history_length, decay_ratio);
    free(job.tree); free(job.palette);
    return status;
}

/* create_threshold_map's cell (y, x) of an n x n map, n = 2^bits: the bits of y and of
   y ^ x interleaved from the top, over n * n, both converted to float32 as `as f32`. */
static float threshold_cell(size_t y, size_t x, unsigned int bits, size_t map_size) {
    uint64_t value = 0;
    unsigned int bit = 0;
    for (unsigned int mask = bits; mask-- > 0;) {
        value |= (uint64_t)((y >> mask) & 1) << bit++;
        value |= (uint64_t)(((y ^ x) >> mask) & 1) << bit++;
    }
    return (float)value / (float)((uint64_t)map_size * map_size);
}

/* Uniform quantization and ordered dithering are per sample (quantize_ndim and
   ordered_dither take any channel count); palettes and diffusion keep 1, 3 or 4. */
static cn_status sample_dimensions(size_t h, size_t w, size_t c) {
    if (!h || !w || !c) return CN_INVALID_ARGUMENT;
    if (h > SIZE_MAX / w || h * w > SIZE_MAX / sizeof(float) / c)
        return CN_SIZE_OVERFLOW;
    return CN_OK;
}

CN_EXPORT cn_status cn_neighborhood_dither(const float *src, float *out,
        size_t h, size_t w, size_t c, uint32_t colors, int mode, size_t map_size,
        int algorithm, const float *palette, size_t palette_count, int *compatible) {
    /* Ordered maps: any power of two up to 2^31, ordered_dither's u32 domain. */
    if (!src || !out || mode < 0 || mode > 2 ||
        algorithm < 0 || algorithm > 7 || (!palette && colors < 2) ||
        (palette && (!palette_count || mode == 1)) ||
        (mode == 1 && (!map_size || (map_size & (map_size - 1)) ||
                       map_size > ((size_t)1 << 31))))
        return CN_INVALID_ARGUMENT;
    if (compatible) *compatible = 1;
    cn_status status = palette || mode == 2 ? dimensions(h, w, c) : sample_dimensions(h, w, c);
    if (status != CN_OK) return status;
    dither_job job = {0};
    job.src = src; job.out = out; job.width = w; job.channels = c;
    job.height = h; job.algorithm = algorithm;
    job.colors = colors; job.mode = mode; job.map_size = map_size;
    job.factor = (float)(colors - 1); job.inverse = 1 / job.factor;
    if (palette) {
        status = prepare_palette(&job, palette, palette_count, 0);
        if (status != CN_OK) { free(job.tree); free(job.palette); return status; }
    }
    float *large_map = NULL;
    if (mode == 1) {
        unsigned int bits = (unsigned int)integer_log2(map_size);
        if (map_size <= 16) {
            for (size_t y = 0; y < map_size; ++y)
                for (size_t x = 0; x < map_size; ++x)
                    job.thresholds[y * map_size + x] = threshold_cell(y, x, bits, map_size);
            job.map = job.thresholds; job.map_stride = map_size;
        } else {
            size_t rows = h < map_size ? h : map_size, columns = w < map_size ? w : map_size;
            large_map = malloc(rows * columns * sizeof(float));
            if (!large_map) return CN_ALLOCATION_FAILED;
            for (size_t y = 0; y < rows; ++y)
                for (size_t x = 0; x < columns; ++x)
                    large_map[y * columns + x] = threshold_cell(y, x, bits, map_size);
            job.map = large_map; job.map_stride = columns;
        }
    }
    if (mode != 2) {
        status = cn_parallel_for(h * w, 16384, independent_dither_range, &job);
        free(large_map); free(job.tree); free(job.palette);
        return status;
    }
    if (w > SIZE_MAX - 4 || w + 4 > SIZE_MAX / sizeof(float) / c / 3) {
        free(job.tree); free(job.palette); return CN_SIZE_OVERFLOW;
    }
    size_t row_size = (w + 4) * c;
    float *memory = calloc(row_size * 3, sizeof(float));
    if (!memory) { free(job.tree); free(job.palette); return CN_ALLOCATION_FAILED; }
    job.error_memory = memory;
    if (palette) diffuse_group(&job, 0, c);
    else status = cn_parallel_for(c, h * w >= 16384 ? 1 : c, diffusion_range, &job);
    free(memory); free(job.tree); free(job.palette);
    return status;
}


/* Prepared palette ownership is separate from per-image diffusion state. The
 * Python bridge holds a strong plan reference across every native call. Plans
 * are immutable after creation and may be read by independent calls/threads. */
typedef struct palette_plan {
    palette_entry *palette;
    palette_tree *tree;
    size_t count, channels, root;
} palette_plan;

static cn_status palette_create(const float *colors, size_t count, size_t channels,
        void **out, size_t *bytes, int ordered) {
    if (!colors || !count || !out || !bytes ||
        (channels != 1 && channels != 3 && channels != 4) ||
        (uintptr_t)colors % _Alignof(float)) return CN_INVALID_ARGUMENT;
    *out = NULL; *bytes = 0;
    if (count > SIZE_MAX / sizeof(float) / channels ||
        (uintptr_t)colors > UINTPTR_MAX - count * channels * sizeof(float))
        return CN_SIZE_OVERFLOW;
    palette_plan *plan = calloc(1, sizeof(palette_plan));
    if (!plan) return CN_ALLOCATION_FAILED;
    dither_job job = {0};
    job.channels = channels;
    cn_status status = prepare_palette(&job, colors, count, ordered);
    if (status != CN_OK) { free(job.tree); free(job.palette); free(plan); return status; }
    size_t entry_bytes = count * sizeof(palette_entry);
    size_t tree_bytes = job.tree ? job.palette_count * sizeof(palette_tree) : 0;
    if (entry_bytes > SIZE_MAX - tree_bytes ||
        entry_bytes + tree_bytes > SIZE_MAX - sizeof(palette_plan)) {
        free(job.tree); free(job.palette); free(plan); return CN_SIZE_OVERFLOW;
    }
    plan->palette = job.palette; plan->tree = job.tree;
    plan->count = job.palette_count; plan->channels = channels; plan->root = job.tree_root;
    *out = plan; *bytes = sizeof(palette_plan) + entry_bytes + tree_bytes;
    return CN_OK;
}

CN_EXPORT cn_status cn_neighborhood_palette_create(const float *colors,
        size_t count, size_t channels, void **out, size_t *bytes) {
    return palette_create(colors, count, channels, out, bytes, 0);
}

/* chainner_ext's PaletteQuantization: colors unique and in its lookup order (its key
   sort over its own channel count, then any 1 -> 3/4 channel expansion), kept as given. */
CN_EXPORT cn_status cn_neighborhood_palette_create_ordered(const float *colors,
        size_t count, size_t channels, void **out, size_t *bytes) {
    return palette_create(colors, count, channels, out, bytes, 1);
}

CN_EXPORT void cn_neighborhood_palette_free(void *handle) {
    palette_plan *plan = handle;
    if (plan) { free(plan->tree); free(plan->palette); free(plan); }
}

CN_EXPORT cn_status cn_neighborhood_palette_apply(const float *src, float *out,
        size_t h, size_t w, size_t c, const void *handle, int mode, int algorithm,
        uint32_t history_length, float decay_ratio, int *compatible) {
    const palette_plan *plan = handle;
    if (!src || !out || !plan || plan->channels != c ||
        (mode != 0 && mode != 2 && mode != 3) || algorithm < 0 || algorithm > 7 ||
        (mode == 3 && history_length < 2) ||
        (uintptr_t)src % _Alignof(float) || (uintptr_t)out % _Alignof(float))
        return CN_INVALID_ARGUMENT;
    cn_status status = dimensions(h, w, c);
    if (status != CN_OK) return status;
    size_t bytes = h * w * c * sizeof(float);
    uintptr_t a = (uintptr_t)src, b = (uintptr_t)out;
    if (a > UINTPTR_MAX - bytes || b > UINTPTR_MAX - bytes) return CN_SIZE_OVERFLOW;
    if (a < b + bytes && b < a + bytes) return CN_INVALID_ARGUMENT;
    if (compatible) *compatible = 1;
    dither_job job = {0};
    job.src = src; job.out = out; job.width = w; job.height = h; job.channels = c;
    job.algorithm = algorithm; job.mode = mode; job.colors = 2;
    job.factor = job.inverse = 1;
    job.palette = plan->palette; job.palette_count = plan->count;
    job.tree = plan->tree; job.tree_root = plan->root;
    job.isa = cn_isa_current();
#if defined(_MSC_VER) && defined(_M_X64)
    /* Without a kd tree (fewer than 300 unique colors, or a nonfinite one), the AVX2
       search reads the sorted palette by channel, up to DITHER_SOA_ENTRIES colors; the
       lanes past the count are zero and masked. */
    float soa[DITHER_SOA_ENTRIES * 4];
    if (job.isa >= CN_ISA_AVX2 && !job.tree && plan->count <= DITHER_SOA_ENTRIES) {
        size_t stride = (plan->count + 7) / 8 * 8;
        for (size_t k = 0; k < c; ++k)
            for (size_t i = 0; i < stride; ++i)
                soa[k * stride + i] = i < plan->count ? plan->palette[i].color[k] : 0;
        job.soa = soa; job.soa_stride = stride;
    }
#endif
    if (mode == 0) return cn_parallel_for(h * w, 16384, independent_dither_range, &job);
    if (mode == 3) return riemersma(&job, history_length, decay_ratio);
    if (w > SIZE_MAX - 4 || w + 4 > SIZE_MAX / sizeof(float) / c / 3)
        return CN_SIZE_OVERFLOW;
    job.error_memory = calloc((w + 4) * c * 3, sizeof(float));
    if (!job.error_memory) return CN_ALLOCATION_FAILED;
    diffuse_group(&job, 0, c);
    free(job.error_memory);
    return CN_OK;
}
