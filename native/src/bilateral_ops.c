/* Altered C extraction of OpenCV 5.0.0 bilateral_filter.dispatch.cpp and
 * bilateral_filter.simd.hpp's float32 bilateral algorithm, with the float32
 * minMaxIdx scan that sets its range (core/src/minmax.simd.hpp and the float
 * reductions of core/hal/intrin_sse.hpp and intrin_avx.hpp). The proprietary
 * default IPP implementation is deliberately not approximated here.
 *
 * Copyright (C) 2000-2008, 2018, Intel Corporation, all rights reserved.
 * Copyright (C) 2009, Willow Garage Inc., all rights reserved.
 * Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
 * Copyright (C) 2025, Advanced Micro Devices, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * * Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * The name of the copyright holders may not be used to endorse or promote
 *   products derived from this software without specific prior written permission.
 * This software is provided by the copyright holders and contributors "as is"
 * and any express or implied warranties, including, but not limited to, the
 * implied warranties of merchantability and fitness for a particular purpose
 * are disclaimed. In no event shall the Intel Corporation or contributors be
 * liable for any direct, indirect, incidental, special, exemplary, or
 * consequential damages (including, but not limited to, procurement of
 * substitute goods or services; loss of use, data, or profits; or business
 * interruption) however caused and on any theory of liability, whether in
 * contract, strict liability, or tort (including negligence or otherwise)
 * arising in any way out of the use of this software, even if advised of
 * the possibility of such damage.
 *
 * minmax.simd.hpp is part of the OpenCV project under the Apache-2.0 license
 * reproduced in chainner_native.LICENSE.txt.
 */
#include "chainner.h"
#include "parallel.h"
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Status 4 is specific to this export: the original would index outside its
 * lookup table (a NaN, infinite or reversed range). The Python boundary
 * preserves OpenCV's exception, without reproducing its invalid gather/access
 * violation, and nothing is written to the output. */
#define CN_BILATERAL_RANGE_ERROR ((cn_status)4)

typedef struct bilateral_context {
    const float *padded, *weights, *lut;
    const ptrdiff_t *offsets;
    float *out;
    unsigned char *invalid;
    size_t width, channels, stride, radius, count, vector_end;
    float scale, top;
    int bins, fused;
} bilateral_context;

static size_t reflect101(ptrdiff_t p, size_t size) {
    if (size == 1) return 0;
    ptrdiff_t period = (ptrdiff_t)(2 * (size - 1));
    p %= period;
    if (p < 0) p += period;
    return (size_t)(p < (ptrdiff_t)size ? p : period - p);
}

/* _mm_min_ps/_mm_max_ps(a, b): b when either operand is NaN. */
static float x86_min(float a, float b) { return a < b ? a : b; }
static float x86_max(float a, float b) { return a > b ? a : b; }

/* v_reduce_min/max: AVX2 folds the high half onto the low one, then pairs;
 * SSE stores the lanes and folds with std::min/max (a when either is NaN). */
static float reduce(const float *v, size_t lanes, int maximum) {
    if (lanes == 8) {
        float m[4];
        for (size_t i = 0; i < 4; ++i)
            m[i] = maximum ? x86_max(v[i], v[i + 4]) : x86_min(v[i], v[i + 4]);
        float a = maximum ? x86_max(m[0], m[2]) : x86_min(m[0], m[2]);
        float b = maximum ? x86_max(m[1], m[3]) : x86_min(m[1], m[3]);
        return maximum ? x86_max(a, b) : x86_min(a, b);
    }
    float a = maximum ? (v[0] < v[1] ? v[1] : v[0]) : (v[1] < v[0] ? v[1] : v[0]);
    float b = maximum ? (v[2] < v[3] ? v[3] : v[2]) : (v[3] < v[2] ? v[3] : v[2]);
    return maximum ? (a < b ? b : a) : (b < a ? b : a);
}

/* minMaxIdx32f over one plane: the whole image, or one row of a Mat whose rows
 * are apart. The first plane seeds both extrema with its first value. Each
 * lane keeps its own extrema; the plane's become their reduction only if an
 * ordered comparison improved one. The remainder is scalar. */
static void range_plane(const float *src, size_t len, size_t lanes, int *started,
                        float *low, float *high) {
    size_t i = 0;
    float lo = *low, hi = *high, lane_lo[8], lane_hi[8];
    int lower = 0, higher = 0;
    if (!*started) {
        lo = hi = src[0];
        i = 1;
        *started = 1;
    }
    for (size_t l = 0; l < lanes; ++l) {
        lane_lo[l] = lo;
        lane_hi[l] = hi;
    }
    for (; i + lanes <= len; i += lanes) {
        for (size_t l = 0; l < lanes; ++l) {
            float value = src[i + l];
            lower |= value < lane_lo[l];
            higher |= value > lane_hi[l];
            lane_lo[l] = x86_min(lane_lo[l], value);
            lane_hi[l] = x86_max(lane_hi[l], value);
        }
    }
    if (lower) lo = reduce(lane_lo, lanes, 0);
    if (higher) hi = reduce(lane_hi, lanes, 1);
    for (; i < len; ++i) {
        if (src[i] < lo) lo = src[i];
        if (src[i] > hi) hi = src[i];
    }
    *low = lo;
    *high = hi;
}

static float multiply_add(float a, float b, float c, int fused) {
    return fused ? fmaf(a, b, c) : a * b + c;
}

/* The vector loop's interpolation: v_trunc (INT_MIN for NaN and out-of-range
 * values), clamped to the table end by v_min; FMA under AVX2 and AVX-512. A
 * negative index would gather outside the table. */
static float vector_lookup(const bilateral_context *j, float alpha, int *invalid) {
    if (!(alpha > -1.0f && alpha < 2147483648.0f)) {
        *invalid = 1;
        return 0;
    }
    int index = (int)alpha;
    if (index > j->bins) index = j->bins;
    alpha -= (float)index;
    return multiply_add(j->lut[index + 1], alpha, j->lut[index] * (1.0f - alpha), j->fused);
}

/* The scalar tail's: std::min with the table end, cvFloor, and a separately
 * rounded interpolation (OpenCV's /fp:precise build contracts nothing). */
static float scalar_lookup(const bilateral_context *j, float alpha, int *invalid) {
    if (j->top < alpha) alpha = j->top;
    if (!(alpha >= 0.0f)) {
        *invalid = 1;
        return 0;
    }
    int index = (int)alpha;
    alpha -= (float)index;
    return j->lut[index] + alpha * (j->lut[index + 1] - j->lut[index]);
}

static float vector_1(const bilateral_context *j, const float *center, int *invalid) {
    float rval = *center, wsum = 0.0f, sum = 0.0f;
    int center_valid = !isnan(rval);
    for (size_t k = 0; k < j->count; ++k) {
        float val = center[j->offsets[k]];
        if (isnan(val)) continue;
        float alpha = center_valid ? fabsf(val - rval) * j->scale : 0.0f;
        float w = j->weights[k] * vector_lookup(j, alpha, invalid);
        wsum += w;
        sum = multiply_add(val, w, sum, j->fused);
    }
    return (sum + (center_valid ? rval : 0.0f)) / (wsum + (center_valid ? 1.0f : 0.0f));
}

static float scalar_1(const bilateral_context *j, const float *center, int *invalid) {
    float rval = *center, wsum = 0.0f, sum = 0.0f;
    int center_valid = !isnan(rval);
    for (size_t k = 0; k < j->count; ++k) {
        float val = center[j->offsets[k]];
        if (isnan(val)) continue;
        float w = j->weights[k] * (center_valid
            ? scalar_lookup(j, fabsf(val - rval) * j->scale, invalid) : 1.0f);
        wsum += w;
        sum += val * w;
    }
    return center_valid ? (sum + rval) / (wsum + 1.0f) : sum / wsum;
}

static float difference_3(const float *a, const float *b) {
    return (fabsf(a[0] - b[0]) + fabsf(a[1] - b[1])) + fabsf(a[2] - b[2]);
}

static void vector_3(const bilateral_context *j, const float *center, float *out,
                     int *invalid) {
    float wsum = 0.0f, sum[3] = {0.0f, 0.0f, 0.0f};
    int center_valid = !isnan(center[0]) && !isnan(center[1]) && !isnan(center[2]);
    for (size_t k = 0; k < j->count; ++k) {
        const float *pixel = center + j->offsets[k];
        if (isnan(pixel[0]) || isnan(pixel[1]) || isnan(pixel[2])) continue;
        float alpha = center_valid ? difference_3(pixel, center) * j->scale : 0.0f;
        float w = j->weights[k] * vector_lookup(j, alpha, invalid);
        wsum += w;
        for (size_t c = 0; c < 3; ++c) sum[c] = multiply_add(pixel[c], w, sum[c], j->fused);
    }
    float reciprocal = 1.0f / (wsum + (center_valid ? 1.0f : 0.0f));
    for (size_t c = 0; c < 3; ++c)
        out[c] = (sum[c] + (center_valid ? center[c] : 0.0f)) * reciprocal;
}

static void scalar_3(const bilateral_context *j, const float *center, float *out,
                     int *invalid) {
    float wsum = 0.0f, sum[3] = {0.0f, 0.0f, 0.0f};
    int center_valid = !isnan(center[0]) && !isnan(center[1]) && !isnan(center[2]);
    for (size_t k = 0; k < j->count; ++k) {
        const float *pixel = center + j->offsets[k];
        if (isnan(pixel[0]) || isnan(pixel[1]) || isnan(pixel[2])) continue;
        float w = j->weights[k] * (center_valid
            ? scalar_lookup(j, difference_3(pixel, center) * j->scale, invalid) : 1.0f);
        wsum += w;
        for (size_t c = 0; c < 3; ++c) sum[c] += pixel[c] * w;
    }
    if (center_valid) {
        float reciprocal = 1.0f / (wsum + 1.0f);
        for (size_t c = 0; c < 3; ++c) out[c] = (sum[c] + center[c]) * reciprocal;
    } else {
        float reciprocal = 1.0f / wsum;
        for (size_t c = 0; c < 3; ++c) out[c] = sum[c] * reciprocal;
    }
}

/* Each row's first width/(2*lanes)*(2*lanes) pixels take the vector loop's
 * formula, the rest the scalar tail's (the AVX-512 build's 4- and 2-vector
 * blocks for one channel and 2-vector blocks for three). */
static void bilateral_range(void *opaque, size_t begin, size_t end) {
    const bilateral_context *j = opaque;
    for (size_t y = begin; y < end; ++y) {
        const float *row = j->padded + (y + j->radius) * j->stride
                         + j->radius * j->channels;
        float *out = j->out + y * j->width * j->channels;
        int invalid = 0;
        size_t x = 0;
        if (j->channels == 1) {
            for (; x < j->vector_end; ++x) out[x] = vector_1(j, row + x, &invalid);
            for (; x < j->width; ++x) out[x] = scalar_1(j, row + x, &invalid);
        } else {
            for (; x < j->vector_end; ++x) vector_3(j, row + 3 * x, out + 3 * x, &invalid);
            for (; x < j->width; ++x) scalar_3(j, row + 3 * x, out + 3 * x, &invalid);
        }
        j->invalid[y] = (unsigned char)invalid;
    }
}

CN_EXPORT cn_status cn_bilateral_f32(const float *src, float *dst,
    size_t height, size_t width, size_t channels, size_t radius,
    double sigma_color, double sigma_space, size_t lanes, size_t range_lanes,
    size_t row_planes) {
    uintptr_t a = (uintptr_t)src, b = (uintptr_t)dst;
    if (!src || !dst || !height || !width || height > INT_MAX || width > INT_MAX
        || (channels != 1 && channels != 3) || !radius || radius > 100
        || (lanes != 4 && lanes != 8 && lanes != 16)
        || (range_lanes != 4 && range_lanes != 8) || row_planes > 1
        || a % _Alignof(float) || b % _Alignof(float)
        || !isfinite(sigma_color) || !isfinite(sigma_space)
        || sigma_color <= 0 || sigma_space <= 0)
        return CN_INVALID_ARGUMENT;
    if (height > SIZE_MAX / width || height * width > SIZE_MAX / channels / sizeof(float))
        return CN_SIZE_OVERFLOW;
    size_t values = height * width * channels, bytes = values * sizeof(float);
    if (a > UINTPTR_MAX - bytes || b > UINTPTR_MAX - bytes) return CN_SIZE_OVERFLOW;
    if (a < b + bytes && b < a + bytes) return CN_INVALID_ARGUMENT;
    if (sigma_color <= 1e-6 || sigma_space <= 1e-6) {
        memcpy(dst, src, bytes);
        return CN_OK;
    }
    size_t planes = row_planes ? height : 1, plane = values / planes;
    float low = 0, high = 0;
    int started = 0, finite = 1;
    for (size_t p = 0; p < planes; ++p)
        range_plane(src + p * plane, plane, range_lanes, &started, &low, &high);
    if (fabs((double)low - (double)high) < FLT_EPSILON) {
        memcpy(dst, src, bytes);
        return CN_OK;
    }
    for (size_t i = 0; i < values && finite; ++i) finite = isfinite(src[i]);
    /* Finite values bound every lookup inside the table; otherwise the result
     * is staged so that an out-of-table index leaves the output untouched. */
    int staged = !finite || !isfinite(high - low);
    size_t padded_h = height + 2 * radius, padded_w = width + 2 * radius;
    if (padded_h > SIZE_MAX / padded_w
        || padded_h * padded_w > PTRDIFF_MAX / sizeof(float) / channels)
        return CN_SIZE_OVERFLOW;
    size_t stride = padded_w * channels, bins = 4096 * channels;
    size_t diameter = radius * 2 + 1, max_taps = diameter * diameter;
    float *padded = malloc(padded_h * stride * sizeof(float));
    float *weights = malloc(max_taps * sizeof(float));
    ptrdiff_t *offsets = malloc(max_taps * sizeof(ptrdiff_t));
    float *lut = malloc((bins + 2) * sizeof(float));
    unsigned char *invalid = malloc(height);
    float *out = staged ? malloc(bytes) : dst;
    if (!padded || !weights || !offsets || !lut || !invalid || !out) {
        free(padded); free(weights); free(offsets); free(lut); free(invalid);
        if (staged) free(out);
        return CN_ALLOCATION_FAILED;
    }
    float length = (float)((double)high - (double)low) * (float)channels;
    float scale = (float)bins / length, last = 1.0f;
    double color_coefficient = -0.5 / (sigma_color * sigma_color);
    double space_coefficient = -0.5 / (sigma_space * sigma_space);
    for (size_t i = 0; i < bins + 2; ++i) {
        if (last > 0) {
            /* The original expression divides int/float before promoting. */
            double value = (float)i / scale;
            last = (float)exp(value * value * color_coefficient);
        } else last = 0;
        lut[i] = last;
    }
    size_t count = 0;
    ptrdiff_t r = (ptrdiff_t)radius;
    for (ptrdiff_t y = -r; y <= r; ++y) {
        for (ptrdiff_t x = -r; x <= r; ++x) {
            double distance = sqrt((double)y * (double)y + (double)x * (double)x);
            if (distance > (double)radius || (x == 0 && y == 0)) continue;
            weights[count] = (float)exp(distance * distance * space_coefficient);
            offsets[count++] = y * (ptrdiff_t)stride + x * (ptrdiff_t)channels;
        }
    }
    for (size_t y = 0; y < padded_h; ++y) {
        const float *row = src + reflect101((ptrdiff_t)y - r, height) * width * channels;
        for (size_t x = 0; x < padded_w; ++x) {
            const float *pixel = row + reflect101((ptrdiff_t)x - r, width) * channels;
            for (size_t c = 0; c < channels; ++c)
                padded[y * stride + x * channels + c] = pixel[c];
        }
    }
    bilateral_context job = {padded, weights, lut, offsets, out, invalid, width, channels,
                             stride, radius, count, width / (2 * lanes) * (2 * lanes),
                             scale, (float)bins, (int)bins, lanes != 4};
    size_t work_per_row = width * count;
    size_t grain = work_per_row >= 32768 ? 1 : 32768 / work_per_row;
    cn_status result = cn_parallel_for(height, grain, bilateral_range, &job);
    for (size_t y = 0; result == CN_OK && y < height; ++y)
        if (invalid[y]) result = CN_BILATERAL_RANGE_ERROR;
    if (result == CN_OK && staged) memcpy(dst, out, bytes);
    free(padded); free(weights); free(offsets); free(lut); free(invalid);
    if (staged) free(out);
    return result;
}
