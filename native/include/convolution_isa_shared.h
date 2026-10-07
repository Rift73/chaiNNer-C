/* Helpers of the spatial convolution's ISA units (convolution_avx2.c,
 * convolution_avx512.c; an independent implementation, no OpenCV source code
 * incorporated): the border runs' copies, static helpers compiled into each including
 * unit, never inline. Every including unit uses every helper (C4505). */
#ifndef CHAINNER_CONVOLUTION_ISA_SHARED_H
#define CHAINNER_CONVOLUTION_ISA_SHARED_H
#include "convolution_shared.h"
#include <string.h>

/* The most floats a border run's copies hold (16 KiB of the run function's stack); a
 * larger run takes its unit's per-group path. */
#define CONVOLUTION_PATCH_FLOATS 4096

/* patch[i] = image row `row`'s value at the output frame's virtual element
 * column + low c + i, i < width: pixel X's channel reads image pixel columns[X]
 * (one copy where that is X - padding, inside the image), or 0 where the column
 * table holds -1. */
static void patch_row(const convolution_job *job, int32_t row, size_t column, int64_t low, size_t width,
    float *patch)
{
    size_t c = job->c, part = column % c, lo, hi, i;
    const float *line = job->src + (size_t)row * job->w * c;
    int64_t start = (int64_t)column + low * (int64_t)c, x = (int64_t)(column / c) + low;
    int64_t first = (int64_t)(job->padding * c), last = (int64_t)((job->padding + job->w) * c);
    lo = first <= start ? 0 : first - start < (int64_t)width ? (size_t)(first - start) : width;
    hi = last <= start + (int64_t)lo ? lo : last - start < (int64_t)width ? (size_t)(last - start) : width;
    for (i = 0; i < lo; ++i) {
        int32_t source = job->columns[x];
        patch[i] = source < 0 ? 0.0f : line[(size_t)source * c + part];
        if (++part == c) {
            part = 0;
            ++x;
        }
    }
    if (hi > lo) memcpy(patch + lo, line + (size_t)(start + (int64_t)lo - first), (hi - lo) * sizeof(float));
    if (hi == width) return;
    x = (start + (int64_t)hi) / (int64_t)c;
    part = (size_t)((start + (int64_t)hi) % (int64_t)c);
    for (i = hi; i < width; ++i) {
        int32_t source = job->columns[x];
        patch[i] = source < 0 ? 0.0f : line[(size_t)source * c + part];
        if (++part == c) {
            part = 0;
            ++x;
        }
    }
}

/* The taps' smallest and largest x, and how many rows they span (taps are in
 * kernel order, row by row). */
static size_t tap_extent(const convolution_job *job, int64_t *low, int64_t *high)
{
    size_t k, rows = 0;
    *low = *high = 0;
    for (k = 0; k < job->tap_count; ++k) {
        const convolution_tap *tap = job->taps + k;
        if (k == 0 || tap->x < *low) *low = tap->x;
        if (k == 0 || tap->x > *high) *high = tap->x;
        if (k == 0 || tap->y != tap[-1].y) ++rows;
    }
    return rows;
}
#endif
