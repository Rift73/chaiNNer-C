/* C adaptation of NumPy 2.5.3 npysort/selection.cpp (introselect, NaN last by
 * numpy_tag.h's floating_point_type::less) and the float32 minimum/maximum reduce
 * of umath/loops_minmax.dispatch.c.src (npyv_reduce_maxn/minn, AVX2 high-half
 * ties). Original quickselect
 * is based on Nicolas Devillard's 1998 public-domain algorithm.
 * Copyright (c) 2005-2022, NumPy Developers. All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * * Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of the NumPy Developers nor the names of any contributors
 *   may be used to endorse or promote products derived from this software
 *   without specific prior written permission.
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#include "chainner.h"
#include "numeric.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef ptrdiff_t index_type;
enum { PIVOT_CAPACITY = 50 };

static void swap(float *a, float *b)
{
    float temporary = *a; *a = *b; *b = temporary;
}

static void store_pivot(index_type pivot, index_type kth, index_type *pivots, int *count)
{
    if (!pivots) return;
    if (pivot == kth && *count == PIVOT_CAPACITY) pivots[*count - 1] = pivot;
    else if (pivot >= kth && *count < PIVOT_CAPACITY) pivots[(*count)++] = pivot;
}

/* NumPy 2.5.3 numpy_tag.h floating_point_type::less: NaN sorts to the end, and
 * NaNs compare equal, so which NaN lands where follows the selection's swaps. */
static int less(float a, float b)
{
    return a < b || (b != b && a == a);
}

static void select_rank(float *v, index_type n, index_type kth, index_type *pivots, int *count);

static index_type median5(float *v)
{
    if (less(v[1], v[0])) swap(v + 1, v);
    if (less(v[4], v[3])) swap(v + 4, v + 3);
    if (less(v[3], v[0])) swap(v + 3, v);
    if (less(v[4], v[1])) swap(v + 4, v + 1);
    if (less(v[2], v[1])) swap(v + 2, v + 1);
    return less(v[3], v[2]) ? (less(v[3], v[1]) ? 1 : 3) : 2;
}

static index_type median_of_medians(float *v, index_type n)
{
    index_type medians = n / 5;
    for (index_type i = 0; i < medians; ++i) {
        index_type start = i * 5;
        swap(v + start + median5(v + start), v + i);
    }
    if (medians > 2) select_rank(v, medians, medians / 2, NULL, NULL);
    return medians / 2;
}

static void select_rank(float *v, index_type n, index_type kth, index_type *pivots, int *count)
{
    index_type low = 0, high = n - 1;
    while (pivots && *count > 0) {
        index_type pivot = pivots[*count - 1];
        if (pivot > kth) { high = pivot - 1; break; }
        if (pivot == kth) return;
        low = pivot + 1;
        --*count;
    }
    if (kth - low < 3) {
        for (index_type i = low; i <= kth; ++i) {
            index_type minimum = i;
            for (index_type j = i + 1; j <= high; ++j)
                if (less(v[j], v[minimum])) minimum = j;
            swap(v + i, v + minimum);
        }
        store_pivot(kth, kth, pivots, count);
        return;
    }
    if (kth == n - 1) {
        index_type maximum = low;
        for (index_type j = low + 1; j < n; ++j)
            if (!less(v[j], v[maximum])) maximum = j;
        swap(v + kth, v + maximum);
        return;
    }
    int depth = 0;
    for (index_type size = n; size > 1; size >>= 1) depth += 2;
    while (low + 1 < high) {
        index_type left = low + 1, right = high;
        if (depth > 0 || right - left < 5) {
            index_type middle = low + (high - low) / 2;
            if (less(v[high], v[middle])) swap(v + high, v + middle);
            if (less(v[high], v[low])) swap(v + high, v + low);
            if (less(v[low], v[middle])) swap(v + low, v + middle);
            swap(v + middle, v + low + 1);
        } else {
            index_type middle = left + median_of_medians(v + left, right - left);
            swap(v + middle, v + low);
            --left; ++right;
        }
        --depth;
        float pivot = v[low];
        for (;;) {
            do { ++left; } while (less(v[left], pivot));
            do { --right; } while (less(pivot, v[right]));
            if (right < left) break;
            swap(v + left, v + right);
        }
        swap(v + low, v + right);
        if (right != kth) store_pivot(right, kth, pivots, count);
        if (right >= kth) high = right - 1;
        if (right <= kth) low = left;
    }
    if (high == low + 1 && less(v[high], v[low])) swap(v + high, v + low);
    store_pivot(kth, kth, pivots, count);
}

/* NumPy 2.5.3 item_selection.c _new_sortlike: one introselect per kth, in
 * ascending order, sharing one pivot stack. */
void cn_numpy_partition_f32(float *v, size_t n, const size_t *kth, size_t count)
{
    index_type pivots[PIVOT_CAPACITY];
    int stored = 0;
    for (size_t i = 0; i < count; ++i)
        select_rank(v, (index_type)n, (index_type)kth[i], pivots, &stored);
}

/* npyv_maxn/minn_f32 and the SSE scalar_max/min_f of NumPy 2.5.3: a NaN first
 * operand wins, else max_ps/min_ps, which returns the second operand on ties
 * (either signed zero) and on a NaN second operand. */
static float extreme(float a, float b, int maximum)
{
    if (isnan(a)) return a;
    return (maximum ? a > b : a < b) ? a : b;
}

/* npyv_reduce_maxn/minn_f32: any NaN lane gives the canonical 0x7FC00000. The
 * SSE/AVX2 shuffle trees prefer the high half on ties. X86_V4 is not built by
 * MSVC wheels; its compiler reduce intrinsic kept the low half under 1.24. */
static float horizontal(float *accumulator, int lanes, int maximum)
{
    for (int j = 0; j < lanes; ++j) {
        if (isnan(accumulator[j])) return cn_f32_from_bits(CN_NPY_NANF_BITS);
    }
    for (int width = lanes; width > 1; width /= 2)
        for (int j = 0; j < width / 2; ++j)
            accumulator[j] = lanes == 16
                ? extreme(accumulator[j + width / 2], accumulator[j], maximum)
                : extreme(accumulator[j], accumulator[j + width / 2], maximum);
    return accumulator[0];
}

/* One FLOAT_maximum/minimum reduce call of NumPy 2.5.3 loops_minmax.dispatch
 * .c.src over a contiguous stream: simd_reduce_c with eight-vector unrolling,
 * single vectors, the horizontal reduce, then scalar leftovers. */
static float reduce_contiguous(float out, const float *x, size_t len, int lanes, int maximum)
{
    if (len < 1) return out;
    size_t vstep = (size_t)lanes, wstep = vstep * 8;
    float accumulator[16];
    for (int j = 0; j < lanes; ++j) accumulator[j] = out;
    for (; len >= wstep; len -= wstep, x += wstep)
        for (size_t j = 0; j < vstep; ++j) {
            float r01 = extreme(x[j], x[vstep + j], maximum);
            float r23 = extreme(x[2 * vstep + j], x[3 * vstep + j], maximum);
            float r45 = extreme(x[4 * vstep + j], x[5 * vstep + j], maximum);
            float r67 = extreme(x[6 * vstep + j], x[7 * vstep + j], maximum);
            accumulator[j] = extreme(accumulator[j], extreme(extreme(r01, r23, maximum),
                extreme(r45, r67, maximum), maximum), maximum);
        }
    for (; len >= vstep; len -= vstep, x += vstep)
        for (size_t j = 0; j < vstep; ++j)
            accumulator[j] = extreme(accumulator[j], x[j], maximum);
    float result = horizontal(accumulator, lanes, maximum);
    for (; len > 0; --len, ++x) result = extreme(result, *x, maximum);
    return result;
}

/* The same call over a strided stream (inner stride != 4 bytes): the scalar
 * eight-accumulator unroll and its pairwise combine. */
static float reduce_strided(float out, const float *x, size_t len, int maximum)
{
    size_t i = 0;
    if (len >= 8) {
        float m[8];
        for (size_t j = 0; j < 8; ++j) m[j] = x[j];
        for (i = 8; i + 8 <= len; i += 8)
            for (size_t j = 0; j < 8; ++j) m[j] = extreme(m[j], x[i + j], maximum);
        m[0] = extreme(m[0], m[1], maximum);
        m[2] = extreme(m[2], m[3], maximum);
        m[4] = extreme(m[4], m[5], maximum);
        m[6] = extreme(m[6], m[7], maximum);
        m[0] = extreme(m[0], m[2], maximum);
        m[4] = extreme(m[4], m[6], maximum);
        m[0] = extreme(m[0], m[4], maximum);
        out = extreme(out, m[0], maximum);
    }
    for (; i < len; ++i) out = extreme(out, x[i], maximum);
    return out;
}

float cn_numpy_extreme_f32(const float *src, size_t n, int lanes, int maximum, size_t block,
                           size_t period)
{
    /* umath/reduction.c copies src[0] into the result and the first inner-loop
     * call skips it; each call then continues from the preceding result. The
     * calls are cn_numpy_sum_f32's: blocks that restart at every period. */
    float result = src[0];
    if (!period || period > n) period = n;
    if (!block || block > period) block = period;
    for (size_t row = 0; row < n; row += period) {
        size_t row_end = n - row < period ? n : row + period;
        for (size_t begin = row; begin < row_end; begin += block) {
            size_t end = row_end - begin < block ? row_end : begin + block;
            size_t first = begin ? begin : 1;
            result = lanes == 1
                ? reduce_strided(result, src + first, end - first, maximum)
                : reduce_contiguous(result, src + first, end - first, lanes, maximum);
        }
    }
    return result;
}

/* Internal callers own aligned scratch, validate n>0 and q in [0,100], and
 * reset the scratch before each independent NumPy percentile query.
 * weak: NumPy 2.5.3 lib/_function_base_impl.py percentile sets
 * weak_q = type(q) in (int, float), and _quantile's gamma = float(gamma)
 * keeps the weight a Python float, so under NEP 50 _lerp runs in float32.
 * A NumPy float64 q keeps the float64 interpolation. */
double cn_numpy_percentile_f32(float *ordered, size_t n, double percentile, int weak)
{
    double at = (double)(n - 1) * (percentile / 100.0);
    index_type lower = (index_type)floor(at), upper = lower + 1;
    double fraction = at - (double)lower;
    if (at >= (double)(n - 1)) {
        lower = upper = (index_type)n - 1;
        fraction = at + 1.0;
    }
    /* _quantile partitions at np.unique([0, -1, prev, next]); these ranks keep
     * duplicates, which select the same: introselect returns at once for a repeated kth. */
    const size_t ranks[4] = {0, (size_t)lower, (size_t)upper, n - 1};
    cn_numpy_partition_f32(ordered, n, ranks, 4);
    /* _quantile: the partition always includes kth -1, so any NaN sorts last;
     * then, on either lerp path, the result is arr[-1], that NaN as stored. */
    if (isnan(ordered[n - 1])) return ordered[n - 1];
    float delta = ordered[upper] - ordered[lower];
    if (weak) {
        /* _lerp: a + d*t, or b - d*(1 - t) where t >= 0.5. The Python floats
         * t and 1 - t round to float32 before each float32 product. */
        if (fraction >= 0.5) return ordered[upper] - delta * (float)(1.0 - fraction);
        return ordered[lower] + delta * (float)fraction;
    }
    return fraction >= 0.5
        ? (double)ordered[upper] - (double)delta * (1.0 - fraction)
        : (double)ordered[lower] + (double)delta * fraction;
}

/* lanes, block and period describe np.min/np.max/np.mean's reduction of the
 * node's operand (cn_numpy_extreme_f32, cn_numpy_sum_f32); weak is
 * np.percentile's weak_q for the q object the node passes. */
CN_EXPORT cn_status cn_analysis_statistics(const float *src, const float *quantile_src,
    size_t n, double percentile, int lanes, size_t block, size_t period, int weak,
    double *result)
{
    if (!src || !quantile_src || !result || !n ||
        !(percentile >= 0 && percentile <= 100) ||
        (lanes != 1 && lanes != 4 && lanes != 8 && lanes != 16) || (weak != 0 && weak != 1) ||
        (uintptr_t)src % _Alignof(float) || (uintptr_t)quantile_src % _Alignof(float) ||
        (uintptr_t)result % _Alignof(double)) return CN_INVALID_ARGUMENT;
    if (n > SIZE_MAX / sizeof(float) || n > PTRDIFF_MAX) return CN_SIZE_OVERFLOW;
    size_t bytes = n * sizeof(float);
    uintptr_t a = (uintptr_t)src, b = (uintptr_t)quantile_src, out = (uintptr_t)result;
    if (a > UINTPTR_MAX - bytes || b > UINTPTR_MAX - bytes || out > UINTPTR_MAX - 4 * sizeof(double))
        return CN_SIZE_OVERFLOW;
    if ((a < out + 4 * sizeof(double) && out < a + bytes) ||
        (b < out + 4 * sizeof(double) && out < b + bytes)) return CN_INVALID_ARGUMENT;
    float *ordered = malloc(bytes);
    if (!ordered) return CN_ALLOCATION_FAILED;
    memcpy(ordered, quantile_src, bytes);
    float minimum = cn_numpy_extreme_f32(src, n, lanes, 0, block, period);
    float maximum = cn_numpy_extreme_f32(src, n, lanes, 1, block, period);
    result[0] = minimum; result[1] = maximum;
    result[2] = numpy_mean_f32(cn_numpy_sum_f32(src, n, 1, block, period), n);
    result[3] = cn_numpy_percentile_f32(ordered, n, percentile, weak);
    free(ordered);
    return CN_OK;
}
