#if !defined(_M_X64)
#error "convolution_avx512.c is an x86-64 /arch:AVX512 unit"
#endif
/* The spatial convolution's AVX-512 unit (/arch:AVX512, spec 4.3, D15): no export,
 * reached only through convolution_range's dispatch on the call's ISA level (no
 * OpenCV source code incorporated). convolution_avx2.c's layout in groups of 16 at
 * absolute columns = 0 (mod 16). The fused mirror (fused_lanes 8, the checked
 * bridge's AVX2/FMA OpenCV dispatch) fixes each output's sequence by its column
 * alone: below fused_columns the taps in order, each one vfmadd; from fused_columns
 * on convolution_scalar's unfused sum. The vector outputs of a row segment [first,
 * last) are [first, end), end = min(last, fused_columns); the groups cover them from
 * first rounded down to end rounded up, and a group reaching past either end (a
 * chunk's first group; the group holding end, which straddles the fused boundary or
 * the chunk's end) loads and stores only its lanes inside [first, end) through a lane
 * mask. The outputs from end on, the straddling group's other lanes among them, run
 * convolution_scalar. */
#include "border_runs_shared.h"
#include "convolution_isa_shared.h"
#include "convolution_shared.h"
#include "group_masks_shared.h"
#include <immintrin.h>

/* A border run is widened into the interior to eight groups (when its copies then fit
 * CONVOLUTION_PATCH_FLOATS). */
#define CONVOLUTION_RUN_BLOCK 128

/* Zeros for the eight groups of a padding row's tap. */
static const float padding_row[128] = {0.0f};

/* Groups [j, end) of a row (multiples of 16) into out: tap k of the group at j is
 * one load at bases[k] + j, or zeros (padding_row) where bases[k] is NULL, a padding
 * row. Eight groups at a time keep eight independent sums (a 4-cycle FMA at 2 per
 * cycle needs 8 in flight); a tail of four or more groups keeps four, then one group
 * at a time. */
static void tap_groups(const convolution_job *job, const float *const *bases, size_t j, size_t end, float *out)
{
    size_t k, taps = job->tap_count;
    for (; end - j >= 128; j += 128) {
        __m512 sum0 = _mm512_setzero_ps(), sum1 = sum0, sum2 = sum0, sum3 = sum0;
        __m512 sum4 = sum0, sum5 = sum0, sum6 = sum0, sum7 = sum0;
        for (k = 0; k < taps; ++k) {
            __m512 coefficient = _mm512_set1_ps(job->taps[k].coefficient);
            const float *values = bases[k] ? bases[k] + j : padding_row;
            sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(values), coefficient, sum0);
            sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(values + 16), coefficient, sum1);
            sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(values + 32), coefficient, sum2);
            sum3 = _mm512_fmadd_ps(_mm512_loadu_ps(values + 48), coefficient, sum3);
            sum4 = _mm512_fmadd_ps(_mm512_loadu_ps(values + 64), coefficient, sum4);
            sum5 = _mm512_fmadd_ps(_mm512_loadu_ps(values + 80), coefficient, sum5);
            sum6 = _mm512_fmadd_ps(_mm512_loadu_ps(values + 96), coefficient, sum6);
            sum7 = _mm512_fmadd_ps(_mm512_loadu_ps(values + 112), coefficient, sum7);
        }
        _mm512_storeu_ps(out + j, sum0);
        _mm512_storeu_ps(out + j + 16, sum1);
        _mm512_storeu_ps(out + j + 32, sum2);
        _mm512_storeu_ps(out + j + 48, sum3);
        _mm512_storeu_ps(out + j + 64, sum4);
        _mm512_storeu_ps(out + j + 80, sum5);
        _mm512_storeu_ps(out + j + 96, sum6);
        _mm512_storeu_ps(out + j + 112, sum7);
    }
    if (end - j >= 64) {
        __m512 sum0 = _mm512_setzero_ps(), sum1 = sum0, sum2 = sum0, sum3 = sum0;
        for (k = 0; k < taps; ++k) {
            __m512 coefficient = _mm512_set1_ps(job->taps[k].coefficient);
            const float *values = bases[k] ? bases[k] + j : padding_row;
            sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(values), coefficient, sum0);
            sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(values + 16), coefficient, sum1);
            sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(values + 32), coefficient, sum2);
            sum3 = _mm512_fmadd_ps(_mm512_loadu_ps(values + 48), coefficient, sum3);
        }
        _mm512_storeu_ps(out + j, sum0);
        _mm512_storeu_ps(out + j + 16, sum1);
        _mm512_storeu_ps(out + j + 32, sum2);
        _mm512_storeu_ps(out + j + 48, sum3);
        j += 64;
    }
    for (; j < end; j += 16) {
        __m512 sum = _mm512_setzero_ps();
        for (k = 0; k < taps; ++k)
            sum = _mm512_fmadd_ps(_mm512_loadu_ps(bases[k] ? bases[k] + j : padding_row),
                _mm512_set1_ps(job->taps[k].coefficient), sum);
        _mm512_storeu_ps(out + j, sum);
    }
}

/* The group at j (as tap_groups), only the lanes in mask loaded and stored. */
static void tap_group_masked(const convolution_job *job, const float *const *bases, size_t j, __mmask16 mask,
    float *out)
{
    size_t k;
    __m512 sum = _mm512_setzero_ps();
    for (k = 0; k < job->tap_count; ++k)
        sum = _mm512_fmadd_ps(bases[k] ? _mm512_maskz_loadu_ps(mask, bases[k] + j) : _mm512_setzero_ps(),
            _mm512_set1_ps(job->taps[k].coefficient), sum);
    _mm512_mask_storeu_ps(out + j, mask, sum);
}

/* Groups [column, stop) of a row (16-lane grid) into out, column's output: tap k of
 * the group at column + j reads bases[k] + j, and the lanes outside [first, end) are
 * masked off (group_masks). */
static void tap_span(const convolution_job *job, const float *const *bases, size_t column, size_t stop,
    size_t first, size_t end, float *out)
{
    __mmask16 head, tail;
    size_t whole, whole_end;
    group_masks(column, stop, first, end, &head, &tail, &whole, &whole_end);
    if (head) tap_group_masked(job, bases, 0, head, out);
    tap_groups(job, bases, whole, whole_end, out);
    if (tail) tap_group_masked(job, bases, whole_end, tail, out);
}

/* Groups [column, stop) of row y whose every lane's pixel lies in the interior
 * columns [left, right), in any row, the lanes outside [first, end) masked off: tap
 * k reads src from its row through the table (once per segment; padding: zeros) at
 * the tap's column offset. */
static void interior_column_groups(const convolution_job *job, size_t y, size_t column, size_t stop,
    size_t first, size_t end)
{
    const float *bases[129];
    ptrdiff_t line = (ptrdiff_t)(job->w * job->c), shift = (ptrdiff_t)column - (ptrdiff_t)(job->padding * job->c);
    size_t k;
    if (column >= stop) return;
    for (k = 0; k < job->tap_count; ++k) {
        const convolution_tap *tap = job->taps + k;
        int32_t row = job->rows[(ptrdiff_t)y + (ptrdiff_t)tap->y];
        bases[k] = row < 0 ? NULL : job->src + (row * line + (ptrdiff_t)tap->x * (ptrdiff_t)job->c + shift);
    }
    tap_span(job, bases, column, stop, first, end, job->out + y * job->ow * job->c + column);
}

/* The group at (y, column), some lane's pixel outside [left, right), only the lanes
 * in mask loaded and stored: per tap, the row through the table (padding: zeros),
 * then one load when all 16 lanes' pixels map into the row without reflection, else
 * the lanes in mask through the column table (-1: zero). */
static void border_group(const convolution_job *job, size_t y, size_t column, __mmask16 mask)
{
    size_t c = job->c, pixel[16], channel[16], x = column / c, part = column % c, lane, k;
    __m512 sum = _mm512_setzero_ps();
    for (lane = 0; lane < 16; ++lane) {
        pixel[lane] = x;
        channel[lane] = part;
        if (++part == c) {
            part = 0;
            ++x;
        }
    }
    for (k = 0; k < job->tap_count; ++k) {
        const convolution_tap *tap = job->taps + k;
        int32_t row = job->rows[(ptrdiff_t)y + (ptrdiff_t)tap->y];
        __m512 values = _mm512_setzero_ps();
        if (row >= 0) {
            const float *line = job->src + (size_t)row * job->w * c;
            int64_t first = (int64_t)pixel[0] + tap->x - (int64_t)job->padding;
            int64_t last = (int64_t)pixel[15] + tap->x - (int64_t)job->padding;
            if (first >= 0 && last < (int64_t)job->w) {
                values = _mm512_maskz_loadu_ps(mask, line + (size_t)first * c + channel[0]);
            } else {
                float lanes[16];
                for (lane = 0; lane < 16; ++lane) {
                    int32_t source = mask >> lane & 1 ? job->columns[(ptrdiff_t)pixel[lane] + (ptrdiff_t)tap->x] : -1;
                    lanes[lane] = source < 0 ? 0.0f : line[(size_t)source * c + channel[lane]];
                }
                values = _mm512_loadu_ps(lanes);
            }
        }
        sum = _mm512_fmadd_ps(values, _mm512_set1_ps(tap->coefficient), sum);
    }
    _mm512_mask_storeu_ps(job->out + y * job->ow * c + column, mask, sum);
}

/* Groups [column, stop) of row y, some lane's pixel possibly outside [left, right)
 * (a border run), the lanes outside [first, end) masked off: each tap row's virtual
 * elements [column + low c, min(stop, end) + high c) (low and high from tap_extent)
 * are copied once into patch (patch_row; a padding row is not copied, its taps read
 * zeros), and tap_span runs on the copies. A run whose copies exceed
 * CONVOLUTION_PATCH_FLOATS takes border_group per group. */
static void border_run(const convolution_job *job, size_t y, size_t column, size_t stop, size_t first,
    size_t end, int64_t low, int64_t high, size_t rows)
{
    float patch[CONVOLUTION_PATCH_FLOATS];
    const float *bases[129], *current = patch;
    size_t c = job->c, width, used = 0, k;
    if (column >= stop) return;
    width = (end < stop ? end : stop) - column + (size_t)(high - low) * c;
    if (rows * width > CONVOLUTION_PATCH_FLOATS) {
        for (; column < stop; column += 16)
            border_group(job, y, column,
                lane_mask(first > column ? first - column : 0, end < column + 16 ? end - column : 16));
        return;
    }
    for (k = 0; k < job->tap_count; ++k) {
        const convolution_tap *tap = job->taps + k;
        int32_t row = job->rows[(ptrdiff_t)y + (ptrdiff_t)tap->y];
        if (row >= 0 && (k == 0 || tap->y != tap[-1].y)) {
            current = patch + used;
            patch_row(job, row, column, low, width, patch + used);
            used += width;
        }
        bases[k] = row < 0 ? NULL : current + (size_t)(tap->x - low) * c;
    }
    tap_span(job, bases, column, stop, first, end, job->out + y * job->ow * c + column);
}

void cn_convolution_row_avx512(const convolution_job *job, size_t y, size_t first, size_t last)
{
    size_t end = last < job->fused_columns ? last : job->fused_columns;
    size_t column = first / 16 * 16, stop = (end + 15) / 16 * 16, head, tail, rows = 0;
    size_t inner = job->left * job->c, outer = job->right * job->c;
    int64_t low = 0, high = 0;
    if (first >= end) {
        convolution_scalar(job, y, first, last);
        return;
    }
    /* Interior groups: column >= left * c and column + 16 <= right * c; the border
     * runs [column, head) and [tail, stop) hold the others, each widened into the
     * interior to a block of eight groups where the segment has one and the widened
     * run's copies fit the patch (else not widened). */
    border_runs(column, stop, inner, outer, 16, CONVOLUTION_RUN_BLOCK, &head, &tail);
    if (head > column || stop > tail) {
        rows = tap_extent(job, &low, &high);
        if (rows * (CONVOLUTION_RUN_BLOCK + (size_t)(high - low) * job->c) > CONVOLUTION_PATCH_FLOATS)
            border_runs(column, stop, inner, outer, 16, 16, &head, &tail);
    }
    border_run(job, y, column, head, first, end, low, high, rows);
    interior_column_groups(job, y, head, tail, first, end);
    border_run(job, y, tail, stop, first, end, low, high, rows);
    convolution_scalar(job, y, end, last);
}
