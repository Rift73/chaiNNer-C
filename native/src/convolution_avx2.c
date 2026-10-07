#if !defined(_M_X64)
#error "convolution_avx2.c is an x86-64 /arch:AVX2 unit"
#endif
/* The spatial convolution's AVX2 unit (/arch:AVX2, spec 4.3): no export, reached
 * only through convolution_range's dispatch on the call's ISA level (no OpenCV
 * source code incorporated). Called only for the matching AVX2/FMA OpenCV CPU
 * dispatch, selected by the checked bridge (fused_lanes 8): every lane of a group
 * below fused_columns runs its output's sequence, the taps in order, each one
 * vfmadd; lower ISA levels and other hosts run the same sequence through
 * convolution_scalar. */
#include "border_runs_shared.h"
#include "convolution_isa_shared.h"
#include "convolution_shared.h"
#include <immintrin.h>

/* Zeros for the eight groups of a padding row's tap. */
static const float padding_row[64] = {0.0f};

/* count outputs (a multiple of 8) of a row into out: tap k of the group at j is
 * one load at bases[k] + j, or zeros (padding_row) where bases[k] is NULL, a
 * padding row. Eight groups at a time keep eight independent sums (a 4-cycle FMA
 * at 2 per cycle needs 8 in flight); a tail of four or more groups keeps four,
 * then one group at a time. */
static void tap_groups(const convolution_job *job, const float *const *bases, size_t count, float *out)
{
    size_t k, taps = job->tap_count, j = 0;
    for (; count - j >= 64; j += 64) {
        __m256 sum0 = _mm256_setzero_ps(), sum1 = sum0, sum2 = sum0, sum3 = sum0;
        __m256 sum4 = sum0, sum5 = sum0, sum6 = sum0, sum7 = sum0;
        for (k = 0; k < taps; ++k) {
            __m256 coefficient = _mm256_set1_ps(job->taps[k].coefficient);
            const float *values = bases[k] ? bases[k] + j : padding_row;
            sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(values), coefficient, sum0);
            sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(values + 8), coefficient, sum1);
            sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(values + 16), coefficient, sum2);
            sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(values + 24), coefficient, sum3);
            sum4 = _mm256_fmadd_ps(_mm256_loadu_ps(values + 32), coefficient, sum4);
            sum5 = _mm256_fmadd_ps(_mm256_loadu_ps(values + 40), coefficient, sum5);
            sum6 = _mm256_fmadd_ps(_mm256_loadu_ps(values + 48), coefficient, sum6);
            sum7 = _mm256_fmadd_ps(_mm256_loadu_ps(values + 56), coefficient, sum7);
        }
        _mm256_storeu_ps(out + j, sum0);
        _mm256_storeu_ps(out + j + 8, sum1);
        _mm256_storeu_ps(out + j + 16, sum2);
        _mm256_storeu_ps(out + j + 24, sum3);
        _mm256_storeu_ps(out + j + 32, sum4);
        _mm256_storeu_ps(out + j + 40, sum5);
        _mm256_storeu_ps(out + j + 48, sum6);
        _mm256_storeu_ps(out + j + 56, sum7);
    }
    if (count - j >= 32) {
        __m256 sum0 = _mm256_setzero_ps(), sum1 = sum0, sum2 = sum0, sum3 = sum0;
        for (k = 0; k < taps; ++k) {
            __m256 coefficient = _mm256_set1_ps(job->taps[k].coefficient);
            const float *values = bases[k] ? bases[k] + j : padding_row;
            sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(values), coefficient, sum0);
            sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(values + 8), coefficient, sum1);
            sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(values + 16), coefficient, sum2);
            sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(values + 24), coefficient, sum3);
        }
        _mm256_storeu_ps(out + j, sum0);
        _mm256_storeu_ps(out + j + 8, sum1);
        _mm256_storeu_ps(out + j + 16, sum2);
        _mm256_storeu_ps(out + j + 24, sum3);
        j += 32;
    }
    for (; j < count; j += 8) {
        __m256 sum = _mm256_setzero_ps();
        for (k = 0; k < taps; ++k)
            sum = _mm256_fmadd_ps(_mm256_loadu_ps(bases[k] ? bases[k] + j : padding_row),
                _mm256_set1_ps(job->taps[k].coefficient), sum);
        _mm256_storeu_ps(out + j, sum);
    }
}

/* Groups [column, stop) of row y whose every lane's pixel lies in the interior
 * columns [left, right), in any row: tap k reads src from its row through the
 * table (once per segment; padding: zeros) at the tap's column offset. */
static void interior_column_groups(const convolution_job *job, size_t y, size_t column, size_t stop)
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
    tap_groups(job, bases, stop - column, job->out + y * job->ow * job->c + column);
}

/* The group at (y, column), some lane's pixel outside [left, right): per tap, the
 * row through the table (padding: zeros), then one load when every lane's pixel
 * maps into the row without reflection, else the 8 lanes through the column
 * table (-1: zero). */
static void border_group(const convolution_job *job, size_t y, size_t column)
{
    size_t c = job->c, pixel[8], channel[8], x = column / c, part = column % c, lane, k;
    __m256 sum = _mm256_setzero_ps();
    for (lane = 0; lane < 8; ++lane) {
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
        __m256 values = _mm256_setzero_ps();
        if (row >= 0) {
            const float *line = job->src + (size_t)row * job->w * c;
            int64_t first = (int64_t)pixel[0] + tap->x - (int64_t)job->padding;
            int64_t last = (int64_t)pixel[7] + tap->x - (int64_t)job->padding;
            if (first >= 0 && last < (int64_t)job->w) {
                values = _mm256_loadu_ps(line + (size_t)first * c + channel[0]);
            } else {
                float lanes[8];
                for (lane = 0; lane < 8; ++lane) {
                    int32_t source = job->columns[(ptrdiff_t)pixel[lane] + (ptrdiff_t)tap->x];
                    lanes[lane] = source < 0 ? 0.0f : line[(size_t)source * c + channel[lane]];
                }
                values = _mm256_loadu_ps(lanes);
            }
        }
        sum = _mm256_fmadd_ps(values, _mm256_set1_ps(tap->coefficient), sum);
    }
    _mm256_storeu_ps(job->out + y * job->ow * c + column, sum);
}

/* Groups [column, stop) of row y, some lane's pixel possibly outside [left, right)
 * (a border run): each tap row's virtual elements [column + low c, stop + high c)
 * (low and high from tap_extent) are copied once into patch (patch_row; a padding
 * row is not copied, its taps read zeros), and tap_groups runs on the copies. A
 * run whose copies exceed CONVOLUTION_PATCH_FLOATS takes border_group per group. */
static void border_run(const convolution_job *job, size_t y, size_t column, size_t stop, int64_t low,
    int64_t high, size_t rows)
{
    float patch[CONVOLUTION_PATCH_FLOATS];
    const float *bases[129], *current = patch;
    size_t c = job->c, width = stop - column + (size_t)(high - low) * c, used = 0, k;
    if (column >= stop) return;
    if (rows * width > CONVOLUTION_PATCH_FLOATS) {
        for (; column < stop; column += 8) border_group(job, y, column);
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
    tap_groups(job, bases, stop - column, job->out + y * job->ow * c + column);
}

void cn_convolution_row_avx2(const convolution_job *job, size_t y, size_t first, size_t last)
{
    size_t column = (first + 7) / 8 * 8;
    size_t stop = (last < job->fused_columns ? last : job->fused_columns) / 8 * 8;
    size_t head, tail, rows = 0;
    int64_t low = 0, high = 0;
    if (column >= stop) {
        convolution_scalar(job, y, first, last);
        return;
    }
    /* Interior groups: column >= left * c and column + 8 <= right * c; the border
     * runs [column, head) and [tail, stop) hold the others, each widened into the
     * interior to a block of eight groups where the segment has one. */
    border_runs(column, stop, job->left * job->c, job->right * job->c, 8, 64, &head, &tail);
    if (head > column || stop > tail) rows = tap_extent(job, &low, &high);
    convolution_scalar(job, y, first, column);
    border_run(job, y, column, head, low, high, rows);
    interior_column_groups(job, y, head, tail);
    border_run(job, y, tail, stop, low, high, rows);
    convolution_scalar(job, y, stop, last);
}
