/* Spatial convolution, moved from convolution_ops.c (an independent
 * implementation; no OpenCV source code has been incorporated). Shared by that
 * baseline unit and the ISA units convolution_avx2.c (/arch:AVX2) and
 * convolution_avx512.c (/arch:AVX512): the job, and static helpers compiled into
 * each including unit, never inline, so the linker cannot fold a VEX copy into
 * baseline callers. Every including unit uses every helper (C4505). */
#ifndef CHAINNER_CONVOLUTION_SHARED_H
#define CHAINNER_CONVOLUTION_SHARED_H
#include "chainner.h"
#include "fused_shared.h"
#include "isa.h"

typedef struct convolution_tap {
    int64_t y, x;
    float coefficient;
} convolution_tap;

/* isa: the level the C entry read once for the call (cn_isa_current).
 * rows[v] and columns[v], for the padded-frame row (pixel column) v of the
 * output frame, reached by some tap from [-kh/2, oh + kh/2) ([-kw/2, ow + kw/2)):
 * reflect101(v, oh) - padding when that lies in [0, h) (the same over ow and w),
 * else -1, a padding zero. The interior is output rows [top, bottom) and pixel
 * columns [left, right), where every tap maps into the image without reflection
 * (both empty when none does), and tap k of an interior output reads
 * src[base + tap_offsets[k]], tap_offsets[k] = (tap.y * w + tap.x) * c. */
typedef struct convolution_job {
    const float *src;
    float *out;
    size_t w, ow, c, padding, tap_count, fused_columns;
    convolution_tap taps[129];
    cn_isa_level isa;
    const int32_t *rows, *columns;
    ptrdiff_t tap_offsets[129];
    size_t top, bottom, left, right;
} convolution_job;

/* Output (y, column) through the tables. Each output keeps B3's operation
 * sequence: the taps in order, fused below fused_columns, else value *
 * coefficient added; a tap whose row or column is -1 reads 0. */
static float convolution_output(const convolution_job *job, size_t y, size_t column)
{
    size_t x = column / job->c, channel = column % job->c, k;
    int fused = column < job->fused_columns;
    float sum = 0;
    for (k = 0; k < job->tap_count; ++k) {
        const convolution_tap *tap = job->taps + k;
        int32_t row = job->rows[(ptrdiff_t)y + (ptrdiff_t)tap->y];
        int32_t pixel = job->columns[(ptrdiff_t)x + (ptrdiff_t)tap->x];
        float value = 0;
        if (row >= 0 && pixel >= 0)
            value = job->src[((size_t)row * job->w + (size_t)pixel) * job->c + channel];
        if (fused) sum = fused_tap(value, tap->coefficient, sum);
        else sum += value * tap->coefficient;
    }
    return sum;
}

/* Outputs [first, last) of row y, one at a time: the interior's through
 * tap_offsets (the index is a ptrdiff_t before the pointer is formed), the others
 * through convolution_output; the same sequence either way. */
static void convolution_scalar(const convolution_job *job, size_t y, size_t first, size_t last)
{
    float *out = job->out + y * job->ow * job->c;
    size_t inner = first, outer = first, column = first, k;
    ptrdiff_t row = ((ptrdiff_t)y - (ptrdiff_t)job->padding) * (ptrdiff_t)(job->w * job->c) -
        (ptrdiff_t)(job->padding * job->c);
    if (y >= job->top && y < job->bottom) {
        size_t left = job->left * job->c, right = job->right * job->c;
        inner = left < first ? first : left < last ? left : last;
        outer = right < inner ? inner : right < last ? right : last;
    }
    for (; column < inner; ++column) out[column] = convolution_output(job, y, column);
    for (; column < outer; ++column) {
        ptrdiff_t base = row + (ptrdiff_t)column;
        float sum = 0;
        if (column < job->fused_columns)
            for (k = 0; k < job->tap_count; ++k)
                sum = fused_tap(job->src[base + job->tap_offsets[k]], job->taps[k].coefficient, sum);
        else
            for (k = 0; k < job->tap_count; ++k)
                sum += job->src[base + job->tap_offsets[k]] * job->taps[k].coefficient;
        out[column] = sum;
    }
    for (; column < last; ++column) out[column] = convolution_output(job, y, column);
}

/* convolution_avx2.c: outputs [first, last) of row y, inside one chunk, for ISA
 * level avx2 and fused_columns > 0: groups of 8 at columns = 0 (mod 8)
 * below fused_columns, every other output through convolution_scalar. */
void cn_convolution_row_avx2(const convolution_job *job, size_t y, size_t first, size_t last);
/* convolution_avx512.c: the same for ISA level avx512, in groups of 16 at columns =
 * 0 (mod 16) over the outputs of [first, last) below fused_columns; a group reaching
 * past them loads and stores only its lanes inside (a lane mask). */
void cn_convolution_row_avx512(const convolution_job *job, size_t y, size_t first, size_t last);
#endif
