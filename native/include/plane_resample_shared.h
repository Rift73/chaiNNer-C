/* One axis of nodes/impl/plane_resample.py's resampler (cn_plane_resample_axis):
 * the job and the scalar rows, compiled into each including unit
 * (plane_resample_ops.c, the baseline entry, and the AVX2 unit
 * plane_resample_avx2.c, which runs them for a row's last columns). Every
 * including unit uses every helper (C4505).
 *
 * The taps (input indices, already mirrored and clamped, and float32 weights) come
 * precomputed from plane_resample.py's _taps, so the weights are its bits. Each
 * output sample keeps _resample_axis's NumPy sequence in float32: from +0, in tap
 * order, acc + (input - bias_in) * weight; then + bias_out when bias_out is not 0.
 * bias_in 0 leaves every input as it is (x - 0 is x, -0 included). */
#ifndef CHAINNER_PLANE_RESAMPLE_SHARED_H
#define CHAINNER_PLANE_RESAMPLE_SHARED_H
#include "chainner.h"
#include "isa.h"

/* src: rows of width float32 samples, row_stride elements apart. index and
 * weights: taps x dst_dim, tap-major (tap t of output i at t * dst_dim + i).
 * Horizontal: out is rows x dst_dim (one output row per source row); vertical:
 * dst_dim x width. out is C-contiguous. isa: the level the entry read once. */
typedef struct plane_resample_job {
    const float *src;
    float *out;
    const int32_t *index;
    const float *weights;
    size_t row_stride, width, dst_dim, taps;
    float bias_in, bias_out;
    int vertical;
    cn_isa_level isa;
} plane_resample_job;

/* Horizontal: output row y's columns [first, dst_dim). */
static void resample_row_horizontal(const plane_resample_job *job, size_t y, size_t first)
{
    const float *row = job->src + y * job->row_stride;
    float *out = job->out + y * job->dst_dim;
    for (size_t x = first; x < job->dst_dim; ++x) {
        float acc = 0.0f;
        for (size_t t = 0; t < job->taps; ++t) {
            size_t k = t * job->dst_dim + x;
            acc = acc + (row[job->index[k]] - job->bias_in) * job->weights[k];
        }
        out[x] = job->bias_out != 0.0f ? acc + job->bias_out : acc;
    }
}

/* Vertical: output row y's columns [first, width). */
static void resample_row_vertical(const plane_resample_job *job, size_t y, size_t first)
{
    float *out = job->out + y * job->width;
    for (size_t x = first; x < job->width; ++x) out[x] = 0.0f;
    for (size_t t = 0; t < job->taps; ++t) {
        size_t k = t * job->dst_dim + y;
        const float *row = job->src + (size_t)job->index[k] * job->row_stride;
        float weight = job->weights[k];
        for (size_t x = first; x < job->width; ++x)
            out[x] = out[x] + (row[x] - job->bias_in) * weight;
    }
    if (job->bias_out != 0.0f)
        for (size_t x = first; x < job->width; ++x) out[x] = out[x] + job->bias_out;
}

#if defined(_MSC_VER) && defined(_M_X64)
/* plane_resample_avx2.c: output rows [begin, end) for ISA level avx2 or above, 8
 * columns at a time with the scalar rows' operations per lane (horizontal: the
 * inputs by vgatherdps); each row's last columns run the scalar row. */
void cn_plane_resample_rows_avx2(const plane_resample_job *job, size_t begin, size_t end);
#endif
#endif
