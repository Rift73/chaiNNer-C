/* Descale: undo a resize with vapoursynth-descale's solver (third_party/descale, MIT), which
 * solves A'A x = A'b per row, then per column. The image is split into padded float planes
 * so the solver's AVX2 unit can use its aligned 8x8 loads, and both passes run on the pool:
 * the row pass in 16-row chunks (deinterleaving those rows first), the column pass in
 * 16-column chunks (interleaving those columns into the output after). Every vector is
 * solved independently, so the result does not depend on the chunking or the thread count.
 */

#include <malloc.h>
#include <stdlib.h>
#include <string.h>

#include "chainner.h"
#include "descale.h"
#include "isa.h"
#include "parallel.h"

#define DESCALE_ROWS 16
#define DESCALE_COLUMNS 16

static size_t round_up(size_t value, size_t multiple) { return (value + multiple - 1) / multiple * multiple; }

/* The solver's AVX2 unit needs FMA, which chaiNNer's AVX2 level includes (src/isa.c). */
static struct DescaleAPI descale_api(void) {
    return get_descale_api(cn_isa_current() >= CN_ISA_AVX2 ? DESCALE_OPT_AVX2 : DESCALE_OPT_NONE);
}

/* One direction's solver for a src_dim -> dst_dim descale; NULL on invalid parameters.
 * mode is DescaleMode (1 bilinear .. 7 point), border DescaleBorder. */
CN_EXPORT void *cn_descale_core_create(int src_dim, int dst_dim, int mode, int taps, double b, double c,
                                       double blur, double shift, double active_dim, int border) {
    if (src_dim < 1 || dst_dim < 1 || dst_dim > src_dim || mode < DESCALE_MODE_BILINEAR || mode > DESCALE_MODE_POINT ||
        !(blur > 0.0) || !(active_dim > 0.0) || border < DESCALE_BORDER_MIRROR || border > DESCALE_BORDER_REPEAT ||
        (mode == DESCALE_MODE_LANCZOS && taps < 1))
        return NULL;
    struct DescaleParams params = {0};
    params.mode = (enum DescaleMode)mode;
    params.taps = taps;
    params.param1 = b;
    params.param2 = c;
    params.blur = blur;
    params.shift = shift;
    params.active_dim = active_dim;
    params.border_handling = (enum DescaleBorder)border;
    return descale_api().create_core(src_dim, dst_dim, &params);
}

CN_EXPORT void cn_descale_core_free(void *core) {
    if (core) descale_api().free_core(core);
}

typedef struct descale_job {
    struct DescaleAPI api;
    struct DescaleCore *core_h, *core_v;
    const float *src;
    float *dst;
    size_t h, w, c, dh, dw;
    size_t rows;                         /* h rounded up to 8: the row pass works in groups of 8 */
    size_t in_stride, mid_stride;        /* floats; multiples of 16, so every row is 64-byte aligned */
    size_t in_plane, mid_plane, out_plane;
    float *in, *mid, *out;               /* channel k's plane starts at base + k * plane */
} descale_job;

static void descale_rows(void *context, size_t begin, size_t end) {
    const descale_job *job = context;
    for (size_t chunk = begin; chunk < end; ++chunk) {
        size_t y0 = chunk * DESCALE_ROWS, y1 = y0 + DESCALE_ROWS < job->rows ? y0 + DESCALE_ROWS : job->rows;
        for (size_t y = y0; y < y1; ++y) {
            for (size_t k = 0; k < job->c; ++k) {
                float *row = job->in + k * job->in_plane + y * job->in_stride;
                if (y < job->h) {
                    const float *pixel = job->src + y * job->w * job->c + k;
                    for (size_t x = 0; x < job->w; ++x) row[x] = pixel[x * job->c];
                    memset(row + job->w, 0, (job->in_stride - job->w) * sizeof(float));
                } else {
                    memset(row, 0, job->in_stride * sizeof(float));
                }
            }
        }
        if (job->core_h) {
            for (size_t k = 0; k < job->c; ++k)
                job->api.process_vectors(job->core_h, DESCALE_DIR_HORIZONTAL, (int)(y1 - y0), (int)job->in_stride, 0,
                                         (int)job->mid_stride, job->in + k * job->in_plane + y0 * job->in_stride, NULL,
                                         job->mid + k * job->mid_plane + y0 * job->mid_stride);
        }
    }
}

static void descale_columns(void *context, size_t begin, size_t end) {
    const descale_job *job = context;
    for (size_t chunk = begin; chunk < end; ++chunk) {
        size_t x0 = chunk * DESCALE_COLUMNS, x1 = x0 + DESCALE_COLUMNS < job->dw ? x0 + DESCALE_COLUMNS : job->dw;
        if (job->core_v) {
            for (size_t k = 0; k < job->c; ++k)
                job->api.process_vectors(job->core_v, DESCALE_DIR_VERTICAL, (int)(x1 - x0), (int)job->mid_stride, 0,
                                         (int)job->mid_stride, job->mid + k * job->mid_plane + x0, NULL,
                                         job->out + k * job->out_plane + x0);
        }
        for (size_t y = 0; y < job->dh; ++y) {
            float *pixel = job->dst + (y * job->dw + x0) * job->c;
            for (size_t k = 0; k < job->c; ++k) {
                const float *row = job->out + k * job->out_plane + y * job->mid_stride;
                for (size_t x = x0; x < x1; ++x) pixel[(x - x0) * job->c + k] = row[x];
            }
        }
    }
}

/* src is an h x w x c float32 image, dst dh x dw x c, where dw and dh are the cores' target
 * sizes (core_h NULL keeps the width, core_v NULL the height). */
CN_EXPORT cn_status cn_descale(const float *src, size_t h, size_t w, size_t c, void *core_h, void *core_v, float *dst) {
    if (!src || !dst || !h || !w || !c || h > (size_t)1 << 24 || w > (size_t)1 << 24 || c > 64) return CN_INVALID_ARGUMENT;
    descale_job job = {0};
    job.api = descale_api();
    job.core_h = core_h;
    job.core_v = core_v;
    if ((job.core_h && (size_t)job.core_h->src_dim != w) || (job.core_v && (size_t)job.core_v->src_dim != h))
        return CN_INVALID_ARGUMENT;
    job.src = src;
    job.dst = dst;
    job.h = h;
    job.w = w;
    job.c = c;
    job.dw = job.core_h ? (size_t)job.core_h->dst_dim : w;
    job.dh = job.core_v ? (size_t)job.core_v->dst_dim : h;
    job.rows = round_up(h, 8);
    job.in_stride = round_up(w, 16);
    job.mid_stride = round_up(job.dw, 16);
    job.in_plane = job.rows * job.in_stride;
    job.mid_plane = job.core_h ? job.rows * job.mid_stride : 0;
    job.out_plane = job.core_v ? job.dh * job.mid_stride : 0;
    size_t floats = c * (job.in_plane + job.mid_plane + job.out_plane);
    float *buffer = _aligned_malloc(floats * sizeof(float), 64);
    if (!buffer) return CN_ALLOCATION_FAILED;
    job.in = buffer;
    job.mid = job.core_h ? job.in + c * job.in_plane : job.in;
    if (!job.core_h) job.mid_plane = job.in_plane;
    job.out = job.core_v ? job.mid + c * job.mid_plane : job.mid;
    if (!job.core_v) job.out_plane = job.mid_plane;

    cn_status status = cn_parallel_for(job.rows / DESCALE_ROWS + (job.rows % DESCALE_ROWS != 0), 1, descale_rows, &job);
    if (status == CN_OK)
        status = cn_parallel_for(round_up(job.dw, DESCALE_COLUMNS) / DESCALE_COLUMNS, 1, descale_columns, &job);
    _aligned_free(buffer);
    return status;
}
