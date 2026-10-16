/* Descale: undo a resize with vapoursynth-descale's solver (third_party/descale, MIT), which
 * solves A'A x = A'b per row, then per column, in the plugin's order. Both passes run on the
 * pool. The row pass deinterleaves a 16-row chunk into a padded local buffer and solves it
 * into the middle planes; the column pass solves a 32-column strip of the middle planes into
 * a local strip and interleaves that into the output. Buffers are 64-byte aligned with rows
 * a multiple of 16 floats, as the solver's aligned SIMD loads require. Every vector is
 * solved independently, so the result does not depend on the chunking or the thread count.
 *
 * The caller passes the workspace (cn_descale_workspace bytes): the bridge takes it from
 * NumPy, whose pool keeps it warm across a batch. Fresh pages cost more than the solve.
 */

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include "chainner.h"
#include "descale.h"
#include "isa.h"
#include "parallel.h"

#define DESCALE_ROWS 16
#define DESCALE_STRIP 32
#define DESCALE_MAX_SLOTS 128

static size_t round_up(size_t value, size_t multiple) { return (value + multiple - 1) / multiple * multiple; }

/* The vendored AVX2 unit needs FMA, which chaiNNer's AVX2 level includes (src/isa.c). */
static struct DescaleAPI descale_api(void) {
    return get_descale_api(cn_isa_current() >= CN_ISA_AVX2 ? DESCALE_OPT_AVX2 : DESCALE_OPT_NONE);
}

/* descale.c's round_halfup (from zimg). */
static double round_halfup(double x) { return x < 0 ? floor(x + 0.5) : floor(x + 0.49999999999999994); }

/* descale's mirror reflects a tap once, so a tap more than dst_dim beyond an edge (a kernel
 * wider than the target, or a shift past an edge) would index outside its weights row and
 * corrupt the heap. Replays scaling_weights' tap positions (descale.c) to refuse those. */
static int mirror_fits(int src_dim, int dst_dim, int mode, int taps, double blur, double shift, double active_dim) {
    static const int mode_taps[] = {0, 1, 2, 0, 2, 3, 4, 0};
    int support = (int)ceil((mode == DESCALE_MODE_LANCZOS ? taps : mode_taps[mode]) * blur);
    int filter_size = support > 0 ? 2 * support : 1;
    double ratio = (double)src_dim / active_dim;
    for (int i = 0; i < src_dim; ++i) {
        double pos = (i + 0.5) / ratio + shift;
        double first = round_halfup(pos - filter_size / 2.0) + 0.5, last = first + (filter_size - 1);
        if (-first >= dst_dim || 2.0 * dst_dim - last < 0.0) return 0;
    }
    return 1;
}

/* One direction's solver for a src_dim -> dst_dim descale; NULL on invalid parameters, and
 * then *mirror_outside is 1 when the mirror's reach (mirror_fits) was the reason, else 0.
 * mode is DescaleMode (1 bilinear .. 7 point), border DescaleBorder. */
CN_EXPORT void *cn_descale_core_create(int src_dim, int dst_dim, int mode, int taps, double b, double c,
                                       double blur, double shift, double active_dim, int border,
                                       int *mirror_outside) {
    if (mirror_outside) *mirror_outside = 0;
    /* descale indexes its src_dim x dst_dim matrices with int products. */
    if (src_dim < 1 || dst_dim < 1 || dst_dim > src_dim || (int64_t)src_dim * dst_dim > INT_MAX ||
        mode < DESCALE_MODE_BILINEAR || mode > DESCALE_MODE_POINT ||
        !(blur > 0.0) || !(active_dim > 0.0) || border < DESCALE_BORDER_MIRROR || border > DESCALE_BORDER_REPEAT ||
        (mode == DESCALE_MODE_LANCZOS && taps < 1))
        return NULL;
    if (border == DESCALE_BORDER_MIRROR && !mirror_fits(src_dim, dst_dim, mode, taps, blur, shift, active_dim)) {
        if (mirror_outside) *mirror_outside = 1;
        return NULL;
    }
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
    void (*process)(struct DescaleCore *core, enum DescaleDir dir, int vector_count, int src_stride, int imask_stride,
                    int dst_stride, const float *srcp, const unsigned char *imaskp, float *dstp);
    struct DescaleCore *core_h, *core_v;
    const float *src;
    float *dst;
    size_t h, w, c, dh, dw;
    size_t rows;           /* h rounded up to 8: the row pass solves groups of 8 rows */
    size_t in_stride;      /* local row buffer stride, floats */
    size_t mid_stride;     /* middle plane stride, floats */
    size_t mid_plane;      /* rows * mid_stride */
    size_t slot_floats;    /* one task's local buffer */
    size_t slots;
    float *mid, *local;
    volatile LONG64 used[DESCALE_MAX_SLOTS / 64];
} descale_job;

/* A free local buffer for one callback; at most CPU count + 1 callbacks run at once. */
static size_t claim_slot(descale_job *job) {
    for (;;) {
        for (size_t word = 0; word * 64 < job->slots; ++word) {
            LONG64 mask = job->used[word];
            size_t limit = job->slots - word * 64 < 64 ? job->slots - word * 64 : 64;
            for (size_t bit = 0; bit < limit; ++bit) {
                LONG64 flag = (LONG64)((uint64_t)1 << bit);
                if (!(mask & flag) && InterlockedCompareExchange64(&job->used[word], mask | flag, mask) == mask)
                    return word * 64 + bit;
                mask = job->used[word];
            }
        }
        YieldProcessor();
    }
}

static void release_slot(descale_job *job, size_t slot) {
    InterlockedAnd64(&job->used[slot / 64], ~(LONG64)((uint64_t)1 << (slot % 64)));
}

/* Rows y0..y1 of the source, channel by channel, into planes of the given stride. */
static void deinterleave_rows(const descale_job *job, size_t y0, size_t y1, float *planes, size_t plane, size_t stride,
                              size_t first_row) {
    for (size_t y = y0; y < y1; ++y) {
        float *row = planes + (y - first_row) * stride;
        if (y < job->h) {
            const float *pixel = job->src + y * job->w * job->c;
            if (job->c == 3) {
                float *r0 = row, *r1 = row + plane, *r2 = row + 2 * plane;
                for (size_t x = 0; x < job->w; ++x) {
                    r0[x] = pixel[3 * x];
                    r1[x] = pixel[3 * x + 1];
                    r2[x] = pixel[3 * x + 2];
                }
            } else {
                for (size_t x = 0; x < job->w; ++x)
                    for (size_t k = 0; k < job->c; ++k) row[k * plane + x] = pixel[x * job->c + k];
            }
            for (size_t k = 0; k < job->c; ++k) memset(row + k * plane + job->w, 0, (stride - job->w) * sizeof(float));
        } else {
            for (size_t k = 0; k < job->c; ++k) memset(row + k * plane, 0, stride * sizeof(float));
        }
    }
}

static void descale_rows(void *context, size_t begin, size_t end) {
    descale_job *job = context;
    size_t slot = job->core_h ? claim_slot(job) : 0;
    float *local = job->local + slot * job->slot_floats;
    for (size_t chunk = begin; chunk < end; ++chunk) {
        size_t y0 = chunk * DESCALE_ROWS, y1 = y0 + DESCALE_ROWS < job->rows ? y0 + DESCALE_ROWS : job->rows;
        if (!job->core_h) {
            deinterleave_rows(job, y0, y1, job->mid + y0 * job->mid_stride, job->mid_plane, job->mid_stride, y0);
            continue;
        }
        size_t plane = DESCALE_ROWS * job->in_stride;
        deinterleave_rows(job, y0, y1, local, plane, job->in_stride, y0);
        for (size_t k = 0; k < job->c; ++k)
            job->process(job->core_h, DESCALE_DIR_HORIZONTAL, (int)(y1 - y0), (int)job->in_stride, 0,
                         (int)job->mid_stride, local + k * plane, NULL, job->mid + k * job->mid_plane + y0 * job->mid_stride);
    }
    if (job->core_h) release_slot(job, slot);
}

static void descale_columns(void *context, size_t begin, size_t end) {
    descale_job *job = context;
    size_t slot = job->core_v ? claim_slot(job) : 0;
    float *local = job->local + slot * job->slot_floats;
    size_t plane = job->dh * DESCALE_STRIP;
    for (size_t chunk = begin; chunk < end; ++chunk) {
        size_t x0 = chunk * DESCALE_STRIP, x1 = x0 + DESCALE_STRIP < job->dw ? x0 + DESCALE_STRIP : job->dw;
        const float *planes = job->mid + x0;
        size_t plane_stride = job->mid_plane, row_stride = job->mid_stride;
        if (job->core_v) {
            for (size_t k = 0; k < job->c; ++k)
                job->process(job->core_v, DESCALE_DIR_VERTICAL, (int)(x1 - x0), (int)job->mid_stride, 0, DESCALE_STRIP,
                             job->mid + k * job->mid_plane + x0, NULL, local + k * plane);
            planes = local;
            plane_stride = plane;
            row_stride = DESCALE_STRIP;
        }
        for (size_t y = 0; y < job->dh; ++y) {
            float *pixel = job->dst + (y * job->dw + x0) * job->c;
            const float *row = planes + y * row_stride;
            if (job->c == 3) {
                const float *r0 = row, *r1 = row + plane_stride, *r2 = row + 2 * plane_stride;
                for (size_t x = 0; x < x1 - x0; ++x) {
                    pixel[3 * x] = r0[x];
                    pixel[3 * x + 1] = r1[x];
                    pixel[3 * x + 2] = r2[x];
                }
            } else {
                for (size_t x = 0; x < x1 - x0; ++x)
                    for (size_t k = 0; k < job->c; ++k) pixel[x * job->c + k] = row[k * plane_stride + x];
            }
        }
    }
    if (job->core_v) release_slot(job, slot);
}

/* Fills the layout; returns the workspace bytes (64 bytes of alignment slack included). */
static size_t layout(descale_job *job, size_t h, size_t w, size_t c, struct DescaleCore *core_h,
                     struct DescaleCore *core_v) {
    job->core_h = core_h;
    job->core_v = core_v;
    job->h = h;
    job->w = w;
    job->c = c;
    job->dw = core_h ? (size_t)core_h->dst_dim : w;
    job->dh = core_v ? (size_t)core_v->dst_dim : h;
    job->rows = round_up(h, 8);
    job->in_stride = round_up(w, 16);
    job->mid_stride = core_h ? round_up(job->dw, 16) : job->in_stride;
    job->mid_plane = job->rows * job->mid_stride;
    size_t row_local = core_h ? c * DESCALE_ROWS * job->in_stride : 0;
    size_t strip_local = core_v ? c * job->dh * DESCALE_STRIP : 0;
    job->slot_floats = round_up(row_local > strip_local ? row_local : strip_local, 16);
    size_t capacity = (size_t)cn_parallel_capacity() + 1;
    job->slots = capacity < DESCALE_MAX_SLOTS ? capacity : DESCALE_MAX_SLOTS;
    return (c * job->mid_plane + job->slots * job->slot_floats) * sizeof(float) + 64;
}

static int valid(size_t h, size_t w, size_t c, struct DescaleCore *core_h, struct DescaleCore *core_v) {
    return h && w && c && h <= (size_t)1 << 24 && w <= (size_t)1 << 24 && c <= 64 &&
           (!core_h || (size_t)core_h->src_dim == w) && (!core_v || (size_t)core_v->src_dim == h) &&
           /* descale's column pass indexes the h-row middle plane with int products. */
           (!core_v || h * round_up(core_h ? (size_t)core_h->dst_dim : w, 16) <= INT_MAX);
}

/* Workspace bytes cn_descale needs for these arguments; 0 if they are invalid. */
CN_EXPORT size_t cn_descale_workspace(size_t h, size_t w, size_t c, void *core_h, void *core_v) {
    descale_job job;
    return valid(h, w, c, core_h, core_v) ? layout(&job, h, w, c, core_h, core_v) : 0;
}

/* src is an h x w x c float32 image, dst dh x dw x c, where dw and dh are the cores' target
 * sizes (core_h NULL keeps the width, core_v NULL the height). */
CN_EXPORT cn_status cn_descale(const float *src, size_t h, size_t w, size_t c, void *core_h, void *core_v, float *dst,
                               void *workspace, size_t workspace_bytes) {
    if (!src || !dst || !workspace || !valid(h, w, c, core_h, core_v)) return CN_INVALID_ARGUMENT;
    descale_job job = {0};
    if (layout(&job, h, w, c, core_h, core_v) > workspace_bytes) return CN_INVALID_ARGUMENT;
    /* No AVX-512 unit: a lane-for-lane 16-wide port measured no end-to-end gain (memory bound;
     * its row pass was slower, its column pass within noise). */
    job.process = descale_api().process_vectors;
    job.src = src;
    job.dst = dst;
    job.mid = (float *)(((uintptr_t)workspace + 63) & ~(uintptr_t)63);
    job.local = job.mid + c * job.mid_plane;

    cn_status status = cn_parallel_for(job.rows / DESCALE_ROWS + (job.rows % DESCALE_ROWS != 0), 1, descale_rows, &job);
    if (status == CN_OK)
        status = cn_parallel_for(round_up(job.dw, DESCALE_STRIP) / DESCALE_STRIP, 1, descale_columns, &job);
    return status;
}
