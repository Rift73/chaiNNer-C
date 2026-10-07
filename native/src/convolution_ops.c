#include "chainner.h"
#include "convolution_shared.h"
#include "parallel.h"
#include <limits.h>
#include <stdlib.h>

/* Independent implementations of box running sums and spatial convolution.
   Explicit double sums / float fma sequences reproduce OpenCV 5.0.0's CPU
   contracts; no OpenCV source code has been incorporated here. */
static cn_status dimensions(size_t h, size_t w, size_t c) {
    if (!h || !w || !c || h > INT_MAX || w > INT_MAX)
        return CN_INVALID_ARGUMENT;
    if (h > SIZE_MAX / w || h * w > SIZE_MAX / sizeof(double) / c)
        return CN_SIZE_OVERFLOW;
    return CN_OK;
}

static size_t reflect101(int64_t coordinate, size_t length) {
    if (length == 1) return 0;
    int64_t period = (int64_t)(length - 1) * 2;
    coordinate %= period;
    if (coordinate < 0) coordinate += period;
    return (size_t)(coordinate < (int64_t)length ? coordinate : period - coordinate);
}

typedef struct box_job {
    const float *src;
    float *out;
    double *rows;
    size_t h, w, c, rx, ry;
    double scale;
    /* OpenCV 5.0.0's boxFilter runs float kernels up to 5x5 through BlockSum
       (box_filter.simd.hpp), larger ones through the row/column FilterEngine. */
    int block;
} box_job;

static void box_rows(void *context, size_t begin, size_t end) {
    box_job *job = (box_job *)context;
    size_t stride = job->w * job->c;
    for (size_t y = begin; y < end; ++y) {
        const float *src = job->src + y * stride;
        double *out = job->rows + y * stride;
        for (size_t c = 0; c < job->c; ++c) {
            /* OpenCV evaluates three/five-tap horizontal sums directly, and
               BlockSum every row sum, one-tap rows included. This matters when
               cancellation spans more than double's precision, and for
               nonfinite one-tap rows. BlockSum's interior 3/5-tap sums start at
               the first tap, not +0; its column sums are never -0, so a zero
               row sum's sign cannot reach the output. */
            if (job->block || job->rx == 1 || job->rx == 2) {
                for (size_t x = 0; x < job->w; ++x) {
                    double sum = 0;
                    for (int64_t dx = -(int64_t)job->rx; dx <= (int64_t)job->rx; ++dx)
                        sum += src[reflect101((int64_t)x + dx, job->w) * job->c + c];
                    out[x * job->c + c] = sum;
                }
                continue;
            }
            double sum = 0;
            for (int64_t x = -(int64_t)job->rx; x <= (int64_t)job->rx; ++x)
                sum += src[reflect101(x, job->w) * job->c + c];
            out[c] = sum;
            for (size_t x = 1; x < job->w; ++x) {
                size_t old = reflect101((int64_t)x - (int64_t)job->rx - 1, job->w);
                size_t next = reflect101((int64_t)x + (int64_t)job->rx, job->w);
                sum += (double)src[next * job->c + c] - (double)src[old * job->c + c];
                out[x * job->c + c] = sum;
            }
        }
    }
}

#define BOX_BAND 128
static void box_columns(void *context, size_t begin, size_t end) {
    box_job *job = (box_job *)context;
    size_t stride = job->w * job->c;
    for (size_t band = begin; band < end; ++band) {
        size_t start = band * BOX_BAND;
        size_t count = stride - start < BOX_BAND ? stride - start : BOX_BAND;
        double sums[BOX_BAND] = {0};
        for (int64_t y = -(int64_t)job->ry; y < (int64_t)job->ry; ++y) {
            const double *row = job->rows + reflect101(y, job->h) * stride + start;
            for (size_t x = 0; x < count; ++x) sums[x] += row[x];
        }
        for (size_t y = 0; y < job->h; ++y) {
            /* With a one-row kernel OpenCV's sumCount stays zero, so the
               FilterEngine resets its accumulator at each four-row batch.
               Preserve that NaN/Inf reset without changing finite sums.
               BlockSum keeps its column sums: no reset. */
            if (job->ry == 0 && !job->block && y % 4 == 0)
                for (size_t x = 0; x < count; ++x) sums[x] = 0;
            size_t next = reflect101((int64_t)y + (int64_t)job->ry, job->h);
            size_t old = reflect101((int64_t)y - (int64_t)job->ry, job->h);
            const double *enter = job->rows + next * stride + start;
            const double *leave = job->rows + old * stride + start;
            float *out = job->out + y * stride + start;
            for (size_t x = 0; x < count; ++x) {
                double total = sums[x] + enter[x];
                out[x] = (float)(total * job->scale);
                sums[x] = total - leave[x];
            }
        }
    }
}

CN_EXPORT cn_status cn_convolution_box(const float *src, float *out,
        size_t h, size_t w, size_t c, size_t rx, size_t ry) {
    if (!src || !out || rx > 1000 || ry > 1000) return CN_INVALID_ARGUMENT;
    cn_status status = dimensions(h, w, c);
    if (status != CN_OK) return status;
    size_t count = h * w * c;
    double *rows = malloc(count * sizeof(double));
    if (!rows) return CN_ALLOCATION_FAILED;
    box_job job = {src, out, rows, h, w, c, rx, ry,
        1.0 / (double)((2 * rx + 1) * (2 * ry + 1)), rx <= 2 && ry <= 2};
    status = cn_parallel_for(h, count < 65536 ? h : 4, box_rows, &job);
    if (status == CN_OK) {
        size_t bands = (w * c + BOX_BAND - 1) / BOX_BAND;
        status = cn_parallel_for(bands, count < 65536 ? bands : 1, box_columns, &job);
    }
    free(rows);
    return status;
}

/* [begin, end) row segment by row segment, so no vector group crosses a chunk:
 * with the fused mirror each segment goes to the AVX-512 row function at avx512,
 * to the AVX2 one at avx2, otherwise to convolution_scalar. */
static void convolution_range(void *context, size_t begin, size_t end) {
    const convolution_job *job = (const convolution_job *)context;
    size_t row_size = job->ow * job->c;
    for (size_t y = begin / row_size, first = begin % row_size; begin < end; ++y, first = 0) {
        size_t last = end - begin < row_size - first ? first + (end - begin) : row_size;
        begin += last - first;
#if defined(_MSC_VER) && defined(_M_X64)
        if (job->isa >= CN_ISA_AVX512 && job->fused_columns > 0) {
            cn_convolution_row_avx512(job, y, first, last);
            continue;
        }
        if (job->isa >= CN_ISA_AVX2 && job->fused_columns > 0) {
            cn_convolution_row_avx2(job, y, first, last);
            continue;
        }
#endif
        convolution_scalar(job, y, first, last);
    }
}

/* One axis's reflected-offset table (D8): entry i is padded-frame position
 * i - reach, mapped to reflect101(i - reach, length) - padding when that lies
 * in [0, size), else -1. */
static void reflected_offsets(int32_t *table, size_t length, size_t size, size_t padding, size_t reach) {
    for (size_t i = 0; i < length + 2 * reach; ++i) {
        size_t position = reflect101((int64_t)i - (int64_t)reach, length);
        table[i] = position >= padding && position - padding < size ? (int32_t)(position - padding) : -1;
    }
}

CN_EXPORT cn_status cn_convolution_spatial(const float *src, float *out,
        size_t h, size_t w, size_t c, const float *kernel,
        size_t kh, size_t kw, size_t padding, size_t fused_lanes) {
    if (!src || !out || !kernel || !kh || !kw || kh > 129 || kw > 129 ||
        kh * kw >= 130 || (fused_lanes != 0 && fused_lanes != 8))
        return CN_INVALID_ARGUMENT;
    cn_status status = dimensions(h, w, c);
    if (status != CN_OK) return status;
    if (padding > ((size_t)INT_MAX - h) / 2 || padding > ((size_t)INT_MAX - w) / 2)
        return CN_SIZE_OVERFLOW;
    size_t oh = h + padding * 2, ow = w + padding * 2;
    status = dimensions(oh, ow, c);
    if (status != CN_OK) return status;
    convolution_job job = {0};
    job.src = src; job.out = out; job.w = w; job.c = c;
    job.ow = ow; job.padding = padding;
    job.isa = cn_isa_current();
    job.fused_columns = fused_lanes ? ow * c / fused_lanes * fused_lanes : 0;
    for (size_t y = 0; y < kh; ++y) {
        for (size_t x = 0; x < kw; ++x) {
            float value = kernel[y * kw + x];
            if (value == 0) continue;
            convolution_tap *tap = job.taps + job.tap_count++;
            tap->y = (int64_t)y - (int64_t)(kh / 2);
            tap->x = (int64_t)x - (int64_t)(kw / 2);
            tap->coefficient = value;
        }
    }
    int64_t top = 0, bottom = (int64_t)oh, left = 0, right = (int64_t)ow;
    for (size_t k = 0; k < job.tap_count; ++k) {
        const convolution_tap *tap = job.taps + k;
        if ((int64_t)padding - tap->y > top) top = (int64_t)padding - tap->y;
        if ((int64_t)(padding + h) - tap->y < bottom) bottom = (int64_t)(padding + h) - tap->y;
        if ((int64_t)padding - tap->x > left) left = (int64_t)padding - tap->x;
        if ((int64_t)(padding + w) - tap->x < right) right = (int64_t)(padding + w) - tap->x;
        job.tap_offsets[k] = (ptrdiff_t)((tap->y * (int64_t)w + tap->x) * (int64_t)c);
    }
    if (bottom <= top) top = bottom = 0;
    if (right <= left) left = right = 0;
    job.top = (size_t)top; job.bottom = (size_t)bottom;
    job.left = (size_t)left; job.right = (size_t)right;
    size_t rows = oh + kh / 2 * 2, columns = ow + kw / 2 * 2;
    int32_t *tables = malloc((rows + columns) * sizeof(int32_t));
    if (!tables) return CN_ALLOCATION_FAILED;
    reflected_offsets(tables, oh, h, padding, kh / 2);
    reflected_offsets(tables + rows, ow, w, padding, kw / 2);
    job.rows = tables + kh / 2;
    job.columns = tables + rows + kw / 2;
    status = cn_parallel_for(oh * ow * c, 16384, convolution_range, &job);
    free(tables);
    return status;
}
