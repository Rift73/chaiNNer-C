/* C extraction of the buffer and packed-spectrum portions of OpenCV 4.8.0
 * crossCorr (imgproc/src/templmatch.cpp) and mulSpectrums (core/src/dxt.cpp).
 * The actual DFT deliberately remains the public OpenCV primitive: its
 * installed Intel IPP implementation changes even final PNG pixels at ties.
 *
 * Copyright (C) 2000, Intel Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * The name of Intel Corporation may not be used to endorse or promote
 *    products derived from this software without specific prior written
 *    permission.
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE INTEL CORPORATION OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#include "chainner.h"
#include "isa.h"
#include "parallel.h"
#include "spectral_shared.h"

static cn_status dimensions(size_t h, size_t w, size_t c, size_t element)
{
    if (!h || !w || !c || c > 512) return CN_INVALID_ARGUMENT;
    if (h > INT32_MAX || w > INT32_MAX || h > SIZE_MAX / w ||
        h * w > SIZE_MAX / c || h * w * c > SIZE_MAX / element)
        return CN_SIZE_OVERFLOW;
    return CN_OK;
}

typedef struct kernel_context {
    const void *src;
    double *dst;
    size_t h, w, dw;
    int is_double;
} kernel_context;

static void kernel_range(void *opaque, size_t begin, size_t end)
{
    const kernel_context *ctx = opaque;
    for (size_t p = begin; p < end; ++p) {
        size_t y = p / ctx->dw, x = p % ctx->dw;
        float value = 0.0f;
        if (y < ctx->h && x < ctx->w) {
            size_t q = y * ctx->w + x;
            /* crossCorr converts a double kernel to the float32 image depth
             * before promoting it to the double DFT plane. */
            value = ctx->is_double ? (float)((const double *)ctx->src)[q]
                                   : ((const float *)ctx->src)[q];
        }
        ctx->dst[p] = (double)value;
    }
}

CN_EXPORT cn_status cn_spectral_kernel(const void *src, int is_double,
    double *dst, size_t h, size_t w, size_t dh, size_t dw)
{
    if (!src || !dst || (is_double != 0 && is_double != 1) || h > dh || w > dw)
        return CN_INVALID_ARGUMENT;
    cn_status status = dimensions(h, w, 1, is_double ? sizeof(double) : sizeof(float));
    if (status != CN_OK) return status;
    status = dimensions(dh, dw, 1, sizeof(double));
    if (status != CN_OK) return status;
    kernel_context ctx = {src, dst, h, w, dw, is_double};
    return cn_parallel_for(dh * dw, 65536, kernel_range, &ctx);
}

/* The period of border modes 2 (reflect), 3 (wrap) and 4 (reflect101) on a length
 * n > 1, and a position q in [0, period) folded into [0, n). */
static int64_t border_period(int64_t n, int border)
{
    return border == 3 ? n : border == 2 ? 2 * n : 2 * (n - 1);
}

static int64_t border_fold(int64_t q, int64_t n, int64_t period, int border)
{
    if (q < n) return q;
    return border == 2 ? period - 1 - q : period - q;
}

static int64_t border_index(int64_t p, size_t length, int border)
{
    int64_t n = (int64_t)length;
    if (p >= 0 && p < n) return p;
    if (border == 0) return -1;
    if (border == 1) return p < 0 ? 0 : n - 1;
    if (n == 1) return 0;
    int64_t period = border_period(n, border);
    p %= period;
    if (p < 0) p += period;
    return border_fold(p, n, period, border);
}

typedef struct tile_context {
    const float *src;
    double *dst;
    size_t h, w, c, padding, y, x, th, tw, kh, kw, dw, channel;
    int border;
} tile_context;

/* The plane value of padded-image column sx (border_index's) in a source row: the
 * image's element, or +0.0 in the zero frame and for a constant border's -1. */
static double tile_value(const tile_context *ctx, const float *row, int64_t sx)
{
    int64_t padding = (int64_t)ctx->padding;
    if (sx < padding || sx >= padding + (int64_t)ctx->w) return 0.0;
    return (double)row[(size_t)(sx - padding) * ctx->c];
}

/* count plane columns from padded-image column p on, all left of [0, ow) or all right
 * of it. A constant or replicated border, or a one-column padded image, has one source
 * column per side; the other borders take p's position in their period once, then step
 * it one column at a time, so no column divides. */
static void tile_edge(const tile_context *ctx, const float *row, double *out,
    size_t count, int64_t p, int64_t ow)
{
    if (ctx->border < 2 || ow == 1) {
        double value = tile_value(ctx, row, border_index(p, (size_t)ow, ctx->border));
        for (size_t i = 0; i < count; ++i) out[i] = value;
        return;
    }
    int64_t period = border_period(ow, ctx->border), q = p % period;
    if (q < 0) q += period;
    for (size_t i = 0; i < count; ++i) {
        out[i] = tile_value(ctx, row, border_fold(q, ow, period, ctx->border));
        if (++q == period) q = 0;
    }
}

/* The plane column of padded-image column p in a row whose column 0 is padded-image
 * column `left`, clamped to [0, columns]. */
static size_t tile_column(int64_t p, int64_t left, size_t columns)
{
    int64_t column = p - left;
    if (column <= 0) return 0;
    return (uint64_t)column < columns ? (size_t)column : columns;
}

/* [begin, end) row segment by row segment. Plane row y's columns hold, by padded-image
 * column, the left border edge (below 0), the zero frame ([0, padding)), the image's
 * channel ([padding, padding + w)), the frame up to ow and the right border edge, then
 * the plane's zero columns from tw + kw - 1 on. Its source row comes from one
 * border_index; a source row in the frame, and the plane's zero rows from th + kh - 1
 * on, are +0.0 throughout. */
static void tile_range(void *opaque, size_t begin, size_t end)
{
    const tile_context *ctx = opaque;
    int64_t padding = (int64_t)ctx->padding;
    int64_t oh = (int64_t)ctx->h + 2 * padding, ow = (int64_t)ctx->w + 2 * padding;
    int64_t top = (int64_t)ctx->y - (int64_t)(ctx->kh / 2);
    int64_t left = (int64_t)ctx->x - (int64_t)(ctx->kw / 2);
    size_t rows = ctx->th + ctx->kh - 1, columns = ctx->tw + ctx->kw - 1;
    size_t dw = ctx->dw, c = ctx->c;
    size_t frame = tile_column(0, left, columns), image = tile_column(padding, left, columns);
    size_t after = tile_column(padding + (int64_t)ctx->w, left, columns);
    size_t edge = tile_column(ow, left, columns);
    for (size_t y = begin / dw, first = begin % dw; begin < end; ++y, first = 0) {
        size_t last = end - begin < dw - first ? first + (end - begin) : dw, i = first;
        double *out = ctx->dst + (begin - first);
        const float *row = NULL;
        begin += last - first;
        if (y < rows) {
            int64_t sy = border_index(top + (int64_t)y, (size_t)oh, ctx->border);
            if (sy >= padding && sy < padding + (int64_t)ctx->h)
                row = ctx->src + (size_t)(sy - padding) * ctx->w * c + ctx->channel;
        }
        if (row) {
            size_t to = last < frame ? last : frame;
            if (i < to) {
                tile_edge(ctx, row, out + i, to - i, left + (int64_t)i, ow);
                i = to;
            }
            for (to = last < image ? last : image; i < to; ++i) out[i] = 0.0;
            to = last < after ? last : after;
            if (i < to) {
                const float *source = row + (size_t)(left + (int64_t)i - padding) * c;
                for (size_t k = 0; i < to; ++i, ++k) out[i] = (double)source[k * c];
            }
            for (to = last < edge ? last : edge; i < to; ++i) out[i] = 0.0;
            to = last < columns ? last : columns;
            if (i < to) {
                tile_edge(ctx, row, out + i, to - i, left + (int64_t)i, ow);
                i = to;
            }
        }
        for (; i < last; ++i) out[i] = 0.0;
    }
}

CN_EXPORT cn_status cn_spectral_tile(const float *src, double *dst,
    size_t h, size_t w, size_t c, size_t padding, size_t y, size_t x,
    size_t th, size_t tw, size_t kh, size_t kw, size_t dh, size_t dw,
    size_t channel, int border)
{
    if (!src || !dst || channel >= c || border < 0 || border > 4)
        return CN_INVALID_ARGUMENT;
    cn_status status = dimensions(h, w, c, sizeof(float));
    if (status != CN_OK) return status;
    status = dimensions(dh, dw, 1, sizeof(double));
    if (status != CN_OK) return status;
    if (padding > ((size_t)INT32_MAX - h) / 2 || padding > ((size_t)INT32_MAX - w) / 2)
        return CN_SIZE_OVERFLOW;
    size_t oh = h + 2 * padding, ow = w + 2 * padding;
    if (!th || !tw || !kh || !kw || kh > dh || kw > dw ||
        y >= oh || x >= ow || th > oh - y || tw > ow - x ||
        th > dh - kh + 1 || tw > dw - kw + 1) return CN_INVALID_ARGUMENT;
    tile_context ctx = {src, dst, h, w, c, padding, y, x, th, tw, kh, kw, dw, channel, border};
    return cn_parallel_for(dh * dw, 65536, tile_range, &ctx);
}

/* isa: the level cn_spectral_multiply read once for the call (cn_isa_current). */
typedef struct spectrum_context {
    double *a;
    const double *b;
    size_t h, w;
    cn_isa_level isa;
} spectrum_context;

/* Job 0 runs the packed columns 0 and (for an even w) w - 1, job y + 1 row y's pairs
 * (through the AVX2 unit at avx2); a, b, h and w are read once, so the stores into a
 * cannot make them reload, and ctx->isa is read per row job. */
static void spectrum_range(void *opaque, size_t begin, size_t end)
{
    const spectrum_context *ctx = opaque;
    double *a = ctx->a;
    const double *b = ctx->b;
    size_t h = ctx->h, w = ctx->w;
    for (size_t job = begin; job < end; ++job) {
        if (job == 0) {
            if (h == 1) {
                a[0] *= b[0];
                if (!(w & 1)) a[w - 1] *= b[w - 1];
            } else {
                size_t cols = (w & 1) ? 1 : 2;
                for (size_t col = 0; col < cols; ++col) {
                    size_t x = col == 0 ? 0 : w - 1;
                    a[x] *= b[x];
                    for (size_t y = 1; y + 1 < h; y += 2)
                        multiply_pair(a + y * w + x, b + y * w + x, w);
                    if (!(h & 1)) {
                        size_t p = (h - 1) * w + x;
                        a[p] *= b[p];
                    }
                }
            }
        } else {
            size_t end_x = w - ((w & 1) ? 0 : 1);
            double *row = a + (job - 1) * w;
            const double *other = b + (job - 1) * w;
#if defined(_MSC_VER) && defined(_M_X64)
            if (ctx->isa >= CN_ISA_AVX2) {
                cn_spectral_pairs_avx2(row, other, end_x);
                continue;
            }
#endif
            for (size_t x = 1; x < end_x; x += 2) multiply_pair(row + x, other + x, 1);
        }
    }
}

CN_EXPORT cn_status cn_spectral_multiply(double *a, const double *b, size_t h, size_t w)
{
    if (!a || !b || a == b) return CN_INVALID_ARGUMENT;
    cn_status status = dimensions(h, w, 1, sizeof(double));
    if (status != CN_OK) return status;
    /* A continuous single column has the same packed layout as one row. */
    if (w == 1) { w = h; h = 1; }
    spectrum_context ctx = {a, b, h, w, cn_isa_current()};
    return cn_parallel_for(h + 1, 65536 / w + 1, spectrum_range, &ctx);
}

typedef struct store_context {
    const double *src; float *dst;
    size_t ow, c, y, x, tw, dw, channel;
} store_context;

/* [begin, end) of the th x tw tile, row segment by row segment: plane row y's columns
 * narrowed into channel `channel` of result row ctx->y + y. */
static void store_range(void *opaque, size_t begin, size_t end)
{
    const store_context *ctx = opaque;
    size_t tw = ctx->tw, c = ctx->c;
    for (size_t y = begin / tw, first = begin % tw; begin < end; ++y, first = 0) {
        size_t last = end - begin < tw - first ? first + (end - begin) : tw;
        const double *source = ctx->src + y * ctx->dw;
        float *target = ctx->dst + ((ctx->y + y) * ctx->ow + ctx->x) * c + ctx->channel;
        begin += last - first;
        for (size_t x = first; x < last; ++x) target[x * c] = (float)source[x];
    }
}

CN_EXPORT cn_status cn_spectral_store(const double *src, float *dst,
    size_t oh, size_t ow, size_t c, size_t y, size_t x, size_t th, size_t tw,
    size_t dh, size_t dw, size_t channel)
{
    if (!src || !dst || channel >= c) return CN_INVALID_ARGUMENT;
    cn_status status = dimensions(oh, ow, c, sizeof(float));
    if (status != CN_OK) return status;
    status = dimensions(dh, dw, 1, sizeof(double));
    if (status != CN_OK) return status;
    if (!th || !tw || y >= oh || x >= ow || th > oh - y || tw > ow - x || th > dh || tw > dw)
        return CN_INVALID_ARGUMENT;
    store_context ctx = {src, dst, ow, c, y, x, tw, dw, channel};
    return cn_parallel_for(th * tw, 65536, store_range, &ctx);
}
