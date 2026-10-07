/* Altered C adaptation of chaiNNer-rs crates/image-ops/src/esdt.rs at
 * 6f6ead6064f81b4049d3deb803c9279c7950736b. The exposed chaiNNer node selects
 * pre_process=false, post_process=true, cutoff=0.5 (cn_distance_esdf);
 * cn_distance_esdf_ex is chainner_ext 0.3.10's esdf with every variant.
 * Original: https://gitlab.com/unconed/use.gpu/-/tree/master/packages/glyph
 *
 * MIT License
 * Copyright (c) 2021-2022 Steven Wittens
 * Copyright (c) 2023 Michael Schmidt
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "chainner.h"
#include "chainner_ext_kernels.h"
#include "parallel.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define DISTANCE_INF 1e10f

typedef struct esdf_context {
    const float *source;
    float *out, *outer, *inner, *xo, *yo, *xi, *yi, *workspace;
    size_t *indices;
    size_t width, height, longest;
    float radius, cutoff;
    int pre_process, post_process;
} esdf_context;

static size_t bounded_index(int64_t position, size_t length)
{
    return position < 0 ? 0 : (uint64_t)position >= length ? length - 1 : (size_t)position;
}

static int sign_of(float value)
{
    return value > 0.0f ? 1 : value < 0.0f ? -1 : 0;
}

static int solid(float value)
{
    return value <= 0.0f || value >= 1.0f;
}

static int crossing(float nx, float ny, float dc, float dl, float dr,
    float dxl, float dyl, float dxr, float dyr)
{
    return (dxl * nx + dyl * ny) * (dc * dl) > 0.0f &&
        (dxr * nx + dyr * ny) * (dc * dr) > 0.0f &&
        (dxl * dxr + dyl * dyr) * (dl * dr) > 0.0f;
}

/* paint_subpixel_offsets' relax pass (pre_process): xo/yo in, xi/yi out. */
static void relax_normals(esdf_context *ctx)
{
    size_t w = ctx->width, h = ctx->height;
    const float *data = ctx->source, *xo = ctx->xo, *yo = ctx->yo;
    for (size_t y = 0; y < h; ++y) {
        size_t ym = y == 0 ? 0 : y - 1, yp = y == h - 1 ? h - 1 : y + 1;
        for (size_t x = 0; x < w; ++x) {
            size_t xm = x == 0 ? 0 : x - 1, xp = x == w - 1 ? w - 1 : x + 1;
            size_t j = y * w + x;
            float nx = xo[j], ny = yo[j];
            if (nx == 0.0f && ny == 0.0f) continue;
            float c = data[j], l = data[y * w + xm], r = data[y * w + xp];
            float t = data[ym * w + x], b = data[yp * w + x];
            float dxl = xo[y * w + xm], dxr = xo[y * w + xp];
            float dxt = xo[ym * w + x], dxb = xo[yp * w + x];
            float dyl = yo[y * w + xm], dyr = yo[y * w + xp];
            float dyt = yo[ym * w + x], dyb = yo[yp * w + x];
            float dx = nx, dy = ny;
            int dw = 1;
            float dc = c - 0.5f, dl = l - 0.5f, dr = r - 0.5f, dt = t - 0.5f, db = b - 0.5f;
            if (!solid(l) && !solid(r) && crossing(nx, ny, dc, dl, dr, dxl, dyl, dxr, dyr)) {
                dx += (dxl + dxr) / 2.0f; dy += (dyl + dyr) / 2.0f; ++dw;
            }
            if (!solid(t) && !solid(b) && crossing(nx, ny, dc, dt, db, dxt, dyt, dxb, dyb)) {
                dx += (dxt + dxb) / 2.0f; dy += (dyt + dyb) / 2.0f; ++dw;
            }
            if (!solid(l) && !solid(t) && crossing(nx, ny, dc, dl, dt, dxl, dyl, dxt, dyt)) {
                dx += (dxl + dxt - 1.0f) / 2.0f; dy += (dyl + dyt - 1.0f) / 2.0f; ++dw;
            }
            if (!solid(r) && !solid(t) && crossing(nx, ny, dc, dr, dt, dxr, dyr, dxt, dyt)) {
                dx += (dxr + dxt + 1.0f) / 2.0f; dy += (dyr + dyt - 1.0f) / 2.0f; ++dw;
            }
            if (!solid(l) && !solid(b) && crossing(nx, ny, dc, dl, db, dxl, dyl, dxb, dyb)) {
                dx += (dxl + dxb - 1.0f) / 2.0f; dy += (dyl + dyb + 1.0f) / 2.0f; ++dw;
            }
            if (!solid(r) && !solid(b) && crossing(nx, ny, dc, dr, db, dxr, dyr, dxb, dyb)) {
                dx += (dxr + dxb + 1.0f) / 2.0f; dy += (dyr + dyb + 1.0f) / 2.0f; ++dw;
            }
            float nn = hypotf(nx, ny);
            float ll = (dx * nx + dy * ny) / nn;
            ctx->xi[j] = nx * ll / (float)dw / nn;
            ctx->yi[j] = ny * ll / (float)dw / nn;
        }
    }
}

static void paint_offsets(esdf_context *ctx)
{
    size_t w = ctx->width, h = ctx->height, count = w * h;
    const float *data = ctx->source;
    for (size_t p = 0; p < count; ++p) {
        ctx->outer[p] = data[p] == 0.0f ? DISTANCE_INF : 0.0f;
        ctx->inner[p] = data[p] >= 1.0f ? DISTANCE_INF : 0.0f;
    }
    /* This pass writes neighboring boundary cells in original row order. */
    for (size_t y = 0; y < h; ++y) {
        size_t ym = y == 0 ? 0 : y - 1, yp = y == h - 1 ? h - 1 : y + 1;
        for (size_t x = 0; x < w; ++x) {
            size_t xm = x == 0 ? 0 : x - 1, xp = x == w - 1 ? w - 1 : x + 1;
            size_t j = y * w + x;
            float c = data[j];
            float l = data[y * w + xm], r = data[y * w + xp];
            float t = data[ym * w + x], b = data[yp * w + x];
            if (!solid(c)) {
                float tl = data[ym * w + xm], tr = data[ym * w + xp];
                float bl = data[yp * w + xm], br = data[yp * w + xp];
                float ll = (tl + l * 2.0f + bl) / 4.0f;
                float rr = (tr + r * 2.0f + br) / 4.0f;
                float tt = (tl + t * 2.0f + tr) / 4.0f;
                float bb = (bl + b * 2.0f + br) / 4.0f;
                float neighbors[8] = {l, r, t, b, tl, tr, bl, br};
                float minimum = l, maximum = l;
                for (size_t i = 1; i < 8; ++i) {
                    minimum = fminf(minimum, neighbors[i]);
                    maximum = fmaxf(maximum, neighbors[i]);
                }
                if (minimum > 0.0f) { ctx->inner[j] = DISTANCE_INF; continue; }
                if (maximum < 1.0f) { ctx->outer[j] = DISTANCE_INF; continue; }
                float dx = rr - ll, dy = bb - tt;
                float dl = 1.0f / hypotf(dx, dy);
                dx *= dl;
                dy *= dl;
                float dc = c - 0.5f;
                ctx->xo[j] = -dc * dx;
                ctx->yo[j] = -dc * dy;
            } else if (c >= 1.0f) {
                if (l <= 0.0f && x > 0) {
                    ctx->xo[j - 1] = 0.4999f;
                    ctx->outer[j - 1] = 0.0f;
                    ctx->inner[j - 1] = 0.0f;
                }
                if (r <= 0.0f && x < w - 1) {
                    ctx->xo[j + 1] = -0.4999f;
                    ctx->outer[j + 1] = 0.0f;
                    ctx->inner[j + 1] = 0.0f;
                }
                if (t <= 0.0f && y > 0) {
                    ctx->yo[j - w] = 0.4999f;
                    ctx->outer[j - w] = 0.0f;
                    ctx->inner[j - w] = 0.0f;
                }
                if (b <= 0.0f && y < h - 1) {
                    ctx->yo[j + w] = -0.4999f;
                    ctx->outer[j + w] = 0.0f;
                    ctx->inner[j + w] = 0.0f;
                }
            }
        }
    }
    if (ctx->pre_process) relax_normals(ctx);
    for (size_t y = 0; y < h; ++y) {
        for (size_t x = 0; x < w; ++x) {
            size_t j = y * w + x;
            float nx = ctx->pre_process ? ctx->xi[j] : ctx->xo[j];
            float ny = ctx->pre_process ? ctx->yi[j] : ctx->yo[j];
            if (nx == 0.0f && ny == 0.0f) continue;
            float nn = hypotf(nx, ny);
            int sx = fabsf(nx / nn) > 0.5f ? sign_of(nx) : 0;
            int sy = fabsf(ny / nn) > 0.5f ? sign_of(ny) : 0;
            size_t xx = bounded_index((int64_t)x + sx, w);
            size_t yy = bounded_index((int64_t)y + sy, h);
            float s = (float)sign_of(data[yy * w + xx] - data[j]);
            float dlo = nn + 0.4999f * s, dli = nn - 0.4999f * s;
            dli /= nn;
            dlo /= nn;
            ctx->xo[j] = nx * dlo;
            ctx->yo[j] = ny * dlo;
            ctx->xi[j] = nx * dli;
            ctx->yi[j] = ny * dli;
        }
    }
}

static void transform_1d(float *mask, float *xs, float *ys,
    size_t offset, size_t stride, size_t length, float *f, float *z,
    float *b, float *t, size_t *v)
{
    v[0] = 0;
    b[0] = xs[offset];
    t[0] = ys[offset];
    z[0] = -DISTANCE_INF;
    z[1] = DISTANCE_INF;
    f[0] = mask[offset] != 0.0f ? DISTANCE_INF : ys[offset] * ys[offset];
    ptrdiff_t k = 0;
    for (size_t q = 1; q < length; ++q) {
        size_t o = offset + q * stride;
        float dx = xs[o], dy = ys[o];
        float fq = mask[o] != 0.0f ? DISTANCE_INF : dy * dy;
        f[q] = fq;
        t[q] = dy;
        float qs = (float)q + dx, q2 = qs * qs;
        b[q] = qs;
        float s;
        for (;;) {
            size_t r = v[k];
            float rs = b[r], r2 = rs * rs;
            s = (fq - f[r] + q2 - r2) / (qs - rs) / 2.0f;
            if (!(s <= z[k])) break;
            if (--k < 0) break;
        }
        ++k;
        v[k] = q;
        z[k] = s;
        z[k + 1] = DISTANCE_INF;
    }
    k = 0;
    for (size_t q = 0; q < length; ++q) {
        while (z[k + 1] < (float)q) ++k;
        size_t r = v[k], o = offset + q * stride;
        xs[o] = b[r] - (float)q;
        ys[o] = t[r];
        if (r != q) mask[o] = 0.0f;
    }
}

static int64_t rust_round_index(float value)
{
    /* Rust saturates float-to-integer casts and maps NaN to zero. */
    if (isnan(value)) return 0;
    if (value >= 9223372036854775808.0f) return INT64_MAX;
    if (value <= -9223372036854775808.0f) return INT64_MIN;
    return (int64_t)roundf(value);
}

static int64_t wrapping_step(int64_t value, int step)
{
    if (step == 1 && value == INT64_MAX) return INT64_MIN;
    if (step == -1 && value == INT64_MIN) return INT64_MAX;
    return value + step;
}

static float check_target(float *xs, float *ys, size_t w, size_t h,
    int64_t x, int64_t y, float dx, float dy, float distance, size_t j)
{
    size_t k = bounded_index(y, h) * w + bounded_index(x, w);
    float dx2 = dx + xs[k], dy2 = dy + ys[k];
    float d2 = hypotf(dx2, dy2);
    if (d2 < distance) {
        xs[j] = dx2;
        ys[j] = dy2;
        return d2;
    }
    return distance;
}

static void relax_offsets(float *xs, float *ys, size_t w, size_t h)
{
    for (size_t y = 0; y < h; ++y) {
        for (size_t x = 0; x < w; ++x) {
            size_t j = y * w + x;
            float dx = xs[j], dy = ys[j];
            if (dx == 0.0f && dy == 0.0f) continue;
            float distance = hypotf(dx, dy);
            float ds = (distance - 0.5f) / distance;
            int64_t ix = rust_round_index((float)x + dx * ds);
            int64_t iy = rust_round_index((float)y + dy * ds);
            dx = (float)ix - (float)x;
            dy = (float)iy - (float)y;
            distance = check_target(xs, ys, w, h, wrapping_step(ix, 1), iy, dx + 1.0f, dy, distance, j);
            distance = check_target(xs, ys, w, h, wrapping_step(ix, -1), iy, dx - 1.0f, dy, distance, j);
            distance = check_target(xs, ys, w, h, ix, wrapping_step(iy, 1), dx, dy + 1.0f, distance, j);
            (void)check_target(xs, ys, w, h, ix, wrapping_step(iy, -1), dx, dy - 1.0f, distance, j);
        }
    }
}

static void distance_fields(void *opaque, size_t begin, size_t end)
{
    esdf_context *ctx = opaque;
    for (size_t field = begin; field < end; ++field) {
        float *mask = field == 0 ? ctx->outer : ctx->inner;
        float *xs = field == 0 ? ctx->xo : ctx->xi;
        float *ys = field == 0 ? ctx->yo : ctx->yi;
        float *f = ctx->workspace + field * (ctx->longest * 4 + 1);
        float *z = f + ctx->longest, *b = z + ctx->longest + 1, *t = b + ctx->longest;
        size_t *v = ctx->indices + field * ctx->longest;
        for (size_t x = 0; x < ctx->width; ++x)
            transform_1d(mask, ys, xs, x, ctx->width, ctx->height, f, z, b, t, v);
        for (size_t y = 0; y < ctx->height; ++y)
            transform_1d(mask, xs, ys, y * ctx->width, 1, ctx->width, f, z, b, t, v);
        if (ctx->post_process) relax_offsets(xs, ys, ctx->width, ctx->height);
    }
}

static void distance_output(void *opaque, size_t begin, size_t end)
{
    const esdf_context *ctx = opaque;
    for (size_t i = begin; i < end; ++i) {
        float outer = fmaxf(0.0f, hypotf(ctx->xo[i], ctx->yo[i]) - 0.5f);
        float inner = fmaxf(0.0f, hypotf(ctx->xi[i], ctx->yi[i]) - 0.5f);
        float d = outer >= inner ? outer : -inner;
        if (!ctx->pre_process && !solid(ctx->source[i])) d = 0.5f - ctx->source[i];
        float result = 1.0f - (d / ctx->radius + ctx->cutoff);
        if (result < 0.0f) result = 0.0f;
        if (result > 1.0f) result = 1.0f;
        ctx->out[i] = result;
    }
}

CN_EXPORT cn_status cn_distance_esdf_ex(const float *source, float *out,
    size_t height, size_t width, float radius, float cutoff, int pre_process, int post_process)
{
    if (!source || !out || !height || !width) return CN_INVALID_ARGUMENT;
    if (height > INT32_MAX || width > INT32_MAX ||
        width > SIZE_MAX / 6 / sizeof(float) / height) return CN_SIZE_OVERFLOW;
    size_t count = height * width, longest = width > height ? width : height;
    if (longest > (SIZE_MAX / sizeof(float) / 2 - 1) / 4 ||
        longest > SIZE_MAX / sizeof(size_t) / 2) return CN_SIZE_OVERFLOW;
    float *arrays = calloc(count * 6, sizeof(float));
    float *work = malloc((longest * 4 + 1) * 2 * sizeof(float));
    size_t *indices = malloc(longest * 2 * sizeof(size_t));
    if (!arrays || !work || !indices) {
        free(arrays); free(work); free(indices);
        return CN_ALLOCATION_FAILED;
    }
    esdf_context ctx = {source, out, arrays, arrays + count,
        arrays + count * 2, arrays + count * 3, arrays + count * 4, arrays + count * 5,
        work, indices, width, height, longest, radius, cutoff, pre_process != 0, post_process != 0};
    paint_offsets(&ctx);
    cn_status status;
    if (count >= 65536) status = cn_parallel_for(2, 1, distance_fields, &ctx);
    else { distance_fields(&ctx, 0, 2); status = CN_OK; }
    if (status == CN_OK) status = cn_parallel_for(count, 65536, distance_output, &ctx);
    free(indices); free(work); free(arrays);
    return status;
}

CN_EXPORT cn_status cn_distance_esdf(const float *source, float *out,
    size_t height, size_t width, float radius)
{
    return cn_distance_esdf_ex(source, out, height, width, radius, 0.5f, 0, 1);
}

/* Integer chamfer adaptation from OpenCV 5.0.0 imgproc/src/distransform.cpp
 * (distanceTransform_5x5, L2 metrics 1, 1.4, 2.1969 in 16.16 fixed point).
 * Copyright (C) 2000, Intel Corporation, all rights reserved.
 * Copyright (C) 2013, OpenCV Foundation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * - The name of the copyright holders may not be used to endorse or promote
 *   products derived from this software without specific prior written permission.
 * This software is provided by the copyright holders and contributors "as is"
 * and any express or implied warranties, including, but not limited to, the
 * implied warranties of merchantability and fitness for a particular purpose
 * are disclaimed. In no event shall the Intel Corporation or contributors be
 * liable for any direct, indirect, incidental, special, exemplary, or
 * consequential damages (including, but not limited to, procurement of
 * substitute goods or services; loss of use, data, or profits; or business
 * interruption) however caused and on any theory of liability, whether in
 * contract, strict liability, or tort (including negligence or otherwise)
 * arising in any way out of the use of this software, even if advised of the
 * possibility of such damage.
 */

typedef struct binary_context {
    const uint8_t *source;
    float *out;
    void *fields[2];
    size_t height, width;
    double spread;
    int external_distances, double_division;
} binary_context;

static const int neighbor_y[8] = {-2, -2, -1, -1, -1, -1, -1, 0};
static const int neighbor_x[8] = {-1, 1, -2, -1, 0, 1, 2, -1};

/* OpenCV 5.0.0 keeps distances unsigned and saturates them at UINT_MAX minus
 * the long step: the border holds that value and the forward pass clamps to it,
 * so a field with no zero pixel reads (float)DIST_MAX / 65536 = 65533.805. */
#define CHAMFER_LONG 143976u
#define CHAMFER_MAX (UINT32_MAX - CHAMFER_LONG)

static void integer_chamfer(const binary_context *ctx, uint32_t *field, int white_zero)
{
    const uint32_t costs[8] = {CHAMFER_LONG, CHAMFER_LONG, CHAMFER_LONG, 91750, 65536, 91750,
        CHAMFER_LONG, 65536};
    size_t count = ctx->width * ctx->height;
    for (size_t p = 0; p < count; ++p) field[p] = UINT32_MAX;
    for (int pass = 0; pass < 2; ++pass) {
        int direction = pass == 0 ? 1 : -1;
        for (size_t i = 0; i < count; ++i) {
            size_t p = pass == 0 ? i : count - 1 - i;
            if (pass == 0 && ((ctx->source[p] >= 128) == white_zero)) { field[p] = 0; continue; }
            uint32_t best = field[p];
            if (pass == 1 && best <= 65536) continue;
            int64_t y = (int64_t)(p / ctx->width), x = (int64_t)(p % ctx->width);
            for (size_t k = 0; k < 8; ++k) {
                int64_t yy = y + direction * neighbor_y[k], xx = x + direction * neighbor_x[k];
                uint32_t neighbor = CHAMFER_MAX;
                if (yy >= 0 && xx >= 0 && (uint64_t)yy < ctx->height && (uint64_t)xx < ctx->width)
                    neighbor = field[(size_t)yy * ctx->width + (size_t)xx];
                uint32_t alternative = neighbor + costs[k];
                if (alternative < best) best = alternative;
            }
            field[p] = best > CHAMFER_MAX ? CHAMFER_MAX : best;
        }
    }
}

static void binary_fields(void *opaque, size_t begin, size_t end)
{
    binary_context *ctx = opaque;
    for (size_t i = begin; i < end; ++i) {
        integer_chamfer(ctx, ctx->fields[i], i == 1);
    }
}

static void binary_output(void *opaque, size_t begin, size_t end)
{
    const binary_context *ctx = opaque;
    for (size_t p = begin; p < end; ++p) {
        int positive = ctx->source[p] >= 128;
        size_t field = positive ? 0 : 1;
        float value;
        if (ctx->external_distances) value = ((const float *)ctx->fields[field])[p];
        else {
            uint32_t distance = ((const uint32_t *)ctx->fields[field])[p];
            value = (float)distance * (1.0f / 65536.0f);
        }
        if (ctx->double_division) {
            double result = (double)value / ctx->spread / 2.0;
            value = (float)(positive ? result + 0.5 : 0.5 - result);
        } else {
            value /= (float)ctx->spread;
            value /= 2.0f;
            value = positive ? value + 0.5f : 0.5f - value;
        }
        if (value < 0.0f) value = 0.0f;
        if (value > 1.0f) value = 1.0f;
        ctx->out[p] = value;
    }
}

CN_EXPORT cn_status cn_distance_binary(const uint8_t *source, float *out,
    size_t height, size_t width, double spread, int double_division)
{
    if (!source || !out || !height || !width || double_division < 0 || double_division > 1) return CN_INVALID_ARGUMENT;
    if (height > INT32_MAX || width > INT32_MAX ||
        width > SIZE_MAX / sizeof(float) / height) return CN_SIZE_OVERFLOW;
    size_t count = height * width;
    void *first = malloc(count * sizeof(float)), *second = malloc(count * sizeof(float));
    if (!first || !second) { free(first); free(second); return CN_ALLOCATION_FAILED; }
    binary_context ctx = {source, out, {first, second}, height, width, spread, 0, double_division};
    cn_status status;
    if (count >= 65536) status = cn_parallel_for(2, 1, binary_fields, &ctx);
    else { binary_fields(&ctx, 0, 2); status = CN_OK; }
    if (status == CN_OK) status = cn_parallel_for(count, 65536, binary_output, &ctx);
    free(first); free(second);
    return status;
}

typedef struct binary_mask_context {
    const uint8_t *source;
    uint8_t *black_mask, *white_mask;
} binary_mask_context;

static void binary_mask_range(void *opaque, size_t begin, size_t end)
{
    const binary_mask_context *ctx = opaque;
    for (size_t p = begin; p < end; ++p) {
        uint8_t value = ctx->source[p] >= 128 ? 255 : 0;
        ctx->black_mask[p] = value;
        ctx->white_mask[p] = 255 - value;
    }
}

CN_EXPORT cn_status cn_distance_binary_masks(const uint8_t *source,
    uint8_t *black_mask, uint8_t *white_mask, size_t height, size_t width)
{
    if (!source || !black_mask || !white_mask || !height || !width) return CN_INVALID_ARGUMENT;
    if (height > INT32_MAX || width > INT32_MAX || width > SIZE_MAX / height) return CN_SIZE_OVERFLOW;
    binary_mask_context ctx = {source, black_mask, white_mask};
    return cn_parallel_for(height * width, 65536, binary_mask_range, &ctx);
}

/* Intel IPP's closed SIMD chamfer ordering remains an external primitive.
 * Thresholding, mask inversion and signed-distance composition are still C;
 * no approximate chamfer path is substituted for those vendor results. */
CN_EXPORT cn_status cn_distance_binary_combine(const uint8_t *source,
    const float *black_distance, const float *white_distance, float *out,
    size_t height, size_t width, double spread, int double_division)
{
    if (!source || !black_distance || !white_distance || !out || !height || !width ||
        double_division < 0 || double_division > 1) return CN_INVALID_ARGUMENT;
    if (height > INT32_MAX || width > INT32_MAX ||
        width > SIZE_MAX / sizeof(float) / height) return CN_SIZE_OVERFLOW;
    binary_context ctx = {source, out, {(void *)black_distance, (void *)white_distance},
        height, width, spread, 1, double_division};
    return cn_parallel_for(height * width, 65536, binary_output, &ctx);
}
