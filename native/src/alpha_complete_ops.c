/* Altered C adaptation of chaiNNer-rs fill_alpha.rs, fragment_blur.rs,
 * blend.rs and util/{grid,bits}.rs at commit
 * 6f6ead6064f81b4049d3deb803c9279c7950736b.
 * https://github.com/chaiNNer-org/chaiNNer-rs
 * A balanced C kd-tree replaces the R-tree while retaining its per-cell
 * candidate circle, float32 distances and lexicographic sampling ties.
 * The C ABI, validation and threadpool integration are GPL-3.0-only.
 *
 * MIT License
 * Copyright (c) 2023 Michael Schmidt
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
#include "numeric.h"
#include "parallel.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static cn_status validate(const float *src, float *dst, size_t h, size_t w,
                          size_t *pixels)
{
    size_t bytes;
    uintptr_t a = (uintptr_t)src, b = (uintptr_t)dst;
    if (h && w > SIZE_MAX / h) return CN_SIZE_OVERFLOW;
    *pixels = h * w;
    if (*pixels > SIZE_MAX / (4 * sizeof(float))) return CN_SIZE_OVERFLOW;
    bytes = *pixels * 4 * sizeof(float);
    if (!bytes) return CN_OK;
    if (!src || !dst || a % _Alignof(float) || b % _Alignof(float))
        return CN_INVALID_ARGUMENT;
    if (a > UINTPTR_MAX - bytes || b > UINTPTR_MAX - bytes)
        return CN_SIZE_OVERFLOW;
    if (a < b + bytes && b < a + bytes) return CN_INVALID_ARGUMENT;
    return CN_OK;
}

typedef struct binary_context {
    const float *src;
    float *dst;
    float threshold;
} binary_context;

static void binary_range(void *raw, size_t begin, size_t end)
{
    const binary_context *ctx = (const binary_context *)raw;
    size_t p, c;
    for (p = begin; p < end; ++p) {
        float a = ctx->src[4 * p + 3] < ctx->threshold ? 0.0f : 1.0f;
        for (c = 0; c < 3; ++c) ctx->dst[4 * p + c] = ctx->src[4 * p + c] * a;
        ctx->dst[4 * p + 3] = a;
    }
}

typedef struct fragment_context {
    const float *original;
    float *output;
    size_t height, width, count;
    int64_t dx[255], dy[255];
} fragment_context;

static void fragment_range(void *raw, size_t begin, size_t end)
{
    const fragment_context *ctx = (const fragment_context *)raw;
    size_t p;
    for (p = begin; p < end; ++p) {
        size_t x = p % ctx->width, y = p / ctx->width, j, c;
        float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        unsigned int count = 0;
        float rgb_scale, a_scale, alpha, inv_alpha, top_alpha, final_alpha, divisor;
        for (j = 0; j < ctx->count; ++j) {
            int64_t xx = (int64_t)x + ctx->dx[j], yy = (int64_t)y + ctx->dy[j];
            if (xx >= 0 && yy >= 0 && (uint64_t)xx < ctx->width && (uint64_t)yy < ctx->height) {
                size_t q = (size_t)yy * ctx->width + (size_t)xx;
                for (c = 0; c < 4; ++c) sum[c] += ctx->original[4 * q + c];
                ++count;
            }
        }
        rgb_scale = sum[3] == 0.0f ? 1.0f : 1.0f / sum[3];
        a_scale = count == 0 ? 1.0f : 1.0f / (float)count;
        for (c = 0; c < 3; ++c) sum[c] *= rgb_scale;
        alpha = sum[3] * a_scale;
        inv_alpha = 1.0f - alpha;
        alpha = 1.0f - inv_alpha * inv_alpha;
        top_alpha = ctx->output[4 * p + 3];
        final_alpha = 1.0f - (1.0f - alpha) * (1.0f - top_alpha);
        divisor = final_alpha == 0.0f ? 1.0f : final_alpha;
        for (c = 0; c < 3; ++c) {
            float top = ctx->output[4 * p + c] * top_alpha;
            float bottom = sum[c] * alpha;
            bottom *= 1.0f - top_alpha;
            ctx->output[4 * p + c] = (top + bottom) / divisor;
        }
        ctx->output[4 * p + 3] = final_alpha;
    }
}

CN_EXPORT cn_status cn_alpha_fragment(const float *src, float *dst,
    size_t height, size_t width, float threshold, size_t iterations, size_t count)
{
    size_t pixels, p, c, i, j;
    float *original;
    binary_context binary = {src, dst, threshold};
    fragment_context ctx;
    cn_status status = validate(src, dst, height, width, &pixels);
    if (status != CN_OK) return status;
    if (iterations > UINT32_MAX || count > UINT32_MAX || (iterations && (!count || count > 255)))
        return CN_INVALID_ARGUMENT;
    if (!pixels) return CN_OK;
    /* A nonempty image's checked byte count bounds each dimension below INT64_MAX. */
    status = cn_parallel_for(pixels, 1024, binary_range, &binary);
    if (status != CN_OK || !iterations) return status;
    original = (float *)malloc(pixels * 4 * sizeof(float));
    if (!original) return CN_ALLOCATION_FAILED;
    for (p = 0; p < pixels; ++p) {
        for (c = 0; c < 3; ++c) original[4 * p + c] = dst[4 * p + c] * dst[4 * p + 3];
        original[4 * p + 3] = dst[4 * p + 3];
    }
    ctx.original = original; ctx.output = dst;
    ctx.height = height; ctx.width = width; ctx.count = count;
    for (i = 0; i < iterations; ++i) {
        /* Rust's inferred i32 shift wraps the shift count in release builds. */
        uint32_t bits = UINT32_C(1) << (i & 31);
        int32_t signed_radius;
        float radius;
        memcpy(&signed_radius, &bits, sizeof(bits));
        radius = (float)signed_radius;
        for (j = 0; j < count; ++j) {
            float angle = (float)j / (float)count;
            angle *= 3.14159265358979323846f;
            angle *= 2.0f;
            angle += (float)i;
            ctx.dx[j] = (int64_t)roundf(sinf(angle) * radius);
            ctx.dy[j] = (int64_t)roundf(cosf(angle) * radius);
        }
        status = cn_parallel_for(pixels, 256, fragment_range, &ctx);
        if (status != CN_OK) break;
    }
    if (status == CN_OK) {
        binary.src = dst; binary.threshold = 0.01f;
        status = cn_parallel_for(pixels, 1024, binary_range, &binary);
    }
    free(original);
    return status;
}

typedef struct point {
    float x, y, color[4];
    size_t index;
} point;

static int compare_xy(const void *pa, const void *pb)
{
    const point *a = (const point *)pa, *b = (const point *)pb;
    if (a->x != b->x) return a->x < b->x ? -1 : 1;
    if (a->y != b->y) return a->y < b->y ? -1 : 1;
    return a->index < b->index ? -1 : a->index != b->index;
}

static int compare_yx(const void *pa, const void *pb)
{
    const point *a = (const point *)pa, *b = (const point *)pb;
    if (a->y != b->y) return a->y < b->y ? -1 : 1;
    return compare_xy(pa, pb);
}

static void exchange(point *a, point *b)
{
    point temp = *a; *a = *b; *b = temp;
}

static void select_point(point *data, size_t count, size_t rank, unsigned int axis)
{
    size_t lo = 0, hi = count, budget = 0, size = count;
    int (*compare)(const void *, const void *) = axis ? compare_yx : compare_xy;
    while (size > 1) { ++budget; size >>= 1; }
    budget *= 2;
    while (hi - lo > 16) {
        size_t lower = lo, upper = hi, scan = lo, mid = lo + (hi - lo) / 2;
        point pivot;
        if (!budget--) { qsort(data + lo, hi - lo, sizeof(point), compare); return; }
        if (compare(&data[mid], &data[lo]) < 0) exchange(&data[mid], &data[lo]);
        if (compare(&data[hi - 1], &data[mid]) < 0) exchange(&data[hi - 1], &data[mid]);
        if (compare(&data[mid], &data[lo]) < 0) exchange(&data[mid], &data[lo]);
        pivot = data[mid];
        while (scan < upper) {
            int cmp = compare(&data[scan], &pivot);
            if (cmp < 0) exchange(&data[scan++], &data[lower++]);
            else if (cmp > 0) exchange(&data[scan], &data[--upper]);
            else ++scan;
        }
        if (rank < lower) hi = lower;
        else if (rank >= upper) lo = upper;
        else return;
    }
    qsort(data + lo, hi - lo, sizeof(point), compare);
}

static void build_tree(point *data, size_t count, unsigned int axis)
{
    size_t mid;
    if (count < 2) return;
    mid = count / 2;
    select_point(data, count, mid, axis);
    build_tree(data, mid, axis ^ 1);
    build_tree(data + mid + 1, count - mid - 1, axis ^ 1);
}

static float distance(float x, float y, const point *p)
{
    float dx = x - p->x, dy = y - p->y;
    return dx * dx + dy * dy;
}

typedef struct query {
    float x, y, center_x, center_y, limit, best_distance;
    const point *best;
} query;

static void nearest(const point *points, size_t count, unsigned int axis, query *q)
{
    size_t mid;
    const point *p;
    float delta, d;
    if (!count) return;
    mid = count / 2; p = points + mid;
    delta = axis ? q->y - p->y : q->x - p->x;
    d = distance(q->x, q->y, p);
    if ((!q->best || d <= q->best_distance) && distance(q->center_x, q->center_y, p) <= q->limit) {
        if (!q->best || d < q->best_distance || compare_xy(p, q->best) < 0) {
            q->best = p; q->best_distance = d;
        }
    }
    if (delta < 0.0f) {
        nearest(points, mid, axis ^ 1, q);
        if (!q->best || delta * delta <= q->best_distance)
            nearest(points + mid + 1, count - mid - 1, axis ^ 1, q);
    } else {
        nearest(points + mid + 1, count - mid - 1, axis ^ 1, q);
        if (!q->best || delta * delta <= q->best_distance)
            nearest(points, mid, axis ^ 1, q);
    }
}

typedef struct cell {
    float x, y, limit;
    unsigned char process;
} cell;

typedef struct nearest_context {
    float *output;
    const point *points;
    const unsigned char *transparent;
    unsigned char *edges;
    cell *cells;
    size_t width, height, grid_width, point_count;
    int stage;
} nearest_context;

static const float *sample(const nearest_context *ctx, const cell *area, float x, float y)
{
    query q = {x, y, area->x, area->y, area->limit, CN_INFINITY_F, NULL};
    nearest(ctx->points, ctx->point_count, 0, &q);
    return q.best->color; /* The cell's closest boundary point is always in its circle. */
}

static void nearest_cells(void *raw, size_t begin, size_t end)
{
    nearest_context *ctx = (nearest_context *)raw;
    size_t k;
    for (k = begin; k < end; ++k) {
        cell *area = &ctx->cells[k];
        size_t x0, x1, y0, y1, x, y, c;
        if (!area->process) continue;
        x0 = (k % ctx->grid_width) * 8; y0 = (k / ctx->grid_width) * 8;
        x1 = ctx->width - x0 < 8 ? ctx->width : x0 + 8;
        y1 = ctx->height - y0 < 8 ? ctx->height : y0 + 8;
        if (ctx->stage == 0) {
            float xmin = (float)x0 - 0.5f, xmax = (float)x1 - 0.5f;
            float ymin = (float)y0 - 0.5f, ymax = (float)y1 - 0.5f;
            float radius = fmaxf(xmax - xmin, ymax - ymin) + 1.0f, max_distance;
            query q;
            area->x = (xmin + xmax) / 2.0f; area->y = (ymin + ymax) / 2.0f;
            q.x = area->x; q.y = area->y; q.center_x = q.x; q.center_y = q.y;
            q.limit = CN_INFINITY_F; q.best_distance = CN_INFINITY_F; q.best = NULL;
            nearest(ctx->points, ctx->point_count, 0, &q);
            max_distance = sqrtf(q.best_distance) + radius * 2.0f;
            area->limit = max_distance * max_distance;
        }
        for (y = y0; y < y1; ++y) for (x = x0; x < x1; ++x) {
            size_t p = y * ctx->width + x;
            if (!ctx->transparent[p]) continue;
            if (ctx->stage == 0) {
                const float *color = sample(ctx, area, (float)x, (float)y);
                memcpy(ctx->output + p * 4, color, 4 * sizeof(float));
            } else if (ctx->edges[p]) {
                float acc[4], xx = (float)x, yy = (float)y;
                const float *color;
                unsigned int j;
                const float dx[8] = {0.333f, 0.333f, -0.333f, -0.333f, 0.0f, 0.0f, 0.333f, -0.333f};
                const float dy[8] = {0.333f, -0.333f, 0.333f, -0.333f, 0.333f, -0.333f, 0.0f, 0.0f};
                memcpy(acc, ctx->output + p * 4, sizeof(acc));
                for (j = 0; j < 8; ++j) {
                    color = sample(ctx, area, xx + dx[j], yy + dy[j]);
                    for (c = 0; c < 4; ++c) acc[c] += color[c];
                }
                for (c = 0; c < 4; ++c) ctx->output[p * 4 + c] = acc[c] / acc[3];
            }
        }
    }
}

static int unequal(const float *a, const float *b)
{
    return a[0] != b[0] || a[1] != b[1] || a[2] != b[2] || a[3] != b[3];
}

static void edge_range(void *raw, size_t begin, size_t end)
{
    const nearest_context *ctx = (const nearest_context *)raw;
    size_t p;
    for (p = begin; p < end; ++p) {
        size_t x = p % ctx->width, y = p / ctx->width;
        const float *color = ctx->output + p * 4;
        ctx->edges[p] = (unsigned char)((x && unequal(color, color - 4)) ||
            (x + 1 < ctx->width && unequal(color, color + 4)) ||
            (y && unequal(color, color - ctx->width * 4)) ||
            (y + 1 < ctx->height && unequal(color, color + ctx->width * 4)));
    }
}

static int is_boundary(const unsigned char *transparent, size_t x, size_t y, size_t w, size_t h)
{
    size_t p = y * w + x;
    return !transparent[p] && ((x && transparent[p - 1]) ||
        (x + 1 < w && transparent[p + 1]) || (y && transparent[p - w]) ||
        (y + 1 < h && transparent[p + w]));
}

/* Preserve FixedBits' word-edge updates and its overallocated tail. In
 * particular, replacing this with an ideal 3x3 dilation changes radius masks. */
static void expand_grid(uint64_t *grid, size_t rows, size_t words, size_t width)
{
    size_t y, j;
    for (y = 0; y + 1 < rows; ++y)
        for (j = 0; j < words; ++j) grid[y * words + j] |= grid[(y + 1) * words + j];
    for (y = rows - 1; y > 0; --y)
        for (j = 0; j < words; ++j) grid[y * words + j] |= grid[(y - 1) * words + j];
    for (y = 0; y < rows; ++y) {
        uint64_t *row = grid + y * words;
        for (j = 0; j < words; ++j) row[j] |= (row[j] >> 1) | (row[j] << 1);
        for (j = 0; j + 1 < words; ++j) {
            row[j] |= row[j + 1] << 63;
            row[j + 1] |= row[j] >> 63;
        }
        if (width % 64) row[words - 1] &= (UINT64_C(1) << (width % 64)) - 1;
    }
}

CN_EXPORT cn_status cn_alpha_nearest(const float *src, float *dst,
    size_t height, size_t width, float threshold, size_t radius, int anti_aliasing)
{
    size_t pixels, gw, gh, cells_count, words, grid_count, x, y, p, k, count = 0;
    unsigned char *transparent = NULL, *edges = NULL;
    uint64_t *grid = NULL;
    point *points = NULL;
    cell *cells = NULL;
    binary_context binary = {src, dst, threshold};
    nearest_context ctx;
    cn_status status = validate(src, dst, height, width, &pixels);
    if (status != CN_OK) return status;
    if (radius > UINT32_MAX || (anti_aliasing != 0 && anti_aliasing != 1)) return CN_INVALID_ARGUMENT;
    if (!pixels) return CN_OK;
    gw = width / 8 + (width % 8 != 0); gh = height / 8 + (height % 8 != 0);
    cells_count = gw * gh;
    words = gw / 8 + (gw % 8 != 0); grid_count = words * gh;
    if (cells_count > SIZE_MAX / sizeof(cell) || grid_count > SIZE_MAX / sizeof(uint64_t))
        return CN_SIZE_OVERFLOW;
    status = cn_parallel_for(pixels, 1024, binary_range, &binary);
    if (status != CN_OK) return status;
    transparent = (unsigned char *)malloc(pixels);
    cells = (cell *)calloc(cells_count, sizeof(cell));
    grid = (uint64_t *)calloc(grid_count, sizeof(uint64_t));
    if (!transparent || !cells || !grid) { status = CN_ALLOCATION_FAILED; goto cleanup; }
    for (y = 0; y < height; ++y) for (x = 0; x < width; ++x) {
        size_t cx = x / 8, cy = y / 8;
        p = y * width + x;
        transparent[p] = (unsigned char)(dst[4 * p + 3] == 0.0f);
        if (transparent[p]) cells[cy * gw + cx].process = 1;
        else grid[cy * words + cx / 64] |= UINT64_C(1) << (cx % 64);
    }
    if (radius < width && radius < height) {
        size_t iterations = radius / 8 + (radius % 8 != 0);
        for (k = 0; k < iterations; ++k) expand_grid(grid, gh, words, gw);
        for (y = 0; y < gh; ++y) for (x = 0; x < gw; ++x)
            cells[y * gw + x].process &= (unsigned char)((grid[y * words + x / 64] >> (x % 64)) & 1);
    }
    free(grid); grid = NULL;
    for (y = 0; y < height; ++y) for (x = 0; x < width; ++x)
        if (is_boundary(transparent, x, y, width, height)) ++count;
    if (!count) goto cleanup;
    if (count > SIZE_MAX / sizeof(point)) { status = CN_SIZE_OVERFLOW; goto cleanup; }
    points = (point *)malloc(count * sizeof(point));
    if (anti_aliasing) edges = (unsigned char *)malloc(pixels);
    if (!points || (anti_aliasing && !edges)) { status = CN_ALLOCATION_FAILED; goto cleanup; }
    k = 0;
    for (y = 0; y < height; ++y) for (x = 0; x < width; ++x) {
        if (is_boundary(transparent, x, y, width, height)) {
            point *pt = points + k++;
            pt->x = (float)x; pt->y = (float)y; pt->index = y * width + x;
            memcpy(pt->color, dst + pt->index * 4, sizeof(pt->color));
        }
    }
    build_tree(points, count, 0);
    ctx.output = dst; ctx.points = points; ctx.transparent = transparent;
    ctx.edges = edges; ctx.cells = cells; ctx.width = width; ctx.height = height;
    ctx.grid_width = gw; ctx.point_count = count; ctx.stage = 0;
    status = cn_parallel_for(cells_count, 8, nearest_cells, &ctx);
    if (status == CN_OK && anti_aliasing) {
        status = cn_parallel_for(pixels, 512, edge_range, &ctx);
        if (status == CN_OK) {
            ctx.stage = 1;
            status = cn_parallel_for(cells_count, 8, nearest_cells, &ctx);
        }
    }
cleanup:
    free(transparent); free(edges); free(grid); free(points); free(cells);
    return status;
}
