/* Normal-map derivative kernel construction, normalization and rotation.
 * Numerical expressions follow chaiNNer nodes/impl/normals/edge_filter.py
 * (GPL-3.0-only). No external image or inference runtime is called here.
 * Each Gaussian sample preserves the original float64 accumulation order;
 * the final coefficient sum follows NumPy's pairwise 128-element leaves.
 * Pairwise traversal adapts this port's analysis_ops.c and NumPy 2.5.3
 * _core/src/umath/loops_utils.h.src (BSD-3-Clause; NumPy notice accompanies the DLL).
 */
#include "chainner.h"
#include "cn_crt_math.h"
#include "parallel.h"
#include <math.h>
#include <string.h>

static cn_status span(const void *pointer, size_t count, size_t item, uintptr_t *end)
{
    if (!pointer || !count || (uintptr_t)pointer % item) return CN_INVALID_ARGUMENT;
    if (count > SIZE_MAX / item || (uintptr_t)pointer > UINTPTR_MAX - count * item) return CN_SIZE_OVERFLOW;
    *end = (uintptr_t)pointer + count * item;
    return CN_OK;
}

static int overlap(const void *a, uintptr_t ae, const void *b, uintptr_t be)
{
    return (uintptr_t)a < be && (uintptr_t)b < ae;
}

typedef struct gauss_context {
    const double *parameters;
    size_t count, radius, width;
    double *out;
} gauss_context;

static void gauss_range(void *opaque, size_t begin, size_t end)
{
    const gauss_context *ctx = opaque;
    for (size_t index = begin; index < end; ++index) {
        double x = (double)(index % ctx->width) - (double)ctx->radius;
        double y = (double)(index / ctx->width) - (double)ctx->radius;
        double sum = 0.0;
        for (int offset = 0; offset < 4; ++offset) {
            double sample_x = fabs(x) - 1.0 + (double)offset * 0.25;
            double value = 0.0;
            for (size_t p = 0; p < ctx->count; ++p) {
                double sigma = ctx->parameters[2 * p], weight = ctx->parameters[2 * p + 1];
                double std2 = 2.0 * sigma * sigma;
                value += weight / (0x1.921fb54442d18p+1 * std2) * cn_crt.exp(-(sample_x * sample_x + y * y) / std2);
            }
            sum += value;
        }
        double sign = x > 0.0 ? -1.0 : x < 0.0 ? 1.0 : 0.0;
        ctx->out[index] = sum / 4.0 * sign;
    }
}

CN_EXPORT cn_status cn_normal_gaussian_kernel(const double *parameters, size_t count,
    double *out, size_t radius)
{
    if (radius > (SIZE_MAX - 1) / 2 || count > SIZE_MAX / 2) return CN_SIZE_OVERFLOW;
    size_t width = radius * 2 + 1;
    if (width > SIZE_MAX / width) return CN_SIZE_OVERFLOW;
    uintptr_t oe, pe = 0;
    cn_status status = span(out, width * width, sizeof(double), &oe);
    if (status != CN_OK) return status;
    if (count) {
        status = span(parameters, count * 2, sizeof(double), &pe);
        if (status != CN_OK) return status;
        if (overlap(parameters, pe, out, oe)) return CN_INVALID_ARGUMENT;
    }
    double total = 0.0, expected_radius = 1.0;
    for (size_t i = 0; i < count; ++i) total += parameters[2 * i + 1];
    if (total == 0.0) {
        if (radius != 0) return CN_INVALID_ARGUMENT;
        out[0] = 0.0;
        return CN_OK;
    }
    for (size_t i = 0; i < count; ++i) {
        double sigma = parameters[2 * i];
        if (2.0 * sigma * sigma == 0.0) return CN_INVALID_ARGUMENT;
        if (parameters[2 * i + 1] > 0.0) {
            double size = ceil(2.0 * sigma);
            if (!isfinite(size)) return CN_INVALID_ARGUMENT;
            if (size > expected_radius) expected_radius = size;
        }
    }
    expected_radius += 1.0;
    if ((double)radius != expected_radius) return CN_INVALID_ARGUMENT;
    gauss_context ctx = {parameters, count, radius, width, out};
    return cn_parallel_for(width * width, 512, gauss_range, &ctx);
}

typedef struct finish_context {
    const double *src;
    double *x, *y;
    size_t height, width;
    double denominator;
} finish_context;

static double left_value(const finish_context *ctx, size_t index)
{
    size_t left = ctx->width / 2;
    return ctx->src[(index / left) * ctx->width + index % left];
}

static double pair_sum(const finish_context *ctx, size_t start, size_t count)
{
    if (count < 8) {
        double result = -0.0;
        for (size_t i = 0; i < count; ++i) result += left_value(ctx, start + i);
        return result;
    }
    if (count <= 128) {
        double r[8];
        for (size_t i = 0; i < 8; ++i) r[i] = left_value(ctx, start + i);
        size_t i = 8;
        for (; i < count - count % 8; i += 8)
            for (size_t j = 0; j < 8; ++j) r[j] += left_value(ctx, start + i + j);
        double result = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < count; ++i) result += left_value(ctx, start + i);
        return result;
    }
    size_t left = count / 2;
    left -= left % 8;
    return pair_sum(ctx, start, left) + pair_sum(ctx, start + left, count - left);
}

static void finish_range(void *opaque, size_t begin, size_t end)
{
    const finish_context *ctx = opaque;
    for (size_t i = begin; i < end; ++i) {
        double value = ctx->src[i] / ctx->denominator;
        ctx->x[i] = value;
        ctx->y[(i % ctx->width) * ctx->height + (ctx->height - 1 - i / ctx->width)] = value;
    }
}

CN_EXPORT cn_status cn_normal_kernel_pair(const double *src, double *x, double *y,
    size_t height, size_t width, int normalize)
{
    if (!height || !width || (normalize != 0 && normalize != 1)) return CN_INVALID_ARGUMENT;
    if (height > SIZE_MAX / width) return CN_SIZE_OVERFLOW;
    size_t count = height * width;
    uintptr_t se, xe, ye;
    cn_status status = span(src, count, sizeof(double), &se);
    if (status != CN_OK) return status;
    status = span(x, count, sizeof(double), &xe);
    if (status != CN_OK) return status;
    status = span(y, count, sizeof(double), &ye);
    if (status != CN_OK) return status;
    if ((src != x && overlap(src, se, x, xe)) || overlap(src, se, y, ye) || overlap(x, xe, y, ye)) return CN_INVALID_ARGUMENT;
    finish_context ctx = {src, x, y, height, width, 1.0};
    if (normalize) {
        size_t n = height * (width / 2);
        ctx.denominator = 0.0;
        for (size_t i = 0; i < n;) {
            size_t chunk = n - i > 8192 ? 8192 : n - i;
            ctx.denominator += pair_sum(&ctx, i, chunk);
            i += chunk;
        }
    }
    return cn_parallel_for(count, 16384, finish_range, &ctx);
}
