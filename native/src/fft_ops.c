#include "fft_native.h"
#include "parallel.h"
#include <limits.h>
#include <stdlib.h>
#include <string.h>

typedef struct fft_context {
    double *data;
    unsigned char *failed;
    size_t height, width;
    int inverse, columns;
    cn_fft_plan plan;
} fft_context;

static void fft_lines(void *opaque, size_t begin, size_t end)
{
    const fft_context *ctx = opaque;
    size_t length = ctx->columns ? ctx->height : ctx->width;
    double scale = ctx->inverse ? 1.0 / (double)length : 1.0;
    double *scratch = ctx->columns ? malloc(length * 2 * sizeof(double)) : NULL;
    if (ctx->columns && !scratch) { ctx->failed[begin] = 1; return; }
    for (size_t line = begin; line < end; ++line) {
        double *values = ctx->columns ? scratch : ctx->data + line * length * 2;
        if (ctx->columns)
            for (size_t y = 0; y < length; ++y) {
                values[y * 2] = ctx->data[(y * ctx->width + line) * 2];
                values[y * 2 + 1] = ctx->data[(y * ctx->width + line) * 2 + 1];
            }
        if (cn_fft_execute(ctx->plan, values, ctx->inverse, scale)) {
            ctx->failed[line] = 1;
            break;
        }
        if (ctx->columns)
            for (size_t y = 0; y < length; ++y) {
                ctx->data[(y * ctx->width + line) * 2] = values[y * 2];
                ctx->data[(y * ctx->width + line) * 2 + 1] = values[y * 2 + 1];
            }
    }
    free(scratch);
}

cn_status cn_fft2_with_plans(double *data, size_t height, size_t width,
                            int inverse, cn_fft_plan row, cn_fft_plan column)
{
    size_t lines = height > width ? height : width;
    unsigned char *failed = calloc(lines, 1);
    if (!failed) return CN_ALLOCATION_FAILED;
    fft_context ctx = {data, failed, height, width, inverse, 0, row};
    size_t grain = width >= 8192 ? 1 : 1 + 8192 / width;
    cn_status result = cn_parallel_for(height, grain, fft_lines, &ctx);
    for (size_t i = 0; i < lines; ++i)
        if (failed[i]) result = CN_ALLOCATION_FAILED;
    if (result == CN_OK) {
        ctx.columns = 1;
        ctx.plan = column;
        grain = height >= 8192 ? 1 : 1 + 8192 / height;
        result = cn_parallel_for(width, grain, fft_lines, &ctx);
        for (size_t i = 0; i < lines; ++i)
            if (failed[i]) result = CN_ALLOCATION_FAILED;
    }
    free(failed);
    return result;
}

CN_EXPORT cn_status cn_fft2(const double *src, double *out, size_t height,
                            size_t width, int inverse)
{
    if (!height || !width || (inverse != 0 && inverse != 1)) return CN_INVALID_ARGUMENT;
    if (height > (size_t)INT_MAX / 16 || width > (size_t)INT_MAX / 16 ||
        height > SIZE_MAX / width || height * width > SIZE_MAX / (2 * sizeof(double)))
        return CN_SIZE_OVERFLOW;
    size_t bytes = height * width * 2 * sizeof(double);
    uintptr_t a = (uintptr_t)src, b = (uintptr_t)out;
    if (!src || !out || a % _Alignof(double) || b % _Alignof(double)) return CN_INVALID_ARGUMENT;
    if (a > UINTPTR_MAX - bytes || b > UINTPTR_MAX - bytes) return CN_SIZE_OVERFLOW;
    if (a != b && a < b + bytes && b < a + bytes) return CN_INVALID_ARGUMENT;
    cn_fft_plan row = cn_fft_make_plan(width), column = cn_fft_make_plan(height);
    if (!row || !column) {
        cn_fft_destroy_plan(row);
        cn_fft_destroy_plan(column);
        return CN_ALLOCATION_FAILED;
    }
    if (src != out) memcpy(out, src, bytes);
    cn_status status = cn_fft2_with_plans(out, height, width, inverse, row, column);
    cn_fft_destroy_plan(row);
    cn_fft_destroy_plan(column);
    return status;
}
