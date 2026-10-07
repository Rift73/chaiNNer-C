/* chaiNNer's two-dimensional void-and-cluster algorithm, adapted to C17
 * under GPL-3.0-only. Source: nodes/impl/noise_functions/blue.py (Ulichney's
 * method, as implemented by Moments in Graphics and retained in chaiNNer).
 * Fourier Gaussian table/order adapted from SciPy 1.18.1 ni_fourier.c,
 * Copyright (C) 2003-2005 Peter J. Verveer, BSD-3-Clause; full notice in
 * chainner_native.LICENSE.txt. FFT code carries its separate BSD notice.
 */
#include "cn_crt_math.h"
#include "fft_native.h"
#include "parallel.h"
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct blue_context {
    size_t height, width, count, ones;
    uint8_t *pattern;
    double *spectrum, *factor_y, *factor_x;
    cn_fft_plan row, column;
} blue_context;

/* SciPy 1.18.1's _nd_image (MinGW gcc 15.2) imports exp from the UCRT
 * (api-ms-win-crt-math), the process's ucrtbase.dll exp that cn_crt holds. */
static void gaussian_table(double *table, size_t length, double sigma)
{
    double parameter = sigma * 3.14159265358979323846 / (double)length;
    parameter = -2.0 * parameter * parameter;
    for (size_t i = 0; i < length; ++i) {
        int64_t frequency = i < (length + 1) / 2 ? (int64_t)i : (int64_t)i - (int64_t)length;
        double value = parameter * (double)frequency * (double)frequency;
        table[i] = length == 1 ? 1.0 : (fabs(value) > 50.0 ? 0.0 : cn_crt.exp(value));
    }
}

static void initialize_spectrum(void *opaque, size_t begin, size_t end)
{
    blue_context *ctx = opaque;
    int invert = ctx->ones >= (ctx->count + 1) / 2;
    for (size_t i = begin; i < end; ++i) {
        int value = ctx->pattern[i] != 0;
        ctx->spectrum[i * 2] = (double)(invert ? !value : value);
        ctx->spectrum[i * 2 + 1] = 0.0;
    }
}

static void filter_spectrum(void *opaque, size_t begin, size_t end)
{
    const blue_context *ctx = opaque;
    for (size_t i = begin; i < end; ++i) {
        double factor = 1.0;
        if (ctx->height > 1) factor *= ctx->factor_y[i / ctx->width];
        if (ctx->width > 1) factor *= ctx->factor_x[i % ctx->width];
        ctx->spectrum[i * 2] *= factor;
        ctx->spectrum[i * 2 + 1] *= factor;
    }
}

static cn_status blue_filter(blue_context *ctx)
{
    cn_status status = cn_parallel_for(ctx->count, 8192, initialize_spectrum, ctx);
    if (status != CN_OK) return status;
    status = cn_fft2_with_plans(ctx->spectrum, ctx->height, ctx->width, 0, ctx->row, ctx->column);
    if (status != CN_OK) return status;
    status = cn_parallel_for(ctx->count, 8192, filter_spectrum, ctx);
    if (status != CN_OK) return status;
    return cn_fft2_with_plans(ctx->spectrum, ctx->height, ctx->width, 1, ctx->row, ctx->column);
}

static cn_status blue_find(blue_context *ctx, int tightest, size_t *index)
{
    cn_status status = blue_filter(ctx);
    if (status != CN_OK) return status;
    int invert = ctx->ones >= (ctx->count + 1) / 2;
    double best = 0.0;
    *index = 0;
    for (size_t i = 0; i < ctx->count; ++i) {
        int value = ctx->pattern[i] != 0;
        if (invert) value = !value;
        double score = ctx->spectrum[i * 2];
        if (tightest ? !value : value) score = tightest ? -1.0 : 2.0;
        if (!i || isnan(score) || (tightest ? score > best : score < best)) {
            best = score;
            *index = i;
            if (isnan(score)) break;
        }
    }
    return CN_OK;
}

static void blue_destroy(blue_context *ctx)
{
    cn_fft_destroy_plan(ctx->row);
    cn_fft_destroy_plan(ctx->column);
    free(ctx->pattern);
    free(ctx->spectrum);
    free(ctx->factor_y);
    free(ctx->factor_x);
}

/* permutation is the seeded arange shuffle, not a random image. Its algorithm
 * is supplied by the shared C PCG64 generator. This entry also permits direct
 * differential validation of every subsequent ranking decision. */
CN_EXPORT cn_status cn_blue_from_permutation(const int64_t *permutation, int32_t *out,
        size_t height, size_t width, double sigma_y, double sigma_x, double initial_fraction)
{
    if (!height || !width || !isfinite(initial_fraction)) return CN_INVALID_ARGUMENT;
    if (height > (size_t)INT_MAX / 16 || width > (size_t)INT_MAX / 16 ||
        height > SIZE_MAX / width || height * width > INT32_MAX ||
        height * width > SIZE_MAX / (2 * sizeof(double))) return CN_SIZE_OVERFLOW;
    size_t count = height * width;
    size_t in_bytes = count * sizeof(int64_t), out_bytes = count * sizeof(int32_t);
    uintptr_t a = (uintptr_t)permutation, b = (uintptr_t)out;
    if (!a || !b || a % _Alignof(int64_t) || b % _Alignof(int32_t)) return CN_INVALID_ARGUMENT;
    if (a > UINTPTR_MAX - in_bytes || b > UINTPTR_MAX - out_bytes) return CN_SIZE_OVERFLOW;
    if (a < b + out_bytes && b < a + in_bytes) return CN_INVALID_ARGUMENT;
    blue_context ctx = {height, width, count, 0, NULL, NULL, NULL, NULL, NULL, NULL};
    ctx.pattern = calloc(count, 1);
    ctx.spectrum = malloc(count * 2 * sizeof(double));
    ctx.factor_y = malloc(height * sizeof(double));
    ctx.factor_x = malloc(width * sizeof(double));
    ctx.row = cn_fft_make_plan(width);
    ctx.column = cn_fft_make_plan(height);
    uint8_t *initial = malloc(count);
    if (!ctx.pattern || !ctx.spectrum || !ctx.factor_y || !ctx.factor_x || !ctx.row || !ctx.column || !initial) {
        blue_destroy(&ctx);
        free(initial);
        return CN_ALLOCATION_FAILED;
    }
    cn_status status = CN_OK;
    for (size_t i = 0; i < count; ++i) {
        if (permutation[i] < 0 || (uint64_t)permutation[i] >= count || ctx.pattern[permutation[i]]) {
            status = CN_INVALID_ARGUMENT;
            goto done;
        }
        ctx.pattern[permutation[i]] = 1;
    }
    double proposed = (double)count * initial_fraction;
    size_t n_initial = proposed < 1.0 ? 1 : proposed >= (double)((count - 1) / 2)
        ? (count - 1) / 2 : (size_t)proposed;
    if (!n_initial) n_initial = 1;
    ctx.ones = n_initial;
    for (size_t i = 0; i < count; ++i) ctx.pattern[i] = (uint64_t)permutation[i] < n_initial;
    gaussian_table(ctx.factor_y, height, sigma_y);
    gaussian_table(ctx.factor_x, width, sigma_x);
    size_t tightest, void_index;
    for (;;) {
        status = blue_find(&ctx, 1, &tightest);
        if (status != CN_OK) goto done;
        ctx.ones -= ctx.pattern[tightest] != 0;
        ctx.pattern[tightest] = 0;
        status = blue_find(&ctx, 0, &void_index);
        if (status != CN_OK) goto done;
        ctx.ones += ctx.pattern[void_index] == 0;
        ctx.pattern[void_index] = 1;
        if (void_index == tightest) break;
    }
    memcpy(initial, ctx.pattern, count);
    size_t saved_ones = ctx.ones;
    for (size_t rank = n_initial; rank > 0;) {
        --rank;
        status = blue_find(&ctx, 1, &tightest);
        if (status != CN_OK) goto done;
        ctx.ones -= ctx.pattern[tightest] != 0;
        ctx.pattern[tightest] = 0;
        out[tightest] = (int32_t)rank;
    }
    memcpy(ctx.pattern, initial, count);
    ctx.ones = saved_ones;
    for (size_t rank = n_initial; rank < (count + 1) / 2; ++rank) {
        status = blue_find(&ctx, 0, &void_index);
        if (status != CN_OK) goto done;
        ctx.ones += ctx.pattern[void_index] == 0;
        ctx.pattern[void_index] = 1;
        out[void_index] = (int32_t)rank;
    }
    for (size_t rank = (count + 1) / 2; rank < count; ++rank) {
        status = blue_find(&ctx, 1, &tightest);
        if (status != CN_OK) goto done;
        ctx.ones += ctx.pattern[tightest] == 0;
        ctx.pattern[tightest] = 1;
        out[tightest] = (int32_t)rank;
    }
done:
    free(initial);
    blue_destroy(&ctx);
    return status;
}

/* Full periodic Gaussian field for the public void/cluster helper contract. */
CN_EXPORT cn_status cn_blue_filtered(const uint8_t *pattern, double *out,
        size_t height, size_t width, double sigma_y, double sigma_x)
{
    if (!height || !width) return CN_INVALID_ARGUMENT;
    if (height > (size_t)INT_MAX / 16 || width > (size_t)INT_MAX / 16 ||
        height > SIZE_MAX / width || height * width > SIZE_MAX / (2 * sizeof(double)))
        return CN_SIZE_OVERFLOW;
    size_t count = height * width, bytes = count * sizeof(double);
    uintptr_t a = (uintptr_t)pattern, b = (uintptr_t)out;
    if (!a || !b || b % _Alignof(double)) return CN_INVALID_ARGUMENT;
    if (a > UINTPTR_MAX - count || b > UINTPTR_MAX - bytes) return CN_SIZE_OVERFLOW;
    if (a < b + bytes && b < a + count) return CN_INVALID_ARGUMENT;
    blue_context ctx = {height, width, count, 0, NULL, NULL, NULL, NULL, NULL, NULL};
    ctx.pattern = malloc(count);
    ctx.spectrum = malloc(count * 2 * sizeof(double));
    ctx.factor_y = malloc(height * sizeof(double));
    ctx.factor_x = malloc(width * sizeof(double));
    ctx.row = cn_fft_make_plan(width);
    ctx.column = cn_fft_make_plan(height);
    if (!ctx.pattern || !ctx.spectrum || !ctx.factor_y || !ctx.factor_x || !ctx.row || !ctx.column) {
        blue_destroy(&ctx);
        return CN_ALLOCATION_FAILED;
    }
    memcpy(ctx.pattern, pattern, count);
    for (size_t i = 0; i < count; ++i) ctx.ones += pattern[i] != 0;
    gaussian_table(ctx.factor_y, height, sigma_y);
    gaussian_table(ctx.factor_x, width, sigma_x);
    cn_status status = blue_filter(&ctx);
    if (status == CN_OK)
        for (size_t i = 0; i < count; ++i) out[i] = ctx.spectrum[i * 2];
    blue_destroy(&ctx);
    return status;
}

CN_EXPORT cn_status cn_blue_find_index(const uint8_t *pattern, size_t height,
        size_t width, double sigma_y, double sigma_x, int tightest, size_t *index)
{
    if (!height || !width || (tightest != 0 && tightest != 1) || !index ||
        (uintptr_t)index % _Alignof(size_t)) return CN_INVALID_ARGUMENT;
    if (height > (size_t)INT_MAX / 16 || width > (size_t)INT_MAX / 16 ||
        height > SIZE_MAX / width || height * width > SIZE_MAX / sizeof(double))
        return CN_SIZE_OVERFLOW;
    size_t count = height * width;
    uintptr_t input_address = (uintptr_t)pattern, result_address = (uintptr_t)index;
    if (!pattern) return CN_INVALID_ARGUMENT;
    if (input_address > UINTPTR_MAX - count || result_address > UINTPTR_MAX - sizeof(size_t))
        return CN_SIZE_OVERFLOW;
    if (input_address < result_address + sizeof(size_t) && result_address < input_address + count)
        return CN_INVALID_ARGUMENT;
    double *field = malloc(count * sizeof(double));
    if (!field) return CN_ALLOCATION_FAILED;
    cn_status status = cn_blue_filtered(pattern, field, height, width, sigma_y, sigma_x);
    if (status == CN_OK) {
        size_t ones = 0, best_index = 0;
        for (size_t i = 0; i < count; ++i) ones += pattern[i] != 0;
        int invert = ones >= (count + 1) / 2;
        double best = 0.0;
        for (size_t i = 0; i < count; ++i) {
            int value = pattern[i] != 0;
            if (invert) value = !value;
            double score = field[i];
            if (tightest ? !value : value) score = tightest ? -1.0 : 2.0;
            if (!i || isnan(score) || (tightest ? score > best : score < best)) {
                best = score;
                best_index = i;
                if (isnan(score)) break;
            }
        }
        *index = best_index;
    }
    free(field);
    return status;
}

CN_EXPORT cn_status cn_blue_normalize(const int32_t *ranks, void *out, size_t count,
                                      int double_output)
{
    if (!count || (double_output != 0 && double_output != 1)) return CN_INVALID_ARGUMENT;
    size_t element = double_output ? sizeof(double) : sizeof(float);
    if (count > INT32_MAX || count > SIZE_MAX / element) return CN_SIZE_OVERFLOW;
    uintptr_t a = (uintptr_t)ranks, b = (uintptr_t)out;
    if (!a || !b || a % _Alignof(int32_t) || b % element) return CN_INVALID_ARGUMENT;
    size_t input_bytes = count * sizeof(int32_t), output_bytes = count * element;
    if (a > UINTPTR_MAX - input_bytes || b > UINTPTR_MAX - output_bytes) return CN_SIZE_OVERFLOW;
    if (a < b + output_bytes && b < a + input_bytes) return CN_INVALID_ARGUMENT;
    for (size_t i = 0; i < count; ++i) {
        float value = (float)ranks[i];
        if (double_output) ((double *)out)[i] = (double)value / (double)(count - 1);
        else ((float *)out)[i] = value / (float)(count - 1);
    }
    return CN_OK;
}
