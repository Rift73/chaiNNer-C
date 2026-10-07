/* Stretch Contrast: NumPy-exact bounds, the float64 stretch and channel assembly. */
#include "chainner.h"
#include "parallel.h"
#include "numeric.h"
#include <stdlib.h>
#include <string.h>

/* Exact NumPy reduction and selection traversal shared with Image Statistics.
 * Declarations are internal link contracts; the public boundary below checks
 * all buffers before invoking them. */
double cn_numpy_percentile_f32(float *scratch, size_t count, double percentile, int weak);

/* mode 0: minimum and maximum; 1: percentiles of a NumPy float64 q; 2:
 * percentiles of a Python-number q, which NumPy 2.5 treats as weak. lanes, block
 * and period describe np.min/np.max's reduction of the node's operand
 * (cn_numpy_extreme_f32). */
CN_EXPORT cn_status cn_contrast_bounds_complete(const float *source, size_t count,
    int mode, double percentile, int lanes, size_t block, size_t period, double *bounds) {
    if (!source || !bounds || !count || mode < 0 || mode > 2 ||
        !(percentile >= 0.0 && percentile <= 50.0) ||
        (lanes != 1 && lanes != 4 && lanes != 8 && lanes != 16) ||
        (uintptr_t)source % _Alignof(float) ||
        (uintptr_t)bounds % _Alignof(double)) return CN_INVALID_ARGUMENT;
    if (count > SIZE_MAX / sizeof(float) || count > PTRDIFF_MAX) return CN_SIZE_OVERFLOW;
    size_t bytes = count * sizeof(float);
    uintptr_t src = (uintptr_t)source, dst = (uintptr_t)bounds;
    if (src > UINTPTR_MAX - bytes || dst > UINTPTR_MAX - 2 * sizeof(double)) return CN_SIZE_OVERFLOW;
    if (src < dst + 2 * sizeof(double) && dst < src + bytes) return CN_INVALID_ARGUMENT;
    if (!mode) {
        bounds[0] = cn_numpy_extreme_f32(source, count, lanes, 0, block, period);
        bounds[1] = cn_numpy_extreme_f32(source, count, lanes, 1, block, period);
    } else {
        float *values = malloc(bytes);
        if (!values) return CN_ALLOCATION_FAILED;
        memcpy(values, source, bytes);
        bounds[0] = cn_numpy_percentile_f32(values, count, percentile, mode == 2);
        /* The node makes two independent scalar percentile calls. Reusing the
         * first partition would change equal-value signed-zero permutations. */
        memcpy(values, source, bytes);
        bounds[1] = cn_numpy_percentile_f32(values, count, 100.0 - percentile, mode == 2);
        free(values);
    }
    return CN_OK;
}

typedef struct stretch_context {
    const float *source;
    double *output;
    float minimum;
    double span;
} stretch_context;

static void stretch_range(void *opaque, size_t begin, size_t end) {
    const stretch_context *ctx = opaque;
    for (size_t i = begin; i < end; ++i) {
        /* NumPy subtracts in float32 before promoting an out-of-range Python
         * scalar denominator to float64. Preserve overflow in that first step. */
        float shifted = ctx->source[i] - ctx->minimum;
        ctx->output[i] = (double)shifted / ctx->span;
    }
}

CN_EXPORT cn_status cn_contrast_stretch_wide(const float *source, double *output,
    size_t count, double minimum, double span) {
    if (!source || !output || !count || (uintptr_t)source % _Alignof(float) ||
        (uintptr_t)output % _Alignof(double)) return CN_INVALID_ARGUMENT;
    if (count > SIZE_MAX / sizeof(double)) return CN_SIZE_OVERFLOW;
    uintptr_t src = (uintptr_t)source, dst = (uintptr_t)output;
    if (src > UINTPTR_MAX - count * sizeof(float) ||
        dst > UINTPTR_MAX - count * sizeof(double)) return CN_SIZE_OVERFLOW;
    if (src < dst + count * sizeof(double) && dst < src + count * sizeof(float)) return CN_INVALID_ARGUMENT;
    stretch_context ctx = {source, output, (float)minimum, span};
    return cn_parallel_for(count, 65536, stretch_range, &ctx);
}

typedef struct contrast_merge_context {
    const void *const *sources;
    const int *f64;
    double *output;
    size_t channels;
} contrast_merge_context;

static void contrast_merge_range(void *opaque, size_t begin, size_t end) {
    const contrast_merge_context *ctx = opaque;
    for (size_t i = begin; i < end; ++i)
        for (size_t c = 0; c < ctx->channels; ++c)
            ctx->output[i * ctx->channels + c] = ctx->f64[c]
                ? ((const double *)ctx->sources[c])[i]
                : (double)((const float *)ctx->sources[c])[i];
}

/* At least one independently stretched channel promoted to float64. This
 * assembly preserves NumPy dstack's resulting precision, including the alpha. */
CN_EXPORT cn_status cn_contrast_merge_wide(const void *const *sources,
    const int *f64, double *output, size_t pixels, size_t channels) {
    if (!sources || !f64 || !output || !pixels || channels < 1 || channels > 4 ||
        (uintptr_t)sources % _Alignof(void *) || (uintptr_t)f64 % _Alignof(int) ||
        (uintptr_t)output % _Alignof(double)) return CN_INVALID_ARGUMENT;
    if (pixels > SIZE_MAX / (channels * sizeof(double))) return CN_SIZE_OVERFLOW;
    size_t bytes = pixels * channels * sizeof(double);
    uintptr_t dst = (uintptr_t)output, meta = (uintptr_t)sources, types = (uintptr_t)f64;
    if (dst > UINTPTR_MAX - bytes || meta > UINTPTR_MAX - channels * sizeof(void *) ||
        types > UINTPTR_MAX - channels * sizeof(int)) return CN_SIZE_OVERFLOW;
    if ((meta < dst + bytes && dst < meta + channels * sizeof(void *)) ||
        (types < dst + bytes && dst < types + channels * sizeof(int))) return CN_INVALID_ARGUMENT;
    for (size_t c = 0; c < channels; ++c) {
        if (f64[c] != 0 && f64[c] != 1) return CN_INVALID_ARGUMENT;
        size_t item = f64[c] ? sizeof(double) : sizeof(float);
        uintptr_t src = (uintptr_t)sources[c];
        if (!src || src % item) return CN_INVALID_ARGUMENT;
        if (src > UINTPTR_MAX - pixels * item) return CN_SIZE_OVERFLOW;
        if (src < dst + bytes && dst < src + pixels * item) return CN_INVALID_ARGUMENT;
    }
    contrast_merge_context ctx = {sources, f64, output, channels};
    return cn_parallel_for(pixels, 16384, contrast_merge_range, &ctx);
}
