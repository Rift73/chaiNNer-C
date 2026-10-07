/* Float32 Unsharp Mask weighted sum. Numerical ordering follows OpenCV 5.0.0
 * modules/core/src/arithm.simd.hpp (Apache-2.0): addWeighted32f narrows its
 * weights to float and stays in float. A row holding at least one vector is
 * vector-processed entirely (the last vector overlaps back), a shorter row is
 * scalar. Vectors compute fma(a, w1, fma(b, w2, 0)) when FMA3 is dispatched
 * (AVX2); without it, and in scalar code, (a * w1 + b * w2) + 0, which gives
 * the same float. This C implementation owns its image loop and uses the
 * shared CPU pool.
 */
#include "chainner.h"
#include "parallel.h"
#include <math.h>

static cn_status span(const void *pointer, size_t count, uintptr_t *end)
{
    if (!pointer || !count || (uintptr_t)pointer % _Alignof(float)) return CN_INVALID_ARGUMENT;
    if (count > SIZE_MAX / sizeof(float) || (uintptr_t)pointer > UINTPTR_MAX - count * sizeof(float)) return CN_SIZE_OVERFLOW;
    *end = (uintptr_t)pointer + count * sizeof(float);
    return CN_OK;
}

typedef struct unsharp_context {
    const float *source, *blurred;
    float *output;
    float alpha, beta;
    int fused;
} unsharp_context;

static void unsharp_range(void *opaque, size_t begin, size_t end)
{
    const unsharp_context *ctx = opaque;
    for (size_t i = begin; i < end; ++i) {
        float a = ctx->source[i], b = ctx->blurred[i];
        ctx->output[i] = ctx->fused ? fmaf(a, ctx->alpha, fmaf(b, ctx->beta, 0.0f)) :
            (a * ctx->alpha + b * ctx->beta) + 0.0f;
    }
}

CN_EXPORT cn_status cn_unsharp_weighted(const float *source, const float *blurred,
    float *output, size_t count, size_t row_width, double amount, int lanes)
{
    if (!row_width || !count || count % row_width || (lanes != 4 && lanes != 8)) return CN_INVALID_ARGUMENT;
    uintptr_t se, be, oe;
    cn_status status = span(source, count, &se);
    if (status != CN_OK) return status;
    status = span(blurred, count, &be);
    if (status != CN_OK) return status;
    status = span(output, count, &oe);
    if (status != CN_OK) return status;
    if (((uintptr_t)source < oe && (uintptr_t)output < se) ||
        ((uintptr_t)blurred < oe && (uintptr_t)output < be)) return CN_INVALID_ARGUMENT;
    unsharp_context ctx = {source, blurred, output, (float)(amount + 1.0), (float)-amount,
        lanes == 8 && row_width >= (size_t)lanes};
    return cn_parallel_for(count, 65536, unsharp_range, &ctx);
}
