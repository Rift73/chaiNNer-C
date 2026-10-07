#include "chainner.h"
#include "cn_crt_math.h"
#include "parallel.h"

#include <math.h>
#include <stdint.h>

typedef struct gradient_context {
    float *out;
    size_t height, width;
    int kind;
    double a, b, inner, outer, start_r, start_c;
    float center_r, center_c, direction_r, direction_c;
} gradient_context;

static void gradient_rows(void *opaque, size_t begin, size_t end)
{
    const gradient_context *ctx = opaque;
    const double pi = 3.14159265358979323846264338327950288;
    for (size_t row = begin; row < end; ++row) {
        for (size_t column = 0; column < ctx->width; ++column) {
            double value;
            switch (ctx->kind) {
            case 0:
                value = (double)column / ((double)ctx->width - 1.0);
                break;
            case 1:
                value = (double)row / ((double)ctx->height - 1.0);
                break;
            case 2: {
                double dr = (double)row - (double)ctx->start_r;
                double dc = (double)column - (double)ctx->start_c;
                value = (dr * (double)ctx->direction_r + dc * (double)ctx->direction_c) / ctx->b;
                if (value < 0.0) value = 0.0;
                if (value > 1.0) value = 1.0;
                break;
            }
            case 3: {
                double dr = (double)row - (double)ctx->center_r;
                double dc = (double)column - (double)ctx->center_c;
                value = (sqrt(dr * dr + dc * dc) - ctx->inner) / (ctx->outer - ctx->inner);
                break;
            }
            default:
                value = cn_crt.atan2((double)row - (double)ctx->center_r,
                              (double)column - (double)ctx->center_c) + ctx->a;
                if (value < 0.0) value += 2.0 * pi;
                value = value / pi / 2.0;
                break;
            }
            ctx->out[row * ctx->width + column] = (float)value;
        }
    }
}

/* The coordinate math intentionally follows the original NumPy float64
 * projections and float32 centers/directions. In particular, a one-pixel axis
 * and zero-width/radius intervals retain the original NaNs/infinities. */
CN_EXPORT cn_status cn_gradient(float *out, size_t height, size_t width,
                                int kind, double a, double b, int wide_start)
{
    const double pi = 3.14159265358979323846264338327950288;
    if (kind < 0 || kind > 4) return CN_INVALID_ARGUMENT;
    if (wide_start != 0 && (wide_start != 1 || kind != 2)) return CN_INVALID_ARGUMENT;
    if (width && height > SIZE_MAX / width) return CN_SIZE_OVERFLOW;
    if (height * width > SIZE_MAX / sizeof(float)) return CN_SIZE_OVERFLOW;
    if (height && width && !out) return CN_INVALID_ARGUMENT;
    if (!height || !width) return CN_OK;
    if ((uintptr_t)out % _Alignof(float)) return CN_INVALID_ARGUMENT;
    if ((uintptr_t)out > UINTPTR_MAX - height * width * sizeof(float)) return CN_SIZE_OVERFLOW;

    gradient_context ctx = {0};
    ctx.out = out;
    ctx.height = height;
    ctx.width = width;
    ctx.kind = kind;
    ctx.a = a;
    ctx.b = b;
    ctx.center_r = (float)height / 2.0f;
    ctx.center_c = (float)width / 2.0f;
    ctx.inner = a * (double)width / 2.0;
    ctx.outer = b * (double)width / 2.0;
    if (kind == 2) {
        ctx.direction_r = (float)cn_crt.cos(a);
        ctx.direction_c = (float)cn_crt.sin(a);
        if (wide_start) {
            /* NumPy 1.x promotes a float32 array multiplied by an integer
             * beyond uint16 (or a large floating scalar) to float64. */
            double offset_r = (double)ctx.direction_r * b;
            double offset_c = (double)ctx.direction_c * b;
            ctx.start_r = (double)ctx.center_r - offset_r / 2.0;
            ctx.start_c = (double)ctx.center_c - offset_c / 2.0;
        } else {
            float offset_r = ctx.direction_r * (float)b;
            float offset_c = ctx.direction_c * (float)b;
            ctx.start_r = ctx.center_r - offset_r / 2.0f;
            ctx.start_c = ctx.center_c - offset_c / 2.0f;
        }
    } else if (kind == 4) {
        ctx.center_r = ((float)height - 1.0f) / 2.0f;
        ctx.center_c = ((float)width - 1.0f) / 2.0f;
        /* This is one adjustment, not a modulo operation, in chaiNNer. */
        if (ctx.a > pi) ctx.a -= 2.0 * pi;
        if (ctx.a < -pi) ctx.a += 2.0 * pi;
    }
    size_t grain = 65536 / width;
    if (!grain) grain = 1;
    return cn_parallel_for(height, grain, gradient_rows, &ctx);
}
