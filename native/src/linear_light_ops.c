/* Linear Light (Descale and Resize): an image's colour channels between encoded values and linear light with
 * linear_light.h's curves, in place, rows in parallel. Each value goes through its curve in double and is rounded to
 * float32 once. The powers are linear_light_shared.h's polynomial, which the AVX2 unit repeats four lanes at a time
 * with the same bits.
 * SPDX-License-Identifier: GPL-3.0-only */
#include "chainner.h"
#include "isa.h"
#include "linear_light_shared.h"
#include "parallel.h"
#include "run_kernels.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

static double from_bits(uint64_t bits)
{
    double value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint64_t to_bits(double value)
{
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

/* c[0] + c[1] x + ... + c[7] x^7 in Estrin's order: pairs, then pairs of pairs. */
static double light_poly8(const double *c, double x)
{
    const double x2 = x * x, x4 = x2 * x2;
    const double low = (c[0] + c[1] * x) + x2 * (c[2] + c[3] * x);
    const double high = (c[4] + c[5] * x) + x2 * (c[6] + c[7] * x);
    return low + x4 * high;
}

/* Through c[11] x^11: light_poly8 plus x^8 times the last four terms in its order. */
static double light_poly12(const double *c, double x)
{
    const double x2 = x * x, x4 = x2 * x2;
    return light_poly8(c, x) + (x4 * x4) * ((c[8] + c[9] * x) + x2 * (c[10] + c[11] * x));
}

/* a^g for a >= 0 (linear_light_shared.h). */
static double light_pow(double a, double g)
{
    const uint64_t bits = to_bits(a);
    double e = from_bits(UINT64_C(0x4330000000000000) | bits >> 52) - LIGHT_EXPONENT_BIAS;
    double m = from_bits((bits & UINT64_C(0x000fffffffffffff)) | UINT64_C(0x3ff0000000000000));
    const int fold = m > LIGHT_SQRT2;
    m = fold ? m * 0.5 : m;
    e = e + (fold ? 1.0 : 0.0);
    const double s = (m - 1.0) / (m + 1.0);
    const double t = g * (e + 2.0 * s * light_poly8(light_log_terms, s * s) * LIGHT_INV_LN2);
    const double shifted = t + LIGHT_ROUND;
    const double f = (t - (shifted - LIGHT_ROUND)) * LIGHT_LN2;
    const double r = light_poly12(light_exp_terms, f) * from_bits((to_bits(shifted) + 1023) << 52);
    return a == 0.0 ? 0.0 : a < INFINITY ? r : a;
}

#define CN_LIGHT_POW light_pow
#include "linear_light.h"

/* values[i] for i in [0, count), values starting at a pixel wherever alpha is skipped. */
static void light_values(float *values, size_t count, size_t channels, int skip_alpha, int transfer, int encode)
{
    const size_t colour = channels - (size_t)skip_alpha;
    for (size_t i = 0; i < count; i += channels)
        for (size_t c = 0; c < colour && i + c < count; ++c) {
            const double value = values[i + c];
            values[i + c] = (float)(encode ? cn_light_encode(value, transfer) : cn_light_linearize(value, transfer));
        }
}

typedef struct light_context {
    float *image;
    size_t row, channels; /* floats per row and per pixel */
    int skip_alpha, transfer, encode;
    cn_isa_level isa;
} light_context;

/* Rows [begin, end): at avx2 and above whole groups of four values through the AVX2 unit (the same bits), the rest
 * here. Four values start at a pixel wherever alpha is skipped (two or four channels). */
static void light_rows(void *opaque, size_t begin, size_t end)
{
    const light_context *ctx = opaque;
    float *values = ctx->image + begin * ctx->row;
    size_t count = (end - begin) * ctx->row, done = 0;
#if defined(_MSC_VER) && defined(_M_X64)
    if (ctx->isa >= CN_ISA_AVX2) {
        done = count / 4 * 4;
        cn_light_values_avx2(values, done, ctx->channels, ctx->skip_alpha, ctx->transfer, ctx->encode);
    }
#endif
    light_values(values + done, count - done, ctx->channels, ctx->skip_alpha, ctx->transfer, ctx->encode);
}

CN_EXPORT cn_status cn_linear_light(float *image, size_t height, size_t width, size_t channels,
    int transfer, int encode, int skip_alpha)
{
    if (!image || (uintptr_t)image % _Alignof(float) || channels < 1 || channels > 4 ||
        transfer < CN_LIGHT_GAMMA24 || transfer > CN_LIGHT_BT709 || (encode != 0 && encode != 1) ||
        (skip_alpha != 0 && skip_alpha != 1) || (skip_alpha && channels != 2 && channels != 4))
        return CN_INVALID_ARGUMENT;
    if (!height || !width) return CN_OK;
    if (width > SIZE_MAX / sizeof(float) / channels / height) return CN_SIZE_OVERFLOW;
    size_t bytes = height * width * channels * sizeof(float);
    if ((uintptr_t)image > UINTPTR_MAX - bytes) return CN_SIZE_OVERFLOW;
    light_context ctx = {image, width * channels, channels, skip_alpha, transfer, encode, cn_isa_current()};
    /* About 16384 values a chunk. */
    size_t grain = 16384 / ctx.row;
    return cn_parallel_for(height, grain ? grain : 1, light_rows, &ctx);
}
