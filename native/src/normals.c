#include "chainner.h"
#include "cn_crt_math.h"
#include "parallel.h"
#include "numeric.h"
#include "numpy_trig_f32.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

static void normalize(float *x, float *y, float *z)
{
    float xx = *x * *x;
    float yy = *y * *y;
    float square = xx + yy;
    /* fmin/fmax would erase NaNs; NumPy minimum/maximum propagate them. */
    float length = sqrtf(square < 1.0f ? 1.0f : square);
    *x /= length;
    *y /= length;
    square = square > 1.0f ? 1.0f : square;
    *z = sqrtf(1.0f - square);
}

static void decode(const float *n, int octahedral, float *x, float *y, float *z)
{
    float r = n[2] * 2.0f;
    float g = n[1] * 2.0f;
    r -= 1.0f;
    g -= 1.0f;
    if (!octahedral) {
        *x = r;
        *y = g;
        normalize(x, y, z);
    } else {
        *x = (r + g) / 2.0f;
        *y = r - *x;
        *z = 1.0f - fabsf(*x);
        *z -= fabsf(*y);
        float xx = *x * *x;
        float yy = *y * *y;
        float zz = *z * *z;
        float square = xx + yy;
        square += zz;
        float length = sqrtf(square);
        *x /= length;
        *y /= length;
        *z /= length;
    }
}

static cn_status validate_count(size_t count)
{
    return count > SIZE_MAX / sizeof(float) ? CN_SIZE_OVERFLOW : CN_OK;
}

typedef struct normalize_context {
    float *x, *y, *z;
} normalize_context;

static void normalize_pixels(void *opaque, size_t begin, size_t end)
{
    const normalize_context *ctx = opaque;
    for (size_t i = begin; i < end; ++i)
        normalize(ctx->x + i, ctx->y + i, ctx->z + i);
}

CN_EXPORT cn_status cn_normalize_normals(float *x, float *y, float *z, size_t count)
{
    if (validate_count(count) != CN_OK) return CN_SIZE_OVERFLOW;
    if (count && (!x || !y || !z)) return CN_INVALID_ARGUMENT;
    normalize_context ctx = {x, y, z};
    return cn_parallel_for(count, 65536, normalize_pixels, &ctx);
}

typedef struct decode_context {
    const float *image;
    size_t channels;
    int octahedral;
    float *x, *y, *z;
} decode_context;

static void decode_pixels(void *opaque, size_t begin, size_t end)
{
    const decode_context *ctx = opaque;
    for (size_t i = begin; i < end; ++i)
        decode(ctx->image + i * ctx->channels, ctx->octahedral,
               ctx->x + i, ctx->y + i, ctx->z + i);
}

CN_EXPORT cn_status cn_normal_decode(const float *image, size_t count,
                                     size_t channels, int octahedral,
                                     float *x, float *y, float *z)
{
    if (channels < 3 || (octahedral != 0 && octahedral != 1)) return CN_INVALID_ARGUMENT;
    if (count > SIZE_MAX / channels || validate_count(count * channels) != CN_OK)
        return CN_SIZE_OVERFLOW;
    if (count && (!image || !x || !y || !z)) return CN_INVALID_ARGUMENT;
    decode_context ctx = {image, channels, octahedral, x, y, z};
    return cn_parallel_for(count, 65536, decode_pixels, &ctx);
}

typedef struct encode_context {
    float *x, *y;
    const float *z;
    int octahedral;
    float *out;
} encode_context;

static void encode_pixels(void *opaque, size_t begin, size_t end)
{
    const encode_context *ctx = opaque;
    float *x = ctx->x, *y = ctx->y, *out = ctx->out;
    const float *z = ctx->z;
    for (size_t i = begin; i < end; ++i) {
        float r, g;
        if (ctx->octahedral) {
            float absolute = fabsf(x[i]) + fabsf(y[i]);
            absolute += fabsf(z[i]);
            x[i] /= absolute;
            y[i] /= absolute;
            r = x[i] + y[i];
            g = x[i] - y[i];
        } else {
            r = x[i];
            g = y[i];
        }
        r += 1.0f;
        g += 1.0f;
        out[i * 3] = ctx->octahedral ? 0.0f : z[i];
        out[i * 3 + 1] = g * 0.5f;
        out[i * 3 + 2] = r * 0.5f;
    }
}

CN_EXPORT cn_status cn_normal_encode(float *x, float *y, const float *z,
                                     size_t count, int octahedral, float *out)
{
    if (octahedral != 0 && octahedral != 1) return CN_INVALID_ARGUMENT;
    if (count > SIZE_MAX / 3 || validate_count(count * 3) != CN_OK) return CN_SIZE_OVERFLOW;
    if (count && (!x || !y || !z || !out)) return CN_INVALID_ARGUMENT;
    encode_context ctx = {x, y, z, octahedral, out};
    return cn_parallel_for(count, 65536, encode_pixels, &ctx);
}

typedef struct add_context {
    const float *n1, *n2;
    size_t channels1, channels2;
    float f1, f2;
    int strengthen, method, numpy_fma;
    float *x, *y, *z;
} add_context;

static float angle_component(float a, float b, const add_context *ctx)
{
    float angle = cn_crt.asinf(a) * ctx->f1;
    if (!ctx->strengthen) {
        float other = cn_crt.asinf(b) * ctx->f2;
        angle += other;
    }
    if (angle < -0x1.921fb6p+0f) angle = -0x1.921fb6p+0f;
    if (angle >  0x1.921fb6p+0f) angle =  0x1.921fb6p+0f;
    /* np.sin of the clamped angle: NumPy's FMA3 form (numpy_trig_f32.h; explicit
     * fmaf, which sinf would not reproduce near the hemisphere boundary) where the
     * bridge reports NumPy dispatches it, else the CRT sinf. */
    return cn_numpy_trig_f32(angle, 0, ctx->numpy_fma);
}

static void add_pixels(void *opaque, size_t begin, size_t end)
{
    const add_context *ctx = opaque;
    float *x = ctx->x, *y = ctx->y, *z = ctx->z;
    for (size_t i = begin; i < end; ++i) {
        float x1, y1, z1, x2 = 0.0f, y2 = 0.0f, z2 = 1.0f;
        decode(ctx->n1 + i * ctx->channels1, 0, &x1, &y1, &z1);
        if (!ctx->strengthen)
            decode(ctx->n2 + i * ctx->channels2, 0, &x2, &y2, &z2);
        if (ctx->method == 1) {
            x[i] = angle_component(x1, x2, ctx);
            y[i] = angle_component(y1, y2, ctx);
            normalize(x + i, y + i, z + i);
            continue;
        }
        if (z1 < 0.001f) z1 = 0.001f;
        if (z2 < 0.001f) z2 = 0.001f;
        float n_f = ctx->f1 / z1;
        float m_f = ctx->f2 / z2;
        x[i] = x1 * n_f;
        y[i] = y1 * n_f;
        if (!ctx->strengthen) {
            float x_part = x2 * m_f;
            float y_part = y2 * m_f;
            x[i] += x_part;
            y[i] += y_part;
        }
        float xx = x[i] * x[i];
        float yy = y[i] * y[i];
        float square = xx + yy;
        square += 1.0f;
        float reciprocal = 1.0f / sqrtf(square);
        x[i] *= reciprocal;
        y[i] *= reciprocal;
        z[i] = reciprocal;
    }
}

/* Both normal-addition methods fuse decode, combine and normalize in C. */
CN_EXPORT cn_status cn_normal_add(const float *n1, const float *n2,
                                  size_t count, size_t channels1, size_t channels2,
                                  int method, float f1, float f2, int strengthen, int numpy_fma,
                                  float *x, float *y, float *z)
{
    if (channels1 < 3 || (!strengthen && channels2 < 3) ||
        (method != 0 && method != 1) || (strengthen != 0 && strengthen != 1) ||
        (numpy_fma != 0 && numpy_fma != 1))
        return CN_INVALID_ARGUMENT;
    if (count > SIZE_MAX / channels1 || validate_count(count * channels1) != CN_OK)
        return CN_SIZE_OVERFLOW;
    if (!strengthen && (count > SIZE_MAX / channels2 ||
                       validate_count(count * channels2) != CN_OK))
        return CN_SIZE_OVERFLOW;
    if (count && (!n1 || (!strengthen && !n2) || !x || !y || !z)) return CN_INVALID_ARGUMENT;
    add_context ctx = {n1, n2, channels1, channels2, f1, f2, strengthen, method, numpy_fma, x, y, z};
    return cn_parallel_for(count, 65536, add_pixels, &ctx);
}

typedef struct normal_map_context {
    const float *image;
    float *out, *x, *y;
    size_t channels;
    int mode, option_a, option_b;
    float mean_x, mean_y;
} normal_map_context;

static void normal_map_decode_balance(void *opaque, size_t begin, size_t end)
{
    normal_map_context *ctx = opaque;
    for (size_t i = begin; i < end; ++i) {
        float z;
        decode(ctx->image + i * ctx->channels, 0, ctx->x + i, ctx->y + i, &z);
    }
}

static void normal_map_pixels(void *opaque, size_t begin, size_t end)
{
    const normal_map_context *ctx = opaque;
    for (size_t i = begin; i < end; ++i) {
        float x, y, z;
        int octahedral = ctx->mode == 2 && ctx->option_b == 2;
        if (ctx->mode == 1) {
            x = ctx->x[i] - ctx->mean_x;
            y = ctx->y[i] - ctx->mean_y;
            normalize(&x, &y, &z);
        } else {
            decode(ctx->image + i * ctx->channels,
                   ctx->mode == 2 && ctx->option_a == 2, &x, &y, &z);
        }
        if (ctx->mode == 2) {
            if (ctx->option_a == 1) y = -y;
            if (ctx->option_b == 1) y = -y;
        }
        float r, g;
        if (octahedral) {
            float sum = fabsf(x) + fabsf(y);
            sum += fabsf(z);
            x /= sum;
            y /= sum;
            r = x + y;
            g = x - y;
            z = 0.0f;
        } else {
            r = x;
            g = y;
        }
        r += 1.0f;
        g += 1.0f;
        if (ctx->mode == 0) {
            if (ctx->option_a == 1) z = (z + 1.0f) / 2.0f;
            else if (ctx->option_a >= 2) z = (float)(ctx->option_a - 2) * 0.5f;
        }
        ctx->out[i * 3] = z;
        ctx->out[i * 3 + 1] = g * 0.5f;
        ctx->out[i * 3 + 2] = r * 0.5f;
    }
}

/* mode 0: normalize with B channel option 0..4; mode 1: balance;
 * mode 2: convert from/to DirectX(0), OpenGL(1), Octahedral(2). */
CN_EXPORT cn_status cn_normal_map(const float *image, float *out, size_t count,
                                  size_t channels, int mode, int option_a, int option_b)
{
    if (channels < 3 || mode < 0 || mode > 2 || option_a < 0 || option_b < 0 ||
        (mode == 0 && (option_a > 4 || option_b != 0)) ||
        (mode == 1 && (option_a != 0 || option_b != 0)) ||
        (mode == 2 && (option_a > 2 || option_b > 2))) return CN_INVALID_ARGUMENT;
    if (count > SIZE_MAX / channels || count * channels > SIZE_MAX / sizeof(float))
        return CN_SIZE_OVERFLOW;
    size_t in_bytes = count * channels * sizeof(float);
    size_t out_bytes = count * 3 * sizeof(float);
    if (!count) return CN_OK;
    uintptr_t a = (uintptr_t)image, b = (uintptr_t)out;
    if (!image || !out || a % _Alignof(float) || b % _Alignof(float))
        return CN_INVALID_ARGUMENT;
    if (a > UINTPTR_MAX - in_bytes || b > UINTPTR_MAX - out_bytes)
        return CN_SIZE_OVERFLOW;
    if (a < b + out_bytes && b < a + in_bytes) return CN_INVALID_ARGUMENT;
    normal_map_context ctx = {image, out, NULL, NULL, channels, mode, option_a, option_b, 0, 0};
    cn_status status = CN_OK;
    if (mode == 1) {
        ctx.x = malloc(count * 2 * sizeof(float));
        if (!ctx.x) return CN_ALLOCATION_FAILED;
        ctx.y = ctx.x + count;
        status = cn_parallel_for(count, 65536, normal_map_decode_balance, &ctx);
        if (status == CN_OK) {
            /* np.mean: x and y are fresh contiguous arrays upstream, so one
             * pairwise tree (block 0). */
            ctx.mean_x = numpy_mean_f32(cn_numpy_sum_f32(ctx.x, count, 1, 0, 0), count);
            ctx.mean_y = numpy_mean_f32(cn_numpy_sum_f32(ctx.y, count, 1, 0, 0), count);
        }
    }
    if (status == CN_OK) status = cn_parallel_for(count, 65536, normal_map_pixels, &ctx);
    free(ctx.x);
    return status;
}
