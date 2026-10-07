/* Altered C implementation of OpenCV 5.0.0 float color conversion contracts.
 * Reference: modules/imgproc/src/color_{rgb,yuv,hsv}.simd.hpp, color_lab.cpp and
 * color_rgb.dispatch.cpp, https://github.com/opencv/opencv/tree/5.0.0 (float
 * arithmetic unchanged from 4.8.0), and the measured float gray conversion of
 * its bundled IPP ICV 2026.0.0.
 * Those sources are Apache-2.0 licensed. The checked ABI and pool integration
 * are new GPL-3.0-only code. Row-local vector/tail arithmetic is intentional:
 * changing its association changes saved pixels at quantization boundaries.
 */
#include "chainner.h"
#include "cn_crt_math.h"
#include "parallel.h"
#include <float.h>
#include <limits.h>
#include <math.h>
#include <string.h>
#include "color_lab_tables.h"
#include "numpy_trig_f32.h"

typedef struct color_context {
    const float *src;
    float *dst;
    size_t width, scn, dcn;
    int op, blue, lanes, ipp;
} color_context;

/* SSE min/max select the second operand on ties and unordered comparisons. */
static float minimum(float a, float b) { return a < b ? a : b; }
static float maximum(float a, float b) { return a > b ? a : b; }
static float multiply_add(float a, float b, float c, int fused)
{
    return fused ? fmaf(a, b, c) : a * b + c;
}

static int truncate_int(float value)
{
    if (!(value >= -2147483648.0f && value < 2147483648.0f)) return INT_MIN;
    return (int)value;
}

static int scalar_sector(float *h)
{
    int sector = truncate_int(*h);
    /* cvFloor uses truncation followed by a comparison/subtraction; reproduce
     * its x86 integer-indefinite wrap for negative infinity without C UB. */
    if ((float)sector > *h) {
        uint32_t wrapped = (uint32_t)sector - UINT32_C(1);
        memcpy(&sector, &wrapped, sizeof(sector));
    }
    *h -= (float)sector;
    sector %= 6;
    return sector < 0 ? sector + 6 : sector;
}

static void extrema(float b, float g, float r, int vector, float *low, float *high)
{
    if (vector) {
        *low = minimum(minimum(r, g), b);
        *high = maximum(maximum(r, g), b);
    } else {
        *low = *high = r;
        if (*high < g) *high = g;
        if (*high < b) *high = b;
        if (*low > g) *low = g;
        if (*low > b) *low = b;
    }
}

static void rgb_hue(const color_context *ctx, const float *src, float *dst, int vector)
{
    float b = src[ctx->blue], g = src[1], r = src[ctx->blue ^ 2];
    float low, high;
    extrema(b, g, r, vector, &low, &high);
    float diff = high - low, h = 0.0f, s = 0.0f;
    int hls = ctx->op == 6, fused = vector && ctx->lanes == 8;
    float l = (high + low) * 0.5f;
    if (!hls || diff > FLT_EPSILON) {
        s = hls ? (l < 0.5f ? diff / (high + low) :
            diff / (vector ? 2.0f - (high + low) : (2.0f - high) - low)) :
            diff / (fabsf(high) + FLT_EPSILON);
        float reciprocal = hls ? 60.0f / diff : vector ?
            60.0f / (diff + FLT_EPSILON) : (float)(60.0 / (double)(diff + FLT_EPSILON));
        float delta = high == r ? g - b : high == g ? b - r : r - g;
        float base = high == r ? (vector && g < b ? 360.0f : 0.0f) :
            high == g ? 120.0f : 240.0f;
        h = !vector && high == r ? delta * reciprocal : multiply_add(delta, reciprocal, base, fused);
        if (!vector && h < 0.0f) h += 360.0f;
    }
    dst[0] = h;
    dst[1] = hls ? l : s;
    dst[2] = hls ? s : high;
}

static void hue_rgb(const color_context *ctx, const float *src, float *dst, int vector)
{
    static const int sectors[6][3] = {{1,3,0},{1,0,2},{3,0,1},{0,2,1},{0,1,3},{2,1,0}};
    int hls = ctx->op == 7;
    float h = src[0], s = src[hls ? 2 : 1], v = src[hls ? 1 : 2];
    float b, g, r;
    if (!vector && s == 0.0f) b = g = r = v;
    else {
        h *= 6.0f / 360.0f;
        float t[4];
        int sector;
        if (vector) {
            float pre = (float)truncate_int(h), original = h;
            h -= pre;
            float sector_value = pre - (float)truncate_int((hls ? original : pre) * (1.0f / 6.0f)) * 6.0f;
            if (hls) {
                float ls = v * s, element = v <= 0.5f ? ls : s - ls;
                float twice = h + h;
                t[0] = v + element; t[1] = v - element;
                t[2] = (v + element) - element * twice;
                t[3] = (v - element) + element * twice;
                b = sector_value < 2.0f ? t[1] : sector_value <= 2.0f ? t[3] : sector_value <= 4.0f ? t[0] : t[2];
                g = sector_value < 1.0f ? t[3] : sector_value <= 2.0f ? t[0] : sector_value < 4.0f ? t[2] : t[1];
                r = sector_value < 1.0f ? t[0] : sector_value < 2.0f ? t[2] : sector_value < 4.0f ? t[1] : sector_value <= 4.0f ? t[3] : t[0];
            } else {
                t[0] = v; t[1] = v * (1.0f - s);
                t[2] = v * (1.0f - s * h); t[3] = v * (1.0f - s * (1.0f - h));
                b = sector_value < 2.0f ? t[1] : sector_value == 2.0f ? t[3] : sector_value == 3.0f || sector_value == 4.0f ? t[0] : sector_value > 4.0f ? t[2] : 0.0f;
                g = sector_value < 1.0f ? t[3] : sector_value == 1.0f || sector_value == 2.0f ? t[0] : sector_value == 3.0f ? t[2] : sector_value > 3.0f ? t[1] : s;
                r = sector_value < 1.0f ? t[0] : sector_value == 1.0f ? t[2] : sector_value == 2.0f || sector_value == 3.0f ? t[1] : sector_value == 4.0f ? t[3] : v;
            }
        } else {
            sector = scalar_sector(&h);
            if (hls) {
                float p2 = v <= 0.5f ? v * (1.0f + s) : v + s - v * s;
                float p1 = 2.0f * v - p2;
                t[0] = p2; t[1] = p1;
                t[2] = p1 + (p2 - p1) * (1.0f - h);
                t[3] = p1 + (p2 - p1) * h;
            } else {
                t[0] = v; t[1] = v * (1.0f - s);
                t[2] = v * (1.0f - s * h); t[3] = v * (1.0f - s * (1.0f - h));
            }
            b = t[sectors[sector][0]]; g = t[sectors[sector][1]]; r = t[sectors[sector][2]];
        }
    }
    dst[ctx->blue] = b; dst[1] = g; dst[ctx->blue ^ 2] = r;
    if (ctx->dcn == 4) dst[3] = 1.0f;
}

static float lab_clip(float value)
{
    return value < 0.0f ? 0.0f : value <= 1.0f ? value : 1.0f;
}

static void rgb_lab(const color_context *ctx, const float *src, float *dst, int vector)
{
    int coordinate[3], origin[3], fraction[3];
    for (int k = 0; k < 3; ++k) {
        float value = src[k == 1 ? 1 : (k == 0 ? ctx->blue : ctx->blue ^ 2)];
        value = vector ? minimum(maximum(value, 0.0f), 1.0f) : lab_clip(value);
        coordinate[k] = (int)nearbyintf(value * 16384.0f);
        origin[k] = coordinate[k] >> 9;
        fraction[k] = (coordinate[k] >> 5) & 15;
    }
    int accum[3] = {0, 0, 0};
    for (int i = 0; i < 8; ++i) {
        int p = origin[0] + ((i >> 2) & 1), q = origin[1] + ((i >> 1) & 1), r = origin[2] + (i & 1);
        if (p > 32) p = 32;
        if (q > 32) q = 32;
        if (r > 32) r = 32;
        int weight = ((i & 4) ? fraction[0] : 16 - fraction[0]) *
            ((i & 2) ? fraction[1] : 16 - fraction[1]) *
            ((i & 1) ? fraction[2] : 16 - fraction[2]);
        const int16_t *entry = lab_lattice + ((r * 33 + q) * 33 + p) * 3;
        for (int k = 0; k < 3; ++k) accum[k] += entry[k] * weight;
    }
    dst[0] = (float)((accum[0] + 2048) >> 12) * (100.0f / 16384.0f);
    dst[1] = (float)((accum[1] + 2048) >> 12) * (256.0f / 16384.0f) - 128.0f;
    dst[2] = (float)((accum[2] + 2048) >> 12) * (256.0f / 16384.0f) - 128.0f;
}

static float inverse_gamma(float value, int fused)
{
    value *= 1024.0f;
    int index = truncate_int(value);
    if (index < 0) index = 0;
    if (index > 1023) index = 1023;
    value -= (float)index;
    const float *table = lab_inverse_gamma + index * 4;
    float result = multiply_add(table[3], value, table[2], fused);
    result = multiply_add(result, value, table[1], fused);
    return multiply_add(result, value, table[0], fused);
}

static void lab_rgb(const color_context *ctx, const float *src, float *dst, int vector)
{
    int fused = vector && ctx->lanes == 8;
    float l = src[0], a = src[1], b = src[2], y, fy;
    if (l <= 8.0f) {
        y = vector ? l * (1.0f / 903.3f) : l / 903.3f;
        fy = multiply_add(7.787f, y, 16.0f / 116.0f, fused);
    } else {
        fy = vector ? (l + 16.0f) * (1.0f / 116.0f) : (l + 16.0f) / 116.0f;
        y = fy * fy * fy;
    }
    float x = vector ? multiply_add(a, 1.0f / 500.0f, fy, fused) : a / 500.0f + fy;
    float z = vector ? multiply_add(b, -1.0f / 200.0f, fy, fused) : fy - b / 200.0f;
    if (x <= 6.0f / 29.0f) x = vector ? (x - 16.0f / 116.0f) * (1.0f / 7.787f) : (x - 16.0f / 116.0f) / 7.787f;
    else x = x * x * x;
    if (z <= 6.0f / 29.0f) z = vector ? (z - 16.0f / 116.0f) * (1.0f / 7.787f) : (z - 16.0f / 116.0f) / 7.787f;
    else z = z * z * z;
    for (int k = 0; k < 3; ++k) {
        const float *c = lab_xyz_to_rgb + 3 * k;
        float result = vector ? multiply_add(c[0], x, multiply_add(c[1], y, c[2] * z, fused), fused) : (c[0] * x + c[1] * y) + c[2] * z;
        result = vector ? maximum(0.0f, minimum(result, 1.0f)) : lab_clip(result);
        dst[k == 1 ? 1 : (k == 0 ? ctx->blue ^ 2 : ctx->blue)] = inverse_gamma(result, fused);
    }
    if (ctx->dcn == 4) dst[3] = 1.0f;
}

static void lab_lch(const color_context *ctx, const float *src, float *dst)
{
    int inverse = ctx->op == 13;
    float a = src[1] - 0.5f, b = src[0] - 0.5f;
    float angle = inverse ? b * 0x1.921fb6p+2f : cn_crt.atan2f(b, a);
    float sine = cn_numpy_trig_f32(angle, 0, ctx->ipp);
    float cosine = cn_numpy_trig_f32(angle, 1, ctx->ipp);
    float upper = isnan(sine) ? fabsf(sine) : isnan(cosine) ? fabsf(cosine) : maximum(fabsf(sine), fabsf(cosine));
    float cmax = 0.5f / upper;
    if (inverse) {
        float chroma = src[1] * cmax;
        dst[0] = chroma * sine + 0.5f;
        dst[1] = chroma * cosine + 0.5f;
    } else {
        dst[0] = angle / 0x1.921fb6p+2f + 0.5f;
        dst[1] = cn_crt.hypotf(a, b) / cmax;
    }
    dst[2] = src[2];
}

static void color_range(void *raw, size_t begin, size_t end)
{
    const color_context *ctx = raw;
    size_t vector_end = ctx->width / (size_t)ctx->lanes * (size_t)ctx->lanes;
    if (ctx->op == 10) vector_end = (ctx->width - 1) / ((size_t)ctx->lanes * 2) * ((size_t)ctx->lanes * 2);
    if (ctx->op == 11) vector_end = ctx->width / ((size_t)ctx->lanes * 2) * ((size_t)ctx->lanes * 2);
    for (size_t y = begin; y < end; ++y) {
        for (size_t x = 0; x < ctx->width; ++x) {
            const float *s = ctx->src + (y * ctx->width + x) * ctx->scn;
            float *d = ctx->dst + (y * ctx->width + x) * ctx->dcn;
            int vector = x < vector_end, fused = vector && ctx->lanes == 8;
            if (ctx->op == 0) {
                if (ctx->scn == 1) d[0] = d[1] = d[2] = s[0];
                else {
                    d[0] = s[ctx->blue]; d[1] = s[1]; d[2] = s[ctx->blue ^ 2];
                }
                if (ctx->dcn == 4) d[3] = ctx->scn == 4 ? s[3] : 1.0f;
            } else if (ctx->op == 1) {
                float cb = ctx->blue == 0 ? 0.114f : 0.299f;
                float cr = ctx->blue == 0 ? 0.299f : 0.114f;
                /* IPP 2026.0.0 (l9 and k0) three-channel rows: 8-pixel vectors,
                   then one 4-pixel block whose even pixels sum blue first. */
                size_t block = ctx->width - ctx->width % 8;
                int ipp_even = ctx->scn == 3 && ctx->width % 8 >= 4 &&
                    x >= block && x < block + 4 && !((x - block) & 1);
                if (ctx->ipp && ipp_even) d[0] = fmaf(s[2], cr, fmaf(s[1], 0.587f, s[0] * cb));
                else if (ctx->ipp) d[0] = fmaf(s[2], cr, fmaf(s[0], cb, s[1] * 0.587f));
                else if (vector) d[0] = multiply_add(s[2], cr, multiply_add(s[1], 0.587f, s[0] * cb, fused), fused);
                else d[0] = (s[0] * cb + s[1] * 0.587f) + s[2] * cr;
            } else if (ctx->op == 2 || ctx->op == 8) {
                float c0 = ctx->blue == 0 ? 0.114f : 0.299f, c2 = ctx->blue == 0 ? 0.299f : 0.114f;
                float yv = vector ? multiply_add(s[0], c0, multiply_add(s[1], 0.587f, s[2] * c2, fused), fused) : (s[0] * c0 + s[1] * 0.587f) + s[2] * c2;
                float cr = multiply_add(s[ctx->blue ^ 2] - yv, ctx->op == 8 ? 0.713f : 0.877f, 0.5f, fused);
                float cb = multiply_add(s[ctx->blue] - yv, ctx->op == 8 ? 0.564f : 0.492f, 0.5f, fused);
                d[0] = yv; d[1] = ctx->op == 8 ? cr : cb; d[2] = ctx->op == 8 ? cb : cr;
            } else if (ctx->op == 3 || ctx->op == 9) {
                float yv = s[0], cr = s[ctx->op == 9 ? 1 : 2] - 0.5f, cb = s[ctx->op == 9 ? 2 : 1] - 0.5f;
                float c0 = ctx->op == 9 ? 1.403f : 1.140f, c1 = ctx->op == 9 ? -0.714f : -0.581f;
                float c2 = ctx->op == 9 ? -0.344f : -0.395f, c3 = ctx->op == 9 ? 1.773f : 2.032f;
                d[ctx->blue] = multiply_add(cb, c3, yv, fused);
                d[1] = multiply_add(cr, c1, multiply_add(cb, c2, yv, fused), fused);
                d[ctx->blue ^ 2] = multiply_add(cr, c0, yv, fused);
                if (ctx->dcn == 4) d[3] = 1.0f;
            } else if (ctx->op == 4 || ctx->op == 6) rgb_hue(ctx, s, d, vector);
            else if (ctx->op == 10) rgb_lab(ctx, s, d, vector);
            else if (ctx->op == 11) lab_rgb(ctx, s, d, vector);
            else if (ctx->op == 12 || ctx->op == 13) lab_lch(ctx, s, d);
            else if (ctx->op == 14) {
                float hsv[3] = {s[0] * 360.0f, 1.0f, 1.0f};
                hue_rgb(ctx, hsv, d, vector);
            }
            else hue_rgb(ctx, s, d, vector);
        }
    }
}

CN_EXPORT cn_status cn_color_convert_f32(const float *src, float *dst,
    size_t height, size_t width, size_t scn, size_t dcn, int op, int blue, int lanes, int ipp)
{
    if (!src || !dst || !height || !width || (scn != 1 && scn != 3 && scn != 4) ||
        (dcn != 1 && dcn != 3 && dcn != 4) || op < 0 || op > 14 ||
        (blue != 0 && blue != 2) || (lanes != 4 && lanes != 8) || (ipp != 0 && ipp != 1)) return CN_INVALID_ARGUMENT;
    if ((op == 0 && dcn == 1) || (op == 1 && (scn == 1 || dcn != 1)) ||
        (op > 1 && op != 14 && (scn == 1 || dcn == 1)) ||
        ((op == 2 || op == 4 || op == 6 || op == 8 || op == 10) && dcn != 3) ||
        ((op == 3 || op == 5 || op == 7 || op == 9 || op == 11) && scn != 3) ||
        ((op == 12 || op == 13) && (scn != 3 || dcn != 3)) ||
        (op == 14 && (scn != 1 || dcn != 3))) return CN_INVALID_ARGUMENT;
    if (height > SIZE_MAX / width) return CN_SIZE_OVERFLOW;
    size_t pixels = height * width;
    if (pixels > SIZE_MAX / sizeof(float) / scn || pixels > SIZE_MAX / sizeof(float) / dcn) return CN_SIZE_OVERFLOW;
    size_t input_bytes = pixels * scn * sizeof(float), output_bytes = pixels * dcn * sizeof(float);
    uintptr_t a = (uintptr_t)src, b = (uintptr_t)dst;
    if (a % _Alignof(float) || b % _Alignof(float)) return CN_INVALID_ARGUMENT;
    if (a > UINTPTR_MAX - input_bytes || b > UINTPTR_MAX - output_bytes) return CN_SIZE_OVERFLOW;
    if (a < b + output_bytes && b < a + input_bytes) return CN_INVALID_ARGUMENT;
    color_context ctx = {src, dst, width, scn, dcn, op, blue, lanes, ipp};
    return cn_parallel_for(height, 8, color_range, &ctx);
}
