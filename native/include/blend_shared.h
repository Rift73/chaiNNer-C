/* chaiNNer blend modes and alpha composition, ported from nodes/impl/blend.py and
 * moved from blend.c (SP4b Task 6). Shared by that baseline unit and the /arch:AVX2
 * unit blend_avx2.c: the job, and static helpers compiled into each including unit,
 * never inline, so the linker cannot fold a VEX copy into baseline callers. Every
 * including unit uses them (C4505): blend.c through mode_range and image_range,
 * blend_avx2.c through image_pixels.
 * SPDX-License-Identifier: GPL-3.0-only
 * Keep float32 operation order: upstream NumPy materializes each expression.
 */
#ifndef CHAINNER_BLEND_SHARED_H
#define CHAINNER_BLEND_SHARED_H
#include "chainner.h"
#include "isa.h"
#include "numeric.h"

#include <math.h>
#include <stdint.h>
#include <stddef.h>
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#define CN_BLEND_SSE 1
#endif

/* Unlike fminf/fmaxf, NumPy minimum/maximum propagate either NaN operand. */
static float minimum(float a, float b)
{
    if (isnan(a)) return a;
    if (isnan(b)) return b;
    return a < b ? a : b;
}

static float maximum(float a, float b)
{
    if (isnan(a)) return a;
    if (isnan(b)) return b;
    return a > b ? a : b;
}

/* np.clip(x, 0, 1) as the enforce clamp computes it (convert_shared.h clamp_unit,
 * NumPy 2.5.3 clip.cpp): x < 0 gives +0, then x > 1 gives 1, so -0 and a NaN's
 * payload pass; maxss(+0, v) then minss(1, v) also take a denormal as the zero of
 * its sign under DAZ, as NumPy's maxps/minps build does. minimum(maximum(x, 0), 1)
 * would give +0 for -0. */
static float clip(float value)
{
#ifdef CN_BLEND_SSE
    __m128 v = _mm_max_ss(_mm_setzero_ps(), _mm_set_ss(value));
    return _mm_cvtss_f32(_mm_min_ss(_mm_set_ss(1.0f), v));
#else
    value = value < 0.0f ? 0.0f : value;
    return value > 1.0f ? 1.0f : value;
#endif
}

/* NumPy round uses ties-to-even. Avoid dependence on the process rounding mode.
 * Inputs normally lie in [0,1]; wrapping also matches finite uint8 casts outside
 * that range. NumPy's invalid float-to-uint8 cast produces zero on our platform.
 */
static unsigned char to_uint8(float value)
{
    float scaled = value * 255.0f;
    if (!isfinite(scaled)) return 0;
    float lower = floorf(scaled);
    float fraction = scaled - lower;
    if (fraction > 0.5f || (fraction == 0.5f && fmodf(lower, 2.0f) != 0.0f))
        lower += 1.0f;
    float wrapped = fmodf(lower, 256.0f);
    if (wrapped < 0.0f) wrapped += 256.0f;
    return (unsigned char)wrapped;
}

static float color_burn(float a, float b)
{
    return a == 0.0f ? 0.0f
        : maximum(0.0f, 1.0f - ((1.0f - b) / maximum(0.0001f, a)));
}

static float color_dodge(float a, float b)
{
    return a == 1.0f ? 1.0f
        : minimum(1.0f, b / maximum(0.0001f, 1.0f - a));
}

static float apply_mode(float a, float b, int mode)
{
    switch (mode) {
    case 0: return a;
    case 1: return a * b;
    case 2: return minimum(a, b);
    case 3: return maximum(a, b);
    case 4: return a + b;
    case 5: return color_burn(a, b);
    case 6: return color_dodge(a, b);
    case 7:
        return a == 1.0f ? 1.0f
            : minimum(1.0f, (b * b) / maximum(0.0001f, 1.0f - a));
    case 8:
        return b == 1.0f ? 1.0f
            : minimum(1.0f, (a * a) / maximum(0.0001f, 1.0f - b));
    case 9:
        return b < 0.5f ? (2.0f * b) * a
            : 1.0f - ((2.0f * (1.0f - b)) * (1.0f - a));
    case 10: return fabsf(a - b);
    case 11: return 1.0f - fabsf((1.0f - b) - a);
    case 12: return (a + b) - (a * b);
    case 13: return (float)(to_uint8(a) ^ to_uint8(b)) / 255.0f;
    case 14: return b - a;
    case 15: return b / maximum(0.0001f, a);
    case 16: return (a * (1.0f - b)) + (b * (1.0f - a));
    case 17:
        return a <= 0.5f
            ? ((2.0f * b) * a) + ((b * b) * (1.0f - (2.0f * a)))
            : (sqrtf(b) * ((2.0f * a) - 1.0f))
                + ((2.0f * b) * (1.0f - a));
    case 18:
        return a <= 0.5f ? (2.0f * a) * b
            : 1.0f - ((2.0f * (1.0f - a)) * (1.0f - b));
    case 19:
        return a <= 0.5f ? color_burn(2.0f * a, b)
            : color_dodge(2.0f * (a - 0.5f), b);
    case 20: return (b + (2.0f * a)) - 1.0f;
    case 21: {
        float x = 2.0f * a;
        float y = x - 1.0f;
        return b < y ? y : (b > x ? x : b);
    }
    case 22: return (a + b) - 1.0f;
    default: return cn_f32_from_bits(CN_NPY_NANF_BITS); /* Entry points validate modes first. */
    }
}

static int normalized_mode(int mode)
{
    switch (mode) {
    case 0: case 1: case 2: case 3: case 9: case 10: case 11: case 12:
    case 13: case 16: case 17: case 18: case 21:
        return 1;
    default:
        return 0;
    }
}

/* isa: the level cn_blend_images read once for the call (cn_isa_current);
 * cn_blend_mode has no ISA path. */
typedef struct blend_job {
    const float *overlay, *base;
    float *output;
    int mode, overlay_channels, base_channels;
    cn_isa_level isa;
} blend_job;

/* Convert one channel without allocating expanded grayscale or opaque-alpha
 * images. RGB/BGR ordering is preserved; blending is channel-independent.
 */
static float channel(const float *image, size_t pixel, int channels, int c)
{
    if (c == 3 && channels < 4) return 1.0f;
    return image[pixel * (size_t)channels + (size_t)(channels == 1 ? 0 : c)];
}

/* cn_blend_images' pixels [begin, end): each output's operations in their order. */
static void image_pixels(const blend_job *job, size_t begin, size_t end)
{
    const float *overlay = job->overlay, *base = job->base;
    float *output = job->output;
    int overlay_channels = job->overlay_channels, base_channels = job->base_channels;
    int mode = job->mode;
    int target = overlay_channels > base_channels ? overlay_channels : base_channels;
    int needs_clipping = !normalized_mode(mode);

    for (size_t i = begin; i < end; ++i) {
        size_t offset = i * (size_t)target;
        if (target < 4) {
            for (int c = 0; c < target; ++c) {
                float value = apply_mode(channel(overlay, i, overlay_channels, c),
                    channel(base, i, base_channels, c), mode);
                output[offset + (size_t)c] = needs_clipping ? clip(value) : value;
            }
        } else if (base_channels < 4) {
            float alpha = overlay[i * 4 + 3];
            for (int c = 0; c < 3; ++c) {
                float b = channel(base, i, base_channels, c);
                float blended = apply_mode(overlay[i * 4 + (size_t)c], b, mode);
                float value = (alpha * blended) + ((1.0f - alpha) * b);
                output[offset + (size_t)c] = needs_clipping ? clip(value) : value;
            }
            output[offset + 3] = 1.0f;
        } else {
            float o_a = channel(overlay, i, overlay_channels, 3);
            float b_a = base[i * 4 + 3];
            float final_a = 1.0f - ((1.0f - o_a) * (1.0f - b_a));
            float blend_strength = o_a * b_a;
            float o_strength = o_a - blend_strength;
            float b_strength = b_a - blend_strength;
            float denominator = maximum(final_a, 0.0001f);
            for (int c = 0; c < 3; ++c) {
                float a = channel(overlay, i, overlay_channels, c);
                float b = base[i * 4 + (size_t)c];
                float blended = apply_mode(a, b, mode);
                float value = ((o_strength * a) + (b_strength * b))
                    + (blend_strength * blended);
                output[offset + (size_t)c] = clip(value / denominator);
            }
            output[offset + 3] = needs_clipping ? clip(final_a) : final_a;
        }
    }
}

/* blend_avx2.c: pixels [begin, end), inside one chunk, for ISA level avx2 or above.
 * NORMAL (mode 0) and MULTIPLY (mode 1) with equal channel counts below 4 (no clip,
 * one element per output) run 8 pixels at a time as image_pixels' operation: a copy
 * of the overlay, or overlay * base; the chunk's last pixels, and every other job,
 * run image_pixels. */
void cn_blend_images_avx2(const blend_job *job, size_t begin, size_t end);
#endif
