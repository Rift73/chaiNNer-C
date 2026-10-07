/* Lens kernels reproduce NumPy 2.5.3 float32 operation order. The exp and
 * sin/cos polynomials below are C adaptations of NumPy's
 * loops_exponent_log.dispatch.c.src, npy_simd_data.h, and
 * loops_trigonometric.dispatch.c.src.
 *
 * Copyright (c) 2005-2022, NumPy Developers. All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * * Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of the NumPy Developers nor the names of any contributors
 *   may be used to endorse or promote products derived from this software
 *   without specific prior written permission.
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#include "chainner.h"
#include "cn_crt_math.h"
#include "lens_shared.h"
#include "parallel.h"
#include "numpy_exp_f32.h"
#include "numpy_trig_f32.h"
#include <math.h>
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#define CN_LENS_SSE 1
#endif

typedef struct kernel_context {
    float *out;
    size_t radius;
    float scale, reciprocal, a, b;
    int use_fma;
} kernel_context;

static void kernel_range(void *opaque, size_t begin, size_t end)
{
    const kernel_context *ctx = opaque;
    for (size_t i = begin; i < end; ++i) {
        float axis = (float)((int64_t)i - (int64_t)ctx->radius);
        axis *= ctx->scale;
        axis *= ctx->reciprocal;
        float square = axis * axis;
        float decay = cn_numpy_exp_f32(-ctx->a * square, ctx->use_fma);
        float angle = ctx->b * square;
        ctx->out[2 * i] = decay * cn_numpy_trig_f32(angle, 1, ctx->use_fma);
        ctx->out[2 * i + 1] = decay * cn_numpy_trig_f32(angle, 0, ctx->use_fma);
    }
}

CN_EXPORT cn_status cn_lens_kernel(float *out, size_t radius,
    float scale, float a, float b, int use_fma)
{
    if (!out || !radius || radius > 1000 || (use_fma != 0 && use_fma != 1))
        return CN_INVALID_ARGUMENT;
    kernel_context ctx = {out, radius, scale, (float)(1.0 / (double)radius), a, b, use_fma};
    return cn_parallel_for(2 * radius + 1, 65536, kernel_range, &ctx);
}

typedef struct power_context {
    const float *src; float *out; float exponent; int finish;
} power_context;

/* NumPy 2.5.3 FLOAT_power (loops_umath_fp.dispatch.c.src) with a scalar exponent:
 * fast paths for -1, 0, 0.5, 1 and 2, else powf. */
typedef enum power_path {
    EXPONENT_MINUS_ONE, EXPONENT_ZERO, EXPONENT_HALF, EXPONENT_ONE, EXPONENT_TWO,
    EXPONENT_OTHER
} power_path;

/* The exponent's path, tested in FLOAT_power's order: -0 takes the 0 path and NaN,
 * failing every test, takes powf. */
static power_path power_path_of(float exponent)
{
    if (exponent == -1.0f) return EXPONENT_MINUS_ONE;
    if (exponent == 0.0f) return EXPONENT_ZERO;
    if (exponent == 0.5f) return EXPONENT_HALF;
    if (exponent == 1.0f) return EXPONENT_ONE;
    if (exponent == 2.0f) return EXPONENT_TWO;
    return EXPONENT_OTHER;
}

/* One element of Lens Blur's power on its exponent's path. With finish, the source
 * is clamped below as B3 executes it, maxss(+0, x) with the constant first (B3
 * 0x180041141), not as its C text (x <= 0 -> 0): -0 stays -0, NaN passes, and under
 * DAZ a denormal comes back as the zero of its sign. Only an exponent whose powf
 * differs on +-0 after the clamp below shows the difference, e.g. -1
 * (test_forced_paths.py, test_lens_power_keeps_b3s_input_clamp). Then the path's
 * FLOAT_power operation, and a result <= 0 becomes +0 and one >= 1 becomes 1, NaN
 * passing (B3's comiss and minss give the same). */
static float power_one(float x, power_path path, float exponent,
    float (*crt_powf)(float, float), int finish)
{
    if (finish) {
#ifdef CN_LENS_SSE
        x = _mm_cvtss_f32(_mm_max_ss(_mm_setzero_ps(), _mm_set_ss(x)));
#else
        x = 0.0f > x ? 0.0f : x;
#endif
    }
    float value;
    switch (path) {
    case EXPONENT_MINUS_ONE: value = (float)(1.0 / (double)x); break;
    case EXPONENT_ZERO: value = 1.0f; break;
    case EXPONENT_HALF: value = sqrtf(x); break;
    case EXPONENT_ONE: value = x; break;
    case EXPONENT_TWO: value = x * x; break;
    default: value = crt_powf(x, exponent); break;
    }
    if (finish) {
        if (value <= 0.0f) value = 0.0f;
        else if (value >= 1.0f) value = 1.0f;
    }
    return value;
}

/* The path is chosen once per range, and each path runs its own loop, in which it is
 * a constant. The context's fields and the CRT's powf are read into locals first: a
 * store to out could alias them, which would make the loop reload them per element. */
#define POWER_LOOP(path) \
    for (size_t i = begin; i < end; ++i) \
        out[i] = power_one(src[i], path, exponent, crt_powf, finish)

static void power_range(void *opaque, size_t begin, size_t end)
{
    const power_context *ctx = opaque;
    const float *const src = ctx->src;
    float *const out = ctx->out;
    const float exponent = ctx->exponent;
    const int finish = ctx->finish;
    float (*const crt_powf)(float, float) = cn_crt.powf;
    switch (power_path_of(exponent)) {
    case EXPONENT_MINUS_ONE: POWER_LOOP(EXPONENT_MINUS_ONE); break;
    case EXPONENT_ZERO: POWER_LOOP(EXPONENT_ZERO); break;
    case EXPONENT_HALF: POWER_LOOP(EXPONENT_HALF); break;
    case EXPONENT_ONE: POWER_LOOP(EXPONENT_ONE); break;
    case EXPONENT_TWO: POWER_LOOP(EXPONENT_TWO); break;
    default: POWER_LOOP(EXPONENT_OTHER); break;
    }
}
#undef POWER_LOOP

CN_EXPORT cn_status cn_lens_power(const float *src, float *out, size_t count,
    float exponent, int finish)
{
    if (!src || !out || (finish != 0 && finish != 1)) return CN_INVALID_ARGUMENT;
    if (count > SIZE_MAX / sizeof(float)) return CN_SIZE_OVERFLOW;
    power_context ctx = {src, out, exponent, finish};
    return cn_parallel_for(count, 65536, power_range, &ctx);
}

static void compose_range(void *opaque, size_t begin, size_t end)
{
    const compose_context *ctx = opaque;
#if defined(_MSC_VER) && defined(_M_X64)
    if (ctx->isa >= CN_ISA_AVX2) {
        cn_lens_compose_avx2(ctx, begin, end);
        return;
    }
#endif
    for (size_t i = begin; i < end; ++i) ctx->out[i] = compose_one(ctx, i);
}

CN_EXPORT cn_status cn_lens_compose(const float *f1, const float *f2,
    const float *f3, const float *f4, float *out, size_t count,
    float a, float b, int accumulate)
{
    if (!f1 || !f2 || !f3 || !f4 || !out || (accumulate != 0 && accumulate != 1))
        return CN_INVALID_ARGUMENT;
    if (count > SIZE_MAX / sizeof(float)) return CN_SIZE_OVERFLOW;
    compose_context ctx = {f1, f2, f3, f4, out, a, b, accumulate, cn_isa_current()};
    return cn_parallel_for(count, 65536, compose_range, &ctx);
}
