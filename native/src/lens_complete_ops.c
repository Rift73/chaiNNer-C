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
#include "lens_fold_shared.h"
#include "lens_shared.h"
#include "parallel.h"
#include "run_kernels.h"
#include "scratch.h"
#include "numpy_exp_f32.h"
#include "numpy_trig_f32.h"
#include <limits.h>
#include <math.h>
#include <string.h>
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

/* The path is chosen once per span, and each path runs its own loop, in which it is
 * a constant. The CRT's powf is read into a local first: a store to out could alias
 * it, which would make the loop reload it per element. */
#define POWER_LOOP(path) \
    for (size_t i = 0; i < count; ++i) \
        out[i] = power_one(src[i], path, exponent, crt_powf, finish)

static void power_span(const float *const src, float *const out, const size_t count, const float exponent,
    const int finish)
{
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

/* An image from an 8-bit source holds a few hundred values, and powf is the first power's whole cost: a chunk keeps
 * the results of the values it has seen, by their bits, so each output is still powf's for its own input. Slot 0, the
 * only one the value 0 reaches, starts with its result, so the other slots' zero keys match nothing. */
#define POWER_MEMO_BITS 12

static void power_memo_span(const float *const src, float *const out, const size_t count, const float exponent)
{
    float (*const crt_powf)(float, float) = cn_crt.powf;
    uint32_t keys[1u << POWER_MEMO_BITS] = {0};
    float values[1u << POWER_MEMO_BITS];
    values[0] = crt_powf(0.0f, exponent);
    for (size_t i = 0; i < count; ++i) {
        uint32_t bits;
        memcpy(&bits, src + i, sizeof bits);
        const uint32_t slot = (bits * UINT32_C(2654435769)) >> (32 - POWER_MEMO_BITS);
        if (keys[slot] != bits) {
            keys[slot] = bits;
            values[slot] = crt_powf(src[i], exponent);
        }
        out[i] = values[slot];
    }
}

static void power_range(void *opaque, size_t begin, size_t end)
{
    const power_context *ctx = opaque;
    if (!ctx->finish && power_path_of(ctx->exponent) == EXPONENT_OTHER)
        power_memo_span(ctx->src + begin, ctx->out + begin, end - begin, ctx->exponent);
    else
        power_span(ctx->src + begin, ctx->out + begin, end - begin, ctx->exponent, ctx->finish);
}

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
#if CN_ISA_UNITS()
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

/* cn_lens_blur's chunks, at most this many: each one's rows are a band of the image, worked through whole. */
#define LENS_MAX_CHUNKS 128
/* A band's fewest rows: its seams recompute 2 * radius rows of the passes across. */
#define LENS_BAND_ROWS 32
/* Tap applications (image values * components * 6 * kernel length) below which a call stays on the caller. */
#define LENS_SERIAL_WORK ((size_t)1 << 24)

/* across and down: per component the real then the imaginary kernel's taps, along a row and down a column of the
 * band's rows. scratch: chunk_floats per chunk. */
typedef struct lens_job {
    const float *src, *weights;
    float *out, *scratch;
    const lens_taps *across, *down;
    size_t h, w, c, radius, components, chunks, chunk_floats, fused;
    float exponent;
    int finish;
    cn_isa_level isa;
} lens_job;

static size_t lens_reflect(ptrdiff_t position, size_t size)
{
    if (size == 1) return 0;
    ptrdiff_t period = (ptrdiff_t)(2 * (size - 1));
    position %= period;
    if (position < 0) position += period;
    return (size_t)(position < (ptrdiff_t)size ? position : period - position);
}

static void lens_fold(const lens_job *job, const float *src, const lens_taps *taps, float *out)
{
#if CN_ISA_UNITS()
    if (job->fused && job->isa >= CN_ISA_AVX512) {
        cn_lens_fold_avx512(src, taps, job->fused, job->w, out);
        return;
    }
    if (job->fused && job->isa >= CN_ISA_AVX2) {
        cn_lens_fold_avx2(src, taps, job->fused, job->w, out);
        return;
    }
#endif
    lens_fold_scalar(src, taps, 0, job->fused, job->w, out);
}

/* Band [first, last) of the output, one channel at a time. The channel's rows, the radius above and below included,
 * are copied with their reflected ends; per component they pass once across, into real and imaginary, and each output
 * row is the four passes down composed at once. Every value keeps its sequence: the taps in order, the components'
 * compositions in order, the finishing power. Scratch: the copies, the two across planes, one row of each of the four
 * passes down, the band's sum. */
static void lens_band(const lens_job *job, size_t first, size_t last, float *scratch)
{
    const size_t w = job->w, c = job->c, r = job->radius, stride = w + 2 * r, band = last - first, rows = band + 2 * r;
    float *const padded = scratch, *const real = padded + rows * stride, *const imaginary = real + rows * w;
    float *const down = imaginary + rows * w, *const sum = down + 4 * w;
    for (size_t channel = 0; channel < c; ++channel) {
        for (size_t i = 0; i < rows; ++i) {
            const float *line = job->src + lens_reflect((ptrdiff_t)(first + i) - (ptrdiff_t)r, job->h) * w * c + channel;
            float *row = padded + i * stride;
            for (size_t x = 0; x < r; ++x) {
                row[x] = line[lens_reflect((ptrdiff_t)x - (ptrdiff_t)r, w) * c];
                row[r + w + x] = line[lens_reflect((ptrdiff_t)(w + x), w) * c];
            }
            for (size_t x = 0; x < w; ++x) row[r + x] = line[x * c];
        }
        for (size_t k = 0; k < job->components; ++k) {
            const lens_taps *across = job->across + 2 * k, *columns = job->down + 2 * k;
            for (size_t i = 0; i < rows; ++i) {
                lens_fold(job, padded + i * stride + r, across, real + i * w);
                lens_fold(job, padded + i * stride + r, across + 1, imaginary + i * w);
            }
            compose_context compose = {down, down + w, down + 2 * w, down + 3 * w, NULL,
                job->weights[2 * k], job->weights[2 * k + 1], k != 0, job->isa};
            for (size_t y = 0; y < band; ++y) {
                lens_fold(job, real + (y + r) * w, columns, down);
                lens_fold(job, real + (y + r) * w, columns + 1, down + w);
                lens_fold(job, imaginary + (y + r) * w, columns, down + 2 * w);
                lens_fold(job, imaginary + (y + r) * w, columns + 1, down + 3 * w);
                compose.out = sum + y * w;
                compose_range(&compose, 0, w);
            }
        }
        if (job->finish) power_span(sum, sum, band * w, job->exponent, 1);
        float *out = job->out + first * w * c + channel;
        for (size_t i = 0; i < band * w; ++i) out[i * c] = sum[i];
    }
}

static void lens_chunks(void *opaque, size_t begin, size_t end)
{
    const lens_job *job = opaque;
    for (size_t chunk = begin; chunk < end; ++chunk)
        lens_band(job, job->h * chunk / job->chunks, job->h * (chunk + 1) / job->chunks,
            job->scratch + chunk * job->chunk_floats);
}

static cn_status lens_run(const float *src, float *out, size_t h, size_t w, size_t c, const float *kernels,
    const float *weights, size_t components, size_t length, size_t fused_lanes, float exponent, int finish)
{
    if (!src || !out || !kernels || !weights || !h || !w || !c || !components || !(length & 1) ||
        length > CN_LENS_MAX_TAPS || h > INT_MAX || w > INT_MAX || (fused_lanes != 0 && fused_lanes != 8))
        return CN_INVALID_ARGUMENT;
    if (h > SIZE_MAX / w || h * w > SIZE_MAX / sizeof(float) / c / 8) return CN_SIZE_OVERFLOW;
    size_t radius = length / 2, band = radius > LENS_BAND_ROWS ? radius : LENS_BAND_ROWS;
    size_t chunks = h / band + (h % band != 0);
    if (chunks > LENS_MAX_CHUNKS) chunks = LENS_MAX_CHUNKS;
    /* Every chunk's rows, its scratch's size: the largest band's copies, across planes, rows down and sum. */
    size_t largest = h / chunks + 1, rows = largest + 2 * radius;
    size_t chunk_floats = rows * (w + 2 * radius + 2 * w) + (4 + largest) * w;
    size_t total = 0, tap_offset, float_offset;
    if (!cn_scratch_carve(&total, 4 * components * sizeof(lens_taps), &tap_offset) ||
        !cn_scratch_carve(&total, chunks * chunk_floats * sizeof(float), &float_offset))
        return CN_ALLOCATION_FAILED;
    cn_scratch lease;
    cn_status status = cn_scratch_lease(total, &lease, NULL);
    if (status != CN_OK) return status;
    lens_taps *taps = (lens_taps *)((char *)lease.data + tap_offset);
    for (size_t i = 0; i < 2 * components; ++i) {
        lens_taps *across = taps + i, *down = taps + 2 * components + i;
        across->count = down->count = 0;
        for (size_t k = 0; k < length; ++k) {
            float value = kernels[(i / 2 * length + k) * 2 + i % 2];
            if (value == 0) continue;
            across->offsets[across->count] = (ptrdiff_t)k - (ptrdiff_t)radius;
            across->coefficients[across->count++] = value;
            down->offsets[down->count] = ((ptrdiff_t)k - (ptrdiff_t)radius) * (ptrdiff_t)w;
            down->coefficients[down->count++] = value;
        }
    }
    lens_job job = {src, weights, out, (float *)((char *)lease.data + float_offset), taps, taps + 2 * components,
        h, w, c, radius, components, chunks, chunk_floats, fused_lanes ? w / fused_lanes * fused_lanes : 0, exponent,
        finish, cn_isa_current()};
    size_t work = h * w * c * components * 6 * length;
    status = cn_parallel_for(chunks, work < LENS_SERIAL_WORK ? chunks : 1, lens_chunks, &job);
    cn_scratch_release(&lease);
    return status;
}

/* The convolutions, compositions and finishing power of Lens Blur on an interleaved image already raised to its
 * power, in one pass over the image: each output value equals, per channel and component, cn_convolution_spatial's
 * two passes across (kernels' real and imaginary) and four down, cn_lens_compose in component order (weights: a and b
 * per component), then cn_lens_power(exponent, finish 1). kernels: per component length complex coefficients, real
 * then imaginary. */
CN_EXPORT cn_status cn_lens_blur(const float *src, float *out, size_t h, size_t w, size_t c,
    const float *kernels, const float *weights, size_t components, size_t length, size_t fused_lanes, float exponent)
{
    return lens_run(src, out, h, w, c, kernels, weights, components, length, fused_lanes, exponent, 1);
}

/* cn_lens_blur without the finishing power: the sums, for cn_lens_straight. */
CN_EXPORT cn_status cn_lens_blur_sums(const float *src, float *out, size_t h, size_t w, size_t c,
    const float *kernels, const float *weights, size_t components, size_t length, size_t fused_lanes)
{
    return lens_run(src, out, h, w, c, kernels, weights, components, length, fused_lanes, 0.0f, 0);
}

typedef struct lens_straight_context {
    const float *sums;
    float *out;
    float exponent;
} lens_straight_context;

static void lens_straight_range(void *opaque, size_t begin, size_t end)
{
    const lens_straight_context *ctx = opaque;
    for (size_t p = begin; p < end; ++p) {
        const float *sums = ctx->sums + p * 4;
        float *out = ctx->out + p * 4;
        float alpha = sums[3];
        for (size_t c = 0; c < 3; ++c) {
            float value = alpha > 0.0f ? sums[c] / alpha : 0.0f;
            out[c] = value > 1.0f ? 1.0f : value;
        }
        out[3] = alpha;
    }
    power_span(ctx->out + begin * 4, ctx->out + begin * 4, (end - begin) * 4, ctx->exponent, 1);
    for (size_t p = begin; p < end; ++p) {
        float *out = ctx->out + p * 4;
        if (!(out[3] > 0.0f)) out[0] = out[1] = out[2] = 0.0f;
    }
}

/* The finish of Lens Blur on premultiplied RGBA from its sums: colour over alpha while both still hold the kernels'
 * overshoot, then the finishing power on both (alpha as cn_lens_power finishes it). A flat colour stays flat where alpha's
 * sum passes 1 and is clipped. Colour 0 where alpha ends at 0. */
CN_EXPORT cn_status cn_lens_straight(const float *sums, float *out, size_t pixels, float exponent)
{
    if (!sums || !out || !pixels) return CN_INVALID_ARGUMENT;
    if (pixels > SIZE_MAX / (4 * sizeof(float))) return CN_SIZE_OVERFLOW;
    lens_straight_context ctx = {sums, out, exponent};
    return cn_parallel_for(pixels, 65536, lens_straight_range, &ctx);
}
