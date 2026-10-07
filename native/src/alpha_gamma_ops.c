/* C adaptation of chaiNNer-rs gamma/fill-alpha numerical behavior.
 * Source: https://github.com/chaiNNer-org/chaiNNer-rs/tree/6f6ead6064f81b4049d3deb803c9279c7950736b
 * Files: crates/image-ops/src/{gamma,fill_alpha}.rs and util/grid.rs.
 * Altered implementation: scalar expressions reproduce the AVX lane operations;
 * an owned, simultaneous frontier replaces the color-extension grid traversal.
 * The surrounding C ABI/validation/threadpool integration is GPL-3.0-only.
 *
 * MIT License
 * Copyright (c) 2023 Michael Schmidt
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Polynomial exp/log originally from avx2_mathfun (altered scalar C version):
 * https://github.com/yuyichao/avx2_mathfun/blob/3c48718fd7fa4f427906e59d43e7cc1ef69cc276/avx2_mathfun.h
 * Based on "sse_mathfun.h", by Julien Pommier
 * http://gruntthepeon.free.fr/ssemath/
 * Copyright (C) 2012 Giovanni Garberoglio
 * Interdisciplinary Laboratory for Computational Science (LISC)
 * Fondazione Bruno Kessler and University of Trento
 * via Sommarive, 18
 * I-38123 Trento (Italy)
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 * 1. The origin of this software must not be misrepresented; you must not
 * claim that you wrote the original software. If you use this software
 * in a product, an acknowledgment in the product documentation would be
 * appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 * misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */
#include "chainner.h"
#include "chainner_ext_kernels.h"
#include "parallel.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#if defined(_WIN32)
#include <windows.h>
#endif

CN_EXPORT int cn_gamma_uses_approximation(void)
{
#if defined(_WIN32) && (defined(_M_X64) || defined(_M_IX86))
    return IsProcessorFeaturePresent(PF_AVX2_INSTRUCTIONS_AVAILABLE) != 0;
#elif (defined(__i386__) || defined(__x86_64__)) && defined(__GNUC__)
    return __builtin_cpu_supports("avx2") != 0;
#else
    return 0;
#endif
}

static float float_bits(uint32_t bits)
{
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint32_t to_bits(float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float gamma_log(float x)
{
    int invalid = x <= 0.0f;
    /* SSE max/min select the second operand on unordered or equal inputs. */
    float minimum = float_bits(UINT32_C(0x00800000));
    x = x > minimum ? x : minimum;
    uint32_t bits = to_bits(x);
    int32_t exponent = (int32_t)(bits >> 23) - 127;
    x = float_bits((bits & ~UINT32_C(0x7f800000)) | UINT32_C(0x3f000000));
    float e = (float)exponent + 1.0f;
    float temporary = x < 0.707106781186547524f ? x : 0.0f;
    e -= x < 0.707106781186547524f ? 1.0f : 0.0f;
    x -= 1.0f;
    x += temporary;
    float z = x * x;
    float y = 7.0376836292E-2f * x;
    y += -1.1514610310E-1f;
    y *= x; y += 1.1676998740E-1f;
    y *= x; y += -1.2420140846E-1f;
    y *= x; y += 1.4249322787E-1f;
    y *= x; y += -1.6668057665E-1f;
    y *= x; y += 2.0000714765E-1f;
    y *= x; y += -2.4999993993E-1f;
    y *= x; y += 3.3333331174E-1f;
    y *= x;
    y *= z;
    y += e * -2.12194440e-4f;
    y -= z * 0.5f;
    x += y;
    x += e * 0.693359375f;
    return invalid ? float_bits(UINT32_MAX) : x;
}

static float gamma_exp(float x)
{
    x = x < 88.3762626647949f ? x : 88.3762626647949f;
    x = x > -88.3762626647949f ? x : -88.3762626647949f;
    float fx = x * 1.44269504088896341f;
    fx += 0.5f;
    float temporary = floorf(fx);
    fx = temporary - (temporary > fx ? 1.0f : 0.0f);
    float z = fx * -2.12194440e-4f;
    x -= fx * 0.693359375f;
    x -= z;
    z = x * x;
    float y = 1.9875691500E-4f * x;
    y += 1.3981999507E-3f;
    y *= x; y += 8.3334519073E-3f;
    y *= x; y += 4.1665795894E-2f;
    y *= x; y += 1.6666665459E-1f;
    y *= x; y += 5.0000001201E-1f;
    y *= z;
    y += x;
    y += 1.0f;
    uint32_t exponent = (uint32_t)((int32_t)fx + 127) << 23;
    return y * float_bits(exponent);
}

static float gamma_approx(float x, float gamma)
{
    float result = gamma_exp(gamma_log(x) * gamma);
    if (x == 0.0f) result = 0.0f;
    result = result > 0.0f ? result : 0.0f;
    return result < 1.0f ? result : 1.0f;
}

typedef struct gamma_context {
    const float *src;
    float *out;
    size_t channels, approximate_count;
    float gamma;
} gamma_context;

static void gamma_range(void *opaque, size_t begin, size_t end)
{
    const gamma_context *ctx = opaque;
    for (size_t i = begin; i < end; ++i) {
        if (ctx->channels == 4 && i % 4 == 3) ctx->out[i] = ctx->src[i];
        else if (i < ctx->approximate_count) ctx->out[i] = gamma_approx(ctx->src[i], ctx->gamma);
        else ctx->out[i] = powf(ctx->src[i], ctx->gamma);
    }
}

CN_EXPORT cn_status cn_alpha_gamma(const float *src, float *out,
    size_t pixels, size_t channels, float gamma)
{
    uintptr_t a = (uintptr_t)src, b = (uintptr_t)out;
    if (!src || !out || !channels || a % _Alignof(float) || b % _Alignof(float))
        return CN_INVALID_ARGUMENT;
    if (channels > SIZE_MAX / sizeof(float) ||
        pixels > SIZE_MAX / sizeof(float) / channels) return CN_SIZE_OVERFLOW;
    size_t count = pixels * channels;
    size_t bytes = count * sizeof(float);
    if (a > UINTPTR_MAX - bytes || b > UINTPTR_MAX - bytes) return CN_SIZE_OVERFLOW;
    /* Resize applies the final gamma in place. Every output reads only its
     * corresponding input, so exact aliasing is safe; shifted overlap is not. */
    if (a != b && a < b + bytes && b < a + bytes) return CN_INVALID_ARGUMENT;
    /* Rust chunks at 8192 elements, divisible by eight. Only the final chunk
     * has a scalar tail. RGBA takes its separate all-scalar RGB path. */
    size_t approximate = channels != 4 && cn_gamma_uses_approximation() ? count - count % 8 : 0;
    gamma_context ctx = {src, out, channels, approximate, gamma};
    return cn_parallel_for(count, 65536, gamma_range, &ctx);
}

typedef struct fill_context {
    float *out, *pending;
    const size_t *queue;
    size_t width, height;
} fill_context;

static void fill_values(void *opaque, size_t begin, size_t end)
{
    const fill_context *ctx = opaque;
    for (size_t k = begin; k < end; ++k) {
        size_t p = ctx->queue[k], x = p % ctx->width, y = p / ctx->width;
        size_t cell_x = x / 8, cell_y = y / 8;
        size_t last_x = (ctx->width - 1) / 8, last_y = (ctx->height - 1) / 8;
        int inner = cell_x > 0 && cell_x < last_x && cell_y > 0 && cell_y < last_y;
        float sum[4] = {0};
        for (size_t c = 0; c < 4; ++c) {
            if (inner) {
                sum[c] = ctx->out[(p - 1) * 4 + c] + ctx->out[(p + 1) * 4 + c];
                sum[c] += ctx->out[(p - ctx->width) * 4 + c];
                sum[c] += ctx->out[(p + ctx->width) * 4 + c];
            } else {
                if (x > 0) sum[c] += ctx->out[(p - 1) * 4 + c];
                if (x + 1 < ctx->width) sum[c] += ctx->out[(p + 1) * 4 + c];
                if (y > 0) sum[c] += ctx->out[(p - ctx->width) * 4 + c];
                if (y + 1 < ctx->height) sum[c] += ctx->out[(p + ctx->width) * 4 + c];
            }
        }
        for (size_t c = 0; c < 4; ++c) ctx->pending[k * 4 + c] = sum[c] / sum[3];
    }
}

static void fill_commit(void *opaque, size_t begin, size_t end)
{
    const fill_context *ctx = opaque;
    for (size_t k = begin; k < end; ++k)
        memcpy(ctx->out + ctx->queue[k] * 4, ctx->pending + k * 4, 4 * sizeof(float));
}

static void enqueue(size_t p, unsigned char *state, size_t *queue, size_t *count)
{
    if (!state[p]) { state[p] = 2; queue[(*count)++] = p; }
}

CN_EXPORT cn_status cn_alpha_extend(const float *src, float *out,
    size_t height, size_t width, float threshold, size_t iterations)
{
    if (!src || !out) return CN_INVALID_ARGUMENT;
    if (width && height > SIZE_MAX / width) return CN_SIZE_OVERFLOW;
    size_t pixels = height * width;
    if (pixels > SIZE_MAX / (4 * sizeof(float)) || pixels > SIZE_MAX / sizeof(size_t))
        return CN_SIZE_OVERFLOW;
    if (!pixels) return CN_OK;
    for (size_t p = 0; p < pixels; ++p) {
        float flag = src[p * 4 + 3] < threshold ? 0.0f : 1.0f;
        for (size_t c = 0; c < 3; ++c) out[p * 4 + c] = src[p * 4 + c] * flag;
        out[p * 4 + 3] = flag;
    }
    if (!iterations) return CN_OK;
    unsigned char *state = malloc(pixels);
    size_t *queue = malloc(pixels * sizeof(size_t));
    float *pending = malloc(pixels * 4 * sizeof(float));
    if (!state || !queue || !pending) {
        free(state); free(queue); free(pending);
        return CN_ALLOCATION_FAILED;
    }
    for (size_t p = 0; p < pixels; ++p) state[p] = (unsigned char)(out[p * 4 + 3] != 0.0f);
    size_t count = 0;
    for (size_t p = 0; p < pixels; ++p) {
        if (state[p] == 1) continue;
        size_t x = p % width, y = p / width;
        if ((x > 0 && state[p - 1] == 1) || (x + 1 < width && state[p + 1] == 1) ||
            (y > 0 && state[p - width] == 1) || (y + 1 < height && state[p + width] == 1))
            enqueue(p, state, queue, &count);
    }
    fill_context ctx = {out, pending, queue, width, height};
    cn_status status = CN_OK;
    size_t start = 0;
    for (size_t iteration = 0; iteration < iterations && start < count; ++iteration) {
        size_t end = count;
        /* Offset both work buffers so each synchronous dispatch starts at zero. */
        ctx.queue = queue + start;
        status = cn_parallel_for(end - start, 16384, fill_values, &ctx);
        if (status != CN_OK) break;
        status = cn_parallel_for(end - start, 16384, fill_commit, &ctx);
        if (status != CN_OK) break;
        for (size_t k = start; k < end; ++k) {
            size_t p = queue[k], x = p % width, y = p / width;
            if (x > 0) enqueue(p - 1, state, queue, &count);
            if (x + 1 < width) enqueue(p + 1, state, queue, &count);
            if (y > 0) enqueue(p - width, state, queue, &count);
            if (y + 1 < height) enqueue(p + width, state, queue, &count);
        }
        start = end;
    }
    free(state); free(queue); free(pending);
    return status;
}
