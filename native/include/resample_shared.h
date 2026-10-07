/* Resample's two-pass context and its scalar vertical output, moved from
 * resample_filters.c (SP4b Task 7). Shared by that baseline unit and the /arch:AVX2
 * unit resample_avx2.c: the coefficient line, the context, and vertical_one, a static
 * helper compiled into each including unit, never inline, so the linker cannot fold a
 * VEX copy into baseline callers. Both includers use it (C4505): resample_filters.c
 * for every vertical output below the AVX2 level, resample_avx2.c for a row's last
 * outputs after its groups of 8.
 *
 * Altered C adaptation of resize 0.8.3 src/lib.rs and chaiNNer-rs
 * crates/image-ops/src/scale/filter.rs at commit
 * 6f6ead6064f81b4049d3deb803c9279c7950736b.
 *
 * The MIT License (MIT)
 * Copyright (c) 2015 PistonDevelopers
 * Copyright (c) 2023 Michael Schmidt
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef CHAINNER_RESAMPLE_SHARED_H
#define CHAINNER_RESAMPLE_SHARED_H
#include "chainner.h"
#include "isa.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* Output x of a pass reads source samples [start, start + count) with weights. */
typedef struct coefficient_line {
    size_t start, count;
    const float *weights;
} coefficient_line;

/* clip: 0 none (the triangle filter), 1 the scalar clamp, 2 Vec4's clamp.
 * isa: the level cn_resample_filtered read once for the call (cn_isa_current). */
typedef struct resample_context {
    const float *source;
    float *intermediate, *out;
    const coefficient_line *horizontal, *vertical;
    size_t source_width, target_width, channels;
    int clip;
    cn_isa_level isa;
} resample_context;

/* Vertical output x (of target_width * channels) of the row whose coefficients are
 * line: B3's sequence verbatim (acc = 0, acc += in * w per tap in order, then the
 * clip). MSVC compiles both clips to B3's maxss/minss forms in each includer.
 * Where the sum's NaN meets the term's, B3's addss keeps the sum's (its first source),
 * but the compiler may commute the sources (as SSE2's two-operand addss does), so a NaN
 * sum, found by its bits, is kept as is: every sum and product is quiet, so that add
 * returns the sum and raises no flag. Each term is still multiplied, with its flags. */
static float vertical_one(const resample_context *ctx, const coefficient_line *line, size_t x)
{
    size_t stride = ctx->target_width * ctx->channels;
    float acc = 0.0f;
    for (size_t i = 0; i < line->count; ++i) {
        float term = ctx->intermediate[(line->start + i) * stride + x] * line->weights[i];
        uint32_t bits;
        memcpy(&bits, &acc, sizeof(bits));
        if ((bits & UINT32_C(0x7fffffff)) <= UINT32_C(0x7f800000)) acc += term;
    }
    if (ctx->clip == 2) {
        /* Original Vec4::clamp uses SSE max/min: unordered selects
         * the second operand, unlike the scalar float clamp. */
        acc = acc > 0.0f ? acc : 0.0f;
        acc = acc < 1.0f ? acc : 1.0f;
    } else if (ctx->clip == 1) {
        if (acc < 0.0f) acc = 0.0f;
        if (acc > 1.0f) acc = 1.0f;
    }
    return acc;
}

/* resample_avx2.c, for ISA level avx2 or above: vertical outputs [first, last) of
 * output row y into ctx->out. Each lane runs vertical_one's sequence (a vmulps then a
 * vaddps per tap, no fused multiply-add, then B3's clip instructions); blocks of 64 outputs
 * keep 8 independent sums, then single groups of 8, then vertical_one, which also
 * redoes a group whose sum holds a NaN (B3's payload where two NaNs meet), unless
 * clip 2 turns it into +0. */
void cn_resample_vertical_avx2(const resample_context *ctx, size_t y, size_t first, size_t last);
#endif
