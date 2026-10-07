/* Shared float32 sin/cos preserves NumPy 2.5.3's operation order. Altered C
 * adaptation of numpy/_core/src/umath/loops_trigonometric.dispatch.cpp. Its MSVC
 * build (/fp:contract, meson_cpu/x86/meson.build) fuses the quadrant's multiply
 * and add into one FMA, so the quadrant is fmaf(x, 2/pi, magic) - magic.
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

#ifndef CHAINNER_NUMPY_TRIG_F32_H
#define CHAINNER_NUMPY_TRIG_F32_H
#include "cn_crt_math.h"
#include "numeric.h"
#include <math.h>
static inline float cn_numpy_trig_f32(float x, int cosine_op, int use_fma)
{
    if (!use_fma || fabsf(x) > (cosine_op ? 71476.0625f : 117435.992f))
        return cosine_op ? cn_crt.cosf(x) : cn_crt.sinf(x);
    if (isnan(x)) return cn_f32_from_bits(CN_NPY_NANF_BITS);
    float q = fmaf(x, 0x1.45f306p-1f, 0x1.800000p+23f);
    q -= 0x1.800000p+23f;
    float r = fmaf(q, -0x1.921fb0p+00f, x);
    r = fmaf(q, -0x1.5110b4p-22f, r);
    r = fmaf(q, -0x1.846988p-48f, r);
    float square = r * r;
    float cosine = fmaf(0x1.98e616p-16f, square, -0x1.6c06dcp-10f);
    cosine = fmaf(cosine, square, 0x1.55553cp-05f);
    cosine = fmaf(cosine, square, -0x1.000000p-01f);
    cosine = fmaf(cosine, square, 1.0f);
    float sine = fmaf(0x1.7d3bbcp-19f, square, -0x1.a06bbap-13f);
    sine = fmaf(sine, square, 0x1.11119ap-07f);
    sine = fmaf(sine, square, -0x1.555556p-03f);
    sine = fmaf(sine, square, 0.0f);
    sine = fmaf(sine, r, r);
    int quadrant = (int)q + cosine_op;
    float value = (quadrant & 1) ? cosine : sine;
    return (quadrant & 2) ? 0.0f - value : value;
}

#endif
