/* NumPy 2.5.3's float32 exp (umath/loops_exponent_log.dispatch.c.src, its
 * AVX2/FMA3 form, which NumPy's dispatch runs where it has FMA3; npy_simd_data.h's
 * constants), for C and C++ callers. Its MSVC build (/fp:contract) fuses the
 * quadrant's multiply and add into one FMA. Without FMA the loop is the CRT expf.
 * The value only: a caller that reports FP events raises NumPy's overflow and
 * underflow itself for a finite x past the limits below.
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
#ifndef CHAINNER_NUMPY_EXP_F32_H
#define CHAINNER_NUMPY_EXP_F32_H
#include "cn_crt_math.h"
#include "numeric.h"
#include <math.h>
#include <stdint.h>
#include <string.h>
#define CN_NUMPY_EXP_F32_MAX 88.72283935546875f
#define CN_NUMPY_EXP_F32_MIN -103.97208404541015625f
static inline float cn_numpy_exp_f32(float x, int use_fma)
{
    if (!use_fma) return cn_crt.expf(x);
    if (isnan(x)) return cn_f32_from_bits(CN_NPY_NANF_BITS);
    if (x >= CN_NUMPY_EXP_F32_MAX) return CN_INFINITY_F;
    if (x <= CN_NUMPY_EXP_F32_MIN) return 0.0f;
    float q = fmaf(x, 1.44269504088896340736f, 0x1.800000p+23f);
    q -= 0x1.800000p+23f;
    x = fmaf(q, -6.93145752e-1f, x);
    x = fmaf(q, -1.42860677e-6f, x);
    x = fmaf(q, 0.0f, x);
    float p = fmaf(5.082762527590693718096e-04f, x, 6.757896990527504603057e-03f);
    p = fmaf(p, x, 5.114512081637298353406e-02f);
    p = fmaf(p, x, 2.473615434895520810817e-01f);
    p = fmaf(p, x, 7.257664613233124478488e-01f);
    p = fmaf(p, x, 9.999999999980870924916e-01f);
    float d = fmaf(2.159509375685829852307e-02f, x, -2.742335390411667452936e-01f);
    d = fmaf(d, x, 1.0f);
    p /= d;
    int exponent = (int)q;
    int adjusted = exponent < -125 ? -125 : exponent;
    uint32_t bits;
    memcpy(&bits, &p, sizeof(bits));
    bits += (uint32_t)adjusted << 23;
    memcpy(&p, &bits, sizeof(p));
    if (exponent <= -125) p /= (float)(UINT32_C(1) << (unsigned)(-125 - exponent));
    return p;
}
#endif
