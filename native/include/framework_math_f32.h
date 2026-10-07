/* Framework float32 exponential/logarithm contracts. exp is numpy_exp_f32.h's value
 * with NumPy's FP flags; the log polynomial is an altered C++ adaptation of NumPy
 * 2.5.3's loops_exponent_log.dispatch.c.src, npy_simd_data.h, and their scalar
 * operation ordering.
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
#ifndef CHAINNER_FRAMEWORK_MATH_F32_H
#define CHAINNER_FRAMEWORK_MATH_F32_H
#include "cn_crt_math.h"
#include "numpy_exp_f32.h"
#include <cmath>
#include <cfenv>
// NumPy's exp loop sets the overflow and underflow flags for a finite x past its
// limits; the value is numpy_exp_f32.h's.
static inline float cn_framework_exp_f32(float x, int use_fma)
{
    if (use_fma && std::isfinite(x)) {
        if (x >= CN_NUMPY_EXP_F32_MAX) std::feraiseexcept(FE_OVERFLOW);
        else if (x <= CN_NUMPY_EXP_F32_MIN) std::feraiseexcept(FE_UNDERFLOW);
    }
    return cn_numpy_exp_f32(x, use_fma);
}

static inline float cn_framework_log_f32(float input, int use_fma) {
    if (!use_fma) return cn_crt.logf(input);
    if (std::isnan(input)) return cn_f32_from_bits(CN_NPY_NANF_BITS);
    if (input == CN_INFINITY_F) return CN_INFINITY_F;
    if (input == 0) { std::feraiseexcept(FE_DIVBYZERO); return -CN_INFINITY_F; }
    if (input < 0) { std::feraiseexcept(FE_INVALID); return cn_f32_from_bits(CN_DEFAULT_NANF_BITS); }
    int e = 0;
    float x = std::frexp(input, &e);
    if (x <= 0.707106781186547524f) { x += x; --e; }
    x -= 1.0f;
    float p = std::fma(2.589979117907922693523e-02f,x,3.808837741388407920751e-01f);
    p=std::fma(p,x,1.480000633576506585156e+00f);
    p=std::fma(p,x,2.112677543073053063722e+00f);
    p=std::fma(p,x,9.999999999999998702752e-01f);
    p=std::fma(p,x,0.0f);
    float q=std::fma(5.875095403124574342950e-03f,x,1.546476374983906719538e-01f);
    q=std::fma(q,x,9.864942958519418960339e-01f);
    q=std::fma(q,x,2.453006071784736363091e+00f);
    q=std::fma(q,x,2.612677543073109236779e+00f);
    q=std::fma(q,x,1.0f);
    return std::fma(static_cast<float>(e),0.693147180559945309f,p/q);
}

#endif
