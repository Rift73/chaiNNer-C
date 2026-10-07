/*
 * vim: syntax=c
 *
 * Implement some C99-compatible complex math functions
 *
 * Most of the code is taken from the msun library in FreeBSD (HEAD @ 4th
 * October 2013), under the following license:
 *
 * Copyright (c) 2007, 2011 David Schultz <das@FreeBSD.ORG>
 * Copyright (c) 2012 Stephen Montgomery-Smith <stephen@FreeBSD.ORG>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#pragma once
#include "cn_crt_math.h"
#include <cmath>
#include <limits>
#include <type_traits>

// Complex square root follows npy_math_complex.c.src, and division follows
// numpy/_core/src/umath/loops.c.src; both are unchanged from NumPy v1.24.4 to
// v2.5.3 (NumPy BSD-3-Clause).
// Pinned source URLs and hashes are recorded with the frozen NCNN oracle.
template<class T> struct cn_ncnn_complex { T real, imag; };
// npy_hypot and npy_hypotf: the process's ucrtbase.dll hypot and _hypotf (cn_crt_math.h).
template<class T> T cn_ncnn_hypot(T a,T b) {
    if constexpr(std::is_same_v<T,float>)return cn_crt.hypotf(a,b);
    else return cn_crt.hypot(a,b);
}
template<class T> cn_ncnn_complex<T> cn_ncnn_complex_divide(cn_ncnn_complex<T> a,cn_ncnn_complex<T> b) {
    T br=std::fabs(b.real),bi=std::fabs(b.imag);
    if(br>=bi){
        if(br==0 && bi==0)return {a.real/br,a.imag/bi};
        T rat=b.imag/b.real,scl=T(1)/(b.real+b.imag*rat);
        return {(a.real+a.imag*rat)*scl,(a.imag-a.real*rat)*scl};
    }
    T rat=b.real/b.imag,scl=T(1)/(b.imag+b.real*rat);
    return {(a.real*rat+a.imag)*scl,(a.imag*rat-a.real)*scl};
}
template<class T> cn_ncnn_complex<T> cn_ncnn_complex_sqrt(cn_ncnn_complex<T> z) {
    T a=z.real,b=z.imag,t;
    if(a==0 && b==0)return {T(0),b};
    if(std::isinf(b))return {std::numeric_limits<T>::infinity(),b};
    if(std::isnan(a)){t=(b-b)/(b-b);return {a,t};}
    if(std::isinf(a)){
        if(std::signbit(a))return {std::fabs(b-b),std::copysign(a,b)};
        return {a,std::copysign(b-b,b)};
    }
    constexpr T sqrt_two=T(1.414213562373095048801688724209698079L);
    constexpr T threshold=std::numeric_limits<T>::max()/(T(1)+sqrt_two);
    bool scale=std::fabs(a)>=threshold || std::fabs(b)>=threshold;
    if(scale){a*=T(0.25);b*=T(0.25);}
    cn_ncnn_complex<T> result;
    if(a>=0){t=std::sqrt((a+cn_ncnn_hypot(a,b))*T(0.5));result={t,b/(T(2)*t)};}
    else {t=std::sqrt((-a+cn_ncnn_hypot(a,b))*T(0.5));result={std::fabs(b)/(T(2)*t),std::copysign(t,b)};}
    if(scale)result.real*=T(2);
    return result;
}
