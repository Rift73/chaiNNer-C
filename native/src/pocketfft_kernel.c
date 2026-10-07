/* Complex FFT kernel: a C port of the complex transforms (sincos_2pibyn, cfftp,
 * fftblue, pocketfft_c) of pocketfft_hdronly.h as vendored by NumPy 2.5.3
 * (numpy/fft/pocketfft), which numpy.fft runs through _pocketfft_umath.cpp.
 * NumPy's Windows wheels are built by MSVC, where POCKETFFT_NO_VECTORS holds, so
 * every line is the scalar plan->exec path ported here. Real, DCT/DST and
 * multi-dimensional code removed; checked plan ABI added.
 *
 * Copyright (C) 2010-2022 Max-Planck-Society
 * Copyright (C) 2019-2020 Peter Bell
 * Authors: Martin Reinecke, Peter Bell
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice, this
 *   list of conditions and the following disclaimer in the documentation and/or
 *   other materials provided with the distribution.
 * * Neither the name of the copyright holder nor the names of its contributors may
 *   be used to endorse or promote products derived from this software without
 *   specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include "cn_crt_math.h"
#include "fft_native.h"
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct cmplx { double r, i; } cmplx;

/* PM, PMINPLACE and the complex operators of the C++ cmplx<T>. */
#define PM(a, b, c, d) do { cmplx pm_c_ = (c), pm_d_ = (d); \
    (a).r = pm_c_.r + pm_d_.r; (a).i = pm_c_.i + pm_d_.i; \
    (b).r = pm_c_.r - pm_d_.r; (b).i = pm_c_.i - pm_d_.i; } while (0)
#define PMINPLACE(a, b) do { cmplx pm_t_ = (a); (a).r += (b).r; (a).i += (b).i; \
    (b).r = pm_t_.r - (b).r; (b).i = pm_t_.i - (b).i; } while (0)

CN_FORCE_INLINE cmplx add(cmplx a, cmplx b) { return (cmplx){a.r + b.r, a.i + b.i}; }
CN_FORCE_INLINE cmplx sub(cmplx a, cmplx b) { return (cmplx){a.r - b.r, a.i - b.i}; }
CN_FORCE_INLINE cmplx scale(cmplx a, double f) { return (cmplx){a.r * f, a.i * f}; }

/* special_mul<fwd>: v times w, or times conj(w) when forward. */
CN_FORCE_INLINE cmplx special_mul(cmplx v, cmplx w, int fwd)
{
    return fwd ? (cmplx){v.r * w.r + v.i * w.i, v.i * w.r - v.r * w.i}
               : (cmplx){v.r * w.r - v.i * w.i, v.r * w.i + v.i * w.r};
}

CN_FORCE_INLINE void rotx90(cmplx *a, int fwd)
{
    double tmp = fwd ? -a->r : a->r;
    a->r = fwd ? a->i : -a->i;
    a->i = tmp;
}

static const double hsqt2 = 0.707106781186547524400844362104849;

CN_FORCE_INLINE void rotx45(cmplx *a, int fwd)
{
    double tmp = a->r;
    if (fwd) { a->r = hsqt2 * (a->r + a->i); a->i = hsqt2 * (a->i - tmp); }
    else { a->r = hsqt2 * (a->r - a->i); a->i = hsqt2 * (a->i + tmp); }
}

CN_FORCE_INLINE void rotx135(cmplx *a, int fwd)
{
    double tmp = a->r;
    if (fwd) { a->r = hsqt2 * (a->i - a->r); a->i = hsqt2 * (-tmp - a->i); }
    else { a->r = hsqt2 * (-a->r - a->i); a->i = hsqt2 * (tmp - a->i); }
}

/* sincos_2pibyn<double>: two tables whose products give exp(2*pi*i*idx/n). */
typedef struct sincos_table {
    size_t n, mask, shift;
    cmplx *v1, *v2;
} sincos_table;

static cmplx sincos_calc(size_t x, size_t n, double ang)
{
    x <<= 3;
    if (x < 4 * n) {
        if (x < 2 * n) {
            if (x < n) return (cmplx){cn_crt.cos((double)x * ang), cn_crt.sin((double)x * ang)};
            return (cmplx){cn_crt.sin((double)(2 * n - x) * ang), cn_crt.cos((double)(2 * n - x) * ang)};
        }
        x -= 2 * n;
        if (x < n) return (cmplx){-cn_crt.sin((double)x * ang), cn_crt.cos((double)x * ang)};
        return (cmplx){-cn_crt.cos((double)(2 * n - x) * ang), cn_crt.sin((double)(2 * n - x) * ang)};
    }
    x = 8 * n - x;
    if (x < 2 * n) {
        if (x < n) return (cmplx){cn_crt.cos((double)x * ang), -cn_crt.sin((double)x * ang)};
        return (cmplx){cn_crt.sin((double)(2 * n - x) * ang), -cn_crt.cos((double)(2 * n - x) * ang)};
    }
    x -= 2 * n;
    if (x < n) return (cmplx){-cn_crt.sin((double)x * ang), -cn_crt.cos((double)x * ang)};
    return (cmplx){-cn_crt.cos((double)(2 * n - x) * ang), -cn_crt.sin((double)(2 * n - x) * ang)};
}

static void sincos_free(sincos_table *t)
{
    free(t->v1);
    free(t->v2);
}

static int sincos_init(sincos_table *t, size_t n)
{
    /* Thigh(0.25L*pi/n); long double is double under MSVC. */
    double ang = 0.25 * 3.141592653589793238462643383279502884197 / (double)n;
    size_t nval = (n + 2) / 2, shift = 1;
    while (((size_t)1 << shift) * ((size_t)1 << shift) < nval) ++shift;
    size_t mask = ((size_t)1 << shift) - 1, count1 = mask + 1;
    size_t count2 = (nval + mask) / (mask + 1);
    t->n = n;
    t->mask = mask;
    t->shift = shift;
    t->v1 = malloc(count1 * sizeof(cmplx));
    t->v2 = malloc(count2 * sizeof(cmplx));
    if (!t->v1 || !t->v2) { sincos_free(t); return -1; }
    t->v1[0] = (cmplx){1.0, 0.0};
    for (size_t i = 1; i < count1; ++i) t->v1[i] = sincos_calc(i, n, ang);
    t->v2[0] = (cmplx){1.0, 0.0};
    for (size_t i = 1; i < count2; ++i) t->v2[i] = sincos_calc(i * (mask + 1), n, ang);
    return 0;
}

static cmplx sincos_at(const sincos_table *t, size_t idx)
{
    if (2 * idx <= t->n) {
        cmplx x1 = t->v1[idx & t->mask], x2 = t->v2[idx >> t->shift];
        return (cmplx){x1.r * x2.r - x1.i * x2.i, x1.r * x2.i + x1.i * x2.r};
    }
    idx = t->n - idx;
    cmplx x1 = t->v1[idx & t->mask], x2 = t->v2[idx >> t->shift];
    return (cmplx){x1.r * x2.r - x1.i * x2.i, -(x1.r * x2.i + x1.i * x2.r)};
}

/* util */
static size_t largest_prime_factor(size_t n)
{
    size_t res = 1;
    while ((n & 1) == 0) { res = 2; n >>= 1; }
    for (size_t x = 3; x * x <= n; x += 2)
        while ((n % x) == 0) { res = x; n /= x; }
    if (n > 1) res = n;
    return res;
}

static double cost_guess(size_t n)
{
    const double lfp = 1.1; /* penalty for non-hardcoded larger factors */
    size_t ni = n;
    double result = 0.;
    while ((n & 1) == 0) { result += 2; n >>= 1; }
    for (size_t x = 3; x * x <= n; x += 2)
        while ((n % x) == 0) {
            result += (x <= 5) ? (double)x : lfp * (double)x;
            n /= x;
        }
    if (n > 1) result += (n <= 5) ? (double)n : lfp * (double)n;
    return result * (double)ni;
}

/* The smallest composite of 2, 3, 5, 7 and 11 which is >= n. */
static size_t good_size_cmplx(size_t n)
{
    if (n <= 12) return n;
    size_t bestfac = 2 * n;
    for (size_t f11 = 1; f11 < bestfac; f11 *= 11)
        for (size_t f117 = f11; f117 < bestfac; f117 *= 7)
            for (size_t f1175 = f117; f1175 < bestfac; f1175 *= 5) {
                size_t x = f1175;
                while (x < n) x *= 2;
                for (;;) {
                    if (x < n) x *= 3;
                    else if (x > n) {
                        if (x < bestfac) bestfac = x;
                        if (x & 1) break;
                        x >>= 1;
                    } else return n;
                }
            }
    return bestfac;
}

/* cfftp */
#define NFCT 25
typedef struct cfftp_fctdata {
    size_t fct;
    cmplx *tw, *tws;
} cfftp_fctdata;

typedef struct cfftp_plan {
    size_t length, nfct;
    cmplx *mem;
    cfftp_fctdata fct[NFCT];
} cfftp_plan;

#define CH(a, b, c) ch[(a) + ido * ((b) + l1 * (c))]
#define CC(a, b, c) cc[(a) + ido * ((b) + cdim * (c))]
#define WA(x, i) wa[(i) - 1 + (x) * (ido - 1)]

CN_FORCE_INLINE void pass2(size_t ido, size_t l1, const cmplx *restrict cc,
    cmplx *restrict ch, const cmplx *restrict wa, int fwd)
{
    const size_t cdim = 2;
    if (ido == 1)
        for (size_t k = 0; k < l1; ++k) {
            CH(0, k, 0) = add(CC(0, 0, k), CC(0, 1, k));
            CH(0, k, 1) = sub(CC(0, 0, k), CC(0, 1, k));
        }
    else
        for (size_t k = 0; k < l1; ++k) {
            CH(0, k, 0) = add(CC(0, 0, k), CC(0, 1, k));
            CH(0, k, 1) = sub(CC(0, 0, k), CC(0, 1, k));
            for (size_t i = 1; i < ido; ++i) {
                CH(i, k, 0) = add(CC(i, 0, k), CC(i, 1, k));
                CH(i, k, 1) = special_mul(sub(CC(i, 0, k), CC(i, 1, k)), WA(0, i), fwd);
            }
        }
}

#define PREP3(idx) \
    cmplx t0 = CC(idx, 0, k), t1, t2; \
    PM(t1, t2, CC(idx, 1, k), CC(idx, 2, k)); \
    CH(idx, k, 0) = add(t0, t1);
#define PARTSTEP3a(u1, u2, twr, twi) { \
    cmplx ca = add(t0, scale(t1, twr)); \
    cmplx cb = {-t2.i * twi, t2.r * twi}; \
    PM(CH(0, k, u1), CH(0, k, u2), ca, cb); }
#define PARTSTEP3b(u1, u2, twr, twi) { \
    cmplx ca = add(t0, scale(t1, twr)); \
    cmplx cb = {-t2.i * twi, t2.r * twi}; \
    CH(i, k, u1) = special_mul(add(ca, cb), WA(u1 - 1, i), fwd); \
    CH(i, k, u2) = special_mul(sub(ca, cb), WA(u2 - 1, i), fwd); }

CN_FORCE_INLINE void pass3(size_t ido, size_t l1, const cmplx *restrict cc,
    cmplx *restrict ch, const cmplx *restrict wa, int fwd)
{
    const size_t cdim = 3;
    const double tw1r = -0.5,
        tw1i = (fwd ? -1 : 1) * 0.8660254037844386467637231707529362;
    if (ido == 1)
        for (size_t k = 0; k < l1; ++k) {
            PREP3(0)
            PARTSTEP3a(1, 2, tw1r, tw1i)
        }
    else
        for (size_t k = 0; k < l1; ++k) {
            {
                PREP3(0)
                PARTSTEP3a(1, 2, tw1r, tw1i)
            }
            for (size_t i = 1; i < ido; ++i) {
                PREP3(i)
                PARTSTEP3b(1, 2, tw1r, tw1i)
            }
        }
}

#undef PARTSTEP3b
#undef PARTSTEP3a
#undef PREP3

CN_FORCE_INLINE void pass4(size_t ido, size_t l1, const cmplx *restrict cc,
    cmplx *restrict ch, const cmplx *restrict wa, int fwd)
{
    const size_t cdim = 4;
    if (ido == 1)
        for (size_t k = 0; k < l1; ++k) {
            cmplx t1, t2, t3, t4;
            PM(t2, t1, CC(0, 0, k), CC(0, 2, k));
            PM(t3, t4, CC(0, 1, k), CC(0, 3, k));
            rotx90(&t4, fwd);
            PM(CH(0, k, 0), CH(0, k, 2), t2, t3);
            PM(CH(0, k, 1), CH(0, k, 3), t1, t4);
        }
    else
        for (size_t k = 0; k < l1; ++k) {
            {
                cmplx t1, t2, t3, t4;
                PM(t2, t1, CC(0, 0, k), CC(0, 2, k));
                PM(t3, t4, CC(0, 1, k), CC(0, 3, k));
                rotx90(&t4, fwd);
                PM(CH(0, k, 0), CH(0, k, 2), t2, t3);
                PM(CH(0, k, 1), CH(0, k, 3), t1, t4);
            }
            for (size_t i = 1; i < ido; ++i) {
                cmplx t1, t2, t3, t4;
                cmplx cc0 = CC(i, 0, k), cc1 = CC(i, 1, k), cc2 = CC(i, 2, k), cc3 = CC(i, 3, k);
                PM(t2, t1, cc0, cc2);
                PM(t3, t4, cc1, cc3);
                rotx90(&t4, fwd);
                CH(i, k, 0) = add(t2, t3);
                CH(i, k, 1) = special_mul(add(t1, t4), WA(0, i), fwd);
                CH(i, k, 2) = special_mul(sub(t2, t3), WA(1, i), fwd);
                CH(i, k, 3) = special_mul(sub(t1, t4), WA(2, i), fwd);
            }
        }
}

#define PREP5(idx) \
    cmplx t0 = CC(idx, 0, k), t1, t2, t3, t4; \
    PM(t1, t4, CC(idx, 1, k), CC(idx, 4, k)); \
    PM(t2, t3, CC(idx, 2, k), CC(idx, 3, k)); \
    CH(idx, k, 0).r = t0.r + t1.r + t2.r; \
    CH(idx, k, 0).i = t0.i + t1.i + t2.i;
#define PARTSTEP5a(u1, u2, twar, twbr, twai, twbi) { \
    cmplx ca, cb; \
    ca.r = t0.r + twar * t1.r + twbr * t2.r; \
    ca.i = t0.i + twar * t1.i + twbr * t2.i; \
    cb.i = twai * t4.r twbi * t3.r; \
    cb.r = -(twai * t4.i twbi * t3.i); \
    PM(CH(0, k, u1), CH(0, k, u2), ca, cb); }
#define PARTSTEP5b(u1, u2, twar, twbr, twai, twbi) { \
    cmplx ca, cb; \
    ca.r = t0.r + twar * t1.r + twbr * t2.r; \
    ca.i = t0.i + twar * t1.i + twbr * t2.i; \
    cb.i = twai * t4.r twbi * t3.r; \
    cb.r = -(twai * t4.i twbi * t3.i); \
    CH(i, k, u1) = special_mul(add(ca, cb), WA(u1 - 1, i), fwd); \
    CH(i, k, u2) = special_mul(sub(ca, cb), WA(u2 - 1, i), fwd); }

CN_FORCE_INLINE void pass5(size_t ido, size_t l1, const cmplx *restrict cc,
    cmplx *restrict ch, const cmplx *restrict wa, int fwd)
{
    const size_t cdim = 5;
    const double tw1r = 0.3090169943749474241022934171828191,
        tw1i = (fwd ? -1 : 1) * 0.9510565162951535721164393333793821,
        tw2r = -0.8090169943749474241022934171828191,
        tw2i = (fwd ? -1 : 1) * 0.5877852522924731291687059546390728;
    if (ido == 1)
        for (size_t k = 0; k < l1; ++k) {
            PREP5(0)
            PARTSTEP5a(1, 4, tw1r, tw2r, +tw1i, +tw2i)
            PARTSTEP5a(2, 3, tw2r, tw1r, +tw2i, -tw1i)
        }
    else
        for (size_t k = 0; k < l1; ++k) {
            {
                PREP5(0)
                PARTSTEP5a(1, 4, tw1r, tw2r, +tw1i, +tw2i)
                PARTSTEP5a(2, 3, tw2r, tw1r, +tw2i, -tw1i)
            }
            for (size_t i = 1; i < ido; ++i) {
                PREP5(i)
                PARTSTEP5b(1, 4, tw1r, tw2r, +tw1i, +tw2i)
                PARTSTEP5b(2, 3, tw2r, tw1r, +tw2i, -tw1i)
            }
        }
}

#undef PARTSTEP5b
#undef PARTSTEP5a
#undef PREP5

#define PREP7(idx) \
    cmplx t1 = CC(idx, 0, k), t2, t3, t4, t5, t6, t7; \
    PM(t2, t7, CC(idx, 1, k), CC(idx, 6, k)); \
    PM(t3, t6, CC(idx, 2, k), CC(idx, 5, k)); \
    PM(t4, t5, CC(idx, 3, k), CC(idx, 4, k)); \
    CH(idx, k, 0).r = t1.r + t2.r + t3.r + t4.r; \
    CH(idx, k, 0).i = t1.i + t2.i + t3.i + t4.i;
#define PARTSTEP7a0(u1, u2, x1, x2, x3, y1, y2, y3, out1, out2) { \
    cmplx ca, cb; \
    ca.r = t1.r + x1 * t2.r + x2 * t3.r + x3 * t4.r; \
    ca.i = t1.i + x1 * t2.i + x2 * t3.i + x3 * t4.i; \
    cb.i = y1 * t7.r y2 * t6.r y3 * t5.r; \
    cb.r = -(y1 * t7.i y2 * t6.i y3 * t5.i); \
    PM(out1, out2, ca, cb); }
#define PARTSTEP7a(u1, u2, x1, x2, x3, y1, y2, y3) \
    PARTSTEP7a0(u1, u2, x1, x2, x3, y1, y2, y3, CH(0, k, u1), CH(0, k, u2))
#define PARTSTEP7(u1, u2, x1, x2, x3, y1, y2, y3) { \
    cmplx da, db; \
    PARTSTEP7a0(u1, u2, x1, x2, x3, y1, y2, y3, da, db) \
    CH(i, k, u1) = special_mul(da, WA(u1 - 1, i), fwd); \
    CH(i, k, u2) = special_mul(db, WA(u2 - 1, i), fwd); }

CN_FORCE_INLINE void pass7(size_t ido, size_t l1, const cmplx *restrict cc,
    cmplx *restrict ch, const cmplx *restrict wa, int fwd)
{
    const size_t cdim = 7;
    const double tw1r = 0.6234898018587335305250048840042398,
        tw1i = (fwd ? -1 : 1) * 0.7818314824680298087084445266740578,
        tw2r = -0.2225209339563144042889025644967948,
        tw2i = (fwd ? -1 : 1) * 0.9749279121818236070181316829939312,
        tw3r = -0.9009688679024191262361023195074451,
        tw3i = (fwd ? -1 : 1) * 0.433883739117558120475768332848359;
    if (ido == 1)
        for (size_t k = 0; k < l1; ++k) {
            PREP7(0)
            PARTSTEP7a(1, 6, tw1r, tw2r, tw3r, +tw1i, +tw2i, +tw3i)
            PARTSTEP7a(2, 5, tw2r, tw3r, tw1r, +tw2i, -tw3i, -tw1i)
            PARTSTEP7a(3, 4, tw3r, tw1r, tw2r, +tw3i, -tw1i, +tw2i)
        }
    else
        for (size_t k = 0; k < l1; ++k) {
            {
                PREP7(0)
                PARTSTEP7a(1, 6, tw1r, tw2r, tw3r, +tw1i, +tw2i, +tw3i)
                PARTSTEP7a(2, 5, tw2r, tw3r, tw1r, +tw2i, -tw3i, -tw1i)
                PARTSTEP7a(3, 4, tw3r, tw1r, tw2r, +tw3i, -tw1i, +tw2i)
            }
            for (size_t i = 1; i < ido; ++i) {
                PREP7(i)
                PARTSTEP7(1, 6, tw1r, tw2r, tw3r, +tw1i, +tw2i, +tw3i)
                PARTSTEP7(2, 5, tw2r, tw3r, tw1r, +tw2i, -tw3i, -tw1i)
                PARTSTEP7(3, 4, tw3r, tw1r, tw2r, +tw3i, -tw1i, +tw2i)
            }
        }
}

#undef PARTSTEP7
#undef PARTSTEP7a0
#undef PARTSTEP7a
#undef PREP7

CN_FORCE_INLINE void pass8(size_t ido, size_t l1, const cmplx *restrict cc,
    cmplx *restrict ch, const cmplx *restrict wa, int fwd)
{
    const size_t cdim = 8;
    for (size_t k = 0; k < l1; ++k) {
        {
            cmplx a0, a1, a2, a3, a4, a5, a6, a7;
            PM(a1, a5, CC(0, 1, k), CC(0, 5, k));
            PM(a3, a7, CC(0, 3, k), CC(0, 7, k));
            PMINPLACE(a1, a3);
            rotx90(&a3, fwd);

            rotx90(&a7, fwd);
            PMINPLACE(a5, a7);
            rotx45(&a5, fwd);
            rotx135(&a7, fwd);

            PM(a0, a4, CC(0, 0, k), CC(0, 4, k));
            PM(a2, a6, CC(0, 2, k), CC(0, 6, k));
            PM(CH(0, k, 0), CH(0, k, 4), add(a0, a2), a1);
            PM(CH(0, k, 2), CH(0, k, 6), sub(a0, a2), a3);
            rotx90(&a6, fwd);
            PM(CH(0, k, 1), CH(0, k, 5), add(a4, a6), a5);
            PM(CH(0, k, 3), CH(0, k, 7), sub(a4, a6), a7);
        }
        for (size_t i = 1; i < ido; ++i) {
            cmplx a0, a1, a2, a3, a4, a5, a6, a7;
            PM(a1, a5, CC(i, 1, k), CC(i, 5, k));
            PM(a3, a7, CC(i, 3, k), CC(i, 7, k));
            rotx90(&a7, fwd);
            PMINPLACE(a1, a3);
            rotx90(&a3, fwd);
            PMINPLACE(a5, a7);
            rotx45(&a5, fwd);
            rotx135(&a7, fwd);
            PM(a0, a4, CC(i, 0, k), CC(i, 4, k));
            PM(a2, a6, CC(i, 2, k), CC(i, 6, k));
            PMINPLACE(a0, a2);
            CH(i, k, 0) = add(a0, a1);
            CH(i, k, 4) = special_mul(sub(a0, a1), WA(3, i), fwd);
            CH(i, k, 2) = special_mul(add(a2, a3), WA(1, i), fwd);
            CH(i, k, 6) = special_mul(sub(a2, a3), WA(5, i), fwd);
            rotx90(&a6, fwd);
            PMINPLACE(a4, a6);
            CH(i, k, 1) = special_mul(add(a4, a5), WA(0, i), fwd);
            CH(i, k, 5) = special_mul(sub(a4, a5), WA(4, i), fwd);
            CH(i, k, 3) = special_mul(add(a6, a7), WA(2, i), fwd);
            CH(i, k, 7) = special_mul(sub(a6, a7), WA(6, i), fwd);
        }
    }
}

#define PREP11(idx) \
    cmplx t1 = CC(idx, 0, k), t2, t3, t4, t5, t6, t7, t8, t9, t10, t11; \
    PM(t2, t11, CC(idx, 1, k), CC(idx, 10, k)); \
    PM(t3, t10, CC(idx, 2, k), CC(idx, 9, k)); \
    PM(t4, t9, CC(idx, 3, k), CC(idx, 8, k)); \
    PM(t5, t8, CC(idx, 4, k), CC(idx, 7, k)); \
    PM(t6, t7, CC(idx, 5, k), CC(idx, 6, k)); \
    CH(idx, k, 0).r = t1.r + t2.r + t3.r + t4.r + t5.r + t6.r; \
    CH(idx, k, 0).i = t1.i + t2.i + t3.i + t4.i + t5.i + t6.i;
#define PARTSTEP11a0(u1, u2, x1, x2, x3, x4, x5, y1, y2, y3, y4, y5, out1, out2) { \
    cmplx ca = add(add(add(add(add(t1, scale(t2, x1)), scale(t3, x2)), scale(t4, x3)), \
        scale(t5, x4)), scale(t6, x5)), cb; \
    cb.i = y1 * t11.r y2 * t10.r y3 * t9.r y4 * t8.r y5 * t7.r; \
    cb.r = -(y1 * t11.i y2 * t10.i y3 * t9.i y4 * t8.i y5 * t7.i); \
    PM(out1, out2, ca, cb); }
#define PARTSTEP11a(u1, u2, x1, x2, x3, x4, x5, y1, y2, y3, y4, y5) \
    PARTSTEP11a0(u1, u2, x1, x2, x3, x4, x5, y1, y2, y3, y4, y5, CH(0, k, u1), CH(0, k, u2))
#define PARTSTEP11(u1, u2, x1, x2, x3, x4, x5, y1, y2, y3, y4, y5) { \
    cmplx da, db; \
    PARTSTEP11a0(u1, u2, x1, x2, x3, x4, x5, y1, y2, y3, y4, y5, da, db) \
    CH(i, k, u1) = special_mul(da, WA(u1 - 1, i), fwd); \
    CH(i, k, u2) = special_mul(db, WA(u2 - 1, i), fwd); }

CN_FORCE_INLINE void pass11(size_t ido, size_t l1, const cmplx *restrict cc,
    cmplx *restrict ch, const cmplx *restrict wa, int fwd)
{
    const size_t cdim = 11;
    const double tw1r = 0.8412535328311811688618116489193677,
        tw1i = (fwd ? -1 : 1) * 0.5406408174555975821076359543186917,
        tw2r = 0.4154150130018864255292741492296232,
        tw2i = (fwd ? -1 : 1) * 0.9096319953545183714117153830790285,
        tw3r = -0.1423148382732851404437926686163697,
        tw3i = (fwd ? -1 : 1) * 0.9898214418809327323760920377767188,
        tw4r = -0.6548607339452850640569250724662936,
        tw4i = (fwd ? -1 : 1) * 0.7557495743542582837740358439723444,
        tw5r = -0.9594929736144973898903680570663277,
        tw5i = (fwd ? -1 : 1) * 0.2817325568414296977114179153466169;
    if (ido == 1)
        for (size_t k = 0; k < l1; ++k) {
            PREP11(0)
            PARTSTEP11a(1, 10, tw1r, tw2r, tw3r, tw4r, tw5r, +tw1i, +tw2i, +tw3i, +tw4i, +tw5i)
            PARTSTEP11a(2, 9, tw2r, tw4r, tw5r, tw3r, tw1r, +tw2i, +tw4i, -tw5i, -tw3i, -tw1i)
            PARTSTEP11a(3, 8, tw3r, tw5r, tw2r, tw1r, tw4r, +tw3i, -tw5i, -tw2i, +tw1i, +tw4i)
            PARTSTEP11a(4, 7, tw4r, tw3r, tw1r, tw5r, tw2r, +tw4i, -tw3i, +tw1i, +tw5i, -tw2i)
            PARTSTEP11a(5, 6, tw5r, tw1r, tw4r, tw2r, tw3r, +tw5i, -tw1i, +tw4i, -tw2i, +tw3i)
        }
    else
        for (size_t k = 0; k < l1; ++k) {
            {
                PREP11(0)
                PARTSTEP11a(1, 10, tw1r, tw2r, tw3r, tw4r, tw5r, +tw1i, +tw2i, +tw3i, +tw4i, +tw5i)
                PARTSTEP11a(2, 9, tw2r, tw4r, tw5r, tw3r, tw1r, +tw2i, +tw4i, -tw5i, -tw3i, -tw1i)
                PARTSTEP11a(3, 8, tw3r, tw5r, tw2r, tw1r, tw4r, +tw3i, -tw5i, -tw2i, +tw1i, +tw4i)
                PARTSTEP11a(4, 7, tw4r, tw3r, tw1r, tw5r, tw2r, +tw4i, -tw3i, +tw1i, +tw5i, -tw2i)
                PARTSTEP11a(5, 6, tw5r, tw1r, tw4r, tw2r, tw3r, +tw5i, -tw1i, +tw4i, -tw2i, +tw3i)
            }
            for (size_t i = 1; i < ido; ++i) {
                PREP11(i)
                PARTSTEP11(1, 10, tw1r, tw2r, tw3r, tw4r, tw5r, +tw1i, +tw2i, +tw3i, +tw4i, +tw5i)
                PARTSTEP11(2, 9, tw2r, tw4r, tw5r, tw3r, tw1r, +tw2i, +tw4i, -tw5i, -tw3i, -tw1i)
                PARTSTEP11(3, 8, tw3r, tw5r, tw2r, tw1r, tw4r, +tw3i, -tw5i, -tw2i, +tw1i, +tw4i)
                PARTSTEP11(4, 7, tw4r, tw3r, tw1r, tw5r, tw2r, +tw4i, -tw3i, +tw1i, +tw5i, -tw2i)
                PARTSTEP11(5, 6, tw5r, tw1r, tw4r, tw2r, tw3r, +tw5i, -tw1i, +tw4i, -tw2i, +tw3i)
            }
        }
}

#undef PARTSTEP11
#undef PARTSTEP11a0
#undef PARTSTEP11a
#undef PREP11

#define CX(a, b, c) cc[(a) + ido * ((b) + l1 * (c))]
#define CX2(a, b) cc[(a) + idl1 * (b)]
#define CH2(a, b) ch[(a) + idl1 * (b)]

CN_FORCE_INLINE int passg(size_t ido, size_t ip, size_t l1, cmplx *restrict cc,
    cmplx *restrict ch, const cmplx *restrict wa, const cmplx *restrict csarr, int fwd)
{
    const size_t cdim = ip;
    size_t ipph = (ip + 1) / 2, idl1 = ido * l1;
    cmplx *wal = malloc(ip * sizeof(cmplx));
    if (!wal) return -1;
    wal[0] = (cmplx){1., 0.};
    for (size_t i = 1; i < ip; ++i)
        wal[i] = (cmplx){csarr[i].r, fwd ? -csarr[i].i : csarr[i].i};

    for (size_t k = 0; k < l1; ++k)
        for (size_t i = 0; i < ido; ++i)
            CH(i, k, 0) = CC(i, 0, k);
    for (size_t j = 1, jc = ip - 1; j < ipph; ++j, --jc)
        for (size_t k = 0; k < l1; ++k)
            for (size_t i = 0; i < ido; ++i)
                PM(CH(i, k, j), CH(i, k, jc), CC(i, j, k), CC(i, jc, k));
    for (size_t k = 0; k < l1; ++k)
        for (size_t i = 0; i < ido; ++i) {
            cmplx tmp = CH(i, k, 0);
            for (size_t j = 1; j < ipph; ++j) tmp = add(tmp, CH(i, k, j));
            CX(i, k, 0) = tmp;
        }
    for (size_t l = 1, lc = ip - 1; l < ipph; ++l, --lc) {
        /* j=0 */
        for (size_t ik = 0; ik < idl1; ++ik) {
            CX2(ik, l).r = CH2(ik, 0).r + wal[l].r * CH2(ik, 1).r + wal[2 * l].r * CH2(ik, 2).r;
            CX2(ik, l).i = CH2(ik, 0).i + wal[l].r * CH2(ik, 1).i + wal[2 * l].r * CH2(ik, 2).i;
            CX2(ik, lc).r = -wal[l].i * CH2(ik, ip - 1).i - wal[2 * l].i * CH2(ik, ip - 2).i;
            CX2(ik, lc).i = wal[l].i * CH2(ik, ip - 1).r + wal[2 * l].i * CH2(ik, ip - 2).r;
        }
        size_t iwal = 2 * l;
        size_t j = 3, jc = ip - 3;
        for (; j < ipph - 1; j += 2, jc -= 2) {
            iwal += l; if (iwal > ip) iwal -= ip;
            cmplx xwal = wal[iwal];
            iwal += l; if (iwal > ip) iwal -= ip;
            cmplx xwal2 = wal[iwal];
            for (size_t ik = 0; ik < idl1; ++ik) {
                CX2(ik, l).r += CH2(ik, j).r * xwal.r + CH2(ik, j + 1).r * xwal2.r;
                CX2(ik, l).i += CH2(ik, j).i * xwal.r + CH2(ik, j + 1).i * xwal2.r;
                CX2(ik, lc).r -= CH2(ik, jc).i * xwal.i + CH2(ik, jc - 1).i * xwal2.i;
                CX2(ik, lc).i += CH2(ik, jc).r * xwal.i + CH2(ik, jc - 1).r * xwal2.i;
            }
        }
        for (; j < ipph; ++j, --jc) {
            iwal += l; if (iwal > ip) iwal -= ip;
            cmplx xwal = wal[iwal];
            for (size_t ik = 0; ik < idl1; ++ik) {
                CX2(ik, l).r += CH2(ik, j).r * xwal.r;
                CX2(ik, l).i += CH2(ik, j).i * xwal.r;
                CX2(ik, lc).r -= CH2(ik, jc).i * xwal.i;
                CX2(ik, lc).i += CH2(ik, jc).r * xwal.i;
            }
        }
    }
    free(wal);

    /* shuffling and twiddling */
    if (ido == 1)
        for (size_t j = 1, jc = ip - 1; j < ipph; ++j, --jc)
            for (size_t ik = 0; ik < idl1; ++ik) {
                cmplx t1 = CX2(ik, j), t2 = CX2(ik, jc);
                PM(CX2(ik, j), CX2(ik, jc), t1, t2);
            }
    else
        for (size_t j = 1, jc = ip - 1; j < ipph; ++j, --jc)
            for (size_t k = 0; k < l1; ++k) {
                cmplx t1 = CX(0, k, j), t2 = CX(0, k, jc);
                PM(CX(0, k, j), CX(0, k, jc), t1, t2);
                for (size_t i = 1; i < ido; ++i) {
                    cmplx x1, x2;
                    PM(x1, x2, CX(i, k, j), CX(i, k, jc));
                    size_t idij = (j - 1) * (ido - 1) + i - 1;
                    CX(i, k, j) = special_mul(x1, wa[idij], fwd);
                    idij = (jc - 1) * (ido - 1) + i - 1;
                    CX(i, k, jc) = special_mul(x2, wa[idij], fwd);
                }
            }
    return 0;
}

#undef CH2
#undef CX2
#undef CX
#undef WA
#undef CC
#undef CH

CN_FORCE_INLINE int pass_all(const cfftp_plan *plan, cmplx c[], double fct, int fwd)
{
    size_t length = plan->length;
    if (length == 1) { c[0] = scale(c[0], fct); return 0; }
    size_t l1 = 1;
    cmplx *ch = malloc(length * sizeof(cmplx));
    if (!ch) return -1;
    cmplx *p1 = c, *p2 = ch, *swap;

    for (size_t k1 = 0; k1 < plan->nfct; k1++) {
        size_t ip = plan->fct[k1].fct;
        size_t l2 = ip * l1;
        size_t ido = length / l2;
        const cmplx *tw = plan->fct[k1].tw;
        if (ip == 4) pass4(ido, l1, p1, p2, tw, fwd);
        else if (ip == 8) pass8(ido, l1, p1, p2, tw, fwd);
        else if (ip == 2) pass2(ido, l1, p1, p2, tw, fwd);
        else if (ip == 3) pass3(ido, l1, p1, p2, tw, fwd);
        else if (ip == 5) pass5(ido, l1, p1, p2, tw, fwd);
        else if (ip == 7) pass7(ido, l1, p1, p2, tw, fwd);
        else if (ip == 11) pass11(ido, l1, p1, p2, tw, fwd);
        else {
            if (passg(ido, ip, l1, p1, p2, tw, plan->fct[k1].tws, fwd)) { free(ch); return -1; }
            swap = p1; p1 = p2; p2 = swap;
        }
        swap = p1; p1 = p2; p2 = swap;
        l1 = l2;
    }
    if (p1 != c) {
        if (fct != 1.)
            for (size_t i = 0; i < length; ++i) c[i] = scale(ch[i], fct);
        else
            memcpy(c, p1, length * sizeof(cmplx));
    } else if (fct != 1.)
        for (size_t i = 0; i < length; ++i) c[i] = scale(c[i], fct);
    free(ch);
    return 0;
}

static CN_NOINLINE int cfftp_forward(const cfftp_plan *plan, cmplx c[], double fct)
{
    return pass_all(plan, c, fct, 1);
}

static CN_NOINLINE int cfftp_backward(const cfftp_plan *plan, cmplx c[], double fct)
{
    return pass_all(plan, c, fct, 0);
}

static int cfftp_add_factor(cfftp_plan *plan, size_t factor)
{
    if (plan->nfct >= NFCT) return -1;
    plan->fct[plan->nfct++] = (cfftp_fctdata){factor, NULL, NULL};
    return 0;
}

static int cfftp_factorize(cfftp_plan *plan)
{
    size_t len = plan->length;
    while ((len & 7) == 0) { if (cfftp_add_factor(plan, 8)) return -1; len >>= 3; }
    while ((len & 3) == 0) { if (cfftp_add_factor(plan, 4)) return -1; len >>= 2; }
    if ((len & 1) == 0) {
        len >>= 1;
        /* factor 2 should be at the front of the factor list */
        if (cfftp_add_factor(plan, 2)) return -1;
        size_t first = plan->fct[0].fct;
        plan->fct[0].fct = plan->fct[plan->nfct - 1].fct;
        plan->fct[plan->nfct - 1].fct = first;
    }
    for (size_t divisor = 3; divisor * divisor <= len; divisor += 2)
        while ((len % divisor) == 0) {
            if (cfftp_add_factor(plan, divisor)) return -1;
            len /= divisor;
        }
    if (len > 1 && cfftp_add_factor(plan, len)) return -1;
    return 0;
}

static size_t cfftp_twsize(const cfftp_plan *plan)
{
    size_t twsize = 0, l1 = 1;
    for (size_t k = 0; k < plan->nfct; ++k) {
        size_t ip = plan->fct[k].fct, ido = plan->length / (l1 * ip);
        twsize += (ip - 1) * (ido - 1);
        if (ip > 11) twsize += ip;
        l1 *= ip;
    }
    return twsize;
}

static int cfftp_comp_twiddle(cfftp_plan *plan)
{
    sincos_table twiddle;
    if (sincos_init(&twiddle, plan->length)) return -1;
    size_t l1 = 1, memofs = 0;
    for (size_t k = 0; k < plan->nfct; ++k) {
        size_t ip = plan->fct[k].fct, ido = plan->length / (l1 * ip);
        plan->fct[k].tw = plan->mem + memofs;
        memofs += (ip - 1) * (ido - 1);
        for (size_t j = 1; j < ip; ++j)
            for (size_t i = 1; i < ido; ++i)
                plan->fct[k].tw[(j - 1) * (ido - 1) + i - 1] = sincos_at(&twiddle, j * l1 * i);
        if (ip > 11) {
            plan->fct[k].tws = plan->mem + memofs;
            memofs += ip;
            for (size_t j = 0; j < ip; ++j)
                plan->fct[k].tws[j] = sincos_at(&twiddle, j * l1 * ido);
        }
        l1 *= ip;
    }
    sincos_free(&twiddle);
    return 0;
}

static void cfftp_destroy(cfftp_plan *plan)
{
    if (!plan) return;
    free(plan->mem);
    free(plan);
}

static cfftp_plan *cfftp_make(size_t length)
{
    cfftp_plan *plan = calloc(1, sizeof *plan);
    if (!plan) return NULL;
    plan->length = length;
    if (length == 1) return plan;
    if (cfftp_factorize(plan)) { free(plan); return NULL; }
    size_t tws = cfftp_twsize(plan);
    /* A prime length has no twiddles; keep a valid allocation. */
    plan->mem = malloc((tws ? tws : 1) * sizeof(cmplx));
    if (!plan->mem || cfftp_comp_twiddle(plan)) { cfftp_destroy(plan); return NULL; }
    return plan;
}

/* fftblue */
typedef struct fftblue_plan {
    size_t n, n2;
    cfftp_plan *plan;
    cmplx *mem, *bk, *bkf;
} fftblue_plan;

static void fftblue_destroy(fftblue_plan *plan)
{
    if (!plan) return;
    cfftp_destroy(plan->plan);
    free(plan->mem);
    free(plan);
}

static fftblue_plan *fftblue_make(size_t length)
{
    fftblue_plan *plan = calloc(1, sizeof *plan);
    if (!plan) return NULL;
    plan->n = length;
    plan->n2 = good_size_cmplx(length * 2 - 1);
    size_t n = plan->n, n2 = plan->n2;
    plan->plan = cfftp_make(n2);
    plan->mem = malloc((n + n2 / 2 + 1) * sizeof(cmplx));
    cmplx *tbkf = malloc(n2 * sizeof(cmplx));
    sincos_table tmp;
    if (!plan->plan || !plan->mem || !tbkf || sincos_init(&tmp, 2 * n)) {
        free(tbkf);
        fftblue_destroy(plan);
        return NULL;
    }
    plan->bk = plan->mem;
    plan->bkf = plan->mem + n;

    /* initialize b_k */
    cmplx *bk = plan->bk;
    bk[0] = (cmplx){1, 0};
    size_t coeff = 0;
    for (size_t m = 1; m < n; ++m) {
        coeff += 2 * m - 1;
        if (coeff >= 2 * n) coeff -= 2 * n;
        bk[m] = sincos_at(&tmp, coeff);
    }
    sincos_free(&tmp);

    /* initialize the zero-padded, Fourier transformed b_k. Add normalisation. */
    double xn2 = 1.0 / (double)n2;
    tbkf[0] = scale(bk[0], xn2);
    for (size_t m = 1; m < n; ++m) tbkf[m] = tbkf[n2 - m] = scale(bk[m], xn2);
    for (size_t m = n; m <= n2 - n; ++m) tbkf[m] = (cmplx){0., 0.};
    if (cfftp_forward(plan->plan, tbkf, 1.)) {
        free(tbkf);
        fftblue_destroy(plan);
        return NULL;
    }
    for (size_t i = 0; i < n2 / 2 + 1; ++i) plan->bkf[i] = tbkf[i];
    free(tbkf);
    return plan;
}

CN_FORCE_INLINE int fftblue_fft(const fftblue_plan *plan, cmplx c[], double fct, int fwd)
{
    size_t n = plan->n, n2 = plan->n2;
    const cmplx *bk = plan->bk, *bkf = plan->bkf;
    cmplx *akf = malloc(n2 * sizeof(cmplx));
    if (!akf) return -1;

    /* initialize a_k and FFT it */
    for (size_t m = 0; m < n; ++m) akf[m] = special_mul(c[m], bk[m], fwd);
    cmplx zero = scale(akf[0], 0.0);
    for (size_t m = n; m < n2; ++m) akf[m] = zero;

    if (cfftp_forward(plan->plan, akf, 1.)) { free(akf); return -1; }

    /* do the convolution */
    akf[0] = special_mul(akf[0], bkf[0], !fwd);
    for (size_t m = 1; m < (n2 + 1) / 2; ++m) {
        akf[m] = special_mul(akf[m], bkf[m], !fwd);
        akf[n2 - m] = special_mul(akf[n2 - m], bkf[m], !fwd);
    }
    if ((n2 & 1) == 0) akf[n2 / 2] = special_mul(akf[n2 / 2], bkf[n2 / 2], !fwd);

    /* inverse FFT */
    if (cfftp_backward(plan->plan, akf, 1.)) { free(akf); return -1; }

    /* multiply by b_k */
    for (size_t m = 0; m < n; ++m) c[m] = scale(special_mul(akf[m], bk[m], fwd), fct);
    free(akf);
    return 0;
}

static CN_NOINLINE int fftblue_forward(const fftblue_plan *plan, cmplx c[], double fct)
{
    return fftblue_fft(plan, c, fct, 1);
}

static CN_NOINLINE int fftblue_backward(const fftblue_plan *plan, cmplx c[], double fct)
{
    return fftblue_fft(plan, c, fct, 0);
}

/* pocketfft_c: FFTPACK or Bluestein by the C++ cost model. */
typedef struct cn_fft_plan_opaque {
    cfftp_plan *packplan;
    fftblue_plan *blueplan;
} pocketfft_c;

cn_fft_plan cn_fft_make_plan(size_t length)
{
    /* The checked ABI's size limit (cn_fft2 validates the same bound). */
    if (!length || length > (size_t)INT_MAX / 16) return NULL;
    pocketfft_c *plan = calloc(1, sizeof *plan);
    if (!plan) return NULL;
    size_t tmp = (length < 50) ? 0 : largest_prime_factor(length);
    int blue = 0;
    if (tmp * tmp > length) {
        double comp1 = cost_guess(length);
        double comp2 = 2 * cost_guess(good_size_cmplx(2 * length - 1));
        comp2 *= 1.5; /* fudge factor that appears to give good overall performance */
        blue = comp2 < comp1;
    }
    if (blue) plan->blueplan = fftblue_make(length);
    else plan->packplan = cfftp_make(length);
    if (!plan->blueplan && !plan->packplan) { free(plan); return NULL; }
    return plan;
}

void cn_fft_destroy_plan(cn_fft_plan plan)
{
    if (!plan) return;
    cfftp_destroy(plan->packplan);
    fftblue_destroy(plan->blueplan);
    free(plan);
}

int cn_fft_execute(cn_fft_plan plan, double *data, int inverse, double fct)
{
    cmplx *c = (cmplx *)data;
    if (plan->packplan)
        return inverse ? cfftp_backward(plan->packplan, c, fct) : cfftp_forward(plan->packplan, c, fct);
    return inverse ? fftblue_backward(plan->blueplan, c, fct) : fftblue_forward(plan->blueplan, c, fct);
}
