/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This file is based on files from packages softfloat and fdlibm
// issued with the following licenses:

/*============================================================================

This C source file is part of the SoftFloat IEEE Floating-Point Arithmetic
Package, Release 3c, by John R. Hauser.

Copyright 2011, 2012, 2013, 2014, 2015, 2016, 2017 The Regents of the
University of California.  All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice,
    this list of conditions, and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions, and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

 3. Neither the name of the University nor the names of its contributors may
    be used to endorse or promote products derived from this software without
    specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS", AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ARE
DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=============================================================================*/

// FDLIBM licenses:

/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

/*
 * ====================================================
 * Copyright (C) 2004 by Sun Microsystems, Inc. All rights reserved.
 *
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

/* Altered C adaptation of OpenCV 4.8.0 Gaussian kernel construction and
 * softfloat f64_exp. IEEE double operations preserve the original explicit
 * rounding order; no fused expressions substitute softdouble steps.
 */
#include "chainner.h"
#include "separable.h"
#include "numeric.h"
#include <math.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

static const uint64_t exp_table[] = {
    0x3ff0000000000000, // 1.000000
    0x3ff02c9a3e778061, // 1.010889
    0x3ff059b0d3158574, // 1.021897
    0x3ff0874518759bc8, // 1.033025
    0x3ff0b5586cf9890f, // 1.044274
    0x3ff0e3ec32d3d1a2, // 1.055645
    0x3ff11301d0125b51, // 1.067140
    0x3ff1429aaea92de0, // 1.078761
    0x3ff172b83c7d517b, // 1.090508
    0x3ff1a35beb6fcb75, // 1.102383
    0x3ff1d4873168b9aa, // 1.114387
    0x3ff2063b88628cd6, // 1.126522
    0x3ff2387a6e756238, // 1.138789
    0x3ff26b4565e27cdd, // 1.151189
    0x3ff29e9df51fdee1, // 1.163725
    0x3ff2d285a6e4030b, // 1.176397
    0x3ff306fe0a31b715, // 1.189207
    0x3ff33c08b26416ff, // 1.202157
    0x3ff371a7373aa9cb, // 1.215247
    0x3ff3a7db34e59ff7, // 1.228481
    0x3ff3dea64c123422, // 1.241858
    0x3ff4160a21f72e2a, // 1.255381
    0x3ff44e086061892d, // 1.269051
    0x3ff486a2b5c13cd0, // 1.282870
    0x3ff4bfdad5362a27, // 1.296840
    0x3ff4f9b2769d2ca7, // 1.310961
    0x3ff5342b569d4f82, // 1.325237
    0x3ff56f4736b527da, // 1.339668
    0x3ff5ab07dd485429, // 1.354256
    0x3ff5e76f15ad2148, // 1.369002
    0x3ff6247eb03a5585, // 1.383910
    0x3ff6623882552225, // 1.398980
    0x3ff6a09e667f3bcd, // 1.414214
    0x3ff6dfb23c651a2f, // 1.429613
    0x3ff71f75e8ec5f74, // 1.445181
    0x3ff75feb564267c9, // 1.460918
    0x3ff7a11473eb0187, // 1.476826
    0x3ff7e2f336cf4e62, // 1.492908
    0x3ff82589994cce13, // 1.509164
    0x3ff868d99b4492ed, // 1.525598
    0x3ff8ace5422aa0db, // 1.542211
    0x3ff8f1ae99157736, // 1.559004
    0x3ff93737b0cdc5e5, // 1.575981
    0x3ff97d829fde4e50, // 1.593142
    0x3ff9c49182a3f090, // 1.610490
    0x3ffa0c667b5de565, // 1.628027
    0x3ffa5503b23e255d, // 1.645755
    0x3ffa9e6b5579fdbf, // 1.663677
    0x3ffae89f995ad3ad, // 1.681793
    0x3ffb33a2b84f15fb, // 1.700106
    0x3ffb7f76f2fb5e47, // 1.718619
    0x3ffbcc1e904bc1d2, // 1.737334
    0x3ffc199bdd85529c, // 1.756252
    0x3ffc67f12e57d14b, // 1.775376
    0x3ffcb720dcef9069, // 1.794709
    0x3ffd072d4a07897c, // 1.814252
    0x3ffd5818dcfba487, // 1.834008
    0x3ffda9e603db3285, // 1.853979
    0x3ffdfc97337b9b5f, // 1.874168
    0x3ffe502ee78b3ff6, // 1.894576
    0x3ffea4afa2a490da, // 1.915207
    0x3ffefa1bee615a27, // 1.936062
    0x3fff50765b6e4540, // 1.957144
    0x3fffa7c1819e90d8, // 1.978456
};


static double from_bits(uint64_t bits) {
    double value; memcpy(&value, &bits, sizeof(value)); return value;
}

static double gaussian_exp(double x) {
    if (isnan(x)) return from_bits(UINT64_C(0x7FFFFFFFFFFFFFFF)); /* softdouble::nan() */
    if (isinf(x)) return x > 0 ? CN_INFINITY_F : 0;
    double factor = from_bits(UINT64_C(0x3f83ce0f3e46f431));
    double a5 = 1 / factor;
    double a4 = from_bits(UINT64_C(0x3fe62e42fefa39f1)) / factor;
    double a3 = from_bits(UINT64_C(0x3fcebfbdff82a45a)) / factor;
    double a2 = from_bits(UINT64_C(0x3fac6b08d81fec75)) / factor;
    double a1 = from_bits(UINT64_C(0x3f83b2a72b4f3cd3)) / factor;
    double a0 = from_bits(UINT64_C(0x3f55e7aa1566c2a4)) / factor;
    double prescale = from_bits(UINT64_C(0x3ff71547652b82fe)) * 64;
    double scaled = fabs(x) >= 2048 ? (x < 0 ? -192000 : 192000) : x * prescale;
    double rounded = nearbyint(scaled);
    int value = (int)rounded;
    int exponent = (int)floor((double)value / 64) + 1023;
    if (exponent < 0) exponent = 0;
    if (exponent > 2047) exponent = 2047;
    double base = from_bits((uint64_t)exponent << 52);
    double tail = (scaled - rounded) * (1.0 / 64);
    double polynomial = (((((a0*tail+a1)*tail+a2)*tail+a3)*tail+a4)*tail+a5);
    return base * factor * from_bits(exp_table[(uint32_t)value & 63]) * polynomial;
}

static cn_status kernel(double *out, size_t n, double sigma) {
    if (!n || n > INT_MAX || !isfinite(sigma)) return CN_INVALID_ARGUMENT;
    if (sigma <= 0 && (n == 1 || n == 3 || n == 5 || n == 7 || n == 9)) {
        static const double fixed[5][9] = {
            {1}, {.25,.5,.25}, {.0625,.25,.375,.25,.0625},
            {.03125,.109375,.21875,.28125,.21875,.109375,.03125},
            {.015625,.05078125,.1171875,.19921875,.234375,.19921875,.1171875,.05078125,.015625}
        };
        memcpy(out, fixed[(n-1)/2], n * sizeof(double));
        return CN_OK;
    }
    double s = sigma > 0 ? sigma : fma((double)n, .15, .35);
    double scale = -.125 / (s * s);
    size_t half = (n-1)/2;
    double sum = 0;
    for (size_t i = 0; i < half; ++i) {
        int64_t x = 1 - (int64_t)n + (int64_t)i * 2;
        uint32_t square = (uint32_t)((uint64_t)x * (uint64_t)x);
        int64_t signed_square = square <= INT32_MAX ? (int64_t)square
            : (int64_t)square - INT64_C(4294967296);
        /* Pinned OpenCV evaluates x*x in int32 before softdouble conversion. */
        out[i] = gaussian_exp((double)signed_square * scale);
        sum += out[i];
    }
    sum *= 2; sum += 1;
    if (!(n & 1)) sum += 1;
    double multiplier = 1 / sum;
    for (size_t i = 0; i < half; ++i) {
        out[i] *= multiplier;
        out[n-1-i] = out[i];
    }
    out[half] = multiplier;
    if (!(n & 1)) out[half+1] = multiplier;
    return CN_OK;
}

static cn_status gaussian_kernel_uncached(void *out, size_t n, double sigma, int is_double) {
    if (!out || !n || n > INT_MAX || !isfinite(sigma) || (is_double != 0 && is_double != 1))
        return CN_INVALID_ARGUMENT;
    size_t item = is_double ? sizeof(double) : sizeof(float);
    if ((uintptr_t)out % item) return CN_INVALID_ARGUMENT;
    if ((uintptr_t)out > UINTPTR_MAX - n * item) return CN_SIZE_OVERFLOW;
    if (is_double) return kernel(out, n, sigma);
    double *temporary = malloc(n * sizeof(double));
    if (!temporary) return CN_ALLOCATION_FAILED;
    cn_status status = kernel(temporary, n, sigma);
    if (status == CN_OK)
        for (size_t i = 0; i < n; ++i) ((float *)out)[i] = (float)temporary[i];
    free(temporary);
    return status;
}


/* Immutable coefficient entries. A hit copies under a short lock, so callers
 * never borrow storage that eviction can free. 32 * 128 KiB = 4 MiB retained;
 * larger kernels keep their original uncached construction. Cache allocation
 * failure only declines retention after successful coefficient computation. */
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
static SRWLOCK gaussian_lock = SRWLOCK_INIT;
static void lock_gaussian(void) { AcquireSRWLockExclusive(&gaussian_lock); }
static void unlock_gaussian(void) { ReleaseSRWLockExclusive(&gaussian_lock); }
#else
#include <stdatomic.h>
static atomic_flag gaussian_lock = ATOMIC_FLAG_INIT;
static void lock_gaussian(void) {
    while (atomic_flag_test_and_set_explicit(&gaussian_lock, memory_order_acquire)) {}
}
static void unlock_gaussian(void) {
    atomic_flag_clear_explicit(&gaussian_lock, memory_order_release);
}
#endif
uint64_t cn_image_fp_state(void);
#define GAUSSIAN_SLOTS 32
#define GAUSSIAN_ENTRY_BYTES ((size_t)128 * 1024)
typedef struct gaussian_entry {
    void *data;
    size_t count, bytes;
    uint64_t sigma, controls;
    int is_double;
} gaussian_entry;
static gaussian_entry gaussian_entries[GAUSSIAN_SLOTS];
static size_t gaussian_next;
static uint64_t gaussian_hits, gaussian_builds;

CN_EXPORT cn_status cn_gaussian_kernel(void *out, size_t n, double sigma, int is_double) {
    if (!out || !n || n > INT_MAX || !isfinite(sigma) || (is_double != 0 && is_double != 1))
        return CN_INVALID_ARGUMENT;
    size_t item = is_double ? sizeof(double) : sizeof(float);
    if ((uintptr_t)out % item) return CN_INVALID_ARGUMENT;
    if (n > SIZE_MAX / item || (uintptr_t)out > UINTPTR_MAX - n * item)
        return CN_SIZE_OVERFLOW;
    uint64_t sigma_bits, controls = cn_image_fp_state();
    memcpy(&sigma_bits, &sigma, sizeof(sigma_bits));
    size_t bytes = n * item;
    lock_gaussian();
    for (size_t i = 0; i < GAUSSIAN_SLOTS; ++i) {
        const gaussian_entry *entry = gaussian_entries + i;
        if (entry->data && entry->count == n && entry->sigma == sigma_bits &&
            entry->controls == controls && entry->is_double == is_double) {
            memcpy(out, entry->data, bytes);
            ++gaussian_hits;
            unlock_gaussian();
            return CN_OK;
        }
    }
    ++gaussian_builds;
    unlock_gaussian();
    cn_status status = gaussian_kernel_uncached(out, n, sigma, is_double);
    if (status != CN_OK || bytes > GAUSSIAN_ENTRY_BYTES) return status;
    void *copy = malloc(bytes);
    if (!copy) return CN_OK;
    memcpy(copy, out, bytes);
    lock_gaussian();
    gaussian_entry *entry = gaussian_entries + gaussian_next;
    free(entry->data);
    *entry = (gaussian_entry){copy, n, bytes, sigma_bits, controls, is_double};
    gaussian_next = (gaussian_next + 1) % GAUSSIAN_SLOTS;
    unlock_gaussian();
    return CN_OK;
}

/* Diagnostic counters verify preparation reuse and bounded storage, not speed. */
CN_EXPORT cn_status cn_gaussian_cache_info(uint64_t *out, int clear) {
    if (!out || (uintptr_t)out % _Alignof(uint64_t)) return CN_INVALID_ARGUMENT;
    lock_gaussian();
    if (clear) {
        for (size_t i = 0; i < GAUSSIAN_SLOTS; ++i) {
            free(gaussian_entries[i].data);
            memset(gaussian_entries + i, 0, sizeof(gaussian_entry));
        }
        gaussian_hits = gaussian_builds = 0;
        gaussian_next = 0;
    }
    out[0] = gaussian_hits; out[1] = gaussian_builds; out[2] = out[3] = 0;
    for (size_t i = 0; i < GAUSSIAN_SLOTS; ++i) {
        if (gaussian_entries[i].data) ++out[2];
        out[3] += gaussian_entries[i].bytes;
    }
    unlock_gaussian();
    return CN_OK;
}

CN_EXPORT cn_status cn_gaussian_f32(const float *src, float *dst, size_t height,
    size_t width, size_t channels, size_t nx, size_t ny,
    double sigma_x, double sigma_y, int border, size_t lanes) {
    uintptr_t a = (uintptr_t)src, b = (uintptr_t)dst;
    if (!src || !dst || !height || !width || !channels
        || height > INT_MAX || width > INT_MAX || (lanes != 4 && lanes != 8)
        || (border != 1 && border != 2 && border != 4)
        || a % _Alignof(float) || b % _Alignof(float)
        || !isfinite(sigma_x) || !isfinite(sigma_y) || sigma_x < 0 || sigma_y < 0)
        return CN_INVALID_ARGUMENT;
    if (height > SIZE_MAX / width || height * width > SIZE_MAX / channels / sizeof(float))
        return CN_SIZE_OVERFLOW;
    size_t bytes = height * width * channels * sizeof(float);
    if (a > UINTPTR_MAX - bytes || b > UINTPTR_MAX - bytes) return CN_SIZE_OVERFLOW;
    if (a < b + bytes && b < a + bytes) return CN_INVALID_ARGUMENT;
    if (sigma_y <= 0) sigma_y = sigma_x;
    if (height == 1) ny = 1;
    if (width == 1) nx = 1;
    if (nx == 1 && ny == 1) { memcpy(dst, src, bytes); return CN_OK; }
    if (!nx && sigma_x > 0 && sigma_x < (INT_MAX - 1.0) / 8)
        nx = (size_t)nearbyint(sigma_x * 8 + 1) | 1;
    if (!ny && sigma_y > 0 && sigma_y < (INT_MAX - 1.0) / 8)
        ny = (size_t)nearbyint(sigma_y * 8 + 1) | 1;
    if (!nx || !ny || nx > INT_MAX || ny > INT_MAX || !(nx & 1) || !(ny & 1))
        return CN_INVALID_ARGUMENT;
    float *coefficients = malloc((nx + ny) * sizeof(float));
    if (!coefficients) return CN_ALLOCATION_FAILED;
    cn_status status = cn_gaussian_kernel(coefficients, nx, sigma_x, 0);
    if (status == CN_OK) status = cn_gaussian_kernel(coefficients + nx, ny, sigma_y, 0);
    if (status == CN_OK) status = cn_symmetric_filter_f32(src, dst, height, width,
        channels, coefficients, nx, coefficients + nx, ny, border, lanes);
    free(coefficients);
    return status;
}

CN_EXPORT cn_status cn_ssim_window(double *out) {
    if (!out || (uintptr_t)out % _Alignof(double)) return CN_INVALID_ARGUMENT;
    if ((uintptr_t)out > UINTPTR_MAX - 121 * sizeof(double)) return CN_SIZE_OVERFLOW;
    double weights[11];
    cn_status status = kernel(weights, 11, 1.5);
    if (status != CN_OK) return status;
    for (size_t y = 0; y < 11; ++y)
        for (size_t x = 0; x < 11; ++x) out[y * 11 + x] = weights[y] * weights[x];
    return CN_OK;
}

CN_EXPORT cn_status cn_gaussian_mean_u8(const uint8_t *src, uint8_t *dst,
    size_t height, size_t width, size_t radius, size_t lanes) {
    if (!src || !dst || !height || !width || height > INT_MAX || width > INT_MAX
        || radius > (INT_MAX - 1) / 2 || (lanes != 4 && lanes != 8))
        return CN_INVALID_ARGUMENT;
    if (height > SIZE_MAX / width || height * width > SIZE_MAX / sizeof(float) / 2)
        return CN_SIZE_OVERFLOW;
    size_t count = height * width;
    uintptr_t a = (uintptr_t)src, b = (uintptr_t)dst;
    if (a > UINTPTR_MAX - count || b > UINTPTR_MAX - count) return CN_SIZE_OVERFLOW;
    if (a < b + count && b < a + count) return CN_INVALID_ARGUMENT;
    float *planes = malloc(count * sizeof(float) * 2);
    if (!planes) return CN_ALLOCATION_FAILED;
    for (size_t p = 0; p < count; ++p) planes[p] = (float)src[p];
    cn_status status = cn_gaussian_f32(planes, planes + count, height, width, 1,
        radius * 2 + 1, radius * 2 + 1, 0, 0, 1, lanes);
    if (status == CN_OK)
        for (size_t p = 0; p < count; ++p) {
            float value = nearbyintf(fabsf(planes[count + p]));
            dst[p] = !(value > 0) ? 0 : value >= 255 ? 255 : (uint8_t)value;
        }
    free(planes);
    return status;
}
