/* The spectral filter's packed-spectrum product, moved from spectral_filter_ops.c
 * (SP4b Task 8). Shared by that baseline unit and the /arch:AVX2 unit spectral_avx2.c:
 * multiply_pair, a static helper compiled into each including unit, never inline, so
 * the linker cannot fold a VEX copy into baseline callers. Both includers use it
 * (C4505): spectral_filter_ops.c for the packed columns and, below the AVX2 level,
 * every row pair; spectral_avx2.c for a row's last pair.
 *
 * C extraction of the packed-spectrum portion of OpenCV 4.8.0 mulSpectrums
 * (core/src/dxt.cpp).
 *
 * Copyright (C) 2000, Intel Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * The name of Intel Corporation may not be used to endorse or promote
 *    products derived from this software without specific prior written
 *    permission.
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE INTEL CORPORATION OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef CHAINNER_SPECTRAL_SHARED_H
#define CHAINNER_SPECTRAL_SHARED_H
#include "chainner.h"

#include <stddef.h>

static void multiply_pair(double *a, const double *b, size_t stride)
{
    double ar = a[0], ai = a[stride], br = b[0], bi = -b[stride];
    /* Keep the original conjugation and operation order, including signed zero. */
    double real = ar * br - ai * bi;
    double imag = ar * bi + ai * br;
    a[0] = real;
    a[stride] = imag;
}

/* spectral_avx2.c, for ISA level avx2 or above: a row's pairs (x, x + 1) for x = 1, 3,
 * ... below end_x as multiply_pair(row + x, other + x, 1), two pairs at a time, the
 * last one through multiply_pair. */
void cn_spectral_pairs_avx2(double *row, const double *other, size_t end_x);
#endif
