#if !defined(_M_X64)
#error "spectral_avx2.c is an x86-64 /arch:AVX2 unit"
#endif
/* The spectral filter's packed-spectrum product, AVX2 unit (/arch:AVX2, spec 4.3): no
 * export, reached only through spectrum_range's dispatch on the call's ISA level (SP4b
 * Task 8). Each value is multiply_pair's: its products, difference and sum in its order
 * (vaddsubpd subtracts in the real lanes and adds in the imaginary ones; no fused
 * multiply-add). Where two NaN inputs meet in a pair, which payload survives may differ
 * from B3's commuted mulsd/addsd (a don't-care: NaN positions and every other bit match).
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
#include "spectral_shared.h"

#include <immintrin.h>

void cn_spectral_pairs_avx2(double *row, const double *other, size_t end_x)
{
    /* Two pairs per vector, lanes (real, imaginary, real, imaginary). The conjugate's
     * sign flips the imaginary lanes of b, as bi = -b[stride] does. */
    const __m256d conjugate = _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);
    size_t x = 1;
    for (; x + 2 < end_x; x += 4) {
        __m256d a = _mm256_loadu_pd(row + x);
        __m256d b = _mm256_xor_pd(_mm256_loadu_pd(other + x), conjugate);
        /* (ar * br, ar * bi) and (ai * bi, ai * br) per pair, then real = ar * br - ai *
         * bi in the even lanes and imag = ar * bi + ai * br in the odd ones. */
        __m256d first = _mm256_mul_pd(_mm256_movedup_pd(a), b);
        __m256d second = _mm256_mul_pd(_mm256_permute_pd(a, 0xF), _mm256_permute_pd(b, 0x5));
        _mm256_storeu_pd(row + x, _mm256_addsub_pd(first, second));
    }
    for (; x < end_x; x += 2) multiply_pair(row + x, other + x, 1);
}
