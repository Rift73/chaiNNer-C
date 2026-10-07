/* The exceptional-value rule, shared by image_setup_ops.c's scalar scan and the
 * /arch:AVX2 unit morphology_avx2.c's tail: a static helper compiled into each
 * including unit, never inline, used by both (C4505). */
#ifndef CHAINNER_EXCEPTIONAL_SHARED_H
#define CHAINNER_EXCEPTIONAL_SHARED_H
#include <stdint.h>

/* A float32 word OpenCV's morphology contracts treat as exceptional: -0, an infinity
 * or a NaN (exponent all ones), and under DAZ a negative denormal, which compares as
 * -0 there. Bit tests: the MXCSR changes no result but through daz. The morphology
 * pre-scan's predicate (!isfinite(v) || (v == 0 && signbit(v)), its compare under the
 * caller's MXCSR) equals it in both states. */
static int exceptional_bits(uint32_t bits, int daz)
{
    return bits == UINT32_C(0x80000000) ||
        (daz && bits > UINT32_C(0x80000000) && bits < UINT32_C(0x80800000)) ||
        (bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000);
}
#endif
