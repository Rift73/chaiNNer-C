/* The exceptional-value scan shared by cn_image_exceptional and the morphology
 * pre-scan (D11): prototypes only. */
#ifndef CHAINNER_EXCEPTIONAL_H
#define CHAINNER_EXCEPTIONAL_H
#include "chainner.h"
#include "isa.h"

/* 1 when src[0, count) holds a word exceptional_bits accepts under the caller's DAZ
 * (read once from the MXCSR): -0, a negative denormal under DAZ, an infinity or a
 * NaN; else 0. It reads in order and stops at the first group holding a match (16
 * words in the x86-64 baseline's SSE2 loop, 32 or 8 in the AVX2 path); serial.
 * Defined in image_setup_ops.c. */
int cn_exceptional_scan(const float *src, size_t count, cn_isa_level isa);
/* morphology_avx2.c: the same scan for ISA level avx2 or above, the caller's DAZ
 * passed in. */
int cn_exceptional_scan_avx2(const float *src, size_t count, int daz);
#endif
