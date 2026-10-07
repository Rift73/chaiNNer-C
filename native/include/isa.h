#ifndef CHAINNER_ISA_H
#define CHAINNER_ISA_H

#include "chainner.h"

/* The ISA level decides only which implementation runs a kernel's operation
 * sequence; mirror parameters (lanes, fused) decide the sequence and its bits.
 * avx2: AVX2, FMA3, BMI1, BMI2 with OS-saved YMM state (XCR0 bits 1-2).
 * avx512: avx2 plus AVX-512 F, CD, BW, DQ, VL with OS-saved opmask and ZMM
 * state (XCR0 bits 5-7). */
typedef enum cn_isa_level { CN_ISA_SCALAR = 0, CN_ISA_AVX2 = 1, CN_ISA_AVX512 = 2 } cn_isa_level;

#ifdef __cplusplus
extern "C" {
#endif

/* The effective level. The first call runs the probe once: CPUID, XGETBV and
 * CHAINNER_C_ISA (exactly scalar, avx2 or avx512; unset or anything else is
 * auto), capped at the CPU's maximum. A C entry reads it once per call and
 * passes it down, so one call never mixes levels. Non-MSVC-x64 builds: scalar. */
cn_isa_level cn_isa_current(void);
/* out[0] the effective level, out[1] the CHAINNER_C_ISA request (-1 = auto),
 * which cn_isa_set leaves unchanged, out[2] the CPU's maximum. */
CN_EXPORT cn_status cn_isa_get(int *out);
/* Test-only, called while no kernel runs: runs the probe, then sets the
 * effective level to min(level, CPU maximum) for calls that start afterwards and
 * returns it. A level outside 0-2 returns -1 and changes nothing. */
CN_EXPORT int cn_isa_set(int level);
/* Test-only and pure: the level the probe derives from these CPUID words and
 * XCR0 (0 when OSXSAVE is clear). leaf7_ebx counts only when max_leaf >= 7. */
CN_EXPORT int cn_isa_classify(uint32_t max_leaf, uint32_t leaf1_ecx, uint32_t leaf7_ebx, uint64_t xcr0);

#ifdef __cplusplus
}
#endif

#endif
