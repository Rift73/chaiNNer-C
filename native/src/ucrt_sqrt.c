/* Linux only: sqrt and sqrtf with the Windows build's results and floating-point flags. There, every module calls
 * the static UCRT's sqrt and sqrtf out of line (clang-cl's /fp:strict keeps math errno, so they are not inlined),
 * and the UCRT classifies its input by its bits, not by comparisons, which DAZ would bend:
 * - a NaN comes back quiet (its quiet bit set, sign and payload kept) and raises nothing, a signaling one included;
 * - a negative value other than -0 (a negative denormal under DAZ included) gives the default NaN and raises invalid
 *   alone, as the instruction does on -1;
 * - anything else (+-0, +inf, positive values) is the instruction's own result and flags under the caller's MXCSR
 *   (DAZ reads a positive denormal as +0).
 * Measured on ucrtbase.dll and the static UCRT of the pinned toolset over the special values with DAZ and FTZ each
 * off and on, results and MXCSR flags alike. cmake/linux.cmake builds this unit outside ThinLTO, with hidden
 * visibility, into every module, and compiles every unit with -fno-builtin-sqrt and -fno-builtin-sqrtf, so that
 * their calls reach these definitions and never glibc's (which compares, so a negative denormal under DAZ gives -0)
 * or an inlined instruction. Explicit SIMD square roots are untouched. */
#include <stdint.h>
#include <string.h>

double sqrt(double x);
float sqrtf(float x);

double sqrt(double x)
{
    uint64_t bits;
    memcpy(&bits, &x, sizeof bits);
    if ((bits & UINT64_C(0x7FFFFFFFFFFFFFFF)) > UINT64_C(0x7FF0000000000000)) {
        bits |= UINT64_C(0x0008000000000000);
        memcpy(&x, &bits, sizeof x);
        return x;
    }
    if (bits >> 63 && bits != UINT64_C(0x8000000000000000)) x = -1.0;
    __asm__ volatile("sqrtsd %1, %0" : "=x"(x) : "x"(x));
    return x;
}

float sqrtf(float x)
{
    uint32_t bits;
    memcpy(&bits, &x, sizeof bits);
    if ((bits & UINT32_C(0x7FFFFFFF)) > UINT32_C(0x7F800000)) {
        bits |= UINT32_C(0x00400000);
        memcpy(&x, &bits, sizeof x);
        return x;
    }
    if (bits >> 31 && bits != UINT32_C(0x80000000)) x = -1.0f;
    __asm__ volatile("sqrtss %1, %0" : "=x"(x) : "x"(x));
    return x;
}
