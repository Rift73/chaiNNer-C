/* The C runtime math that mirrors share with their reference libraries (Consult 6 D-4).
 *
 * Every target links the static UCRT. NumPy, SciPy, CPython and ncnn link it
 * dynamically: their scalar math is the one ucrtbase.dll of the process, which
 * Windows services. A mirror of such a call takes the same export of that loaded
 * module through cn_crt, so it follows the reference on every Windows version by
 * construction; the static UCRT agrees only while the build SDK's libucrt and the
 * user's ucrtbase.dll give the same bits. The call sites form three groups:
 *  1. A mirror of a reference library's own CRT call that reaches ucrtbase.dll
 *     (NumPy's scalar loops and npymath, its pocketfft, linalg and Generator code,
 *     SciPy's _nd_image, CPython's float pow, and resize's filters' exp, sin and cos,
 *     which the real chainner_ext imports from it): calls through cn_crt.
 *  2. The chainner_ext-API kernels (alpha_complete_ops.c,
 *     alpha_gamma_ops.c, chainner_ext/dither.c, neighborhood_ops.c's Riemersma
 *     decay, threshold_complete_ops.c's coverage, distance_complete_ops.c): the
 *     static CRT. Their reference is this project's own module, whose contract is
 *     the tracked manifests, the same on every machine.
 *  3. Exact operations (sqrt, fmod, floor, ceil, fma, fabs, copysign): the static
 *     CRT; every conforming implementation gives the same bits.
 * OpenCV's mirrors (bilateral_ops.c, denoise_ops.cpp) are not group 1: cv2.pyd
 * links its own static CRT, so its exp does not follow ucrtbase.dll.
 *
 * chainner_native.dll fills the table as it loads (crt_math.c): GetModuleHandleW
 * finds the ucrtbase.dll that python3xx.dll loaded (never a LoadLibrary), and a
 * missing module or export fails the load with a message on stderr. The pyds read
 * the DLL's exported table. Outside Windows the table holds the libm functions.
 * Each entry is named as its member; the export differs where noted. */
#ifndef CHAINNER_CN_CRT_MATH_H
#define CHAINNER_CN_CRT_MATH_H
#include "chainner.h"

#if defined(_WIN32) && defined(CN_TENSOR_IMPORTS)
#define CN_CRT_API __declspec(dllimport)
#else
#define CN_CRT_API CN_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif
/* The layout of the CRT's struct _complex. */
typedef struct cn_crt_complex { double real, imag; } cn_crt_complex;

typedef struct cn_crt_table {
    double (*cos)(double);
    double (*sin)(double);
    double (*exp)(double);
    double (*log)(double);
    double (*log1p)(double);
    double (*log2)(double);
    double (*atan2)(double, double);
    double (*hypot)(double, double);
    double (*pow)(double, double);
    double (*cabs)(cn_crt_complex); /* _cabs: NumPy's npy_cabs on Win64 */
    float (*expf)(float);
    float (*logf)(float);
    float (*powf)(float, float);
    float (*log10f)(float);
    float (*atan2f)(float, float);
    float (*hypotf)(float, float); /* _hypotf: the CRT's hypotf is an inline wrapper */
    float (*asinf)(float);
    float (*cosf)(float);
    float (*sinf)(float);
} cn_crt_table;

CN_CRT_API extern cn_crt_table cn_crt;
#ifdef __cplusplus
}
#endif
#endif
