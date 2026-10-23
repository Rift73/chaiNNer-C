/* Linear Light's power for images and its AVX2 unit (linear_light_avx2.c), reached only through cn_linear_light's
 * dispatch on the call's ISA level (linear_light_ops.c). Both evaluate a^g = 2^(g * log2(a)) in double with the same
 * operations in the same order, so they give the same bits: log2(a) from the exponent field plus 2 atanh(s) / ln 2
 * on the mantissa m folded into [sqrt(1/2), sqrt(2)) (s = (m - 1) / (m + 1), |s| <= 0.1716, the series through
 * s^15), then 2^t as 2^n exp((t - n) ln 2) with n the nearest integer and the exponential's series through the 11th
 * power, both polynomials in Estrin's order (light_poly). Its relative error stays below 1e-13 for the positive
 * doubles a float32 gives, so a float32 result is pow's except within that distance of a rounding tie; 0, infinity
 * and NaN pass through as pow gives them for g > 0. */
#ifndef CHAINNER_LINEAR_LIGHT_SHARED_H
#define CHAINNER_LINEAR_LIGHT_SHARED_H
#include <stddef.h>

#define LIGHT_LN2 0x1.62e42fefa39efp-1
#define LIGHT_INV_LN2 0x1.71547652b82fep0
#define LIGHT_SQRT2 0x1.6a09e667f3bcdp0
/* 2^52 + 1023: an exponent field ORed into 2^52's bits, minus this, is the unbiased exponent. */
#define LIGHT_EXPONENT_BIAS 4503599627371519.0
/* 1.5 * 2^52: adding and subtracting it rounds a double below 2^51 to the nearest integer. */
#define LIGHT_ROUND 0x1.8p52

/* The atanh series in s^2: 1 / (2k + 1), k = 0 to 7. */
static const double light_log_terms[8] = {1.0, 1.0 / 3, 1.0 / 5, 1.0 / 7, 1.0 / 9, 1.0 / 11, 1.0 / 13, 1.0 / 15};
/* The exponential series: 1 / k!, k = 0 to 11. */
static const double light_exp_terms[12] = {1.0,          1.0,          1.0 / 2.0,        1.0 / 6.0,
                                           1.0 / 24.0,   1.0 / 120.0,  1.0 / 720.0,      1.0 / 5040.0,
                                           1.0 / 40320.0, 1.0 / 362880.0, 1.0 / 3628800.0, 1.0 / 39916800.0};

/* values[i] for i in [0, count), count a multiple of 4 starting at a pixel: each converted as cn_linear_light's
 * scalar path converts it (linear_light.h's curve with this power), the last channel of a pixel left as is when
 * skip_alpha (channels 2 or 4). */
void cn_light_values_avx2(float *values, size_t count, size_t channels, int skip_alpha, int transfer, int encode);
#endif
