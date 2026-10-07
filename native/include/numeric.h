#ifndef CHAINNER_NUMERIC_H
#define CHAINNER_NUMERIC_H
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* NaN bit patterns spelled out, never the SDK's quiet-NaN macro (0xFFC00000 up to
   Windows SDK 10.0.19041, 0x7FC00000 from 10.0.22621): NumPy's NPY_NANF, and
   x86's default NaN, which an invalid operation such as 0/0 or log(-1) gives. */
#define CN_NPY_NANF_BITS UINT32_C(0x7FC00000)
#define CN_DEFAULT_NANF_BITS UINT32_C(0xFFC00000)
static inline float cn_f32_from_bits(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof value);
    return value;
}
/* +infinity spelled out, never the SDK's INFINITY macro: the UCRT defines it as
   (float)1e+300, a conversion that the strict-FP build performs at run time, raising
   overflow inside the kernels' FP-flag windows (MSVC folded it; clang /fp:strict may
   not fold an inexact conversion). */
#define CN_INFINITY_F cn_f32_from_bits(UINT32_C(0x7F800000))

/* Internal numerical contracts shared by statistics and palette extraction.
   Sum uses NumPy float32 pairwise blocks: the inner-loop calls NumPy's reduce
   iterator makes (native_numpy_reduce.numpy_sum_block). The calls cut each run of
   period elements (0 = all n) into blocks of block elements (0 = the whole run),
   the last one shorter: NumPy never buffers past its outer dimension. No
   allocation/global state. */
float cn_numpy_sum_f32(const float *x, size_t n, size_t stride, size_t block, size_t period);
/* np.min (maximum 0) or np.max (1) of contiguous x: lanes 4/8/16 for NumPy's
   SIMD path at that float32 width, 1 for its strided scalar path; block and
   period as in cn_numpy_sum_f32. n > 0. */
float cn_numpy_extreme_f32(const float *x, size_t n, int lanes, int maximum, size_t block,
                           size_t period);
/* np.partition of one contiguous float32 lane for count ascending, normalized
   kth (NumPy's introselect, NaN last). n > 0, every kth < n. */
void cn_numpy_partition_f32(float *v, size_t n, const size_t *kth, size_t count);
/* extract_unique_const's sort key (chainner_ext 0.3.10 image-ops palette) for 1, 3 or
   4 channels: the color, or r r 0.2126 + g g 0.7152 + b b 0.0722 (+ a 10). A NaN key's
   bits are the ones the real module's compiled sums give, measured on every ordered
   pair of NaN channels: alpha's term, then blue's, then red's, then green's NaN wins,
   quieted, and +inf + -inf gives the default NaN. They are derived from the channels'
   bits after the arithmetic, never read from it: the compiler orders a sum's operands
   either way, and a NaN key's sign and payload order the palette (f32::total_cmp). */
static inline float cn_palette_key(const float *color, size_t channels) {
    static const size_t order[2][4] = {{2, 0, 1, 0}, {3, 2, 0, 1}};
    if (channels == 1) return color[0];
    float key = color[0] * color[0] * 0.2126f + color[1] * color[1] * 0.7152f +
        color[2] * color[2] * 0.0722f;
    if (channels == 4) key += color[3] * 10.0f;
    uint32_t bits;
    memcpy(&bits, &key, sizeof bits);
    if ((bits & UINT32_C(0x7FFFFFFF)) <= UINT32_C(0x7F800000)) return key;
    for (size_t k = 0; k < channels; ++k) {
        memcpy(&bits, &color[order[channels == 4][k]], sizeof bits);
        if ((bits & UINT32_C(0x7FFFFFFF)) > UINT32_C(0x7F800000))
            return cn_f32_from_bits(bits | UINT32_C(0x00400000));
    }
    return cn_f32_from_bits(CN_DEFAULT_NANF_BITS);
}
/* NumPy 2.5.3 _methods._mean/_var divide the float32 sum by an np.intp count:
   in float64, then rounded to float32 (not exact through float32 past 2^24). */
static inline float numpy_mean_f32(float sum, size_t n) {
    return (float)((double)sum / (double)n);
}
#ifdef __cplusplus
}
#endif
#endif
