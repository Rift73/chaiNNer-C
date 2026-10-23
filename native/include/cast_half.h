/* float32 -> float16 casts' AVX2 lanes (cast_half_avx2.c) for cn_tensor_cast_typed. */
#ifndef CHAINNER_CAST_HALF_H
#define CHAINNER_CAST_HALF_H
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Converts n float32 values (src, from bytes apart) to float16 bits (out, to bytes
 * apart) in groups of 8, as long as every value of a group is +-0 or rounds to a
 * normal finite half (|x| in [2^-14, 65520)): the bits the scalar cast gives, which
 * raises no condition for them. Stops before the first group holding another value
 * (and before a tail under 8) and returns how many values it converted; the caller
 * converts that group the scalar way. The caller ensures ISA level avx2 or above. */
size_t cn_float_to_half_avx2(const char *src, ptrdiff_t from, char *out, ptrdiff_t to,
                             size_t n);

#ifdef __cplusplus
}
#endif

#endif
