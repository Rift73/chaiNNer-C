/* np.clip of float32 with constant bounds, as NumPy 2.5.3 computes it
 * (umath/clip.cpp, _npy_clip_const_minmax_): a NaN bound fills every element with
 * the NaN one (low's first); else x < low gives low, then x > high gives high, so a
 * NaN x and a zero equal to a bound pass. Its build runs that as maxps(low, x) then
 * minps(high, x); maxss and minss give the same bits, the caller's DAZ included (a
 * denormal x is taken as a zero of its sign). The loop clears the FP status, so no
 * caller reports one for a clip; a NaN x returns as stored, as the max and min give it. */
#ifndef CHAINNER_NUMPY_CLIP_F32_H
#define CHAINNER_NUMPY_CLIP_F32_H
#include <math.h>
#include <xmmintrin.h>
static inline float cn_numpy_clip_f32(float x, float low, float high)
{
    if (isnan(low) || isnan(high)) return isnan(low) ? low : high;
    if (isnan(x)) return x;
    __m128 value = _mm_max_ss(_mm_set_ss(low), _mm_set_ss(x));
    return _mm_cvtss_f32(_mm_min_ss(_mm_set_ss(high), value));
}
#endif
