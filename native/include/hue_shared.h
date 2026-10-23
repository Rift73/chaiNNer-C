/* The float colour conversions' AVX2 code (color_avx2.c), reached only through color_range's dispatch on the call's
 * ISA level (color_complete_ops.c). */
#ifndef CHAINNER_HUE_SHARED_H
#define CHAINNER_HUE_SHARED_H
#include <stddef.h>
#include <stdint.h>

/* rgb_hue's vector form at 8 lanes (fused multiply-add) for count pixels (a multiple of 8) of scn samples at src
 * into 3 samples each at dst: (h, l, s) when hls, else (h, s, v); blue is the blue sample's index (0 or 2). Each lane
 * runs rgb_hue's operations in its order: vminps/vmaxps for its minimum/maximum (the second operand on ties and
 * NaN, as theirs), vdivps, vfmadd231ps for its fmaf, compare masks and blends for its conditions. */
void cn_rgb_hue_avx2(const float *src, float *dst, size_t count, size_t scn, int blue, int hls);
/* hue_rgb's vector form for count pixels (a multiple of 8) of (h, l, s) when hls, else (h, s, v), at src into dcn
 * samples each at dst (alpha 1 when dcn is 4): each lane runs hue_rgb's vector operations in its order, its
 * truncations as vcvttps2dq (INT_MIN outside the int range and for NaN, as truncate_int) and its selections as blends
 * of the same comparisons. */
void cn_hue_rgb_avx2(const float *src, float *dst, size_t count, size_t dcn, int blue, int hls);
/* The fused forms (vfmadd for each fmaf, in color_range's order) of BGR to grey (cb and cr the blue and red weights;
 * ipp: IPP's 8-pixel order, else OpenCV's vector one), BGR to YUV or YCrCb (3 samples out) and YUV or YCrCb to BGR
 * (3 samples in, dcn out, alpha 1), for count pixels (a multiple of 8). */
void cn_bgr_gray_avx2(const float *src, float *dst, size_t count, size_t scn, float cb, float cr, int ipp);
void cn_bgr_yuv_avx2(const float *src, float *dst, size_t count, size_t scn, int blue, int ycrcb);
void cn_yuv_bgr_avx2(const float *src, float *dst, size_t count, size_t dcn, int blue, int ycrcb);
/* rgb_lab's vector form for count pixels (a multiple of 8) of scn samples at src into (L, a, b) at dst: the clamp as
 * vmaxps/vminps, nearbyintf as vcvtps2dq (round to nearest even), the lattice's int16 entries gathered (lattice is
 * color_lab_tables.h's lab_lattice) and the trilinear sums in int32, exact integer arithmetic. */
void cn_bgr_lab_avx2(const float *src, float *dst, size_t count, size_t scn, int blue, const int16_t *lattice);
/* lab_rgb's vector form at OpenCV's baseline 4 lanes (no fused multiply-add) for count pixels (a multiple of 8) of
 * (L, a, b) at src into dcn samples each at dst (alpha 1 when dcn is 4): each lane runs its products and sums in its
 * order, its branches as blends of the same comparisons, the clip as vminps/vmaxps (a NaN clips to 1, as minimum)
 * and inverse_gamma's truncation as vcvttps2dq, the cubic's coefficients gathered (xyz is lab_xyz_to_rgb, gamma_table
 * lab_inverse_gamma). */
void cn_lab_bgr_avx2(const float *src, float *dst, size_t count, size_t dcn, int blue, const float *xyz,
                     const float *gamma_table);
#endif
