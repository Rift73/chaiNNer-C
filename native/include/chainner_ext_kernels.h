/* The chainner_native.dll entries that chainner_ext.pyd binds: the kernels its image API
 * shares with the node bridges (one implementation each). Every entry is defined, and
 * documented, in the kernel file named beside it; that file includes this header, so a
 * declaration cannot drift from its definition. */
#ifndef CHAINNER_EXT_KERNELS_H
#define CHAINNER_EXT_KERNELS_H
#include "chainner.h"
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(CN_TENSOR_IMPORTS)
#define CN_EXT_KERNEL __declspec(dllimport)
#else
#define CN_EXT_KERNEL CN_EXPORT
#endif

#ifdef __cplusplus
/* pixel_art_ops.cpp defines its entry noexcept, part of a C++ function's type. */
#define CN_EXT_NOEXCEPT noexcept
extern "C" {
#else
#define CN_EXT_NOEXCEPT
#endif
/* alpha_complete_ops.c */
CN_EXT_KERNEL cn_status cn_alpha_fragment(const float *src, float *dst,
    size_t height, size_t width, float threshold, size_t iterations, size_t count);
CN_EXT_KERNEL cn_status cn_alpha_nearest(const float *src, float *dst,
    size_t height, size_t width, float threshold, size_t radius, int anti_aliasing);
/* alpha_gamma_ops.c */
CN_EXT_KERNEL cn_status cn_alpha_gamma(const float *src, float *out,
    size_t pixels, size_t channels, float gamma);
CN_EXT_KERNEL cn_status cn_alpha_extend(const float *src, float *out,
    size_t height, size_t width, float threshold, size_t iterations);
/* threshold_complete_ops.c */
CN_EXT_KERNEL cn_status cn_threshold_aa(const float *src, float *dst,
    size_t height, size_t width, size_t channels, float threshold, float smoothness);
CN_EXT_KERNEL cn_status cn_threshold_hard(const float *src, float *dst, size_t count,
    int type, float threshold, float maximum, int ipp);
/* distance_complete_ops.c */
CN_EXT_KERNEL cn_status cn_distance_esdf_ex(const float *source, float *out,
    size_t height, size_t width, float radius, float cutoff, int pre_process, int post_process);
/* pixel_art_ops.cpp */
CN_EXT_KERNEL cn_status cn_pixel_art_f32(const float *src, float *dst,
    size_t height, size_t width, size_t channels, int mode) CN_EXT_NOEXCEPT;
/* neighborhood_ops.c. compatible, on the palette entries: optional, written 1 when given
   (every input dithers since Consult 8 D-5; kept for the B3 goldens, Consult 10 D-8). */
CN_EXT_KERNEL cn_status cn_neighborhood_dither(const float *src, float *out,
    size_t h, size_t w, size_t c, uint32_t colors, int mode, size_t map_size,
    int algorithm, const float *palette, size_t palette_count, int *compatible);
CN_EXT_KERNEL cn_status cn_neighborhood_uniform_riemersma(const float *src, float *out,
    size_t h, size_t w, size_t c, uint32_t colors, uint32_t history_length, float decay_ratio);
CN_EXT_KERNEL cn_status cn_neighborhood_palette_create_ordered(const float *colors,
    size_t count, size_t channels, void **out, size_t *bytes);
CN_EXT_KERNEL void cn_neighborhood_palette_free(void *handle);
CN_EXT_KERNEL cn_status cn_neighborhood_palette_apply(const float *src, float *out,
    size_t h, size_t w, size_t c, const void *handle, int mode, int algorithm,
    uint32_t history_length, float decay_ratio, int *compatible);
/* resample_ops.c, resample_filters.c */
CN_EXT_KERNEL cn_status cn_resample_nearest(const float *src, float *out,
    size_t h, size_t w, size_t c, size_t out_h, size_t out_w);
CN_EXT_KERNEL cn_status cn_resample_filtered(const float *source, float *out,
    size_t height, size_t width, size_t channels, size_t target_height,
    size_t target_width, int filter, int gamma, int vector_clip);
#ifdef __cplusplus
}
#endif
#endif
