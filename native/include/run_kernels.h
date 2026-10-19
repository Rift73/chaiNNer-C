/* The chainner_native.dll entries that the run engine and the native node entry points (native/src/run) call,
 * grouped by the kernel file that defines and documents them; that file includes this header, so a declaration
 * cannot drift from its definition. A node lane adds its kernels under their file's heading (kept in file order). */
#ifndef CHAINNER_RUN_KERNELS_H
#define CHAINNER_RUN_KERNELS_H
#include "chainner.h"
#include <stddef.h>

#if defined(_WIN32) && defined(CN_TENSOR_IMPORTS)
#define CN_RUN_KERNEL __declspec(dllimport)
#else
#define CN_RUN_KERNEL CN_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif
/* analysis_ops.c */
CN_RUN_KERNEL cn_status cn_analysis_distinct(const float *src, float *out,
    size_t pixels, size_t channels, size_t *count);
/* buffers.c */
CN_RUN_KERNEL cn_status cn_pixels_convert_checked(const void *src, void *out, size_t count,
    int type, int output, int normalize, int *events);
CN_RUN_KERNEL cn_status cn_pixels_normalized_f32(const float *src, size_t count, int *same);
/* channels.c */
/* A null source denotes a constant channel. The Python boundary owns every
 * source buffer until the synchronous parallel call returns. */
typedef struct cn_channel_source {
    const float *data;
    size_t channels, channel;
    float constant;
} cn_channel_source;
CN_RUN_KERNEL cn_status cn_channels_assemble(const cn_channel_source *sources,
    float *out, size_t pixels, size_t channels);
CN_RUN_KERNEL cn_status cn_channels_flip(const float *src, float *out,
    size_t height, size_t width, size_t channels, int axis);
/* composite_complete_ops.c */
typedef struct canvas_layer { const float *data; size_t height, width, channels; int constant; } canvas_layer;
typedef struct canvas_geometry {
    size_t height, width, channels, base_top, base_left, paste_top, paste_left;
    size_t overlay_top, overlay_left, region_height, region_width;
    int padded;
} canvas_geometry;
CN_RUN_KERNEL cn_status cn_composite_canvas(const canvas_layer *base, const canvas_layer *overlay,
    const canvas_geometry *g, float *out, int mode);
/* convolution_ops.c */
CN_RUN_KERNEL cn_status cn_convolution_spatial(const float *src, float *out,
    size_t h, size_t w, size_t c, const float *kernel,
    size_t kh, size_t kw, size_t padding, size_t fused_lanes);
/* cv_resize_ops.c */
CN_RUN_KERNEL cn_status cn_cv_resize_area(const float *src, float *out,
    size_t h, size_t w, size_t channels, size_t dh, size_t dw);
/* filter_ops.c */
CN_RUN_KERNEL cn_status cn_filter_lens_normalize(const float *complex_values,
    const double *parameters, float *out, size_t components, size_t length);
CN_RUN_KERNEL cn_status cn_filter_morphology(const float *src, float *out,
    size_t height, size_t width, size_t channels, size_t radius, size_t iterations,
    int cross, int maximum);
/* gaussian_complete_ops.c */
CN_RUN_KERNEL cn_status cn_gaussian_f32(const float *src, float *dst, size_t height,
    size_t width, size_t channels, size_t nx, size_t ny,
    double sigma_x, double sigma_y, int border, size_t lanes);
/* image_setup_ops.c */
CN_RUN_KERNEL cn_status cn_image_exceptional(const float *src, size_t count, int *result);
CN_RUN_KERNEL cn_status cn_image_distinct_exceeds(const float *src, size_t pixels,
    size_t channels, size_t limit, int *result);
/* kmeans_ops.c */
CN_RUN_KERNEL cn_status cn_palette_kmeans(const float *data, float *out,
    size_t pixels, size_t channels, size_t colors);
/* lens_complete_ops.c */
CN_RUN_KERNEL cn_status cn_lens_kernel(float *out, size_t radius,
    float scale, float a, float b, int use_fma);
CN_RUN_KERNEL cn_status cn_lens_power(const float *src, float *out, size_t count,
    float exponent, int finish);
CN_RUN_KERNEL cn_status cn_lens_compose(const float *f1, const float *f2,
    const float *f3, const float *f4, float *out, size_t count,
    float a, float b, int accumulate);
/* morphology_ops.c */
CN_RUN_KERNEL cn_status cn_morphology_complete(const float *src, float *out,
    size_t height, size_t width, size_t channels, size_t radius, size_t iterations,
    int shape, int maximum, size_t lanes);
/* neighborhood_ops.c (its other palette entries: chainner_ext_kernels.h) */
CN_RUN_KERNEL cn_status cn_neighborhood_palette_create(const float *colors,
    size_t count, size_t channels, void **out, size_t *bytes);
/* palette_ops.c */
CN_RUN_KERNEL cn_status cn_palette_median_cut(const float *src, float *out,
    size_t pixels, size_t channels, size_t capacity,
    size_t root_block, size_t root_period, size_t child_block, size_t *written);
CN_RUN_KERNEL cn_status cn_palette_pad_edge(const float *source, float *out,
    size_t colors, size_t channels, size_t capacity);
/* resample_ops.c (nearest and filtered: chainner_ext_kernels.h) */
CN_RUN_KERNEL cn_status cn_resample_alpha(const float *src, float *out,
    size_t pixels, int undo);
#ifdef __cplusplus
}
#endif

#endif
