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
/* The entries that canny_ops.cpp and inpaint_ops.cpp define noexcept, part of a C++ function's type. */
#define CN_RUN_NOEXCEPT noexcept
extern "C" {
#else
#define CN_RUN_NOEXCEPT
#endif
/* adjustments.c */
CN_RUN_KERNEL cn_status cn_adjust_f32(const float *src, float *dst, size_t pixels,
    size_t channels, size_t channel_stride, int op, float a, float b, int should_clip);
CN_RUN_KERNEL cn_status cn_opacity_f32(const float *src, float *dst, size_t pixels,
    size_t channels, float opacity);
CN_RUN_KERNEL cn_status cn_levels_f32(const float *src, float *dst, size_t pixels,
    size_t channels, size_t channel_stride, unsigned int mask, float black, float white,
    float gamma, float out_black, float out_white);
CN_RUN_KERNEL cn_status cn_log_linear_f32(const float *src, float *dst, size_t count,
    float white, float gamma, float offset, float gain, float inverse_scale, int invert);
/* analysis_ops.c */
CN_RUN_KERNEL cn_status cn_analysis_mean(const float *src, size_t n, size_t block, size_t period,
    float *result);
CN_RUN_KERNEL cn_status cn_analysis_binary(const float *a, const float *b, float *out,
    size_t n, int op, float scale, float threshold);
CN_RUN_KERNEL cn_status cn_analysis_ssim(const float *mu1, const float *mu2,
    const float *v1, const float *v2, const float *cov, float *out, size_t n);
CN_RUN_KERNEL cn_status cn_analysis_palette(const float *image, const float *palette,
    float *out, size_t pixels, size_t levels, size_t channels);
CN_RUN_KERNEL cn_status cn_analysis_distinct(const float *src, float *out,
    size_t pixels, size_t channels, size_t *count);
CN_RUN_KERNEL cn_status cn_analysis_cas(const void *low, const void *high,
    const float *image, const float *sharpened, void *out, size_t pixels, size_t channels,
    double exponent, int is_double, int is_mix);
CN_RUN_KERNEL cn_status cn_analysis_channel_stats(const float *image, size_t pixels,
    size_t block, size_t period, float *out);
CN_RUN_KERNEL cn_status cn_analysis_transfer(const float *image, float *out,
    const unsigned char *mask, size_t pixels, const float *stats, const float *reference_stats,
    const float *limits, int reciprocal, int scale);
/* blue_noise_ops.c */
CN_RUN_KERNEL cn_status cn_blue_from_permutation(const int64_t *permutation, int32_t *out,
    size_t height, size_t width, double sigma_y, double sigma_x, double initial_fraction);
CN_RUN_KERNEL cn_status cn_blue_normalize(const int32_t *ranks, void *out, size_t count,
    int double_output);
/* box_complete_ops.c */
CN_RUN_KERNEL cn_status cn_box_mean_u8(const uint8_t *src, uint8_t *dst,
    size_t height, size_t width, size_t radius);
CN_RUN_KERNEL cn_status cn_box_separable(const float *src, float *dst,
    size_t height, size_t width, size_t channels, double radius_x, double radius_y, size_t lanes);
/* buffers.c */
CN_RUN_KERNEL cn_status cn_pixels_convert_checked(const void *src, void *out, size_t count,
    int type, int output, int normalize, int *events);
CN_RUN_KERNEL cn_status cn_pixels_normalized_f32(const float *src, size_t count, int *same);
/* canny_ops.cpp */
CN_RUN_KERNEL cn_status cn_canny_u8_policy(const uint8_t *source, uint8_t *output,
    size_t height, size_t width, size_t channels, double lower, double upper,
    int ipp_thresholds) CN_RUN_NOEXCEPT;
/* cas_complete_ops.c */
CN_RUN_KERNEL cn_status cn_cas_luminance(const float *src, double *out,
    size_t count, size_t channels, int fused);
CN_RUN_KERNEL cn_status cn_cas_extrema_f64(const double *src, double *low, double *high,
    size_t h, size_t w, const uint8_t *kernel, size_t kh, size_t kw);
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
CN_RUN_KERNEL cn_status cn_channels_shift(const float *src, float *out,
    size_t height, size_t width, size_t channels, int64_t dx, int64_t dy, int fill);
CN_RUN_KERNEL cn_status cn_channels_bbox(const float *src, size_t height, size_t width,
    size_t channels, size_t channel, float threshold, size_t *bounds);
CN_RUN_KERNEL cn_status cn_channels_crop(const float *src, float *out, size_t height,
    size_t width, size_t channels, size_t x, size_t y, size_t crop_width, size_t crop_height,
    size_t out_width);
/* color_complete_ops.c */
CN_RUN_KERNEL cn_status cn_color_convert_f32(const float *src, float *dst,
    size_t height, size_t width, size_t scn, size_t dcn, int op, int blue, int lanes, int ipp);
/* color_ops.c */
CN_RUN_KERNEL cn_status cn_color_linear(const float *src, float *dst, size_t count,
    float factor, float offset);
CN_RUN_KERNEL cn_status cn_hls_adjust(float *image, size_t pixels, float hue,
    float saturation_factor);
CN_RUN_KERNEL cn_status cn_threshold_f32(const float *src, const float *binary,
    float *dst, size_t count, int type, float threshold, float maximum);
CN_RUN_KERNEL cn_status cn_auto_threshold(const float *src, size_t pixels,
    size_t channels, int method, int *out);
CN_RUN_KERNEL cn_status cn_adaptive_apply(const uint8_t *src, const uint8_t *mean,
    uint8_t *dst, size_t count, int delta, int inverted, int maximum);
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
CN_RUN_KERNEL cn_status cn_convolution_box(const float *src, float *out,
    size_t h, size_t w, size_t c, size_t rx, size_t ry);
CN_RUN_KERNEL cn_status cn_convolution_spatial(const float *src, float *out,
    size_t h, size_t w, size_t c, const float *kernel,
    size_t kh, size_t kw, size_t padding, size_t fused_lanes);
/* cv_resize_ops.c */
CN_RUN_KERNEL cn_status cn_cv_resize_area(const float *src, float *out,
    size_t h, size_t w, size_t channels, size_t dh, size_t dw);
/* denoise_ops.cpp */
CN_RUN_KERNEL cn_status cn_denoise_u8(const uint8_t *src, uint8_t *out,
    size_t height, size_t width, size_t channels, float strength, float color_strength,
    int patch, int search);
/* descale_ops.c */
CN_RUN_KERNEL void *cn_descale_core_create(int src_dim, int dst_dim, int mode, int taps, double b,
    double c, double blur, double shift, double active_dim, int border, int *mirror_outside);
CN_RUN_KERNEL void cn_descale_core_free(void *core);
CN_RUN_KERNEL size_t cn_descale_workspace(size_t h, size_t w, size_t c, void *core_h, void *core_v);
CN_RUN_KERNEL cn_status cn_descale(const float *src, size_t h, size_t w, size_t c, void *core_h,
    void *core_v, float *dst, void *workspace, size_t workspace_bytes);
/* distance_complete_ops.c */
CN_RUN_KERNEL cn_status cn_distance_esdf(const float *source, float *out,
    size_t height, size_t width, float radius);
CN_RUN_KERNEL cn_status cn_distance_binary(const uint8_t *source, float *out,
    size_t height, size_t width, double spread, int double_division);
/* filter_ops.c */
CN_RUN_KERNEL cn_status cn_filter_arithmetic(const float *a, const float *b,
    const float *c, const float *d, float *out, size_t n, int operation, float scale,
    float second, int clip);
CN_RUN_KERNEL cn_status cn_filter_lens_normalize(const float *complex_values,
    const double *parameters, float *out, size_t components, size_t length);
CN_RUN_KERNEL cn_status cn_filter_morphology(const float *src, float *out,
    size_t height, size_t width, size_t channels, size_t radius, size_t iterations,
    int cross, int maximum);
CN_RUN_KERNEL cn_status cn_filter_quantize_reference(const float *src, const float *reference,
    float *out, size_t height, size_t width, size_t ref_height, size_t ref_width,
    size_t channels, size_t radius, double spatial);
/* gaussian_complete_ops.c */
CN_RUN_KERNEL cn_status cn_gaussian_f32(const float *src, float *dst, size_t height,
    size_t width, size_t channels, size_t nx, size_t ny,
    double sigma_x, double sigma_y, int border, size_t lanes);
CN_RUN_KERNEL cn_status cn_ssim_window(double *out);
CN_RUN_KERNEL cn_status cn_gaussian_mean_u8(const uint8_t *src, uint8_t *dst,
    size_t height, size_t width, size_t radius, size_t lanes);
/* generation.c */
CN_RUN_KERNEL cn_status cn_image_fill(float *out, size_t height, size_t width,
    size_t channels, const float *a, const float *b, const float *gradient,
    int mode, size_t square);
/* gradients.c */
CN_RUN_KERNEL cn_status cn_gradient(float *out, size_t height, size_t width,
    int kind, double a, double b, int wide_start);
/* image_setup_ops.c */
CN_RUN_KERNEL cn_status cn_image_exceptional(const float *src, size_t count, int *result);
CN_RUN_KERNEL cn_status cn_image_distinct_exceeds(const float *src, size_t pixels,
    size_t channels, size_t limit, int *result);
/* inpaint_ops.cpp */
CN_RUN_KERNEL cn_status cn_inpaint_u8(const uint8_t *source, const uint8_t *input_mask,
    uint8_t *output, size_t height, size_t width, size_t channels, double radius,
    int method) CN_RUN_NOEXCEPT;
/* kmeans_ops.c */
CN_RUN_KERNEL cn_status cn_palette_kmeans(const float *data, float *out,
    size_t pixels, size_t channels, size_t colors);
/* layout_ops.c */
CN_RUN_KERNEL cn_status cn_layout_pad(const float *src, float *out, const float *color,
    size_t h, size_t w, size_t c, size_t out_h, size_t out_w, size_t out_c,
    size_t top, size_t left, int mode);
CN_RUN_KERNEL cn_status cn_layout_paste(const float *src, float *out,
    size_t h, size_t w, size_t c, size_t out_h, size_t out_w, size_t out_c,
    size_t top, size_t left, size_t target_h, size_t target_w, int expand_gray);
CN_RUN_KERNEL cn_status cn_layout_zstack(const float *const *sources, float *out,
    size_t elements, size_t count, int mode);
CN_RUN_KERNEL cn_status cn_layout_pixelate(const float *src, float *out,
    size_t h, size_t w, size_t c, size_t sx, size_t sy);
CN_RUN_KERNEL cn_status cn_layout_rotate_u8(const uint8_t *src, float *out,
    size_t h, size_t w, size_t c, size_t out_c, int quarter);
/* lens_complete_ops.c */
CN_RUN_KERNEL cn_status cn_lens_kernel(float *out, size_t radius,
    float scale, float a, float b, int use_fma);
CN_RUN_KERNEL cn_status cn_lens_power(const float *src, float *out, size_t count,
    float exponent, int finish);
CN_RUN_KERNEL cn_status cn_lens_compose(const float *f1, const float *f2,
    const float *f3, const float *f4, float *out, size_t count,
    float a, float b, int accumulate);
/* matting_complete_ops.c */
CN_RUN_KERNEL cn_status cn_matting_erode_mask_u8(const uint8_t *source, uint8_t *out,
    size_t height, size_t width, size_t radius);
CN_RUN_KERNEL cn_status cn_matting_trimap_output(const float *image, size_t pixel_stride,
    size_t channel_stride, const double *trimap, size_t pixels, float *out,
    size_t out_pixel_stride, size_t out_channel_stride);
/* morphology_ops.c */
CN_RUN_KERNEL cn_status cn_morphology_complete(const float *src, float *out,
    size_t height, size_t width, size_t channels, size_t radius, size_t iterations,
    int shape, int maximum, size_t lanes);
/* neighborhood_ops.c (its other palette entries: chainner_ext_kernels.h) */
CN_RUN_KERNEL cn_status cn_neighborhood_median_f32(const float *src, float *out,
    size_t h, size_t w, size_t c, size_t radius, size_t vector_lanes);
CN_RUN_KERNEL cn_status cn_neighborhood_median_u8(const uint8_t *src, uint8_t *out,
    size_t h, size_t w, size_t c, size_t radius);
CN_RUN_KERNEL cn_status cn_neighborhood_palette_create(const float *colors,
    size_t count, size_t channels, void **out, size_t *bytes);
/* noise_complete_ops.c */
CN_RUN_KERNEL cn_status cn_seeded_permutation(const uint32_t *seed, size_t words,
    int64_t *out, size_t count);
CN_RUN_KERNEL cn_status cn_add_seeded_noise(const float *image, size_t pixels,
    size_t channels, size_t noise_channels, int mode, double amount,
    const uint32_t *seed, size_t words, float *out);
/* palette_ops.c */
CN_RUN_KERNEL cn_status cn_palette_median_cut(const float *src, float *out,
    size_t pixels, size_t channels, size_t capacity,
    size_t root_block, size_t root_period, size_t child_block, size_t *written);
CN_RUN_KERNEL cn_status cn_palette_pad_edge(const float *source, float *out,
    size_t colors, size_t channels, size_t capacity);
/* procedural_complete_ops.c */
CN_RUN_KERNEL cn_status cn_create_procedural(void *out, size_t height, size_t width,
    uint64_t seed, int method, double scale, double brightness,
    int horizontal, int vertical, int spherical, int layers,
    double scale_ratio, double brightness_ratio, int increment_seed);
/* resample_ops.c (nearest and filtered: chainner_ext_kernels.h) */
CN_RUN_KERNEL cn_status cn_resample_alpha(const float *src, float *out,
    size_t pixels, int undo);
CN_RUN_KERNEL cn_status cn_resample_border(const float *src, size_t h, size_t w,
    size_t c, float tolerance, int mode, size_t *bounds, int pairwise);
/* rotate_ops.c */
CN_RUN_KERNEL cn_status cn_rotate_affine_u8(const uint8_t *source, float *out,
    size_t height, size_t width, size_t channels, size_t out_height, size_t out_width,
    const double *matrix, int interpolation, const uint8_t *fill, int quarter);
/* statistics_ops.c */
CN_RUN_KERNEL cn_status cn_analysis_statistics(const float *src, const float *quantile_src,
    size_t n, double percentile, int lanes, size_t block, size_t period, int weak,
    double *result);
/* transfer_complete_ops.c */
CN_RUN_KERNEL cn_status cn_transfer_gather_f32(const float *source, const unsigned char *mask,
    size_t pixels, float *out, size_t *count);
CN_RUN_KERNEL cn_status cn_transfer_mask(const void *source, size_t pixels, size_t channels,
    size_t pixel_stride, size_t channel_stride, int f64, unsigned char *mask);
CN_RUN_KERNEL cn_status cn_transfer_alpha(const void *rgb, size_t rgb_pixel_stride,
    size_t rgb_channel_stride, const void *original, size_t original_pixel_stride,
    size_t original_channel_stride, size_t pixels, int rgb_kind, int source_f64,
    void *out, size_t out_pixel_stride, size_t out_channel_stride);
/* transfer_ops.c */
CN_RUN_KERNEL cn_status cn_contrast_bounds_complete(const float *source, size_t count,
    int mode, double percentile, int lanes, size_t block, size_t period, double *bounds);
/* transparency_color.c */
CN_RUN_KERNEL cn_status cn_chroma_key(const float *image, const float *key, size_t pixels,
    size_t pixel_stride, size_t channel_stride, float bg_threshold, float fg_threshold,
    int masks, float *out, size_t out_pixel_stride, size_t out_channel_stride,
    uint8_t *bg, uint8_t *fg);
CN_RUN_KERNEL cn_status cn_chroma_trimap(const uint8_t *bg, const uint8_t *fg,
    double *out, size_t pixels);
CN_RUN_KERNEL cn_status cn_color_representation(const float *image, float *out,
    size_t pixels, int operation, const float *a, const float *b, const float *c, const float *d);
/* unsharp_complete_ops.c */
CN_RUN_KERNEL cn_status cn_unsharp_weighted(const float *source, const float *blurred,
    float *output, size_t count, size_t row_width, double amount, int lanes);
#ifdef __cplusplus
}
#endif

#endif
