/* chaiNNer blend modes and alpha composition, ported from nodes/impl/blend.py.
 * SPDX-License-Identifier: GPL-3.0-only
 * Keep float32 operation order: upstream NumPy materializes each expression.
 * The per-element helpers and image_pixels live in blend_shared.h (SP4b Task 6),
 * shared with the AVX2 unit blend_avx2.c.
 */
#include "blend_shared.h"
#include "parallel.h"

static void mode_range(void *context, size_t begin, size_t end)
{
    blend_job *job = (blend_job *)context;
    for (size_t i = begin; i < end; ++i)
        job->output[i] = apply_mode(job->overlay[i], job->base[i], job->mode);
}

CN_EXPORT cn_status cn_blend_mode(const float *overlay, const float *base,
    float *output, size_t count, int mode)
{
    if (overlay == NULL || base == NULL || output == NULL || mode < 0 || mode > 22)
        return CN_INVALID_ARGUMENT;
    if (count > SIZE_MAX / sizeof(float)) return CN_SIZE_OVERFLOW;
    blend_job job = {overlay, base, output, mode, 0, 0, CN_ISA_SCALAR};
    return cn_parallel_for(count, 65536, mode_range, &job);
}

static int supported_channels(int channels)
{
    return channels == 1 || channels == 3 || channels == 4;
}

static void image_range(void *context, size_t begin, size_t end)
{
    const blend_job *job = context;
#if defined(_MSC_VER) && defined(_M_X64)
    if (job->isa >= CN_ISA_AVX2) {
        cn_blend_images_avx2(job, begin, end);
        return;
    }
#endif
    image_pixels(job, begin, end);
}

CN_EXPORT cn_status cn_blend_images(const float *overlay, const float *base,
    float *output, size_t pixels, int overlay_channels, int base_channels, int mode)
{
    if (overlay == NULL || base == NULL || output == NULL || mode < 0 || mode > 22
        || !supported_channels(overlay_channels) || !supported_channels(base_channels))
        return CN_INVALID_ARGUMENT;
    int target = overlay_channels > base_channels ? overlay_channels : base_channels;
    if (pixels > SIZE_MAX / ((size_t)target * sizeof(float))) return CN_SIZE_OVERFLOW;
    blend_job job = {overlay, base, output, mode, overlay_channels, base_channels,
        cn_isa_current()};
    return cn_parallel_for(pixels, 65536, image_range, &job);
}
