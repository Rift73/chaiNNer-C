/* Planar half or float32 samples to interleaved float32: a model's NCHW output
 * plane set (for example the TensorRT node's pinned output, cropped) as an HWC
 * image, in one pass. See planar_shared.h for the exactness rules. */
#include "planar_shared.h"
#include "parallel.h"

static void planar_range(void *context, size_t begin, size_t end)
{
    const planar_job *job = context;
#if defined(_MSC_VER) && defined(_M_X64)
    if (job->isa >= CN_ISA_AVX2) {
        cn_planar_rows_avx2(job, begin, end);
        return;
    }
#endif
    planar_rows(job, begin, end, 0);
}

/* Strides are in elements. Refused: a null pointer, no channel, rows wider than
 * row_stride, a misaligned pointer, and a source that overlaps the output. */
CN_EXPORT cn_status cn_planar_to_interleaved_f32(const void *src, int half,
    size_t plane_stride, size_t row_stride, size_t height, size_t width,
    size_t channels, int reverse, float *out)
{
    size_t item = half ? sizeof(uint16_t) : sizeof(float);
    if (src == NULL || out == NULL || channels == 0 || width > row_stride
        || (uintptr_t)src % item != 0 || (uintptr_t)out % sizeof(float) != 0)
        return CN_INVALID_ARGUMENT;
    if (height == 0 || width == 0) return CN_OK;
    if (height - 1 > (SIZE_MAX - width) / row_stride) return CN_SIZE_OVERFLOW;
    size_t plane = (height - 1) * row_stride + width;
    if (plane_stride != 0 && channels - 1 > (SIZE_MAX - plane) / plane_stride)
        return CN_SIZE_OVERFLOW;
    size_t extent = (channels - 1) * plane_stride + plane;
    if (extent > SIZE_MAX / item || height > SIZE_MAX / width
        || height * width > SIZE_MAX / channels / sizeof(float))
        return CN_SIZE_OVERFLOW;
    uintptr_t s = (uintptr_t)src, o = (uintptr_t)out;
    size_t src_bytes = extent * item, out_bytes = height * width * channels * sizeof(float);
    if (s > UINTPTR_MAX - src_bytes || o > UINTPTR_MAX - out_bytes) return CN_SIZE_OVERFLOW;
    if (s < o + out_bytes && o < s + src_bytes) return CN_INVALID_ARGUMENT;
    planar_job job = {src, out, plane_stride, row_stride, width, channels,
        half != 0, reverse != 0, cn_isa_current()};
    size_t grain = 65536 / (width * channels) + 1;
    return cn_parallel_for(height, grain, planar_range, &job);
}
