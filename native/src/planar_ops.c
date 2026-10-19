/* Planar half or float32 samples to interleaved float32: a model's NCHW output
 * plane set (every upscaling backend's output, cropped, red and blue swapped where
 * asked) as an HWC image in one pass, reporting whether every sample is finite.
 * See planar_shared.h for the exactness rules. */
#include "planar_shared.h"
#include "parallel.h"
#ifdef _MSC_VER
#include <intrin.h>
#endif

typedef struct {
    planar_job job;
    volatile long special; /* 1 once a chunk saw an Inf or NaN sample */
} planar_call;

static void planar_range(void *context, size_t begin, size_t end)
{
    planar_call *call = context;
    int special;
#if defined(_MSC_VER) && defined(_M_X64)
    if (call->job.isa >= CN_ISA_AVX2)
        special = cn_planar_rows_avx2(&call->job, begin, end);
    else
#endif
        special = planar_rows(&call->job, begin, end, 0);
    if (special) {
#ifdef _MSC_VER
        _InterlockedOr(&call->special, 1);
#else
        __atomic_fetch_or(&call->special, 1, __ATOMIC_RELAXED);
#endif
    }
}

/* Strides are in elements; order as planar_job's. finite (may be null): 1 when every
 * sample is finite, else 0 (NaN or Inf). Refused: a null pointer, no channel, an
 * order other than 0-2 (2 with fewer than three channels), rows wider than
 * row_stride, a misaligned pointer, and a source that overlaps the output. */
CN_EXPORT cn_status cn_planar_to_interleaved_f32(const void *src, int half,
    size_t plane_stride, size_t row_stride, size_t height, size_t width,
    size_t channels, int order, float *out, int *finite)
{
    size_t item = half ? sizeof(uint16_t) : sizeof(float);
    if (src == NULL || out == NULL || channels == 0 || width > row_stride
        || order < 0 || order > 2 || (order == 2 && channels < 3)
        || (uintptr_t)src % item != 0 || (uintptr_t)out % sizeof(float) != 0)
        return CN_INVALID_ARGUMENT;
    if (height == 0 || width == 0) {
        if (finite) *finite = 1;
        return CN_OK;
    }
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
    planar_call call = {{src, out, plane_stride, row_stride, width, channels,
        half != 0, order, cn_isa_current()}, 0};
    size_t grain = 65536 / (width * channels) + 1;
    cn_status status = cn_parallel_for(height, grain, planar_range, &call);
    if (finite) *finite = !call.special;
    return status;
}
