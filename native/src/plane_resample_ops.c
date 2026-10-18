/* One axis of the plane resampler (nodes/impl/plane_resample.py), bit for bit.
 * See plane_resample_shared.h for the operation sequence. */
#include "plane_resample_shared.h"
#include "parallel.h"

static void resample_range(void *context, size_t begin, size_t end)
{
    const plane_resample_job *job = context;
#if defined(_MSC_VER) && defined(_M_X64)
    if (job->isa >= CN_ISA_AVX2) {
        cn_plane_resample_rows_avx2(job, begin, end);
        return;
    }
#endif
    for (size_t y = begin; y < end; ++y) {
        if (job->vertical)
            resample_row_vertical(job, y, 0);
        else
            resample_row_horizontal(job, y, 0);
    }
}

static int overlaps(uintptr_t a, size_t a_bytes, uintptr_t b, size_t b_bytes)
{
    return a < b + b_bytes && b < a + a_bytes;
}

/* src: height rows of width float32 samples, row_stride elements apart. vertical
 * 0 resamples each row to dst_dim samples (out height x dst_dim), 1 the columns to
 * dst_dim rows (out dst_dim x width). index and weights: taps x dst_dim,
 * tap-major; every index lies inside the resampled axis. Refused: a null or
 * misaligned pointer, no tap, rows wider than row_stride, an index outside the
 * axis or beyond INT32_MAX rows, and an output that overlaps an input. */
CN_EXPORT cn_status cn_plane_resample_axis(const float *src, size_t row_stride,
    size_t height, size_t width, int vertical, const int32_t *index, const float *weights,
    size_t taps, size_t dst_dim, float bias_in, float bias_out, float *out)
{
    if (src == NULL || index == NULL || weights == NULL || out == NULL || taps == 0
        || width > row_stride || (uintptr_t)src % sizeof(float) != 0
        || (uintptr_t)index % sizeof(int32_t) != 0 || (uintptr_t)weights % sizeof(float) != 0
        || (uintptr_t)out % sizeof(float) != 0)
        return CN_INVALID_ARGUMENT;
    size_t rows = vertical ? dst_dim : height, columns = vertical ? width : dst_dim;
    size_t axis = vertical ? height : width;
    if (rows == 0 || columns == 0) return CN_OK;
    if (axis == 0 || axis > INT32_MAX) return CN_INVALID_ARGUMENT;
    if (taps > SIZE_MAX / dst_dim || taps * dst_dim > SIZE_MAX / sizeof(float)
        || height - 1 > (SIZE_MAX - width) / row_stride || rows > SIZE_MAX / columns
        || rows * columns > SIZE_MAX / sizeof(float))
        return CN_SIZE_OVERFLOW;
    size_t table = taps * dst_dim;
    for (size_t k = 0; k < table; ++k)
        if (index[k] < 0 || (size_t)index[k] >= axis) return CN_INVALID_ARGUMENT;
    size_t extent = (height - 1) * row_stride + width;
    if (extent > SIZE_MAX / sizeof(float)) return CN_SIZE_OVERFLOW;
    size_t src_bytes = extent * sizeof(float), table_bytes = table * sizeof(float);
    size_t out_bytes = rows * columns * sizeof(float);
    uintptr_t o = (uintptr_t)out;
    if ((uintptr_t)src > UINTPTR_MAX - src_bytes || o > UINTPTR_MAX - out_bytes
        || (uintptr_t)index > UINTPTR_MAX - table_bytes
        || (uintptr_t)weights > UINTPTR_MAX - table_bytes)
        return CN_SIZE_OVERFLOW;
    if (overlaps((uintptr_t)src, src_bytes, o, out_bytes)
        || overlaps((uintptr_t)index, table_bytes, o, out_bytes)
        || overlaps((uintptr_t)weights, table_bytes, o, out_bytes))
        return CN_INVALID_ARGUMENT;
    plane_resample_job job = {src, out, index, weights, row_stride, width, dst_dim, taps,
        bias_in, bias_out, vertical != 0, cn_isa_current()};
    size_t grain = 65536 / columns / taps + 1;
    return cn_parallel_for(rows, grain, resample_range, &job);
}
