/* YUV plane codes <-> float32 plane values (nodes/impl/colour_info.py's
 * code_to_float and float_to_code) and the YUV <-> BGR matrix step of
 * nodes/impl/yuv_rgb.py, bit for bit. See yuv_planes_shared.h for the operation
 * sequences. */
#include "yuv_planes_shared.h"
#include "parallel.h"

typedef void (*segment_fn)(const plane_codes_job *job, size_t y, size_t first,
    size_t last, size_t start);

/* Output elements [begin, end), split at row ends. */
static void codes_range(const plane_codes_job *job, segment_fn segment, size_t begin, size_t end)
{
    while (begin < end) {
        size_t y = begin / job->width, first = begin % job->width;
        size_t last = end - begin < job->width - first ? first + (end - begin) : job->width;
        segment(job, y, first, last, begin);
        begin += last - first;
    }
}

static void codes_to_f32_range(void *context, size_t begin, size_t end)
{
    const plane_codes_job *job = context;
#if defined(_MSC_VER) && defined(_M_X64)
    if (job->isa >= CN_ISA_AVX2) {
        codes_range(job, cn_codes_to_f32_segment_avx2, begin, end);
        return;
    }
#endif
    codes_range(job, codes_to_f32_segment, begin, end);
}

static void f32_to_codes_range(void *context, size_t begin, size_t end)
{
    const plane_codes_job *job = context;
#if defined(_MSC_VER) && defined(_M_X64)
    if (job->isa >= CN_ISA_AVX2) {
        codes_range(job, cn_f32_to_codes_segment_avx2, begin, end);
        return;
    }
#endif
    codes_range(job, f32_to_codes_segment, begin, end);
}

static void yuv_to_bgr_range(void *context, size_t begin, size_t end)
{
    const yuv_matrix_job *job = context;
    for (size_t row = begin; row < end; ++row) {
#if defined(_MSC_VER) && defined(_M_X64)
        if (job->isa >= CN_ISA_AVX2) {
            cn_yuv_to_bgr_row_avx2(job, row);
            continue;
        }
#endif
        yuv_to_bgr_row(job, row, 0);
    }
}

static void bgr_to_yuv_range(void *context, size_t begin, size_t end)
{
    const yuv_matrix_job *job = context;
    for (size_t row = begin; row < end; ++row) {
#if defined(_MSC_VER) && defined(_M_X64)
        if (job->isa >= CN_ISA_AVX2) {
            cn_bgr_to_yuv_row_avx2(job, row);
            continue;
        }
#endif
        bgr_to_yuv_row(job, row, 0);
    }
}

/* *bytes: the span of height rows of width item-byte elements, row_stride
 * elements apart, from p (0 for no rows or columns). Refused: a null or
 * misaligned pointer and rows wider than row_stride. */
static cn_status span(const void *p, size_t item, size_t row_stride, size_t height,
    size_t width, size_t *bytes)
{
    *bytes = 0;
    if (p == NULL || (uintptr_t)p % item != 0 || width > row_stride) return CN_INVALID_ARGUMENT;
    if (height == 0 || width == 0) return CN_OK;
    if (height - 1 > (SIZE_MAX - width) / row_stride) return CN_SIZE_OVERFLOW;
    size_t extent = (height - 1) * row_stride + width;
    if (extent > SIZE_MAX / item || (uintptr_t)p > UINTPTR_MAX - extent * item)
        return CN_SIZE_OVERFLOW;
    *bytes = extent * item;
    return CN_OK;
}

static int overlaps(const void *a, size_t a_bytes, const void *b, size_t b_bytes)
{
    uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return x < y + b_bytes && y < x + a_bytes;
}

/* The two code entries' checks: src rows, row_stride apart, and a C-contiguous
 * output that does not overlap them. */
static cn_status codes_check(const void *src, size_t src_item, size_t row_stride,
    size_t height, size_t width, const void *out, size_t out_item)
{
    size_t src_bytes, out_bytes;
    cn_status status = span(src, src_item, row_stride, height, width, &src_bytes);
    if (status == CN_OK) status = span(out, out_item, width, height, width, &out_bytes);
    if (status != CN_OK) return status;
    return overlaps(src, src_bytes, out, out_bytes) ? CN_INVALID_ARGUMENT : CN_OK;
}

/* Codes (uint8, or uint16 when wide) to float32 values: the strides are in
 * elements; out is height x width, C-contiguous. */
CN_EXPORT cn_status cn_plane_codes_to_f32(const void *src, int wide, size_t row_stride,
    size_t height, size_t width, float offset, float scale, int chroma, float *out)
{
    cn_status status = codes_check(src, wide ? sizeof(uint16_t) : sizeof(uint8_t),
        row_stride, height, width, out, sizeof(float));
    if (status != CN_OK || height == 0 || width == 0) return status;
    plane_codes_job job = {src, out, row_stride, width, offset, scale, 0.0f,
        wide != 0, chroma != 0, cn_isa_current()};
    return cn_parallel_for(height * width, 65536, codes_to_f32_range, &job);
}

/* Float32 values to codes (uint8, or uint16 when wide) in [0, maximum]: maximum
 * is at most 255, or 65535 when wide. */
CN_EXPORT cn_status cn_plane_f32_to_codes(const float *src, size_t row_stride,
    size_t height, size_t width, float offset, float scale, int chroma, uint32_t maximum,
    int wide, void *out)
{
    if (maximum > (wide ? 65535u : 255u)) return CN_INVALID_ARGUMENT;
    cn_status status = codes_check(src, sizeof(float), row_stride, height, width, out,
        wide ? sizeof(uint16_t) : sizeof(uint8_t));
    if (status != CN_OK || height == 0 || width == 0) return status;
    plane_codes_job job = {src, out, row_stride, width, offset, scale, (float)maximum,
        wide != 0, chroma != 0, cn_isa_current()};
    return cn_parallel_for(height * width, 65536, f32_to_codes_range, &job);
}

/* The matrix entries' checks: three planes, the BGR image (rows of width x 3,
 * bgr_stride apart) and a misaligned matrix; outputs may overlap nothing. */
static cn_status matrix_check(const float *const planes[3], const size_t strides[3],
    const float *bgr, size_t bgr_stride, size_t height, size_t width, const float *matrix,
    int planes_out)
{
    size_t plane_bytes[3], bgr_bytes;
    if (matrix == NULL || (uintptr_t)matrix % sizeof(float) != 0 || width > SIZE_MAX / 3)
        return CN_INVALID_ARGUMENT;
    cn_status status = span(bgr, sizeof(float), bgr_stride, height, 3 * width, &bgr_bytes);
    for (int k = 0; k < 3 && status == CN_OK; ++k)
        status = span(planes[k], sizeof(float), strides[k], height, width, &plane_bytes[k]);
    if (status != CN_OK) return status;
    const size_t matrix_bytes = 9 * sizeof(float);
    for (int k = 0; k < 3; ++k) {
        if (planes_out) {
            if (overlaps(planes[k], plane_bytes[k], bgr, bgr_bytes)
                || overlaps(planes[k], plane_bytes[k], matrix, matrix_bytes)
                || overlaps(planes[k], plane_bytes[k], planes[(k + 1) % 3], plane_bytes[(k + 1) % 3]))
                return CN_INVALID_ARGUMENT;
        } else if (overlaps(bgr, bgr_bytes, planes[k], plane_bytes[k])) {
            return CN_INVALID_ARGUMENT;
        }
    }
    if (!planes_out && overlaps(bgr, bgr_bytes, matrix, matrix_bytes)) return CN_INVALID_ARGUMENT;
    return CN_OK;
}

/* Y, U and V planes (strides in elements) to a C-contiguous height x width x 3
 * BGR image. matrix: R, G, B by Y, U, V. */
CN_EXPORT cn_status cn_yuv_to_bgr(const float *y, size_t y_stride, const float *u,
    size_t u_stride, const float *v, size_t v_stride, size_t height, size_t width,
    const float *matrix, float centre, float *out)
{
    const float *planes[3] = {y, u, v};
    const size_t strides[3] = {y_stride, u_stride, v_stride};
    cn_status status = matrix_check(planes, strides, out, 3 * width, height, width, matrix, 0);
    if (status != CN_OK || height == 0 || width == 0) return status;
    yuv_matrix_job job = {y, u, v, NULL, NULL, NULL, NULL, out, y_stride, u_stride, v_stride,
        3 * width, width, {0}, centre, cn_isa_current()};
    for (int k = 0; k < 9; ++k) job.matrix[k] = matrix[k];
    return cn_parallel_for(height, 65536 / width + 1, yuv_to_bgr_range, &job);
}

/* A BGR image (rows of width x 3, row_stride elements apart) to C-contiguous
 * height x width Y, U and V planes. matrix: Y, U, V by R, G, B. */
CN_EXPORT cn_status cn_bgr_to_yuv(const float *src, size_t row_stride, size_t height,
    size_t width, const float *matrix, float centre, float *y, float *u, float *v)
{
    const float *planes[3] = {y, u, v};
    const size_t strides[3] = {width, width, width};
    cn_status status = matrix_check(planes, strides, src, row_stride, height, width, matrix, 1);
    if (status != CN_OK || height == 0 || width == 0) return status;
    yuv_matrix_job job = {NULL, NULL, NULL, src, y, u, v, NULL, width, width, width,
        row_stride, width, {0}, centre, cn_isa_current()};
    for (int k = 0; k < 9; ++k) job.matrix[k] = matrix[k];
    return cn_parallel_for(height, 65536 / width + 1, bgr_to_yuv_range, &job);
}
