/* YUV plane codes and values (nodes/impl/colour_info.py's convention):
 * cn_plane_codes_to_f32 and cn_plane_f32_to_codes; and the matrix step of
 * nodes/impl/yuv_rgb.py: cn_yuv_to_bgr and cn_bgr_to_yuv. The jobs and the scalar
 * helpers, compiled into each including unit (yuv_planes_ops.c, the baseline
 * entries, and the AVX2 unit yuv_planes_avx2.c, which runs them for a segment's
 * last elements). Every including unit uses every helper (C4505).
 *
 * Each element keeps the NumPy sequence in float32, so the results are its bits.
 * A code becomes ((float)code - offset) / scale, + 0.5 for chroma; a value
 * becomes (value - 0.5 for chroma) * scale + offset, rounded half to even
 * (np.rint), clipped to [0, maximum] (np.clip) and cast. Clipping before the
 * rounding gives the same codes (the rounding is monotonic and keeps the integers
 * 0 and maximum), and it sends NaN to 0, NumPy's cast of a NaN on x86-64; NumPy
 * also warns there, these kernels do not. A matrix output is
 * (m0 * a + m1 * b) + m2 * c. No helper calls the CRT. */
#ifndef CHAINNER_YUV_PLANES_SHARED_H
#define CHAINNER_YUV_PLANES_SHARED_H
#include "chainner.h"
#include "isa.h"

/* src: codes (uint8, or uint16 when wide) or float32 values; row y of the source
 * starts at element y * row_stride. out: height x width, C-contiguous, values or
 * codes. maximum: the largest code (2^bits - 1). isa: the level the entry read
 * once for the call. */
typedef struct plane_codes_job {
    const void *src;
    void *out;
    size_t row_stride, width;
    float offset, scale, maximum;
    int wide, chroma;
    cn_isa_level isa;
} plane_codes_job;

static float code_value(float code, float offset, float scale, int chroma)
{
    float value = (code - offset) / scale;
    return chroma ? value + 0.5f : value;
}

/* The scaled value clipped (NaN fails the first test), then rounded half to even
 * by integer parts, exact in any rounding mode: the clipped value is below 2^16. */
static uint32_t value_code(float value, float offset, float scale, float maximum, int chroma)
{
    if (chroma) value = value - 0.5f;
    float scaled = value * scale + offset;
    scaled = scaled > 0.0f ? scaled : 0.0f;
    scaled = scaled < maximum ? scaled : maximum;
    uint32_t whole = (uint32_t)scaled;
    float fraction = scaled - (float)whole;
    return whole + (fraction > 0.5f || (fraction == 0.5f && (whole & 1u)));
}

/* Row y's columns [first, last) into output elements [start, start + last - first). */
static void codes_to_f32_segment(const plane_codes_job *job, size_t y, size_t first,
    size_t last, size_t start)
{
    float *out = (float *)job->out + start;
    if (job->wide) {
        const uint16_t *row = (const uint16_t *)job->src + y * job->row_stride;
        for (size_t x = first; x < last; ++x)
            *out++ = code_value((float)row[x], job->offset, job->scale, job->chroma);
    } else {
        const uint8_t *row = (const uint8_t *)job->src + y * job->row_stride;
        for (size_t x = first; x < last; ++x)
            *out++ = code_value((float)row[x], job->offset, job->scale, job->chroma);
    }
}

static void f32_to_codes_segment(const plane_codes_job *job, size_t y, size_t first,
    size_t last, size_t start)
{
    const float *row = (const float *)job->src + y * job->row_stride;
    if (job->wide) {
        uint16_t *out = (uint16_t *)job->out + start;
        for (size_t x = first; x < last; ++x)
            *out++ = (uint16_t)value_code(row[x], job->offset, job->scale, job->maximum, job->chroma);
    } else {
        uint8_t *out = (uint8_t *)job->out + start;
        for (size_t x = first; x < last; ++x)
            *out++ = (uint8_t)value_code(row[x], job->offset, job->scale, job->maximum, job->chroma);
    }
}

/* matrix: 3 x 3, rows the outputs and columns the inputs: R, G, B by Y, U, V
 * (yuv_to_bgr), Y, U, V by R, G, B (bgr_to_yuv). centre, when not 0, is
 * subtracted from U and V before the products (yuv_to_bgr) or added to them last
 * (bgr_to_yuv), as yuv_rgb.py's separate - 0.5 and + 0.5 of 4:4:4 planes are.
 * Planes: rows of width float32 values, their stride apart; the BGR image: rows of
 * width x 3 values (B, G, R), bgr_stride apart. Outputs are C-contiguous. */
typedef struct yuv_matrix_job {
    const float *y, *u, *v, *bgr;
    float *y_out, *u_out, *v_out, *bgr_out;
    size_t y_stride, u_stride, v_stride, bgr_stride, width;
    float matrix[9], centre;
    cn_isa_level isa;
} yuv_matrix_job;

/* Row `row`'s pixels [first, width). */
static void yuv_to_bgr_row(const yuv_matrix_job *job, size_t row, size_t first)
{
    const float *y = job->y + row * job->y_stride, *u = job->u + row * job->u_stride;
    const float *v = job->v + row * job->v_stride, *m = job->matrix;
    float *out = job->bgr_out + row * job->width * 3;
    for (size_t x = first; x < job->width; ++x) {
        float luma = y[x], cb = u[x], cr = v[x];
        if (job->centre != 0.0f) {
            cb = cb - job->centre;
            cr = cr - job->centre;
        }
        out[3 * x + 2] = m[0] * luma + m[1] * cb + m[2] * cr;
        out[3 * x + 1] = m[3] * luma + m[4] * cb + m[5] * cr;
        out[3 * x] = m[6] * luma + m[7] * cb + m[8] * cr;
    }
}

static void bgr_to_yuv_row(const yuv_matrix_job *job, size_t row, size_t first)
{
    const float *bgr = job->bgr + row * job->bgr_stride, *m = job->matrix;
    size_t offset = row * job->width;
    for (size_t x = first; x < job->width; ++x) {
        float b = bgr[3 * x], g = bgr[3 * x + 1], r = bgr[3 * x + 2];
        float cb = m[3] * r + m[4] * g + m[5] * b, cr = m[6] * r + m[7] * g + m[8] * b;
        if (job->centre != 0.0f) {
            cb = cb + job->centre;
            cr = cr + job->centre;
        }
        job->y_out[offset + x] = m[0] * r + m[1] * g + m[2] * b;
        job->u_out[offset + x] = cb;
        job->v_out[offset + x] = cr;
    }
}

#if CN_ISA_UNITS()
/* yuv_planes_avx2.c: the segment functions above for ISA level avx2 or above, 8
 * elements at a time with the same operations per lane (the rounding is vroundps
 * to nearest even, independent of MXCSR; vmaxps(scaled, 0) sends NaN to 0); the
 * last (last - first) % 8 elements run the scalar segment. */
void cn_codes_to_f32_segment_avx2(const plane_codes_job *job, size_t y, size_t first,
    size_t last, size_t start);
void cn_f32_to_codes_segment_avx2(const plane_codes_job *job, size_t y, size_t first,
    size_t last, size_t start);
/* The matrix rows above for the whole row, 8 pixels at a time with the same
 * operations per lane (the BGR pixels interleaved and split by permutes and
 * blends, which only move bits); the last width % 8 pixels run the scalar row. */
void cn_yuv_to_bgr_row_avx2(const yuv_matrix_job *job, size_t row);
void cn_bgr_to_yuv_row_avx2(const yuv_matrix_job *job, size_t row);
#endif
#endif
