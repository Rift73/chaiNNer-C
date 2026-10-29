/* Planar samples to interleaved float32 (cn_planar_to_interleaved_f32 and
 * cn_planes_to_interleaved_f32): the job and the scalar helpers, compiled into each
 * including unit (planar_ops.c, the baseline entry, and the AVX2 unit planar_avx2.c,
 * which runs planar_rows for a row's tail and for channel counts other than 1 and
 * 3). Every including unit uses every helper (C4505).
 *
 * Every value is exact: float32 samples are moved bit for bit, and a half becomes
 * NumPy's float32 for it (npy_halfbits_to_floatbits): normal values by integer
 * rebiasing, Inf and NaN with the payload kept (never quieted), a subnormal as
 * (float)mantissa * 2^-24, whose operands and result are normal float32 values,
 * so it is exact under any rounding mode, FTZ or DAZ. */
#ifndef CHAINNER_PLANAR_SHARED_H
#define CHAINNER_PLANAR_SHARED_H
#include "chainner.h"
#include "isa.h"
#include <string.h>

/* Samples are half bits (uint16) or float32. One buffer (planes NULL): src, whose
 * channel k's row y starts at element k * plane_stride + y * row_stride; output
 * channel k reads plane source_plane(job, k). order: 0 keeps the planes' order, 1
 * reverses it, 2 swaps the first and third planes (RGB(A) <-> BGR(A); three or more
 * channels). Separate planes (cn_planes_to_interleaved_f32, order 0): output channel
 * k reads planes[k], whose row y starts at element y * row_strides[k]. out: height x
 * width x channels, C-contiguous. isa: the level the entry read once for the call. */
typedef struct planar_job {
    const void *src;
    const void *const *planes;
    const size_t *row_strides;
    float *out;
    size_t plane_stride, row_stride, width, channels;
    int half, order;
    cn_isa_level isa;
} planar_job;

static size_t source_plane(const planar_job *job, size_t k)
{
    if (job->order == 1) return job->channels - 1 - k;
    if (job->order == 2 && k < 3) return 2 - k;
    return k;
}

/* The first sample of output channel k's row y. */
static const void *planar_row(const planar_job *job, size_t k, size_t y)
{
    size_t item = job->half ? sizeof(uint16_t) : sizeof(float);
    if (job->planes != NULL)
        return (const unsigned char *)job->planes[k] + y * job->row_strides[k] * item;
    return (const unsigned char *)job->src
        + (source_plane(job, k) * job->plane_stride + y * job->row_stride) * item;
}

static uint32_t half_bits_to_f32(uint32_t h)
{
    uint32_t sign = (h & 0x8000u) << 16, exponent = h & 0x7c00u;
    uint32_t rest = (h & 0x7fffu) << 13;
    if (exponent == 0x7c00u) return sign | 0x7f800000u | rest;
    if (exponent != 0) return sign | (rest + (112u << 23));
    float value = (float)(h & 0x3ffu) * 0x1p-24f;
    uint32_t bits;
    memcpy(&bits, &value, sizeof bits);
    return sign | bits;
}

/* Rows [begin, end), columns [first, width); returns 1 when a sample there is Inf
 * or NaN (its float32 exponent all ones), else 0. */
static int planar_rows(const planar_job *job, size_t begin, size_t end, size_t first)
{
    size_t channels = job->channels;
    uint32_t special = 0;
    for (size_t y = begin; y < end; ++y) {
        float *row = job->out + y * job->width * channels;
        for (size_t k = 0; k < channels; ++k) {
            const void *source = planar_row(job, k, y);
            for (size_t x = first; x < job->width; ++x) {
                uint32_t bits;
                if (job->half)
                    bits = half_bits_to_f32(((const uint16_t *)source)[x]);
                else
                    memcpy(&bits, (const float *)source + x, sizeof bits);
                special |= (bits & 0x7f800000u) == 0x7f800000u;
                memcpy(row + x * channels + k, &bits, sizeof bits);
            }
        }
    }
    return special != 0;
}

#if CN_ISA_UNITS()
/* planar_avx2.c: rows [begin, end), inside one chunk, for ISA level avx2 or above;
 * returns planar_rows' Inf/NaN flag. One and three channels run 8 pixels at a time
 * with half_bits_to_f32's operation per lane; each row's last width % 8 pixels, and
 * every other channel count, run planar_rows. */
int cn_planar_rows_avx2(const planar_job *job, size_t begin, size_t end);
#endif
#endif
