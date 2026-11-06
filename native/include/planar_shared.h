/* Planar samples to interleaved float32 (cn_planar_to_interleaved_f32 and
 * cn_planes_to_interleaved_f32), and interleaved float32 to planar bfloat16
 * (cn_interleaved_to_bfloat_planes): the jobs and the scalar helpers, compiled into
 * each including unit (planar_ops.c, the baseline entries, and the AVX2 unit
 * planar_avx2.c, which runs planar_rows and bfloat_rows for a row's tail and
 * planar_rows for channel counts other than 1 and 3). Every including unit uses every
 * helper (C4505).
 *
 * Every value is exact: float32 samples are moved bit for bit, a bfloat16 is the top
 * half of its float32, and a half becomes NumPy's float32 for it
 * (npy_halfbits_to_floatbits): normal values by integer rebiasing, Inf and NaN with
 * the payload kept (never quieted), a subnormal as (float)mantissa * 2^-24, whose
 * operands and result are normal float32 values, so it is exact under any rounding
 * mode, FTZ or DAZ. A gain other than 1 or an offset other than 0 (a video frame's,
 * cn_planes_to_interleaved_f32) gives value * gain + offset in the same pass, between
 * 0 and 1. */
#ifndef CHAINNER_PLANAR_SHARED_H
#define CHAINNER_PLANAR_SHARED_H
#include "chainner.h"
#include "isa.h"
#include <string.h>

enum { CN_PLANAR_FLOAT = 0, CN_PLANAR_HALF = 1, CN_PLANAR_BFLOAT = 2 };

/* Samples are float32, or half or bfloat16 bits (uint16). One buffer (planes NULL): src, whose
 * channel k's row y starts at element k * plane_stride + y * row_stride; output
 * channel k reads plane source_plane(job, k). order: 0 keeps the planes' order, 1
 * reverses it, 2 swaps the first and third planes (RGB(A) <-> BGR(A); three or more
 * channels). Separate planes (cn_planes_to_interleaved_f32, order 0): output channel
 * k reads planes[k], whose row y starts at element y * row_strides[k]. out: height x
 * width x channels, C-contiguous. isa: the level the entry read once for the call.
 * gain 1 and offset 0 move the values as above. */
typedef struct planar_job {
    const void *src;
    const void *const *planes;
    const size_t *row_strides;
    float *out;
    size_t plane_stride, row_stride, width, channels;
    int sample, order;
    cn_isa_level isa;
    float gain, offset;
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
    size_t item = job->sample == CN_PLANAR_FLOAT ? sizeof(float) : sizeof(uint16_t);
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
                if (job->sample == CN_PLANAR_HALF)
                    bits = half_bits_to_f32(((const uint16_t *)source)[x]);
                else if (job->sample == CN_PLANAR_BFLOAT)
                    bits = (uint32_t)((const uint16_t *)source)[x] << 16;
                else
                    memcpy(&bits, (const float *)source + x, sizeof bits);
                special |= (bits & 0x7f800000u) == 0x7f800000u;
                if (job->gain != 1.0f || job->offset != 0.0f) {
                    float value;
                    memcpy(&value, &bits, sizeof value);
                    value = value * job->gain + job->offset;
                    value = value < 1.0f ? value : 1.0f;
                    value = value > 0.0f ? value : 0.0f;
                    memcpy(&bits, &value, sizeof bits);
                }
                memcpy(row + x * channels + k, &bits, sizeof bits);
            }
        }
    }
    return special != 0;
}

/* Interleaved float32 (a row every row_stride elements, channels samples per pixel) to
 * planes of bfloat16 bits, height x width each, one after another in out. Output plane
 * k reads channel k, the first and third swapped when swap is set. */
typedef struct bfloat_job {
    const float *src;
    uint16_t *out;
    size_t row_stride, height, width, channels;
    int swap;
    cn_isa_level isa;
} bfloat_job;

static size_t bfloat_channel(const bfloat_job *job, size_t k)
{
    return job->swap && k < 3 ? 2 - k : k;
}

/* Rows [begin, end), columns [first, width). Round to nearest, ties to even, as
 * ml_dtypes rounds; a NaN becomes the quiet NaN 0x7fc0 with its sign, since rounding
 * its payload could carry it into Inf or wrap it to zero. */
static void bfloat_rows(const bfloat_job *job, size_t begin, size_t end, size_t first)
{
    for (size_t k = 0; k < job->channels; ++k) {
        const float *source = job->src + bfloat_channel(job, k);
        uint16_t *plane = job->out + k * job->height * job->width;
        for (size_t y = begin; y < end; ++y) {
            const float *row = source + y * job->row_stride;
            uint16_t *out = plane + y * job->width;
            for (size_t x = first; x < job->width; ++x) {
                uint32_t bits;
                memcpy(&bits, row + x * job->channels, sizeof bits);
                out[x] = (bits & 0x7fffffffu) > 0x7f800000u
                    ? (uint16_t)((bits >> 16 & 0x8000u) | 0x7fc0u)
                    : (uint16_t)((bits + 0x7fffu + (bits >> 16 & 1u)) >> 16);
            }
        }
    }
}

#if CN_ISA_UNITS()
/* planar_avx2.c: rows [begin, end), inside one chunk, for ISA level avx2 or above;
 * returns planar_rows' Inf/NaN flag. One and three channels run 8 pixels at a time
 * with half_bits_to_f32's operation per lane; each row's last width % 8 pixels, and
 * every other channel count, run planar_rows. */
int cn_planar_rows_avx2(const planar_job *job, size_t begin, size_t end);
/* planar_avx2.c: bfloat_rows' rows, 8 pixels at a time; each row's last width % 8
 * pixels run bfloat_rows. */
void cn_bfloat_rows_avx2(const bfloat_job *job, size_t begin, size_t end);
#endif
#endif
