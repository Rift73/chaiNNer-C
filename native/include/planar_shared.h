/* Planar samples to interleaved float32 (cn_planar_to_interleaved_f32): the job
 * and the scalar helpers, compiled into each including unit (planar_ops.c, the
 * baseline entry, and the AVX2 unit planar_avx2.c, which runs planar_rows for a
 * row's tail and for channel counts other than 1 and 3). Every including unit
 * uses every helper (C4505).
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

/* src: half bits (uint16) or float32 samples; channel k's row y starts at
 * element k * plane_stride + y * row_stride. out: height x width x channels,
 * C-contiguous; output channel k reads plane channels - 1 - k when reverse is
 * set, else plane k. isa: the level the entry read once for the call. */
typedef struct planar_job {
    const void *src;
    float *out;
    size_t plane_stride, row_stride, width, channels;
    int half, reverse;
    cn_isa_level isa;
} planar_job;

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

/* Rows [begin, end), columns [first, width). */
static void planar_rows(const planar_job *job, size_t begin, size_t end, size_t first)
{
    size_t channels = job->channels;
    for (size_t y = begin; y < end; ++y) {
        float *row = job->out + y * job->width * channels;
        for (size_t k = 0; k < channels; ++k) {
            size_t plane = job->reverse ? channels - 1 - k : k;
            size_t offset = plane * job->plane_stride + y * job->row_stride;
            for (size_t x = first; x < job->width; ++x) {
                uint32_t bits;
                if (job->half)
                    bits = half_bits_to_f32(((const uint16_t *)job->src)[offset + x]);
                else
                    memcpy(&bits, (const float *)job->src + offset + x, sizeof bits);
                memcpy(row + x * channels + k, &bits, sizeof bits);
            }
        }
    }
}

#if defined(_MSC_VER) && defined(_M_X64)
/* planar_avx2.c: rows [begin, end), inside one chunk, for ISA level avx2 or above.
 * One and three channels run 8 pixels at a time with half_bits_to_f32's operation
 * per lane; each row's last width % 8 pixels, and every other channel count, run
 * planar_rows. */
void cn_planar_rows_avx2(const planar_job *job, size_t begin, size_t end);
#endif
#endif
