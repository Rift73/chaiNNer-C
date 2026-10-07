/* cn_normal_output's job and per-pixel sequence, moved from color_ops.c (SP4b Task 8).
 * Shared by that unit and the 8-lane unit color_avx2.c: normal_pixels and the NaN
 * outputs' rule, static helpers compiled into each including unit. Both includers use
 * them: color_ops.c for every pixel below the AVX2 level, color_avx2.c for a chunk's
 * last pixels after its groups of 8 and for a group's non-finite pixels. */
#ifndef CHAINNER_COLOR_SHARED_H
#define CHAINNER_COLOR_SHARED_H
#include "chainner.h"
#include "isa.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* isa: the level cn_normal_output read once for the call (cn_isa_current). */
typedef struct normal_output_job {
    const float *dx, *dy, *alpha;
    float *dst;
    int invert_r, invert_g, channels;
    cn_isa_level isa;
} normal_output_job;

static int normal_nan_word(uint32_t word) { return (word & 0x7FFFFFFFu) > 0x7F800000u; }

/* The NaN outputs of pixel i, whose dx or dy is not finite: the bits NumPy's sequence
 * gives on x86, derived from dx and dy as integers after the arithmetic. Each operation
 * keeps its first NaN operand, quieted (x * x + y * y, sqrt, then x / length, y / length
 * and 2 / length); inf / inf gives the default NaN 0xFFC00000; an inversion flips the
 * sign bit and fabsf clears it; + 1 and * 0.5 keep a NaN. The arithmetic's NaN signs
 * and payloads are never read: the compiler leaves them unspecified (it may move a
 * negation across the division or swap an operation's operands). */
static void normal_nan_outputs(const normal_output_job *job, size_t i, float *pixel)
{
    uint32_t x, y;
    memcpy(&x, &job->dx[i], sizeof x);
    memcpy(&y, &job->dy[i], sizeof y);
    const uint32_t quiet = 0x00400000u, sign = 0x80000000u, generated = 0xFFC00000u;
    uint32_t length = normal_nan_word(x) ? x | quiet : normal_nan_word(y) ? y | quiet : 0;
    uint32_t words[3] = {0, 0, 0}; /* z, y, x; 0: not a NaN */
    if (length) words[0] = length & ~sign;
    const uint32_t inputs[2] = {y, x};
    for (int c = 0; c < 2; ++c) {
        uint32_t word = normal_nan_word(inputs[c]) ? inputs[c] | quiet : length ? length
            : (inputs[c] & ~sign) == 0x7F800000u ? generated : 0;
        if (word && (c == 0 ? job->invert_g : job->invert_r)) word ^= sign;
        words[c + 1] = word;
    }
    for (int c = 0; c < 3; ++c)
        if (words[c]) memcpy(&pixel[c], &words[c], sizeof words[c]);
}

/* Whether pixel i's dx or dy is not finite, from their bits. */
static int normal_special(const normal_output_job *job, size_t i)
{
    uint32_t x, y;
    memcpy(&x, &job->dx[i], sizeof x);
    memcpy(&y, &job->dy[i], sizeof y);
    return (x & 0x7F800000u) == 0x7F800000u || (y & 0x7F800000u) == 0x7F800000u;
}

/* Pixels [begin, end); the job's fields are read once, so the stores into dst cannot
 * make them reload; each pixel's operations are the original's, in order. B3 runs
 * sqrtf as the CRT's (sqrtss on every non-negative or NaN input, a NaN quieted) and
 * fabsf as (float)fabs((double)z) (andps on the double, a sign-clear). A pixel with a
 * non-finite input then takes its NaN outputs from normal_nan_outputs. */
static void normal_pixels(const normal_output_job *job, size_t begin, size_t end)
{
    const float *dx = job->dx, *dy = job->dy, *alpha = job->alpha;
    int invert_r = job->invert_r, invert_g = job->invert_g;
    size_t channels = (size_t)job->channels;
    float *pixel = job->dst + begin * channels;
    for (size_t i = begin; i < end; ++i, pixel += channels) {
        float x = dx[i], y = dy[i];
        float square = x * x;
        float yy = y * y;
        square += yy;
        square += 4.0f;
        float length = sqrtf(square);
        x /= length;
        y /= length;
        float z = 2.0f / length;
        if (invert_r) x = -x;
        if (invert_g) y = -y;
        x += 1.0f;
        y += 1.0f;
        pixel[0] = fabsf(z);
        pixel[1] = y * 0.5f;
        pixel[2] = x * 0.5f;
        if (channels == 4) pixel[3] = alpha ? alpha[i] : 1.0f;
        if (normal_special(job, i)) normal_nan_outputs(job, i, pixel);
    }
}

/* color_avx2.c: pixels [begin, end) of one chunk, for ISA level avx2 or above, 8 at a
 * time as normal_pixels' operations (vsqrtps and vdivps are correctly rounded as sqrtss
 * and divss; the inversions and fabsf are sign xors and a sign clear; the channels are
 * interleaved by permutes, which move bits); the chunk's last pixels run
 * normal_pixels. */
void cn_normal_output_avx2(const normal_output_job *job, size_t begin, size_t end);
#endif
