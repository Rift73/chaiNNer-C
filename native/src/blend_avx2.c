#if !defined(_M_X64)
#error "blend_avx2.c is an x86-64 /arch:AVX2 unit"
#endif
/* Blend Images' per-element blend, AVX2 unit (/arch:AVX2, spec 4.3): no export,
 * reached only through image_range's dispatch on the call's ISA level (SP4b D13).
 * NORMAL (mode 0) and MULTIPLY (mode 1) are normalized modes (no clip), so with equal
 * channel counts below 4 each output element is image_pixels' one operation on the
 * same element of both inputs: the overlay's value (moved, every bit kept), or
 * overlay * base (multiply). Whole blocks of 8 pixels (channels vectors) run inside
 * the chunk; the chunk's last pixels, and every other job, run image_pixels. */
#include "blend_shared.h"
#include <immintrin.h>

/* overlay * base as B3's apply_mode computes it (0x18000223B: mulss xmm7 = overlay,
 * xmm8 = base): a NaN overlay gives itself, quieted (the first source's NaN wins),
 * else the product (which keeps a NaN base, quieted). The compiler may commute
 * vmulps's sources, so the overlay's NaN is selected explicitly. */
static __m256 multiply(__m256 overlay, __m256 base)
{
    __m256 product = _mm256_mul_ps(overlay, base);
    __m256 nan = _mm256_cmp_ps(overlay, overlay, _CMP_UNORD_Q);
    __m256 quiet = _mm256_or_ps(overlay, _mm256_castsi256_ps(_mm256_set1_epi32(0x00400000)));
    return _mm256_blendv_ps(product, quiet, nan);
}

void cn_blend_images_avx2(const blend_job *job, size_t begin, size_t end)
{
    size_t channels = (size_t)job->overlay_channels;
    if ((job->mode != 0 && job->mode != 1) || job->overlay_channels != job->base_channels ||
        channels > 3) {
        image_pixels(job, begin, end);
        return;
    }
    size_t blocks = (end - begin) / 8;
    size_t i = begin * channels, last = (begin + blocks * 8) * channels;
    const float *overlay = job->overlay, *base = job->base;
    float *output = job->output;
    if (job->mode == 0) {
        for (; last - i >= 32; i += 32) {
            __m256 a0 = _mm256_loadu_ps(overlay + i), a1 = _mm256_loadu_ps(overlay + i + 8);
            __m256 a2 = _mm256_loadu_ps(overlay + i + 16), a3 = _mm256_loadu_ps(overlay + i + 24);
            _mm256_storeu_ps(output + i, a0);
            _mm256_storeu_ps(output + i + 8, a1);
            _mm256_storeu_ps(output + i + 16, a2);
            _mm256_storeu_ps(output + i + 24, a3);
        }
        for (; i < last; i += 8) _mm256_storeu_ps(output + i, _mm256_loadu_ps(overlay + i));
    } else {
        for (; last - i >= 32; i += 32) {
            __m256 p0 = multiply(_mm256_loadu_ps(overlay + i), _mm256_loadu_ps(base + i));
            __m256 p1 = multiply(_mm256_loadu_ps(overlay + i + 8), _mm256_loadu_ps(base + i + 8));
            __m256 p2 = multiply(_mm256_loadu_ps(overlay + i + 16), _mm256_loadu_ps(base + i + 16));
            __m256 p3 = multiply(_mm256_loadu_ps(overlay + i + 24), _mm256_loadu_ps(base + i + 24));
            _mm256_storeu_ps(output + i, p0);
            _mm256_storeu_ps(output + i + 8, p1);
            _mm256_storeu_ps(output + i + 16, p2);
            _mm256_storeu_ps(output + i + 24, p3);
        }
        for (; i < last; i += 8)
            _mm256_storeu_ps(output + i, multiply(_mm256_loadu_ps(overlay + i), _mm256_loadu_ps(base + i)));
    }
    image_pixels(job, begin + blocks * 8, end);
}
