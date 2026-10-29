#if !defined(_M_X64) && !(defined(__linux__) && defined(__x86_64__))
#error "wavelet_avx2.c is an x86-64 /arch:AVX2 unit"
#endif
/* Wavelet Color Fix's blur interior, AVX2 unit (/arch:AVX2, spec 4.3): no export, reached only through blur_range's
 * dispatch on the call's ISA level (wavelet_shared.h). */
#include "wavelet_shared.h"

#include <immintrin.h>
#include <math.h>

static const float weights[9] = {
    0.0625f, 0.125f, 0.0625f, 0.125f, 0.25f, 0.125f, 0.0625f, 0.125f, 0.0625f
};

void cn_wavelet_interior_avx2(const wavelet_job *job, size_t from, size_t to)
{
    const size_t width = job->width, height = job->height, radius = job->radius, plane_size = height * width;
    const size_t y = from / width % height, plane = from / plane_size * plane_size;
    const size_t rows[3] = {y < radius ? 0 : y - radius, y, radius >= height - y ? height - 1 : y + radius};
    const size_t x = from - plane - y * width;
    const float *tap[9];
    for (size_t dy = 0; dy < 3; ++dy)
        for (size_t dx = 0; dx < 3; ++dx) tap[dy * 3 + dx] = job->source + plane + rows[dy] * width + x - radius + dx * radius;
    size_t i = from, k = 0;
    for (; to - i >= 8; i += 8, k += 8) {
        __m256 value = _mm256_setzero_ps();
        for (size_t t = 0; t < 9; ++t) {
            const __m256 sample = _mm256_loadu_ps(tap[t] + k), weight = _mm256_set1_ps(weights[t]);
            value = job->fused ? _mm256_fmadd_ps(sample, weight, value)
                               : _mm256_add_ps(value, _mm256_mul_ps(sample, weight));
        }
        _mm256_storeu_ps(job->low + i, value);
        if (job->high) {
            const __m256 difference = _mm256_sub_ps(_mm256_loadu_ps(job->source + i), value);
            const __m256 previous = job->first ? _mm256_setzero_ps() : _mm256_loadu_ps(job->high + i);
            _mm256_storeu_ps(job->high + i, _mm256_add_ps(previous, difference));
        }
    }
    for (; i < to; ++i, ++k) {
        float value = 0.0f;
        for (size_t t = 0; t < 9; ++t) {
            if (job->fused) value = fmaf(tap[t][k], weights[t], value);
            else {
                float product = tap[t][k] * weights[t];
                value = value + product;
            }
        }
        job->low[i] = value;
        if (job->high) {
            float difference = job->source[i] - value;
            float previous = job->first ? 0.0f : job->high[i];
            job->high[i] = previous + difference;
        }
    }
}
