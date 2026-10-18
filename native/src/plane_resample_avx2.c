#if !defined(_M_X64)
#error "plane_resample_avx2.c is an x86-64 /arch:AVX2 unit"
#endif
/* The plane resampler's AVX2 unit (/arch:AVX2, spec 4.3): no export, reached only
 * through cn_plane_resample_axis's dispatch on the call's ISA level. Each lane runs
 * its output's sequence of plane_resample_shared.h (vsubps, vmulps, vaddps, no
 * contraction); groups of 8 columns start at column 0, and each row's last
 * columns run the scalar row. */
#include "plane_resample_shared.h"
#include <immintrin.h>

/* Columns [0, dst_dim / 16 * 16) in pairs of 8-lane groups (two gathers in
 * flight), then one more group of 8 when it fits; returns the first column left. */
static size_t horizontal_groups(const plane_resample_job *job, size_t y)
{
    const float *row = job->src + y * job->row_stride;
    float *out = job->out + y * job->dst_dim;
    const __m256 bias_in = _mm256_set1_ps(job->bias_in), bias_out = _mm256_set1_ps(job->bias_out);
    int add = job->bias_out != 0.0f;
    size_t x = 0, dst = job->dst_dim;
    for (; dst - x >= 16; x += 16) {
        __m256 a = _mm256_setzero_ps(), b = _mm256_setzero_ps();
        for (size_t t = 0; t < job->taps; ++t) {
            const int32_t *index = job->index + t * dst + x;
            const float *weight = job->weights + t * dst + x;
            __m256 ga = _mm256_i32gather_ps(row, _mm256_loadu_si256((const __m256i *)index), 4);
            __m256 gb = _mm256_i32gather_ps(row, _mm256_loadu_si256((const __m256i *)(index + 8)), 4);
            a = _mm256_add_ps(a, _mm256_mul_ps(_mm256_sub_ps(ga, bias_in), _mm256_loadu_ps(weight)));
            b = _mm256_add_ps(b, _mm256_mul_ps(_mm256_sub_ps(gb, bias_in), _mm256_loadu_ps(weight + 8)));
        }
        if (add) {
            a = _mm256_add_ps(a, bias_out);
            b = _mm256_add_ps(b, bias_out);
        }
        _mm256_storeu_ps(out + x, a);
        _mm256_storeu_ps(out + x + 8, b);
    }
    if (dst - x >= 8) {
        __m256 a = _mm256_setzero_ps();
        for (size_t t = 0; t < job->taps; ++t) {
            const int32_t *index = job->index + t * dst + x;
            __m256 ga = _mm256_i32gather_ps(row, _mm256_loadu_si256((const __m256i *)index), 4);
            a = _mm256_add_ps(a, _mm256_mul_ps(_mm256_sub_ps(ga, bias_in),
                _mm256_loadu_ps(job->weights + t * dst + x)));
        }
        _mm256_storeu_ps(out + x, add ? _mm256_add_ps(a, bias_out) : a);
        x += 8;
    }
    return x;
}

/* Columns [0, width / 8 * 8): four groups of 8 at a time (32 columns), then single
 * groups; every tap's row is read at the same columns. Returns the first column
 * left. */
static size_t vertical_groups(const plane_resample_job *job, size_t y)
{
    float *out = job->out + y * job->width;
    const __m256 bias_in = _mm256_set1_ps(job->bias_in), bias_out = _mm256_set1_ps(job->bias_out);
    int add = job->bias_out != 0.0f;
    size_t x = 0, width = job->width;
    for (; width - x >= 32; x += 32) {
        __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
            _mm256_setzero_ps()};
        for (size_t t = 0; t < job->taps; ++t) {
            size_t k = t * job->dst_dim + y;
            const float *row = job->src + (size_t)job->index[k] * job->row_stride + x;
            __m256 weight = _mm256_set1_ps(job->weights[k]);
            for (int g = 0; g < 4; ++g)
                acc[g] = _mm256_add_ps(acc[g], _mm256_mul_ps(
                    _mm256_sub_ps(_mm256_loadu_ps(row + 8 * g), bias_in), weight));
        }
        for (int g = 0; g < 4; ++g)
            _mm256_storeu_ps(out + x + 8 * g, add ? _mm256_add_ps(acc[g], bias_out) : acc[g]);
    }
    for (; width - x >= 8; x += 8) {
        __m256 acc = _mm256_setzero_ps();
        for (size_t t = 0; t < job->taps; ++t) {
            size_t k = t * job->dst_dim + y;
            const float *row = job->src + (size_t)job->index[k] * job->row_stride + x;
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(row), bias_in),
                _mm256_set1_ps(job->weights[k])));
        }
        _mm256_storeu_ps(out + x, add ? _mm256_add_ps(acc, bias_out) : acc);
    }
    return x;
}

void cn_plane_resample_rows_avx2(const plane_resample_job *job, size_t begin, size_t end)
{
    for (size_t y = begin; y < end; ++y) {
        if (job->vertical)
            resample_row_vertical(job, y, vertical_groups(job, y));
        else
            resample_row_horizontal(job, y, horizontal_groups(job, y));
    }
}
