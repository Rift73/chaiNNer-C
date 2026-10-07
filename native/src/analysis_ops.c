#include "chainner.h"
#include "parallel.h"
#include "numeric.h"
#include "cn_crt_math.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Match NumPy's float32 pairwise reduction tree, including its 128-element
   leaves. Keeping the tree independent of the worker count makes all summary
   values deterministic. Pixel transforms, unlike reductions, use the pool. */
static float pair_sum(const float *x, size_t n, size_t stride) {
    if (n < 8) {
        float total = -0.0f;
        for (size_t i = 0; i < n; ++i) total += x[i * stride];
        return total;
    }
    if (n <= 128) {
        float r[8];
        for (size_t j = 0; j < 8; ++j) r[j] = x[j * stride];
        size_t i = 8;
        for (; i < n - n % 8; i += 8)
            for (size_t j = 0; j < 8; ++j) r[j] += x[(i + j) * stride];
        float total = ((r[0] + r[1]) + (r[2] + r[3])) +
                      ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) total += x[i * stride];
        return total;
    }
    size_t left = n / 2;
    left -= left % 8;
    return pair_sum(x, left, stride) + pair_sum(x + left * stride, n - left, stride);
}

float cn_numpy_sum_f32(const float *x, size_t n, size_t stride, size_t block, size_t period) {
    /* NumPy 2.5.3 nditer_constr.c npyiter_find_buffering_setup decides the
       length each reduce inner-loop call receives: the whole array when the
       operand coalesces to one stride unbuffered, else buffer-sized blocks that
       restart at every outer index ("Never buffer beyond the first outer
       dimension", :1950). The block sums are added to the identity in order. */
    if (!period || period > n) period = n;
    if (!block || block > period) block = period;
    float total = 0.0f;
    for (size_t row = 0; row < n; row += period) {
        size_t end = n - row < period ? n : row + period;
        for (size_t begin = row; begin < end; begin += block) {
            size_t count = end - begin < block ? end - begin : block;
            total += pair_sum(x + begin * stride, count, stride);
        }
    }
    return total;
}

static int float_compare(const void *a, const void *b) {
    float x = *(const float *)a, y = *(const float *)b;
    if (isnan(x)) return isnan(y) ? 0 : 1;
    if (isnan(y)) return -1;
    return (x > y) - (x < y);
}

CN_EXPORT cn_status cn_analysis_mean(const float *src, size_t n, size_t block, size_t period,
                                     float *result) {
    if (!src || !result) return CN_INVALID_ARGUMENT;
    if (n > SIZE_MAX / sizeof(float)) return CN_SIZE_OVERFLOW;
    /* np.mean of an empty array: _mean's 0/0 in float64 gives x86's default NaN. */
    *result = n ? numpy_mean_f32(cn_numpy_sum_f32(src, n, 1, block, period), n)
               : cn_f32_from_bits(CN_DEFAULT_NANF_BITS);
    return CN_OK;
}

CN_EXPORT cn_status cn_analysis_grayscale(const float *src, size_t pixels,
                                         size_t channels, float tolerance,
                                         int *result) {
    if (!src || !result || !pixels || (channels != 1 && channels < 3))
        return CN_INVALID_ARGUMENT;
    if (channels > SIZE_MAX / sizeof(float) ||
        pixels > SIZE_MAX / sizeof(float) / channels) return CN_SIZE_OVERFLOW;
    *result = 1;
    if (channels == 1) return CN_OK;
    for (size_t p = 0; p < pixels; ++p) {
        const float *v = src + p * channels;
        if (!(fabsf(v[0] - v[1]) <= tolerance) ||
            !(fabsf(v[0] - v[2]) <= tolerance)) { *result = 0; break; }
    }
    return CN_OK;
}

typedef struct binary_job {
    const float *a, *b;
    float *out;
    int op;
    float scale, threshold;
} binary_job;

static void binary_range(void *context, size_t begin, size_t end) {
    binary_job *j = (binary_job *)context;
    for (size_t i = begin; i < end; ++i) {
        float x = j->a[i], y = j->b[i], v;
        switch (j->op) {
        case 0: v = x - y; v *= j->scale; v += 0.5f; break;
        case 1: {
            float difference = x - y;
            float magnitude = fabsf(difference) - j->threshold;
            magnitude = magnitude < 0 ? 0 : magnitude;
            float sign = isnan(difference) ? difference :
                (difference > 0 ? 1.0f : (difference < 0 ? -1.0f : 0.0f));
            v = sign * magnitude; v *= j->scale; v += x;
            break;
        }
        case 2: v = x - y; v *= v; break;
        default: v = x * y; break;
        }
        j->out[i] = v;
    }
}

CN_EXPORT cn_status cn_analysis_binary(const float *a, const float *b, float *out,
                                      size_t n, int op, float scale, float threshold) {
    if (!a || !b || !out || op < 0 || op > 3) return CN_INVALID_ARGUMENT;
    if (n > SIZE_MAX / sizeof(float)) return CN_SIZE_OVERFLOW;
    binary_job job = {a, b, out, op, scale, threshold};
    return cn_parallel_for(n, 65536, binary_range, &job);
}

typedef struct ssim_job {
    const float *mu1, *mu2, *v1, *v2, *cov;
    float *out;
} ssim_job;

static void ssim_range(void *context, size_t begin, size_t end) {
    ssim_job *j = (ssim_job *)context;
    for (size_t i = begin; i < end; ++i) {
        float x2 = j->mu1[i] * j->mu1[i], y2 = j->mu2[i] * j->mu2[i];
        float xy = j->mu1[i] * j->mu2[i];
        float vx = j->v1[i] - x2, vy = j->v2[i] - y2, cov = j->cov[i] - xy;
        float top1 = 2.0f * xy; top1 += 0.0001f;
        float top2 = 2.0f * cov; top2 += 0.0009f;
        float bottom1 = x2 + y2; bottom1 += 0.0001f;
        float bottom2 = vx + vy; bottom2 += 0.0009f;
        j->out[i] = (top1 * top2) / (bottom1 * bottom2);
    }
}

CN_EXPORT cn_status cn_analysis_ssim(const float *mu1, const float *mu2,
                                    const float *v1, const float *v2,
                                    const float *cov, float *out, size_t n) {
    if (!mu1 || !mu2 || !v1 || !v2 || !cov || !out) return CN_INVALID_ARGUMENT;
    if (n > SIZE_MAX / sizeof(float)) return CN_SIZE_OVERFLOW;
    ssim_job job = {mu1, mu2, v1, v2, cov, out};
    return cn_parallel_for(n, 65536, ssim_range, &job);
}

static double round_even(double x) {
    double floor_x = floor(x), fraction = x - floor_x;
    return floor_x + ((fraction > 0.5 ||
        (fraction == 0.5 && fmod(floor_x, 2.0) != 0)) ? 1.0 : 0.0);
}

static size_t palette_index(float x, size_t levels) {
    double scaled;
    if (levels <= 65536) scaled = (double)(x * (float)(levels - 1));
    else scaled = (double)x * (double)(levels - 1);
    /* The installed NumPy cast loop narrows through signed int32 for U8/U16
       and signed int64 for U32. Invalid conversions yield the signed minimum,
       whose low destination bits are zero. Express it without undefined C casts. */
    if (!isfinite(scaled)) return 0;
    double rounded = round_even(scaled);
    double limit = levels <= 65536 ? 2147483648.0 : 9223372036854775808.0;
    if (rounded < -limit || rounded >= limit) return 0;
    double modulus = levels <= 256 ? 256.0 : (levels <= 65536 ? 65536.0 : 4294967296.0);
    rounded = fmod(rounded, modulus);
    if (rounded < 0) rounded += modulus;
    return (size_t)rounded;
}

typedef struct palette_job {
    const float *image, *palette;
    float *out;
    size_t levels, channels;
} palette_job;

static void palette_range(void *context, size_t begin, size_t end) {
    palette_job *j = (palette_job *)context;
    for (size_t p = begin; p < end; ++p) {
        size_t index = palette_index(j->image[p], j->levels);
        memcpy(j->out + p * j->channels, j->palette + index * j->channels,
               j->channels * sizeof(float));
    }
}

CN_EXPORT cn_status cn_analysis_palette(const float *image, const float *palette,
                                       float *out, size_t pixels, size_t levels,
                                       size_t channels) {
    if (!image || !palette || !out || !pixels || !channels ||
        !levels || levels > 16777216 ||
        (uintptr_t)image % _Alignof(float) || (uintptr_t)palette % _Alignof(float) ||
        (uintptr_t)out % _Alignof(float)) return CN_INVALID_ARGUMENT;
    if (channels > SIZE_MAX / sizeof(float) ||
        pixels > SIZE_MAX / sizeof(float) / channels ||
        levels > SIZE_MAX / sizeof(float) / channels) return CN_SIZE_OVERFLOW;
    size_t source_bytes = pixels * sizeof(float), palette_bytes = levels * channels * sizeof(float);
    size_t output_bytes = pixels * channels * sizeof(float);
    uintptr_t a = (uintptr_t)image, b = (uintptr_t)palette, c = (uintptr_t)out;
    if (a > UINTPTR_MAX - source_bytes || b > UINTPTR_MAX - palette_bytes ||
        c > UINTPTR_MAX - output_bytes) return CN_SIZE_OVERFLOW;
    if ((a < c + output_bytes && c < a + source_bytes) ||
        (b < c + output_bytes && c < b + palette_bytes)) return CN_INVALID_ARGUMENT;
    /* Validate before dispatch: malformed float indices never reach a read. */
    for (size_t p = 0; p < pixels; ++p)
        if (palette_index(image[p], levels) >= levels) return CN_INVALID_ARGUMENT;
    palette_job job = {image, palette, out, levels, channels};
    return cn_parallel_for(pixels, 16384, palette_range, &job);
}

/* VOID_compare of two rows viewed as structured voids: FLOAT_compare field by field,
   so NaN sorts last and +-0, or two NaNs, compare equal. */
static int row_compare(const float *a, const float *b, size_t channels) {
    for (size_t c = 0; c < channels; ++c) {
        int order = float_compare(a + c, b + c);
        if (order) return order;
    }
    return 0;
}

/* GENERIC_COPY and GENERIC_SWAP: rows move as bytes, so every payload is kept. */
static void row_copy(float *dst, const float *src, size_t channels) {
    memcpy(dst, src, channels * sizeof(float));
}

static void row_swap(float *a, float *b, float *tmp, size_t channels) {
    row_copy(tmp, a, channels);
    row_copy(a, b, channels);
    row_copy(b, tmp, channels);
}

/* NumPy 2.5.3's generic npy_heapsort (heapsort.cpp) of the num rows at start; its
   one-based a[k] is start's row k - 1. */
static void rows_heapsort(float *start, size_t num, size_t channels, float *tmp) {
    size_t i, j, l;
    for (l = num >> 1; l > 0; --l) {
        row_copy(tmp, start + (l - 1) * channels, channels);
        for (i = l, j = l << 1; j <= num;) {
            if (j < num &&
                row_compare(start + (j - 1) * channels, start + j * channels, channels) < 0)
                j += 1;
            if (row_compare(tmp, start + (j - 1) * channels, channels) < 0) {
                row_copy(start + (i - 1) * channels, start + (j - 1) * channels, channels);
                i = j;
                j += j;
            } else {
                break;
            }
        }
        row_copy(start + (i - 1) * channels, tmp, channels);
    }
    for (; num > 1;) {
        row_copy(tmp, start + (num - 1) * channels, channels);
        row_copy(start + (num - 1) * channels, start, channels);
        num -= 1;
        for (i = 1, j = 2; j <= num;) {
            if (j < num &&
                row_compare(start + (j - 1) * channels, start + j * channels, channels) < 0)
                j++;
            if (row_compare(tmp, start + (j - 1) * channels, channels) < 0) {
                row_copy(start + (i - 1) * channels, start + (j - 1) * channels, channels);
                i = j;
                j += j;
            } else {
                break;
            }
        }
        row_copy(start + (i - 1) * channels, tmp, channels);
    }
}

/* NumPy 2.5.3's generic npy_quicksort (quicksort.cpp) of the num rows at start:
   median-of-three partitions while more than SMALL_QUICKSORT = 16 rows remain, the
   larger part pushed with the depth left from 2 floor(log2 num); a part popped past
   that depth goes to npy_heapsort, every other part ends in insertion sort. pl, pr,
   pm, pi, pj and pk are row indices; pivot is vp. PYA_QS_STACK is 128.
   test_analysis_ops reaches the heapsort with McIlroy's adversary on NumPy's sort. */
static void rows_quicksort(float *start, size_t num, size_t channels, float *pivot,
                           float *tmp) {
    size_t stack[128], *sptr = stack, pl = 0, pr = num - 1, pm, pi, pj, pk;
    int depth[128], *psdepth = depth, cdepth = 0;
    for (size_t unum = num; unum >>= 1;) cdepth++;
    cdepth *= 2;
    for (;;) {
        if (cdepth < 0) {
            rows_heapsort(start + pl * channels, pr - pl + 1, channels, tmp);
            goto stack_pop;
        }
        while (pr - pl > 15) {
            pm = pl + ((pr - pl) >> 1);
            if (row_compare(start + pm * channels, start + pl * channels, channels) < 0)
                row_swap(start + pm * channels, start + pl * channels, tmp, channels);
            if (row_compare(start + pr * channels, start + pm * channels, channels) < 0)
                row_swap(start + pr * channels, start + pm * channels, tmp, channels);
            if (row_compare(start + pm * channels, start + pl * channels, channels) < 0)
                row_swap(start + pm * channels, start + pl * channels, tmp, channels);
            row_copy(pivot, start + pm * channels, channels);
            pi = pl;
            pj = pr - 1;
            row_swap(start + pm * channels, start + pj * channels, tmp, channels);
            for (;;) {
                do {
                    ++pi;
                } while (row_compare(start + pi * channels, pivot, channels) < 0 && pi < pj);
                do {
                    --pj;
                } while (row_compare(pivot, start + pj * channels, channels) < 0 && pi < pj);
                if (pi >= pj) break;
                row_swap(start + pi * channels, start + pj * channels, tmp, channels);
            }
            pk = pr - 1;
            row_swap(start + pi * channels, start + pk * channels, tmp, channels);
            /* push largest partition on stack */
            if (pi - pl < pr - pi) {
                *sptr++ = pi + 1;
                *sptr++ = pr;
                pr = pi - 1;
            } else {
                *sptr++ = pl;
                *sptr++ = pi - 1;
                pl = pi + 1;
            }
            *psdepth++ = --cdepth;
        }
        /* insertion sort */
        for (pi = pl + 1; pi <= pr; ++pi) {
            row_copy(pivot, start + pi * channels, channels);
            pj = pi;
            pk = pi - 1;
            while (pj > pl && row_compare(pivot, start + pk * channels, channels) < 0) {
                row_copy(start + pj * channels, start + pk * channels, channels);
                pj--;
                pk--;
            }
            row_copy(start + pj * channels, pivot, channels);
        }
    stack_pop:
        if (sptr == stack) break;
        pr = *(--sptr);
        pl = *(--sptr);
        cdepth = *(--psdepth);
    }
}

/* np.unique(image.reshape(-1, channels), axis=0) on NumPy 2.5.3 (_unique1d): the
   rows, viewed as one structured void each, take no hash path; ar.sort() sorts a
   copy with npy_quicksort and VOID_compare, then mask[1:] = aux[1:] != aux[:-1]
   (field by field, NaN never equal) keeps the first row of each run. The copy is
   sorted in out (pixels x channels, not overlapping src) and compacted there. */
CN_EXPORT cn_status cn_analysis_distinct(const float *src, float *out,
                                        size_t pixels, size_t channels, size_t *count) {
    if (!src || !out || !count || !pixels || !channels) return CN_INVALID_ARGUMENT;
    if (channels > SIZE_MAX / sizeof(float) / 2 ||
        pixels > SIZE_MAX / sizeof(float) / channels) return CN_SIZE_OVERFLOW;
    float *pivot = (float *)malloc(2 * channels * sizeof(float));
    if (!pivot) return CN_ALLOCATION_FAILED;
    memcpy(out, src, pixels * channels * sizeof(float));
    rows_quicksort(out, pixels, channels, pivot, pivot + channels);
    /* Rows before written hold the kept rows. written <= i, and a kept row is copied
       only when written < i, so row i - 1 still holds the sorted row i - 1. */
    size_t written = 1;
    for (size_t i = 1; i < pixels; ++i) {
        const float *row = out + i * channels, *previous = row - channels;
        int same = 1;
        for (size_t c = 0; same && c < channels; ++c) same = row[c] == previous[c];
        if (same) continue;
        if (written != i) row_copy(out + written * channels, row, channels);
        ++written;
    }
    *count = written;
    free(pivot);
    return CN_OK;
}

typedef struct cas_job {
    const void *low, *high;
    const float *image, *sharpened;
    void *out;
    size_t channels;
    double exponent;
    int is_double, is_mix;
} cas_job;

/* NumPy 2.5.3's FLOAT/DOUBLE_power (loops_umath_fp.dispatch.c.src) answers a
   scalar exponent of -1, 0, 0.5, 1 or 2, compared in the array's own type,
   without pow: 0.5 is sqrt (so -0 stays -0 and -inf gives NaN), 2 is x * x.
   Any other exponent is the process's ucrtbase.dll pow (no SVML on Windows). */
static double numpy_power_f64(double x, double e) {
    if (e == -1.0) return 1.0 / x;
    if (e == 0.0) return 1.0;
    if (e == 0.5) return sqrt(x);
    if (e == 1.0) return x;
    if (e == 2.0) return x * x;
    return cn_crt.pow(x, e);
}

static float numpy_power_f32(float x, float e) {
    if (e == -1.0f) return (float)(1.0 / x);
    if (e == 0.0f) return 1.0f;
    if (e == 0.5f) return sqrtf(x);
    if (e == 1.0f) return x;
    if (e == 2.0f) return x * x;
    return cn_crt.powf(x, e);
}

static void cas_range(void *context, size_t begin, size_t end) {
    cas_job *j = (cas_job *)context;
    for (size_t p = begin; p < end; ++p) {
        if (j->is_double) {
            const double *a = (const double *)j->low, *b = (const double *)j->high;
            double *out = (double *)j->out;
            if (j->is_mix) {
                double inverse = 1.0 - a[p];
                for (size_t c = 0; c < j->channels; ++c) {
                    size_t i = p * j->channels + c;
                    out[i] = j->image[i] * inverse + j->sharpened[i] * a[p];
                }
            } else {
                double x = 1.0 - b[p];
                x = isnan(x) || x < a[p] ? x : a[p];
                x /= b[p] + 1e-8;
                out[p] = numpy_power_f64(x, j->exponent);
            }
        } else {
            const float *a = (const float *)j->low, *b = (const float *)j->high;
            float *out = (float *)j->out;
            if (j->is_mix) {
                float inverse = 1.0f - a[p];
                for (size_t c = 0; c < j->channels; ++c) {
                    size_t i = p * j->channels + c;
                    out[i] = j->image[i] * inverse + j->sharpened[i] * a[p];
                }
            } else {
                float x = 1.0f - b[p];
                x = isnan(x) || x < a[p] ? x : a[p];
                x /= b[p] + 1e-8f;
                out[p] = numpy_power_f32(x, (float)j->exponent);
            }
        }
    }
}

CN_EXPORT cn_status cn_analysis_cas(const void *low, const void *high,
                                   const float *image, const float *sharpened,
                                   void *out, size_t pixels, size_t channels,
                                   double exponent, int is_double, int is_mix) {
    if (!low || !out || !pixels || !channels ||
        (!is_mix && !high) || (is_mix && (!image || !sharpened))) return CN_INVALID_ARGUMENT;
    if (channels > SIZE_MAX / sizeof(double) ||
        pixels > SIZE_MAX / sizeof(double) / channels) return CN_SIZE_OVERFLOW;
    cas_job job = {low, high, image, sharpened, out, channels, exponent, is_double, is_mix};
    return cn_parallel_for(pixels, 16384, cas_range, &job);
}

typedef struct correction_job {
    const float *image, *reference, *alpha, *reference_alpha;
    float *out, *alpha_out;
    int add;
} correction_job;

static void correction_range(void *context, size_t begin, size_t end) {
    correction_job *j = (correction_job *)context;
    for (size_t p = begin; p < end; ++p) {
        for (size_t c = 0; c < 3; ++c) {
            size_t i = p * 3 + c;
            if (j->add) j->out[i] = j->image[i] + j->reference[i];
            else j->out[i] = j->reference[i] - j->image[i];
        }
        if (!j->add && ((j->alpha && j->alpha[p] == 0) ||
                       (j->reference_alpha && j->reference_alpha[p] == 0)))
            /* Preserve upstream's (H,W,1) nonzero indexing: only channel 0
               is cleared. Changing all three would change existing output. */
            j->out[p * 3] = 0;
        if (j->alpha_out) {
            if (j->add) j->alpha_out[p] = j->alpha[p] +
                (j->alpha[p] == 0 ? 0.0f : j->reference_alpha[p]);
            else j->alpha_out[p] = j->reference_alpha[p] - j->alpha[p];
        }
    }
}

CN_EXPORT cn_status cn_analysis_correction(const float *image, const float *reference,
                                          const float *alpha, const float *reference_alpha,
                                          float *out, float *alpha_out, size_t pixels, int add) {
    if (!image || !reference || !out || !pixels ||
        (alpha_out && (!alpha || !reference_alpha))) return CN_INVALID_ARGUMENT;
    if (pixels > SIZE_MAX / sizeof(float) / 3) return CN_SIZE_OVERFLOW;
    correction_job job = {image, reference, alpha, reference_alpha, out, alpha_out, add};
    return cn_parallel_for(pixels, 16384, correction_range, &job);
}

/* block and period: the reduce blocks of one upstream np.split column
   (numpy_sum_block). */
CN_EXPORT cn_status cn_analysis_channel_stats(const float *image, size_t pixels,
                                             size_t block, size_t period, float *out) {
    if (!image || !out) return CN_INVALID_ARGUMENT;
    if (pixels > SIZE_MAX / sizeof(float) / 3) return CN_SIZE_OVERFLOW;
    if (!pixels) {
        /* .mean() and .std() of an empty column: _mean's 0/0, x86's default NaN. */
        for (size_t c = 0; c < 6; ++c) out[c] = cn_f32_from_bits(CN_DEFAULT_NANF_BITS);
        return CN_OK;
    }
    float *squared = (float *)malloc(pixels * sizeof(float));
    if (!squared) return CN_ALLOCATION_FAILED;
    for (size_t c = 0; c < 3; ++c) {
        float mean = numpy_mean_f32(cn_numpy_sum_f32(image + c, pixels, 3, block, period), pixels);
        for (size_t p = 0; p < pixels; ++p) {
            float difference = image[p * 3 + c] - mean;
            squared[p] = difference * difference;
        }
        out[c * 2] = mean;
        /* _var squares into a fresh aligned (N, 1) array (subtract with out=...),
           which NumPy reduces unbuffered: one pairwise tree (block 0). */
        out[c * 2 + 1] = sqrtf(numpy_mean_f32(cn_numpy_sum_f32(squared, pixels, 1, 0, 0), pixels));
    }
    free(squared);
    return CN_OK;
}

typedef struct transfer_job {
    const float *image;
    float *out;
    float mean[3], target[3], ratio[3];
    float low[3], high[3], range_low[3], range_high[3];
    int scale, finish;
} transfer_job;

static void transfer_range(void *context, size_t begin, size_t end) {
    transfer_job *j = (transfer_job *)context;
    for (size_t p = begin; p < end; ++p) {
        for (size_t c = 0; c < 3; ++c) {
            size_t i = p * 3 + c;
            float x;
            if (!j->finish) {
                x = j->image[i] - j->mean[c];
                x *= j->ratio[c]; x += j->target[c];
            } else {
                x = j->out[i];
                if (!j->scale) {
                    x = x < j->low[c] ? j->low[c] : x;
                    x = x > j->high[c] ? j->high[c] : x;
                } else if (j->range_low[c] < j->low[c] || j->range_high[c] > j->high[c]) {
                    float span = j->high[c] - j->low[c];
                    x -= j->range_low[c]; x *= span;
                    x /= j->range_high[c] - j->range_low[c]; x += j->low[c];
                }
            }
            j->out[i] = x;
        }
    }
}

CN_EXPORT cn_status cn_analysis_transfer(const float *image, float *out,
                                        const unsigned char *mask, size_t pixels,
                                        const float *stats, const float *reference_stats,
                                        const float *limits, int reciprocal, int scale) {
    if (!image || !out || !mask || !pixels || !stats || !reference_stats || !limits)
        return CN_INVALID_ARGUMENT;
    if (pixels > SIZE_MAX / sizeof(float) / 3) return CN_SIZE_OVERFLOW;
    transfer_job job = {0};
    job.image = image; job.out = out; job.scale = scale;
    for (size_t c = 0; c < 3; ++c) {
        job.mean[c] = stats[c * 2]; job.target[c] = reference_stats[c * 2];
        job.ratio[c] = reciprocal ? reference_stats[c * 2 + 1] / stats[c * 2 + 1]
                                 : stats[c * 2 + 1] / reference_stats[c * 2 + 1];
        job.low[c] = limits[c * 2]; job.high[c] = limits[c * 2 + 1];
    }
    cn_status status = cn_parallel_for(pixels, 16384, transfer_range, &job);
    if (status != CN_OK) return status;
    if (scale) {
        size_t first = 0;
        while (first < pixels && !mask[first]) ++first;
        if (first == pixels) return CN_INVALID_ARGUMENT;
        for (size_t c = 0; c < 3; ++c) {
            float low = out[first * 3 + c], high = low;
            for (size_t p = first + 1; p < pixels; ++p) if (mask[p]) {
                float x = out[p * 3 + c];
                if (x < low || isnan(x)) low = x;
                if (x > high || isnan(x)) high = x;
            }
            job.range_low[c] = low; job.range_high[c] = high;
            /* Python max([nan, bound]) keeps nan, as these comparisons do. */
            job.low[c] = low < job.low[c] ? job.low[c] : low;
            job.high[c] = high > job.high[c] ? job.high[c] : high;
        }
    }
    job.finish = 1;
    return cn_parallel_for(pixels, 16384, transfer_range, &job);
}
