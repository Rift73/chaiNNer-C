/* Color-transfer preparation and matrix construction, adapted from the
 * project's MIT colortrans helpers (Daniel Steinberg, 2022) and NumPy 2.5.3
 * covariance/linalg/BLAS dispatch (NumPy Developers, BSD-3-Clause).
 * Public OpenBLAS ILP64 exports of NumPy's bundled scipy-openblas 0.3.34
 * perform the established BLAS/LAPACK kernels; they remain an external
 * native dependency, not rewritten C eigensolvers.
 * References at numpy/numpy tag v2.5.3: numpy/lib/_function_base_impl.py,
 * numpy/_core/_methods.py, numpy/linalg/_linalg.py,
 * numpy/linalg/umath_linalg.cpp, numpy/_core/src/common/cblasfuncs.c,
 * numpy/_core/src/umath/{loops.c.src,clip.cpp},
 * numpy/_core/src/npymath/npy_math_complex.c.src.
 * NumPy 2 np.linalg.eig always returns complex128, so every step after the
 * eigensystem follows NumPy's complex arithmetic and zgemm/zgetrf/zgesv.
 */
#include "chainner.h"
#include "cn_crt_math.h"
#include "parallel.h"
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef int64_t lapack_int;
typedef struct { double real, imag; } complex128;
typedef void (*gemm_complex_fn)(int, int, int, lapack_int, lapack_int, lapack_int,
    const complex128 *, const complex128 *, lapack_int, const complex128 *, lapack_int,
    const complex128 *, complex128 *, lapack_int);
typedef void (*syrk_fn)(int, int, int, lapack_int, lapack_int, double,
    const double *, lapack_int, double, double *, lapack_int);
typedef void (*geev_fn)(const char *, const char *, const lapack_int *, double *,
    const lapack_int *, double *, double *, double *, const lapack_int *, double *,
    const lapack_int *, double *, const lapack_int *, lapack_int *);
typedef void (*getrf_complex_fn)(const lapack_int *, const lapack_int *, complex128 *,
    const lapack_int *, lapack_int *, lapack_int *);
typedef void (*gesv_complex_fn)(const lapack_int *, const lapack_int *, complex128 *,
    const lapack_int *, lapack_int *, complex128 *, const lapack_int *, lapack_int *);
typedef struct {
    gemm_complex_fn zgemm;
    syrk_fn syrk;
    geev_fn geev;
    getrf_complex_fn zgetrf;
    gesv_complex_fn zgesv;
} transfer_backend;

static int backend_valid(const transfer_backend *b) {
    return b && b->zgemm && b->syrk && b->geev && b->zgetrf && b->zgesv;
}

static double read_value(const void *source, size_t i, int f64) {
    return f64 ? ((const double *)source)[i] : (double)((const float *)source)[i];
}

/* Images are row-major pixel grids: pixel i's channel c at i * pixel + c * channel
 * elements, interleaved (channels, 1) or planar (1, plane), read in place (D-16). */
typedef struct { size_t pixel, channel; } pixel_layout;

static size_t element(pixel_layout layout, size_t i, size_t c) {
    return i * layout.pixel + c * layout.channel;
}

static int layout_valid(pixel_layout layout) {
    return layout.pixel && layout.channel;
}

/* This axis-zero reduction is sequential in NumPy's Nx3 C-order array (the
 * boolean-mask gather, row-major whatever the image's layout); np.cov separately
 * promotes its data and mean to float64. */
static cn_status covariance(const void *image, pixel_layout layout,
        const unsigned char *mask, size_t pixels, int f64, void *mean, double *cov,
        const transfer_backend *b, int *error) {
    size_t count = 0;
    for (size_t i = 0; i < pixels; ++i) count += mask[i] != 0;
    /* np.cov transposes when `not rowvar and m.ndim != 1`, so fewer than two
     * pixels leave no degrees of freedom: the covariance is 0 * inf = NaN,
     * which eig rejects as non-finite. */
    if (count < 2) { *error = 1; return CN_OK; }
    double *data = malloc(count * 3 * sizeof(double));
    if (!data) return CN_ALLOCATION_FAILED;
    double average[3] = {0, 0, 0};
    float float_average[3] = {0, 0, 0};
    size_t at = 0;
    for (size_t i = 0; i < pixels; ++i) if (mask[i]) {
        for (size_t c = 0; c < 3; ++c) {
            double value = read_value(image, element(layout, i, c), f64);
            data[at * 3 + c] = value;
            average[c] += value;
            if (!f64) float_average[c] += (float)value;
        }
        ++at;
    }
    for (size_t c = 0; c < 3; ++c) {
        average[c] /= (double)count;
        /* np.mean divides by an np.intp count, which NEP 50 no longer treats
         * as weak: the float32 sum divides in float64, then casts back. */
        if (f64) ((double *)mean)[c] = average[c];
        else ((float *)mean)[c] = (float)((double)float_average[c] / (double)count);
    }
    for (size_t i = 0; i < count; ++i)
        for (size_t c = 0; c < 3; ++c) data[i * 3 + c] -= average[c];
    /* NumPy recognizes X @ X.T and calls the upper-triangle SYRK kernel. */
    b->syrk(101, 121, 112, 3, (lapack_int)count, 1, data, 3, 0, cov, 3);
    double factor = 1.0 / (double)(count - 1);
    for (size_t r = 0; r < 3; ++r)
        for (size_t c = 0; c < 3; ++c) {
            if (r > c) cov[r * 3 + c] = cov[c * 3 + r];
        }
    for (size_t i = 0; i < 9; ++i) {
        cov[i] *= factor;
        if (!isfinite(cov[i])) *error = 1;
    }
    free(data);
    return CN_OK;
}

/* umath_linalg's real geev result, made complex as NumPy always returns it. */
static cn_status eigen(const double *matrix, complex128 *values, complex128 *vectors,
        const transfer_backend *b, int *error) {
    double a[9], wr[3], wi[3], vr[9], vl[9], query;
    const lapack_int n = 3;
    lapack_int length = -1, info;
    for (size_t r = 0; r < 3; ++r)
        for (size_t c = 0; c < 3; ++c) a[c * 3 + r] = matrix[r * 3 + c];
    b->geev("N", "V", &n, a, &n, wr, wi, vl, &n, vr, &n, &query, &length, &info);
    if (info || !isfinite(query) || query < 1 || query > 1000000) {
        *error = 3; return CN_OK;
    }
    length = (lapack_int)query;
    double *work = malloc((size_t)length * sizeof(double));
    if (!work) return CN_ALLOCATION_FAILED;
    b->geev("N", "V", &n, a, &n, wr, wi, vl, &n, vr, &n, work, &length, &info);
    free(work);
    if (info) { *error = 3; return CN_OK; }
    for (size_t c = 0; c < 3; ++c) { values[c].real = wr[c]; values[c].imag = wi[c]; }
    for (size_t c = 0; c < 3; ++c) {
        if (wi[c] == 0) {
            for (size_t r = 0; r < 3; ++r) {
                vectors[r * 3 + c].real = vr[c * 3 + r];
                vectors[r * 3 + c].imag = 0;
            }
        } else {
            if (c + 1 == 3 || wi[c] < 0) { *error = 3; return CN_OK; }
            for (size_t r = 0; r < 3; ++r) {
                vectors[r * 3 + c].real = vectors[r * 3 + c + 1].real = vr[c * 3 + r];
                vectors[r * 3 + c].imag = vr[(c + 1) * 3 + r];
                vectors[r * 3 + c + 1].imag = -vr[(c + 1) * 3 + r];
            }
            ++c;
        }
    }
    return CN_OK;
}

/* NumPy's complex true_divide loop (loops.c.src). */
static complex128 divide_complex(complex128 a, complex128 b) {
    complex128 out;
    if (fabs(b.real) >= fabs(b.imag)) {
        if (b.real == 0 && b.imag == 0) {
            out.real = a.real / fabs(b.real); out.imag = a.imag / fabs(b.imag);
        } else {
            double ratio = b.imag / b.real;
            double scale = 1 / (b.real + b.imag * ratio);
            out.real = (a.real + a.imag * ratio) * scale;
            out.imag = (a.imag - a.real * ratio) * scale;
        }
    } else {
        double ratio = b.real / b.imag;
        double scale = 1 / (b.imag + b.real * ratio);
        out.real = (a.real * ratio + a.imag) * scale;
        out.imag = (a.imag * ratio - a.real) * scale;
    }
    return out;
}

/* np.sqrt(z.clip(min=0)). A minimum-only clip is np.maximum(z, 0)
 * (_methods._clip), whose complex loop keeps z when a part is NaN or z >= 0
 * lexicographically (loops.c.src CGE), else gives 0+0j. MSVC builds undefine
 * HAVE_CSQRT (npy_config.h), so np.sqrt is npymath's npy_csqrt. */
static complex128 clipped_root(complex128 z) {
    /* npy_csqrt risks spurious overflow at DBL_MAX / (1 + sqrt(2)). */
    const double threshold = DBL_MAX / (1 + 1.414213562373095048801688724209698807);
    double a = z.real, b = z.imag;
    complex128 result;
    if (!isnan(a) && !isnan(b) && !(a > 0 || (a == 0 && b >= 0))) a = b = 0;
    if (a == 0 && b == 0) { result.real = 0; result.imag = b; return result; }
    if (isinf(b)) { result.real = fabs(b); result.imag = b; return result; }
    if (isnan(a)) { result.real = a; result.imag = (b - b) / (b - b); return result; }
    if (isinf(a)) {
        if (signbit(a)) { result.real = fabs(b - b); result.imag = copysign(a, b); }
        else { result.real = a; result.imag = copysign(b - b, b); }
        return result;
    }
    int scale = fabs(a) >= threshold || fabs(b) >= threshold;
    if (scale) { a *= 0.25; b *= 0.25; }
    if (a >= 0) {
        double t = sqrt((a + cn_crt.hypot(a, b)) * 0.5);
        result.real = t; result.imag = b / (2 * t);
    } else {
        double t = sqrt((-a + cn_crt.hypot(a, b)) * 0.5);
        result.real = fabs(b) / (2 * t); result.imag = copysign(t, b);
    }
    if (scale) result.real *= 2;
    return result;
}

static void product(const complex128 *a, const complex128 *b, int transpose_b,
        complex128 *out, const transfer_backend *backend) {
    const complex128 one = {1, 0}, zero = {0, 0};
    backend->zgemm(101, 111, transpose_b ? 112 : 111, 3, 3, 3,
        &one, a, 3, b, 3, &zero, out, 3);
}

static int singular(const complex128 *matrix, const transfer_backend *b) {
    const lapack_int n = 3;
    lapack_int pivots[3], info;
    double logs = 0;
    complex128 a[9];
    for (size_t r = 0; r < 3; ++r)
        for (size_t c = 0; c < 3; ++c) a[c * 3 + r] = matrix[r * 3 + c];
    b->zgetrf(&n, &n, a, &n, pivots, &info);
    if (info) return 1;
    for (size_t i = 0; i < 3; ++i)
        logs += cn_crt.log(cn_crt.cabs((cn_crt_complex){a[i * 4].real, a[i * 4].imag}));
    /* np.linalg.det is sign*exp(sum(log(abs(LU diagonal)))), abs being npy_cabs (the
     * CRT's _cabs on Win64); its zero test includes underflow, not merely a zero LU
     * pivot. */
    return cn_crt.exp(logs) == 0;
}

static void inverse(complex128 *matrix, const transfer_backend *b, int *error) {
    const lapack_int n = 3;
    lapack_int pivots[3], info;
    complex128 a[9], eye[9] = {{0, 0}};
    for (size_t r = 0; r < 3; ++r) {
        eye[r * 4].real = 1;
        for (size_t c = 0; c < 3; ++c) a[c * 3 + r] = matrix[r * 3 + c];
    }
    b->zgesv(&n, &n, a, &n, pivots, eye, &n, &info);
    for (size_t r = 0; r < 3; ++r)
        for (size_t c = 0; c < 3; ++c) matrix[r * 3 + c] = eye[c * 3 + r];
    if (info) *error = 4;
}

CN_EXPORT cn_status cn_transfer_parameters(const void *image, const void *reference,
        const unsigned char *mask, const unsigned char *reference_mask,
        size_t pixels, size_t reference_pixels, size_t pixel_stride, size_t channel_stride,
        size_t reference_pixel_stride, size_t reference_channel_stride, int f64, int principal,
        void *mean, void *reference_mean, complex128 *matrix,
        int *error, const transfer_backend *b) {
    pixel_layout layout = {pixel_stride, channel_stride};
    pixel_layout reference_layout = {reference_pixel_stride, reference_channel_stride};
    if (!image || !reference || !mask || !reference_mask || !mean || !reference_mean
        || !matrix || !error || !backend_valid(b) || !pixels || !reference_pixels
        || !layout_valid(layout) || !layout_valid(reference_layout)
        || (f64 != 0 && f64 != 1) || (principal != 0 && principal != 1)) return CN_INVALID_ARGUMENT;
    if (pixels > SIZE_MAX / 3 / sizeof(double) || reference_pixels > SIZE_MAX / 3 / sizeof(double)
        || pixels > INT64_MAX || reference_pixels > INT64_MAX) return CN_SIZE_OVERFLOW;
    *error = 0;
    double cov[9], reference_cov[9];
    int content_error = 0, reference_error = 0;
    cn_status status = covariance(image, layout, mask, pixels, f64, mean, cov, b, &content_error);
    if (status != CN_OK) return status;
    status = covariance(reference, reference_layout, reference_mask, reference_pixels, f64,
        reference_mean, reference_cov, b, &reference_error);
    if (status != CN_OK) return status;
    /* Both covariances are computed first; Linear checks the reference
     * eigensystem first, while Principal checks the content first. */
    *error = principal ? (content_error ? content_error : reference_error)
                       : (reference_error ? reference_error : content_error);
    if (*error) return CN_OK;
    complex128 values[3], vectors[9], reference_values[3], reference_vectors[9];
    status = eigen(cov, values, vectors, b, error);
    if (status != CN_OK || *error) return status;
    status = eigen(reference_cov, reference_values, reference_vectors, b, error);
    if (status != CN_OK || *error) return status;
    complex128 diagonal[9] = {{0, 0}}, temporary[9];
    if (principal) {
        for (size_t i = 0; i < 3; ++i) {
            /* np.where(values == 0, 1e-42, values): the weak 1e-42 is 1e-42+0j. */
            if (values[i].real == 0 && values[i].imag == 0) {
                values[i].real = 1e-42; values[i].imag = 0;
            }
            diagonal[i * 4] = clipped_root(divide_complex(reference_values[i], values[i]));
        }
        product(reference_vectors, diagonal, 0, temporary, b);
        product(temporary, vectors, 1, matrix, b);
    } else {
        complex128 content_root[9], reference_root[9];
        for (size_t i = 0; i < 3; ++i) diagonal[i * 4] = clipped_root(reference_values[i]);
        product(reference_vectors, diagonal, 0, temporary, b);
        product(temporary, reference_vectors, 1, reference_root, b);
        for (size_t i = 0; i < 3; ++i) diagonal[i * 4] = clipped_root(values[i]);
        product(vectors, diagonal, 0, temporary, b);
        product(temporary, vectors, 1, content_root, b);
        /* += identity / 255.0 adds 1/255+0j on the diagonal and 0+0j
         * elsewhere, which turns any -0 part into +0. */
        if (singular(content_root, b))
            for (size_t i = 0; i < 9; ++i) {
                content_root[i].real += i % 4 == 0 ? 1.0 / 255.0 : 0.0;
                content_root[i].imag += 0.0;
            }
        inverse(content_root, b, error);
        if (*error) return CN_OK;
        product(reference_root, content_root, 0, matrix, b);
    }
    return CN_OK;
}

CN_EXPORT cn_status cn_transfer_gather_f32(const float *source, const unsigned char *mask,
        size_t pixels, float *out, size_t *count) {
    if (!source || !mask || !out || !count) return CN_INVALID_ARGUMENT;
    if (pixels > SIZE_MAX / 3 / sizeof(float)) return CN_SIZE_OVERFLOW;
    size_t n = 0;
    for (size_t i = 0; i < pixels; ++i) if (mask[i]) {
        memcpy(out + n * 3, source + i * 3, 3 * sizeof(float)); ++n;
    }
    *count = n;
    return CN_OK;
}

CN_EXPORT cn_status cn_transfer_mask(const void *source, size_t pixels, size_t channels,
        size_t pixel_stride, size_t channel_stride, int f64, unsigned char *mask) {
    pixel_layout layout = {pixel_stride, channel_stride};
    if (!source || !mask || (channels != 3 && channels != 4) || (f64 != 0 && f64 != 1)
        || !layout_valid(layout)) return CN_INVALID_ARGUMENT;
    if (pixels > SIZE_MAX / channels / sizeof(double)) return CN_SIZE_OVERFLOW;
    for (size_t i = 0; i < pixels; ++i)
        mask[i] = (unsigned char)(channels == 3 || read_value(source, element(layout, i, 3), f64) > 0);
    return CN_OK;
}

/* np.dstack((rgb, alpha)): out keeps rgb's layout (the caller allocates it). */
CN_EXPORT cn_status cn_transfer_alpha(const void *rgb, size_t rgb_pixel_stride,
        size_t rgb_channel_stride, const void *original, size_t original_pixel_stride,
        size_t original_channel_stride, size_t pixels, int rgb_kind, int source_f64,
        void *out, size_t out_pixel_stride, size_t out_channel_stride) {
    pixel_layout from = {rgb_pixel_stride, rgb_channel_stride};
    pixel_layout source = {original_pixel_stride, original_channel_stride};
    pixel_layout to = {out_pixel_stride, out_channel_stride};
    if (!rgb || !original || !out || rgb_kind < 0 || rgb_kind > 2
        || (source_f64 != 0 && source_f64 != 1) || (rgb_kind == 0 && source_f64)
        || !layout_valid(from)
        || !layout_valid(source) || !layout_valid(to)) return CN_INVALID_ARGUMENT;
    if (pixels > SIZE_MAX / 4 / sizeof(complex128)) return CN_SIZE_OVERFLOW;
    for (size_t i = 0; i < pixels; ++i) {
        double alpha = read_value(original, element(source, i, 3), source_f64);
        if (rgb_kind == 2) {
            complex128 *result = out;
            for (size_t c = 0; c < 3; ++c)
                result[element(to, i, c)] = ((const complex128 *)rgb)[element(from, i, c)];
            result[element(to, i, 3)].real = alpha; result[element(to, i, 3)].imag = 0;
        } else if (rgb_kind == 1) {
            for (size_t c = 0; c < 3; ++c)
                ((double *)out)[element(to, i, c)] = ((const double *)rgb)[element(from, i, c)];
            ((double *)out)[element(to, i, 3)] = alpha;
        } else {
            /* A float32 result has a float32 original (np.result_type): np.dstack
             * copies its alpha's bits, a denormal one included under DAZ, where
             * the float64 round trip above would flush it. */
            for (size_t c = 0; c < 3; ++c)
                ((float *)out)[element(to, i, c)] = ((const float *)rgb)[element(from, i, c)];
            ((float *)out)[element(to, i, 3)] = ((const float *)original)[element(source, i, 3)];
        }
    }
    return CN_OK;
}

/* Linear's result is upstream's transfer.dot((content - mu).T).T, planar whatever
 * the image's layout; Principal's (content - mu).dot(transform.T) is interleaved.
 * The caller allocates out in that layout. */
CN_EXPORT cn_status cn_transfer_apply_complex(const void *image, size_t pixel_stride,
        size_t channel_stride, size_t pixels, int f64,
        const void *mean, const void *reference_mean, const complex128 *matrix,
        complex128 *out, size_t out_pixel_stride, size_t out_channel_stride,
        int principal, const transfer_backend *b) {
    pixel_layout layout = {pixel_stride, channel_stride};
    pixel_layout to = {out_pixel_stride, out_channel_stride};
    if (!image || !mean || !reference_mean || !matrix || !out || !backend_valid(b)
        || !pixels || (f64 != 0 && f64 != 1) || (principal != 0 && principal != 1)
        || !layout_valid(layout) || !layout_valid(to)) return CN_INVALID_ARGUMENT;
    if (pixels > SIZE_MAX / 3 / sizeof(complex128) || pixels > INT64_MAX) return CN_SIZE_OVERFLOW;
    /* The product lands in out directly when out has its layout: interleaved for
     * Principal, planar (3 x pixels) for Linear. */
    int in_place = principal ? to.pixel == 3 && to.channel == 1
                             : to.pixel == 1 && to.channel == pixels;
    complex128 *centered = malloc(pixels * 3 * sizeof(complex128));
    complex128 *temporary = in_place ? out : malloc(pixels * 3 * sizeof(complex128));
    if (!centered || !temporary) {
        free(centered);
        if (!in_place) free(temporary);
        return CN_ALLOCATION_FAILED;
    }
    for (size_t i = 0; i < pixels; ++i)
        for (size_t c = 0; c < 3; ++c) {
            double value;
            size_t index = element(layout, i, c);
            if (f64) value = ((const double *)image)[index] - ((const double *)mean)[c];
            else { float f = ((const float *)image)[index] - ((const float *)mean)[c]; value = f; }
            centered[i * 3 + c].real = value; centered[i * 3 + c].imag = 0;
        }
    /* The original BLAS orientation: (content - mu) @ transform.T for
     * Principal, transform @ (content - mu).T for Linear. */
    const complex128 one = {1, 0}, zero = {0, 0};
    if (principal)
        b->zgemm(101, 111, 112, (lapack_int)pixels, 3, 3, &one,
            centered, 3, matrix, 3, &zero, temporary, 3);
    else
        b->zgemm(101, 111, 112, 3, (lapack_int)pixels, 3, &one,
            matrix, 3, centered, 3, &zero, temporary, (lapack_int)pixels);
    for (size_t i = 0; i < pixels; ++i)
        for (size_t c = 0; c < 3; ++c) {
            complex128 value = temporary[principal ? i * 3 + c : c * pixels + i];
            /* + mu_reference adds mu+0j, which turns a -0 imaginary part into +0. */
            value.real += read_value(reference_mean, c, f64);
            value.imag += 0.0;
            /* clip(0, 1) is minimum(maximum(z, 0), 1) with lexicographic
             * comparisons; a NaN part keeps z (clip.cpp). */
            if (!isnan(value.real) && !isnan(value.imag)) {
                if (value.real < 0 || (value.real == 0 && value.imag <= 0)) value.real = value.imag = 0;
                if (value.real > 1 || (value.real == 1 && value.imag >= 0)) { value.real = 1; value.imag = 0; }
            }
            out[element(to, i, c)] = value;
        }
    free(centered); if (!in_place) free(temporary);
    return CN_OK;
}
