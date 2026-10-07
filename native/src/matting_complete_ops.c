/* Closed-form alpha and multilevel foreground matting adapted from
 * PyMatting 1.1.16 (The PyMatting Developers, MIT): cf_laplacian.py,
 * estimate_alpha_cf.py, ichol.py, cg.py and estimate_foreground_ml.py.
 * The foreground starts from 1.1.16's mean foreground and background colours.
 * Public NumPy OpenBLAS ddot preserves the original CG reduction order.
 */
#include "chainner.h"
#include "cn_crt_math.h"
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef double (*matting_dot_fn)(int64_t, const double *, int64_t, const double *, int64_t);
/* Existing exact ellipse span generation and finite morphology implementation. */
cn_status cn_morphology_complete(const float *, float *, size_t, size_t, size_t,
    size_t, size_t, int, int, size_t);
typedef struct { double *values; int64_t *rows; size_t *starts; size_t n, count, capacity; } sparse_matrix;

static void sparse_free(sparse_matrix *matrix) {
    free(matrix->values); free(matrix->rows); free(matrix->starts);
    memset(matrix, 0, sizeof(*matrix));
}

static cn_status reserve_sparse(sparse_matrix *matrix, size_t required, size_t maximum) {
    if (required > maximum) return CN_SIZE_OVERFLOW;
    if (required <= matrix->capacity) return CN_OK;
    size_t capacity = matrix->capacity ? matrix->capacity : 256;
    while (capacity < required) capacity = capacity > maximum / 2 ? maximum : capacity * 2;
    double *values = realloc(matrix->values, capacity * sizeof(double));
    if (!values) return CN_ALLOCATION_FAILED;
    matrix->values = values;
    int64_t *rows = realloc(matrix->rows, capacity * sizeof(int64_t));
    if (!rows) return CN_ALLOCATION_FAILED;
    matrix->rows = rows; matrix->capacity = capacity;
    return CN_OK;
}

static cn_status laplacian(const double *image, const double *trimap, size_t height,
        size_t width, double *values, int64_t *map, size_t *unknown, int *error) {
    size_t pixels = height * width, foreground = 0, background = 0, count = 0;
    for (size_t i = 0; i < pixels; ++i) {
        int fg = trimap[i] >= 0.9, bg = trimap[i] <= 0.1;
        foreground += fg; background += bg;
        map[i] = fg || bg ? -1 : (int64_t)count++;
    }
    *unknown = count;
    if (!background) { *error = 1; return CN_OK; }
    if (!foreground) { *error = 2; return CN_OK; }
    memset(values, 0, pixels * 25 * sizeof(double));
    for (size_t y = 1; y + 1 < height; ++y) {
        for (size_t x = 1; x + 1 < width; ++x) {
            int needed = 0;
            for (size_t dy = 0; dy < 3; ++dy)
                for (size_t dx = 0; dx < 3; ++dx)
                    if (map[(y + dy - 1) * width + x + dx - 1] >= 0) needed = 1;
            if (!needed) continue;
            double color[9][3];
            for (size_t channel = 0; channel < 3; ++channel) {
                double sum = 0;
                for (size_t dy = 0; dy < 3; ++dy)
                    for (size_t dx = 0; dx < 3; ++dx)
                        sum += image[((y + dy - 1) * width + x + dx - 1) * 3 + channel];
                for (size_t dy = 0; dy < 3; ++dy)
                    for (size_t dx = 0; dx < 3; ++dx)
                        color[dy * 3 + dx][channel] = image[((y + dy - 1) * width + x + dx - 1) * 3 + channel] - sum / 9;
            }
            double a00 = 1e-7, a01 = 0, a02 = 0, a11 = 1e-7, a12 = 0, a22 = 1e-7;
            for (size_t k = 0; k < 9; ++k) {
                a00 += color[k][0] * color[k][0]; a01 += color[k][0] * color[k][1];
                a02 += color[k][0] * color[k][2]; a11 += color[k][1] * color[k][1];
                a12 += color[k][1] * color[k][2]; a22 += color[k][2] * color[k][2];
            }
            a00 /= 9; a01 /= 9; a02 /= 9; a11 /= 9; a12 /= 9; a22 /= 9;
            double determinant = a00 * a12 * a12 + a01 * a01 * a22 + a02 * a02 * a11
                - a00 * a11 * a22 - 2 * a01 * a02 * a12;
            if (determinant == 0) { *error = 6; return CN_OK; }
            double inverse = 1.0 / determinant;
            double m00 = (a12 * a12 - a11 * a22) * inverse;
            double m01 = (a01 * a22 - a02 * a12) * inverse;
            double m02 = (a02 * a11 - a01 * a12) * inverse;
            double m11 = (a02 * a02 - a00 * a22) * inverse;
            double m12 = (a00 * a12 - a01 * a02) * inverse;
            double m22 = (a01 * a01 - a00 * a11) * inverse;
            for (size_t yi = 0; yi < 3; ++yi) for (size_t xi = 0; xi < 3; ++xi) {
                size_t i = (y + yi - 1) * width + x + xi - 1;
                double s = color[yi * 3 + xi][0], t = color[yi * 3 + xi][1], u = color[yi * 3 + xi][2];
                double c0 = m00 * s + m01 * t + m02 * u;
                double c1 = m01 * s + m11 * t + m12 * u;
                double c2 = m02 * s + m12 * t + m22 * u;
                for (size_t yj = 0; yj < 3; ++yj) for (size_t xj = 0; xj < 3; ++xj) {
                    size_t j = (y + yj - 1) * width + x + xj - 1;
                    double temp = c0 * color[yj * 3 + xj][0] + c1 * color[yj * 3 + xj][1] + c2 * color[yj * 3 + xj][2];
                    double value = (i == j ? 1.0 : 0.0) - (1 + temp) / 9;
                    size_t slot = (yj + 2 - yi) * 5 + xj + 2 - xi;
                    values[i * 25 + slot] += value;
                }
            }
        }
    }
    return CN_OK;
}

static cn_status build_system(const double *trimap, size_t height, size_t width,
        const double *stencil, const int64_t *map, sparse_matrix *matrix, double *rhs) {
    matrix->starts = calloc(matrix->n + 1, sizeof(size_t));
    if (!matrix->starts) return CN_ALLOCATION_FAILED;
    for (size_t y = 0; y < height; ++y) for (size_t x = 0; x < width; ++x) {
        size_t p = y * width + x;
        if (map[p] < 0) continue;
        int64_t columns[25]; double values[25]; size_t count = 0;
        double sum = 0;
        for (int dy = -2; dy <= 2; ++dy) for (int dx = -2; dx <= 2; ++dx) {
            int64_t yy = (int64_t)y + dy, xx = (int64_t)x + dx;
            size_t j = 0;
            if (yy >= 0 && xx >= 0 && (uint64_t)yy < height && (uint64_t)xx < width)
                j = (size_t)yy * width + (size_t)xx;
            double value = stencil[p * 25 + (size_t)(dy + 2) * 5 + (size_t)(dx + 2)];
            if (map[j] >= 0) {
                size_t at = 0;
                while (at < count && columns[at] < map[j]) ++at;
                if (at < count && columns[at] == map[j]) values[at] += value;
                else {
                    for (size_t k = count; k > at; --k) { columns[k] = columns[k - 1]; values[k] = values[k - 1]; }
                    columns[at] = map[j]; values[at] = value; ++count;
                }
            } else sum += value * (trimap[j] >= 0.9 ? 1.0 : 0.0);
        }
        rhs[(size_t)map[p]] = -sum;
        cn_status status = reserve_sparse(matrix, matrix->count + count, SIZE_MAX / sizeof(double));
        if (status != CN_OK) return status;
        memcpy(matrix->rows + matrix->count, columns, count * sizeof(int64_t));
        memcpy(matrix->values + matrix->count, values, count * sizeof(double));
        matrix->count += count;
        matrix->starts[(size_t)map[p] + 1] = matrix->count;
    }
    return CN_OK;
}

static int compare_i64(const void *a, const void *b) {
    int64_t x = *(const int64_t *)a, y = *(const int64_t *)b;
    return (x > y) - (x < y);
}

static cn_status ichol(const sparse_matrix *a_matrix, sparse_matrix *factor, double shift, int *failed) {
    size_t n = a_matrix->n;
    size_t *s = calloc(n ? n : 1, sizeof(size_t)), *t = calloc(n ? n : 1, sizeof(size_t));
    int64_t *links = malloc((n ? n : 1) * sizeof(int64_t)), *columns = malloc((n ? n : 1) * sizeof(int64_t));
    double *a = calloc(n ? n : 1, sizeof(double)), *r = calloc(n ? n : 1, sizeof(double)), *diagonal = malloc((n ? n : 1) * sizeof(double));
    unsigned char *marked = calloc(n ? n : 1, 1);
    cn_status status = CN_OK;
    if (!s || !t || !links || !columns || !a || !r || !diagonal || !marked) { status = CN_ALLOCATION_FAILED; goto cleanup; }
    factor->count = 0; memset(factor->starts, 0, (n + 1) * sizeof(size_t)); *failed = 0;
    for (size_t j = 0; j < n; ++j) {
        links[j] = -1; diagonal[j] = shift;
        for (size_t index = a_matrix->starts[j]; index < a_matrix->starts[j + 1]; ++index) {
            size_t i = (size_t)a_matrix->rows[index];
            if (i == j) { diagonal[j] += a_matrix->values[index]; t[j] = index + 1; }
            if (i >= j) r[j] += fabs(a_matrix->values[index]);
        }
    }
    for (size_t j = 0; j < n; ++j) {
        size_t count = 0;
        for (size_t index = t[j]; index < a_matrix->starts[j + 1]; ++index) {
            size_t i = (size_t)a_matrix->rows[index]; double value = a_matrix->values[index];
            if (value != 0 && i > j) {
                a[i] += value;
                if (!marked[i]) { marked[i] = 1; columns[count++] = (int64_t)i; }
            }
        }
        int64_t k = links[j];
        while (k != -1) {
            size_t start = s[(size_t)k], end = factor->starts[(size_t)k + 1];
            int64_t next = links[(size_t)k]; double ljk = factor->values[start++];
            if (start < end) {
                s[(size_t)k] = start; size_t i = (size_t)factor->rows[start];
                links[(size_t)k] = links[i]; links[i] = k;
                for (size_t index = start; index < end; ++index) {
                    i = (size_t)factor->rows[index]; a[i] -= factor->values[index] * ljk;
                    if (!marked[i]) { marked[i] = 1; columns[count++] = (int64_t)i; }
                }
            }
            k = next;
        }
        if (diagonal[j] <= 0) { *failed = 1; goto cleanup; }
        status = reserve_sparse(factor, factor->count + 1 + count, 250000000);
        if (status != CN_OK) goto cleanup;
        diagonal[j] = sqrt(diagonal[j]);
        factor->values[factor->count] = diagonal[j]; factor->rows[factor->count++] = (int64_t)j;
        s[j] = factor->count;
        qsort(columns, count, sizeof(int64_t), compare_i64);
        for (size_t index = 0; index < count; ++index) {
            size_t i = (size_t)columns[index]; double value = a[i] / diagonal[j];
            diagonal[i] -= value * value;
            double relative = 0.0 * r[j];
            if (fabs(value) > 1e-4 && fabs(a[i]) > relative) {
                factor->values[factor->count] = value; factor->rows[factor->count++] = (int64_t)i;
            }
            a[i] = 0; marked[i] = 0;
        }
        factor->starts[j + 1] = factor->count;
        if (factor->starts[j] + 1 < factor->starts[j + 1]) {
            size_t i = (size_t)factor->rows[factor->starts[j] + 1];
            links[j] = links[i]; links[i] = (int64_t)j;
        }
    }
cleanup:
    free(s); free(t); free(links); free(columns); free(a); free(r); free(diagonal); free(marked);
    return status;
}

static void precondition(const sparse_matrix *factor, const double *b, double *x) {
    size_t n = factor->n;
    memcpy(x, b, n * sizeof(double));
    for (size_t j = 0; j < n; ++j) {
        double temp = x[j] / factor->values[factor->starts[j]];
        x[j] = temp;
        for (size_t k = factor->starts[j] + 1; k < factor->starts[j + 1]; ++k)
            x[(size_t)factor->rows[k]] -= factor->values[k] * temp;
    }
    for (size_t ii = n; ii > 0;) {
        size_t i = --ii; double sum = x[i];
        for (size_t k = factor->starts[i] + 1; k < factor->starts[i + 1]; ++k)
            sum -= factor->values[k] * x[(size_t)factor->rows[k]];
        x[i] = sum / factor->values[factor->starts[i]];
    }
}

static void sparse_multiply(const sparse_matrix *a, const double *x, double *out) {
    for (size_t i = 0; i < a->n; ++i) {
        double sum = 0;
        for (size_t k = a->starts[i]; k < a->starts[i + 1]; ++k)
            sum += a->values[k] * x[(size_t)a->rows[k]];
        out[i] = sum;
    }
}

static cn_status conjugate_gradient(const sparse_matrix *a, const sparse_matrix *factor,
        const double *b, double *x, matting_dot_fn dot, int *error) {
    size_t n = a->n;
    if (!n) { *error = 5; return CN_OK; }
    double *storage = malloc(n * 4 * sizeof(double));
    if (!storage) return CN_ALLOCATION_FAILED;
    double *r = storage, *z = storage + n, *p = storage + n * 2, *ap = storage + n * 3;
    memset(x, 0, n * sizeof(double));
    double norm_b = sqrt(dot((int64_t)n, b, 1, b, 1));
    sparse_multiply(a, x, ap);
    for (size_t i = 0; i < n; ++i) r[i] = b[i] - ap[i];
    double norm_r = sqrt(dot((int64_t)n, r, 1, r, 1));
    if (norm_r < 1e-7 * norm_b) { free(storage); return CN_OK; }
    /* The original zero-RHS path divides 0/0 and reaches its iteration error. */
    if (norm_b == 0 && norm_r == 0) { *error = 5; free(storage); return CN_OK; }
    precondition(factor, r, z); memcpy(p, z, n * sizeof(double));
    double rz = dot((int64_t)n, r, 1, z, 1);
    for (size_t iteration = 0; iteration < 10000; ++iteration) {
        sparse_multiply(a, p, ap);
        double alpha = rz / dot((int64_t)n, p, 1, ap, 1);
        for (size_t i = 0; i < n; ++i) { x[i] += alpha * p[i]; r[i] -= alpha * ap[i]; }
        norm_r = sqrt(dot((int64_t)n, r, 1, r, 1));
        if (norm_r < 1e-7 * norm_b) { free(storage); return CN_OK; }
        precondition(factor, r, z);
        double beta = 1.0 / rz;
        rz = dot((int64_t)n, r, 1, z, 1); beta *= rz;
        for (size_t i = 0; i < n; ++i) { p[i] *= beta; p[i] += z[i]; }
    }
    *error = 5; free(storage); return CN_OK;
}

CN_EXPORT cn_status cn_matting_cf_stencil(const double *image, const double *trimap,
        size_t height, size_t width, double *values, int64_t *map, size_t *unknown, int *error) {
    if (!image || !trimap || !values || !map || !unknown || !error || !height || !width)
        return CN_INVALID_ARGUMENT;
    if (height > SIZE_MAX / width || height * width > SIZE_MAX / 25 / sizeof(double)
        || height > INT64_MAX || width > INT64_MAX) return CN_SIZE_OVERFLOW;
    *error = 0;
    return laplacian(image, trimap, height, width, values, map, unknown, error);
}

CN_EXPORT cn_status cn_matting_cf_alpha(const double *image, const double *trimap,
        size_t height, size_t width, double *out, matting_dot_fn dot, int *error, int *retries) {
    if (!image || !trimap || !out || !dot || !error || !retries || !height || !width) return CN_INVALID_ARGUMENT;
    if (height > SIZE_MAX / width || height * width > SIZE_MAX / 25 / sizeof(double)
        || height > INT64_MAX || width > INT64_MAX) return CN_SIZE_OVERFLOW;
    size_t pixels = height * width, unknown = 0;
    *error = 0; *retries = 0;
    double *values = malloc(pixels * 25 * sizeof(double));
    int64_t *map = malloc(pixels * sizeof(int64_t));
    double *rhs = NULL, *solution = NULL;
    sparse_matrix a = {0}, factor = {0};
    cn_status status = CN_OK;
    if (!values || !map) { status = CN_ALLOCATION_FAILED; goto cleanup; }
    status = laplacian(image, trimap, height, width, values, map, &unknown, error);
    if (status != CN_OK || *error) goto cleanup;
    a.n = factor.n = unknown;
    rhs = malloc((unknown ? unknown : 1) * sizeof(double));
    solution = malloc((unknown ? unknown : 1) * sizeof(double));
    factor.starts = calloc(unknown + 1, sizeof(size_t));
    if (!rhs || !solution || !factor.starts) { status = CN_ALLOCATION_FAILED; goto cleanup; }
    status = build_system(trimap, height, width, values, map, &a, rhs);
    if (status != CN_OK) goto cleanup;
    free(values); values = NULL;
    const double shifts[] = {0, 1e-4, 1e-3, 1e-2, .1, .5, 1, 10, 100, 1e3, 1e4, 1e5};
    int failed = 1;
    for (size_t attempt = 0; attempt < sizeof(shifts) / sizeof(shifts[0]); ++attempt) {
        status = ichol(&a, &factor, shifts[attempt], &failed);
        if (status != CN_OK) { if (status == CN_SIZE_OVERFLOW) { *error = 4; status = CN_OK; } goto cleanup; }
        if (!failed) break;
        ++*retries;
    }
    if (failed) { *error = 3; goto cleanup; }
    status = conjugate_gradient(&a, &factor, rhs, solution, dot, error);
    if (status != CN_OK || *error) goto cleanup;
    for (size_t i = 0; i < pixels; ++i) {
        double value = map[i] >= 0 ? solution[(size_t)map[i]] : trimap[i];
        out[i] = value <= 0 ? 0 : value >= 1 ? 1 : value;
    }
cleanup:
    sparse_free(&a); sparse_free(&factor); free(values); free(map); free(rhs); free(solution);
    return status;
}

static size_t round_even_positive(double value) {
    double lower = floor(value), fraction = value - lower;
    size_t integer = (size_t)lower;
    return integer + (fraction > .5 || (fraction == .5 && (integer & 1)));
}

CN_EXPORT cn_status cn_matting_foreground(const double *input, const double *input_alpha,
        size_t height, size_t width, float *foreground, float *background) {
    if (!input || !input_alpha || !foreground || !height || !width || (height == 1 && width == 1))
        return CN_INVALID_ARGUMENT;
    if (height > SIZE_MAX / width || height * width > SIZE_MAX / 16 / sizeof(double))
        return CN_SIZE_OVERFLOW;
    size_t levels = (size_t)ceil(cn_crt.log2((double)(height > width ? height : width)));
    float *previous_f = calloc(3, sizeof(float)), *previous_b = calloc(3, sizeof(float));
    if (!previous_f || !previous_b) { free(previous_f); free(previous_b); return CN_ALLOCATION_FAILED; }
    /* PyMatting 1.1.16's 1x1 start: the mean colours of the alpha > 0.9 and the
     * alpha < 0.1 pixels. Numba sums the float32 values in y, x order and compares
     * the float32 alpha as a double. `mean /= count + 1e-5` divides in float64 and
     * rounds to float32, and np.zeros(...) + mean is a float32 addition to +0. */
    float sum_f[3] = {0}, sum_b[3] = {0};
    size_t count_f = 0, count_b = 0;
    for (size_t p = 0; p < height * width; ++p) {
        double a0 = (double)(float)input_alpha[p];
        if (a0 > 0.9) {
            for (size_t c = 0; c < 3; ++c) sum_f[c] += (float)input[p * 3 + c];
            ++count_f;
        }
        if (a0 < 0.1) {
            for (size_t c = 0; c < 3; ++c) sum_b[c] += (float)input[p * 3 + c];
            ++count_b;
        }
    }
    for (size_t c = 0; c < 3; ++c) {
        previous_f[c] = 0.0f + (float)((double)sum_f[c] / ((double)count_f + 1e-5));
        previous_b[c] = 0.0f + (float)((double)sum_b[c] / ((double)count_b + 1e-5));
    }
    size_t previous_width = 1, previous_height = 1;
    const float regularization = 1e-5f, gradient_weight = 1.0f;
    for (size_t level = 0; level <= levels; ++level) {
        double fraction = (double)level / (double)levels;
        size_t w = round_even_positive(cn_crt.pow((double)width, fraction));
        size_t h = round_even_positive(cn_crt.pow((double)height, fraction));
        if (!h || !w || h > height || w > width || h > SIZE_MAX / w || h * w > SIZE_MAX / 3 / sizeof(float)) {
            free(previous_f); free(previous_b); return CN_SIZE_OVERFLOW;
        }
        size_t pixels = h * w;
        float *image = malloc(pixels * 3 * sizeof(float)), *alpha = malloc(pixels * sizeof(float));
        float *f = malloc(pixels * 3 * sizeof(float)), *b = malloc(pixels * 3 * sizeof(float));
        if (!image || !alpha || !f || !b) {
            free(image); free(alpha); free(f); free(b); free(previous_f); free(previous_b);
            return CN_ALLOCATION_FAILED;
        }
        for (size_t y = 0; y < h; ++y) for (size_t x = 0; x < w; ++x) {
            size_t source = (y * height / h) * width + x * width / w;
            size_t prior = (y * previous_height / h) * previous_width + x * previous_width / w;
            size_t p = y * w + x;
            alpha[p] = (float)input_alpha[source];
            for (size_t c = 0; c < 3; ++c) {
                image[p * 3 + c] = (float)input[source * 3 + c];
                f[p * 3 + c] = previous_f[prior * 3 + c];
                b[p * 3 + c] = previous_b[prior * 3 + c];
            }
        }
        size_t iterations = w <= 32 && h <= 32 ? 10 : 2;
        for (size_t iteration = 0; iteration < iterations; ++iteration) {
            /* Gauss-Seidel traversal is deliberately ordered: later pixels
             * observe this iteration's earlier foreground/background writes. */
            for (size_t y = 0; y < h; ++y) for (size_t x = 0; x < w; ++x) {
                size_t p = y * w + x;
                float a0 = alpha[p], a00 = a0 * a0;
                double a1 = 1.0 - (double)a0, a01 = (double)a0 * a1, a11 = a1 * a1;
                float rhs_f[3], rhs_b[3];
                for (size_t c = 0; c < 3; ++c) {
                    rhs_f[c] = a0 * image[p * 3 + c];
                    rhs_b[c] = (float)(a1 * (double)image[p * 3 + c]);
                }
                size_t neighbors[4] = {
                    y * w + (x ? x - 1 : 0), y * w + (x + 1 < w ? x + 1 : w - 1),
                    (y ? y - 1 : 0) * w + x, (y + 1 < h ? y + 1 : h - 1) * w + x
                };
                for (size_t direction = 0; direction < 4; ++direction) {
                    size_t q = neighbors[direction];
                    float difference = a0 - alpha[q];
                    float gradient = fabsf(difference);
                    float product = gradient_weight * gradient;
                    float da = regularization + product;
                    a00 += da; a11 += (double)da;
                    for (size_t c = 0; c < 3; ++c) {
                        float left = da * f[q * 3 + c], right = da * b[q * 3 + c];
                        rhs_f[c] += left; rhs_b[c] += right;
                    }
                }
                double determinant = (double)a00 * a11 - a01 * a01;
                double inv_det = 1.0 / determinant;
                double b00 = inv_det * a11, b01 = inv_det * -a01, b11 = inv_det * (double)a00;
                for (size_t c = 0; c < 3; ++c) {
                    double value_f = b00 * rhs_f[c] + b01 * rhs_b[c];
                    double value_b = b01 * rhs_f[c] + b11 * rhs_b[c];
                    /* Numba's min(1,NaN) keeps its first argument, as do
                     * these ordered comparisons; then max(0, ...) follows. */
                    value_f = value_f < 1 ? value_f : 1;
                    value_b = value_b < 1 ? value_b : 1;
                    value_f = value_f > 0 ? value_f : 0;
                    value_b = value_b > 0 ? value_b : 0;
                    f[p * 3 + c] = (float)value_f;
                    b[p * 3 + c] = (float)value_b;
                }
            }
        }
        free(image); free(alpha); free(previous_f); free(previous_b);
        previous_f = f; previous_b = b; previous_width = w; previous_height = h;
    }
    memcpy(foreground, previous_f, height * width * 3 * sizeof(float));
    if (background) memcpy(background, previous_b, height * width * 3 * sizeof(float));
    free(previous_f); free(previous_b);
    return CN_OK;
}

CN_EXPORT cn_status cn_matting_output(const float *foreground, const double *alpha,
        size_t pixels, int float64_output, void *out) {
    if (!foreground || !alpha || !out || (float64_output != 0 && float64_output != 1)) return CN_INVALID_ARGUMENT;
    if (pixels > SIZE_MAX / 4 / sizeof(double)) return CN_SIZE_OVERFLOW;
    for (size_t i = 0; i < pixels; ++i) {
        if (float64_output) {
            double *destination = out;
            destination[i * 4] = foreground[i * 3 + 2]; destination[i * 4 + 1] = foreground[i * 3 + 1];
            destination[i * 4 + 2] = foreground[i * 3]; destination[i * 4 + 3] = alpha[i];
        } else {
            float *destination = out;
            destination[i * 4] = foreground[i * 3 + 2]; destination[i * 4 + 1] = foreground[i * 3 + 1];
            destination[i * 4 + 2] = foreground[i * 3]; destination[i * 4 + 3] = (float)alpha[i];
        }
    }
    return CN_OK;
}

CN_EXPORT cn_status cn_matting_erode_mask_u8(const uint8_t *source, uint8_t *out,
        size_t height, size_t width, size_t radius) {
    if (!source || !out || !height || !width || radius > 46340) return CN_INVALID_ARGUMENT;
    if (height > SIZE_MAX / width || height * width > SIZE_MAX / sizeof(size_t)
        || height > INT_MAX || width > INT_MAX) return CN_SIZE_OVERFLOW;
    size_t pixels = height * width;
    if (!radius) { memmove(out, source, pixels); return CN_OK; }
    float *input = malloc(pixels * sizeof(float)), *eroded = malloc(pixels * sizeof(float));
    if (!input || !eroded) { free(input); free(eroded); return CN_ALLOCATION_FAILED; }
    for (size_t i = 0; i < pixels; ++i) input[i] = source[i];
    /* Integer byte masks have identical minima at every floating SIMD lane
     * width, and FLT_MAX is equivalent to uint8's neutral erosion border. */
    cn_status status = cn_morphology_complete(input, eroded, height, width, 1, radius, 1, 2, 0, 4);
    if (status == CN_OK) for (size_t i = 0; i < pixels; ++i) out[i] = (uint8_t)eroded[i];
    free(input); free(eroded);
    return status;
}

/* Row-major pixel grids: pixel i's channel c at i * pixel + c * channel elements,
 * interleaved (channels, 1) or planar (1, plane): upstream's np.dstack((img,
 * trimap)) keeps a planar image's layout (D-16). The trimap is row-major. */
static inline void trimap_pixel(const float *image, const double *trimap, float *out,
        size_t i, size_t pixel, size_t channel, size_t out_pixel, size_t out_channel) {
    for (size_t c = 0; c < 3; ++c)
        out[i * out_pixel + c * out_channel] = image[i * pixel + c * channel];
    out[i * out_pixel + 3 * out_channel] = (float)trimap[i];
}

CN_EXPORT cn_status cn_matting_trimap_output(const float *image, size_t pixel_stride,
        size_t channel_stride, const double *trimap, size_t pixels, float *out,
        size_t out_pixel_stride, size_t out_channel_stride) {
    if (!image || !trimap || !out || !pixel_stride || !channel_stride
        || !out_pixel_stride || !out_channel_stride) return CN_INVALID_ARGUMENT;
    if (pixels > SIZE_MAX / 4 / sizeof(double)) return CN_SIZE_OVERFLOW;
    if (pixel_stride == 3 && channel_stride == 1 && out_pixel_stride == 4 && out_channel_stride == 1)
        for (size_t i = 0; i < pixels; ++i) trimap_pixel(image, trimap, out, i, 3, 1, 4, 1);
    else
        for (size_t i = 0; i < pixels; ++i)
            trimap_pixel(image, trimap, out, i, pixel_stride, channel_stride,
                out_pixel_stride, out_channel_stride);
    return CN_OK;
}
