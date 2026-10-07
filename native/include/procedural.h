#ifndef CHAINNER_PROCEDURAL_H
#define CHAINNER_PROCEDURAL_H
#include "chainner.h"

typedef struct {
    const double *points, *gradients;
    const float *values;
    const int32_t *table;
    double *out;
    size_t table_size, gradient_count, value_count;
    int dimensions, smooth, simplex;
    double f, g, r2, scale;
} procedural_job;

/* The caller validates bounded coordinates and owns the immutable tables. */
double cn_procedural_sample(const procedural_job *job, const double *point);
#endif
