#ifndef CHAINNER_FFT_NATIVE_H
#define CHAINNER_FFT_NATIVE_H
#include "chainner.h"

typedef struct cn_fft_plan_opaque *cn_fft_plan;
cn_fft_plan cn_fft_make_plan(size_t length);
void cn_fft_destroy_plan(cn_fft_plan plan);
int cn_fft_execute(cn_fft_plan plan, double *data, int inverse, double scale);
/* In-place row-major complex128 transform; plans are immutable and caller-owned. */
cn_status cn_fft2_with_plans(double *data, size_t height, size_t width,
                            int inverse, cn_fft_plan row, cn_fft_plan column);
#endif
