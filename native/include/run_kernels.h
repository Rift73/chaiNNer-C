/* The chainner_native.dll entries that the run engine and the native node entry points (native/src/run) call,
 * grouped by the kernel file that defines and documents them; that file includes this header, so a declaration
 * cannot drift from its definition. A node lane adds its kernels under their file's heading (kept in file order). */
#ifndef CHAINNER_RUN_KERNELS_H
#define CHAINNER_RUN_KERNELS_H
#include "chainner.h"
#include <stddef.h>

#if defined(_WIN32) && defined(CN_TENSOR_IMPORTS)
#define CN_RUN_KERNEL __declspec(dllimport)
#else
#define CN_RUN_KERNEL CN_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif
/* buffers.c */
CN_RUN_KERNEL cn_status cn_pixels_convert_checked(const void *src, void *out, size_t count,
    int type, int output, int normalize, int *events);
CN_RUN_KERNEL cn_status cn_pixels_normalized_f32(const float *src, size_t count, int *same);
/* channels.c */
CN_RUN_KERNEL cn_status cn_channels_flip(const float *src, float *out,
    size_t height, size_t width, size_t channels, int axis);
#ifdef __cplusplus
}
#endif

#endif
