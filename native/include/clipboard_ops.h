#ifndef CHAINNER_CLIPBOARD_OPS_H
#define CHAINNER_CLIPBOARD_OPS_H
#include <stddef.h>
#include <stdint.h>
#if defined(_WIN32) && defined(CN_TENSOR_IMPORTS)
#define CN_CLIPBOARD_API __declspec(dllimport)
#elif defined(_WIN32)
#define CN_CLIPBOARD_API __declspec(dllexport)
#else
#define CN_CLIPBOARD_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif
/* Float32 half-up quantization to bottom-up, straight-alpha BGRA.
 * Strides are bytes and may be negative or unaligned. Output is packed.
 * Status values follow the existing cn_status ABI. */
CN_CLIPBOARD_API int cn_clipboard_pixels_f32(const void *source,
    size_t height, size_t width, size_t channels,
    ptrdiff_t row_stride, ptrdiff_t column_stride, ptrdiff_t channel_stride,
    int rgb, uint8_t *output, size_t output_bytes);
#ifdef __cplusplus
}
#endif
#endif
