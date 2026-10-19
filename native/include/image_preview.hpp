/* LargeImageOutput's UI previews, natively (GPL-3.0): numpy_outputs.py get_broadcast_data's preview sizes and
 * preview_encode's steps (BOX resize with premultiplied alpha, to_uint8, OpenCV's default PNG or JPEG encoder,
 * base64) with the same kernels and encoders, so every URL is today's. Nothing here touches Python. */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cn::image_preview {

struct Preview {
    std::string url; // "data:image/png;base64,..." or "data:image/jpg;base64,..."
    uint32_t width = 0, height = 0;
};

/* The previews of a normalized float32 image (height x width x channels, C-contiguous, channels 1 to 4) in
 * get_broadcast_data's order. False when a step declines (a kernel's error, a floating-point event that NumPy would
 * report, an encoder's fallback): the caller then runs today's Python path, which reports it as today. */
bool previews(const float *data, uint32_t height, uint32_t width, uint32_t channels, std::vector<Preview> &out);

} // namespace cn::image_preview
