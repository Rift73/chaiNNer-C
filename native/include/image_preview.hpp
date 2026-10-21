/* LargeImageOutput's UI previews, natively (GPL-3.0): numpy_outputs.py get_broadcast_data's preview sizes and
 * preview_encode's steps (BOX resize with premultiplied alpha, to_uint8, base64) with the same kernels; a JPEG is
 * OpenCV's default encoder's (today's URL), a PNG chaiNNer-C's default PNG (cn::image_encode::png: today's pixels in
 * other bytes). Nothing here touches Python. */
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
 * report, an encoder's fallback): the caller then runs today's Python path, which reports it as today. The PNG
 * encoder's failures throw (std::bad_alloc when memory runs out, else std::runtime_error). */
bool previews(const float *data, uint32_t height, uint32_t width, uint32_t channels, std::vector<Preview> &out);

} // namespace cn::image_preview
