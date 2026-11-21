/* LargeImageOutput's UI previews, natively (GPL-3.0): numpy_outputs.py get_broadcast_data's preview sizes and
 * preview_encode's steps (BOX resize with premultiplied alpha, to_uint8, base64) with the same kernels; a JPEG is
 * OpenCV's default encoder's (the Python path's URL), a PNG chaiNNer-C's default PNG (cn::image_encode::png: the
 * Python path's pixels in other bytes). Nothing here touches Python. */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cn::image_preview {

struct Preview {
    std::string url; // "data:image/png;base64,..." or "data:image/jpg;base64,..."
    uint32_t width = 0, height = 0;
};

/* The most pixels a full-size preview has (8192 x 8192; numpy_outputs.FULL_SIZE_PREVIEW_PIXELS): a 16K picture as
 * base64 inside one event is too large to send and decode. */
inline constexpr uint64_t full_size_pixels = uint64_t(8192) * 8192;

/* One node's previews travel in one event, which the window parses as one JavaScript string (at most 536,870,888
 * characters in V8). Its previews may take at most node_preview_chars characters together (90 % of that), counted as
 * their pixels times preview_centichars_per_pixel / 100: the base64 of incompressible RGBA as a PNG, the most a
 * preview takes (5.336 measured on RGBA noise with this encoder on 2026-11-20; stored deflate's 4 bytes a pixel and a
 * filter byte a row give 5.334; a JPEG tier takes fewer). Every image's ordinary previews (tier_pixels) count; the
 * full-size ones are granted in output order while they fit (Run::broadcast_now and numpy_outputs.full_size_previews,
 * with numpy_outputs's copies of these constants): ten large images share about 34 M pixels of them, while one image
 * of full_size_pixels keeps its own. */
inline constexpr uint64_t preview_centichars_per_pixel = 534;
inline constexpr uint64_t node_preview_chars = 483'000'000;

struct Shape {
    uint32_t height = 0, width = 0;
};

/* The shapes of a height x width image's previews other than the full-size one, in get_broadcast_data's order: from
 * the largest size that applies, each from the one before, resized when a side exceeds its size by more than the
 * grace. previews encodes exactly these. */
std::vector<Shape> tier_shapes(uint32_t height, uint32_t width);

/* The pixels of those previews together. */
uint64_t tier_pixels(uint32_t height, uint32_t width);

/* The pixels the full-size preview of a height x width image holds (previews' full_size): its own when it is larger
 * than the largest preview size allows and has at most full_size_pixels, else 0 (it gets none). */
uint64_t full_size_cost(uint32_t height, uint32_t width);

/* The previews of a normalized float32 image (height x width x channels, C-contiguous, channels 1 to 4) in
 * get_broadcast_data's order. full_size (LargeImageOutput.full_size): an image larger than the largest size allows
 * and of at most full_size_pixels gets one more preview in front, at its own size and lossless; the others then
 * follow as without it, all of them lossy (a JPEG) unless 4 channels. sizes (a seek request's previewSizes, the
 * widths of the views that show the image): when not empty, only the previews those views show are made, each the
 * bytes it has without sizes: for each width the smallest tier at least that wide (the largest when none is), and the
 * full-size one only for a width past the largest tier's. Each tier is still resized from the one before (a box
 * resize straight from the image gives other pixels). False when a step declines (a kernel's error, a floating-point
 * event that NumPy would report, an encoder's fallback): the caller then runs the Python path, which reports it. The
 * PNG encoder's failures throw (std::bad_alloc when memory runs out, else std::runtime_error). */
bool previews(const float *data, uint32_t height, uint32_t width, uint32_t channels, bool full_size,
              std::vector<Preview> &out, const std::vector<uint32_t> &sizes = {});

} // namespace cn::image_preview
