/* Save Image's encoding steps shared by the Python glue (image_io.cpp), the native Save Image entry point
 * (run/nodes/io.cpp) and the UI previews (image_preview.cpp): chaiNNer-C's default PNG and chaiNNer's ExtraSamples fix
 * of OpenCV's TIFFs (GPL-3.0). Nothing here touches Python. */
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace cn::image_encode {

enum class Png { declined, encoded, out_of_memory, failed };

/* The default PNG (cn_png_encode_u8, png_encode.c) of an 8-bit grey, grey + alpha, BGR or BGRA image (strides in
 * bytes, any sign), written to the buffer allocate(size) returns once the file's size is known (on the calling
 * thread; it does not throw, and returns nullptr when it cannot allocate): `declined` (allocate not called) beyond the
 * encoder's range (each side at most 2^31 - 1, rows of fewer than 2^29 samples), so another encoder runs;
 * `out_of_memory` when an allocation failed; `failed` for any other error, which none of the encoder's inputs here
 * gives. */
Png png(const uint8_t *source, ptrdiff_t height, ptrdiff_t width, ptrdiff_t channels, ptrdiff_t row_stride,
        ptrdiff_t pixel_stride, ptrdiff_t channel_stride, const std::function<uint8_t *(size_t)> &allocate);
/* The same into `encoded` (untouched unless `encoded`). */
Png png(const uint8_t *source, ptrdiff_t height, ptrdiff_t width, ptrdiff_t channels, ptrdiff_t row_stride,
        ptrdiff_t pixel_stride, ptrdiff_t channel_stride, std::vector<uint8_t> &encoded);

/* OpenCV writes 4-sample RGB TIFFs without ExtraSamples (tag 338), which TIFF 6.0 requires, so readers may take the
 * fourth sample for an unknown one (upstream chaiNNer #2950). Adds 338 = SHORT 2 (unassociated alpha: chaiNNer's
 * straight alpha) to a classic TIFF's first IFD: a copy of the IFD with the entry in tag order goes to the end of the
 * file, on a word boundary, and the header points to it; every other byte stays where it was. False (and `fixed`
 * untouched) for every other file, which stays as it is. */
bool tiff_extra_samples(std::span<const uint8_t> file, std::vector<uint8_t> &fixed);

} // namespace cn::image_encode
