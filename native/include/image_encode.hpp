/* Save Image's encoding steps shared by the Python glue (image_io.cpp) and the native Save Image entry point
 * (run/nodes/io.cpp): fpng 1.0.6's PNG and chaiNNer's ExtraSamples fix of OpenCV's TIFFs (GPL-3.0). Nothing here
 * touches Python. */
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace cn::image_encode {

enum class Fpng { declined, encoded, failed };

/* fpng's PNG of an 8-bit BGR or BGRA image (strides in bytes, any sign): `declined` beyond the limits this fpng
 * version handles (each side at most 65535, the filtered stream below INT32_MAX - 1 MiB), so OpenCV's encoder runs;
 * `failed` when fpng reports an error, which today's path raises. */
Fpng fpng_png(const uint8_t *source, ptrdiff_t height, ptrdiff_t width, ptrdiff_t channels, ptrdiff_t row_stride,
              ptrdiff_t pixel_stride, ptrdiff_t channel_stride, std::vector<uint8_t> &encoded);

/* OpenCV writes 4-sample RGB TIFFs without ExtraSamples (tag 338), which TIFF 6.0 requires, so readers may take the
 * fourth sample for an unknown one (upstream chaiNNer #2950). Adds 338 = SHORT 2 (unassociated alpha: chaiNNer's
 * straight alpha) to a classic TIFF's first IFD: a copy of the IFD with the entry in tag order goes to the end of the
 * file, on a word boundary, and the header points to it; every other byte stays where it was. False (and `fixed`
 * untouched) for every other file, which stays as it is. */
bool tiff_extra_samples(std::span<const uint8_t> file, std::vector<uint8_t> &fixed);

} // namespace cn::image_encode
