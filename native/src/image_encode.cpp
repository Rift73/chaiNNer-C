/* Save Image's shared encoding steps (GPL-3.0); see native/include/image_encode.hpp. */
#include "image_encode.hpp"

#include "fpng.h"
#include "image_decode.hpp"

#include <climits>
#include <cstring>
#include <mutex>

namespace cn::image_encode {

Fpng fpng_png(const uint8_t *source, ptrdiff_t height, ptrdiff_t width, ptrdiff_t channels, ptrdiff_t row_stride,
              ptrdiff_t pixel_stride, ptrdiff_t channel_stride, std::vector<uint8_t> &encoded) {
    // Vendor 1.0.6 writes only two dimension bytes into IHDR despite accepting larger values. It also uses uint32
    // offsets and signed row sizes. Keep valid dimensions and worst-case uncompressed framing below those limits.
    constexpr uint64_t maximum_dimension = 65535;
    constexpr uint64_t maximum_filtered = INT32_MAX - UINT64_C(1048576);
    if ((channels != 3 && channels != 4) || height <= 0 || width <= 0 || uint64_t(height) > maximum_dimension ||
        uint64_t(width) > maximum_dimension ||
        (uint64_t(width) * uint64_t(channels) + 1) * uint64_t(height) > maximum_filtered)
        return Fpng::declined;
    static std::once_flag initialized;
    std::call_once(initialized, fpng::fpng_init);
    // fpng takes packed RGB(A); reading through byte pointers also takes readonly, reversed and broadcast views.
    std::vector<uint8_t> pixels(size_t(height) * size_t(width) * size_t(channels));
    uint8_t *destination = pixels.data();
    for (ptrdiff_t y = 0; y < height; ++y)
        for (ptrdiff_t x = 0; x < width; ++x) {
            const uint8_t *pixel = source + y * row_stride + x * pixel_stride;
            *destination++ = pixel[2 * channel_stride];
            *destination++ = pixel[channel_stride];
            *destination++ = pixel[0];
            if (channels == 4) *destination++ = pixel[3 * channel_stride];
        }
    return fpng::fpng_encode_image_to_memory(pixels.data(), uint32_t(width), uint32_t(height), uint32_t(channels),
                                             encoded, 0)
               ? Fpng::encoded
               : Fpng::failed;
}

bool tiff_extra_samples(std::span<const uint8_t> file, std::vector<uint8_t> &fixed) {
    const uint8_t *p = file.data();
    const size_t n = file.size();
    const cn::image_decode::ClassicTiff tiff(p, n);
    if (!tiff.valid) return false;
    const size_t entries = tiff.ifd + 2, count = tiff.u16(tiff.ifd);
    if (count * 12 + 4 > n - entries) return false;
    uint32_t samples = 0, photometric = 0;
    size_t insert = count;
    for (size_t i = 0; i < count; ++i) {
        const size_t entry = entries + i * 12;
        const uint32_t tag = tiff.u16(entry);
        if (tag == 338) return false;
        if (tag == 277) samples = tiff.u16(entry + 8);
        if (tag == 262) photometric = tiff.u16(entry + 8);
        if (tag > 338 && insert == count) insert = i;
    }
    const size_t moved = n + (n & 1), size = moved + 2 + (count + 1) * 12 + 4;
    if (samples != 4 || photometric != 2 || count == 0xFFFF || moved > UINT32_MAX) return false;
    fixed.resize(size);
    uint8_t *out = fixed.data(), *at = out + n;
    const auto put = [&](uint32_t value, int bytes) {
        for (int i = 0; i < bytes; ++i) *at++ = uint8_t(value >> (8 * (tiff.little ? i : bytes - 1 - i)));
    };
    std::memcpy(out, p, n);
    if (n & 1) *at++ = 0;
    put(uint32_t(count + 1), 2);
    std::memcpy(at, p + entries, insert * 12);
    at += insert * 12;
    put(338, 2);
    put(3, 2);
    put(1, 4);
    put(2, 2);
    put(0, 2);
    std::memcpy(at, p + entries + insert * 12, (count - insert) * 12 + 4); // and the next IFD's offset
    at = out + 4;
    put(uint32_t(moved), 4);
    return true;
}

} // namespace cn::image_encode
