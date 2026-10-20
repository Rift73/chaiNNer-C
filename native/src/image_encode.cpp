/* Save Image's shared encoding steps (GPL-3.0); see native/include/image_encode.hpp. */
#include "image_encode.hpp"

#include "image_decode.hpp"
#include "run_kernels.h"

#include <cstring>
#include <new>
#include <utility>

namespace cn::image_encode {

Png png(const uint8_t *source, ptrdiff_t height, ptrdiff_t width, ptrdiff_t channels, ptrdiff_t row_stride,
        ptrdiff_t pixel_stride, ptrdiff_t channel_stride, const std::function<uint8_t *(size_t)> &allocate) {
    if (channels < 1 || channels > 4 || height < 1 || width < 1 || size_t(height) > CN_PNG_MAX_SIDE ||
        size_t(width) > CN_PNG_MAX_SIDE || size_t(width) > CN_PNG_MAX_ROW_SAMPLES / size_t(channels))
        return Png::declined;
    // The encoder reads through the strides (readonly, reversed and broadcast views too) and swaps BGR(A) itself.
    const cn_png_allocate trampoline = [](void *context, size_t size) {
        return (*static_cast<const std::function<uint8_t *(size_t)> *>(context))(size);
    };
    const cn_status status = cn_png_encode_u8(source, size_t(height), size_t(width), size_t(channels), row_stride,
                                              pixel_stride, channel_stride, 1, trampoline,
                                              const_cast<std::function<uint8_t *(size_t)> *>(&allocate));
    if (status == CN_ALLOCATION_FAILED) return Png::out_of_memory;
    return status == CN_OK ? Png::encoded : Png::failed;
}

Png png(const uint8_t *source, ptrdiff_t height, ptrdiff_t width, ptrdiff_t channels, ptrdiff_t row_stride,
        ptrdiff_t pixel_stride, ptrdiff_t channel_stride, std::vector<uint8_t> &encoded) {
    std::vector<uint8_t> bytes;
    const Png result = png(source, height, width, channels, row_stride, pixel_stride, channel_stride,
                           [&](size_t size) -> uint8_t * {
                               try {
                                   bytes.resize(size);
                                   return bytes.data();
                               } catch (const std::bad_alloc &) {
                                   return nullptr;
                               }
                           });
    if (result == Png::encoded) encoded = std::move(bytes);
    return result;
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
