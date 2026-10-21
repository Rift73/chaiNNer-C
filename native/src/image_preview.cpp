/* LargeImageOutput's UI previews, natively (GPL-3.0); see native/include/image_preview.hpp. */
#include "image_preview.hpp"

#include "chainner_ext_kernels.h"
#include "codec_output.hpp"
#include "image_codecs.h"
#include "image_encode.hpp"
#include "run_kernels.h"

#include <algorithm>
#include <cmath>
#include <new>
#include <span>
#include <stdexcept>

namespace cn::image_preview {
namespace {

constexpr int filter_box = 4;    // ResizeFilter.BOX
constexpr int type_float32 = 0;  // native_buffers._TYPES
constexpr int output_uint8 = 1;  // cn_pixels_convert_checked's uint8 output

struct Pixels {
    const float *data;
    uint32_t height, width, channels;
};

// resize(img, (width, height), ResizeFilter.BOX) for an image it resizes: four channels go through premultiplied
// alpha (native_resample.premultiply, then finish_alpha_inplace); the result is height x width x channels.
bool box_resize(const Pixels &in, uint32_t height, uint32_t width, std::vector<float> &out) {
    const size_t pixels = size_t(in.height) * in.width, channels = in.channels;
    std::vector<float> premultiplied;
    const float *source = in.data;
    if (channels == 4) {
        premultiplied.resize(pixels * 4);
        if (cn_resample_alpha(in.data, premultiplied.data(), pixels, 0) != CN_OK) return false;
        source = premultiplied.data();
    }
    out.resize(size_t(height) * width * channels);
    if (cn_resample_filtered(source, out.data(), in.height, in.width, channels, height, width, filter_box, 0, 0) != CN_OK)
        return false;
    return channels != 4 || cn_resample_alpha(out.data(), out.data(), size_t(height) * width, 1) == CN_OK;
}

std::string base64(std::span<const uint8_t> bytes) {
    static constexpr char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string text;
    text.reserve((bytes.size() + 2) / 3 * 4);
    size_t i = 0;
    for (; i + 3 <= bytes.size(); i += 3) {
        const uint32_t v = uint32_t(bytes[i]) << 16 | uint32_t(bytes[i + 1]) << 8 | bytes[i + 2];
        for (int shift = 18; shift >= 0; shift -= 6) text.push_back(alphabet[v >> shift & 63]);
    }
    if (const size_t rest = bytes.size() - i) {
        const uint32_t v = uint32_t(bytes[i]) << 16 | (rest == 2 ? uint32_t(bytes[i + 1]) << 8 : 0);
        text.push_back(alphabet[v >> 18 & 63]);
        text.push_back(alphabet[v >> 12 & 63]);
        text.push_back(rest == 2 ? alphabet[v >> 6 & 63] : '=');
        text.push_back('=');
    }
    return text;
}

// preview_encode(img, target_size, grace, lossless): a resize when a side exceeds target_size * grace, then
// to_uint8(img) as a PNG (more than 3 channels, or lossless; chaiNNer-C's default PNG, cn::image_encode::png) or as
// cv2.imencode's JPEG with OpenCV's defaults (quality 95, 4:2:0, baseline), as a data URL. `image` becomes the
// encoded image, held in `owned` when resized.
bool encode_step(Pixels &image, int target, double grace, bool lossless, std::vector<float> &owned, std::string &url) {
    const double largest = target * grace;
    if (image.width > largest || image.height > largest) {
        const double f = std::max(double(image.width) / target, double(image.height) / target);
        const auto side = [&](uint32_t value) {
            return uint32_t(std::max(1.0, std::floor(double(value) / f + 0.5))); // max(1, round_half_up(value / f))
        };
        const uint32_t width = side(image.width), height = side(image.height);
        std::vector<float> resized;
        if (!box_resize(image, height, width, resized)) return false;
        owned.swap(resized);
        image = Pixels{owned.data(), height, width, image.channels};
    }
    const size_t count = size_t(image.height) * image.width * image.channels;
    std::vector<uint8_t> pixels(count);
    int events = 0;
    if (cn_pixels_convert_checked(image.data, pixels.data(), count, type_float32, output_uint8, 1, &events) != CN_OK ||
        events)
        return false; // NumPy would report the event (warn, raise, print or call): today's path does
    if (image.channels > 3 || lossless) {
        const ptrdiff_t c = ptrdiff_t(image.channels);
        std::vector<uint8_t> png;
        switch (cn::image_encode::png(pixels.data(), image.height, image.width, c, ptrdiff_t(image.width) * c, c, 1, png)) {
        case cn::image_encode::Png::encoded:
            url = "data:image/png;base64," + base64(png);
            return true;
        case cn::image_encode::Png::declined:
            return false; // today's path runs
        case cn::image_encode::Png::out_of_memory:
            throw std::bad_alloc();
        default:
            throw std::runtime_error("the PNG encoder failed to encode the preview");
        }
    }
    const cn_codec_image view{pixels.data(), image.height, image.width, image.channels, 1, size_t(image.width) * image.channels};
    cn::codecs::Output output;
    auto &out = output.value;
    const cn_codec_status status = cn_jpeg_encode(&view, 95, 0x221111, 0, &out);
    if (status == CN_CODEC_OK) url = "data:image/jpg;base64," + base64({out.data, out.size});
    return status == CN_CODEC_OK;
}

} // namespace

bool previews(const float *data, uint32_t height, uint32_t width, uint32_t channels, std::vector<Preview> &out) {
    static constexpr int sizes[] = {2048, 1024, 512, 256};
    constexpr double grace = 1.2;
    const uint32_t image_size = std::max(height, width);
    size_t start = std::size(sizes) - 1;
    for (size_t i = 0; i < std::size(sizes); ++i) {
        if (sizes[i] <= int64_t(image_size) && image_size <= sizes[i] * grace) {
            start = i;
            break;
        }
        if (image_size > uint32_t(sizes[i])) {
            start = i ? i - 1 : 0;
            break;
        }
    }
    Pixels image{data, height, width, channels};
    std::vector<float> owned;
    for (size_t i = start; i < std::size(sizes); ++i) {
        Preview preview;
        if (!encode_step(image, sizes[i], grace, i == start, owned, preview.url)) return false;
        preview.width = image.width;
        preview.height = image.height;
        out.push_back(std::move(preview));
    }
    return true;
}

} // namespace cn::image_preview
