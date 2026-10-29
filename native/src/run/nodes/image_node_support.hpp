/* Shared pieces of the image node entry points (native/src/run/nodes): enforced inputs as the Python bodies read
 * them, and the glue their NumPy slicing does. */
#pragma once

#include "../kernel_status.hpp"
#include "dispatch.hpp"
#include "image_resize.hpp"

#include "cn_crt_math.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#ifndef _WIN32
#include <cerrno>
#include <cstring>
#endif

namespace cn::run {

/* A number input as a Python float: an enforced int (precision 0) or float. */
inline double number(const Value &value)
{
    if (const int64_t *integer = std::get_if<int64_t>(&value)) return double(*integer);
    return std::get<double>(value);
}

/* An int input; a Python int beyond int64 (a PyRef) declines to the Python body. */
inline int64_t integer(const Value &value)
{
    if (const int64_t *integer = std::get_if<int64_t>(&value)) return *integer;
    throw RunPython{};
}

/* native_noise.seed_words of a seed input's value: its base-2**32 digits, little-endian; ValueError for a negative
 * value. A seed beyond int64 (a PyRef) declines to the Python body. */
inline std::vector<uint32_t> seed_words(const Value &value)
{
    const Seed *seed = std::get_if<Seed>(&value);
    if (!seed) throw RunPython{};
    if (seed->value < 0) throw NodeError("ValueError", "expected non-negative integer");
    std::vector<uint32_t> words{uint32_t(seed->value)};
    if (seed->value >> 32) words.push_back(uint32_t(uint64_t(seed->value) >> 32));
    return words;
}

/* An int input as an Extent; a Python int beyond int64 (a PyRef) declines to the Python body. */
inline Extent extent(const Value &value) { return Extent::of(integer(value)); }

/* max(extent, 1). */
inline Extent at_least_one(Extent extent)
{
    if (extent.mantissa == 0) extent.mantissa = 1;
    return extent;
}

/* repr(image.shape) of the array the Python body holds: (h, w) for one channel, else (h, w, c). */
inline std::string shape_text(const Image &image)
{
    std::string text = "(" + std::to_string(image.height) + ", " + std::to_string(image.width);
    if (image.channels != 1) text += ", " + std::to_string(image.channels);
    return text + ")";
}

inline size_t pixels(const Image &image) { return size_t(image.height) * size_t(image.width); }

/* native_layout._shape, native_channels._shape and native_resample._shape: a nonempty image. */
inline void nonempty(const Image &image)
{
    if (image.height == 0 || image.width == 0)
        throw NodeError("ValueError", "Expected a nonempty two- or three-dimensional image");
}

/* linear_light.linearize of a nonempty image: a copy whose colour channels went through the LightTransfer value's
 * curve to linear light (alpha, a fourth channel, as is). */
inline Image linearize(const Image &image, int64_t transfer)
{
    MutableImage linear = MutableImage::allocate(image.height, image.width, image.channels);
    std::copy_n(image.data, image.values(), linear.data());
    check(cn_linear_light(linear.data(), size_t(image.height), size_t(image.width), size_t(image.channels),
                          int(transfer), 0, image.channels == 4));
    return std::move(linear).publish();
}

/* linear_light.encode_inplace of an owned linear-light result. */
inline void encode_light(MutableImage &image, int64_t transfer)
{
    check(cn_linear_light(image.data(), size_t(image.height()), size_t(image.width()), size_t(image.channels()),
                          int(transfer), 1, image.channels() == 4));
}

/* The image's channels in the given order as a new image (NumPy's img[:, :, i] or img[:, :, a:b] slices, copied as
 * the output enforce copies them); an empty image gives an empty result of the same size. */
inline Image select_channels(const Image &image, std::initializer_list<int> channels)
{
    MutableImage out = MutableImage::allocate(image.height, image.width, int32_t(channels.size()));
    if (pixels(image) == 0) return std::move(out).publish();
    std::vector<cn_channel_source> sources;
    for (int channel : channels) sources.push_back({image.data, size_t(image.channels), size_t(channel), 0.0f});
    check(cn_channels_assemble(sources.data(), out.data(), pixels(image), sources.size()));
    return std::move(out).publish();
}

/* native_numpy_reduce's full-reduction setup of an aligned float32 NumPy view: the inner-loop block and the run
 * period (0 for none) and whether the loop reads a contiguous stream. shape and strides (in bytes) of the view. */
struct ReduceSetup {
    size_t block = 0, period = 0;
    bool contiguous = true;
};

inline ReduceSetup reduce_setup(std::vector<int64_t> shape, std::vector<int64_t> strides)
{
    constexpr int64_t buffer = 8192, itemsize = 4;  // NPY_BUFSIZE; float32
    // The iterator drops length-1 axes, stably orders the rest (innermost first) by |stride| and coalesces axes that
    // continue one stride.
    std::vector<std::pair<int64_t, int64_t>> order, axes;
    int64_t size = 1;
    for (size_t i = shape.size(); i-- > 0;) {
        size *= shape[i];
        if (shape[i] != 1) order.emplace_back(strides[i], shape[i]);
    }
    std::stable_sort(order.begin(), order.end(),
                     [](const auto &a, const auto &b) { return std::abs(a.first) < std::abs(b.first); });
    for (auto [stride, length] : order) {
        if (!axes.empty() && axes.back().first * axes.back().second == stride) axes.back().second *= length;
        else axes.emplace_back(stride, length);
    }
    if (axes.empty()) return {};
    int64_t cost = 1, current = axes[0].second;
    int64_t best_cost = cost, best_size = current, best_core = 1;
    size_t best_dim = 0;
    for (size_t dim = 1; dim < axes.size(); ++dim) {
        if (current >= buffer && cost > 1) break;
        if (dim == 1) ++cost;  // a coalesced axis boundary: the input must be buffered
        int64_t core = current;
        current *= axes[dim].second;
        int64_t limit = current > buffer && cost > 1 ? buffer : current;
        if (cost * best_size <= best_cost * limit) best_cost = cost, best_size = current, best_core = core, best_dim = dim;
    }
    int64_t period = best_size, block = period;
    bool buffered = best_dim > 0;
    if (buffered && block > buffer) block = best_core * (buffer / best_core);
    return {size_t(block >= period ? 0 : block), size_t(period >= size ? 0 : period),
            buffered || axes[0].first == itemsize};
}

/* A three-channel image with one alpha channel appended (native_channels.concatenate_channels(image, alpha) and
 * combine_rgb_alpha of a three-channel image), both nonempty and of one size. */
inline Image with_alpha(const Image &image, const Image &alpha)
{
    std::vector<cn_channel_source> sources{{image.data, 3, 0, 0.0f}, {image.data, 3, 1, 0.0f},
                                           {image.data, 3, 2, 0.0f}, {alpha.data, 1, 0, 0.0f}};
    MutableImage out = MutableImage::allocate(image.height, image.width, 4);
    check(cn_channels_assemble(sources.data(), out.data(), pixels(image), sources.size()));
    return std::move(out).publish();
}

/* Python's start:stop (step 1) over a sequence of size n as the [begin, end) it selects: negative indices count from
 * the end, both clamp to [0, n]. */
inline std::pair<int64_t, int64_t> slice_bounds(__int128 start, __int128 stop, int64_t n)
{
    auto index = [n](__int128 i) -> int64_t {
        if (i < 0) i += n;
        return int64_t(i < 0 ? 0 : i > n ? n : i);
    };
    int64_t begin = index(start), end = index(stop);
    return {begin, end < begin ? begin : end};
}

/* image[y0:y1, x0:x1] (Python slices): the input itself when the slice is the whole image, rows of full width
 * without a copy, else a copy of the region. */
inline Value slice(const Value &input, __int128 y0, __int128 y1, __int128 x0, __int128 x1)
{
    const Image &image = std::get<Image>(input);
    auto [top, bottom] = slice_bounds(y0, y1, image.height);
    auto [left, right] = slice_bounds(x0, x1, image.width);
    if (top == 0 && bottom == image.height && left == 0 && right == image.width) return input;
    int32_t height = int32_t(bottom - top), width = int32_t(right - left);
    if (height == 0 || width == 0) return MutableImage::allocate(height, width, image.channels).publish();
    if (width == image.width)
        return Image{image.storage, image.data + size_t(top) * size_t(image.width) * size_t(image.channels), height,
                     width, image.channels};
    MutableImage out = MutableImage::allocate(height, width, image.channels);
    check(cn_channels_crop(image.data, out.data(), size_t(image.height), size_t(image.width), size_t(image.channels),
                           size_t(left), size_t(top), size_t(width), size_t(height), size_t(width)));
    return std::move(out).publish();
}

/* Python's round(x, ndigits) of a float for 0 <= ndigits <= 22 (CPython's double_round): x as the decimal string
 * correctly rounded to ndigits places (ties to even on the exact binary value), read back correctly rounded; NaN
 * and the infinities unchanged. */
inline double python_round(double x, int ndigits)
{
    if (!std::isfinite(x)) return x;
    char text[352];
    char *end = std::to_chars(text, text + sizeof text, x, std::chars_format::fixed, ndigits).ptr;
    double result = 0;
    std::from_chars(text, end, result);
    return result;
}

/* CPython's float ** float (float_pow): its special cases for a zero exponent, NaN, infinite exponents and bases and
 * a zero base, then the CRT's pow of the base's magnitude (negated for a negative base and an odd exponent), whose
 * infinite result raises OverflowError (errno ERANGE; a denormal does not). A negative base with a fractional exponent
 * (a complex result) declines to the Python body. */
inline double python_pow(double base, double exponent)
{
    auto odd = [](double x) { return std::fmod(std::fabs(x), 2.0) == 1.0; };
    if (exponent == 0) return 1.0;
    if (std::isnan(base)) return base;
    if (std::isnan(exponent)) return base == 1.0 ? 1.0 : exponent;
    if (std::isinf(exponent)) {
        double magnitude = std::fabs(base);
        if (magnitude == 1.0) return 1.0;
        return (exponent > 0) == (magnitude > 1.0) ? std::fabs(exponent) : 0.0;
    }
    if (std::isinf(base)) {
        if (exponent > 0) return odd(exponent) ? base : std::fabs(base);
        return odd(exponent) ? std::copysign(0.0, base) : 0.0;
    }
    if (base == 0) {
        if (exponent < 0) throw NodeError("ZeroDivisionError", "zero to a negative power");
        return odd(exponent) ? base : 0.0;
    }
    bool negate = false;
    if (base < 0) {
        if (exponent != std::floor(exponent)) throw RunPython{};
        base = -base, negate = odd(exponent);
    }
    if (base == 1.0) return negate ? -1.0 : 1.0;
    double result = cn_crt.pow(base, exponent);
#ifdef _WIN32
    if (std::isinf(result)) throw NodeError("OverflowError", "(34, 'Result too large')");
#else
    // CPython raises float pow's ERANGE with the C library's text for it
    if (std::isinf(result))
        throw NodeError("OverflowError", "(" + std::to_string(ERANGE) + ", '" + std::strerror(ERANGE) + "')");
#endif
    return negate ? -result : result;
}

/* native_analysis.binary of two images (operation 0 high pass, 1 thresholded unsharp, 2 squared difference, 3 product;
 * scale and threshold as c_floats): each operand nonempty, both of one shape. */
inline Image analysis_binary(const Image &image, const Image &other, int op, double scale = 0, double threshold = 0)
{
    for (const Image *operand : {&image, &other})
        if (operand->height == 0 || operand->width == 0)
            throw NodeError("ValueError", "Invalid image shape " + shape_text(*operand));
    if (image.height != other.height || image.width != other.width || image.channels != other.channels)
        throw NodeError("ValueError", "Native image operands must have the same shape");
    MutableImage out = MutableImage::allocate(image.height, image.width, image.channels);
    check(cn_analysis_binary(image.data, other.data, out.data(), image.values(), op, float(scale), float(threshold)));
    return std::move(out).publish();
}

/* An ErrorDiffusionMap value's index in native_neighborhood.DIFFUSION_IDS. */
inline int diffusion_map(const std::string &value)
{
    static constexpr std::array<std::string_view, 8> maps{"FS", "JJN", "ST", "A", "B", "S", "S2", "SL"};
    return int(std::find(maps.begin(), maps.end(), value) - maps.begin());
}

/* An integer input that native_neighborhood checks against [2, 2**32) (a colour count, a Riemersma history length):
 * its value, or ValueError with the message (a Python int beyond int64 is out of range too). */
inline uint32_t dither_count(const Value &value, const char *message)
{
    const int64_t *count = std::get_if<int64_t>(&value);
    if (!count || *count < 2 || *count > 0xFFFFFFFF) throw NodeError("ValueError", message);
    return uint32_t(*count);
}

/* image_utils.to_uint8 of a float32 image: always saturate before quantizing.
 * The compatibility flag is ignored. NumPy's NaN warnings are not raised here. */
inline std::vector<uint8_t> to_uint8(const Image &image, bool /* normalized */)
{
    std::vector<uint8_t> out(std::max<size_t>(image.values(), 1));
    check(cn_pixels_convert_checked(image.data, out.data(), image.values(), 0, 1, 1, nullptr));
    return out;
}

/* native_color_complete.cvt_color of a float32 image for one code of its _CODES table, given as (operation, blue
 * index, output channels): OpenCV's row loop lanes of the operation's dispatched family (Lab's baseline: 4) and its
 * IPP state. An untested OpenCV declines to the Python body (cv2.cvtColor). */
inline Image cvt_color(const Image &image, int op, int blue, int32_t output_channels)
{
    if (!dispatch::cv2_tested("nodes.impl.native_color_complete")) throw RunPython{};
    if (image.height == 0 || image.width == 0) throw NodeError("ValueError", "Color conversion requires a nonempty image");
    const char *family = op <= 1 ? "color_rgb" : op <= 3 || op == 8 || op == 9 ? "color_yuv"
                         : op <= 7 || op == 14                                ? "color_hsv"
                                                                              : nullptr;
    MutableImage out = MutableImage::allocate(image.height, image.width, output_channels);
    check(cn_color_convert_f32(image.data, out.data(), size_t(image.height), size_t(image.width),
                               size_t(image.channels), size_t(output_channels), op, blue,
                               int(family ? dispatch::opencv_lanes(family) : 4), dispatch::cv2_use_ipp() ? 1 : 0));
    return std::move(out).publish();
}

}  // namespace cn::run
