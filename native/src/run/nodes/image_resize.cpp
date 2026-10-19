/* nodes/impl/resize.py's resize() and its native_resample bridges, line by line (image_resize.hpp). */
#include "image_resize.hpp"
#include "image_node_support.hpp"

#include "../kernel_status.hpp"
#include "chainner_ext_kernels.h"

#include <charconv>
#include <cmath>
#include <limits>
#include <string>

namespace cn::run {
namespace {

constexpr uint64_t int32_limit = uint64_t(std::numeric_limits<int32_t>::max());
constexpr uint64_t coordinate_limit = (uint64_t(1) << 32) - 1;

/* Python's repr() of a positive finite double: its shortest round-trip digits, in fixed notation when the decimal
 * point falls within 16 digits (with ".0" for an integer), else as d.ddde+XX (float_repr_style "short"). */
std::string python_repr(double value)
{
    char buffer[64];
    auto end = std::to_chars(buffer, buffer + sizeof(buffer), value, std::chars_format::scientific).ptr;
    std::string scientific(buffer, end);
    size_t e = scientific.find('e');
    std::string digits = scientific.substr(0, 1) + (e > 2 ? scientific.substr(2, e - 2) : "");
    int exponent = std::stoi(scientific.substr(e + 1));
    int point = exponent + 1;
    if (point <= -4 || point > 16) {
        std::string text = digits.substr(0, 1) + (digits.size() > 1 ? "." + digits.substr(1) : "");
        std::string power = std::to_string(exponent < 0 ? -exponent : exponent);
        return text + (exponent < 0 ? "e-" : "e+") + (power.size() < 2 ? "0" : "") + power;
    }
    if (point <= 0) return "0." + std::string(size_t(-point), '0') + digits;
    if (size_t(point) >= digits.size()) return digits + std::string(size_t(point) - digits.size(), '0') + ".0";
    return digits.substr(0, size_t(point)) + "." + digits.substr(size_t(point));
}

/* round(value, 3) of a finite double: the decimal rounded half to even at three places, read back (CPython's
 * double_round through dtoa mode 3). */
double python_round3(double value)
{
    char buffer[400];
    auto end = std::to_chars(buffer, buffer + sizeof(buffer), value, std::chars_format::fixed, 3).ptr;
    double rounded = 0;
    std::from_chars(buffer, end, rounded);
    return rounded;
}

/* The exact product mantissa_a * mantissa_b * factor (factor 1 or 3, operands below 2^63) rounded half to even to a
 * double, times 2^exponent: an overflow is the infinity. */
double scaled_product(uint64_t mantissa_a, uint64_t mantissa_b, uint64_t factor, int exponent)
{
    unsigned __int128 product = static_cast<unsigned __int128>(mantissa_a) * mantissa_b * factor;
    if (product == 0) return 0.0;
    int bits = 0;
    for (unsigned __int128 rest = product; rest; rest >>= 1) ++bits;
    int shift = bits > 53 ? bits - 53 : 0;
    uint64_t top = uint64_t(product >> shift);
    if (shift) {
        unsigned __int128 remainder = product & ((static_cast<unsigned __int128>(1) << shift) - 1);
        unsigned __int128 half = static_cast<unsigned __int128>(1) << (shift - 1);
        if (remainder > half || (remainder == half && (top & 1))) ++top;
    }
    return std::ldexp(double(top), shift + exponent);
}

/* resize()'s memory check: new_w * new_h * c * 4 bytes against 16 GiB, with the message's round(.. / GB, 3). */
void check_memory(Extent width, Extent height, int32_t channels)
{
    uint64_t factor = channels == 3 ? 3 : 1;
    int exponent = width.exponent + height.exponent + (channels == 4 ? 4 : 2);
    /* new_memory = width.mantissa * height.mantissa * factor * 2^exponent; the limit is 2^34. */
    unsigned __int128 product = static_cast<unsigned __int128>(width.mantissa) * height.mantissa * factor;
    bool over = exponent > 34 ? product != 0 : product > (static_cast<unsigned __int128>(1) << (34 - exponent));
    if (!over) return;
    double gigabytes = scaled_product(width.mantissa, height.mantissa, factor, exponent - 30);
    if (std::isinf(gigabytes)) throw NodeError("OverflowError", "integer division result too large for a float");
    throw NodeError("RuntimeError", "Resize would require " + python_repr(python_round3(gigabytes)) +
                                        " GB of memory, but only 16 GB are allowed.");
}

/* The bridges' coordinate check, then the engine's image limit (decline: the body would return a larger image). */
void check_target(uint64_t width, uint64_t height)
{
    if (width > coordinate_limit || height > coordinate_limit)
        throw NodeError("OverflowError", "Resize dimensions exceed the native coordinate range");
    if (width > int32_limit || height > int32_limit) throw RunPython{};
}

/* native_resample.nearest (dimensions positive). */
Image nearest(const Image &image, uint64_t width, uint64_t height)
{
    nonempty(image);
    check_target(width, height);
    MutableImage out = MutableImage::allocate(int32_t(height), int32_t(width), image.channels);
    check(cn_resample_nearest(image.data, out.data(), size_t(image.height), size_t(image.width),
                              size_t(image.channels), size_t(height), size_t(width)));
    return std::move(out).publish();
}

/* native_resample.filtered. The engine's images are C-contiguous, so the bridge's vector_clip (a strided RGBA
 * upscale) is 0. */
MutableImage filtered(const Image &image, uint64_t width, uint64_t height, int64_t filter, bool gamma_correction)
{
    nonempty(image);
    check_target(width, height);
    MutableImage out = MutableImage::allocate(int32_t(height), int32_t(width), image.channels);
    if (width == 0 || height == 0) return out;
    check(cn_resample_filtered(image.data, out.data(), size_t(image.height), size_t(image.width),
                               size_t(image.channels), size_t(height), size_t(width), int(filter),
                               gamma_correction ? 1 : 0, 0));
    return out;
}

}  // namespace

Extent Extent::of(int64_t value)
{
    return Extent{uint64_t(value), 0};
}

Extent Extent::round_half_up(double value)
{
    double rounded = std::floor(value + 0.5);
    if (std::isnan(rounded)) throw NodeError("ValueError", "cannot convert float NaN to integer");
    if (std::isinf(rounded)) throw NodeError("OverflowError", "cannot convert float infinity to integer");
    if (rounded < 0x1p63) return Extent{uint64_t(rounded), 0};
    int exponent = 0;
    double fraction = std::frexp(rounded, &exponent);
    return Extent{uint64_t(std::ldexp(fraction, 53)), exponent - 53};
}

Value resize(const Image &image, Extent width, Extent height, int64_t filter, bool separate_alpha,
             bool gamma_correction)
{
    check_memory(width, height, image.channels);
    /* A dimension beyond 2^63 compares as UINT64_MAX: above every image size and coordinate limit. */
    uint64_t new_w = width.exponent ? std::numeric_limits<uint64_t>::max() : width.mantissa;
    uint64_t new_h = height.exponent ? std::numeric_limits<uint64_t>::max() : height.mantissa;
    uint64_t w = uint64_t(image.width), h = uint64_t(image.height);
    if (filter == filter_auto) filter = new_w > w || new_h > h ? filter_lanczos : filter_box;
    if (new_w == w && new_h == h && (filter == filter_nearest || filter == filter_box)) return image;
    if (filter == filter_nearest) {
        /* The original binding returns HWC even for a 2D grayscale input. */
        if (new_w == 0 || new_h == 0) return std::move(filtered(image, new_w, new_h, filter_box, false)).publish();
        return nearest(image, new_w, new_h);
    }
    bool premultiplied = !separate_alpha && image.channels == 4;
    Image source = image;
    if (premultiplied) {
        nonempty(image);
        MutableImage alpha = MutableImage::allocate(image.height, image.width, 4);
        check(cn_resample_alpha(image.data, alpha.data(), size_t(image.height) * size_t(image.width), 0));
        source = std::move(alpha).publish();
    }
    MutableImage out = filtered(source, new_w, new_h, filter, gamma_correction);
    size_t pixels = size_t(out.height()) * size_t(out.width());
    if (premultiplied && pixels) check(cn_resample_alpha(out.data(), out.data(), pixels, 1));
    return std::move(out).publish();
}

}  // namespace cn::run
