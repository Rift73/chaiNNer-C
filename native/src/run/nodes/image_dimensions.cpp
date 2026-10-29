/* Native entry points of image_dimension nodes (pattern: image_utility.cpp). */
#include "image_node_support.hpp"
#include "image_resize.hpp"

#include "chainner_ext_kernels.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>

namespace {

using namespace cn::run;

constexpr int64_t opencv_int_max = 2147483647;

/* An output size that an Image cannot hold declines to the Python body (NumPy allocates it or raises its own
 * MemoryError). */
int32_t dimension(int64_t size)
{
    if (size > std::numeric_limits<int32_t>::max()) throw RunPython{};
    return int32_t(size);
}

/* Python's min(v, 1.0) then max(0.0, ...): NaN passes min and becomes 0.0 in max. */
double clamp01(double v)
{
    double low = 1.0 < v ? 1.0 : v;
    return low > 0.0 ? low : 0.0;
}

/* native_layout.pad (BorderType values; 5 transparent, 6 white, 7 the custom colour). */
Value pad(const Value &input, int64_t border, const Color &color, int64_t top, int64_t right, int64_t bottom,
          int64_t left, std::optional<std::pair<int64_t, int64_t>> output_size)
{
    const Image &image = std::get<Image>(input);
    nonempty(image);
    int64_t h = image.height, w = image.width, c = image.channels;
    if (std::min({top, right, bottom, left}) < 0) throw NodeError("ValueError", "Padding must not be negative");
    if (std::max({top, right, bottom, left}) > opencv_int_max)
        throw NodeError("OverflowError", "Padding exceeds the original OpenCV integer range");
    auto [out_h, out_w] = output_size ? *output_size : std::pair{h + top + bottom, w + left + right};
    if (std::min(out_h, out_w) < 1) throw NodeError("ValueError", "Padding must produce a nonempty image");
    bool padded = top || right || bottom || left;
    if (!padded && out_h == h && out_w == w) return input;
    int64_t out_c = c, mode = border;
    std::vector<double> values(size_t(c), 0.0);
    if (padded) {
        if (border == 0 && c == 4) {
            values = {0.0, 0.0, 0.0, 1.0};
        } else if (border == 5) {
            out_c = 4, mode = 0, values.assign(4, 0.0);
        } else if (border == 6) {
            mode = 0, values.assign(size_t(c), 1.0);
        } else if (border == 7) {
            int64_t color_channels = int64_t(color.values.size());
            out_c = std::max(c, color_channels), mode = 0;
            values = color.values;
            if (color_channels < out_c) {
                // Upstream widens a float32 colour image, then reconstructs the Color.
                for (double &v : values) v = double(float(v));
                if (values.size() == 1 && (out_c == 3 || out_c == 4)) values.assign(3, values[0]);
                if (values.size() == 3 && out_c == 4) values.push_back(1.0);
                for (double &v : values) v = clamp01(v);
            }
        }
    } else {
        mode = 0;  // Offset mode may crop an image without ever creating a border.
    }
    std::vector<float> fill(values.begin(), values.end());
    MutableImage out = MutableImage::allocate(dimension(out_h), dimension(out_w), int32_t(out_c));
    check(cn_layout_pad(image.data, out.data(), fill.data(), size_t(h), size_t(w), size_t(c), size_t(out_h),
                        size_t(out_w), size_t(out_c), size_t(top), size_t(left), int(mode)));
    return std::move(out).publish();
}

/* chainner:image:pad (border/pad.py): BorderMode 0 one amount, 1 per edge, 2 offsets into a width x height canvas. */
void pad_node(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image &image = std::get<Image>(inputs[0]);
    int64_t border = std::get<int64_t>(inputs[1]);
    const Color &color = std::get<Color>(inputs[2]);
    int64_t mode = std::get<int64_t>(inputs[3]);
    if (mode == 0) {
        int64_t amount = integer(inputs[4]);
        outputs.push_back(pad(inputs[0], border, color, amount, amount, amount, amount, std::nullopt));
        return;
    }
    int64_t left = integer(inputs[5]), top = integer(inputs[6]), right = integer(inputs[7]), bottom = integer(inputs[8]);
    if (mode == 1) {
        outputs.push_back(pad(inputs[0], border, color, top, right, bottom, left, std::nullopt));
        return;
    }
    int64_t width = integer(inputs[9]), height = integer(inputs[10]);
    int64_t right_pad = std::max<int64_t>(0, width - left - image.width);
    int64_t bottom_pad = std::max<int64_t>(0, height - top - image.height);
    outputs.push_back(pad(inputs[0], border, color, top, right_pad, bottom_pad, left, std::pair{height, width}));
}

void assertion(bool condition, const char *message)
{
    if (!condition) throw NodeError("AssertionError", message);
}

/* chainner:image:crop (crop/crop.py): CropMode 0 border, 1 edges, 2 offsets; NumPy slices. */
void crop(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image &image = std::get<Image>(inputs[0]);
    __int128 h = image.height, w = image.width;
    int64_t mode = std::get<int64_t>(inputs[1]);
    const char *no_height = "Cropped area would result in an image with no height";
    const char *no_width = "Cropped area would result in an image with no width";
    if (mode == 0) {
        __int128 amount = integer(inputs[2]);
        if (amount == 0) {
            outputs.push_back(inputs[0]);
            return;
        }
        assertion(2 * amount < h, no_height);
        assertion(2 * amount < w, no_width);
        outputs.push_back(slice(inputs[0], amount, h - amount, amount, w - amount));
    } else if (mode == 1) {
        __int128 left = integer(inputs[3]), top = integer(inputs[4]), right = integer(inputs[5]),
                 bottom = integer(inputs[6]);
        if (top == 0 && bottom == 0 && left == 0 && right == 0) {
            outputs.push_back(inputs[0]);
            return;
        }
        assertion(top + bottom < h, no_height);
        assertion(left + right < w, no_width);
        outputs.push_back(slice(inputs[0], top, h - bottom, left, w - right));
    } else {
        __int128 left = integer(inputs[3]), top = integer(inputs[4]), width = integer(inputs[7]),
                 height = integer(inputs[8]);
        assertion(top < h, no_height);
        assertion(left < w, no_width);
        outputs.push_back(slice(inputs[0], top, top + height, left, left + width));
    }
}

/* chainner:image:remove_border (crop/crop_border.py, native_resample.border_region): every enforced image takes the
 * native path (float32, 2-D or at least two channels); the detected region grows by the padding and is clipped to
 * the image (utils.Region). A C-ordered image reduces pairwise. */
void crop_border(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image &image = std::get<Image>(inputs[0]);
    double tolerance = number(inputs[1]) / 100;
    int64_t select = std::get<int64_t>(inputs[2]);
    int64_t padding = integer(inputs[3]);
    if (image.height == 0 || image.width == 0)
        throw NodeError("ValueError", "Border detection requires a nonempty float32 image");
    std::array<size_t, 4> bounds{};
    check(cn_resample_border(image.data, size_t(image.height), size_t(image.width), size_t(image.channels),
                             float(tolerance), int(select), bounds.data(), 1));
    __int128 x = __int128(bounds[0]) - padding, y = __int128(bounds[1]) - padding;
    __int128 right = x + __int128(bounds[2]) + 2 * padding, bottom = y + __int128(bounds[3]) + 2 * padding;
    __int128 left = std::max<__int128>(x, 0), top = std::max<__int128>(y, 0);
    __int128 width = std::min<__int128>(right, image.width) - left, height = std::min<__int128>(bottom, image.height) - top;
    if (width == image.width && height == image.height) {
        outputs.push_back(inputs[0]);
        return;
    }
    outputs.push_back(slice(inputs[0], top, top + height, left, left + width));
}

/* native_channels._bbox's bounds (x, y, width, height) of one channel above min(threshold / 100, 0.99999). */
std::array<size_t, 4> bounding_box(const Image &image, size_t channel, double threshold)
{
    double value = threshold / 100;
    std::array<size_t, 4> bounds{};
    check(cn_channels_bbox(image.data, size_t(image.height), size_t(image.width), size_t(image.channels), channel,
                           float(0.99999 < value ? 0.99999 : value), bounds.data()));
    return bounds;
}

/* chainner:image:crop_content (crop/crop_to_content.py, native_channels.crop_to_content): an image without alpha is
 * returned as it is. */
void crop_to_content(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image &image = std::get<Image>(inputs[0]);
    nonempty(image);
    if (image.channels < 4) {
        outputs.push_back(inputs[0]);
        return;
    }
    auto [x, y, width, height] = bounding_box(image, 3, number(inputs[1]));
    if (width == 0) throw NodeError("RuntimeError", "Crop results in empty image.");
    outputs.push_back(slice(inputs[0], y, y + height, x, x + width));
}

/* chainner:image:get_bbox (utility/get_bounding_box.py, native_channels.bounding_box). */
void get_bounding_box(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image &image = std::get<Image>(inputs[0]);
    nonempty(image);
    std::array<size_t, 4> bounds = bounding_box(image, 0, number(inputs[1]));
    if (bounds[2] == 0) throw NodeError("RuntimeError", "Resulting bounding box is empty.");
    for (size_t bound : bounds) outputs.push_back(int64_t(bound));
}

/* chainner:image:get_dims (utility/get_dimensions.py). */
void get_dimensions(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image &image = std::get<Image>(inputs[0]);
    outputs.push_back(int64_t(image.width));
    outputs.push_back(int64_t(image.height));
    outputs.push_back(int64_t(image.channels));
}

/* chainner:image:resize_pixel_art (resize/resize_pixel_art.py, native_pixel_art.pixel_art_upscale): ResizeAlgorithm
 * values with their kernel mode and scale. */
void resize_pixel_art(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    struct Algorithm {
        std::string_view value;
        int mode;
        int32_t scale;
    };
    static constexpr Algorithm algorithms[] = {
        {"adv_mame2x", 0, 2}, {"adv_mame3x", 1, 3}, {"adv_mame4x", 2, 4},   {"eagle2x", 3, 2},
        {"eagle3x", 4, 3},    {"super_eagle2x", 5, 2}, {"sai2x", 6, 2},     {"super_sai2x", 7, 2},
        {"hqx2x", 8, 2},      {"hqx3x", 9, 3},       {"hqx4x", 10, 4},
    };
    const Image &image = std::get<Image>(inputs[0]);
    const std::string &value = std::get<std::string>(inputs[1]);
    const Algorithm &algorithm =
        *std::find_if(std::begin(algorithms), std::end(algorithms), [&](const Algorithm &a) { return a.value == value; });
    if (image.height == 0 || image.width == 0) throw NodeError("ValueError", "Pixel-art images must be nonempty");
    MutableImage out = MutableImage::allocate(dimension(int64_t(image.height) * algorithm.scale),
                                              dimension(int64_t(image.width) * algorithm.scale), image.channels);
    check(cn_pixel_art_f32(image.data, out.data(), size_t(image.height), size_t(image.width), size_t(image.channels),
                           algorithm.mode));
    outputs.push_back(std::move(out).publish());
}

/* chainner:image:resize_to_side (resize/resize_to_side.py): resize_to_side_conditional's size (SideSelection and
 * ResizeCondition values), then resize.resize with DPID's Lambda (input 5). A scaled side is
 * round_half_up((target / side) * other), at least 1; a target beyond 2**53 (Python's exact int division) declines to
 * the Python body. */
void resize_to_side(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image &image = std::get<Image>(inputs[0]);
    int64_t target = integer(inputs[1]);
    const std::string &side = std::get<std::string>(inputs[2]), &condition = std::get<std::string>(inputs[3]);
    int64_t w = image.width, h = image.height;
    auto keep = [&](int64_t size) {
        return condition == "downscale" ? target > size : condition == "upscale" ? target < size : false;
    };
    auto scaled = [&](int64_t base, int64_t other) {
        if (base == 0) throw NodeError("ZeroDivisionError", "division by zero");
        if (target > (int64_t(1) << 53)) throw RunPython{};
        return at_least_one(Extent::round_half_up(double(target) / double(base) * double(other)));
    };
    Extent width = Extent::of(w), height = Extent::of(h);
    if (side == "width") {
        if (!keep(w)) width = Extent::of(target), height = scaled(w, h);
    } else if (side == "height") {
        if (!keep(h)) width = scaled(h, w), height = Extent::of(target);
    } else {
        int64_t reference = side == "shorter side" ? std::min(h, w) : std::max(h, w);
        if (!keep(reference)) width = scaled(reference, w), height = scaled(reference, h);
    }
    outputs.push_back(
        resize(image, width, height, std::get<int64_t>(inputs[4]), false, std::nullopt, number(inputs[5])));
}

/* native_descale._core's factorized one-direction systems, shared like its lru_cache(maxsize=16): the 16 most
 * recently used, each freed when the cache and every caller drop it. */
class DescaleCores {
public:
    struct Key {
        int64_t src, dst, mode, taps;
        double b, c, blur, shift, active;
        int64_t border;
        bool operator==(const Key &) const = default;
    };

    /* native_descale._core's message for cn_descale_core_create's refusal reason. */
    static std::string refusal_message(const Key &key, int refusal)
    {
        static constexpr std::array<std::string_view, 7> kernels{"Bilinear", "Bicubic",  "Lanczos", "Spline16",
                                                                 "Spline36", "Spline64", "Point"};
        if (refusal == 0) return "Invalid descale parameters";
        std::string kernel(kernels[size_t(key.mode - 1)]);
        if (refusal == 1) {
            char text[32];
            std::string blur(text, std::to_chars(text, text + sizeof text, key.blur, std::chars_format::general, 6).ptr);
            return "Descale: Blur " + blur + " is too narrow for the " + kernel +
                   " kernel at this size: some target pixels get no weight (the result would be black). Use a larger "
                   "Blur (1 is the kernel's own width).";
        }
        if (refusal == 2)
            return "Descale: with this source window (Source Left, Top, Width, Height) some target pixels get no "
                   "weight from the " + kernel + " kernel (the result would be black). Use a window that covers the "
                   "target.";
        return "Descale: the " + kernel +
               " kernel's weights are not finite with these parameters (the result would not be finite either).";
    }

    std::shared_ptr<void> get(const Key &key)
    {
        {
            std::lock_guard lock(mutex_);
            for (auto entry = entries_.begin(); entry != entries_.end(); ++entry) {
                if (entry->first == key) {
                    entries_.splice(entries_.begin(), entries_, entry);
                    return entry->second;
                }
            }
        }
        int refusal = 0;
        void *handle = cn_descale_core_create(int(key.src), int(key.dst), int(key.mode), int(key.taps), key.b, key.c,
                                              key.blur, key.shift, key.active, int(key.border), &refusal);
        if (!handle) throw NodeError("ValueError", refusal_message(key, refusal));
        std::shared_ptr<void> core(handle, cn_descale_core_free);
        std::lock_guard lock(mutex_);
        entries_.emplace_front(key, core);
        if (entries_.size() > 16) entries_.pop_back();
        return core;
    }

private:
    std::mutex mutex_;
    std::list<std::pair<Key, std::shared_ptr<void>>> entries_;
};

/* chainner:image:descale (resize/descale.py, native_descale.descale): DescaleKernel and DescaleBorder values; a
 * direction whose size, shift and window are unchanged is skipped. A source width or height of 0 means the target's.
 * Linear Light (13) solves on the Transfer (14) curve's linear copy and encodes the result. */
void descale(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    static DescaleCores cores;
    const Image &image = std::get<Image>(inputs[0]);
    int64_t width = integer(inputs[1]), height = integer(inputs[2]), mode = std::get<int64_t>(inputs[3]);
    double b = number(inputs[4]), c = number(inputs[5]);
    int64_t taps = integer(inputs[6]);
    double blur = number(inputs[7]), left = number(inputs[8]), top = number(inputs[9]);
    double active_w = number(inputs[10]), active_h = number(inputs[11]);
    int64_t border = std::get<int64_t>(inputs[12]);
    if (active_w == 0) active_w = double(width);
    if (active_h == 0) active_h = double(height);
    int64_t h = image.height, w = image.width;
    if (!(1 <= width && width <= w) || !(1 <= height && height <= h))
        throw NodeError("ValueError", "Descale needs a target no larger than the image (" + std::to_string(w) + "x" +
                                          std::to_string(h) + "), got " + std::to_string(width) + "x" +
                                          std::to_string(height));
    std::shared_ptr<void> core_h, core_v;
    if (width != w || left != 0 || active_w != double(width))
        core_h = cores.get({w, width, mode, taps, b, c, blur, left, active_w, border});
    if (height != h || top != 0 || active_h != double(height))
        core_v = cores.get({h, height, mode, taps, b, c, blur, top, active_h, border});
    size_t channels = size_t(image.channels);
    size_t size = cn_descale_workspace(size_t(h), size_t(w), channels, core_h.get(), core_v.get());
    if (size == 0) throw NodeError("ValueError", "Invalid descale image or parameters");
    std::unique_ptr<uint8_t[]> workspace = std::make_unique_for_overwrite<uint8_t[]>(size);
    MutableImage out = MutableImage::allocate(int32_t(height), int32_t(width), image.channels);
    bool linear_light = std::get<bool>(inputs[13]);
    int64_t transfer = std::get<int64_t>(inputs[14]);
    Image source = linear_light ? linearize(image, transfer) : image;
    check(cn_descale(source.data, size_t(h), size_t(w), channels, core_h.get(), core_v.get(), out.data(),
                     workspace.get(), size));
    if (linear_light) encode_light(out, transfer);
    outputs.push_back(std::move(out).publish());
}

}  // namespace

CN_RUN_NODES_DECLARE(image_dimensions)
{
    registry.add("chainner:image:crop", crop);
    registry.add("chainner:image:crop_content", crop_to_content);
    registry.add("chainner:image:descale", descale);
    registry.add("chainner:image:get_bbox", get_bounding_box);
    registry.add("chainner:image:get_dims", get_dimensions);
    registry.add("chainner:image:pad", pad_node);
    registry.add("chainner:image:remove_border", crop_border);
    registry.add("chainner:image:resize_pixel_art", resize_pixel_art);
    registry.add("chainner:image:resize_to_side", resize_to_side);
}
