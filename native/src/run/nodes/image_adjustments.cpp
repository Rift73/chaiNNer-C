/* Native entry points of image_adjustment nodes (pattern: image_utility.cpp). Python's float arithmetic is double
 * arithmetic; ctypes passes a float to a c_float parameter as (float)double. */
#include "../kernel_status.hpp"
#include "image_node_support.hpp"

#include "chainner_ext_kernels.h"

#include <algorithm>
#include <cmath>
#include <optional>

namespace {

using namespace cn::run;

/* native_adjustments._channels: a two- or three-dimensional image without an empty axis. */
void adjustable_shape(const Image &image)
{
    if (image.height == 0 || image.width == 0) throw NodeError("ValueError", "Invalid image shape " + shape_text(image));
}

/* native_adjustments.adjust: the result has the image's shape; an enforced image is C-ordered, so the channel
 * stride is 1. */
Value adjust(const Image &image, int op, double a = 0, double b = 0, bool clip = false)
{
    adjustable_shape(image);
    MutableImage out = MutableImage::allocate(image.height, image.width, image.channels);
    check(cn_adjust_f32(image.data, out.data(), pixels(image), size_t(image.channels), 1, op, float(a), float(b),
                        clip ? 1 : 0));
    return std::move(out).publish();
}

/* chainner:image:brightness_and_contrast (adjustments/brightness_and_contrast.py). */
void brightness_and_contrast(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    double brightness = number(inputs[1]) / 100;
    double contrast = number(inputs[2]) / 100;
    if (brightness == 0 && contrast == 0) {
        outputs.push_back(inputs[0]);
        return;
    }
    double max_c = 259.0 / 255.0;
    double factor = (max_c * (contrast + 1)) / (max_c - contrast);
    double add = factor * brightness + 0.5 * (1 - factor);
    outputs.push_back(adjust(std::get<Image>(inputs[0]), 3, factor, add, add < 0 || factor + add > 1));
}

/* chainner:image:clamp (adjustments/clamp.py). */
void clamp(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    double minimum = number(inputs[1]), maximum = number(inputs[2]);
    if (minimum <= 0 && maximum >= 1) {
        outputs.push_back(inputs[0]);
        return;
    }
    outputs.push_back(adjust(std::get<Image>(inputs[0]), 4, minimum, maximum));
}

/* chainner:image:color_levels (adjustments/color_levels.py, native_adjustments.levels). */
void color_levels(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image &image = std::get<Image>(inputs[0]);
    unsigned mask = unsigned(std::get<bool>(inputs[3])) | unsigned(std::get<bool>(inputs[2])) << 1 |
                    unsigned(std::get<bool>(inputs[1])) << 2 | unsigned(std::get<bool>(inputs[4])) << 3;
    double gamma = number(inputs[7]);
    adjustable_shape(image);
    MutableImage out = MutableImage::allocate(image.height, image.width, image.channels);
    check(cn_levels_f32(image.data, out.data(), pixels(image), size_t(image.channels), 1, mask, float(number(inputs[5])),
                        float(number(inputs[6])), float(gamma > 0.001 ? gamma : 0.001), float(number(inputs[8])),
                        float(number(inputs[9]))));
    outputs.push_back(std::move(out).publish());
}

/* chainner:image:invert (adjustments/invert_color.py). */
void invert(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    outputs.push_back(adjust(std::get<Image>(inputs[0]), 2));
}

/* chainner:image:opacity (adjustments/opacity.py, native_adjustments.opacity). */
void opacity(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image &image = std::get<Image>(inputs[0]);
    double value = number(inputs[1]);
    if (value == 100 && image.channels == 4) {
        outputs.push_back(inputs[0]);
        return;
    }
    adjustable_shape(image);
    MutableImage out = MutableImage::allocate(image.height, image.width, 4);
    check(cn_opacity_f32(image.data, out.data(), pixels(image), size_t(image.channels), float(value / 100)));
    outputs.push_back(std::move(out).publish());
}

/* chainner:image:add (arithmetic/add.py). */
void add(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    double value = number(inputs[1]);
    if (value == 0) {
        outputs.push_back(inputs[0]);
        return;
    }
    outputs.push_back(adjust(std::get<Image>(inputs[0]), 0, value / 100));
}

/* chainner:image:divide (arithmetic/divide.py); the enforced divisor is at least 0.0001. */
void divide(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    double value = number(inputs[1]);
    if (value == 1.0) {
        outputs.push_back(inputs[0]);
        return;
    }
    outputs.push_back(adjust(std::get<Image>(inputs[0]), 1, 1.0 / value));
}

/* chainner:image:multiply (arithmetic/multiply.py). */
void multiply(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    double value = number(inputs[1]);
    if (value == 1.0) {
        outputs.push_back(inputs[0]);
        return;
    }
    outputs.push_back(adjust(std::get<Image>(inputs[0]), 1, value));
}

/* chainner:image:premultiplied_alpha (arithmetic/premultiplied_alpha.py): AlphaAssociation's two values. */
void premultiplied_alpha(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    bool unpremultiply = std::get<std::string>(inputs[1]) == "Premultiplied -> Straight";
    outputs.push_back(adjust(std::get<Image>(inputs[0]), unpremultiply ? 6 : 5));
}

/* chainner:image:gamma (gamma/gamma.py, native_alpha_gamma.fast_gamma); the enforced gamma is at least 0.01. */
void gamma(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    double value = number(inputs[1]);
    if (value == 1) {
        outputs.push_back(inputs[0]);
        return;
    }
    if (std::get<bool>(inputs[2])) value = 1 / value;
    const Image &image = std::get<Image>(inputs[0]);
    MutableImage out = MutableImage::allocate(image.height, image.width, image.channels);
    check(cn_alpha_gamma(image.data, out.data(), pixels(image), size_t(image.channels), float(value)));
    outputs.push_back(std::move(out).publish());
}

/* chainner:image:log2lin (gamma/log_to_linear.py, native_adjustments.log_linear): the offset's pow and the gain's
 * check (ValueError when black equals white) raise before the image is checked. */
void log_to_linear(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    double black = number(inputs[1]), white = number(inputs[2]), gamma = number(inputs[3]);
    double offset = python_pow(10.0, (black - white) * 0.002 / gamma);
    if (offset == 1.0) throw NodeError("ValueError", "Black and White must differ");
    double gain = 1.0 / (1.0 - offset);
    const Image &image = std::get<Image>(inputs[0]);
    adjustable_shape(image);
    MutableImage out = MutableImage::allocate(image.height, image.width, image.channels);
    check(cn_log_linear_f32(image.data, out.data(), image.values(), float(white), float(gamma), float(offset),
                            float(gain), float(0.002 / gamma), std::get<bool>(inputs[4]) ? 1 : 0));
    outputs.push_back(std::move(out).publish());
}

/* chainner:image:generate_threshold (threshold/generate_threshold.py, native_color_ops.auto_threshold); enforced
 * images have at most four channels, so the channel-mean path for more than seven never runs. */
void generate_threshold(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image &image = std::get<Image>(inputs[0]);
    adjustable_shape(image);
    int value = 0;
    check(cn_auto_threshold(image.data, pixels(image), size_t(image.channels), int(std::get<int64_t>(inputs[1])),
                            &value));
    outputs.push_back(double(value) / 255 * 100);
}

/* native_color_ops.linear: image * factor + offset (c_floats). */
Image linear(const Image &image, double factor, double offset)
{
    adjustable_shape(image);
    MutableImage out = MutableImage::allocate(image.height, image.width, image.channels);
    check(cn_color_linear(image.data, out.data(), image.values(), float(factor), float(offset)));
    return std::move(out).publish();
}

/* hue_and_saturation.with_lightness: towards white for a positive lightness, towards black for a negative one. */
Image with_lightness(const Image &image, double lightness)
{
    if (lightness > 0) return linear(image, 1 - lightness, lightness);
    if (lightness < 0) return linear(image, 1 + lightness, 0);
    return image;
}

/* chainner:image:hue_and_saturation (adjustments/hue_and_saturation.py): hue and saturation in HLS (cvt_color
 * BGR2HLS, native_color_ops.hls_adjust, HLS2BGR), then lightness; grey only takes the lightness; alpha passes
 * through. */
void hue_and_saturation(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image &input = std::get<Image>(inputs[0]);
    double hue = number(inputs[1]), saturation = number(inputs[2]) / 100, lightness = number(inputs[3]) / 100;
    if (hue == 0 && saturation == 0 && lightness == 0) {
        outputs.push_back(inputs[0]);
        return;
    }
    if (input.channels == 1) {
        outputs.push_back(with_lightness(input, lightness));
        return;
    }
    Image image = input;
    std::optional<Image> alpha;
    if (input.channels > 3) {
        alpha = select_channels(input, {3});
        image = select_channels(input, {0, 1, 2});
    }
    if (hue != 0 || saturation != 0) {
        Image hls = cvt_color(image, 6, 0, 3);
        MutableImage adjusted = MutableImage::allocate(hls.height, hls.width, 3);
        std::copy(hls.data, hls.data + hls.values(), adjusted.data());
        check(cn_hls_adjust(adjusted.data(), pixels(hls), float(hue), float(1 + saturation)));
        image = cvt_color(std::move(adjusted).publish(), 7, 0, 3);
    }
    if (lightness != 0) image = with_lightness(image, lightness);
    outputs.push_back(alpha ? with_alpha(image, *alpha) : image);
}

/* chainner:image:threshold (threshold/threshold.py): ThresholdType values 0-4 (cv2.THRESH_*); without
 * anti-aliasing native_threshold.hard_threshold (IPP's state), else the anti-aliased mask through
 * native_color_ops.threshold. */
void threshold(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image &image = std::get<Image>(inputs[0]);
    double value = number(inputs[1]) / 100, maximum = number(inputs[3]) / 100;
    int kind = int(std::get<int64_t>(inputs[2]));
    double smoothness = double(integer(inputs[5])) / 10;
    MutableImage out = MutableImage::allocate(image.height, image.width, image.channels);
    if (!std::get<bool>(inputs[4])) {
        adjustable_shape(image);
        check(cn_threshold_hard(image.data, out.data(), image.values(), kind, float(value), float(maximum),
                                dispatch::cv2_use_ipp() ? 1 : 0));
        outputs.push_back(std::move(out).publish());
        return;
    }
    if (image.height == 0 || image.width == 0) throw NodeError("ValueError", "Threshold requires a nonempty image");
    MutableImage binary = MutableImage::allocate(image.height, image.width, image.channels);
    check(cn_threshold_aa(image.data, binary.data(), size_t(image.height), size_t(image.width), size_t(image.channels),
                          float(value), float(smoothness)));
    check(cn_threshold_f32(image.data, binary.data(), out.data(), image.values(), kind, float(value), float(maximum)));
    outputs.push_back(std::move(out).publish());
}

/* chainner:image:threshold_adaptive (threshold/threshold_adaptive.py, native_color_ops.adaptive_threshold): the
 * uint8 image's box (AdaptiveMethod 0) or Gaussian (1) mean, cn_adaptive_apply; the uint8 result is normalized here
 * as the output enforce converts it. The maximum and the constant round as Python's round() (half to even). */
void threshold_adaptive(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image &image = std::get<Image>(inputs[0]);
    int kind = int(std::get<int64_t>(inputs[1]));
    double maximum = number(inputs[2]) / 100 * 255;
    int method = int(std::get<int64_t>(inputs[3]));
    int64_t radius = integer(inputs[4]);
    int delta = int(std::nearbyint(number(inputs[5]) / 100 * 255));
    adjustable_shape(image);
    std::vector<uint8_t> source = to_uint8(image, true), mean(source.size()), result(source.size());
    size_t height = size_t(image.height), width = size_t(image.width);
    if (method == 0) {
        if (radius > (0x7FFFFFFF - 1) / 2)
            throw NodeError("ValueError", "Adaptive mean radius exceeds the supported integer range");
        check(cn_box_mean_u8(source.data(), mean.data(), height, width, size_t(radius)));
    } else {
        check(cn_gaussian_mean_u8(source.data(), mean.data(), height, width, size_t(radius),
                                  dispatch::opencv_lanes("smooth")));
    }
    int maximum_u8 = int(std::max(0.0, std::min(255.0, std::nearbyint(maximum))));
    check(cn_adaptive_apply(source.data(), mean.data(), result.data(), image.values(), delta, kind, maximum_u8));
    MutableImage out = MutableImage::allocate(image.height, image.width, 1);
    check(cn_pixels_convert_checked(result.data(), out.data(), image.values(), 2, 0, 1, nullptr));
    outputs.push_back(std::move(out).publish());
}

/* native_transfer.contrast_bounds of a view of the input (its values in C order, and the view's NumPy reduce setup):
 * minimum and maximum (no percentile) or the percentile pair, NumPy's minimum/maximum loop lanes (1 for a strided
 * stream). */
std::pair<double, double> contrast_bounds(const Image &values, const ReduceSetup &setup, const double *percentile)
{
    auto lanes_of = [](std::string_view function) {
        std::string target = dispatch::text("nodes.impl.native_numpy_simd", "loop_target", {function, "fff"});
        return int(dispatch::number("nodes.impl.native_numpy_simd", "float32_lanes", {target}));
    };
    int lanes = lanes_of("minimum");
    if (lanes_of("maximum") != lanes)
        throw NodeError("RuntimeError", "NumPy dispatches minimum and maximum at different widths");
    if (!setup.contiguous) lanes = 1;
    double bounds[2] = {};
    check(cn_contrast_bounds_complete(values.data, values.values(), percentile ? 2 : 0, percentile ? *percentile : 0,
                                      lanes, setup.block, setup.period, bounds));
    return {bounds[0], bounds[1]};
}

/* native_transfer.stretch_range of float32 values with Python-float bounds (op 7, or op 1 by 0 for equal bounds). */
Image stretch_range(const Image &image, double minimum, double maximum)
{
    if (minimum > maximum) throw NodeError("ValueError", "min must be less than max");
    if (minimum == maximum) return std::get<Image>(adjust(image, 1, 0));
    return std::get<Image>(adjust(image, 7, minimum, maximum - minimum));
}

/* chainner:image:stretch_contrast (adjustments/stretch_contrast.py): StretchMode 0 auto, 1 percentile, 2 manual; the
 * colour channels together (keep colours, manual or grey) or one by one; alpha passes through. The bounds reduce
 * the views the body reduces (the colour channels of an RGBA image, a channel plane), with their NumPy setups. An
 * empty image takes NumPy's own reductions for the automatic modes, so it declines to the Python body. */
void stretch_contrast(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image &input = std::get<Image>(inputs[0]);
    int64_t mode = std::get<int64_t>(inputs[1]);
    bool keep_colors = std::get<bool>(inputs[2]);
    double percentile = number(inputs[3]), manual_min = number(inputs[4]), manual_max = number(inputs[5]);
    int64_t h = input.height, w = input.width, c = input.channels;
    auto range_of = [&](const Image &values, const ReduceSetup &setup) -> std::pair<double, double> {
        if (mode == 2) {
            if (manual_min > manual_max) throw NodeError("ValueError", "Minimum must be less than Maximum");
            return {manual_min / 255, manual_max / 255};
        }
        if (values.height == 0 || values.width == 0) throw RunPython{};
        return contrast_bounds(values, setup, mode == 1 ? &percentile : nullptr);
    };
    Image image = input;
    std::optional<Image> alpha;
    if (c == 4) {
        alpha = select_channels(input, {3});
        image = select_channels(input, {0, 1, 2});
    }
    // The colour view's element strides in the input (rows of w * c values, pixels of c).
    int64_t pixel = c * 4, row = w * pixel;
    if (keep_colors || mode == 2 || c == 1) {
        ReduceSetup setup = c == 1 ? reduce_setup({h, w}, {row, pixel}) : reduce_setup({h, w, 3}, {row, pixel, 4});
        auto [low, high] = range_of(image, setup);
        image = stretch_range(image, low, high);
    } else {
        std::vector<Image> channels;
        for (int channel = 0; channel < 3; ++channel) {
            Image plane = select_channels(image, {channel});
            auto [low, high] = range_of(plane, reduce_setup({h, w}, {row, pixel}));
            channels.push_back(stretch_range(plane, low, high));
        }
        std::vector<cn_channel_source> sources;
        for (const Image &plane : channels) sources.push_back({plane.data, 1, 0, 0.0f});
        MutableImage merged = MutableImage::allocate(image.height, image.width, 3);
        check(cn_channels_assemble(sources.data(), merged.data(), pixels(image), sources.size()));
        image = std::move(merged).publish();
    }
    outputs.push_back(alpha ? with_alpha(image, *alpha) : image);
}

}  // namespace

CN_RUN_NODES_DECLARE(image_adjustments)
{
    registry.add("chainner:image:add", add);
    registry.add("chainner:image:brightness_and_contrast", brightness_and_contrast);
    registry.add("chainner:image:clamp", clamp);
    registry.add("chainner:image:color_levels", color_levels);
    registry.add("chainner:image:divide", divide);
    registry.add("chainner:image:gamma", gamma);
    registry.add("chainner:image:generate_threshold", generate_threshold);
    registry.add("chainner:image:hue_and_saturation", hue_and_saturation);
    registry.add("chainner:image:invert", invert);
    registry.add("chainner:image:log2lin", log_to_linear);
    registry.add("chainner:image:multiply", multiply);
    registry.add("chainner:image:opacity", opacity);
    registry.add("chainner:image:premultiplied_alpha", premultiplied_alpha);
    registry.add("chainner:image:stretch_contrast", stretch_contrast);
    registry.add("chainner:image:threshold", threshold);
    registry.add("chainner:image:threshold_adaptive", threshold_adaptive);
}
