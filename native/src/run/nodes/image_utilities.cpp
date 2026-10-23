/* Native entry points of image_utility nodes other than Flip (image_utility.cpp), and View Image (pattern:
 * image_utility.cpp). */
#include "image_convolve.hpp"
#include "image_node_support.hpp"

#include "cn_crt_math.h"

#include <array>
#include <limits>

namespace {

using namespace cn::run;

/* utils.round_half_up(numerator / denominator) of Python ints: math.floor(x + 0.5) of the true quotient. */
int64_t round_half_up(int64_t numerator, int64_t denominator)
{
    return int64_t(std::floor(double(numerator) / double(denominator) + 0.5));
}

/* Python's a % b for b > 0. */
int64_t python_mod(int64_t a, int64_t b)
{
    int64_t r = a % b;
    return r < 0 ? r + b : r;
}

/* chainner:image:shift (modification/shift.py, native_channels.shift): ShiftFill -1 auto, 0 black, 1 transparent,
 * 2 wrap. */
void shift(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image &image = std::get<Image>(inputs[0]);
    int64_t h = image.height, w = image.width, dx, dy;
    if (std::get<int64_t>(inputs[1]) == 0) {
        dx = round_half_up(w * integer(inputs[2]), 100);
        dy = round_half_up(h * integer(inputs[3]), 100);
    } else {
        dx = integer(inputs[4]);
        dy = integer(inputs[5]);
    }
    int64_t fill = std::get<int64_t>(inputs[6]);
    nonempty(image);
    int32_t out_channels = image.channels;
    if (fill == 2) {
        dx = python_mod(dx, w), dy = python_mod(dy, h);
        if (dx == 0 && dy == 0) {
            outputs.push_back(inputs[0]);
            return;
        }
    } else {
        if (h >= 32767 || w >= 32767) throw NodeError("ValueError", "Unsupported affine shift dimensions or fill");
        // Translations farther away than this cannot sample the image.
        dx = std::max(-w - 1, std::min(w + 1, dx));
        dy = std::max(-h - 1, std::min(h + 1, dy));
        if (fill == 1) out_channels = 4;
    }
    MutableImage out = MutableImage::allocate(image.height, image.width, out_channels);
    check(cn_channels_shift(image.data, out.data(), size_t(h), size_t(w), size_t(image.channels), dx, dy, int(fill)));
    outputs.push_back(std::move(out).publish());
}

/* Color.gray/bgr/bgra's _norm, max(0, min(float(n), 1)), into a colour channel: min keeps the float unless 1 is
 * smaller (then int 1), max keeps that unless it is not above 0 (then int 0: 0.0, -0.0 and NaN become int 0). */
void push_color_norm(Color &color, float value)
{
    double x = double(value);
    bool integral = 1 < x;
    double low = integral ? 1.0 : x;
    if (!(low > 0)) {
        low = 0.0;
        integral = true;
    }
    color.values.push_back(low);
    color.integral.push_back(integral);
}

/* chainner:image:pick_color (miscellaneous/pick_color.py): CoordinateType 1 relative, 2 absolute. */
void pick_color(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image &image = std::get<Image>(inputs[0]);
    int64_t h = image.height, w = image.width, x, y;
    if (std::get<int64_t>(inputs[1]) == 1) {
        x = round_half_up((w - 1) * integer(inputs[2]), 100);
        y = round_half_up((h - 1) * integer(inputs[3]), 100);
    } else {
        x = integer(inputs[4]);
        y = integer(inputs[5]);
    }
    if (!(x < w && y < h))
        throw NodeError("AssertionError", "The given coordinates (X, Y) are outside the image. Expect X=" +
                                              std::to_string(x) + " to be to 0 to " + std::to_string(w - 1) +
                                              " and Y=" + std::to_string(y) + " to be to 0 to " +
                                              std::to_string(h - 1) + ".");
    // orig_img[y, x]: NumPy's negative indices count from the end; an empty axis has no valid index.
    for (auto [index, size, axis] : {std::array<int64_t, 3>{y, h, 0}, std::array<int64_t, 3>{x, w, 1}})
        if (index < -size)
            throw NodeError("IndexError", "index " + std::to_string(index) + " is out of bounds for axis " +
                                              std::to_string(axis) + " with size " + std::to_string(size));
    if (y < 0) y += h;
    if (x < 0) x += w;
    const float *pixel = image.data + (size_t(y) * size_t(w) + size_t(x)) * size_t(image.channels);
    Color color;
    for (int32_t c = 0; c < image.channels; ++c) push_color_norm(color, pixel[c]);
    outputs.push_back(std::move(color));
}

/* chainner:image:stack (compositing/stack_images.py, native_layout.stack): images scale (nearest) to the largest
 * height (horizontal) or width (vertical); grey becomes three channels; an image with a narrower channel count
 * broadcasts a single row or column to the largest height and width, as NumPy's assignment into the maxima does,
 * and otherwise keeps its own size (upstream failed to broadcast it). */
void stack(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    bool horizontal = std::get<std::string>(inputs[0]) == "horizontal";
    std::vector<const Image *> items;
    for (const Value &input : inputs.subspan(1))
        if (const Image *image = std::get_if<Image>(&input)) items.push_back(image);
    for (const Image *image : items) nonempty(*image);
    int64_t max_h = 0, max_w = 0, max_c = 0;
    for (const Image *image : items) {
        max_h = std::max<int64_t>(max_h, image->height);
        max_w = std::max<int64_t>(max_w, image->width);
        max_c = std::max<int64_t>(max_c, image->channels == 1 ? 3 : image->channels);
    }
    std::vector<std::pair<int64_t, int64_t>> targets;
    for (const Image *image : items) {
        int64_t h = image->height, w = image->width, th = h, tw = w;
        if (horizontal && h < max_h) {
            th = max_h, tw = int64_t(std::floor(double(w * max_h) / double(h) + 0.5));
        } else if (!horizontal && w < max_w) {
            th = int64_t(std::floor(double(h * max_w) / double(w) + 0.5)), tw = max_w;
        }
        int64_t expanded = image->channels == 1 ? 3 : image->channels;
        if (expanded < max_c && (th == 1 || th == max_h) && (tw == 1 || tw == max_w)) th = max_h, tw = max_w;
        targets.emplace_back(th, tw);
    }
    int64_t out_h = horizontal ? max_h : 0, out_w = horizontal ? 0 : max_w;
    for (auto [th, tw] : targets) (horizontal ? out_w : out_h) += horizontal ? tw : th;
    if (out_h > std::numeric_limits<int32_t>::max() || out_w > std::numeric_limits<int32_t>::max()) throw RunPython{};
    MutableImage out = MutableImage::allocate(int32_t(out_h), int32_t(out_w), int32_t(max_c));
    int64_t top = 0, left = 0;
    for (size_t i = 0; i < items.size(); ++i) {
        const Image &image = *items[i];
        auto [th, tw] = targets[i];
        check(cn_layout_paste(image.data, out.data(), size_t(image.height), size_t(image.width),
                              size_t(image.channels), size_t(out_h), size_t(out_w), size_t(max_c), size_t(top),
                              size_t(left), size_t(th), size_t(tw), 1));
        (horizontal ? left : top) += horizontal ? tw : th;
    }
    outputs.push_back(std::move(out).publish());
}

/* chainner:image:z_stack (compositing/z_stack_images.py, native_layout.z_stack): two required images and up to
 * twelve more, all of one shape. */
void z_stack(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const std::string &expression = std::get<std::string>(inputs[0]);
    std::vector<const Image *> items;
    for (const Value &input : inputs.subspan(1))
        if (const Image *image = std::get_if<Image>(&input)) items.push_back(image);
    for (const Image *image : items) nonempty(*image);
    const Image &first = *items[0];
    for (const Image *image : items)
        if (image->height != first.height || image->width != first.width || image->channels != first.channels)
            throw NodeError("AssertionError", "All images must have the same dimensions and channels");
    int mode = expression == "mean" ? 0 : expression == "median" ? 1 : expression == "minimum" ? 2 : 3;
    std::vector<const float *> sources;
    for (const Image *image : items) sources.push_back(image->data);
    MutableImage out = MutableImage::allocate(first.height, first.width, first.channels);
    check(cn_layout_zstack(sources.data(), out.data(), first.values(), sources.size(), mode));
    outputs.push_back(std::move(out).publish());
}

/* native_analysis._image: a nonempty image. */
void analysis_image(const Image &image)
{
    if (image.height == 0 || image.width == 0) throw NodeError("ValueError", "Invalid image shape " + shape_text(image));
}

/* chainner:image:lut (miscellaneous/apply_palette.py, native_analysis.apply_palette): the palette's first row. */
void apply_palette(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image &image = std::get<Image>(inputs[0]);
    const Image &palette = std::get<Image>(inputs[1]);
    analysis_image(image);
    analysis_image(palette);
    if (palette.width > (1 << 24))
        throw NodeError("AssertionError", "Quantizing float32 values with more than 2**24 levels doesn't make sense, "
                                          "because only integers up to 2**24 can be represented exactly using float32.");
    MutableImage out = MutableImage::allocate(image.height, image.width, palette.channels);
    cn_status status = cn_analysis_palette(image.data, palette.data, out.data(), image.values(), size_t(palette.width),
                                           size_t(palette.channels));
    if (status == CN_INVALID_ARGUMENT)
        throw NodeError("IndexError", "Quantized palette index is out of bounds for axis 0");
    check(status);
    outputs.push_back(std::move(out).publish());
}

/* chainner:image:image_statistics (miscellaneous/image_statistics.py, native_analysis.statistics): a C-ordered
 * image is one contiguous stream (numpy_reduce: no blocks, the SIMD min/max loop of NumPy's dispatched target);
 * each statistic is round(value * 255, 4). */
void image_statistics(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image &image = std::get<Image>(inputs[0]);
    double percentile = number(inputs[1]);
    analysis_image(image);
    std::string target = dispatch::text("nodes.impl.native_numpy_simd", "loop_target", {"minimum", "fff"});
    int lanes = int(dispatch::number("nodes.impl.native_numpy_simd", "float32_lanes", {target}));
    std::array<double, 4> result{};
    check(cn_analysis_statistics(image.data, image.data, image.values(), percentile, lanes, 0, 0, 1, result.data()));
    for (double value : result) outputs.push_back(python_round(value * 255, 4));
}

/* FillColor.get_color(channels) scaled to bytes: -1 auto zeros, 0 black (opaque for four channels), 1 transparent. */
std::vector<uint8_t> fill_bytes(int64_t fill, int32_t channels)
{
    if (fill == 1) return {0, 0, 0, 0};
    std::vector<uint8_t> bytes(size_t(channels), 0);
    if (fill == 0 && channels >= 4) bytes = {0, 0, 0, 255};
    return bytes;
}

/* Python's float % 360 (the result takes the divisor's sign; zero is +0.0). */
double mod_360(double angle)
{
    double mod = std::fmod(angle, 360.0);
    if (mod != 0) return mod < 0 ? mod + 360.0 : mod;
    return 0.0;
}

/* native_rotate.rotate's transform for an angle that is not a right angle it rotates exactly (Pillow 9.2.0's
 * Image.rotate): the matrix and the output size. */
struct Affine {
    std::array<double, 6> matrix;
    int64_t out_w, out_h;
};

Affine affine(double angle, int64_t w, int64_t h, bool expand)
{
    double radians = -(mod_360(angle) * (3.141592653589793 / 180.0));
    std::array<double, 6> m{python_round(cn_crt.cos(radians), 15), python_round(cn_crt.sin(radians), 15), 0.0,
                            python_round(-cn_crt.sin(radians), 15), python_round(cn_crt.cos(radians), 15), 0.0};
    auto transform = [&m](double x, double y) {
        return std::pair{m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5]};
    };
    std::tie(m[2], m[5]) = transform(-double(w) / 2.0, -double(h) / 2.0);
    m[2] += double(w) / 2.0;
    m[5] += double(h) / 2.0;
    int64_t out_w = w, out_h = h;
    if (expand) {
        std::array<std::pair<double, double>, 4> corners{transform(0, 0), transform(double(w), 0),
                                                         transform(double(w), double(h)), transform(0, double(h))};
        double min_x = corners[0].first, max_x = min_x, min_y = corners[0].second, max_y = min_y;
        for (auto [x, y] : corners) {
            min_x = std::min(min_x, x), max_x = std::max(max_x, x);
            min_y = std::min(min_y, y), max_y = std::max(max_y, y);
        }
        out_w = int64_t(std::ceil(max_x)) - int64_t(std::floor(min_x));
        out_h = int64_t(std::ceil(max_y)) - int64_t(std::floor(min_y));
        std::tie(m[2], m[5]) = transform(-double(out_w - w) / 2.0, -double(out_h - h) / 2.0);
    }
    return {m, out_w, out_h};
}

/* chainner:image:rotate (modification/rotate.py): right angles through native_layout.rotate_quarters, any other
 * angle through pil_utils.rotate (native_rotate.rotate's Pillow-compatible affine sampling of the uint8 image;
 * a transparent fill first converts the image to BGRA). RotationInterpolationMethod 3 cubic, 2 linear, 0 nearest;
 * RotateSizeChange 1 expand, 0 crop; FillColor as fill_bytes. */
void rotate(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image &input = std::get<Image>(inputs[0]);
    double angle = number(inputs[1]);
    int interpolation = int(std::get<int64_t>(inputs[2]));
    bool expand = std::get<int64_t>(inputs[3]) == 1;
    int64_t fill = std::get<int64_t>(inputs[4]);
    double normalized = mod_360(angle);
    int64_t h = input.height, w = input.width;
    if (normalized == 0 || normalized == 180 || ((normalized == 90 || normalized == 270) && (expand || h == w))) {
        nonempty(input);
        int quarter = int(normalized / 90);
        int32_t out_c = fill == 1 ? 4 : input.channels;
        std::vector<uint8_t> source = to_uint8(input, true);
        bool odd = quarter % 2 == 1;
        MutableImage out = MutableImage::allocate(odd ? input.width : input.height, odd ? input.height : input.width,
                                                  out_c);
        check(cn_layout_rotate_u8(source.data(), out.data(), size_t(h), size_t(w), size_t(input.channels),
                                  size_t(out_c), quarter));
        outputs.push_back(std::move(out).publish());
        return;
    }
    std::vector<uint8_t> color = fill_bytes(fill, input.channels);
    Image image = input;
    if (fill == 1 && input.channels != 4) image = cvt_color(input, 0, 0, 4);
    std::vector<uint8_t> source = to_uint8(image, true);
    if (h == 0 || w == 0) throw NodeError("ValueError", "Expected a nonempty image");
    Affine transform = affine(angle, w, h, expand);
    for (double value : transform.matrix)
        if (!std::isfinite(value)) throw NodeError("ValueError", "Rotation transform must be finite");
    if (std::max(transform.out_h, transform.out_w) > 2147483647)
        throw NodeError("OverflowError", "Rotation output exceeds the affine coordinate range");
    MutableImage out = MutableImage::allocate(int32_t(transform.out_h), int32_t(transform.out_w), image.channels);
    check(cn_rotate_affine_u8(source.data(), out.data(), size_t(h), size_t(w), size_t(image.channels),
                              size_t(transform.out_h), size_t(transform.out_w), transform.matrix.data(), interpolation,
                              color.data(), -1));
    outputs.push_back(std::move(out).publish());
}

/* chainner:image:inpaint (miscellaneous/inpaint.py, native_repair.inpaint): InpaintAlgorithm 0 Navier-Stokes,
 * 1 Telea on the uint8 image and mask; the uint8 result is normalized here as the output enforce converts it. */
void inpaint(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image &image = std::get<Image>(inputs[0]);
    const Image &mask = std::get<Image>(inputs[1]);
    int method = int(std::get<int64_t>(inputs[2]));
    double radius = number(inputs[3]);
    if (image.height != mask.height || image.width != mask.width)
        throw NodeError("AssertionError", "Input image and mask must have the same resolution");
    std::vector<uint8_t> source = to_uint8(image, true), stencil = to_uint8(mask, true);
    if (image.height == 0 || image.width == 0)
        throw NodeError("ValueError",
                        "Inpaint requires matching nonempty gray/BGR image and gray mask, using NS or Telea");
    std::vector<uint8_t> result(image.values());
    check(cn_inpaint_u8(source.data(), stencil.data(), result.data(), size_t(image.height), size_t(image.width),
                        size_t(image.channels), radius, method));
    MutableImage out = MutableImage::allocate(image.height, image.width, image.channels);
    check(cn_pixels_convert_checked(result.data(), out.data(), result.size(), 2, 0, 1, nullptr));
    outputs.push_back(std::move(out).publish());
}

/* native_analysis.mean of a fresh C-ordered image (NumPy's float32 mean, one contiguous run). */
float analysis_mean(const Image &image)
{
    float result = 0;
    check(cn_analysis_mean(image.data, image.values(), 0, 0, &result));
    return result;
}

/* native_analysis.calculate_ssim of two grey images: the 11 x 11 Gaussian window's spatial convolutions of the
 * images, their squares and product, cropped by 5 on every side, then the SSIM map's mean. */
double calculate_ssim(const Image &image, const Image &other)
{
    analysis_image(image);
    analysis_image(other);
    Kernel window{11, 11, std::vector<double>(121)};
    check(cn_ssim_window(window.values.data()));
    std::array<Image, 5> sources{image, other, analysis_binary(image, image, 3), analysis_binary(other, other, 3),
                                 analysis_binary(image, other, 3)};
    std::array<Image, 5> maps;
    for (size_t i = 0; i < sources.size(); ++i)
        maps[i] = std::get<Image>(slice(convolve(sources[i], window, 0), 5, -5, 5, -5));
    MutableImage out = MutableImage::allocate(maps[0].height, maps[0].width, 1);
    check(cn_analysis_ssim(maps[0].data, maps[1].data, maps[2].data, maps[3].data, maps[4].data, out.data(),
                           maps[0].values()));
    return double(analysis_mean(std::move(out).publish()));
}

/* round(np.float32, 6): NumPy's float32 multiply by 1e6, rint (half to even) and divide. */
float numpy_round6(float value)
{
    return std::nearbyint(value * 1000000.0f) / 1000000.0f;
}

/* chainner:image:image_metrics (miscellaneous/image_metrics.py): MSE, PSNR and SSIM of the luma (cvt_color
 * BGR2YCrCb's first channel) or of the grey images. */
void image_metrics(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    Image original = std::get<Image>(inputs[0]), comparison = std::get<Image>(inputs[1]);
    if (original.height != comparison.height || original.width != comparison.width ||
        original.channels != comparison.channels)
        throw NodeError("AssertionError", "Images must have same dimensions and color depth");
    if (original.channels > 1) {
        original = select_channels(cvt_color(original, 8, 0, 3), {0});
        comparison = select_channels(cvt_color(comparison, 8, 0, 3), {0});
    }
    float mse = analysis_mean(analysis_binary(comparison, original, 2));
    double reciprocal = 1.0 / double(mse);
    if (reciprocal == 0) throw NodeError("ValueError", "expected a positive input, got 0.0");
    double psnr = mse == 0 ? std::numeric_limits<double>::infinity() : 10 * std::log10(reciprocal);
    double ssim = python_round(calculate_ssim(comparison, original), 6);
    outputs.push_back(double(numpy_round6(mse)));
    outputs.push_back(python_round(psnr, 6));
    outputs.push_back(ssim);
}

/* nodes/impl/color: the colour spaces by id (convert_data.py), their detectors and conversions. */
struct ColorSpace {
    const char *name;
    int32_t channels;
};
constexpr ColorSpace color_spaces[] = {
    {"Gray", 1}, {"RGB", 3},  {"RGBA", 4}, {"YUV", 3},     {"HSV", 3},       {"HSL", 3},           {"CMYK", 4},
    {"YUVA", 4}, {"HSVA", 4}, {"HSLA", 4}, {"L*a*b*", 3}, {"L*a*b*A", 4}, {"L*C*h\xC2\xB0", 3}, {"L*C*h\xC2\xB0" "A", 4},
};
enum : int { gray, rgb, rgba, yuv, hsv, hsl, cmyk, yuva, hsva, hsla, lab, laba, lch, lcha };

/* utils.format.format_image_with_channels(channels) (conj "and", singular). */
std::string image_with_channels(const std::vector<int32_t> &channels)
{
    auto named = [](int32_t c) { return c == 1 ? "grayscale" : c == 3 ? "RGB" : c == 4 ? "RGBA" : nullptr; };
    bool all_named = std::all_of(channels.begin(), channels.end(), [&](int32_t c) { return named(c) != nullptr; });
    std::vector<std::string> words;
    for (int32_t c : channels) words.push_back(all_named ? named(c) : std::to_string(c));
    std::string joined = words.size() == 1   ? words[0]
                         : words.size() == 2 ? words[0] + " and " + words[1]
                                             : words[0] + ", " + words[1] + ", and " + words[2];
    return all_named ? "a " + joined + " image" : "an image with " + joined + " channel(s)";
}

/* native_transparency.representation(image, operation): CMYK (0, 1), HSV (2, 3), HSL (4, 5) and Lab (6, 7)
 * representations of three-channel images (four for operation 1). */
Image representation(const Image &image, int operation)
{
    if (image.channels != (operation == 1 ? 4 : 3) || image.height == 0 || image.width == 0)
        throw NodeError("ValueError", "Invalid image shape for transparency/color kernel");
    MutableImage out = MutableImage::allocate(image.height, image.width, operation == 0 ? 4 : 3);
    check(cn_color_representation(image.data, out.data(), pixels(image), operation, nullptr, nullptr, nullptr, nullptr));
    return std::move(out).publish();
}

/* native_color_complete.lab_lch: NumPy's sin/cos arithmetic, in its FMA3 form where NumPy's sin loop dispatches
 * FMA3. */
Image lab_lch(const Image &image, bool inverse)
{
    if (image.channels != 3 || image.height == 0 || image.width == 0)
        throw NodeError("ValueError", "Lab/LCH conversion requires a nonempty three-channel image");
    std::string target = dispatch::text("nodes.impl.native_numpy_simd", "loop_target", {"sin", "ff"});
    int fma = int(dispatch::number("nodes.impl.native_numpy_simd", "fma3", {target}));
    MutableImage out = MutableImage::allocate(image.height, image.width, 3);
    check(cn_color_convert_f32(image.data, out.data(), size_t(image.height), size_t(image.width), 3, 3,
                               inverse ? 13 : 12, 0, 4, fma));
    return std::move(out).publish();
}

/* Single-channel images merged in order (native_channels.merge_channels of three planes). */
Image merge3(const Image &a, const Image &b, const Image &c)
{
    std::vector<cn_channel_source> sources{{a.data, 1, 0, 0.0f}, {b.data, 1, 0, 0.0f}, {c.data, 1, 0, 0.0f}};
    MutableImage out = MutableImage::allocate(a.height, a.width, 3);
    check(cn_channels_assemble(sources.data(), out.data(), pixels(a), sources.size()));
    return std::move(out).publish();
}

/* convert_data's float32 conversions. cvt_color codes as (operation, blue, channels): BGR2RGB (0, 2, 3),
 * BGR2YUV (2, 0, 3), YUV2BGR (3, 0, 3), BGR2HSV (4, 0, 3), HSV2BGR (5, 0, 3), BGR2HLS (6, 0, 3), HLS2BGR (7, 0, 3),
 * BGR2LAB (10, 0, 3), LAB2BGR (11, 0, 3). */
Image rev3(const Image &image) { return cvt_color(image, 0, 2, 3); }
Image hsv_to_rgb(const Image &image) { return cvt_color(representation(image, 3), 5, 0, 3); }
Image hsl_to_rgb(const Image &image) { return cvt_color(representation(image, 5), 7, 0, 3); }

Image yuv_to_rgb(const Image &image)
{
    Image result = cvt_color(rev3(image), 3, 0, 3);
    MutableImage out = MutableImage::allocate(result.height, result.width, 3);
    check(cn_adjust_f32(result.data, out.data(), pixels(result), 3, 1, 4, 0.0f, 1.0f, 0));
    return std::move(out).publish();
}

Image hsv_to_hsl(const Image &image)
{
    Image h = select_channels(image, {2});
    Image hls = cvt_color(hsv_to_rgb(image), 6, 0, 3);
    return merge3(select_channels(hls, {1}), select_channels(hls, {2}), h);
}

Image hsl_to_hsv(const Image &image)
{
    Image h = select_channels(image, {2});
    Image hsv = cvt_color(hsl_to_rgb(image), 4, 0, 3);
    return merge3(select_channels(hsv, {2}), select_channels(hsv, {1}), h);
}

/* A conversion: its function, or for a generated XA -> YA conversion the index of its X -> Y source. */
struct Conversion {
    int input, output, cost;
    Image (*convert)(const Image &);
    size_t source = 0;
};

/* convert_data.conversions in its order: the listed ones, the alpha pairs' additions and removals, then XA -> YA for
 * every X -> Y between alpha-pair spaces (applied to the colour channels; alpha passes through). */
const std::vector<Conversion> &conversions()
{
    static const std::vector<Conversion> list = [] {
        constexpr int lost = 1000, chroma = 100;
        std::vector<Conversion> all{
            {rgb, gray, lost * 2, [](const Image &i) { return cvt_color(i, 1, 0, 1); }},
            {gray, rgb, 1, [](const Image &i) { return cvt_color(i, 0, 0, 3); }},
            {rgba, gray, lost * 3, [](const Image &i) { return cvt_color(i, 1, 0, 1); }},
            {gray, rgba, 1, [](const Image &i) { return cvt_color(i, 0, 0, 4); }},
            {rgb, yuv, 1, [](const Image &i) { return rev3(cvt_color(i, 2, 0, 3)); }},
            {yuv, rgb, chroma, yuv_to_rgb},
            {rgb, hsv, 1, [](const Image &i) { return representation(cvt_color(i, 4, 0, 3), 2); }},
            {hsv, rgb, chroma, hsv_to_rgb},
            {rgb, hsl, 1, [](const Image &i) { return representation(cvt_color(i, 6, 0, 3), 4); }},
            {hsl, rgb, chroma, hsl_to_rgb},
            {hsv, hsl, 1, hsv_to_hsl},
            {hsl, hsv, 1, hsl_to_hsv},
            {rgb, cmyk, 1, [](const Image &i) { return representation(i, 0); }},
            {cmyk, rgb, chroma, [](const Image &i) { return representation(i, 1); }},
            {rgb, lab, 1, [](const Image &i) { return representation(cvt_color(i, 10, 0, 3), 6); }},
            {lab, rgb, 1, [](const Image &i) { return cvt_color(representation(i, 7), 11, 0, 3); }},
            {lab, lch, 1, [](const Image &i) { return lab_lch(i, false); }},
            {lch, lab, chroma, [](const Image &i) { return lab_lch(i, true); }},
        };
        constexpr std::array<std::pair<int, int>, 6> pairs{
            {{rgb, rgba}, {yuv, yuva}, {hsv, hsva}, {hsl, hsla}, {lab, laba}, {lch, lcha}}};
        for (auto [three, four] : pairs) {
            all.push_back({three, four, 1, [](const Image &i) { return cvt_color(i, 0, 0, 4); }});
            all.push_back({four, three, lost, [](const Image &i) { return cvt_color(i, 0, 0, 3); }});
        }
        auto partner = [&](int space) {
            for (auto [three, four] : pairs)
                if (three == space) return four;
            return -1;
        };
        size_t count = all.size();
        for (size_t i = 0; i < count; ++i) {
            int input = partner(all[i].input), output = partner(all[i].output);
            if (input >= 0 && output >= 0) all.push_back({input, output, all[i].cost, nullptr, i});
        }
        return all;
    }();
    return list;
}

/* convert.get_shortest_path: Dijkstra over the conversions with Python's tie-breaking (the first cheapest entry of
 * the insertion-ordered front; a cheaper path to a queued space keeps its place). The path as the conversions it
 * takes (one per pair of spaces); every space is reachable. */
std::vector<size_t> shortest_path(int start, int destination)
{
    struct Item {
        int space;
        int cost;
        std::vector<size_t> path;
    };
    std::vector<Item> front{{start, 0, {}}};
    std::vector<bool> processed(std::size(color_spaces), false);
    while (!front.empty()) {
        size_t best = 0;
        for (size_t i = 1; i < front.size(); ++i)
            if (front[i].cost < front[best].cost) best = i;
        Item current = std::move(front[best]);
        front.erase(front.begin() + std::ptrdiff_t(best));
        processed[size_t(current.space)] = true;
        if (current.space == destination) return current.path;
        const std::vector<Conversion> &all = conversions();
        for (size_t index = 0; index < all.size(); ++index) {
            const Conversion &conversion = all[index];
            if (conversion.input != current.space) continue;
            int total = current.cost + conversion.cost;
            auto old = std::find_if(front.begin(), front.end(),
                                    [&](const Item &item) { return item.space == conversion.output; });
            std::vector<size_t> path = current.path;
            path.push_back(index);
            if (old == front.end()) {
                if (!processed[size_t(conversion.output)]) front.push_back({conversion.output, total, std::move(path)});
            } else if (old->cost > total) {
                old->cost = total;
                old->path = std::move(path);
            }
        }
    }
    return {};
}

/* A conversion applied (a generated XA -> YA conversion converts the colour channels with its X -> Y source and
 * appends the alpha). */
Image convert_step(const Image &image, size_t index)
{
    const Conversion &conversion = conversions()[index];
    if (conversion.convert) return conversion.convert(image);
    Image color = select_channels(image, {0, 1, 2}), alpha = select_channels(image, {3});
    return with_alpha(convert_step(color, conversion.source), alpha);
}

/* chainner:image:change_colorspace (miscellaneous/change_color_model.py, color/convert.convert): the input space or
 * detector (ids 1000-1005 pick by channel count), the output space or, with alpha, its alpha partner; the shortest
 * conversion path, step by step. */
void change_color_model(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image &image = std::get<Image>(inputs[0]);
    int64_t input = std::get<int64_t>(inputs[1]);
    int output = int(std::get<int64_t>(inputs[2]));
    if (std::get<bool>(inputs[3])) {
        static constexpr int partners[] = {-1, rgba, -1, yuva, hsva, hsla, -1, -1, -1, -1, laba, -1, lcha, -1};
        if (partners[output] >= 0) output = partners[output];
    }
    int from;
    if (input >= 1000) {
        static const std::vector<std::vector<int>> detectors{{gray, rgb, rgba}, {yuv, yuva}, {hsv, hsva},
                                                             {hsl, hsla},       {lab, laba}, {lch, lcha}};
        static constexpr const char *names[] = {"RGB", "YUV", "HSV", "HSL", "L*a*b*", "L*C*h\xC2\xB0"};
        const std::vector<int> &candidates = detectors[size_t(input - 1000)];
        auto found = std::find_if(candidates.begin(), candidates.end(),
                                  [&](int space) { return color_spaces[space].channels == image.channels; });
        if (found == candidates.end()) {
            std::vector<int32_t> channels;
            for (int space : candidates) channels.push_back(color_spaces[space].channels);
            throw NodeError("ValueError", std::string("Expected the input image for ") + names[input - 1000] +
                                              " to be " + image_with_channels(channels) + " but found " +
                                              image_with_channels({image.channels}) + ".");
        }
        from = *found;
    } else {
        from = int(input);
    }
    if (image.channels != color_spaces[from].channels)
        throw NodeError("ValueError", std::string("Expected the input image for a ") + color_spaces[from].name + " -> " +
                                          color_spaces[output].name + " conversion to be " +
                                          image_with_channels({color_spaces[from].channels}) + " but found " +
                                          image_with_channels({image.channels}) + ".");
    if (from == output) {
        outputs.push_back(inputs[0]);
        return;
    }
    Image result = image;
    for (size_t index : shortest_path(from, output)) result = convert_step(result, index);
    outputs.push_back(result);
}

/* chainner:image:view (image/io/view_image.py): returns its input; the engine's LargeImageOutput broadcasts it. */
void view_image(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    outputs.push_back(inputs[0]);
}

}  // namespace

CN_RUN_NODES_DECLARE(image_utilities)
{
    registry.add("chainner:image:change_colorspace", change_color_model);
    registry.add("chainner:image:image_metrics", image_metrics);
    registry.add("chainner:image:image_statistics", image_statistics);
    registry.add("chainner:image:inpaint", inpaint);
    registry.add("chainner:image:lut", apply_palette);
    registry.add("chainner:image:pick_color", pick_color);
    registry.add("chainner:image:rotate", rotate);
    registry.add("chainner:image:shift", shift);
    registry.add("chainner:image:stack", stack);
    registry.add("chainner:image:view", view_image);
    registry.add("chainner:image:z_stack", z_stack);
}
