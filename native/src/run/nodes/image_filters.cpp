/* Native entry points of image_filter nodes outside the benchmark set (pattern: image_utility.cpp). */
#include "image_blur.hpp"
#include "image_convolve.hpp"
#include "image_morphology.hpp"
#include "image_node_support.hpp"

#include "chainner_ext_kernels.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <optional>
#include <string_view>

namespace {

using namespace cn::run;

/* chainner:image:add_noise (noise/add_noise.py, nodes/impl/noise.py, native_noise.add_noise): NoiseType values as the
 * kernel's modes, NoiseColor rgb or gray. An empty image takes noise.py's NumPy fallback, so it declines to the
 * Python body. */
void add_noise(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image &image = std::get<Image>(inputs[0]);
    const std::string &type = std::get<std::string>(inputs[1]);
    int noise_channels = std::get<std::string>(inputs[2]) == "rgb" ? 3 : 1;
    double amount = number(inputs[3]) / 100.0;
    int mode = type == "gaussian" ? 0 : type == "uniform" ? 1 : type == "salt_and_pepper" ? 2 : type == "poisson" ? 3 : 4;
    if (image.height == 0 || image.width == 0) throw RunPython{};
    std::vector<uint32_t> words = seed_words(inputs[4]);
    if (std::signbit(amount) && (mode == 0 || mode == 1 || mode == 4))
        throw NodeError("ValueError", mode == 1 ? "high - low < 0" : "scale < 0");
    MutableImage out = MutableImage::allocate(image.height, image.width, std::max(image.channels, noise_channels));
    check(cn_add_seeded_noise(image.data, pixels(image), size_t(image.channels), size_t(noise_channels), mode, amount,
                              words.data(), words.size(), out.data()));
    outputs.push_back(std::move(out).publish());
}

/* chainner:image:pixelate (miscellaneous/pixelate.py, native_layout.pixelate); the enforced block sizes are within
 * [1, 1024]. */
void pixelate(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image &image = std::get<Image>(inputs[0]);
    nonempty(image);
    MutableImage out = MutableImage::allocate(image.height, image.width, image.channels);
    check(cn_layout_pixelate(image.data, out.data(), size_t(image.height), size_t(image.width), size_t(image.channels),
                             size_t(integer(inputs[1])), size_t(integer(inputs[2]))));
    outputs.push_back(std::move(out).publish());
}

/* chainner:image:sharpen (sharpen/unsharp_mask.py): native_unsharp.weighted (OpenCV's addWeighted on the "arithm"
 * family's lanes; C-ordered operands form one row) without a threshold, else native_analysis.binary. */
void unsharp_mask(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image &image = std::get<Image>(inputs[0]);
    double radius = number(inputs[1]), amount = number(inputs[2]);
    if (radius == 0 || amount == 0) {
        outputs.push_back(inputs[0]);
        return;
    }
    Image blurred = fast_gaussian_blur(image, radius, radius);
    double threshold = number(inputs[3]) / 100;
    if (threshold != 0) {
        outputs.push_back(analysis_binary(image, blurred, 1, amount, threshold));
        return;
    }
    MutableImage out = MutableImage::allocate(image.height, image.width, image.channels);
    check(cn_unsharp_weighted(image.data, blurred.data, out.data(), image.values(), image.values(), amount,
                              int(dispatch::opencv_lanes("arithm"))));
    outputs.push_back(std::move(out).publish());
}

/* chainner:image:high_pass (miscellaneous/high_pass.py): BlurMode 0 Gaussian (fast_gaussian_blur), 1 a given blurred
 * image; alpha passes through. */
void high_pass(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image &input = std::get<Image>(inputs[0]);
    Image image = input;
    std::optional<Image> alpha;
    if (input.channels > 3) {
        alpha = select_channels(input, {3});
        image = select_channels(input, {0, 1, 2});
    }
    Image blurred;
    if (std::get<int64_t>(inputs[1]) == 0) {
        double radius = number(inputs[2]);
        blurred = fast_gaussian_blur(image, radius, radius);
    } else {
        const Image *given = std::get_if<Image>(&inputs[3]);
        if (!given) throw NodeError("AssertionError", "Expected a blurred image to be given.");
        blurred = input.channels > 3 && given->channels > 3 ? select_channels(*given, {0, 1, 2}) : *given;
    }
    if (blurred.height != image.height || blurred.width != image.width || blurred.channels != image.channels)
        throw NodeError("AssertionError", "Expected blurred image to have same shape as the input image.");
    Image result = analysis_binary(image, blurred, 0, number(inputs[4]), 0);
    outputs.push_back(alpha ? with_alpha(result, *alpha) : result);
}

/* float() of a decimal token (parse_kernel's syntax) beyond the doubles' range, where from_chars leaves its value
 * unset: +-inf when the place of its leading nonzero digit plus its exponent is 0 or more (an overflow), else +-0 (an
 * underflow). */
double beyond_doubles(std::string_view token)
{
    size_t e = token.find_first_of("eE");
    std::string_view mantissa = token.substr(0, e);
    int64_t exponent = 0;
    if (e != std::string_view::npos) {
        bool negative = token[e + 1] == '-';
        for (size_t i = e + 1 + (negative || token[e + 1] == '+'); i < token.size(); ++i)
            exponent = std::min<int64_t>(exponent * 10 + (token[i] - '0'), 1'000'000);
        if (negative) exponent = -exponent;
    }
    size_t point = std::min(mantissa.find('.'), mantissa.size());
    size_t lead = mantissa.find_first_not_of("+-0.");
    int64_t place = lead < point ? int64_t(point - lead) - 1 : -int64_t(lead - point);
    return std::copysign(place + exponent >= 0 ? HUGE_VAL : 0.0, token[0] == '-' ? -1.0 : 1.0);
}

/* The kernel text of chainner:image:image_convolve as np.stack([l.split() for l in text.splitlines()]).astype(float):
 * rows of decimal numbers. Anything else (other characters, other number syntax, ragged or missing rows) declines to
 * the Python body, which parses or rejects it with NumPy. */
Kernel parse_kernel(const std::string &text)
{
    for (char ch : text)
        if (!(std::isdigit(static_cast<unsigned char>(ch)) || ch == '+' || ch == '-' || ch == '.' || ch == 'e' ||
              ch == 'E' || ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r'))
            throw RunPython{};
    // A decimal number: sign, digits with an optional fraction (or a fraction alone), optional exponent.
    auto decimal = [](std::string_view token) {
        size_t i = 0, n = token.size();
        if (i < n && (token[i] == '+' || token[i] == '-')) ++i;
        size_t integer = 0, fraction = 0;
        while (i < n && std::isdigit(static_cast<unsigned char>(token[i]))) ++i, ++integer;
        if (i < n && token[i] == '.') {
            ++i;
            while (i < n && std::isdigit(static_cast<unsigned char>(token[i]))) ++i, ++fraction;
        }
        if (integer + fraction == 0) return false;
        if (i < n && (token[i] == 'e' || token[i] == 'E')) {
            ++i;
            if (i < n && (token[i] == '+' || token[i] == '-')) ++i;
            size_t exponent = 0;
            while (i < n && std::isdigit(static_cast<unsigned char>(token[i]))) ++i, ++exponent;
            if (exponent == 0) return false;
        }
        return i == n;
    };
    Kernel kernel;
    size_t start = 0;
    while (start < text.size()) {
        size_t end = text.find_first_of("\r\n", start);
        std::string_view line(text.data() + start, (end == std::string::npos ? text.size() : end) - start);
        start = end == std::string::npos ? text.size() : end + (text.compare(end, 2, "\r\n") == 0 ? 2 : 1);
        size_t columns = 0;
        for (size_t i = 0; i < line.size();) {
            if (line[i] == ' ' || line[i] == '\t') {
                ++i;
                continue;
            }
            size_t j = line.find_first_of(" \t", i);
            std::string_view token = line.substr(i, (j == std::string_view::npos ? line.size() : j) - i);
            if (!decimal(token)) throw RunPython{};
            const char *first = token.data() + (token[0] == '+' ? 1 : 0);
            double value = 0;
            if (std::from_chars(first, token.data() + token.size(), value).ec == std::errc::result_out_of_range)
                value = beyond_doubles(token);
            kernel.values.push_back(value);
            ++columns;
            i += token.size();
        }
        if (columns == 0 || (kernel.rows > 0 && columns != kernel.columns)) throw RunPython{};
        kernel.columns = columns;
        ++kernel.rows;
    }
    if (kernel.rows == 0) throw RunPython{};
    return kernel;
}

/* chainner:image:image_convolve (miscellaneous/convolve.py): the parsed kernel flipped on both axes (a reversal of its
 * row-major values), then native_convolution.convolve. */
void convolve_node(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    Kernel kernel = parse_kernel(std::get<std::string>(inputs[1]));
    std::reverse(kernel.values.begin(), kernel.values.end());
    outputs.push_back(convolve(std::get<Image>(inputs[0]), kernel, integer(inputs[2])));
}

/* native_filters.arithmetic(a, operation, scale, b=, c=, d=, clip=) (second 0; scale and second as c_floats). */
Image arithmetic(const Image &a, int operation, double scale, const Image *b = nullptr, const Image *c = nullptr,
                 const Image *d = nullptr, bool clip = false)
{
    auto valid = [](const Image &image) {
        if (image.height == 0 || image.width == 0)
            throw NodeError("ValueError", "Invalid image shape " + shape_text(image));
    };
    valid(a);
    for (const Image *operand : {b, c, d}) {
        if (!operand) continue;
        valid(*operand);
        if (operand->height != a.height || operand->width != a.width || operand->channels != a.channels)
            throw NodeError("ValueError", "Filter operands must have identical shapes");
    }
    MutableImage out = MutableImage::allocate(a.height, a.width, a.channels);
    check(cn_filter_arithmetic(a.data, b ? b->data : nullptr, c ? c->data : nullptr, d ? d->data : nullptr, out.data(),
                               a.values(), operation, float(scale), 0.0f, clip ? 1 : 0));
    return std::move(out).publish();
}

/* A 3 x 3 kernel of the given coefficients times a factor (NumPy's weak-scalar float32 product: exact here). */
Kernel kernel3(std::array<double, 9> values, double factor = 1)
{
    Kernel kernel{3, 3, {}};
    for (double value : values) kernel.values.push_back(value * factor);
    return kernel;
}

/* chainner:image:sharpen_hbf (sharpen/high_boost_filter.py): KernelType 0 the 4-neighbour kernel, 1 the 8-neighbour
 * one; contrast adaptive mixing through cas.cas_mix (CAS mask from the luminance's 3 x 3 cross or square extrema).
 * A colour image's mask, and so the mix, is float64, normalized here as the output enforce converts it. */
void high_boost_filter(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image &image = std::get<Image>(inputs[0]);
    bool strong = std::get<int64_t>(inputs[1]) == 1;
    double amount = number(inputs[2]);
    if (amount == 0) {
        outputs.push_back(inputs[0]);
        return;
    }
    constexpr std::array<double, 9> identity{0, 0, 0, 0, 1, 0, 0, 0, 0};
    constexpr std::array<double, 9> cross{0, 1, 0, 1, 1, 1, 0, 1, 0};
    Kernel kernel{3, 3, {}};
    for (size_t i = 0; i < 9; ++i) {
        double neighbourhood = strong ? identity[i] - 1.0 / 9 : identity[i] - cross[i] / 5;
        kernel.values.push_back(neighbourhood * amount + identity[i]);
    }
    Image sharpened = convolve(image, kernel, 0);
    if (!std::get<bool>(inputs[3])) {
        outputs.push_back(std::move(sharpened));
        return;
    }
    double exponent = 1 / number(inputs[4]);
    size_t count = pixels(image);
    if (image.channels == 1) {
        int64_t shape = strong ? 0 : 1;  // cv2.MORPH_RECT, cv2.MORPH_CROSS
        Image low = morphology(image, shape, 1, 1, false), high = morphology(image, shape, 1, 1, true);
        MutableImage mask = MutableImage::allocate(image.height, image.width, 1);
        check(cn_analysis_cas(low.data, high.data, nullptr, nullptr, mask.data(), count, 1, exponent, 0, 0));
        Image weights = std::move(mask).publish();
        MutableImage out = MutableImage::allocate(image.height, image.width, 1);
        check(cn_analysis_cas(weights.data, nullptr, image.data, sharpened.data, out.data(), count, 1, 1, 0, 1));
        outputs.push_back(std::move(out).publish());
        return;
    }
    std::vector<double> luminance(count), low(count), high(count), mask(count), mixed(image.values());
    check(cn_cas_luminance(image.data, luminance.data(), count, size_t(image.channels), 1));
    std::array<uint8_t, 9> taps{};
    for (size_t i = 0; i < 9; ++i) taps[i] = strong || cross[i] != 0 ? 1 : 0;
    check(cn_cas_extrema_f64(luminance.data(), low.data(), high.data(), size_t(image.height), size_t(image.width),
                             taps.data(), 3, 3));
    check(cn_analysis_cas(low.data(), high.data(), nullptr, nullptr, mask.data(), count, 1, exponent, 1, 0));
    check(cn_analysis_cas(mask.data(), nullptr, image.data, sharpened.data, mixed.data(), count,
                          size_t(image.channels), 1, 1, 1));
    MutableImage out = MutableImage::allocate(image.height, image.width, image.channels);
    check(cn_pixels_convert_checked(mixed.data(), out.data(), mixed.size(), 1, 0, 1, nullptr));
    outputs.push_back(std::move(out).publish());
}

/* chainner:image:edge_detection (miscellaneous/edge_detection.py): Algorithm 1 Sobel, 2 Scharr, 3 gradient,
 * 4 differential, 5 Roberts (gradient filters; GradientComponent 1 magnitude, 2 x, 3 y), 6 Prewitt compass,
 * 7 Laplacian, 8 Laplacian denoise, 9 difference of Gaussians; alpha passes through. */
void edge_detection(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image &input = std::get<Image>(inputs[0]);
    double amount = number(inputs[1]);
    int64_t algorithm = std::get<int64_t>(inputs[2]), component = std::get<int64_t>(inputs[3]);
    Image image = input;
    std::optional<Image> alpha;
    if (input.channels >= 4) {
        alpha = select_channels(input, {3});
        image = select_channels(input, {0, 1, 2});
    }
    const Kernel laplace = kernel3({1, 1, 1, 1, -8, 1, 1, 1, 1}, 0.25);
    Image result;
    if (algorithm <= 5) {
        static const std::array<Kernel, 5> filters_x{
            kernel3({-1, 0, 1, -2, 0, 2, -1, 0, 1}, 0.25),
            kernel3({-47, 0, 47, -162, 0, 162, -47, 0, 47}, 1.0 / 256),
            kernel3({0, 0, 0, -1, 1, 0, 0, 0, 0}),
            kernel3({0, 0, 0, 0, -1, 1, 0, -1, 1}, 0.5),
            kernel3({0, 0, 0, 0, 1, 0, 0, 0, -1}),
        };
        const Kernel &filter_x = filters_x[size_t(algorithm - 1)];
        Kernel filter_y = algorithm == 4   ? kernel3({0, 0, 0, 0, 1, 1, 0, -1, -1}, 0.5)
                          : algorithm == 5 ? kernel3({0, 0, 0, 0, 0, 1, 0, -1, 0})
                                           : rotate90(filter_x);
        if (algorithm == 5) component = 1;
        if (component == 1) {
            Image gx = convolve(image, filter_x, 0);
            Image gy = convolve(image, filter_y, 0);
            result = arithmetic(gx, 0, amount / 2, &gy);
        } else {
            result = arithmetic(convolve(image, component == 2 ? filter_x : filter_y, 0), 1, amount / 2);
        }
    } else if (algorithm == 6) {
        Kernel filter_0 = kernel3({-1, 1, 1, -1, -2, 1, -1, 1, 1}, 0.25);
        Kernel filter_45 = kernel3({1, 1, 1, -1, -2, 1, -1, -1, 1}, 0.25);
        Image filtered = convolve(image, filter_0, 0);
        Image g = arithmetic(filtered, 5, 1, &filtered);
        Kernel turned = filter_0;
        for (int i = 1; i < 4; ++i) {
            turned = rotate90(turned);
            Image next = convolve(image, turned, 0);
            g = arithmetic(g, 5, 1, &next);
        }
        turned = filter_45;
        for (int i = 0; i < 4; ++i) {
            Image next = convolve(image, turned, 0);
            g = arithmetic(g, 5, 1, &next);
            turned = rotate90(turned);
        }
        result = arithmetic(g, 2, amount / 2);
    } else if (algorithm == 7) {
        result = arithmetic(convolve(image, laplace, 0), 2, amount);
    } else if (algorithm == 8) {
        Image maximum = morphology(image, 1, 1, 1, true);
        Image minimum = morphology(image, 1, 1, 1, false);
        Image laplacian = convolve(image, laplace, 0);
        result = arithmetic(arithmetic(image, 4, 1, &maximum, &minimum, &laplacian), 2, amount);
    } else {
        double first = number(inputs[4]), second = number(inputs[5]);
        Image g1 = fast_gaussian_blur(image, first, first);
        Image g2 = fast_gaussian_blur(image, second, second);
        result = arithmetic(g1, 3, amount, &g2);
    }
    result = arithmetic(result, 6, 1, nullptr, nullptr, nullptr, true);
    outputs.push_back(alpha ? with_alpha(result, *alpha) : result);
}

/* native_convolution.box_blur with integer radii in [0, 1000] (OpenCV's blur, BORDER_REFLECT_101). */
Image box_blur_integer(const Image &image, int64_t radius_x, int64_t radius_y)
{
    if (image.height == 0 || image.width == 0)
        throw NodeError("ValueError", "Invalid convolution image shape " + shape_text(image));
    MutableImage out = MutableImage::allocate(image.height, image.width, image.channels);
    check(cn_convolution_box(image.data, out.data(), size_t(image.height), size_t(image.width), size_t(image.channels),
                             size_t(radius_x), size_t(radius_y)));
    return std::move(out).publish();
}

/* chainner:image:blur (blur/box_blur.py): integer radii through native_convolution.box_blur; fractional ones
 * through native_box.separable_box (sepFilter2D on the "filter" dispatch lanes), integral radii of 15 and more
 * blurred apart first; an average radius of 70 and more filters with the 2-D kernel through the spectral filter
 * (cv2.dft), which declines to the Python body (after its own empty-image error). The result is the raw one: the
 * body's freeze_normalized(clamp=True) clamps it as the output enforce does. */
void box_blur(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    double radius_x = number(inputs[1]), radius_y = number(inputs[2]);
    if (radius_x == 0 && radius_y == 0) {
        outputs.push_back(inputs[0]);
        return;
    }
    // Python's round() (half to even) beyond 200.
    if (radius_x > 200) radius_x = std::nearbyint(radius_x);
    if (radius_y > 200) radius_y = std::nearbyint(radius_y);
    auto integral = [](double value) { return std::trunc(value) == value; };
    Image image = std::get<Image>(inputs[0]);
    if (integral(radius_x) && integral(radius_y)) {
        outputs.push_back(box_blur_integer(image, int64_t(radius_x), int64_t(radius_y)));
        return;
    }
    if (radius_x >= 15 && integral(radius_x)) {
        image = box_blur_integer(image, int64_t(radius_x), 0);
        radius_x = 1;
    }
    if (radius_y >= 15 && integral(radius_y)) {
        image = box_blur_integer(image, 0, int64_t(radius_y));
        radius_y = 1;
    }
    if (std::sqrt(radius_x * radius_y) >= 70) {
        if (image.height == 0 || image.width == 0)
            throw NodeError("ValueError", "Spectral filtering requires a nonempty float32 image");
        throw RunPython{};
    }
    if (image.height == 0 || image.width == 0) throw NodeError("ValueError", "Separable Box Blur requires a nonempty image");
    MutableImage out = MutableImage::allocate(image.height, image.width, image.channels);
    check(cn_box_separable(image.data, out.data(), size_t(image.height), size_t(image.width), size_t(image.channels),
                           radius_x, radius_y, dispatch::opencv_lanes("filter")));
    outputs.push_back(std::move(out).publish());
}

/* native_neighborhood._image: a nonempty image. */
void neighborhood_image(const Image &image)
{
    if (image.height == 0 || image.width == 0) throw NodeError("ValueError", "Invalid image shape " + shape_text(image));
}

/* chainner:image:median_blur (blur/median_blur.py, native_neighborhood.median): radii 1 and 2 on float32 (OpenCV's
 * median_blur dispatch lanes), larger ones on the uint8 image, whose uint8 result is normalized here as the output
 * enforce converts it. */
void median_blur(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image &image = std::get<Image>(inputs[0]);
    int64_t radius = integer(inputs[1]);
    if (radius == 0) {
        outputs.push_back(inputs[0]);
        return;
    }
    neighborhood_image(image);
    size_t height = size_t(image.height), width = size_t(image.width), channels = size_t(image.channels);
    MutableImage out = MutableImage::allocate(image.height, image.width, image.channels);
    if (radius <= 2) {
        check(cn_neighborhood_median_f32(image.data, out.data(), height, width, channels, size_t(radius),
                                         dispatch::opencv_lanes("median_blur")));
    } else {
        std::vector<uint8_t> source = to_uint8(image, true), result(source.size());
        check(cn_neighborhood_median_u8(source.data(), result.data(), height, width, channels, size_t(radius)));
        check(cn_pixels_convert_checked(result.data(), out.data(), image.values(), 2, 0, 1, nullptr));
    }
    outputs.push_back(std::move(out).publish());
}

/* chainner:image:dither (quantize/dither.py, native_neighborhood.uniform_dither): UniformDitherAlgorithm "None",
 * "Ordered" (ThresholdMap B2-B16), "Diffusion" (ErrorDiffusionMap values) or "Riemersma". */
void dither(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image &image = std::get<Image>(inputs[0]);
    const std::string &algorithm = std::get<std::string>(inputs[2]);
    int mode = algorithm == "None" ? 0 : algorithm == "Ordered" ? 1 : algorithm == "Diffusion" ? 2 : 3;
    size_t map_size = 2;
    if (mode == 1) map_size = size_t(std::stoi(std::get<std::string>(inputs[3]).substr(1)));
    int map = 0;
    if (mode == 2) map = diffusion_map(std::get<std::string>(inputs[4]));
    neighborhood_image(image);
    uint32_t colors = dither_count(inputs[1], "Dither colors must be an integer in [2,4294967295]");
    MutableImage out = MutableImage::allocate(image.height, image.width, image.channels);
    size_t height = size_t(image.height), width = size_t(image.width), channels = size_t(image.channels);
    if (mode == 3) {
        uint32_t history = dither_count(inputs[5], "History length must be an integer in [2,4294967295]");
        check(cn_neighborhood_uniform_riemersma(image.data, out.data(), height, width, channels, colors, history,
                                                float(1.0 / history)));
    } else {
        check(cn_neighborhood_dither(image.data, out.data(), height, width, channels, colors, mode, map_size, map,
                                     nullptr, 0, nullptr));
    }
    outputs.push_back(std::move(out).publish());
}

/* A uint8 result as the output enforce normalizes it (value / 255). */
Image normalized_u8(const std::vector<uint8_t> &values, int32_t height, int32_t width, int32_t channels)
{
    MutableImage out = MutableImage::allocate(height, width, channels);
    check(cn_pixels_convert_checked(values.data(), out.data(), size_t(height) * size_t(width) * size_t(channels), 2, 0,
                                    1, nullptr));
    return std::move(out).publish();
}

/* chainner:image:canny_edge_detection (miscellaneous/canny_edge_detection.py, native_repair.canny) on the uint8
 * image; IPP's threshold policy applies to a single-threaded OpenCV only. An empty image goes to cv2.Canny, so it
 * declines to the Python body. */
void canny(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image &image = std::get<Image>(inputs[0]);
    double lower = double(integer(inputs[1])), upper = double(integer(inputs[2]));
    if (image.height == 0 || image.width == 0) throw RunPython{};
    std::vector<uint8_t> source = to_uint8(image, true), edges(pixels(image));
    bool ipp = image.channels == 1 && image.height > 3 && image.width > 3 &&
               dispatch::number("cv2", "getNumThreads", {}) <= 1 && dispatch::cv2_use_ipp();
    check(cn_canny_u8_policy(source.data(), edges.data(), size_t(image.height), size_t(image.width),
                             size_t(image.channels), lower, upper, ipp ? 1 : 0));
    outputs.push_back(normalized_u8(edges, image.height, image.width, 1));
}

/* chainner:image:distance_transform (miscellaneous/distance_transform.py): the sub-pixel ESDF (native_filters
 * .subpixel_sdf, radius spread * 2) or the binary SDF of the uint8 image, whose distances under IPP (OpenCV's
 * default) come from cv2.distanceTransform, so a nonempty image declines to the Python body there. */
void distance_transform(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image &image = std::get<Image>(inputs[0]);
    int64_t spread = integer(inputs[1]);
    MutableImage out = MutableImage::allocate(image.height, image.width, 1);
    if (std::get<bool>(inputs[2])) {
        if (image.height == 0 || image.width == 0)
            throw NodeError("ValueError", "Invalid image shape " + shape_text(image));
        check(cn_distance_esdf(image.data, out.data(), size_t(image.height), size_t(image.width),
                               float(double(spread) * 2)));
    } else {
        if (image.height == 0 || image.width == 0)
            throw NodeError("ValueError", "Binary SDF requires a nonempty grayscale image");
        if (dispatch::cv2_use_ipp()) throw RunPython{};
        std::vector<uint8_t> source = to_uint8(image, true);
        check(cn_distance_binary(source.data(), out.data(), size_t(image.height), size_t(image.width), double(spread),
                                 0));
    }
    outputs.push_back(std::move(out).publish());
}

/* chainner:image:fast_nlmeans (noise/denoise.py, native_denoise.denoise) on the clipped uint8 image. */
void denoise(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image &image = std::get<Image>(inputs[0]);
    double strength = number(inputs[1]), color_strength = number(inputs[2]);
    int64_t patch = integer(inputs[3]), search = integer(inputs[4]);
    std::vector<uint8_t> source = to_uint8(image, false);
    if (image.height == 0 || image.width == 0) throw NodeError("ValueError", "Non-local means requires a nonempty image");
    std::vector<uint8_t> result(source.size());
    check(cn_denoise_u8(source.data(), result.data(), size_t(image.height), size_t(image.width), size_t(image.channels),
                        float(strength), float(color_strength), int(patch), int(search)));
    outputs.push_back(normalized_u8(result, image.height, image.width, image.channels));
}

/* chainner:image:quantize_to_referece (quantize/quantize_to_reference.py, native_filters.quantize_reference): the
 * reference's size divides the image's (Python's % raises ZeroDivisionError for an empty reference side). */
void quantize_to_reference(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image &image = std::get<Image>(inputs[0]);
    const Image &reference = std::get<Image>(inputs[1]);
    int64_t radius = integer(inputs[2]);
    auto require = [](bool condition, const char *message) {
        if (!condition) throw NodeError("AssertionError", message);
    };
    require(image.channels == reference.channels, "Image and reference image must have the same number of channels");
    require(image.height >= reference.height, "Image height must be larger than reference image height");
    if (reference.height == 0) throw NodeError("ZeroDivisionError", "division by zero");
    require(image.height % reference.height == 0, "Image height must be a multiple of reference image height");
    require(image.width >= reference.width, "Image width must be larger than reference image width");
    if (reference.width == 0) throw NodeError("ZeroDivisionError", "division by zero");
    require(image.width % reference.width == 0, "Image width must be a multiple of reference image width");
    double scale = number(inputs[3]) / 100;
    scale = scale * scale;
    for (const Image *operand : {&image, &reference})
        if (operand->height == 0 || operand->width == 0)
            throw NodeError("ValueError", "Invalid image shape " + shape_text(*operand));
    MutableImage out = MutableImage::allocate(image.height, image.width, image.channels);
    check(cn_filter_quantize_reference(image.data, reference.data, out.data(), size_t(image.height), size_t(image.width),
                                       size_t(reference.height), size_t(reference.width), size_t(image.channels),
                                       size_t(radius), double(reference.width) * scale));
    outputs.push_back(std::move(out).publish());
}

/* native_transfer_complete.transfer_mask: the pixels whose colour is determinate (not fully transparent). */
std::vector<uint8_t> transfer_mask(const Image &image)
{
    std::vector<uint8_t> mask(std::max<size_t>(pixels(image), 1));
    check(cn_transfer_mask(image.data, pixels(image), size_t(image.channels), size_t(image.channels), 1, 0, mask.data()));
    return mask;
}

/* native_transfer_complete.mean_std_apply's statistics of an image's masked pixels: native_analysis.channel_stats of
 * the gathered (N, 3) values, reduced as NumPy reduces their first column. */
std::array<float, 6> masked_stats(const Image &image, const std::vector<uint8_t> &mask)
{
    std::vector<float> gathered(std::max<size_t>(image.values(), 1));
    size_t count = 0;
    check(cn_transfer_gather_f32(image.data, mask.data(), pixels(image), gathered.data(), &count));
    ReduceSetup setup = reduce_setup({int64_t(count), 1}, {12, 4});
    std::array<float, 6> stats{};
    check(cn_analysis_channel_stats(gathered.data(), count, setup.block, setup.period, stats.data()));
    return stats;
}

/* chainner:image:color_transfer (correction/color_transfer.py): the Mean+Std algorithm (mean_std.mean_std_transfer's
 * native path: TransferColorSpace "L*a*b*" through cvt_color BGR2LAB/LAB2BGR or "RGB"; OverflowMethod 1 clip, 0
 * scale); alpha passes through. Linear Histogram and Principal Color (complex matrices through NumPy's OpenBLAS
 * LAPACK) and empty images (NumPy's helpers) decline to the Python body. */
void color_transfer(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image &image = std::get<Image>(inputs[0]);
    const Image &reference = std::get<Image>(inputs[1]);
    if (std::get<std::string>(inputs[2]) != "mean_std" || pixels(image) == 0 || pixels(reference) == 0)
        throw RunPython{};
    bool lab = std::get<std::string>(inputs[3]) == "L*a*b*";
    bool scale = std::get<int64_t>(inputs[4]) == 0;
    bool reciprocal = std::get<bool>(inputs[5]);
    std::vector<uint8_t> valid = transfer_mask(image), reference_valid = transfer_mask(reference);
    Image source = image.channels == 4 ? select_channels(image, {0, 1, 2}) : image;
    Image target = reference.channels == 4 ? select_channels(reference, {0, 1, 2}) : reference;
    std::array<float, 6> limits{0, 1, 0, 1, 0, 1};
    if (lab) {
        limits = {0, 100, -127, 127, -127, 127};
        source = cvt_color(source, 10, 0, 3);
        target = cvt_color(target, 10, 0, 3);
    }
    std::array<float, 6> stats = masked_stats(source, valid), reference_stats = masked_stats(target, reference_valid);
    MutableImage transferred = MutableImage::allocate(source.height, source.width, 3);
    check(cn_analysis_transfer(source.data, transferred.data(), valid.data(), pixels(source), stats.data(),
                               reference_stats.data(), limits.data(), reciprocal ? 1 : 0, scale ? 1 : 0));
    Image result = std::move(transferred).publish();
    if (lab) result = cvt_color(result, 11, 0, 3);
    if (image.channels == 4) {
        MutableImage out = MutableImage::allocate(image.height, image.width, 4);
        check(cn_transfer_alpha(result.data, 3, 1, image.data, 4, 1, pixels(image), 0, 0, out.data(), 4, 1));
        result = std::move(out).publish();
    }
    outputs.push_back(result);
}

}  // namespace

CN_RUN_NODES_DECLARE(image_filters)
{
    registry.add("chainner:image:add_noise", add_noise);
    registry.add("chainner:image:blur", box_blur);
    registry.add("chainner:image:canny_edge_detection", canny);
    registry.add("chainner:image:color_transfer", color_transfer);
    registry.add("chainner:image:distance_transform", distance_transform);
    registry.add("chainner:image:dither", dither);
    registry.add("chainner:image:edge_detection", edge_detection);
    registry.add("chainner:image:fast_nlmeans", denoise);
    registry.add("chainner:image:high_pass", high_pass);
    registry.add("chainner:image:image_convolve", convolve_node);
    registry.add("chainner:image:median_blur", median_blur);
    registry.add("chainner:image:pixelate", pixelate);
    registry.add("chainner:image:quantize_to_referece", quantize_to_reference);
    registry.add("chainner:image:sharpen", unsharp_mask);
    registry.add("chainner:image:sharpen_hbf", high_boost_filter);
}
