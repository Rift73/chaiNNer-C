/* native_convolution.convolve (image_convolve.hpp). */
#include "image_convolve.hpp"

#include "image_node_support.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace cn::run {
namespace {

/* NumPy's float64 -> float32 cast (round to nearest; beyond float32's range an infinity, as OpenCV's silent
 * narrowing of double coefficients gives). */
float narrow(double value)
{
    constexpr double overflow = 0x1.ffffffp127;  // FLT_MAX plus half its unit in the last place
    constexpr float infinity = std::numeric_limits<float>::infinity();
    if (std::fabs(value) >= overflow) return value > 0 ? infinity : -infinity;
    return float(value);
}

}  // namespace

Image convolve(const Image &image, const Kernel &kernel, int64_t padding)
{
    if (image.height == 0 || image.width == 0)
        throw NodeError("ValueError", "Invalid convolution image shape " + shape_text(image));
    if (kernel.values.empty())
        throw NodeError("ValueError", "Convolution kernel must be a nonempty two-dimensional array");
    if (padding > (int64_t(std::numeric_limits<int32_t>::max()) - std::max(image.height, image.width)) / 2)
        throw NodeError("OverflowError", "Padded image exceeds OpenCV's dimension range");
    size_t size = kernel.values.size();
    bool tested = dispatch::cv2_tested("nodes.impl.native_convolution");
    // filter.dispatch.cpp's DFT threshold: area 130 for float32 with SSE3 active, else 50 (spectral: cv2.dft).
    if (tested && size >= (dispatch::cv2_sse3() ? 130u : 50u)) throw RunPython{};
    // _spatial_lanes: the mirrored dispatch modes are the baseline (0) and AVX2.
    if (!tested || dispatch::cv2_use_ipp_not_exact()) throw RunPython{};
    std::string mode = dispatch::opencv_target("filter");
    if ((mode != "BASELINE" && mode != "AVX2") || size >= 130) throw RunPython{};
    size_t lanes = mode == "BASELINE" ? 0 : dispatch::opencv_lanes("filter");
    std::vector<float> coefficients;
    for (double value : kernel.values) coefficients.push_back(narrow(value));
    MutableImage out = MutableImage::allocate(int32_t(image.height + 2 * padding), int32_t(image.width + 2 * padding),
                                              image.channels);
    check(cn_convolution_spatial(image.data, out.data(), size_t(image.height), size_t(image.width),
                                 size_t(image.channels), coefficients.data(), kernel.rows, kernel.columns,
                                 size_t(padding), lanes));
    return std::move(out).publish();
}

Kernel rotate90(const Kernel &kernel)
{
    Kernel turned{kernel.columns, kernel.rows, std::vector<double>(kernel.values.size())};
    for (size_t i = 0; i < turned.rows; ++i)
        for (size_t j = 0; j < turned.columns; ++j)
            turned.values[i * turned.columns + j] = kernel.values[j * kernel.columns + (kernel.columns - 1 - i)];
    return turned;
}

}  // namespace cn::run
