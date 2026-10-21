/* image_utils.fast_gaussian_blur and native_gaussian.gaussian, line by line (image_blur.hpp). */
#include "image_blur.hpp"

#include "../kernel_status.hpp"
#include "dispatch.hpp"

#include <algorithm>
#include <cmath>

namespace cn::run {
namespace {

/* cv2.BORDER_REFLECT. */
constexpr int border_reflect = 2;

/* get_scale_factor: the downsampling factor for a sigma (every comparison false for NaN gives 8). */
double scale_factor(double sigma)
{
    if (sigma < 11) return 1;
    if (sigma < 15) return 1.25;
    if (sigma < 20) return 1.5;
    if (sigma < 25) return 2;
    if (sigma < 30) return 2.5;
    if (sigma < 50) return 3;
    if (sigma < 100) return 4;
    if (sigma < 200) return 6;
    return 8;
}

/* get_sizing's size_down and sigma_up. Its sigma_down only serves the downsampled path, which declines. */
struct Sizing {
    int64_t size;
    double up;
};

Sizing sizing(int64_t size, double sigma, double factor)
{
    if (factor <= 1) return {size, sigma};
    int64_t size_down = int64_t(std::ceil(double(size) / factor));
    if (size_down == 0) throw NodeError("ZeroDivisionError", "division by zero");
    return {size_down, double(size) / double(size_down)};
}

}  // namespace

Image gaussian(const Image &image, double sigma_x, double sigma_y)
{
    if (image.height == 0 || image.width == 0) throw NodeError("ValueError", "Gaussian filtering requires a nonempty image");
    MutableImage out = MutableImage::allocate(image.height, image.width, image.channels);
    size_t lanes = dispatch::opencv_lanes("filter");
    check(cn_gaussian_f32(image.data, out.data(), size_t(image.height), size_t(image.width), size_t(image.channels), 0,
                          0, sigma_x, sigma_y, border_reflect, lanes));
    return std::move(out).publish();
}

Image fast_gaussian_blur(const Image &image, double sigma_x, double sigma_y)
{
    if (sigma_x == 0 || sigma_y == 0) return image;
    double factor = std::min(scale_factor(sigma_x), scale_factor(sigma_y));
    Sizing y = sizing(image.height, sigma_y, factor);
    Sizing x = sizing(image.width, sigma_x, factor);
    if (image.height != y.size || image.width != x.size) throw RunPython{};
    /* The body's post blur runs when either up-sigma is nonzero: always here (a nonzero sigma, or size / size_down). */
    return gaussian(image, x.up, y.up);
}

}  // namespace cn::run
