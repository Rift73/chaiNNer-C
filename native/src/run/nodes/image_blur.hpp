/* image_utils.fast_gaussian_blur and native_gaussian.gaussian for native entry points (Gaussian Blur, Unsharp Mask,
 * High Pass, ...). */
#pragma once

#include "run_engine.hpp"

namespace cn::run {

/* native_gaussian.gaussian(image, sigma_x, sigma_y) with its defaults (size (0, 0), BORDER_REFLECT): ValueError for
 * an empty image; OpenCV's "filter" dispatch lanes from dispatch.hpp. */
Image gaussian(const Image &image, double sigma_x, double sigma_y);

/* fast_gaussian_blur(image, sigma_x, sigma_y)'s raw result, as _fast_gaussian_blur returns it (a caller with
 * normalized=True returns it to its ImageOutput, whose enforce clamps it as freeze_normalized(clamp=True) would):
 * image itself where the body returns img.copy() (a sigma of 0). Where the body downsamples (both sigmas at least
 * 11 and the image larger than one pixel along a side) its upsampling is cv2.resize under IPP, so this throws
 * RunPython (before any output) after the body's own errors (ZeroDivisionError for an empty side). */
Image fast_gaussian_blur(const Image &image, double sigma_x, double sigma_y);

}  // namespace cn::run
