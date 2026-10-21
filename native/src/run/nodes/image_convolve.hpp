/* native_convolution.convolve for native entry points (Convolve, High Boost Filter, Edge Detection, Image Metrics). */
#pragma once

#include "run_engine.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cn::run {

/* A convolution kernel as the body's float64 (or float32-valued) array: rows x columns coefficients, row-major. */
struct Kernel {
    size_t rows = 0, columns = 0;
    std::vector<double> values;
};

/* native_convolution.convolve(image, kernel, padding) with padding >= 0: in its order, ValueError for an empty image
 * or kernel, OverflowError for a padding beyond OpenCV's dimensions; OpenCV's spatial filter2D through
 * cn_convolution_spatial on the "filter" dispatch lanes. Where the body calls OpenCV (the spectral filter2D's
 * cv2.dft, or cv2.filter2D for an untested OpenCV, IPP's inexact mode, a dispatch mode the kernel does not mirror or
 * 130 coefficients and more) this throws RunPython before any output. */
Image convolve(const Image &image, const Kernel &kernel, int64_t padding);

/* np.rot90(kernel, 1): the kernel turned a quarter counterclockwise. */
Kernel rotate90(const Kernel &kernel);

}  // namespace cn::run
