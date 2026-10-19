/* Image storage on the NumPy pool, MutableImage (run_engine.hpp) and the engine's image output normalization. */
#include "engine.hpp"
#include "kernel_status.hpp"
#include "numpy_pool.hpp"

#include <pybind11/numpy.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <new>
#include <string>

namespace cn::run {
namespace {

/* NumPy's _ArrayMemoryError._size_to_string (numpy/_core/_exceptions.py): a byte count in binary units. */
std::string size_to_string(uint64_t bytes)
{
    constexpr const char *units[] = {"bytes", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB"};
    constexpr int count = int(std::size(units));
    int unit = std::max(int(std::bit_width(bytes)) - 1, 1) / 10;
    double value = std::ldexp(double(bytes), -10 * unit);  // num_bytes / unit_val, correctly rounded
    if (std::nearbyint(value) == 1024.0) {  // round(n_units) == STEP (half to even)
        unit += 1;
        value /= 1024.0;
    }
    if (unit >= count) {
        value = std::ldexp(value, 10 * (unit - (count - 1)));
        unit = count - 1;
    }
    char text[64];
    if (unit == 0)
        std::snprintf(text, sizeof text, "%.0f %s", value, units[unit]);
    else if (std::nearbyint(value) < 1000.0)
        std::snprintf(text, sizeof text, "%#.3g %s", value, units[unit]);
    else
        std::snprintf(text, sizeof text, "%#.0f %s", value, units[unit]);
    return text;
}

/* The MemoryError np.empty raises for the float32 array an image is in Python (2-D for one channel): NumPy's own
 * _ArrayMemoryError, made under the GIL (as the bridges' dispatch facts take it), or a MemoryError with its text
 * when Python cannot make it. */
[[noreturn]] void out_of_memory(int32_t height, int32_t width, int32_t channels, size_t bytes)
{
    std::string shape = "(" + std::to_string(height) + ", " + std::to_string(width);
    if (channels != 1) shape += ", " + std::to_string(channels);
    NodeError error("MemoryError", "Unable to allocate " + size_to_string(bytes) + " for an array with shape " +
                                       shape + ") and data type float32");
    if (Py_IsInitialized() && !Py_IsFinalizing()) {
        namespace py = pybind11;
        py::gil_scoped_acquire held;
        try {
            py::tuple dims = channels == 1 ? py::tuple(py::make_tuple(height, width))
                                           : py::tuple(py::make_tuple(height, width, channels));
            py::object exception =
                py::module_::import("numpy._core._exceptions").attr("_ArrayMemoryError")(dims, py::dtype::of<float>());
            if (py::str(exception).cast<std::string>() == error.what()) {
                error.python_type = "numpy._core._exceptions._ArrayMemoryError";
                error.exception = py_ref(exception);
            }
        } catch (py::error_already_set &) {
        }
    }
    throw error;
}

}  // namespace

/* An empty image still gets a block (64 bytes), as NumPy's empty arrays have data: kernels refuse a null pointer
 * even for a zero count. */
ImageStorage::ImageStorage(size_t bytes) : bytes_(bytes ? bytes : 64), pooled_(true)
{
    data_ = static_cast<float *>(cn_numpy_pool_allocate(bytes_));
    if (!data_) throw std::bad_alloc();
}

ImageStorage::~ImageStorage()
{
    if (pooled_) cn_numpy_pool_free(data_);
}

MutableImage MutableImage::allocate(int32_t height, int32_t width, int32_t channels)
{
    if (height < 0 || width < 0 || (channels != 1 && channels != 3 && channels != 4))
        throw NodeError("ValueError", "Invalid image dimensions or arguments passed to C kernels");
    size_t values = size_t(height) * size_t(width) * size_t(channels);
    if (values > std::numeric_limits<size_t>::max() / sizeof(float))
        throw NodeError("OverflowError", "Image size exceeds the C kernel addressable buffer range");
    MutableImage image;
    try {
        image.storage_ = std::make_shared<ImageStorage>(values * sizeof(float));
    } catch (const std::bad_alloc &) {
        out_of_memory(height, width, channels, values * sizeof(float));
    }
    image.data_ = image.storage_->data();
    image.height_ = height;
    image.width_ = width;
    image.channels_ = channels;
    return image;
}

Image MutableImage::publish() &&
{
    Image image{std::move(storage_), data_, height_, width_, channels_};
    data_ = nullptr;
    height_ = width_ = channels_ = 0;
    return image;
}

Image normalized_output(const Image &image)
{
    int same = 0;
    check(cn_pixels_normalized_f32(image.data, image.values(), &same));
    if (same) return image;
    MutableImage copy = MutableImage::allocate(image.height, image.width, image.channels);
    check(cn_pixels_convert_checked(image.data, copy.data(), image.values(), 0, 0, 1, nullptr));
    return std::move(copy).publish();
}

}  // namespace cn::run
