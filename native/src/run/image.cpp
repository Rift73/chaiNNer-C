/* Image storage on the NumPy pool, and MutableImage (run_engine.hpp). */
#include "engine.hpp"
#include "numpy_pool.hpp"

#include <cstdint>
#include <limits>

namespace cn::run {

/* An empty image still gets a block (64 bytes), as NumPy's empty arrays have data: kernels refuse a null pointer
 * even for a zero count. */
ImageStorage::ImageStorage(size_t bytes) : pooled_(true)
{
    data_ = static_cast<float *>(cn_numpy_pool_allocate(bytes ? bytes : 64));
    if (!data_) throw NodeError("MemoryError", "Unable to allocate " + std::to_string(bytes) + " bytes for an image");
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
    image.storage_ = std::make_shared<ImageStorage>(values * sizeof(float));
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

}  // namespace cn::run
