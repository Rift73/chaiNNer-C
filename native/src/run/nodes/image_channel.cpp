/* Native entry points of image_channel nodes (the pattern: image_utility.cpp's note). */
#include "run_engine.hpp"

#include <algorithm>

namespace {

using namespace cn::run;

/* An image of ones the size of image (the body's np.ones((h, w), np.float32)). */
Image ones(const Image &image)
{
    MutableImage out = MutableImage::allocate(image.height, image.width, 1);
    std::fill_n(out.data(), size_t(image.height) * size_t(image.width), 1.0f);
    return std::move(out).publish();
}

/* source's pixels of interleaved channels as one plane per channel. */
template <size_t channels>
void deinterleave(const float *source, float *const *targets, size_t pixels)
{
    for (size_t p = 0; p < pixels; ++p)
        for (size_t c = 0; c < channels; ++c) targets[c][p] = source[p * channels + c];
}

/* chainner:image:split_channels (image_channel/all/separate_rgba.py): outputs R, G, B, A of a BGR(A) image (the
 * body's views img[:, :, i], copied in one pass; ones for a missing alpha); a grey image is R itself, the others one
 * shared image of ones. */
void split_channels(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image &image = std::get<Image>(inputs[0]);
    if (image.channels == 1) {
        Image one = ones(image);
        outputs = {inputs[0], one, one, one};
        return;
    }
    size_t channels = size_t(image.channels), pixels = size_t(image.height) * size_t(image.width);
    MutableImage planes[4] = {MutableImage::allocate(image.height, image.width, 1),
                              MutableImage::allocate(image.height, image.width, 1),
                              MutableImage::allocate(image.height, image.width, 1),
                              channels == 4 ? MutableImage::allocate(image.height, image.width, 1) : MutableImage()};
    float *targets[4] = {planes[0].data(), planes[1].data(), planes[2].data(), planes[3].data()};
    if (channels == 3)
        deinterleave<3>(image.data, targets, pixels);
    else
        deinterleave<4>(image.data, targets, pixels);
    Image alpha = channels == 4 ? std::move(planes[3]).publish() : ones(image);
    outputs = {std::move(planes[2]).publish(), std::move(planes[1]).publish(), std::move(planes[0]).publish(), alpha};
}

}  // namespace

CN_RUN_NODES_DECLARE(image_channel)
{
    registry.add("chainner:image:split_channels", split_channels);
}
