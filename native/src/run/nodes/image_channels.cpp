/* Native entry points of image_channel nodes (pattern: image_utility.cpp). */
#include "image_node_support.hpp"

#include "chainner_ext_kernels.h"

#include <algorithm>
#include <optional>
#include <utility>

namespace {

using namespace cn::run;

/* A channel input of native_channels' assembly: an image or a colour (ImageInput(allow_colors=True)). */
const Image *as_image(const Value &value) { return std::get_if<Image>(&value); }

/* native_channels._same_size: every image is nonempty (checked in input order), at least one input is an image,
 * and all images have the first one's size. Returns the first image. */
const Image &same_size(std::initializer_list<const Value *> inputs, const char *no_image)
{
    const Image *first = nullptr;
    for (const Value *value : inputs) {
        if (const Image *image = as_image(*value)) {
            nonempty(*image);
            if (!first) first = image;
        }
    }
    if (!first) throw NodeError("ValueError", no_image);
    for (const Value *value : inputs) {
        const Image *image = as_image(*value);
        if (image && (image->height != first->height || image->width != first->width))
            throw NodeError("AssertionError", "All channel images must have the same resolution");
    }
    return *first;
}

/* native_channels._assemble: output channel i is channel selections[i].second of input selections[i].first, or that
 * channel of a colour input as a constant (a c_float of the Python float). */
Value assemble(const Image &size, const std::vector<const Value *> &inputs,
               const std::vector<std::pair<size_t, size_t>> &selections)
{
    std::vector<cn_channel_source> sources;
    for (auto [input, channel] : selections) {
        if (const Image *image = as_image(*inputs[input]))
            sources.push_back({image->data, size_t(image->channels), channel, 0.0f});
        else
            sources.push_back({nullptr, 0, 0, float(std::get<Color>(*inputs[input]).values[channel])});
    }
    MutableImage out = MutableImage::allocate(size.height, size.width, int32_t(selections.size()));
    check(cn_channels_assemble(sources.data(), out.data(), pixels(size), sources.size()));
    return std::move(out).publish();
}

/* chainner:image:combine_rgba (all/combine_rgba.py, native_channels.combine_rgba): B, G, R and A (a grey 1 colour
 * when unconnected); enforced inputs are single-channel images or colours. */
void combine_rgba(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Value opaque = Color{{1.0}, {}};
    const Value &alpha = std::holds_alternative<std::monostate>(inputs[3]) ? opaque : inputs[3];
    const Image &size = same_size({&inputs[2], &inputs[1], &inputs[0], &alpha},
                                  "At least one channels must be an image, but all given channels are colors.");
    outputs.push_back(assemble(size, {&inputs[2], &inputs[1], &inputs[0], &alpha}, {{0, 0}, {1, 0}, {2, 0}, {3, 0}}));
}

/* chainner:image:merge_channels (all/merge_channels.py, native_channels.merge_channels): the first four channels of
 * the connected images; two channels become B, G, G (cv2.merge's order). */
void merge_channels(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    std::vector<const Value *> images;
    for (const Value &input : inputs)
        if (as_image(input)) images.push_back(&input);
    const Image &first = std::get<Image>(*images[0]);
    nonempty(first);
    for (const Value *value : images) {
        const Image &image = std::get<Image>(*value);
        nonempty(image);
        if (image.height != first.height || image.width != first.width)
            throw NodeError("AssertionError", "All images to be merged must be the same resolution");
    }
    std::vector<std::pair<size_t, size_t>> selections;
    for (size_t i = 0; i < images.size(); ++i)
        for (int32_t c = 0; c < std::get<Image>(*images[i]).channels && selections.size() < 4; ++c)
            selections.emplace_back(i, size_t(c));
    if (selections.size() == 2) selections.push_back(selections[1]);
    outputs.push_back(assemble(first, images, selections));
}

/* chainner:image:merge_transparency (transparency/merge_transparency.py, native_channels.combine_rgb_alpha): the
 * RGB input's first three channels (its one channel thrice when grey) and the alpha channel. */
void merge_transparency(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image &size =
        same_size({&inputs[0], &inputs[1]}, "At least one input must be an image, but both RGB and Alpha are colors.");
    const Image *rgb = as_image(inputs[0]);
    size_t channels = rgb ? size_t(rgb->channels) : std::get<Color>(inputs[0]).values.size();
    size_t step = channels == 1 ? 0 : 1;
    outputs.push_back(assemble(size, {&inputs[0], &inputs[1]}, {{0, 0}, {0, step}, {0, 2 * step}, {1, 0}}));
}

/* chainner:image:fill_alpha (misc/fill_alpha.py, native_alpha_gamma's fill kernels at threshold 0.05): the filled
 * image's colour channels and the input's alpha. */
void fill_alpha(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image &image = std::get<Image>(inputs[0]);
    int64_t method = std::get<int64_t>(inputs[1]);
    size_t height = size_t(image.height), width = size_t(image.width);
    MutableImage filled = MutableImage::allocate(image.height, image.width, 4);
    if (method == 3) {
        check(cn_alpha_nearest(image.data, filled.data(), height, width, 0.05f, 100000, 1));
    } else {
        std::optional<Image> fragments;
        if (method == 1) {
            MutableImage blurred = MutableImage::allocate(image.height, image.width, 4);
            check(cn_alpha_fragment(image.data, blurred.data(), height, width, 0.05f, 6, 5));
            fragments = std::move(blurred).publish();
        }
        check(cn_alpha_extend(fragments ? fragments->data : image.data, filled.data(), height, width, 0.05f, 100000));
    }
    outputs.push_back(select_channels(std::move(filled).publish(), {0, 1, 2}));
    outputs.push_back(select_channels(image, {3}));
}

/* chainner:image:split_transparency (transparency/split_transparency.py): RGB is returned as it is with an opaque
 * alpha; grey becomes BGRA through cvt_color (GRAY2BGRA) first. */
void split_transparency(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image &image = std::get<Image>(inputs[0]);
    if (image.channels == 3) {
        MutableImage opaque = MutableImage::allocate(image.height, image.width, 1);
        std::fill(opaque.data(), opaque.data() + pixels(image), 1.0f);
        outputs.push_back(inputs[0]);
        outputs.push_back(std::move(opaque).publish());
        return;
    }
    Image bgra = image.channels == 1 ? cvt_color(image, 0, 0, 4) : image;
    outputs.push_back(select_channels(bgra, {0, 1, 2}));
    outputs.push_back(select_channels(bgra, {3}));
}

/* chainner:image:chroma_key (misc/chroma_key.py, native_transparency/native_matting): KeyMethod 1 binary keying,
 * 2 trimap matting, native when it outputs the trimap; its alpha matting (closed-form alpha through NumPy's OpenBLAS
 * dot, with printed diagnostics) declines to the Python body. */
void chroma_key(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image &image = std::get<Image>(inputs[0]);
    const Color &key_color = std::get<Color>(inputs[1]);
    bool binary = std::get<int64_t>(inputs[2]) == 1;
    bool output_trimap = std::get<bool>(inputs[8]);
    if (!binary && !output_trimap) throw RunPython{};
    if (image.height == 0 || image.width == 0)
        throw NodeError("ValueError", "Invalid image shape for transparency/color kernel");
    std::vector<float> key(key_color.values.begin(), key_color.values.end());
    size_t count = pixels(image);
    MutableImage out = MutableImage::allocate(image.height, image.width, 4);
    if (binary) {
        check(cn_chroma_key(image.data, key.data(), count, 3, 1, float(number(inputs[3]) / 100), 0, 0, out.data(), 4, 1,
                            nullptr, nullptr));
        outputs.push_back(std::move(out).publish());
        return;
    }
    std::vector<uint8_t> background(count), foreground(count);
    check(cn_chroma_key(image.data, key.data(), count, 3, 1, float(number(inputs[4]) / 100),
                        float(number(inputs[5]) / 100), 1, nullptr, 0, 0, background.data(), foreground.data()));
    size_t height = size_t(image.height), width = size_t(image.width);
    for (auto [mask, input] : {std::pair{&background, 6}, std::pair{&foreground, 7}}) {
        int64_t radius = integer(inputs[size_t(input)]);
        if (radius > 0) {
            std::vector<uint8_t> eroded(count);
            check(cn_matting_erode_mask_u8(mask->data(), eroded.data(), height, width, size_t(radius)));
            *mask = std::move(eroded);
        }
    }
    std::vector<double> trimap(count);
    check(cn_chroma_trimap(background.data(), foreground.data(), trimap.data(), count));
    check(cn_matting_trimap_output(image.data, 3, 1, trimap.data(), count, out.data(), 4, 1));
    outputs.push_back(std::move(out).publish());
}

}  // namespace

CN_RUN_NODES_DECLARE(image_channels)
{
    registry.add("chainner:image:chroma_key", chroma_key);
    registry.add("chainner:image:combine_rgba", combine_rgba);
    registry.add("chainner:image:fill_alpha", fill_alpha);
    registry.add("chainner:image:merge_channels", merge_channels);
    registry.add("chainner:image:merge_transparency", merge_transparency);
    registry.add("chainner:image:split_transparency", split_transparency);
}
