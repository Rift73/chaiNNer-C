/* Native entry points of image_filter nodes (the pattern: image_utility.cpp's note). */
#include "../kernel_status.hpp"
#include "chainner_ext_kernels.h"
#include "dispatch.hpp"
#include "image_blur.hpp"
#include "image_convolve.hpp"
#include "image_morphology.hpp"
#include "image_node_support.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {

using namespace cn::run;

/* chainner:image:gaussian_blur (image_filter/blur/gaussian_blur.py): the input itself for two zero radii, else
 * fast_gaussian_blur, on premultiplied colours for an image with transparency, whose fresh result the output enforce
 * clamps (the body's freeze_normalized(clamp=True)). */
void gaussian_blur(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    double sigma_x = std::get<double>(inputs[1]), sigma_y = std::get<double>(inputs[2]);
    if (sigma_x == 0 && sigma_y == 0) {
        outputs.push_back(inputs[0]);
        return;
    }
    const Image &image = std::get<Image>(inputs[0]);
    if (has_transparency(image))
        outputs.push_back(finish_alpha(fast_gaussian_blur(premultiplied(image), sigma_x, sigma_y), true));
    else
        outputs.push_back(fast_gaussian_blur(image, sigma_x, sigma_y));
}

/* Dilate (maximum) and Erode (image_filter/miscellaneous/dilate.py, erode.py): the input itself for a zero radius
 * or iteration count, else native_filters.morphology, over an image whose transparent pixels hold no colour, whose
 * fresh result the output enforce clamps (the bodies' freeze_normalized(clamp=True)). */
template <bool maximum>
void morphology_node(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    int64_t radius = std::get<int64_t>(inputs[2]), iterations = std::get<int64_t>(inputs[3]);
    if (radius == 0 || iterations == 0) {
        outputs.push_back(inputs[0]);
        return;
    }
    const Image &image = std::get<Image>(inputs[0]);
    outputs.push_back(morphology(has_transparency(image) ? finish_alpha(image, false) : image, std::get<int64_t>(inputs[1]),
                                 radius, iterations, maximum));
}

/* Lens Blur's kernel_params and kernel_scales (image_filter/blur/lens_blur.py): per component count, each
 * component's (a, b, A, B). */
using LensComponent = std::array<double, 4>;
const std::vector<std::vector<LensComponent>> lens_parameters = {
    {{0.862325, 1.624835, 0.767583, 1.862321}},
    {
        {0.886528, 5.268909, 0.411259, -0.548794},
        {1.960518, 1.558213, 0.513282, 4.56111},
    },
    {
        {2.17649, 5.043495, 1.621035, -2.105439},
        {1.019306, 9.027613, -0.28086, -0.162882},
        {2.81511, 1.597273, -0.366471, 10.300301},
    },
    {
        {4.338459, 1.553635, -5.767909, 46.164397},
        {3.839993, 4.693183, 9.795391, -15.227561},
        {2.79188, 8.178137, -3.048324, 0.302959},
        {1.34219, 12.328289, 0.010001, 0.24465},
    },
    {
        {4.892608, 1.685979, -22.356787, 85.91246},
        {4.71187, 4.998496, 35.918936, -28.875618},
        {4.052795, 8.244168, -13.212253, -1.578428},
        {2.929212, 11.900859, 0.507991, 1.816328},
        {1.512961, 16.116382, 0.138051, -0.01},
    },
    {
        {5.143778, 2.079813, -82.326596, 111.231024},
        {5.612426, 6.153387, 113.878661, 58.004879},
        {5.982921, 9.802895, 39.479083, -162.028887},
        {6.505167, 11.059237, -71.286026, 95.027069},
        {3.869579, 14.81052, 1.405746, -3.704914},
        {2.201904, 19.032909, -0.152784, -0.107988},
    },
};
constexpr double lens_scales[] = {1.4, 1.2, 1.2, 1.2, 1.2, 1.2};

/* chainner:image:lens_blur (image_filter/blur/lens_blur.py): the input itself for radius 0; else the body's values
 * from one pass over the image: the power on the interleaved image, then cn_lens_blur's convolutions, compositions and
 * finishing power; with transparency, on premultiplied colours and finished by cn_lens_straight. The body's planar
 * copies have no effect on any value (the output enforce clips the result as it
 * clips the body's frozen view). Native on native_convolution.convolve's spatial path only (spatial_lanes declines the
 * others) and without an exceptional value after the power: where a NaN's payload meets another NaN the body's vector
 * groups decide which survives, so the body runs. */
void lens_blur(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    Image image = std::get<Image>(inputs[0]);
    int64_t radius = std::get<int64_t>(inputs[1]), count = std::get<int64_t>(inputs[2]);
    double gamma = std::get<double>(inputs[3]);
    if (radius == 0) {
        outputs.push_back(inputs[0]);
        return;
    }
    size_t index = size_t(std::clamp<int64_t>(count - 1, 0, 5));
    const std::vector<LensComponent> &parameters = lens_parameters[index];
    size_t length = size_t(2 * radius + 1), components = parameters.size();
    int use_fma = int(dispatch::number("nodes.impl.native_lens", "kernel_dispatch", {}));
    std::vector<float> kernels(components * length * 2), normalized(kernels.size()), weights(components * 2);
    std::vector<double> coefficients(components * 2);
    for (size_t i = 0; i < components; ++i) {
        const LensComponent &p = parameters[i];
        check(cn_lens_kernel(kernels.data() + i * length * 2, size_t(radius), float(lens_scales[index]), float(p[0]),
                             float(p[1]), use_fma));
        coefficients[i * 2] = p[2];
        coefficients[i * 2 + 1] = p[3];
        weights[i * 2] = float(p[2]);
        weights[i * 2 + 1] = float(p[3]);
    }
    check(cn_filter_lens_normalize(kernels.data(), coefficients.data(), normalized.data(), components, length));
    if (image.height == 0 || image.width == 0)
        throw NodeError("ValueError", "Invalid convolution image shape " +
                                          shape_text(Image{image.storage, image.data, image.height, image.width, 1}));
    size_t lanes = spatial_lanes(length);
    bool transparent = has_transparency(image);
    if (transparent) image = premultiplied(image);
    MutableImage power = MutableImage::allocate(image.height, image.width, image.channels);
    check(cn_lens_power(image.data, power.data(), image.values(), float(gamma), 0));
    int exceptional = 0;
    check(cn_image_exceptional(power.data(), image.values(), &exceptional));
    if (exceptional) throw RunPython{};
    MutableImage out = MutableImage::allocate(image.height, image.width, image.channels);
    if (transparent) {
        MutableImage sums = MutableImage::allocate(image.height, image.width, 4);
        check(cn_lens_blur_sums(power.data(), sums.data(), size_t(image.height), size_t(image.width), 4, normalized.data(),
                                weights.data(), components, length, lanes));
        check(cn_lens_straight(sums.data(), out.data(), pixels(image), float(1.0 / gamma)));
    } else {
        check(cn_lens_blur(power.data(), out.data(), size_t(image.height), size_t(image.width), size_t(image.channels),
                           normalized.data(), weights.data(), components, length, lanes, float(1.0 / gamma)));
    }
    outputs.push_back(std::move(out).publish());
}

/* chainner:image:palette_dither (image_filter/quantize/dither_palette.py) through native_neighborhood.palette_dither:
 * the palette's first row prepared by cn_neighborhood_palette_create, applied by cn_neighborhood_palette_apply in
 * mode 0 (none), 2 (diffusion, with its map) or 3 (Riemersma, with its history length). */
void palette_dither(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image &image = std::get<Image>(inputs[0]);
    const Image &palette = std::get<Image>(inputs[1]);
    if (image.channels != palette.channels)
        throw NodeError("AssertionError", "Image and palette must have the same number of channels.");
    const std::string &algorithm = std::get<std::string>(inputs[2]);
    shaped(image);
    /* palette[:1, ...] */
    Image top = palette;
    top.height = std::min(palette.height, 1);
    shaped(top);
    int mode = algorithm == "None" ? 0 : algorithm == "Diffusion" ? 2 : 3;
    int diffusion = 0;
    if (mode == 2) diffusion = diffusion_map(std::get<std::string>(inputs[3]));
    MutableImage out = MutableImage::allocate(image.height, image.width, image.channels);
    uint32_t length = 16;
    if (mode == 3) length = history_length(inputs[4]);
    void *handle = nullptr;
    size_t bytes = 0;
    check(cn_neighborhood_palette_create(top.data, size_t(top.width), size_t(top.channels), &handle, &bytes));
    std::unique_ptr<void, void (*)(void *)> plan(handle, cn_neighborhood_palette_free);
    check(cn_neighborhood_palette_apply(image.data, out.data(), size_t(image.height), size_t(image.width),
                                        size_t(image.channels), plan.get(), mode, diffusion, length,
                                        float(1.0 / double(length)), nullptr));
    outputs.push_back(std::move(out).publish());
}

}  // namespace

CN_RUN_NODES_DECLARE(image_filter)
{
    registry.add("chainner:image:dilate", morphology_node<true>);
    registry.add("chainner:image:erode", morphology_node<false>);
    registry.add("chainner:image:gaussian_blur", gaussian_blur);
    registry.add("chainner:image:lens_blur", lens_blur);
    registry.add("chainner:image:palette_dither", palette_dither);
}
