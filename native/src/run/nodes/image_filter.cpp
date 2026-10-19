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
#include <string_view>
#include <vector>

namespace {

using namespace cn::run;

/* chainner:image:gaussian_blur (image_filter/blur/gaussian_blur.py): the input itself for two zero radii, else
 * fast_gaussian_blur, whose fresh result the output enforce clamps (the body's freeze_normalized(clamp=True)). */
void gaussian_blur(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    double sigma_x = std::get<double>(inputs[1]), sigma_y = std::get<double>(inputs[2]);
    if (sigma_x == 0 && sigma_y == 0) {
        outputs.push_back(inputs[0]);
        return;
    }
    outputs.push_back(fast_gaussian_blur(std::get<Image>(inputs[0]), sigma_x, sigma_y));
}

/* Dilate (maximum) and Erode (image_filter/miscellaneous/dilate.py, erode.py): the input itself for a zero radius
 * or iteration count, else native_filters.morphology, whose fresh result the output enforce clamps (the bodies'
 * freeze_normalized(clamp=True)). */
template <bool maximum>
void morphology_node(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    int64_t radius = std::get<int64_t>(inputs[2]), iterations = std::get<int64_t>(inputs[3]);
    if (radius == 0 || iterations == 0) {
        outputs.push_back(inputs[0]);
        return;
    }
    outputs.push_back(morphology(std::get<Image>(inputs[0]), std::get<int64_t>(inputs[1]), radius, iterations, maximum));
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

/* chainner:image:lens_blur (image_filter/blur/lens_blur.py): the input itself for radius 0; else the body's planar
 * pipeline: the power, per component and channel four separable convolutions composed into the planar output, the
 * finishing power and the transpose back to HWC (the output enforce clips it as it clips the body's frozen view).
 * Native on native_convolution.convolve's spatial path only (image_convolve.hpp declines the others). */
void lens_blur(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image &image = std::get<Image>(inputs[0]);
    int64_t radius = std::get<int64_t>(inputs[1]), count = std::get<int64_t>(inputs[2]);
    double gamma = std::get<double>(inputs[3]);
    if (radius == 0) {
        outputs.push_back(inputs[0]);
        return;
    }
    size_t channels = size_t(image.channels), plane = size_t(image.height) * size_t(image.width);
    MutableImage planar = MutableImage::allocate(image.height, image.width, image.channels);
    for (size_t c = 0; c < channels; ++c)
        for (size_t p = 0; p < plane; ++p) planar.data()[c * plane + p] = image.data[p * channels + c];

    size_t index = size_t(std::clamp<int64_t>(count - 1, 0, 5));
    const std::vector<LensComponent> &parameters = lens_parameters[index];
    size_t length = size_t(2 * radius + 1), components = parameters.size();
    int use_fma = int(dispatch::number("nodes.impl.native_lens", "kernel_dispatch", {}));
    std::vector<float> kernels(components * length * 2), normalized(kernels.size());
    std::vector<double> coefficients(components * 2);
    for (size_t i = 0; i < components; ++i) {
        const LensComponent &p = parameters[i];
        check(cn_lens_kernel(kernels.data() + i * length * 2, size_t(radius), float(lens_scales[index]), float(p[0]),
                             float(p[1]), use_fma));
        coefficients[i * 2] = p[2];
        coefficients[i * 2 + 1] = p[3];
    }
    check(cn_filter_lens_normalize(kernels.data(), coefficients.data(), normalized.data(), components, length));

    MutableImage power = MutableImage::allocate(image.height, image.width, image.channels);
    check(cn_lens_power(planar.data(), power.data(), image.values(), float(gamma), 0));
    Image powered = std::move(power).publish();
    MutableImage composed = MutableImage::allocate(image.height, image.width, image.channels);
    for (size_t i = 0; i < components; ++i) {
        Kernel row{1, length, {}}, imaginary_row{1, length, {}};
        for (size_t k = 0; k < length; ++k) {
            row.values.push_back(double(normalized[(i * length + k) * 2]));
            imaginary_row.values.push_back(double(normalized[(i * length + k) * 2 + 1]));
        }
        Kernel column{length, 1, row.values}, imaginary_column{length, 1, imaginary_row.values};
        for (size_t c = 0; c < channels; ++c) {
            // Channel c's plane of the planar image, a one-channel image of its own.
            Image source{powered.storage, powered.data + c * plane, image.height, image.width, 1};
            Image inter_real = convolve(source, row, 0);
            Image inter_imaginary = convolve(source, imaginary_row, 0);
            Image f1 = convolve(inter_real, column, 0);
            Image f2 = convolve(inter_real, imaginary_column, 0);
            Image f3 = convolve(inter_imaginary, column, 0);
            Image f4 = convolve(inter_imaginary, imaginary_column, 0);
            check(cn_lens_compose(f1.data, f2.data, f3.data, f4.data, composed.data() + c * plane, plane,
                                  float(parameters[i][2]), float(parameters[i][3]), i != 0));
        }
    }
    MutableImage finished = MutableImage::allocate(image.height, image.width, image.channels);
    check(cn_lens_power(composed.data(), finished.data(), image.values(), float(1.0 / gamma), 1));
    MutableImage out = MutableImage::allocate(image.height, image.width, image.channels);
    for (size_t c = 0; c < channels; ++c)
        for (size_t p = 0; p < plane; ++p) out.data()[p * channels + c] = finished.data()[c * plane + p];
    outputs.push_back(std::move(out).publish());
}

/* Palette Dither's error diffusion maps (ErrorDiffusionMap values) in native_neighborhood.DIFFUSION_IDS order. */
constexpr std::array<std::string_view, 8> diffusion_maps = {"FS", "JJN", "ST", "A", "B", "S", "S2", "SL"};

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
    if (image.height == 0 || image.width == 0) throw NodeError("ValueError", "Invalid image shape " + shape_text(image));
    /* palette[:1, ...] */
    Image top = palette;
    top.height = std::min(palette.height, 1);
    if (top.height == 0 || top.width == 0) throw NodeError("ValueError", "Invalid image shape " + shape_text(top));
    int mode = algorithm == "None" ? 0 : algorithm == "Diffusion" ? 2 : 3;
    int diffusion = 0;
    if (mode == 2) {
        const std::string &map = std::get<std::string>(inputs[3]);
        diffusion = int(std::find(diffusion_maps.begin(), diffusion_maps.end(), map) - diffusion_maps.begin());
    }
    MutableImage out = MutableImage::allocate(image.height, image.width, image.channels);
    int64_t length = 16;
    if (mode == 3) {
        const int64_t *history = std::get_if<int64_t>(&inputs[4]);
        /* A Python int beyond int64 (a PyRef) is beyond the range too. */
        if (!history || *history < 2 || *history > int64_t(UINT32_MAX))
            throw NodeError("ValueError", "History length must be an integer in [2,4294967295]");
        length = *history;
    }
    void *handle = nullptr;
    size_t bytes = 0;
    check(cn_neighborhood_palette_create(top.data, size_t(top.width), size_t(top.channels), &handle, &bytes));
    std::unique_ptr<void, void (*)(void *)> plan(handle, cn_neighborhood_palette_free);
    check(cn_neighborhood_palette_apply(image.data, out.data(), size_t(image.height), size_t(image.width),
                                        size_t(image.channels), plan.get(), mode, diffusion, uint32_t(length),
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
