/* Native entry points of image_dimension nodes (the pattern: image_utility.cpp's note). */
#include "image_node_support.hpp"
#include "image_resize.hpp"

namespace {

using namespace cn::run;

/* chainner:image:resize (image_dimension/resize/resize.py): Percentage (0) scales both sides by scale / 100 with
 * round_half_up and at least 1 pixel, Absolute (1) takes the width and height; Lambda (7) is DPID's; Linear Light (8)
 * resizes with the Transfer (9) curve. */
void resize_node(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image &image = std::get<Image>(inputs[0]);
    Extent width, height;
    if (std::get<int64_t>(inputs[1]) == 0) {
        double scale = std::get<double>(inputs[2]);
        width = at_least_one(Extent::round_half_up(double(image.width) * (scale / 100)));
        height = at_least_one(Extent::round_half_up(double(image.height) * (scale / 100)));
    } else {
        width = extent(inputs[3]);
        height = extent(inputs[4]);
    }
    std::optional<int64_t> light;
    if (std::get<bool>(inputs[8])) light = std::get<int64_t>(inputs[9]);
    outputs.push_back(resize(image, width, height, std::get<int64_t>(inputs[5]), std::get<bool>(inputs[6]), light,
                             number(inputs[7])));
}

}  // namespace

CN_RUN_NODES_DECLARE(image_dimension)
{
    registry.add("chainner:image:resize", resize_node);
}
