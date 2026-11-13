/* Native entry points of image_dimension nodes (the pattern: image_utility.cpp's note). */
#include "image_node_support.hpp"
#include "image_resize.hpp"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string_view>

namespace {

using namespace cn::run;

/* Whether text ends in six of run, a digit from low to high and one more digit. */
bool ends_in_run(std::string_view text, char run, char low, char high)
{
    size_t size = text.size();
    if (size < 8 || text[size - 1] < '0' || text[size - 1] > '9' || text[size - 2] < low || text[size - 2] > high)
        return false;
    return text.substr(size - 8, 6).find_first_not_of(run) == std::string_view::npos;
}

/* The clean-up the type system (Navi's fixRoundingError) gives every product: a number whose 16 significant digits
 * end in 000000, a digit 0-3 and one more keeps 13 of them; a fraction whose shortest digits (JavaScript's String,
 * plain digits from 1e-6 up) end in 999999 and 6-9, or 000000 and 0-3, and one more digit keeps 12. */
double fix_rounding_error(double n)
{
    if (!std::isfinite(n)) return n;
    char text[64];
    int length = std::snprintf(text, sizeof text, "%.15e", n);
    std::string_view digits(text, length);
    if (ends_in_run(digits.substr(0, digits.find('e')), '0', '0', '3')) {
        std::snprintf(text, sizeof text, "%.12e", n);
        return std::strtod(text, nullptr);
    }
    if (n == std::floor(n) || std::fabs(n) < 1e-6) return n;
    std::string_view shortest(text, std::to_chars(text, text + sizeof text, n, std::chars_format::fixed).ptr);
    if (ends_in_run(shortest, '9', '6', '9') || ends_in_run(shortest, '0', '0', '3')) {
        std::snprintf(text, sizeof text, "%.11e", n);
        return std::strtod(text, nullptr);
    }
    return n;
}

/* side * scale / 100 as the type system computes it: side * scale, then that times 1 / 100, each product through
 * fix_rounding_error. */
double percent_of(double side, double scale)
{
    return fix_rounding_error(fix_rounding_error(side * scale) * (1.0 / 100));
}

/* chainner:image:resize (image_dimension/resize/resize.py): Percentage (0) scales both sides, percent_of(side, scale)
 * with round_half_up and at least 1 pixel, Absolute (1) takes the width and height; Lambda (7) is DPID's; Linear Light (8)
 * resizes with the Transfer (9) curve. */
void resize_node(NodeContext &, std::span<const Value> inputs, std::vector<Value> &outputs)
{
    const Image &image = std::get<Image>(inputs[0]);
    Extent width, height;
    if (std::get<int64_t>(inputs[1]) == 0) {
        double scale = std::get<double>(inputs[2]);
        width = at_least_one(Extent::round_half_up(percent_of(image.width, scale)));
        height = at_least_one(Extent::round_half_up(percent_of(image.height, scale)));
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
