/* nodes/impl/resize.py's resize() for native entry points (Resize, Resize To Side, Average Color Fix, ...). */
#pragma once

#include "run_engine.hpp"

#include <cstdint>
#include <optional>

namespace cn::run {

/* A nonnegative Python int that a node computes, exactly: mantissa * 2^exponent, with mantissa < 2^63 (exponent is
 * 0 for every value below 2^63). Python's ints are unbounded; a percentage of an image size can exceed int64. */
struct Extent {
    uint64_t mantissa = 0;
    int exponent = 0;

    static Extent of(int64_t value);  // value >= 0
    /* math.floor(value + 0.5) (utils.round_half_up) of a value >= -0.5: ValueError for NaN and OverflowError for an
     * infinity, as math.floor raises them. */
    static Extent round_half_up(double value);
};

/* ResizeFilter values (nodes/impl/resize.py). */
enum ResizeFilterValue : int64_t {
    filter_auto = -1,
    filter_nearest = 0,
    filter_lanczos = 1,
    filter_box = 4,
    filter_dpid = 12
};

/* resize(img, (width, height), filter, separate_alpha, False, dpid_lambda, light) as the nodes call it, light a
 * LightTransfer value (linear_light.h's CN_LIGHT_*) or none: the resized image, or img itself where the body returns
 * img.copy() (a same-size Nearest, Box or DPID resize); the body's errors in its order. Throws RunPython before any
 * output when the result would exceed the engine's int32 image dimensions. */
Value resize(const Image &image, Extent width, Extent height, int64_t filter, bool separate_alpha,
             std::optional<int64_t> light, double dpid_lambda);

}  // namespace cn::run
