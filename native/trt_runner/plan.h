// The runner's tile planner: nodes/impl/tensorrt/tiling.py in C++, with the same tiles, the
// same padding and bit-identical float32 blend weights. No CUDA or TensorRT.
#pragma once

#include <vector>

namespace cn_trt {

// tiling.TILE_ALIGNMENT and TILE_OVERLAP.
constexpr int tile_alignment = 64;
constexpr int tile_overlap = 16;

// tiling.ShapeBounds: the input height/width range of the engine's optimization profile
// and the multiple its input sizes must be.
struct Bounds {
    int min_h;
    int min_w;
    int max_h;
    int max_w;
    int alignment;
};

int tile_extent(int max_size, int min_size, int alignment);
int padded_extent(int length, int min_size, int tile, int alignment);
// Throws std::invalid_argument when the engine input is too small to tile the image.
std::vector<int> tile_starts(int length, int tile, int overlap);
std::vector<std::vector<float>> axis_weights(std::vector<int> const& starts, int tile, int length,
                                             int scale, int overlap);

// The tiles of a frame: tiled_upscale's tile sizes, start offsets and blend weights (empty
// for a frame that fits one tile, which needs none).
struct Plan {
    int tile_h;
    int tile_w;
    std::vector<int> ys;
    std::vector<int> xs;
    std::vector<std::vector<float>> wy;
    std::vector<std::vector<float>> wx;
};

Plan plan_tiles(int height, int width, int scale, Bounds const& bounds, int overlap);

}  // namespace cn_trt
