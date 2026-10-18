#include "plan.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace cn_trt {

int tile_extent(int max_size, int min_size, int alignment) {
    int const aligned = max_size / alignment * alignment;
    return aligned >= min_size ? aligned : max_size;
}

int padded_extent(int length, int min_size, int tile, int alignment) {
    int const aligned = (length + alignment - 1) / alignment * alignment;
    return std::min(std::max(aligned, min_size), tile);
}

std::vector<int> tile_starts(int length, int tile, int overlap) {
    if (length <= tile) return {0};
    int const step = tile - 3 * overlap;
    if (step <= 0) {
        std::string const t = std::to_string(tile), o = std::to_string(3 * overlap);
        throw std::invalid_argument(
            "This image (" + std::to_string(length) + " px) is larger than the TensorRT engine's input (" +
            t + " px), and tiling it needs an engine input over " + o + " px. The engine still upscales "
            "images up to " + t + " px; for larger ones, build it with a size over " + o + " px.");
    }
    std::vector<int> starts{0};
    while (starts.back() + tile < length) starts.push_back(std::min(starts.back() + step, length - tile));
    return starts;
}

// Each operation in tiling.axis_weights' float32 order: the ramps' (k + 0.5) / band and
// (k - 0.5) / band, the running total in tile order, then each weight over the total.
std::vector<std::vector<float>> axis_weights(std::vector<int> const& starts, int tile, int length,
                                             int scale, int overlap) {
    size_t const count = starts.size();
    std::vector<std::vector<float>> raw(count);
    int const cut = overlap * scale;
    for (size_t i = 0; i < count; ++i) {
        int const n = std::min(tile, length - starts[i]) * scale;
        std::vector<float>& w = raw[i];
        w.assign(static_cast<size_t>(n), 1.0f);
        if (i > 0) {
            int const band = (starts[i - 1] + tile - starts[i]) * scale - 2 * cut;
            std::fill(w.begin(), w.begin() + cut, 0.0f);
            for (int k = 0; k < band; ++k)
                w[static_cast<size_t>(cut + k)] *= (static_cast<float>(k) + 0.5f) / static_cast<float>(band);
        }
        if (i + 1 < count) {
            int const band = (starts[i] + tile - starts[i + 1]) * scale - 2 * cut;
            std::fill(w.end() - cut, w.end(), 0.0f);
            for (int k = 0; k < band; ++k)
                w[static_cast<size_t>(n - cut - band + k)] *=
                    (static_cast<float>(band - k) - 0.5f) / static_cast<float>(band);
        }
    }
    std::vector<float> total(static_cast<size_t>(length) * static_cast<size_t>(scale), 0.0f);
    for (size_t i = 0; i < count; ++i) {
        size_t const offset = static_cast<size_t>(starts[i]) * static_cast<size_t>(scale);
        for (size_t k = 0; k < raw[i].size(); ++k) total[offset + k] += raw[i][k];
    }
    for (size_t i = 0; i < count; ++i) {
        size_t const offset = static_cast<size_t>(starts[i]) * static_cast<size_t>(scale);
        for (size_t k = 0; k < raw[i].size(); ++k) raw[i][k] /= total[offset + k];
    }
    return raw;
}

Plan plan_tiles(int height, int width, int scale, Bounds const& bounds, int overlap) {
    Plan plan;
    plan.tile_h = tile_extent(bounds.max_h, bounds.min_h, bounds.alignment);
    plan.tile_w = tile_extent(bounds.max_w, bounds.min_w, bounds.alignment);
    plan.ys = tile_starts(height, plan.tile_h, overlap);
    plan.xs = tile_starts(width, plan.tile_w, overlap);
    if (plan.ys.size() > 1 || plan.xs.size() > 1) {
        plan.wy = axis_weights(plan.ys, plan.tile_h, height, scale, overlap);
        plan.wx = axis_weights(plan.xs, plan.tile_w, width, scale, overlap);
    }
    return plan;
}

}  // namespace cn_trt
