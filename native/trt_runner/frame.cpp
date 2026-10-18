#include "frame.h"

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace cn_trt {
namespace {

void require(bool condition, std::string const& message) {
    if (!condition) throw std::invalid_argument(message);
}

char* address(HostPlane const& plane, int64_t y, int64_t x) {
    return static_cast<char*>(plane.data) + y * plane.row_stride + x * plane.column_stride;
}

// Three planes of one buffer, one sample apart in each pixel (a BGR image's channels).
bool interleaved(HostPlane const* planes, int count, int width, int bytes) {
    if (count != 3) return false;
    char const* base = static_cast<char const*>(planes[0].data);
    for (int p = 0; p < 3; ++p) {
        if (planes[p].column_stride != 3 * bytes || planes[p].row_stride != planes[0].row_stride) return false;
        base = std::min(base, static_cast<char const*>(planes[p].data));
    }
    if (std::llabs(planes[0].row_stride) < 3LL * width * bytes) return false;
    bool used[3] = {};
    for (int p = 0; p < 3; ++p) {
        std::ptrdiff_t const offset = static_cast<char const*>(planes[p].data) - base;
        if (offset % bytes || offset / bytes > 2 || used[offset / bytes]) return false;
        used[offset / bytes] = true;
    }
    return true;
}

Packing packing_of(HostPlane const* planes, int count, int width, int height, int bytes) {
    int64_t const pixels = static_cast<int64_t>(width) * height;
    Packing packing{};
    packing.interleaved = interleaved(planes, count, width, bytes);
    if (packing.interleaved) {
        char const* base = std::min({static_cast<char const*>(planes[0].data), static_cast<char const*>(planes[1].data),
                                     static_cast<char const*>(planes[2].data)});
        for (int p = 0; p < 3; ++p) packing.layout.offset[p] = (static_cast<char const*>(planes[p].data) - base) / bytes;
        packing.layout.pitch = 3LL * width;
        packing.layout.step = 3;
    } else {
        for (int p = 0; p < count; ++p) packing.layout.offset[p] = p * pixels;
        packing.layout.pitch = width;
        packing.layout.step = 1;
    }
    packing.bytes = static_cast<std::size_t>(count * pixels * bytes);
    return packing;
}

// The interleaved planes' first byte: the plane at offset 0.
HostPlane const& interleaved_base(HostPlane const* planes, Packing const& packing) {
    int p = 0;
    while (packing.layout.offset[p]) ++p;
    return planes[p];
}

}  // namespace

void check_runner_arguments(std::size_t engine_size, int device, int streams) {
    require(engine_size > 0, "The TensorRT engine is empty.");
    require(device >= 0, "The GPU index must be 0 or more, not " + std::to_string(device) + ".");
    require(streams >= 1 && streams <= max_streams, "The stream count must be 1 to " + std::to_string(max_streams) +
                                                        ", not " + std::to_string(streams) + ".");
}

void check_engine_channels(int input_channels, int output_channels) {
    require((input_channels == 1 || input_channels == 3) && (output_channels == 1 || output_channels == 3),
            "The TensorRT runner takes engines with 1 or 3 input and output channels, not " +
                std::to_string(input_channels) + " -> " + std::to_string(output_channels) + ".");
}

int result_planes(int planes, int input_channels, int output_channels) {
    require(planes == 1 || planes == 3,
            "The TensorRT runner takes 1 (gray) or 3 (R, G, B) planes, not " + std::to_string(planes) + ".");
    return input_channels != output_channels ? output_channels : planes;
}

FrameJob describe_frame(HostPlane const* planes, int count, HostPlane const* result, int result_count,
                        int width, int height, Format const& format, Bounds const& bounds, int scale,
                        int input_channels, int output_channels) {
    FrameJob job{};
    check_engine_channels(input_channels, output_channels);
    job.planes = count;
    job.result_planes = result_planes(count, input_channels, output_channels);
    require(result_count == job.result_planes, "This engine makes " + std::to_string(job.result_planes) +
                                                   " result planes from " + std::to_string(count) + ", not " +
                                                   std::to_string(result_count) + ".");
    require(width >= 1 && height >= 1, "Cannot upscale an empty image.");
    require(static_cast<int64_t>(width) * scale <= INT_MAX && static_cast<int64_t>(height) * scale <= INT_MAX,
            "The upscaled image would be too large.");
    if (format.sample == Sample::U8) {
        require(format.bits >= 1 && format.bits <= 8, "8-bit samples hold 1 to 8 bits, not " + std::to_string(format.bits) + ".");
    } else if (format.sample == Sample::U16) {
        require(format.bits >= 9 && format.bits <= 16,
                "16-bit samples hold 9 to 16 bits, not " + std::to_string(format.bits) + ".");
    }
    job.width = width;
    job.height = height;
    job.scale = scale;
    job.format = format;
    if (format.sample == Sample::U8 || format.sample == Sample::U16) job.codes = codes_for(format.bits, format.full_range);
    int const bytes = sample_bytes(format.sample);
    job.source = packing_of(planes, count, width, height, bytes);
    job.result = packing_of(result, result_count, width * scale, height * scale, bytes);

    Plan const plan = plan_tiles(height, width, scale, bounds, tile_overlap);
    bool const single = plan.ys.size() == 1 && plan.xs.size() == 1;
    std::size_t wx_start = 0;
    if (!single) {
        for (auto const& w : plan.wy) {
            job.weights.insert(job.weights.end(), w.begin(), w.end());
            wx_start += w.size();
        }
        for (auto const& w : plan.wx) job.weights.insert(job.weights.end(), w.begin(), w.end());
    }
    std::size_t wy_offset = 0;
    for (std::size_t iy = 0; iy < plan.ys.size(); ++iy) {
        std::size_t wx_offset = wx_start;
        for (std::size_t ix = 0; ix < plan.xs.size(); ++ix) {
            TileIn tile{};
            tile.y0 = plan.ys[iy];
            tile.x0 = plan.xs[ix];
            tile.real_h = std::min(plan.tile_h, height - tile.y0);
            tile.real_w = std::min(plan.tile_w, width - tile.x0);
            tile.height = padded_extent(tile.real_h, bounds.min_h, plan.tile_h, bounds.alignment);
            tile.width = padded_extent(tile.real_w, bounds.min_w, plan.tile_w, bounds.alignment);
            // tiling._pad_tile: reflect needs two pixels along a padded axis
            tile.edge = !((tile.real_h > 1 || tile.height == tile.real_h) && (tile.real_w > 1 || tile.width == tile.real_w));
            job.tiles.push_back(tile);
            if (!single) {
                int const ny = static_cast<int>(plan.wy[iy].size()), nx = static_cast<int>(plan.wx[ix].size());
                job.places.push_back({tile.x0 * scale, tile.y0 * scale, nx, ny, wx_offset, wy_offset});
                wx_offset += static_cast<std::size_t>(nx);
            }
        }
        if (!single) wy_offset += plan.wy[iy].size();
    }
    return job;
}

void pack(HostPlane const* planes, int count, int width, int height, Sample sample, Packing const& packing, char* block) {
    int const bytes = sample_bytes(sample);
    if (packing.interleaved) {
        HostPlane const& base = interleaved_base(planes, packing);
        std::size_t const row = static_cast<std::size_t>(3) * width * bytes;
        for (int y = 0; y < height; ++y) std::memcpy(block + y * row, address(base, y, 0), row);
        return;
    }
    for (int p = 0; p < count; ++p)
        for (int y = 0; y < height; ++y) {
            char* dst = block + (packing.layout.offset[p] + static_cast<int64_t>(y) * width) * bytes;
            if (planes[p].column_stride == bytes) {
                std::memcpy(dst, address(planes[p], y, 0), static_cast<std::size_t>(width) * bytes);
            } else {
                for (int x = 0; x < width; ++x) std::memcpy(dst + x * bytes, address(planes[p], y, x), bytes);
            }
        }
}

void unpack(char const* block, Packing const& packing, HostPlane const* planes, int count, int width, int height,
            Sample sample) {
    int const bytes = sample_bytes(sample);
    if (packing.interleaved) {
        HostPlane const& base = interleaved_base(planes, packing);
        std::size_t const row = static_cast<std::size_t>(3) * width * bytes;
        for (int y = 0; y < height; ++y) std::memcpy(address(base, y, 0), block + y * row, row);
        return;
    }
    for (int p = 0; p < count; ++p)
        for (int y = 0; y < height; ++y) {
            char const* src = block + (packing.layout.offset[p] + static_cast<int64_t>(y) * width) * bytes;
            if (planes[p].column_stride == bytes) {
                std::memcpy(address(planes[p], y, 0), src, static_cast<std::size_t>(width) * bytes);
            } else {
                for (int x = 0; x < width; ++x) std::memcpy(address(planes[p], y, x), src + x * bytes, bytes);
            }
        }
}

}  // namespace cn_trt
