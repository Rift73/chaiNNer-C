// One frame's work, described on the CPU before any CUDA call: the checked arguments, how
// the host planes pack into a device block and back, and the tiles in submission order.
// The GPU runner (runner.cpp) and the CPU reference (reference.cpp) both run it.
#pragma once

#include "convert.h"
#include "plan.h"

#include <cstddef>
#include <vector>

namespace cn_trt {

// A plane in host memory, with any row and column strides in bytes.
struct HostPlane {
    void* data;
    std::ptrdiff_t row_stride;
    std::ptrdiff_t column_stride;
};

struct Format {
    Sample sample;
    int bits;         // integer samples: 1-8 for U8, 9-16 for U16
    bool full_range;  // integer samples
};

// How a frame's host planes map to a packed device block: three planes interleaved in one
// buffer (a BGR image) stay interleaved, other planes become packed planar. Packing and
// unpacking touch only the planes' own samples, never row padding or bytes between them.
struct Packing {
    Layout layout;
    bool interleaved;
    std::size_t bytes;
};

// A tile's place in the upscaled frame and its blend weights' offsets in FrameJob::weights.
struct Place {
    int x0;
    int y0;
    int nx;
    int ny;
    std::size_t wx;
    std::size_t wy;
};

struct FrameJob {
    int width;
    int height;
    int scale;
    Format format;
    Codes codes;
    int planes;
    int result_planes;
    Packing source;
    Packing result;
    std::vector<TileIn> tiles;  // row-major, as tiled_upscale submits them
    std::vector<Place> places;  // one per tile; empty when one tile covers the frame
    std::vector<float> weights;
};

constexpr int max_streams = 16;

// Each throws std::invalid_argument with a message for the user.
void check_runner_arguments(std::size_t engine_size, int device, int streams);
void check_engine_channels(int input_channels, int output_channels);
// convenient_upscale's channel rule: the engine's output channels when they differ from its
// input's, else as many planes as came in.
int result_planes(int planes, int input_channels, int output_channels);
FrameJob describe_frame(HostPlane const* planes, int count, HostPlane const* result, int result_count,
                        int width, int height, Format const& format, Bounds const& bounds, int scale,
                        int input_channels, int output_channels);

void pack(HostPlane const* planes, int count, int width, int height, Sample sample, Packing const& packing, char* block);
void unpack(char const* block, Packing const& packing, HostPlane const* planes, int count, int width, int height,
            Sample sample);

}  // namespace cn_trt
