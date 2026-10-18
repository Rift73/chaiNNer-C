// CPU references of the runner's three GPU kernels (kernels.cu), on host memory, pixel by
// pixel through the same convert.h functions, and of its frame loop (runner.cpp) with a
// stand-in for TensorRT; the GPU tests compare the kernels and the runner with them.
#pragma once

#include "convert.h"
#include "frame.h"

#include <functional>

namespace cn_trt {

// One tile of the frame as the engine's input tensor (channels x tile.height x tile.width).
void to_tensor(Source const& source, TileIn const& tile, void* tensor, Element element, int channels, bool clip);

// tiling._blend: acc += tile output * (wy[y] * wx[x]) over the ny x nx output pixels at
// (x0, y0) of the float32 accumulator (channel planes of acc_plane elements, rows of acc_w).
void accumulate(Planar const& tile, float* acc, int acc_w, int64_t acc_plane, int x0, int y0,
                float const* wy, int ny, float const* wx, int nx);

// The upscaled frame (the single tile's output or the accumulator) as result planes.
void store(Planar const& source, Result const& result, bool clip);

// The engine for reference_upscale: it reads a tile's input tensor (input channels x
// tile.height x tile.width) and writes its output (output channels x tile.height * scale x
// tile.width * scale).
using Engine = std::function<void(TileIn const& tile, void const* input, void* output)>;

// The runner's frame loop on the CPU: what Runner::upscale makes from the same frame when
// TensorRT computes what `engine` does.
void reference_upscale(FrameJob const& job, HostPlane const* planes, HostPlane const* result, int input_channels,
                       int output_channels, Element input, Element output, Engine const& engine, bool clip);

}  // namespace cn_trt
