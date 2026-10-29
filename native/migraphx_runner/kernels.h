// The MIGraphX runner's GPU kernels (kernels.hip), the TensorRT runner's (trt_runner/kernels.cu) in HIP, enqueued on a
// stream; trt_runner/reference.h has their CPU references with the same arguments.
#pragma once

#include "convert.h"

#include <hip/hip_runtime_api.h>

namespace cn_migx {

using cn_trt::Element;
using cn_trt::Planar;
using cn_trt::Result;
using cn_trt::Source;
using cn_trt::TileIn;

hipError_t launch_to_tensor(Source const& source, TileIn const& tile, void* tensor, Element element, int channels,
                            bool clip, hipStream_t stream);
hipError_t launch_accumulate(Planar const& tile, float* acc, int acc_w, int64_t acc_plane, int x0, int y0,
                             float const* wy, int ny, float const* wx, int nx, hipStream_t stream);
hipError_t launch_store(Planar const& source, Result const& result, bool clip, hipStream_t stream);

}  // namespace cn_migx
