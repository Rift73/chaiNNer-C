// The runner's GPU kernels (kernels.cu), enqueued on a stream; reference.h has their CPU
// references with the same arguments.
#pragma once

#include "convert.h"

#include <cuda_runtime_api.h>

namespace cn_trt {

cudaError_t launch_to_tensor(Source const& source, TileIn const& tile, void* tensor, Element element,
                             int channels, bool clip, cudaStream_t stream);
cudaError_t launch_accumulate(Planar const& tile, float* acc, int acc_w, int64_t acc_plane, int x0, int y0,
                              float const* wy, int ny, float const* wx, int nx, cudaStream_t stream);
cudaError_t launch_store(Planar const& source, Result const& result, bool clip, cudaStream_t stream);

}  // namespace cn_trt
