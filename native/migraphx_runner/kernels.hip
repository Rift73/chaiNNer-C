// The MIGraphX runner's GPU kernels: trt_runner/kernels.cu in HIP, one thread per pixel, each through the convert.h
// function its CPU reference (trt_runner/reference.cpp) uses. No lane intrinsics or shared memory, so a wavefront of
// 32 (RDNA) or 64 (CDNA) runs them alike; blocks of 256 threads are whole wavefronts on both. convert.h marks its
// functions __host__ __device__ for nvcc only, so clang's pragma makes them so here: convert.h is included first, its
// standard headers before it, outside the pragma.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>

#pragma clang force_cuda_host_device begin
#include "convert.h"
#pragma clang force_cuda_host_device end

#include "kernels.h"

namespace cn_migx {
namespace {

constexpr int block = 256;

unsigned blocks(int64_t threads) { return static_cast<unsigned>((threads + block - 1) / block); }

__global__ void to_tensor_kernel(Source source, TileIn tile, void* tensor, Element element, int channels, bool clip) {
    int64_t const plane = static_cast<int64_t>(tile.width) * tile.height;
    int64_t const index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= plane) return;
    int const y = static_cast<int>(index / tile.width), x = static_cast<int>(index % tile.width);
    float v[3];
    cn_trt::tensor_pixel(source, tile.y0 + cn_trt::pad_index(y, tile.real_h, tile.edge),
                         tile.x0 + cn_trt::pad_index(x, tile.real_w, tile.edge), channels, clip, v);
    for (int c = 0; c < channels; ++c) cn_trt::store_element(tensor, element, c * plane + index, v[c]);
}

__global__ void accumulate_kernel(Planar tile, float* acc, int acc_w, int64_t acc_plane, int x0, int y0,
                                  float const* wy, int ny, float const* wx, int nx) {
    int64_t const index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<int64_t>(ny) * nx) return;
    int const y = static_cast<int>(index / nx), x = static_cast<int>(index % nx);
    float const weight = wy[y] * wx[x];
    for (int c = 0; c < tile.channels; ++c) {
        float const value =
            cn_trt::load_element(tile.data, tile.element, c * tile.plane + static_cast<int64_t>(y) * tile.pitch + x);
        float& total = acc[c * acc_plane + static_cast<int64_t>(y0 + y) * acc_w + x0 + x];
        total = total + value * weight;
    }
}

__global__ void store_kernel(Planar source, Result result, bool clip) {
    int64_t const index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<int64_t>(result.width) * result.height) return;
    cn_trt::result_pixel(source, result, index / result.width, index % result.width, clip);
}

}  // namespace

hipError_t launch_to_tensor(Source const& source, TileIn const& tile, void* tensor, Element element, int channels,
                            bool clip, hipStream_t stream) {
    to_tensor_kernel<<<blocks(static_cast<int64_t>(tile.width) * tile.height), block, 0, stream>>>(
        source, tile, tensor, element, channels, clip);
    return hipGetLastError();
}

hipError_t launch_accumulate(Planar const& tile, float* acc, int acc_w, int64_t acc_plane, int x0, int y0,
                             float const* wy, int ny, float const* wx, int nx, hipStream_t stream) {
    accumulate_kernel<<<blocks(static_cast<int64_t>(ny) * nx), block, 0, stream>>>(tile, acc, acc_w, acc_plane, x0,
                                                                                    y0, wy, ny, wx, nx);
    return hipGetLastError();
}

hipError_t launch_store(Planar const& source, Result const& result, bool clip, hipStream_t stream) {
    store_kernel<<<blocks(static_cast<int64_t>(result.width) * result.height), block, 0, stream>>>(source, result,
                                                                                                   clip);
    return hipGetLastError();
}

}  // namespace cn_migx
