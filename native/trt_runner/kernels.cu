// The runner's GPU kernels: one thread per pixel, each through the convert.h function its CPU
// reference (reference.cpp) uses.
#include "kernels.h"

namespace cn_trt {
namespace {

constexpr int block = 256;

unsigned blocks(int64_t threads) { return static_cast<unsigned>((threads + block - 1) / block); }

}  // namespace

// External linkage: nvcc's host stub redeclares an internal kernel (C4211).
__global__ void to_tensor_kernel(Source source, TileIn tile, void* tensor, Element element, int channels, bool clip) {
    int64_t const plane = static_cast<int64_t>(tile.width) * tile.height;
    int64_t const index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= plane) return;
    int const y = static_cast<int>(index / tile.width), x = static_cast<int>(index % tile.width);
    float v[3];
    tensor_pixel(source, tile.y0 + pad_index(y, tile.real_h, tile.edge), tile.x0 + pad_index(x, tile.real_w, tile.edge),
                 channels, clip, v);
    for (int c = 0; c < channels; ++c) store_element(tensor, element, c * plane + index, v[c]);
}

__global__ void accumulate_kernel(Planar tile, float* acc, int acc_w, int64_t acc_plane, int x0, int y0,
                                  float const* wy, int ny, float const* wx, int nx) {
    int64_t const index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<int64_t>(ny) * nx) return;
    int const y = static_cast<int>(index / nx), x = static_cast<int>(index % nx);
    float const weight = wy[y] * wx[x];
    for (int c = 0; c < tile.channels; ++c) {
        float const value = load_element(tile.data, tile.element, c * tile.plane + static_cast<int64_t>(y) * tile.pitch + x);
        float& total = acc[c * acc_plane + static_cast<int64_t>(y0 + y) * acc_w + x0 + x];
        total = total + value * weight;
    }
}

__global__ void store_kernel(Planar source, Result result, bool clip) {
    int64_t const index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<int64_t>(result.width) * result.height) return;
    result_pixel(source, result, index / result.width, index % result.width, clip);
}

cudaError_t launch_to_tensor(Source const& source, TileIn const& tile, void* tensor, Element element,
                             int channels, bool clip, cudaStream_t stream) {
    to_tensor_kernel<<<blocks(static_cast<int64_t>(tile.width) * tile.height), block, 0, stream>>>(
        source, tile, tensor, element, channels, clip);
    return cudaGetLastError();
}

cudaError_t launch_accumulate(Planar const& tile, float* acc, int acc_w, int64_t acc_plane, int x0, int y0,
                              float const* wy, int ny, float const* wx, int nx, cudaStream_t stream) {
    accumulate_kernel<<<blocks(static_cast<int64_t>(ny) * nx), block, 0, stream>>>(
        tile, acc, acc_w, acc_plane, x0, y0, wy, ny, wx, nx);
    return cudaGetLastError();
}

cudaError_t launch_store(Planar const& source, Result const& result, bool clip, cudaStream_t stream) {
    store_kernel<<<blocks(static_cast<int64_t>(result.width) * result.height), block, 0, stream>>>(source, result, clip);
    return cudaGetLastError();
}

}  // namespace cn_trt
