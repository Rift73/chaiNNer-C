#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <new>

#if defined(_WIN32)
#define DRFT_PLUGIN_EXPORT extern "C" __declspec(dllexport)
#else
#define DRFT_PLUGIN_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace
{
#if defined(DRFT_ENABLE_OCAB_QKV_PACK_TOKEN_MAJOR)
constexpr char kPluginName[] = "DrftOCABQKVPackTokenMajor_TRT";
#else
constexpr char kPluginName[] = "DrftOCABQKVPack_TRT";
#endif
constexpr char kPluginVersion[] = "1";
constexpr int32_t kTactic = 1;
constexpr int32_t kLinearThreads = 256;
constexpr int32_t kBf16PerVector = 8;
#if !defined(DRFT_ENABLE_OCAB_QKV_PACK_TOKEN_MAJOR)
constexpr int32_t kTransposeTile = 32;
constexpr int32_t kTransposeRowsPerBlock = 8;
#endif

__device__ __forceinline__ int64_t qkvOffset(int32_t batch, int32_t y,
    int32_t x, int32_t component, int32_t channel, int32_t height,
    int32_t width, int32_t channels)
{
    return (((static_cast<int64_t>(batch) * height + y) * width + x) * 3
               + component)
              * channels
        + channel;
}

#if !defined(DRFT_ENABLE_OCAB_QKV_PACK_TOKEN_MAJOR)
__global__ void packQueryKernel(__nv_bfloat16 const* input,
    __nv_bfloat16* query, int32_t height, int32_t width, int32_t channels,
    int32_t heads, int32_t queryWindow, int64_t elements)
{
    int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (linear >= elements)
    {
        return;
    }

    int32_t const headDim = channels / heads;
    int32_t const queryTokens = queryWindow * queryWindow;
    int32_t const windowsY = height / queryWindow;
    int32_t const windowsX = width / queryWindow;
    int32_t const windowsPerImage = windowsY * windowsX;

    int64_t index = linear;
    int32_t const depth = static_cast<int32_t>(index % headDim);
    index /= headDim;
    int32_t const token = static_cast<int32_t>(index % queryTokens);
    index /= queryTokens;
    int32_t const head = static_cast<int32_t>(index % heads);
    index /= heads;
    int32_t const window = static_cast<int32_t>(index);
    int32_t const batch = window / windowsPerImage;
    int32_t const imageWindow = window - batch * windowsPerImage;
    int32_t const windowY = imageWindow / windowsX;
    int32_t const windowX = imageWindow - windowY * windowsX;
    int32_t const y = windowY * queryWindow + token / queryWindow;
    int32_t const x = windowX * queryWindow + token % queryWindow;
    int32_t const channel = head * headDim + depth;
    query[linear] = input[qkvOffset(
        batch, y, x, 0, channel, height, width, channels)];
}

__global__ void packValueKernel(__nv_bfloat16 const* input,
    __nv_bfloat16* value, int32_t height, int32_t width, int32_t channels,
    int32_t heads, int32_t queryWindow, int32_t keyWindow, int32_t padding,
    int64_t elements)
{
    int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (linear >= elements)
    {
        return;
    }

    int32_t const headDim = channels / heads;
    int32_t const keyTokens = keyWindow * keyWindow;
    int32_t const windowsY = height / queryWindow;
    int32_t const windowsX = width / queryWindow;
    int32_t const windowsPerImage = windowsY * windowsX;

    int64_t index = linear;
    int32_t const depth = static_cast<int32_t>(index % headDim);
    index /= headDim;
    int32_t const token = static_cast<int32_t>(index % keyTokens);
    index /= keyTokens;
    int32_t const head = static_cast<int32_t>(index % heads);
    index /= heads;
    int32_t const window = static_cast<int32_t>(index);
    int32_t const batch = window / windowsPerImage;
    int32_t const imageWindow = window - batch * windowsPerImage;
    int32_t const windowY = imageWindow / windowsX;
    int32_t const windowX = imageWindow - windowY * windowsX;
    int32_t const y = windowY * queryWindow + token / keyWindow - padding;
    int32_t const x = windowX * queryWindow + token % keyWindow - padding;
    int32_t const channel = head * headDim + depth;
    value[linear] = (static_cast<uint32_t>(y) < static_cast<uint32_t>(height)
            && static_cast<uint32_t>(x) < static_cast<uint32_t>(width))
        ? input[qkvOffset(batch, y, x, 2, channel, height, width, channels)]
        : __float2bfloat16(0.0F);
}
#endif

#if defined(DRFT_ENABLE_OCAB_QKV_PACK_VECTORIZED)
__global__ void packQueryVectorKernel(__nv_bfloat16 const* input,
    uint4* query, int32_t height, int32_t width, int32_t channels,
    int32_t heads, int32_t queryWindow, int64_t vectors)
{
    int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (linear >= vectors)
    {
        return;
    }

    int32_t const headDim = channels / heads;
    int32_t const vectorsPerHead = headDim / kBf16PerVector;
    int32_t const queryTokens = queryWindow * queryWindow;
    int32_t const windowsY = height / queryWindow;
    int32_t const windowsX = width / queryWindow;
    int32_t const windowsPerImage = windowsY * windowsX;

    int64_t index = linear;
    int32_t const depthVector
        = static_cast<int32_t>(index % vectorsPerHead);
    index /= vectorsPerHead;
    int32_t const token = static_cast<int32_t>(index % queryTokens);
    index /= queryTokens;
    int32_t const head = static_cast<int32_t>(index % heads);
    index /= heads;
    int32_t const window = static_cast<int32_t>(index);
    int32_t const batch = window / windowsPerImage;
    int32_t const imageWindow = window - batch * windowsPerImage;
    int32_t const windowY = imageWindow / windowsX;
    int32_t const windowX = imageWindow - windowY * windowsX;
    int32_t const y = windowY * queryWindow + token / queryWindow;
    int32_t const x = windowX * queryWindow + token % queryWindow;
    int32_t const channel
        = head * headDim + depthVector * kBf16PerVector;
    int64_t const sourceOffset
        = qkvOffset(batch, y, x, 0, channel, height, width, channels);
    query[linear] = *reinterpret_cast<uint4 const*>(input + sourceOffset);
}

__global__ void packValueVectorKernel(__nv_bfloat16 const* input,
    uint4* value, int32_t height, int32_t width, int32_t channels,
    int32_t heads, int32_t queryWindow, int32_t keyWindow, int32_t padding,
    int64_t vectors)
{
    int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (linear >= vectors)
    {
        return;
    }

    int32_t const headDim = channels / heads;
    int32_t const vectorsPerHead = headDim / kBf16PerVector;
    int32_t const keyTokens = keyWindow * keyWindow;
    int32_t const windowsY = height / queryWindow;
    int32_t const windowsX = width / queryWindow;
    int32_t const windowsPerImage = windowsY * windowsX;

    int64_t index = linear;
    int32_t const depthVector
        = static_cast<int32_t>(index % vectorsPerHead);
    index /= vectorsPerHead;
    int32_t const token = static_cast<int32_t>(index % keyTokens);
    index /= keyTokens;
    int32_t const head = static_cast<int32_t>(index % heads);
    index /= heads;
    int32_t const window = static_cast<int32_t>(index);
    int32_t const batch = window / windowsPerImage;
    int32_t const imageWindow = window - batch * windowsPerImage;
    int32_t const windowY = imageWindow / windowsX;
    int32_t const windowX = imageWindow - windowY * windowsX;
    int32_t const y = windowY * queryWindow + token / keyWindow - padding;
    int32_t const x = windowX * queryWindow + token % keyWindow - padding;

    uint4 packed = make_uint4(0U, 0U, 0U, 0U);
    if (static_cast<uint32_t>(y) < static_cast<uint32_t>(height)
        && static_cast<uint32_t>(x) < static_cast<uint32_t>(width))
    {
        int32_t const channel
            = head * headDim + depthVector * kBf16PerVector;
        int64_t const sourceOffset
            = qkvOffset(batch, y, x, 2, channel, height, width, channels);
        packed = *reinterpret_cast<uint4 const*>(input + sourceOffset);
    }
    value[linear] = packed;
}
#endif

#if defined(DRFT_ENABLE_OCAB_QKV_PACK_TOKEN_MAJOR)
__global__ void packQueryTokenMajorVectorKernel(
    __nv_bfloat16 const* input, uint4* query, int32_t height,
    int32_t width, int32_t channels, int32_t heads, int32_t queryWindow,
    int32_t batchHeads, int64_t vectors)
{
    int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (linear >= vectors)
    {
        return;
    }

    int32_t const headDim = channels / heads;
    int32_t const vectorsPerHead = headDim / kBf16PerVector;
    int32_t const windowsX = width / queryWindow;
    int32_t const windowsPerImage = (height / queryWindow) * windowsX;

    int64_t index = linear;
    int32_t const depthVector
        = static_cast<int32_t>(index % vectorsPerHead);
    index /= vectorsPerHead;
    int32_t const windowHead = static_cast<int32_t>(index % batchHeads);
    int32_t const token = static_cast<int32_t>(index / batchHeads);
    int32_t const head = windowHead % heads;
    int32_t const window = windowHead / heads;
    int32_t const batch = window / windowsPerImage;
    int32_t const imageWindow = window - batch * windowsPerImage;
    int32_t const windowY = imageWindow / windowsX;
    int32_t const windowX = imageWindow - windowY * windowsX;
    int32_t const y = windowY * queryWindow + token / queryWindow;
    int32_t const x = windowX * queryWindow + token % queryWindow;
    int32_t const channel
        = head * headDim + depthVector * kBf16PerVector;
    int64_t const sourceOffset
        = qkvOffset(batch, y, x, 0, channel, height, width, channels);
    query[linear] = *reinterpret_cast<uint4 const*>(input + sourceOffset);
}

__global__ void packKeyValueTokenMajorVectorKernel(
    __nv_bfloat16 const* input, uint4* key, uint4* value,
    int32_t height, int32_t width, int32_t channels, int32_t heads,
    int32_t queryWindow, int32_t keyWindow, int32_t padding,
    int32_t batchHeads, int64_t vectors)
{
    int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (linear >= vectors)
    {
        return;
    }

    int32_t const headDim = channels / heads;
    int32_t const vectorsPerHead = headDim / kBf16PerVector;
    int32_t const windowsX = width / queryWindow;
    int32_t const windowsPerImage = (height / queryWindow) * windowsX;

    int64_t index = linear;
    int32_t const depthVector
        = static_cast<int32_t>(index % vectorsPerHead);
    index /= vectorsPerHead;
    int32_t const windowHead = static_cast<int32_t>(index % batchHeads);
    int32_t const token = static_cast<int32_t>(index / batchHeads);
    int32_t const head = windowHead % heads;
    int32_t const window = windowHead / heads;
    int32_t const batch = window / windowsPerImage;
    int32_t const imageWindow = window - batch * windowsPerImage;
    int32_t const windowY = imageWindow / windowsX;
    int32_t const windowX = imageWindow - windowY * windowsX;
    int32_t const y = windowY * queryWindow + token / keyWindow - padding;
    int32_t const x = windowX * queryWindow + token % keyWindow - padding;

    uint4 packedKey = make_uint4(0U, 0U, 0U, 0U);
    uint4 packedValue = make_uint4(0U, 0U, 0U, 0U);
    if (static_cast<uint32_t>(y) < static_cast<uint32_t>(height)
        && static_cast<uint32_t>(x) < static_cast<uint32_t>(width))
    {
        int32_t const channel
            = head * headDim + depthVector * kBf16PerVector;
        int64_t const keyOffset
            = qkvOffset(batch, y, x, 1, channel, height, width, channels);
        int64_t const valueOffset
            = qkvOffset(batch, y, x, 2, channel, height, width, channels);
        packedKey = *reinterpret_cast<uint4 const*>(input + keyOffset);
        packedValue = *reinterpret_cast<uint4 const*>(input + valueOffset);
    }
    key[linear] = packedKey;
    value[linear] = packedValue;
}

cudaError_t launchPackQKVTokenMajor(void const* input, void* query,
    void* key, void* value, int32_t batch, int32_t height, int32_t width,
    int32_t channels, int32_t heads, int32_t queryWindow,
    int32_t keyWindow, cudaStream_t stream)
{
    int32_t const headDim = channels / heads;
    if (input == nullptr || query == nullptr || key == nullptr
        || value == nullptr || batch <= 0 || height <= 0 || width <= 0
        || channels <= 0 || heads <= 0 || channels % heads != 0
        || channels % kBf16PerVector != 0
        || headDim % kBf16PerVector != 0 || queryWindow <= 0
        || keyWindow < queryWindow || (keyWindow - queryWindow) % 2 != 0
        || height % queryWindow != 0 || width % queryWindow != 0)
    {
        return cudaErrorInvalidValue;
    }

    int32_t const windows = batch * (height / queryWindow)
        * (width / queryWindow);
    int32_t const batchHeads = windows * heads;
    int32_t const vectorsPerHead = headDim / kBf16PerVector;
    int32_t const queryTokens = queryWindow * queryWindow;
    int32_t const keyTokens = keyWindow * keyWindow;
    int32_t const padding = (keyWindow - queryWindow) / 2;
    int64_t const queryVectors = static_cast<int64_t>(queryTokens)
        * batchHeads * vectorsPerHead;
    int64_t const keyValueVectors = static_cast<int64_t>(keyTokens)
        * batchHeads * vectorsPerHead;
    int32_t const queryBlocks = static_cast<int32_t>(
        (queryVectors + kLinearThreads - 1) / kLinearThreads);
    int32_t const keyValueBlocks = static_cast<int32_t>(
        (keyValueVectors + kLinearThreads - 1) / kLinearThreads);

    packQueryTokenMajorVectorKernel<<<queryBlocks, kLinearThreads, 0, stream>>>(
        static_cast<__nv_bfloat16 const*>(input),
        static_cast<uint4*>(query), height, width, channels, heads,
        queryWindow, batchHeads, queryVectors);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        return error;
    }
    packKeyValueTokenMajorVectorKernel<<<
        keyValueBlocks, kLinearThreads, 0, stream>>>(
        static_cast<__nv_bfloat16 const*>(input), static_cast<uint4*>(key),
        static_cast<uint4*>(value), height, width, channels, heads,
        queryWindow, keyWindow, padding, batchHeads, keyValueVectors);
    return cudaGetLastError();
}
#endif

#if !defined(DRFT_ENABLE_OCAB_QKV_PACK_TOKEN_MAJOR)
__global__ void packKeyTransposedKernel(__nv_bfloat16 const* input,
    __nv_bfloat16* key, int32_t height, int32_t width, int32_t channels,
    int32_t heads, int32_t queryWindow, int32_t keyWindow, int32_t padding)
{
    __shared__ __nv_bfloat16 tile[kTransposeTile][kTransposeTile + 1];

    int32_t const headDim = channels / heads;
    int32_t const keyTokens = keyWindow * keyWindow;
    int32_t const windowsY = height / queryWindow;
    int32_t const windowsX = width / queryWindow;
    int32_t const windowsPerImage = windowsY * windowsX;
    int32_t const windowHead = static_cast<int32_t>(blockIdx.x);
    int32_t const head = windowHead % heads;
    int32_t const window = windowHead / heads;
    int32_t const batch = window / windowsPerImage;
    int32_t const imageWindow = window - batch * windowsPerImage;
    int32_t const windowY = imageWindow / windowsX;
    int32_t const windowX = imageWindow - windowY * windowsX;
    int32_t const tokenBase = static_cast<int32_t>(blockIdx.y) * kTransposeTile;
    int32_t const lane = static_cast<int32_t>(threadIdx.x);
    int32_t const rowLane = static_cast<int32_t>(threadIdx.y);

    for (int32_t row = rowLane; row < kTransposeTile;
         row += kTransposeRowsPerBlock)
    {
        int32_t const token = tokenBase + row;
        __nv_bfloat16 value = __float2bfloat16(0.0F);
        if (token < keyTokens && lane < headDim)
        {
            int32_t const y
                = windowY * queryWindow + token / keyWindow - padding;
            int32_t const x
                = windowX * queryWindow + token % keyWindow - padding;
            if (static_cast<uint32_t>(y) < static_cast<uint32_t>(height)
                && static_cast<uint32_t>(x) < static_cast<uint32_t>(width))
            {
                int32_t const channel = head * headDim + lane;
                value = input[qkvOffset(
                    batch, y, x, 1, channel, height, width, channels)];
            }
        }
        tile[row][lane] = value;
    }
    __syncthreads();

    int32_t const token = tokenBase + lane;
    if (token < keyTokens)
    {
        int64_t const outputBase
            = (static_cast<int64_t>(window) * heads + head)
            * headDim * keyTokens;
        for (int32_t depth = rowLane; depth < headDim;
             depth += kTransposeRowsPerBlock)
        {
            key[outputBase + static_cast<int64_t>(depth) * keyTokens + token]
                = tile[lane][depth];
        }
    }
}

cudaError_t launchPackQKV(void const* input, void* query,
    void* keyTransposed, void* value, int32_t batch, int32_t height,
    int32_t width, int32_t channels, int32_t heads, int32_t queryWindow,
    int32_t keyWindow, cudaStream_t stream)
{
    if (input == nullptr || query == nullptr || keyTransposed == nullptr
        || value == nullptr || batch <= 0 || height <= 0 || width <= 0
        || channels <= 0 || heads <= 0 || channels % heads != 0
        || queryWindow <= 0 || keyWindow < queryWindow
        || (keyWindow - queryWindow) % 2 != 0
        || height % queryWindow != 0 || width % queryWindow != 0)
    {
        return cudaErrorInvalidValue;
    }

    int32_t const windows = batch * (height / queryWindow)
        * (width / queryWindow);
    int32_t const headDim = channels / heads;
    int32_t const queryTokens = queryWindow * queryWindow;
    int32_t const keyTokens = keyWindow * keyWindow;
    int32_t const padding = (keyWindow - queryWindow) / 2;
    int64_t const queryElements = static_cast<int64_t>(windows) * heads
        * queryTokens * headDim;
    int64_t const valueElements = static_cast<int64_t>(windows) * heads
        * keyTokens * headDim;

#if defined(DRFT_ENABLE_OCAB_QKV_PACK_VECTORIZED)
    bool const vectorized = channels % kBf16PerVector == 0
        && headDim % kBf16PerVector == 0;
    if (vectorized)
    {
        int64_t const queryVectors = queryElements / kBf16PerVector;
        int64_t const valueVectors = valueElements / kBf16PerVector;
        int32_t const queryBlocks = static_cast<int32_t>(
            (queryVectors + kLinearThreads - 1) / kLinearThreads);
        int32_t const valueBlocks = static_cast<int32_t>(
            (valueVectors + kLinearThreads - 1) / kLinearThreads);
        packQueryVectorKernel<<<queryBlocks, kLinearThreads, 0, stream>>>(
            static_cast<__nv_bfloat16 const*>(input),
            static_cast<uint4*>(query), height, width, channels, heads,
            queryWindow, queryVectors);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            return error;
        }
        packValueVectorKernel<<<valueBlocks, kLinearThreads, 0, stream>>>(
            static_cast<__nv_bfloat16 const*>(input),
            static_cast<uint4*>(value), height, width, channels, heads,
            queryWindow, keyWindow, padding, valueVectors);
        error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            return error;
        }
    }
    else
#endif
    {
        int32_t const queryBlocks = static_cast<int32_t>(
            (queryElements + kLinearThreads - 1) / kLinearThreads);
        int32_t const valueBlocks = static_cast<int32_t>(
            (valueElements + kLinearThreads - 1) / kLinearThreads);
        packQueryKernel<<<queryBlocks, kLinearThreads, 0, stream>>>(
            static_cast<__nv_bfloat16 const*>(input),
            static_cast<__nv_bfloat16*>(query), height, width, channels,
            heads, queryWindow, queryElements);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            return error;
        }

        packValueKernel<<<valueBlocks, kLinearThreads, 0, stream>>>(
            static_cast<__nv_bfloat16 const*>(input),
            static_cast<__nv_bfloat16*>(value), height, width, channels,
            heads, queryWindow, keyWindow, padding, valueElements);
        error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            return error;
        }
    }

    dim3 const keyGrid(static_cast<uint32_t>(windows * heads),
        static_cast<uint32_t>((keyTokens + kTransposeTile - 1)
            / kTransposeTile));
    dim3 const keyBlock(kTransposeTile, kTransposeRowsPerBlock);
    packKeyTransposedKernel<<<keyGrid, keyBlock, 0, stream>>>(
        static_cast<__nv_bfloat16 const*>(input),
        static_cast<__nv_bfloat16*>(keyTransposed), height, width, channels,
        heads, queryWindow, keyWindow, padding);
    return cudaGetLastError();
}
#endif

bool concreteOrWildcard(int64_t actual, int64_t expected)
{
    return actual == -1 || actual == expected;
}

bool validateDescriptors(nvinfer1::PluginTensorDesc const* inputs,
    int32_t nbInputs, nvinfer1::PluginTensorDesc const* outputs,
    int32_t nbOutputs, int32_t channels, int32_t heads,
    int32_t queryWindow, int32_t keyWindow)
{
    if (inputs == nullptr || outputs == nullptr || nbInputs != 1
        || nbOutputs != 3 || channels <= 0 || heads <= 0
        || channels % heads != 0)
    {
        return false;
    }
    auto const& input = inputs[0].dims;
    auto const& query = outputs[0].dims;
    auto const& key = outputs[1].dims;
    auto const& value = outputs[2].dims;
#if defined(DRFT_ENABLE_OCAB_QKV_PACK_TOKEN_MAJOR)
    if (input.nbDims != 5 || query.nbDims != 4 || key.nbDims != 4
        || value.nbDims != 4)
    {
        return false;
    }
    int32_t const headDim = channels / heads;
    int32_t const queryTokens = queryWindow * queryWindow;
    int32_t const keyTokens = keyWindow * keyWindow;
    int64_t expectedWindows = -1;
    if (input.d[0] > 0 && input.d[1] > 0 && input.d[2] > 0)
    {
        expectedWindows = input.d[0] * (input.d[1] / queryWindow)
            * (input.d[2] / queryWindow);
    }
    return concreteOrWildcard(input.d[3], 3)
        && concreteOrWildcard(input.d[4], channels)
        && concreteOrWildcard(query.d[0], queryTokens)
        && concreteOrWildcard(query.d[1], expectedWindows)
        && concreteOrWildcard(query.d[2], heads)
        && concreteOrWildcard(query.d[3], headDim)
        && concreteOrWildcard(key.d[0], keyTokens)
        && concreteOrWildcard(key.d[1], expectedWindows)
        && concreteOrWildcard(key.d[2], heads)
        && concreteOrWildcard(key.d[3], headDim)
        && concreteOrWildcard(value.d[0], keyTokens)
        && concreteOrWildcard(value.d[1], expectedWindows)
        && concreteOrWildcard(value.d[2], heads)
        && concreteOrWildcard(value.d[3], headDim);
#else
    if (input.nbDims != 5 || query.nbDims != 4 || key.nbDims != 4
        || value.nbDims != 4)
    {
        return false;
    }
    int32_t const headDim = channels / heads;
    int32_t const queryTokens = queryWindow * queryWindow;
    int32_t const keyTokens = keyWindow * keyWindow;
    return concreteOrWildcard(input.d[3], 3)
        && concreteOrWildcard(input.d[4], channels)
        && concreteOrWildcard(query.d[1], heads)
        && concreteOrWildcard(query.d[2], queryTokens)
        && concreteOrWildcard(query.d[3], headDim)
        && concreteOrWildcard(key.d[1], heads)
        && concreteOrWildcard(key.d[2], headDim)
        && concreteOrWildcard(key.d[3], keyTokens)
        && concreteOrWildcard(value.d[1], heads)
        && concreteOrWildcard(value.d[2], keyTokens)
        && concreteOrWildcard(value.d[3], headDim);
#endif
}

class DrftOCABQKVPackPlugin final : public nvinfer1::IPluginV3,
                                    public nvinfer1::IPluginV3OneCore,
                                    public nvinfer1::IPluginV3OneBuild,
                                    public nvinfer1::IPluginV3OneRuntime
{
public:
    DrftOCABQKVPackPlugin(int32_t channels, int32_t heads,
        int32_t queryWindow, int32_t keyWindow)
        : mChannels(channels), mHeads(heads), mQueryWindow(queryWindow),
          mKeyWindow(keyWindow)
    {
        refreshSerializationFields();
    }

    DrftOCABQKVPackPlugin(DrftOCABQKVPackPlugin const& other)
        : mChannels(other.mChannels), mHeads(other.mHeads),
          mQueryWindow(other.mQueryWindow), mKeyWindow(other.mKeyWindow),
          mTactic(other.mTactic)
    {
        refreshSerializationFields();
    }

    nvinfer1::IPluginCapability* getCapabilityInterface(
        nvinfer1::PluginCapabilityType type) noexcept override
    {
        switch (type)
        {
        case nvinfer1::PluginCapabilityType::kCORE:
            return static_cast<nvinfer1::IPluginV3OneCore*>(this);
        case nvinfer1::PluginCapabilityType::kBUILD:
            return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
        case nvinfer1::PluginCapabilityType::kRUNTIME:
            return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
        default:
            return nullptr;
        }
    }

    nvinfer1::IPluginV3* clone() noexcept override
    {
        return new (std::nothrow) DrftOCABQKVPackPlugin(*this);
    }

    char const* getPluginName() const noexcept override { return kPluginName; }
    char const* getPluginVersion() const noexcept override { return kPluginVersion; }
    char const* getPluginNamespace() const noexcept override { return ""; }
    int32_t getNbOutputs() const noexcept override { return 3; }

    int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes,
        int32_t nbOutputs, nvinfer1::DataType const* inputTypes,
        int32_t nbInputs) const noexcept override
    {
        if (outputTypes == nullptr || inputTypes == nullptr || nbOutputs != 3
            || nbInputs != 1)
        {
            return 1;
        }
        outputTypes[0] = inputTypes[0];
        outputTypes[1] = inputTypes[0];
        outputTypes[2] = inputTypes[0];
        return 0;
    }

    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs,
        int32_t nbInputs, nvinfer1::DimsExprs const*, int32_t nbShapeInputs,
        nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
        nvinfer1::IExprBuilder& builder) noexcept override
    {
        if (inputs == nullptr || outputs == nullptr || nbInputs != 1
            || nbShapeInputs != 0 || nbOutputs != 3 || inputs[0].nbDims != 5)
        {
            return 1;
        }
        auto const* queryWindow = builder.constant(mQueryWindow);
        auto const* windowsY = builder.operation(nvinfer1::DimensionOperation::kFLOOR_DIV,
            *inputs[0].d[1], *queryWindow);
        auto const* windowsX = builder.operation(nvinfer1::DimensionOperation::kFLOOR_DIV,
            *inputs[0].d[2], *queryWindow);
        auto const* windowsPerImage = builder.operation(
            nvinfer1::DimensionOperation::kPROD, *windowsY, *windowsX);
        auto const* windows = builder.operation(nvinfer1::DimensionOperation::kPROD,
            *inputs[0].d[0], *windowsPerImage);
        auto const* heads = builder.constant(mHeads);
        auto const* queryTokens
            = builder.constant(mQueryWindow * mQueryWindow);
        auto const* keyTokens = builder.constant(mKeyWindow * mKeyWindow);
        auto const* headDim = builder.constant(mChannels / mHeads);

#if defined(DRFT_ENABLE_OCAB_QKV_PACK_TOKEN_MAJOR)
        for (int32_t output = 0; output < 3; ++output)
        {
            outputs[output].nbDims = 4;
            outputs[output].d[1] = windows;
            outputs[output].d[2] = heads;
            outputs[output].d[3] = headDim;
        }
        outputs[0].d[0] = queryTokens;
        outputs[1].d[0] = keyTokens;
        outputs[2].d[0] = keyTokens;
#else
        for (int32_t output = 0; output < 3; ++output)
        {
            outputs[output].nbDims = 4;
            outputs[output].d[0] = windows;
            outputs[output].d[1] = heads;
        }
        outputs[0].d[2] = queryTokens;
        outputs[0].d[3] = headDim;
        outputs[1].d[2] = headDim;
        outputs[1].d[3] = keyTokens;
        outputs[2].d[2] = keyTokens;
        outputs[2].d[3] = headDim;
#endif
        return 0;
    }

    bool supportsFormatCombination(int32_t position,
        nvinfer1::DynamicPluginTensorDesc const* inOut,
        int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        if (inOut == nullptr || nbInputs != 1 || nbOutputs != 3
            || position < 0 || position >= 4)
        {
            return false;
        }
        // BF16 or FP16 (the strongly typed ONNX's), the same for all four: the
        // kernels move 16-bit values as bits, and their zero padding is +0 in both.
        nvinfer1::DataType const type = inOut[position].desc.type;
        bool const admittedType = position == 0
            ? type == nvinfer1::DataType::kBF16 || type == nvinfer1::DataType::kHALF
            : type == inOut[0].desc.type;
        return inOut[position].desc.format == nvinfer1::TensorFormat::kLINEAR
            && admittedType;
    }

    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* inputs,
        int32_t nbInputs, nvinfer1::DynamicPluginTensorDesc const* outputs,
        int32_t nbOutputs) noexcept override
    {
        if (inputs == nullptr || outputs == nullptr)
        {
            return 1;
        }
        std::array<nvinfer1::PluginTensorDesc, 1> inputDescriptions{};
        std::array<nvinfer1::PluginTensorDesc, 3> outputDescriptions{};
        if (nbInputs == 1)
        {
            inputDescriptions[0] = inputs[0].desc;
        }
        for (int32_t index = 0; index < nbOutputs && index < 3; ++index)
        {
            outputDescriptions[index] = outputs[index].desc;
        }
        return validateDescriptors(inputDescriptions.data(), nbInputs,
                   outputDescriptions.data(), nbOutputs, mChannels, mHeads,
                   mQueryWindow, mKeyWindow)
            ? 0
            : 1;
    }

    size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
        nvinfer1::DynamicPluginTensorDesc const*, int32_t) const noexcept override
    {
        return 0;
    }

    int32_t getNbTactics() noexcept override { return 1; }
    int32_t getValidTactics(int32_t* tactics, int32_t nbTactics) noexcept override
    {
        if (tactics == nullptr || nbTactics != 1)
        {
            return 1;
        }
        tactics[0] = kTactic;
        return 0;
    }
    char const* getTimingCacheID() noexcept override
    {
#if defined(DRFT_ENABLE_OCAB_QKV_PACK_TOKEN_MAJOR)
        return "bf16_direct_ocab_qkv_pack_token_major_vector16_v3";
#elif defined(DRFT_ENABLE_OCAB_QKV_PACK_VECTORIZED)
        return "bf16_direct_ocab_qkv_pack_vector16_v2";
#else
        return "bf16_direct_ocab_qkv_pack_v1";
#endif
    }
    int32_t getFormatCombinationLimit() noexcept override { return 1; }
    char const* getMetadataString() noexcept override
    {
#if defined(DRFT_ENABLE_OCAB_QKV_PACK_TOKEN_MAJOR)
        return "{\"algorithm\":\"direct_window_pack_token_major_vector16\",\"arithmetic\":\"none\"}";
#elif defined(DRFT_ENABLE_OCAB_QKV_PACK_VECTORIZED)
        return "{\"algorithm\":\"direct_window_pack_vector16_qv\",\"arithmetic\":\"none\"}";
#else
        return "{\"algorithm\":\"direct_window_pack\",\"arithmetic\":\"none\"}";
#endif
    }
    int32_t setTactic(int32_t tactic) noexcept override
    {
        if (tactic != kTactic)
        {
            return 1;
        }
        mTactic = tactic;
        return 0;
    }

    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* inputs,
        int32_t nbInputs, nvinfer1::PluginTensorDesc const* outputs,
        int32_t nbOutputs) noexcept override
    {
        return validateDescriptors(inputs, nbInputs, outputs, nbOutputs,
                   mChannels, mHeads, mQueryWindow, mKeyWindow)
            ? 0
            : 1;
    }

    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDescriptions,
        nvinfer1::PluginTensorDesc const*, void const* const* inputs,
        void* const* outputs, void*, cudaStream_t stream) noexcept override
    {
        if (mTactic != kTactic || inputDescriptions == nullptr
            || inputs == nullptr || outputs == nullptr)
        {
            return 1;
        }
        auto const& dims = inputDescriptions[0].dims;
        if (dims.nbDims != 5)
        {
            return 1;
        }
#if defined(DRFT_ENABLE_OCAB_QKV_PACK_TOKEN_MAJOR)
        cudaError_t const error = launchPackQKVTokenMajor(inputs[0], outputs[0],
            outputs[1], outputs[2], static_cast<int32_t>(dims.d[0]),
            static_cast<int32_t>(dims.d[1]), static_cast<int32_t>(dims.d[2]),
            mChannels, mHeads, mQueryWindow, mKeyWindow, stream);
#else
        cudaError_t const error = launchPackQKV(inputs[0], outputs[0], outputs[1],
            outputs[2], dims.d[0], dims.d[1], dims.d[2], mChannels, mHeads,
            mQueryWindow, mKeyWindow, stream);
#endif
        return error == cudaSuccess ? 0 : 1;
    }

    nvinfer1::IPluginV3* attachToContext(
        nvinfer1::IPluginResourceContext*) noexcept override
    {
        return clone();
    }

    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override
    {
        refreshSerializationFields();
        return &mSerializationCollection;
    }

private:
    void refreshSerializationFields() noexcept
    {
        mSerializationFields = {
            nvinfer1::PluginField{"channels", &mChannels,
                nvinfer1::PluginFieldType::kINT32, 1},
            nvinfer1::PluginField{"heads", &mHeads,
                nvinfer1::PluginFieldType::kINT32, 1},
            nvinfer1::PluginField{"query_window_size", &mQueryWindow,
                nvinfer1::PluginFieldType::kINT32, 1},
            nvinfer1::PluginField{"key_window_size", &mKeyWindow,
                nvinfer1::PluginFieldType::kINT32, 1},
        };
        mSerializationCollection.nbFields
            = static_cast<int32_t>(mSerializationFields.size());
        mSerializationCollection.fields = mSerializationFields.data();
    }

    int32_t mChannels{0};
    int32_t mHeads{0};
    int32_t mQueryWindow{0};
    int32_t mKeyWindow{0};
    int32_t mTactic{kTactic};
    std::array<nvinfer1::PluginField, 4> mSerializationFields{};
    nvinfer1::PluginFieldCollection mSerializationCollection{};
};

class DrftOCABQKVPackPluginCreator final
    : public nvinfer1::IPluginCreatorV3One
{
public:
    DrftOCABQKVPackPluginCreator()
    {
        mFields = {
            nvinfer1::PluginField{"channels", nullptr,
                nvinfer1::PluginFieldType::kINT32, 1},
            nvinfer1::PluginField{"heads", nullptr,
                nvinfer1::PluginFieldType::kINT32, 1},
            nvinfer1::PluginField{"query_window_size", nullptr,
                nvinfer1::PluginFieldType::kINT32, 1},
            nvinfer1::PluginField{"key_window_size", nullptr,
                nvinfer1::PluginFieldType::kINT32, 1},
        };
        mCollection.nbFields = static_cast<int32_t>(mFields.size());
        mCollection.fields = mFields.data();
    }

    nvinfer1::IPluginV3* createPlugin(char const*,
        nvinfer1::PluginFieldCollection const* fields,
        nvinfer1::TensorRTPhase) noexcept override
    {
        int32_t channels = 0;
        int32_t heads = 0;
        int32_t queryWindow = 0;
        int32_t keyWindow = 0;
        if (fields == nullptr)
        {
            return nullptr;
        }
        for (int32_t index = 0; index < fields->nbFields; ++index)
        {
            auto const& field = fields->fields[index];
            if (field.name == nullptr || field.data == nullptr
                || field.length != 1
                || field.type != nvinfer1::PluginFieldType::kINT32)
            {
                continue;
            }
            int32_t const value = *static_cast<int32_t const*>(field.data);
            if (std::strcmp(field.name, "channels") == 0)
            {
                channels = value;
            }
            else if (std::strcmp(field.name, "heads") == 0)
            {
                heads = value;
            }
            else if (std::strcmp(field.name, "query_window_size") == 0)
            {
                queryWindow = value;
            }
            else if (std::strcmp(field.name, "key_window_size") == 0)
            {
                keyWindow = value;
            }
        }
        if (channels <= 0 || heads <= 0 || channels % heads != 0
            || queryWindow <= 0 || keyWindow < queryWindow
            || (keyWindow - queryWindow) % 2 != 0)
        {
            return nullptr;
        }
        return new (std::nothrow) DrftOCABQKVPackPlugin(
            channels, heads, queryWindow, keyWindow);
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override
    {
        return &mCollection;
    }
    char const* getPluginName() const noexcept override { return kPluginName; }
    char const* getPluginVersion() const noexcept override { return kPluginVersion; }
    char const* getPluginNamespace() const noexcept override { return ""; }

private:
    std::array<nvinfer1::PluginField, 4> mFields{};
    nvinfer1::PluginFieldCollection mCollection{};
};
} // namespace

DRFT_PLUGIN_EXPORT nvinfer1::IPluginCreatorInterface*
drftGetOCABQKVPackPluginCreator()
{
    static DrftOCABQKVPackPluginCreator creator;
    return &creator;
}
