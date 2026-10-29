#include "drft_hip_kernels.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#if defined(_WIN32) && defined(__HIP_DEVICE_COMPILE__)
// The device pass of a Windows build sees the host's declarations; a GPU target has no DLL exports.
#define DRFT_HIP_EXPORT extern "C"
#elif defined(_WIN32)
#define DRFT_HIP_EXPORT extern "C" __declspec(dllexport)
#else
#define DRFT_HIP_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace drft_hip
{
namespace
{
constexpr int32_t kTile = 16;
constexpr int32_t kLinearThreads = 256;
// Grid-stride kernels launch at most this many blocks (HIP caps a dimension's threads at 2^32).
constexpr int64_t kMaxLinearBlocks = 65535;
constexpr int32_t kElementsPerVector = 8;
constexpr int32_t kHeadDim = 32;
constexpr int32_t kAttentionThreads = kTile * kTile;

// BF16 as its bits: widening is exact; narrowing rounds to nearest even and makes a NaN the
// canonical quiet NaN, as CUDA's __float2bfloat16_rn.
struct Bf16
{
    uint16_t bits;
};

__device__ __forceinline__ float toFloat(Bf16 value)
{
    return __uint_as_float(static_cast<uint32_t>(value.bits) << 16);
}

__device__ __forceinline__ float toFloat(__half value) { return __half2float(value); }

template <typename Scalar>
__device__ __forceinline__ Scalar fromFloat(float value);

template <>
__device__ __forceinline__ Bf16 fromFloat(float value)
{
    uint32_t const bits = __float_as_uint(value);
    if ((bits & 0x7fffffffU) > 0x7f800000U)
    {
        return Bf16{0x7fffU};
    }
    return Bf16{static_cast<uint16_t>((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16)};
}

template <>
__device__ __forceinline__ __half fromFloat(float value) { return __float2half_rn(value); }

int32_t linearBlocks(int64_t work) noexcept
{
    return static_cast<int32_t>(
        std::min<int64_t>((work + kLinearThreads - 1) / kLinearThreads, kMaxLinearBlocks));
}

// The tiles hold the factors widened to FP32 (exact), summed in the CUDA kernel's order.
template <typename Scalar>
__global__ void windowNeuralBiasKernel(Scalar const* queryFactor, Scalar const* keyFactor,
    Scalar* output, int32_t queryTokens, int32_t keyTokens, int32_t rank)
{
    __shared__ float queryTile[kTile][kTile];
    __shared__ float keyTile[kTile][kTile];

    int32_t const column = static_cast<int32_t>(blockIdx.x) * kTile
        + static_cast<int32_t>(threadIdx.x);
    int32_t const row = static_cast<int32_t>(blockIdx.y) * kTile
        + static_cast<int32_t>(threadIdx.y);
    int32_t const head = static_cast<int32_t>(blockIdx.z);
    float accumulator = 0.0F;

    for (int32_t base = 0; base < rank; base += kTile)
    {
        int32_t const queryRank = base + static_cast<int32_t>(threadIdx.x);
        int32_t const keyRank = base + static_cast<int32_t>(threadIdx.y);
        queryTile[threadIdx.y][threadIdx.x] = row < queryTokens && queryRank < rank
            ? toFloat(queryFactor[(static_cast<int64_t>(head) * queryTokens + row) * rank
                + queryRank])
            : 0.0F;
        keyTile[threadIdx.y][threadIdx.x] = column < keyTokens && keyRank < rank
            ? toFloat(keyFactor[(static_cast<int64_t>(head) * keyTokens + column) * rank
                + keyRank])
            : 0.0F;
        __syncthreads();

#pragma unroll
        for (int32_t index = 0; index < kTile; ++index)
        {
            accumulator += queryTile[threadIdx.y][index] * keyTile[index][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < queryTokens && column < keyTokens)
    {
        output[(static_cast<int64_t>(head) * queryTokens + row) * keyTokens + column]
            = fromFloat<Scalar>(accumulator);
    }
}

// One block per (query token, head) plane; eight groups of 32 threads walk the key rows (groups,
// not wavefronts: no cross-lane operation). The compact table keeps the ONNX's [relative
// position, head] layout; values are copied as bits.
template <typename Bits, int32_t keyWindow>
__global__ void expandOCABRelativeBiasQ32(Bits const* table, Bits* output, int32_t heads)
{
    int32_t constexpr queryWindow = 32;
    int32_t constexpr queryTokens = queryWindow * queryWindow;
    int32_t constexpr keyTokens = keyWindow * keyWindow;
    int32_t constexpr biasSpan = queryWindow + keyWindow - 1;

    int32_t const queryToken = static_cast<int32_t>(blockIdx.x);
    int32_t const head = static_cast<int32_t>(blockIdx.y);
    int32_t const lane = static_cast<int32_t>(threadIdx.x) & 31;
    int32_t const group = static_cast<int32_t>(threadIdx.x) >> 5;
    int32_t const queryY = queryToken >> 5;
    int32_t const queryX = queryToken & 31;
    int64_t const outputBase
        = (static_cast<int64_t>(head) * queryTokens + queryToken) * keyTokens;

    for (int32_t keyY = group; keyY < keyWindow; keyY += 8)
    {
        int32_t const biasRow
            = (keyY - queryY + queryWindow - 1) * biasSpan + (queryWindow - 1 - queryX);
        int32_t const keyRow = keyY * keyWindow;
        output[outputBase + keyRow + lane] = table[(biasRow + lane) * heads + head];
        if (lane < keyWindow - 32)
        {
            output[outputBase + keyRow + 32 + lane]
                = table[(biasRow + 32 + lane) * heads + head];
        }
    }
}

template <typename Bits>
hipError_t launchRelativeBiasBits(void const* table, void* output, int32_t heads,
    int32_t keyWindow, hipStream_t stream) noexcept
{
    dim3 const blocks(32U * 32U, static_cast<uint32_t>(heads));
    if (keyWindow == 40)
    {
        expandOCABRelativeBiasQ32<Bits, 40><<<blocks, kLinearThreads, 0, stream>>>(
            static_cast<Bits const*>(table), static_cast<Bits*>(output), heads);
    }
    else
    {
        expandOCABRelativeBiasQ32<Bits, 48><<<blocks, kLinearThreads, 0, stream>>>(
            static_cast<Bits const*>(table), static_cast<Bits*>(output), heads);
    }
    return hipGetLastError();
}

__device__ __forceinline__ int64_t qkvOffset(int32_t batch, int32_t y, int32_t x,
    int32_t component, int32_t channel, int32_t height, int32_t width, int32_t channels)
{
    return (((static_cast<int64_t>(batch) * height + y) * width + x) * 3 + component) * channels
        + channel;
}

// Eight 16-bit values (one uint4) per thread and step: the CUDA token-major kernels' indexing, with
// the q pack as the window pack at the query window (no padding).
__global__ void ocabWindowPackKernel(uint16_t const* qkv, uint4* output, int32_t height,
    int32_t width, int32_t channels, int32_t heads, int32_t queryWindow, int32_t window,
    int32_t padding, int32_t component, int32_t batchHeads, int64_t vectors)
{
    int32_t const headDim = channels / heads;
    int32_t const vectorsPerHead = headDim / kElementsPerVector;
    int32_t const windowsX = width / queryWindow;
    int32_t const windowsPerImage = (height / queryWindow) * windowsX;
    int64_t const stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < vectors; linear += stride)
    {
        int64_t index = linear;
        int32_t const depthVector = static_cast<int32_t>(index % vectorsPerHead);
        index /= vectorsPerHead;
        int32_t const windowHead = static_cast<int32_t>(index % batchHeads);
        int32_t const token = static_cast<int32_t>(index / batchHeads);
        int32_t const head = windowHead % heads;
        int32_t const windowIndex = windowHead / heads;
        int32_t const batch = windowIndex / windowsPerImage;
        int32_t const imageWindow = windowIndex - batch * windowsPerImage;
        int32_t const windowY = imageWindow / windowsX;
        int32_t const windowX = imageWindow - windowY * windowsX;
        int32_t const y = windowY * queryWindow + token / window - padding;
        int32_t const x = windowX * queryWindow + token % window - padding;

        uint4 packed = make_uint4(0U, 0U, 0U, 0U);
        if (static_cast<uint32_t>(y) < static_cast<uint32_t>(height)
            && static_cast<uint32_t>(x) < static_cast<uint32_t>(width))
        {
            int32_t const channel = head * headDim + depthVector * kElementsPerVector;
            packed = *reinterpret_cast<uint4 const*>(
                qkv + qkvOffset(batch, y, x, component, channel, height, width, channels));
        }
        output[linear] = packed;
    }
}

template <typename Scalar>
__global__ void finalAffineCropKernel(Scalar const* input, float* output, int64_t totalElements,
    int32_t channels, int32_t sourceHeight, int32_t sourceWidth, int32_t outputHeight,
    int32_t outputWidth, float mean0, float mean1, float mean2)
{
    int64_t const stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < totalElements; index += stride)
    {
        int64_t value = index;
        int32_t const x = static_cast<int32_t>(value % outputWidth);
        value /= outputWidth;
        int32_t const y = static_cast<int32_t>(value % outputHeight);
        value /= outputHeight;
        int32_t const channel = static_cast<int32_t>(value % channels);
        int32_t const batch = static_cast<int32_t>(value / channels);
        int64_t const sourceIndex
            = ((static_cast<int64_t>(batch) * channels + channel) * sourceHeight + y) * sourceWidth
            + x;
        float const mean = channel == 0 ? mean0 : (channel == 1 ? mean1 : mean2);
        output[index] = toFloat(input[sourceIndex]) + mean;
    }
}

// One block of 256 threads per (batch, head, 16 query tokens): each thread scores one (query, key)
// pair of a 16x16 tile, the tile's 16 rows update their running maximum and sum, and every thread
// accumulates two of the tile's 16x32 outputs. Operands are widened to FP32 in LDS (exact).
__global__ __launch_bounds__(kAttentionThreads) void ocabAttentionOnline(Bf16 const* query,
    Bf16 const* keyTransposed, Bf16 const* value, Bf16 const* relativeBiasTable, Bf16* output,
    int32_t heads, int32_t queryWindow, int32_t keyWindow)
{
    __shared__ float queryTile[kTile][kHeadDim];
    __shared__ float keyTile[kHeadDim][kTile];
    __shared__ float valueTile[kTile][kHeadDim];
    __shared__ float scores[kTile][kTile];
    __shared__ float numerator[kTile][kHeadDim];
    __shared__ float rowMax[kTile];
    __shared__ float rowSum[kTile];
    __shared__ float rowAlpha[kTile];

    int32_t const queryTokens = queryWindow * queryWindow;
    int32_t const keyTokens = keyWindow * keyWindow;
    int32_t const queryTiles = queryTokens / kTile;
    int32_t const biasSpan = queryWindow + keyWindow - 1;
    int32_t const block = static_cast<int32_t>(blockIdx.x);
    int32_t const queryStart = (block % queryTiles) * kTile;
    int64_t const batchHead = block / queryTiles;
    int32_t const head = static_cast<int32_t>(batchHead % heads);
    int32_t const thread = static_cast<int32_t>(threadIdx.x);
    int32_t const row = thread / kTile;
    int32_t const column = thread % kTile;

    Bf16 const* queryBase = query + batchHead * queryTokens * kHeadDim;
    Bf16 const* keyBase = keyTransposed + batchHead * kHeadDim * keyTokens;
    Bf16 const* valueBase = value + batchHead * keyTokens * kHeadDim;
    Bf16* outputBase = output + batchHead * queryTokens * kHeadDim;

    for (int32_t index = thread; index < kTile * kHeadDim; index += kAttentionThreads)
    {
        queryTile[index / kHeadDim][index % kHeadDim]
            = toFloat(queryBase[queryStart * kHeadDim + index]);
        numerator[index / kHeadDim][index % kHeadDim] = 0.0F;
    }
    if (thread < kTile)
    {
        rowMax[thread] = -INFINITY;
        rowSum[thread] = 0.0F;
        rowAlpha[thread] = 0.0F;
    }
    int32_t const queryToken = queryStart + row;
    int32_t const queryY = queryToken / queryWindow;
    int32_t const queryX = queryToken % queryWindow;
    __syncthreads();

    for (int32_t keyStart = 0; keyStart < keyTokens; keyStart += kTile)
    {
        for (int32_t index = thread; index < kHeadDim * kTile; index += kAttentionThreads)
        {
            keyTile[index / kTile][index % kTile]
                = toFloat(keyBase[(index / kTile) * keyTokens + keyStart + index % kTile]);
            valueTile[index / kHeadDim][index % kHeadDim]
                = toFloat(valueBase[keyStart * kHeadDim + index]);
        }
        __syncthreads();

        float score = 0.0F;
#pragma unroll
        for (int32_t depth = 0; depth < kHeadDim; ++depth)
        {
            score += queryTile[row][depth] * keyTile[depth][column];
        }
        int32_t const keyToken = keyStart + column;
        int32_t const biasIndex = (keyToken / keyWindow - queryY + queryWindow - 1) * biasSpan
            + (keyToken % keyWindow - queryX + queryWindow - 1);
        scores[row][column] = score + toFloat(relativeBiasTable[biasIndex * heads + head]);
        __syncthreads();

        if (thread < kTile)
        {
            float tileMax = -INFINITY;
#pragma unroll
            for (int32_t key = 0; key < kTile; ++key)
            {
                tileMax = fmaxf(tileMax, scores[thread][key]);
            }
            float const oldMax = rowMax[thread];
            float const newMax = fmaxf(oldMax, tileMax);
            float const alpha = isinf(oldMax) ? 0.0F : expf(oldMax - newMax);
            float tileSum = 0.0F;
#pragma unroll
            for (int32_t key = 0; key < kTile; ++key)
            {
                float const probability = expf(scores[thread][key] - newMax);
                tileSum += probability;
                scores[thread][key] = toFloat(fromFloat<Bf16>(probability));
            }
            rowAlpha[thread] = alpha;
            rowMax[thread] = newMax;
            rowSum[thread] = rowSum[thread] * alpha + tileSum;
        }
        __syncthreads();

        for (int32_t index = thread; index < kTile * kHeadDim; index += kAttentionThreads)
        {
            int32_t const outputRow = index / kHeadDim;
            int32_t const outputColumn = index % kHeadDim;
            float product = 0.0F;
#pragma unroll
            for (int32_t key = 0; key < kTile; ++key)
            {
                product += scores[outputRow][key] * valueTile[key][outputColumn];
            }
            float const scaled = numerator[outputRow][outputColumn] * rowAlpha[outputRow];
            numerator[outputRow][outputColumn] = scaled + product;
        }
        __syncthreads();
    }

    for (int32_t index = thread; index < kTile * kHeadDim; index += kAttentionThreads)
    {
        int32_t const outputRow = index / kHeadDim;
        outputBase[queryStart * kHeadDim + index] = fromFloat<Bf16>(
            numerator[outputRow][index % kHeadDim] / rowSum[outputRow]);
    }
}

bool isOCABWindow(int32_t queryWindow, int32_t keyWindow) noexcept
{
    return queryWindow == 32 && (keyWindow == 40 || keyWindow == 48);
}
} // namespace

hipError_t launchWindowNeuralBias(void const* queryFactor, void const* keyFactor, void* output,
    ElementType type, int32_t heads, int32_t queryTokens, int32_t keyTokens, int32_t rank,
    hipStream_t stream) noexcept
{
    if (queryFactor == nullptr || keyFactor == nullptr || output == nullptr || heads <= 0
        || queryTokens <= 0 || keyTokens <= 0 || rank <= 0
        || (type != ElementType::kFloat16 && type != ElementType::kBFloat16))
    {
        return hipErrorInvalidValue;
    }
    dim3 const block(kTile, kTile);
    dim3 const grid(static_cast<uint32_t>((keyTokens + kTile - 1) / kTile),
        static_cast<uint32_t>((queryTokens + kTile - 1) / kTile), static_cast<uint32_t>(heads));
    if (type == ElementType::kFloat16)
    {
        windowNeuralBiasKernel<__half><<<grid, block, 0, stream>>>(
            static_cast<__half const*>(queryFactor), static_cast<__half const*>(keyFactor),
            static_cast<__half*>(output), queryTokens, keyTokens, rank);
    }
    else
    {
        windowNeuralBiasKernel<Bf16><<<grid, block, 0, stream>>>(
            static_cast<Bf16 const*>(queryFactor), static_cast<Bf16 const*>(keyFactor),
            static_cast<Bf16*>(output), queryTokens, keyTokens, rank);
    }
    return hipGetLastError();
}

hipError_t launchOCABRelativeBias(void const* table, void* output, ElementType type,
    int32_t heads, int32_t queryWindow, int32_t keyWindow, hipStream_t stream) noexcept
{
    if (table == nullptr || output == nullptr || heads <= 0
        || !isOCABWindow(queryWindow, keyWindow))
    {
        return hipErrorInvalidValue;
    }
    switch (type)
    {
    case ElementType::kFloat32:
        return launchRelativeBiasBits<uint32_t>(table, output, heads, keyWindow, stream);
    case ElementType::kFloat16:
    case ElementType::kBFloat16:
        return launchRelativeBiasBits<uint16_t>(table, output, heads, keyWindow, stream);
    default:
        return hipErrorInvalidValue;
    }
}

hipError_t launchOCABWindowPack(void const* qkv, void* output, int32_t batch, int32_t height,
    int32_t width, int32_t channels, int32_t heads, int32_t queryWindow, int32_t window,
    int32_t component, hipStream_t stream) noexcept
{
    if (qkv == nullptr || output == nullptr || batch <= 0 || height <= 0 || width <= 0
        || channels <= 0 || heads <= 0 || channels % heads != 0
        || (channels / heads) % kElementsPerVector != 0 || queryWindow <= 0
        || window < queryWindow || (window - queryWindow) % 2 != 0
        || height % queryWindow != 0 || width % queryWindow != 0 || component < 0
        || component > 2)
    {
        return hipErrorInvalidValue;
    }
    int64_t const windows
        = static_cast<int64_t>(batch) * (height / queryWindow) * (width / queryWindow);
    if (windows * heads > std::numeric_limits<int32_t>::max())
    {
        return hipErrorInvalidValue;
    }
    int32_t const batchHeads = static_cast<int32_t>(windows * heads);
    int64_t const vectors = static_cast<int64_t>(window) * window * batchHeads
        * ((channels / heads) / kElementsPerVector);
    ocabWindowPackKernel<<<linearBlocks(vectors), kLinearThreads, 0, stream>>>(
        static_cast<uint16_t const*>(qkv), static_cast<uint4*>(output), height, width, channels,
        heads, queryWindow, window, (window - queryWindow) / 2, component, batchHeads, vectors);
    return hipGetLastError();
}

hipError_t launchFinalAffineCrop(void const* input, float* output, ElementType type,
    int32_t batch, int32_t channels, int32_t sourceHeight, int32_t sourceWidth,
    int32_t outputHeight, int32_t outputWidth, float const (&mean)[3],
    hipStream_t stream) noexcept
{
    if (input == nullptr || output == nullptr || batch < 0 || channels != 3
        || outputHeight < 0 || outputWidth < 0 || sourceHeight < outputHeight
        || sourceWidth < outputWidth
        || (type != ElementType::kFloat16 && type != ElementType::kBFloat16))
    {
        return hipErrorInvalidValue;
    }
    if (batch == 0 || outputHeight == 0 || outputWidth == 0)
    {
        return hipSuccess;
    }
    int64_t const totalElements
        = static_cast<int64_t>(batch) * channels * outputHeight * outputWidth;
    int32_t const blocks = linearBlocks(totalElements);
    if (type == ElementType::kFloat16)
    {
        finalAffineCropKernel<__half><<<blocks, kLinearThreads, 0, stream>>>(
            static_cast<__half const*>(input), output, totalElements, channels, sourceHeight,
            sourceWidth, outputHeight, outputWidth, mean[0], mean[1], mean[2]);
    }
    else
    {
        finalAffineCropKernel<Bf16><<<blocks, kLinearThreads, 0, stream>>>(
            static_cast<Bf16 const*>(input), output, totalElements, channels, sourceHeight,
            sourceWidth, outputHeight, outputWidth, mean[0], mean[1], mean[2]);
    }
    return hipGetLastError();
}

hipError_t launchOCABAttention(void const* query, void const* keyTransposed, void const* value,
    void const* relativeBiasTable, void* output, int32_t batch, int32_t heads,
    int32_t queryWindow, int32_t keyWindow, hipStream_t stream) noexcept
{
    if (query == nullptr || keyTransposed == nullptr || value == nullptr
        || relativeBiasTable == nullptr || output == nullptr || batch <= 0 || heads <= 0
        || !isOCABWindow(queryWindow, keyWindow))
    {
        return hipErrorInvalidValue;
    }
    int64_t const blocks
        = static_cast<int64_t>(batch) * heads * (queryWindow * queryWindow / kTile);
    if (blocks > std::numeric_limits<int32_t>::max())
    {
        return hipErrorInvalidValue;
    }
    ocabAttentionOnline<<<static_cast<uint32_t>(blocks), kAttentionThreads, 0, stream>>>(
        static_cast<Bf16 const*>(query), static_cast<Bf16 const*>(keyTransposed),
        static_cast<Bf16 const*>(value), static_cast<Bf16 const*>(relativeBiasTable),
        static_cast<Bf16*>(output), heads, queryWindow, keyWindow);
    return hipGetLastError();
}
} // namespace drft_hip

// The launchers for tests (ctypes, device pointers): element types as drft_hip::ElementType's
// values, results as hipError_t's.
DRFT_HIP_EXPORT int32_t drftHipLaunchWindowNeuralBias(void const* queryFactor,
    void const* keyFactor, void* output, int32_t type, int32_t heads, int32_t queryTokens,
    int32_t keyTokens, int32_t rank, hipStream_t stream)
{
    return static_cast<int32_t>(drft_hip::launchWindowNeuralBias(queryFactor, keyFactor, output,
        static_cast<drft_hip::ElementType>(type), heads, queryTokens, keyTokens, rank, stream));
}

DRFT_HIP_EXPORT int32_t drftHipLaunchOCABRelativeBias(void const* table, void* output,
    int32_t type, int32_t heads, int32_t queryWindow, int32_t keyWindow, hipStream_t stream)
{
    return static_cast<int32_t>(drft_hip::launchOCABRelativeBias(table, output,
        static_cast<drft_hip::ElementType>(type), heads, queryWindow, keyWindow, stream));
}

DRFT_HIP_EXPORT int32_t drftHipLaunchOCABWindowPack(void const* qkv, void* output,
    int32_t batch, int32_t height, int32_t width, int32_t channels, int32_t heads,
    int32_t queryWindow, int32_t window, int32_t component, hipStream_t stream)
{
    return static_cast<int32_t>(drft_hip::launchOCABWindowPack(qkv, output, batch, height, width,
        channels, heads, queryWindow, window, component, stream));
}

DRFT_HIP_EXPORT int32_t drftHipLaunchFinalAffineCrop(void const* input, float* output,
    int32_t type, int32_t batch, int32_t channels, int32_t sourceHeight, int32_t sourceWidth,
    int32_t outputHeight, int32_t outputWidth, float meanR, float meanG, float meanB,
    hipStream_t stream)
{
    float const mean[3] = {meanR, meanG, meanB};
    return static_cast<int32_t>(drft_hip::launchFinalAffineCrop(input, output,
        static_cast<drft_hip::ElementType>(type), batch, channels, sourceHeight, sourceWidth,
        outputHeight, outputWidth, mean, stream));
}

DRFT_HIP_EXPORT int32_t drftHipLaunchOCABAttention(void const* query, void const* keyTransposed,
    void const* value, void const* relativeBiasTable, void* output, int32_t batch, int32_t heads,
    int32_t queryWindow, int32_t keyWindow, hipStream_t stream)
{
    return static_cast<int32_t>(drft_hip::launchOCABAttention(query, keyTransposed, value,
        relativeBiasTable, output, batch, heads, queryWindow, keyWindow, stream));
}
