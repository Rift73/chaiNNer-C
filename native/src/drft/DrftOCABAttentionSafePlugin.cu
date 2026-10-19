#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <cuda_runtime_api.h>
#include <mma.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <vector>

#if defined(_WIN32)
#define DRFT_PLUGIN_EXPORT extern "C" __declspec(dllexport)
#else
#define DRFT_PLUGIN_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace
{
constexpr char kPluginName[] = "DrftOCABAttentionSafe_TRT";
constexpr char kPluginVersion[] = "1";
constexpr char kBiasPluginName[] = "DrftOCABRelativeBiasSafe_TRT";
constexpr char kBiasPluginVersion[] = "1";
constexpr int32_t kHeadDim = 32;
constexpr int32_t kQueryTile = 16;
constexpr int32_t kKeyTile = 16;
constexpr int32_t kWmmaOnlineTactic = 1;
constexpr int32_t kRelativeBiasTactic = 2;

struct alignas(16) AttentionSharedStorage
{
    float scores[kQueryTile * kKeyTile];
    __nv_bfloat16 probabilities[kQueryTile * kKeyTile];
    float numerator[kQueryTile * kHeadDim];
    float product[kQueryTile * kKeyTile];
    float rowMax[kQueryTile];
    float rowSum[kQueryTile];
    float rowAlpha[kQueryTile];
};

__global__ void ocabAttentionWmmaOnline(__nv_bfloat16 const* query,
    __nv_bfloat16 const* keyTransposed, __nv_bfloat16 const* value,
    __nv_bfloat16 const* relativeBiasTable, __nv_bfloat16* output,
    int32_t batch, int32_t heads, int32_t queryWindow, int32_t keyWindow)
{
#if __CUDA_ARCH__ >= 800
    using namespace nvcuda;
    __shared__ AttentionSharedStorage storage;

    int32_t const queryTokens = queryWindow * queryWindow;
    int32_t const keyTokens = keyWindow * keyWindow;
    int32_t const queryTiles = queryTokens / kQueryTile;
    int32_t const keyTiles = keyTokens / kKeyTile;
    int32_t const biasSpan = queryWindow + keyWindow - 1;
    int32_t const block = static_cast<int32_t>(blockIdx.x);
    int32_t const queryTileIndex = block % queryTiles;
    int32_t const batchHead = block / queryTiles;
    int32_t const head = batchHead % heads;
    int32_t const batchIndex = batchHead / heads;
    int32_t const lane = static_cast<int32_t>(threadIdx.x);

    auto const* queryBase = query
        + (static_cast<int64_t>(batchIndex) * heads + head) * queryTokens * kHeadDim;
    auto const* keyBase = keyTransposed
        + (static_cast<int64_t>(batchIndex) * heads + head) * kHeadDim * keyTokens;
    auto const* valueBase = value
        + (static_cast<int64_t>(batchIndex) * heads + head) * keyTokens * kHeadDim;
    auto* outputBase = output
        + (static_cast<int64_t>(batchIndex) * heads + head) * queryTokens * kHeadDim;
    int32_t const queryStart = queryTileIndex * kQueryTile;

    for (int32_t index = lane; index < kQueryTile * kHeadDim; index += 32)
    {
        storage.numerator[index] = 0.0F;
    }
    if (lane < kQueryTile)
    {
        storage.rowMax[lane] = -CUDART_INF_F;
        storage.rowSum[lane] = 0.0F;
        storage.rowAlpha[lane] = 0.0F;
    }
    __syncwarp();

    for (int32_t keyTileIndex = 0; keyTileIndex < keyTiles; ++keyTileIndex)
    {
        int32_t const keyStart = keyTileIndex * kKeyTile;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> scoreAccumulator;
        wmma::fill_fragment(scoreAccumulator, 0.0F);

        #pragma unroll
        for (int32_t depth = 0; depth < kHeadDim; depth += 16)
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> queryFragment;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> keyFragment;
            wmma::load_matrix_sync(
                queryFragment, queryBase + queryStart * kHeadDim + depth, kHeadDim);
            wmma::load_matrix_sync(
                keyFragment, keyBase + depth * keyTokens + keyStart, keyTokens);
            wmma::mma_sync(scoreAccumulator, queryFragment, keyFragment, scoreAccumulator);
        }
        wmma::store_matrix_sync(
            storage.scores, scoreAccumulator, kKeyTile, wmma::mem_row_major);
        __syncwarp();

        if (lane < kQueryTile)
        {
            int32_t const queryToken = queryStart + lane;
            int32_t const queryY = queryToken / queryWindow;
            int32_t const queryX = queryToken % queryWindow;
            float tileMax = -CUDART_INF_F;

            #pragma unroll
            for (int32_t column = 0; column < kKeyTile; ++column)
            {
                int32_t const keyToken = keyStart + column;
                int32_t const keyY = keyToken / keyWindow;
                int32_t const keyX = keyToken % keyWindow;
                int32_t const biasIndex
                    = (keyY - queryY + queryWindow - 1) * biasSpan
                    + (keyX - queryX + queryWindow - 1);
                float const bias = __bfloat162float(relativeBiasTable[biasIndex * heads + head]);
                float const score = storage.scores[lane * kKeyTile + column] + bias;
                storage.scores[lane * kKeyTile + column] = score;
                tileMax = fmaxf(tileMax, score);
            }

            float const oldMax = storage.rowMax[lane];
            float const newMax = fmaxf(oldMax, tileMax);
            float const alpha = isinf(oldMax) ? 0.0F : expf(oldMax - newMax);
            float tileSum = 0.0F;
            #pragma unroll
            for (int32_t column = 0; column < kKeyTile; ++column)
            {
                float const probability
                    = expf(storage.scores[lane * kKeyTile + column] - newMax);
                tileSum += probability;
                storage.probabilities[lane * kKeyTile + column]
                    = __float2bfloat16_rn(probability);
            }
            storage.rowAlpha[lane] = alpha;
            storage.rowMax[lane] = newMax;
            storage.rowSum[lane] = storage.rowSum[lane] * alpha + tileSum;
        }
        __syncwarp();

        for (int32_t index = lane; index < kQueryTile * kHeadDim; index += 32)
        {
            int32_t const row = index / kHeadDim;
            storage.numerator[index] *= storage.rowAlpha[row];
        }
        __syncwarp();

        #pragma unroll
        for (int32_t depthTile = 0; depthTile < kHeadDim; depthTile += 16)
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> probabilityFragment;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> valueFragment;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> productAccumulator;
            wmma::load_matrix_sync(
                probabilityFragment, storage.probabilities, kKeyTile);
            wmma::load_matrix_sync(
                valueFragment, valueBase + keyStart * kHeadDim + depthTile, kHeadDim);
            wmma::fill_fragment(productAccumulator, 0.0F);
            wmma::mma_sync(
                productAccumulator, probabilityFragment, valueFragment, productAccumulator);
            wmma::store_matrix_sync(
                storage.product, productAccumulator, 16, wmma::mem_row_major);
            __syncwarp();

            for (int32_t index = lane; index < kQueryTile * 16; index += 32)
            {
                int32_t const row = index / 16;
                int32_t const column = index % 16;
                storage.numerator[row * kHeadDim + depthTile + column]
                    += storage.product[index];
            }
            __syncwarp();
        }
    }

    for (int32_t index = lane; index < kQueryTile * kHeadDim; index += 32)
    {
        int32_t const row = index / kHeadDim;
        int32_t const column = index % kHeadDim;
        float const normalized = storage.numerator[index] / storage.rowSum[row];
        outputBase[(queryStart + row) * kHeadDim + column]
            = __float2bfloat16_rn(normalized);
    }
#endif
}

cudaError_t launchOCABAttention(void const* query, void const* keyTransposed,
    void const* value, void const* relativeBiasTable, void* output,
    int32_t batch, int32_t heads, int32_t queryWindow, int32_t keyWindow,
    cudaStream_t stream) noexcept
{
    if (query == nullptr || keyTransposed == nullptr || value == nullptr
        || relativeBiasTable == nullptr || output == nullptr || batch <= 0 || heads <= 0
        || queryWindow != 32 || keyWindow != 40)
    {
        return cudaErrorInvalidValue;
    }
    int32_t const queryTokens = queryWindow * queryWindow;
    int32_t const blocks = batch * heads * (queryTokens / kQueryTile);
    ocabAttentionWmmaOnline<<<blocks, 32, 0, stream>>>(
        static_cast<__nv_bfloat16 const*>(query),
        static_cast<__nv_bfloat16 const*>(keyTransposed),
        static_cast<__nv_bfloat16 const*>(value),
        static_cast<__nv_bfloat16 const*>(relativeBiasTable),
        static_cast<__nv_bfloat16*>(output),
        batch, heads, queryWindow, keyWindow);
    return cudaPeekAtLastError();
}

__global__ void expandOCABRelativeBiasFlat(__nv_bfloat16 const* table,
    __nv_bfloat16* output, int32_t heads, int32_t queryWindow, int32_t keyWindow)
{
    int32_t const queryTokens = queryWindow * queryWindow;
    int32_t const keyTokens = keyWindow * keyWindow;
    int32_t const biasSpan = queryWindow + keyWindow - 1;
    int64_t const elementCount
        = static_cast<int64_t>(heads) * queryTokens * keyTokens;
    int64_t const stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elementCount; index += stride)
    {
        int32_t const keyToken = static_cast<int32_t>(index % keyTokens);
        int64_t const queryAndHead = index / keyTokens;
        int32_t const queryToken = static_cast<int32_t>(queryAndHead % queryTokens);
        int32_t const head = static_cast<int32_t>(queryAndHead / queryTokens);
        int32_t const queryY = queryToken / queryWindow;
        int32_t const queryX = queryToken % queryWindow;
        int32_t const keyY = keyToken / keyWindow;
        int32_t const keyX = keyToken % keyWindow;
        int32_t const biasIndex
            = (keyY - queryY + queryWindow - 1) * biasSpan
            + (keyX - queryX + queryWindow - 1);
        output[index] = table[biasIndex * heads + head];
    }
}

// Assigning one block to a (head, query-token) plane removes the 64-bit
// div/mod chain from every output element. Eight warps walk the key rows;
// their stores are contiguous and the compact table remains in the exact ONNX
// [relative-position, head] layout. Separate template instantiations preserve
// fully constant Q32/K40 and Q32/K48 address arithmetic.
template <typename Scalar, int32_t keyWindow>
__global__ void expandOCABRelativeBiasQ32(Scalar const* table,
    Scalar* output, int32_t heads)
{
    int32_t constexpr queryWindow = 32;
    int32_t constexpr queryTokens = queryWindow * queryWindow;
    int32_t constexpr keyTokens = keyWindow * keyWindow;
    int32_t constexpr biasSpan = queryWindow + keyWindow - 1;

    int32_t const queryToken = static_cast<int32_t>(blockIdx.x);
    int32_t const head = static_cast<int32_t>(blockIdx.y);
    int32_t const lane = static_cast<int32_t>(threadIdx.x) & 31;
    int32_t const warp = static_cast<int32_t>(threadIdx.x) >> 5;
    int32_t const queryY = queryToken >> 5;
    int32_t const queryX = queryToken & 31;
    int64_t const outputBase
        = (static_cast<int64_t>(head) * queryTokens + queryToken) * keyTokens;

    for (int32_t keyY = warp; keyY < keyWindow; keyY += 8)
    {
        int32_t const biasRow
            = (keyY - queryY + queryWindow - 1) * biasSpan
            + (queryWindow - 1 - queryX);
        int32_t const keyRow = keyY * keyWindow;
        output[outputBase + keyRow + lane]
            = table[(biasRow + lane) * heads + head];
        if (lane < keyWindow - 32)
        {
            output[outputBase + keyRow + 32 + lane]
                = table[(biasRow + 32 + lane) * heads + head];
        }
    }
}

cudaError_t launchOCABRelativeBiasFlat(void const* table, void* output,
    int32_t heads, int32_t queryWindow, int32_t keyWindow,
    cudaStream_t stream) noexcept
{
    if (table == nullptr || output == nullptr || heads <= 0
        || queryWindow != 32 || (keyWindow != 40 && keyWindow != 48))
    {
        return cudaErrorInvalidValue;
    }
    int64_t const elementCount = static_cast<int64_t>(heads)
        * queryWindow * queryWindow * keyWindow * keyWindow;
    constexpr int32_t threads = 256;
    int32_t const blocks = static_cast<int32_t>((elementCount + threads - 1) / threads);
    expandOCABRelativeBiasFlat<<<blocks, threads, 0, stream>>>(
        static_cast<__nv_bfloat16 const*>(table),
        static_cast<__nv_bfloat16*>(output), heads, queryWindow, keyWindow);
    return cudaPeekAtLastError();
}

template <typename Scalar>
cudaError_t launchOCABRelativeBiasTyped(void const* table, void* output,
    int32_t heads, int32_t queryWindow, int32_t keyWindow,
    cudaStream_t stream) noexcept
{
    if (table == nullptr || output == nullptr || heads <= 0
        || queryWindow != 32 || (keyWindow != 40 && keyWindow != 48))
    {
        return cudaErrorInvalidValue;
    }
    constexpr int32_t threads = 256;
    dim3 const blocks(static_cast<uint32_t>(queryWindow * queryWindow),
        static_cast<uint32_t>(heads));
    if (keyWindow == 40)
    {
        expandOCABRelativeBiasQ32<Scalar, 40><<<blocks, threads, 0, stream>>>(
            static_cast<Scalar const*>(table), static_cast<Scalar*>(output), heads);
    }
    else
    {
        expandOCABRelativeBiasQ32<Scalar, 48><<<blocks, threads, 0, stream>>>(
            static_cast<Scalar const*>(table), static_cast<Scalar*>(output), heads);
    }
    return cudaPeekAtLastError();
}

cudaError_t launchOCABRelativeBias(void const* table, void* output,
    int32_t heads, int32_t queryWindow, int32_t keyWindow,
    cudaStream_t stream) noexcept
{
    return launchOCABRelativeBiasTyped<__nv_bfloat16>(
        table, output, heads, queryWindow, keyWindow, stream);
}

bool isConcreteOrWildcard(int64_t actual, int64_t expected) noexcept
{
    return actual == -1 || actual == expected;
}

bool validateDescriptors(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs,
    int32_t queryWindow, int32_t keyWindow) noexcept
{
    if (inputs == nullptr || outputs == nullptr || nbInputs != 4 || nbOutputs != 1)
    {
        return false;
    }
    auto const& q = inputs[0].dims;
    auto const& kt = inputs[1].dims;
    auto const& v = inputs[2].dims;
    auto const& table = inputs[3].dims;
    auto const& out = outputs[0].dims;
    int32_t const queryTokens = queryWindow * queryWindow;
    int32_t const keyTokens = keyWindow * keyWindow;
    int32_t const biasSpan = queryWindow + keyWindow - 1;
    if (q.nbDims != 4 || kt.nbDims != 4 || v.nbDims != 4 || table.nbDims != 2
        || out.nbDims != 4)
    {
        return false;
    }
    return isConcreteOrWildcard(q.d[2], queryTokens)
        && isConcreteOrWildcard(q.d[3], kHeadDim)
        && isConcreteOrWildcard(kt.d[0], q.d[0])
        && isConcreteOrWildcard(kt.d[1], q.d[1])
        && isConcreteOrWildcard(kt.d[2], kHeadDim)
        && isConcreteOrWildcard(kt.d[3], keyTokens)
        && isConcreteOrWildcard(v.d[0], q.d[0])
        && isConcreteOrWildcard(v.d[1], q.d[1])
        && isConcreteOrWildcard(v.d[2], keyTokens)
        && isConcreteOrWildcard(v.d[3], kHeadDim)
        && isConcreteOrWildcard(table.d[0], biasSpan * biasSpan)
        && isConcreteOrWildcard(table.d[1], q.d[1])
        && isConcreteOrWildcard(out.d[0], q.d[0])
        && isConcreteOrWildcard(out.d[1], q.d[1])
        && isConcreteOrWildcard(out.d[2], queryTokens)
        && isConcreteOrWildcard(out.d[3], kHeadDim);
}

bool validateBiasDescriptors(nvinfer1::PluginTensorDesc const* inputs,
    int32_t nbInputs, nvinfer1::PluginTensorDesc const* outputs,
    int32_t nbOutputs, int32_t queryWindow, int32_t keyWindow) noexcept
{
    if (inputs == nullptr || outputs == nullptr || nbInputs != 1 || nbOutputs != 1)
    {
        return false;
    }
    auto const& table = inputs[0].dims;
    auto const& output = outputs[0].dims;
    int32_t const queryTokens = queryWindow * queryWindow;
    int32_t const keyTokens = keyWindow * keyWindow;
    int32_t const biasSpan = queryWindow + keyWindow - 1;
    if (table.nbDims != 2 || output.nbDims != 4)
    {
        return false;
    }
    return isConcreteOrWildcard(table.d[0], biasSpan * biasSpan)
        && (table.d[1] == -1 || table.d[1] > 0)
        && isConcreteOrWildcard(output.d[0], 1)
        && isConcreteOrWildcard(output.d[1], table.d[1])
        && isConcreteOrWildcard(output.d[2], queryTokens)
        && isConcreteOrWildcard(output.d[3], keyTokens);
}

class DrftOCABAttentionPlugin final : public nvinfer1::IPluginV3,
                                      public nvinfer1::IPluginV3OneCore,
                                      public nvinfer1::IPluginV3OneBuild,
                                      public nvinfer1::IPluginV3OneRuntime
{
public:
    DrftOCABAttentionPlugin(int32_t queryWindow, int32_t keyWindow)
        : mQueryWindow(queryWindow), mKeyWindow(keyWindow)
    {
        refreshSerializationFields();
    }

    DrftOCABAttentionPlugin(DrftOCABAttentionPlugin const& other)
        : mQueryWindow(other.mQueryWindow), mKeyWindow(other.mKeyWindow), mTactic(other.mTactic)
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
        return new (std::nothrow) DrftOCABAttentionPlugin(*this);
    }

    char const* getPluginName() const noexcept override { return kPluginName; }
    char const* getPluginVersion() const noexcept override { return kPluginVersion; }
    char const* getPluginNamespace() const noexcept override { return ""; }

    int32_t getNbOutputs() const noexcept override { return 1; }

    int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs,
        nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override
    {
        if (outputTypes == nullptr || inputTypes == nullptr || nbOutputs != 1 || nbInputs != 4)
        {
            return 1;
        }
        outputTypes[0] = inputTypes[0];
        return 0;
    }

    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
        nvinfer1::DimsExprs const*, int32_t nbShapeInputs,
        nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
        nvinfer1::IExprBuilder&) noexcept override
    {
        if (inputs == nullptr || outputs == nullptr || nbInputs != 4
            || nbShapeInputs != 0 || nbOutputs != 1)
        {
            return 1;
        }
        outputs[0] = inputs[0];
        return 0;
    }

    bool supportsFormatCombination(int32_t position,
        nvinfer1::DynamicPluginTensorDesc const* inOut,
        int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        if (inOut == nullptr || nbInputs != 4 || nbOutputs != 1
            || position < 0 || position >= 5)
        {
            return false;
        }
        return inOut[position].desc.format == nvinfer1::TensorFormat::kLINEAR
            && inOut[position].desc.type == nvinfer1::DataType::kBF16;
    }

    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* inputs,
        int32_t nbInputs, nvinfer1::DynamicPluginTensorDesc const* outputs,
        int32_t nbOutputs) noexcept override
    {
        if (inputs == nullptr || outputs == nullptr)
        {
            return 1;
        }
        std::array<nvinfer1::PluginTensorDesc, 4> inputDescriptions{};
        std::array<nvinfer1::PluginTensorDesc, 1> outputDescriptions{};
        for (int32_t index = 0; index < nbInputs && index < 4; ++index)
        {
            inputDescriptions[index] = inputs[index].desc;
        }
        for (int32_t index = 0; index < nbOutputs && index < 1; ++index)
        {
            outputDescriptions[index] = outputs[index].desc;
        }
        return validateDescriptors(inputDescriptions.data(), nbInputs,
                   outputDescriptions.data(), nbOutputs, mQueryWindow, mKeyWindow)
            ? 0 : 1;
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
        tactics[0] = kWmmaOnlineTactic;
        return 0;
    }

    char const* getTimingCacheID() noexcept override
    {
        return "wmma_online_bf16_q32_k40_d32_v1";
    }

    int32_t getFormatCombinationLimit() noexcept override { return 1; }

    char const* getMetadataString() noexcept override
    {
        return "{\"algorithm\":\"wmma_online_softmax\",\"accumulation\":\"fp32\"}";
    }

    int32_t setTactic(int32_t tactic) noexcept override
    {
        if (tactic != kWmmaOnlineTactic)
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
        return validateDescriptors(
                   inputs, nbInputs, outputs, nbOutputs, mQueryWindow, mKeyWindow)
            ? 0 : 1;
    }

    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDescriptions,
        nvinfer1::PluginTensorDesc const*, void const* const* inputs,
        void* const* outputs, void*, cudaStream_t stream) noexcept override
    {
        if (mTactic != kWmmaOnlineTactic || inputDescriptions == nullptr
            || inputs == nullptr || outputs == nullptr)
        {
            return 1;
        }
        int32_t const batch = static_cast<int32_t>(inputDescriptions[0].dims.d[0]);
        int32_t const heads = static_cast<int32_t>(inputDescriptions[0].dims.d[1]);
        cudaError_t const error = launchOCABAttention(
            inputs[0], inputs[1], inputs[2], inputs[3], outputs[0],
            batch, heads, mQueryWindow, mKeyWindow, stream);
        return error == cudaSuccess ? 0 : 1;
    }

    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext*) noexcept override
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
            nvinfer1::PluginField{"q_window_size", &mQueryWindow,
                nvinfer1::PluginFieldType::kINT32, 1},
            nvinfer1::PluginField{"kv_window_size", &mKeyWindow,
                nvinfer1::PluginFieldType::kINT32, 1},
        };
        mSerializationCollection.nbFields
            = static_cast<int32_t>(mSerializationFields.size());
        mSerializationCollection.fields = mSerializationFields.data();
    }

    int32_t mQueryWindow{32};
    int32_t mKeyWindow{40};
    int32_t mTactic{kWmmaOnlineTactic};
    std::array<nvinfer1::PluginField, 2> mSerializationFields{};
    nvinfer1::PluginFieldCollection mSerializationCollection{};
};

class DrftOCABAttentionPluginCreator final : public nvinfer1::IPluginCreatorV3One
{
public:
    DrftOCABAttentionPluginCreator()
    {
        mFields = {
            nvinfer1::PluginField{"q_window_size", nullptr,
                nvinfer1::PluginFieldType::kINT32, 1},
            nvinfer1::PluginField{"kv_window_size", nullptr,
                nvinfer1::PluginFieldType::kINT32, 1},
        };
        mCollection.nbFields = static_cast<int32_t>(mFields.size());
        mCollection.fields = mFields.data();
    }

    nvinfer1::IPluginV3* createPlugin(char const*,
        nvinfer1::PluginFieldCollection const* fields,
        nvinfer1::TensorRTPhase) noexcept override
    {
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
                || field.type != nvinfer1::PluginFieldType::kINT32 || field.length != 1)
            {
                continue;
            }
            if (std::strcmp(field.name, "q_window_size") == 0)
            {
                queryWindow = *static_cast<int32_t const*>(field.data);
            }
            else if (std::strcmp(field.name, "kv_window_size") == 0)
            {
                keyWindow = *static_cast<int32_t const*>(field.data);
            }
        }
        if (queryWindow != 32 || keyWindow != 40)
        {
            return nullptr;
        }
        return new (std::nothrow) DrftOCABAttentionPlugin(queryWindow, keyWindow);
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override
    {
        return &mCollection;
    }

    char const* getPluginName() const noexcept override { return kPluginName; }
    char const* getPluginVersion() const noexcept override { return kPluginVersion; }
    char const* getPluginNamespace() const noexcept override { return ""; }

private:
    std::array<nvinfer1::PluginField, 2> mFields{};
    nvinfer1::PluginFieldCollection mCollection{};
};

class DrftOCABRelativeBiasPlugin final : public nvinfer1::IPluginV3,
                                         public nvinfer1::IPluginV3OneCore,
                                         public nvinfer1::IPluginV3OneBuild,
                                         public nvinfer1::IPluginV3OneRuntime
{
public:
    DrftOCABRelativeBiasPlugin(int32_t queryWindow, int32_t keyWindow)
        : mQueryWindow(queryWindow), mKeyWindow(keyWindow)
    {
        refreshSerializationFields();
    }

    DrftOCABRelativeBiasPlugin(DrftOCABRelativeBiasPlugin const& other)
        : mQueryWindow(other.mQueryWindow), mKeyWindow(other.mKeyWindow),
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
        return new (std::nothrow) DrftOCABRelativeBiasPlugin(*this);
    }

    char const* getPluginName() const noexcept override { return kBiasPluginName; }
    char const* getPluginVersion() const noexcept override { return kBiasPluginVersion; }
    char const* getPluginNamespace() const noexcept override { return ""; }

    int32_t getNbOutputs() const noexcept override { return 1; }

    int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs,
        nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override
    {
        if (outputTypes == nullptr || inputTypes == nullptr || nbOutputs != 1 || nbInputs != 1)
        {
            return 1;
        }
        outputTypes[0] = inputTypes[0];
        return 0;
    }

    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
        nvinfer1::DimsExprs const*, int32_t nbShapeInputs,
        nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
        nvinfer1::IExprBuilder& expressionBuilder) noexcept override
    {
        if (inputs == nullptr || outputs == nullptr || nbInputs != 1
            || nbShapeInputs != 0 || nbOutputs != 1 || inputs[0].nbDims != 2)
        {
            return 1;
        }
        outputs[0].nbDims = 4;
        outputs[0].d[0] = expressionBuilder.constant(1);
        outputs[0].d[1] = inputs[0].d[1];
        outputs[0].d[2] = expressionBuilder.constant(mQueryWindow * mQueryWindow);
        outputs[0].d[3] = expressionBuilder.constant(mKeyWindow * mKeyWindow);
        return 0;
    }

    bool supportsFormatCombination(int32_t position,
        nvinfer1::DynamicPluginTensorDesc const* inOut,
        int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        if (inOut == nullptr || nbInputs != 1 || nbOutputs != 1
            || position < 0 || position >= 2)
        {
            return false;
        }
        bool const admittedType
            = inOut[position].desc.type == nvinfer1::DataType::kBF16
#if defined(DRFT_ENABLE_RELATIVE_BIAS_FP32)
            || inOut[position].desc.type == nvinfer1::DataType::kFLOAT
#endif
            ;
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
        nvinfer1::PluginTensorDesc inputDescription = inputs[0].desc;
        nvinfer1::PluginTensorDesc outputDescription = outputs[0].desc;
        return validateBiasDescriptors(&inputDescription, nbInputs,
                   &outputDescription, nbOutputs, mQueryWindow, mKeyWindow)
            ? 0 : 1;
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
        tactics[0] = kRelativeBiasTactic;
        return 0;
    }

    char const* getTimingCacheID() noexcept override
    {
        return mKeyWindow == 40
            ? "expand_bf16_q32_k40_v2"
            : "expand_bf16_q32_k48_v1";
    }

    int32_t getFormatCombinationLimit() noexcept override { return 1; }

    char const* getMetadataString() noexcept override
    {
        return "{\"algorithm\":\"compact_relative_bias_expand\",\"exact\":true}";
    }

    int32_t setTactic(int32_t tactic) noexcept override
    {
        if (tactic != kRelativeBiasTactic)
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
        return validateBiasDescriptors(
                   inputs, nbInputs, outputs, nbOutputs, mQueryWindow, mKeyWindow)
            ? 0 : 1;
    }

    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDescriptions,
        nvinfer1::PluginTensorDesc const*, void const* const* inputs,
        void* const* outputs, void*, cudaStream_t stream) noexcept override
    {
        if (mTactic != kRelativeBiasTactic || inputDescriptions == nullptr
            || inputs == nullptr || outputs == nullptr)
        {
            return 1;
        }
        int32_t const heads = static_cast<int32_t>(inputDescriptions[0].dims.d[1]);
        cudaError_t error = cudaErrorInvalidValue;
        if (inputDescriptions[0].type == nvinfer1::DataType::kBF16)
        {
            error = launchOCABRelativeBiasTyped<__nv_bfloat16>(
                inputs[0], outputs[0], heads, mQueryWindow, mKeyWindow, stream);
        }
#if defined(DRFT_ENABLE_RELATIVE_BIAS_FP32)
        else if (inputDescriptions[0].type == nvinfer1::DataType::kFLOAT)
        {
            error = launchOCABRelativeBiasTyped<float>(
                inputs[0], outputs[0], heads, mQueryWindow, mKeyWindow, stream);
        }
#endif
        return error == cudaSuccess ? 0 : 1;
    }

    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext*) noexcept override
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
            nvinfer1::PluginField{"q_window_size", &mQueryWindow,
                nvinfer1::PluginFieldType::kINT32, 1},
            nvinfer1::PluginField{"kv_window_size", &mKeyWindow,
                nvinfer1::PluginFieldType::kINT32, 1},
        };
        mSerializationCollection.nbFields
            = static_cast<int32_t>(mSerializationFields.size());
        mSerializationCollection.fields = mSerializationFields.data();
    }

    int32_t mQueryWindow{32};
    int32_t mKeyWindow{40};
    int32_t mTactic{kRelativeBiasTactic};
    std::array<nvinfer1::PluginField, 2> mSerializationFields{};
    nvinfer1::PluginFieldCollection mSerializationCollection{};
};

class DrftOCABRelativeBiasPluginCreator final : public nvinfer1::IPluginCreatorV3One
{
public:
    DrftOCABRelativeBiasPluginCreator()
    {
        mFields = {
            nvinfer1::PluginField{"q_window_size", nullptr,
                nvinfer1::PluginFieldType::kINT32, 1},
            nvinfer1::PluginField{"kv_window_size", nullptr,
                nvinfer1::PluginFieldType::kINT32, 1},
        };
        mCollection.nbFields = static_cast<int32_t>(mFields.size());
        mCollection.fields = mFields.data();
    }

    nvinfer1::IPluginV3* createPlugin(char const*,
        nvinfer1::PluginFieldCollection const* fields,
        nvinfer1::TensorRTPhase) noexcept override
    {
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
                || field.type != nvinfer1::PluginFieldType::kINT32 || field.length != 1)
            {
                continue;
            }
            if (std::strcmp(field.name, "q_window_size") == 0)
            {
                queryWindow = *static_cast<int32_t const*>(field.data);
            }
            else if (std::strcmp(field.name, "kv_window_size") == 0)
            {
                keyWindow = *static_cast<int32_t const*>(field.data);
            }
        }
        if (queryWindow != 32 || (keyWindow != 40 && keyWindow != 48))
        {
            return nullptr;
        }
        return new (std::nothrow) DrftOCABRelativeBiasPlugin(queryWindow, keyWindow);
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override
    {
        return &mCollection;
    }

    char const* getPluginName() const noexcept override { return kBiasPluginName; }
    char const* getPluginVersion() const noexcept override { return kBiasPluginVersion; }
    char const* getPluginNamespace() const noexcept override { return ""; }

private:
    std::array<nvinfer1::PluginField, 2> mFields{};
    nvinfer1::PluginFieldCollection mCollection{};
};
} // namespace

DRFT_PLUGIN_EXPORT int32_t drftLaunchOCABAttention(
    void const* query, void const* keyTransposed, void const* value,
    void const* relativeBiasTable, void* output, int32_t batch, int32_t heads,
    int32_t queryWindow, int32_t keyWindow, cudaStream_t stream)
{
    return static_cast<int32_t>(launchOCABAttention(
        query, keyTransposed, value, relativeBiasTable, output,
        batch, heads, queryWindow, keyWindow, stream));
}

DRFT_PLUGIN_EXPORT int32_t drftLaunchOCABRelativeBias(
    void const* table, void* output, int32_t heads,
    int32_t queryWindow, int32_t keyWindow, cudaStream_t stream)
{
    return static_cast<int32_t>(launchOCABRelativeBias(
        table, output, heads, queryWindow, keyWindow, stream));
}

DRFT_PLUGIN_EXPORT int32_t drftLaunchOCABRelativeBiasFlat(
    void const* table, void* output, int32_t heads,
    int32_t queryWindow, int32_t keyWindow, cudaStream_t stream)
{
    return static_cast<int32_t>(launchOCABRelativeBiasFlat(
        table, output, heads, queryWindow, keyWindow, stream));
}

#if defined(DRFT_ENABLE_IMAGE_LN_PLUGIN)
extern "C" nvinfer1::IPluginCreatorInterface*
drftGetImageLayerNormPluginCreator();
#endif
#if defined(DRFT_ENABLE_IMAGE_LN_FAST_PLUGIN)
extern "C" nvinfer1::IPluginCreatorInterface*
drftGetImageLayerNormFastPluginCreator();
#endif
#if defined(DRFT_ENABLE_OCAB_QKV_PACK_PLUGIN)
extern "C" nvinfer1::IPluginCreatorInterface*
drftGetOCABQKVPackPluginCreator();
#endif
#if defined(DRFT_ENABLE_IMAGE_LN_COOPERATIVE_PLUGIN)
extern "C" nvinfer1::IPluginCreatorInterface*
drftGetImageLayerNormCooperativePluginCreator();
#endif
#if defined(DRFT_ENABLE_FFN_SWIGLU_DEPTHWISE_PLUGIN)
extern "C" nvinfer1::IPluginCreatorInterface*
drftGetFFNSwiGLUDepthwisePluginCreator();
#endif
#if defined(DRFT_ENABLE_RESIDUAL_IMAGE_LN_PLUGIN)
extern "C" nvinfer1::IPluginCreatorInterface*
drftGetResidualImageLayerNormPluginCreator();
#endif
#if defined(DRFT_ENABLE_IDENTITY_BARRIER_PLUGIN)
extern "C" nvinfer1::IPluginCreatorInterface*
drftGetIdentityBarrierPluginCreator();
#endif
#if defined(DRFT_ENABLE_FINAL_AFFINE_CROP_PLUGIN)
extern "C" nvinfer1::IPluginCreatorInterface*
drftGetFinalAffineCropPluginCreator();
#endif
#if defined(DRFT_ENABLE_WINDOW_NEURAL_BIAS_PLUGIN)
extern "C" nvinfer1::IPluginCreatorInterface*
drftGetWindowNeuralBiasPluginCreator();
#endif

DRFT_PLUGIN_EXPORT nvinfer1::IPluginCreatorInterface* const* getCreators(
    int32_t& nbCreators)
{
    static DrftOCABAttentionPluginCreator attentionCreator;
    static DrftOCABRelativeBiasPluginCreator relativeBiasCreator;
    static nvinfer1::IPluginCreatorInterface* const creators[] = {
        &attentionCreator,
        &relativeBiasCreator,
#if defined(DRFT_ENABLE_IMAGE_LN_PLUGIN)
        drftGetImageLayerNormPluginCreator(),
#endif
#if defined(DRFT_ENABLE_IMAGE_LN_FAST_PLUGIN)
        drftGetImageLayerNormFastPluginCreator(),
#endif
#if defined(DRFT_ENABLE_IMAGE_LN_COOPERATIVE_PLUGIN)
        drftGetImageLayerNormCooperativePluginCreator(),
#endif
#if defined(DRFT_ENABLE_OCAB_QKV_PACK_PLUGIN)
        drftGetOCABQKVPackPluginCreator(),
#endif
#if defined(DRFT_ENABLE_FFN_SWIGLU_DEPTHWISE_PLUGIN)
        drftGetFFNSwiGLUDepthwisePluginCreator(),
#endif
#if defined(DRFT_ENABLE_RESIDUAL_IMAGE_LN_PLUGIN)
        drftGetResidualImageLayerNormPluginCreator(),
#endif
#if defined(DRFT_ENABLE_IDENTITY_BARRIER_PLUGIN)
        drftGetIdentityBarrierPluginCreator(),
#endif
#if defined(DRFT_ENABLE_FINAL_AFFINE_CROP_PLUGIN)
        drftGetFinalAffineCropPluginCreator(),
#endif
#if defined(DRFT_ENABLE_WINDOW_NEURAL_BIAS_PLUGIN)
        drftGetWindowNeuralBiasPluginCreator(),
#endif
    };
    nbCreators = static_cast<int32_t>(sizeof(creators) / sizeof(creators[0]));
    return creators;
}

DRFT_PLUGIN_EXPORT void setLoggerFinder(nvinfer1::ILoggerFinder*)
{
}
