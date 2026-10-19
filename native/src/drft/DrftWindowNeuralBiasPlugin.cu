#include <NvInfer.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <new>

namespace
{
constexpr char kPluginName[] = "DrftWindowNeuralBias_TRT";
constexpr char kPluginVersion[] = "1";
constexpr int32_t kTactic = 3;
constexpr int32_t kTile = 16;

__global__ void windowNeuralBiasKernel(__nv_bfloat16 const* queryFactor,
    __nv_bfloat16 const* keyFactor, __nv_bfloat16* output,
    int32_t heads, int32_t queryTokens, int32_t keyTokens, int32_t rank)
{
    __shared__ __nv_bfloat16 queryTile[kTile][kTile];
    __shared__ __nv_bfloat16 keyTile[kTile][kTile];

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
        queryTile[threadIdx.y][threadIdx.x]
            = row < queryTokens && queryRank < rank
            ? queryFactor[(head * queryTokens + row) * rank + queryRank]
            : __float2bfloat16(0.0F);
        keyTile[threadIdx.y][threadIdx.x]
            = column < keyTokens && keyRank < rank
            ? keyFactor[(head * keyTokens + column) * rank + keyRank]
            : __float2bfloat16(0.0F);
        __syncthreads();

#pragma unroll
        for (int32_t index = 0; index < kTile; ++index)
        {
            accumulator += __bfloat162float(queryTile[threadIdx.y][index])
                * __bfloat162float(keyTile[index][threadIdx.x]);
        }
        __syncthreads();
    }

    if (row < queryTokens && column < keyTokens && head < heads)
    {
        output[(head * queryTokens + row) * keyTokens + column]
            = __float2bfloat16_rn(accumulator);
    }
}

bool validateDescriptors(nvinfer1::PluginTensorDesc const* inputs,
    int32_t nbInputs, nvinfer1::PluginTensorDesc const* outputs,
    int32_t nbOutputs) noexcept
{
    if (inputs == nullptr || outputs == nullptr || nbInputs != 2 || nbOutputs != 1
        || inputs[0].dims.nbDims != 3 || inputs[1].dims.nbDims != 3
        || outputs[0].dims.nbDims != 3)
    {
        return false;
    }
    if (inputs[0].type != nvinfer1::DataType::kBF16
        || inputs[1].type != inputs[0].type || outputs[0].type != inputs[0].type
        || inputs[0].format != nvinfer1::TensorFormat::kLINEAR
        || inputs[1].format != inputs[0].format || outputs[0].format != inputs[0].format)
    {
        return false;
    }
    auto const& query = inputs[0].dims;
    auto const& key = inputs[1].dims;
    auto const& output = outputs[0].dims;
    return query.d[0] > 0 && query.d[1] > 0 && query.d[2] > 0
        && key.d[0] == query.d[0] && key.d[1] > 0 && key.d[2] == query.d[2]
        && output.d[0] == query.d[0] && output.d[1] == query.d[1]
        && output.d[2] == key.d[1];
}

class DrftWindowNeuralBiasPlugin final
    : public nvinfer1::IPluginV3,
      public nvinfer1::IPluginV3OneCore,
      public nvinfer1::IPluginV3OneBuild,
      public nvinfer1::IPluginV3OneRuntime
{
public:
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
        return new (std::nothrow) DrftWindowNeuralBiasPlugin(*this);
    }

    char const* getPluginName() const noexcept override { return kPluginName; }
    char const* getPluginVersion() const noexcept override { return kPluginVersion; }
    char const* getPluginNamespace() const noexcept override { return ""; }
    int32_t getNbOutputs() const noexcept override { return 1; }

    int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes,
        int32_t nbOutputs, nvinfer1::DataType const* inputTypes,
        int32_t nbInputs) const noexcept override
    {
        if (outputTypes == nullptr || inputTypes == nullptr || nbOutputs != 1
            || nbInputs != 2 || inputTypes[0] != nvinfer1::DataType::kBF16
            || inputTypes[1] != inputTypes[0])
        {
            return 1;
        }
        outputTypes[0] = inputTypes[0];
        return 0;
    }

    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs,
        int32_t nbInputs, nvinfer1::DimsExprs const*, int32_t nbShapeInputs,
        nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
        nvinfer1::IExprBuilder&) noexcept override
    {
        if (inputs == nullptr || outputs == nullptr || nbInputs != 2
            || nbShapeInputs != 0 || nbOutputs != 1 || inputs[0].nbDims != 3
            || inputs[1].nbDims != 3)
        {
            return 1;
        }
        outputs[0].nbDims = 3;
        outputs[0].d[0] = inputs[0].d[0];
        outputs[0].d[1] = inputs[0].d[1];
        outputs[0].d[2] = inputs[1].d[1];
        return 0;
    }

    bool supportsFormatCombination(int32_t position,
        nvinfer1::DynamicPluginTensorDesc const* inOut,
        int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        if (inOut == nullptr || nbInputs != 2 || nbOutputs != 1
            || position < 0 || position >= 3)
        {
            return false;
        }
        return inOut[position].desc.type == nvinfer1::DataType::kBF16
            && inOut[position].desc.format == nvinfer1::TensorFormat::kLINEAR;
    }

    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* inputs,
        int32_t nbInputs, nvinfer1::DynamicPluginTensorDesc const* outputs,
        int32_t nbOutputs) noexcept override
    {
        if (inputs == nullptr || outputs == nullptr)
        {
            return 1;
        }
        nvinfer1::PluginTensorDesc inputDescriptions[2]
            = {inputs[0].desc, inputs[1].desc};
        nvinfer1::PluginTensorDesc outputDescription = outputs[0].desc;
        return validateDescriptors(inputDescriptions, nbInputs,
                   &outputDescription, nbOutputs)
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
        tactics[0] = kTactic;
        return 0;
    }

    char const* getTimingCacheID() noexcept override
    {
        return "bf16_hnr_hmr_tiled16_v1";
    }

    int32_t getFormatCombinationLimit() noexcept override { return 1; }

    char const* getMetadataString() noexcept override
    {
        return "{\"algorithm\":\"bf16_low_rank_bias_tiled16\",\"accumulator\":\"fp32\"}";
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
        return validateDescriptors(inputs, nbInputs, outputs, nbOutputs) ? 0 : 1;
    }

    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDescriptions,
        nvinfer1::PluginTensorDesc const*, void const* const* inputs,
        void* const* outputs, void*, cudaStream_t stream) noexcept override
    {
        if (mTactic != kTactic || inputDescriptions == nullptr || inputs == nullptr
            || outputs == nullptr || inputs[0] == nullptr || inputs[1] == nullptr
            || outputs[0] == nullptr)
        {
            return 1;
        }
        auto const& query = inputDescriptions[0].dims;
        auto const& key = inputDescriptions[1].dims;
        int32_t const heads = static_cast<int32_t>(query.d[0]);
        int32_t const queryTokens = static_cast<int32_t>(query.d[1]);
        int32_t const keyTokens = static_cast<int32_t>(key.d[1]);
        int32_t const rank = static_cast<int32_t>(query.d[2]);
        dim3 const block(kTile, kTile);
        dim3 const grid(
            static_cast<uint32_t>((keyTokens + kTile - 1) / kTile),
            static_cast<uint32_t>((queryTokens + kTile - 1) / kTile),
            static_cast<uint32_t>(heads));
        windowNeuralBiasKernel<<<grid, block, 0, stream>>>(
            static_cast<__nv_bfloat16 const*>(inputs[0]),
            static_cast<__nv_bfloat16 const*>(inputs[1]),
            static_cast<__nv_bfloat16*>(outputs[0]), heads,
            queryTokens, keyTokens, rank);
        return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
    }

    nvinfer1::IPluginV3* attachToContext(
        nvinfer1::IPluginResourceContext*) noexcept override
    {
        return clone();
    }

    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override
    {
        return &mEmptyFields;
    }

private:
    int32_t mTactic{kTactic};
    nvinfer1::PluginFieldCollection mEmptyFields{0, nullptr};
};

class DrftWindowNeuralBiasPluginCreator final
    : public nvinfer1::IPluginCreatorV3One
{
public:
    nvinfer1::IPluginV3* createPlugin(char const*,
        nvinfer1::PluginFieldCollection const*,
        nvinfer1::TensorRTPhase) noexcept override
    {
        return new (std::nothrow) DrftWindowNeuralBiasPlugin();
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override
    {
        return &mEmptyFields;
    }

    char const* getPluginName() const noexcept override { return kPluginName; }
    char const* getPluginVersion() const noexcept override { return kPluginVersion; }
    char const* getPluginNamespace() const noexcept override { return ""; }

private:
    nvinfer1::PluginFieldCollection mEmptyFields{0, nullptr};
};
} // namespace

#if defined(_WIN32)
#define DRFT_WINDOW_BIAS_EXPORT extern "C" __declspec(dllexport)
#else
#define DRFT_WINDOW_BIAS_EXPORT extern "C" __attribute__((visibility("default")))
#endif

DRFT_WINDOW_BIAS_EXPORT int32_t drftLaunchWindowNeuralBias(
    void const* queryFactor, void const* keyFactor, void* output,
    int32_t heads, int32_t queryTokens, int32_t keyTokens, int32_t rank,
    cudaStream_t stream)
{
    if (queryFactor == nullptr || keyFactor == nullptr || output == nullptr
        || heads <= 0 || queryTokens <= 0 || keyTokens <= 0 || rank <= 0)
    {
        return static_cast<int32_t>(cudaErrorInvalidValue);
    }
    dim3 const block(kTile, kTile);
    dim3 const grid(
        static_cast<uint32_t>((keyTokens + kTile - 1) / kTile),
        static_cast<uint32_t>((queryTokens + kTile - 1) / kTile),
        static_cast<uint32_t>(heads));
    windowNeuralBiasKernel<<<grid, block, 0, stream>>>(
        static_cast<__nv_bfloat16 const*>(queryFactor),
        static_cast<__nv_bfloat16 const*>(keyFactor),
        static_cast<__nv_bfloat16*>(output), heads,
        queryTokens, keyTokens, rank);
    return static_cast<int32_t>(cudaPeekAtLastError());
}

extern "C" nvinfer1::IPluginCreatorInterface*
drftGetWindowNeuralBiasPluginCreator()
{
    static DrftWindowNeuralBiasPluginCreator creator;
    return &creator;
}
