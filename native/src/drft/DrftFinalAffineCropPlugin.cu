#include <NvInfer.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <string>

namespace
{
constexpr char kPluginName[] = "DrftFinalAffineCrop_TRT";
constexpr char kPluginVersion[] = "1";

struct SerializedState
{
    int32_t scale;
    float mean[3];
};

__global__ void finalAffineCropKernel(__nv_bfloat16 const* input, float* output,
    int64_t totalElements, int32_t channels, int32_t sourceHeight,
    int32_t sourceWidth, int32_t outputHeight, int32_t outputWidth,
    float mean0, float mean1, float mean2)
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
        int64_t const sourceIndex =
            ((static_cast<int64_t>(batch) * channels + channel) * sourceHeight + y)
            * sourceWidth + x;
        float const mean = channel == 0 ? mean0 : (channel == 1 ? mean1 : mean2);
        output[index] = __bfloat162float(input[sourceIndex]) + mean;
    }
}

class DrftFinalAffineCropPlugin final : public nvinfer1::IPluginV2DynamicExt
{
public:
    DrftFinalAffineCropPlugin(int32_t scale, std::array<float, 3> mean)
        : mScale(scale), mMean(mean)
    {
    }

    DrftFinalAffineCropPlugin(DrftFinalAffineCropPlugin const&) = default;

    char const* getPluginType() const noexcept override { return kPluginName; }
    char const* getPluginVersion() const noexcept override { return kPluginVersion; }
    int32_t getNbOutputs() const noexcept override { return 1; }

    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex,
        nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
        nvinfer1::IExprBuilder& expressionBuilder) noexcept override
    {
        if (outputIndex != 0 || nbInputs != 2 || inputs[1].nbDims != 4)
        {
            return nvinfer1::DimsExprs{-1, {}};
        }
        nvinfer1::DimsExprs output = inputs[1];
        auto const* scale = expressionBuilder.constant(mScale);
        output.d[2] = expressionBuilder.operation(
            nvinfer1::DimensionOperation::kPROD, *inputs[1].d[2], *scale);
        output.d[3] = expressionBuilder.operation(
            nvinfer1::DimensionOperation::kPROD, *inputs[1].d[3], *scale);
        return output;
    }

    bool supportsFormatCombination(int32_t pos,
        nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs,
        int32_t nbOutputs) noexcept override
    {
        if (nbInputs != 2 || nbOutputs != 1 || pos < 0 || pos >= 3)
        {
            return false;
        }
        if (inOut[pos].format != nvinfer1::TensorFormat::kLINEAR)
        {
            return false;
        }
        if (pos == 0)
        {
            return inOut[0].type == nvinfer1::DataType::kBF16;
        }
        return inOut[pos].type == nvinfer1::DataType::kFLOAT;
    }

    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
        nvinfer1::DynamicPluginTensorDesc const*, int32_t) noexcept override
    {
    }

    std::size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
        nvinfer1::PluginTensorDesc const*, int32_t) const noexcept override
    {
        return 0;
    }

    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
        nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void*, cudaStream_t stream) noexcept override
    {
        if (inputDesc == nullptr || outputDesc == nullptr || inputs == nullptr
            || outputs == nullptr || inputs[0] == nullptr || outputs[0] == nullptr
            || inputDesc[0].dims.nbDims != 4 || inputDesc[1].dims.nbDims != 4
            || outputDesc[0].dims.nbDims != 4)
        {
            return 1;
        }

        int32_t const batch = static_cast<int32_t>(outputDesc[0].dims.d[0]);
        int32_t const channels = static_cast<int32_t>(outputDesc[0].dims.d[1]);
        int32_t const outputHeight = static_cast<int32_t>(outputDesc[0].dims.d[2]);
        int32_t const outputWidth = static_cast<int32_t>(outputDesc[0].dims.d[3]);
        int32_t const sourceHeight = static_cast<int32_t>(inputDesc[0].dims.d[2]);
        int32_t const sourceWidth = static_cast<int32_t>(inputDesc[0].dims.d[3]);
        if (batch < 0 || channels != 3 || outputHeight < 0 || outputWidth < 0
            || sourceHeight < outputHeight || sourceWidth < outputWidth)
        {
            return 1;
        }
        if (batch == 0 || outputHeight == 0 || outputWidth == 0)
        {
            return 0;
        }

        int64_t const totalElements = static_cast<int64_t>(batch) * channels
            * outputHeight * outputWidth;
        constexpr int32_t threads = 256;
        int64_t const requestedBlocks = (totalElements + threads - 1) / threads;
        int32_t const blocks = static_cast<int32_t>(
            std::min<int64_t>(requestedBlocks, 65535));
        finalAffineCropKernel<<<blocks, threads, 0, stream>>>(
            static_cast<__nv_bfloat16 const*>(inputs[0]),
            static_cast<float*>(outputs[0]),
            totalElements, channels, sourceHeight, sourceWidth, outputHeight,
            outputWidth, mMean[0], mMean[1], mMean[2]);
        return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
    }

    int32_t initialize() noexcept override { return 0; }
    void terminate() noexcept override {}

    std::size_t getSerializationSize() const noexcept override
    {
        return sizeof(SerializedState);
    }

    void serialize(void* buffer) const noexcept override
    {
        if (buffer == nullptr)
        {
            return;
        }
        SerializedState const state{
            mScale, {mMean[0], mMean[1], mMean[2]}};
        std::memcpy(buffer, &state, sizeof(state));
    }

    void destroy() noexcept override { delete this; }

    DrftFinalAffineCropPlugin* clone() const noexcept override
    {
        return new (std::nothrow) DrftFinalAffineCropPlugin(*this);
    }

    void setPluginNamespace(char const* pluginNamespace) noexcept override
    {
        mNamespace = pluginNamespace == nullptr ? "" : pluginNamespace;
    }

    char const* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

    nvinfer1::DataType getOutputDataType(int32_t index,
        nvinfer1::DataType const*, int32_t) const noexcept override
    {
        return index == 0 ? nvinfer1::DataType::kFLOAT
                          : nvinfer1::DataType::kFLOAT;
    }

private:
    int32_t mScale;
    std::array<float, 3> mMean;
    std::string mNamespace;
};

class DrftFinalAffineCropPluginCreator final : public nvinfer1::IPluginCreator
{
public:
    DrftFinalAffineCropPluginCreator()
    {
        mFields = {{
            {"scale", nullptr, nvinfer1::PluginFieldType::kINT32, 1},
            {"mean_r", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1},
            {"mean_g", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1},
            {"mean_b", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1},
        }};
        mCollection.nbFields = static_cast<int32_t>(mFields.size());
        mCollection.fields = mFields.data();
    }

    char const* getPluginName() const noexcept override { return kPluginName; }
    char const* getPluginVersion() const noexcept override { return kPluginVersion; }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override
    {
        return &mCollection;
    }

    nvinfer1::IPluginV2* createPlugin(char const*,
        nvinfer1::PluginFieldCollection const* fields) noexcept override
    {
        int32_t scale = 4;
        std::array<float, 3> mean{0.44921875F, 0.4375F, 0.404296875F};
        if (fields != nullptr)
        {
            for (int32_t i = 0; i < fields->nbFields; ++i)
            {
                auto const& field = fields->fields[i];
                if (field.name == nullptr || field.data == nullptr || field.length != 1)
                {
                    continue;
                }
                if (std::strcmp(field.name, "scale") == 0)
                {
                    scale = *static_cast<int32_t const*>(field.data);
                }
                else if (std::strcmp(field.name, "mean_r") == 0)
                {
                    mean[0] = *static_cast<float const*>(field.data);
                }
                else if (std::strcmp(field.name, "mean_g") == 0)
                {
                    mean[1] = *static_cast<float const*>(field.data);
                }
                else if (std::strcmp(field.name, "mean_b") == 0)
                {
                    mean[2] = *static_cast<float const*>(field.data);
                }
            }
        }
        if (scale <= 0 || !std::isfinite(mean[0]) || !std::isfinite(mean[1])
            || !std::isfinite(mean[2]))
        {
            return nullptr;
        }
        auto* plugin = new (std::nothrow) DrftFinalAffineCropPlugin(scale, mean);
        if (plugin != nullptr)
        {
            plugin->setPluginNamespace(mNamespace.c_str());
        }
        return plugin;
    }

    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
        std::size_t length) noexcept override
    {
        if (data == nullptr || length != sizeof(SerializedState))
        {
            return nullptr;
        }
        SerializedState state{};
        std::memcpy(&state, data, sizeof(state));
        std::array<float, 3> const mean{
            state.mean[0], state.mean[1], state.mean[2]};
        if (state.scale <= 0 || !std::isfinite(mean[0]) || !std::isfinite(mean[1])
            || !std::isfinite(mean[2]))
        {
            return nullptr;
        }
        auto* plugin = new (std::nothrow) DrftFinalAffineCropPlugin(state.scale, mean);
        if (plugin != nullptr)
        {
            plugin->setPluginNamespace(mNamespace.c_str());
        }
        return plugin;
    }

    void setPluginNamespace(char const* pluginNamespace) noexcept override
    {
        mNamespace = pluginNamespace == nullptr ? "" : pluginNamespace;
    }

    char const* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

private:
    std::array<nvinfer1::PluginField, 4> mFields{};
    nvinfer1::PluginFieldCollection mCollection{};
    std::string mNamespace;
};
} // namespace

extern "C" nvinfer1::IPluginCreatorInterface*
drftGetFinalAffineCropPluginCreator()
{
    static DrftFinalAffineCropPluginCreator creator;
    return &creator;
}
