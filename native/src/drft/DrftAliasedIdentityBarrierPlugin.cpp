#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>

namespace
{
constexpr char kPluginName[] = "DrftIdentityBarrier_TRT";
constexpr char kPluginVersion[] = "1";

std::size_t elementSize(nvinfer1::DataType type) noexcept
{
    switch (type)
    {
    case nvinfer1::DataType::kFLOAT:
    case nvinfer1::DataType::kINT32:
        return 4;
    case nvinfer1::DataType::kHALF:
    case nvinfer1::DataType::kBF16:
        return 2;
    case nvinfer1::DataType::kINT8:
    case nvinfer1::DataType::kFP8:
        return 1;
    default:
        return 0;
    }
}

bool supportedType(nvinfer1::DataType type) noexcept
{
    return type == nvinfer1::DataType::kFLOAT
        || type == nvinfer1::DataType::kHALF
        || type == nvinfer1::DataType::kBF16
        || type == nvinfer1::DataType::kFP8;
}

class DrftAliasedIdentityBarrierPlugin final
    : public nvinfer1::IPluginV3,
      public nvinfer1::IPluginV3OneCore,
      public nvinfer1::IPluginV3OneBuildV2,
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
            return static_cast<nvinfer1::IPluginV3OneBuildV2*>(this);
        case nvinfer1::PluginCapabilityType::kRUNTIME:
            return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
        default:
            return nullptr;
        }
    }

    nvinfer1::IPluginV3* clone() noexcept override
    {
        return new (std::nothrow) DrftAliasedIdentityBarrierPlugin(*this);
    }

    char const* getPluginName() const noexcept override { return kPluginName; }
    char const* getPluginVersion() const noexcept override { return kPluginVersion; }
    char const* getPluginNamespace() const noexcept override { return ""; }
    int32_t getNbOutputs() const noexcept override { return 1; }

    int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes,
        int32_t nbOutputs, nvinfer1::DataType const* inputTypes,
        int32_t nbInputs) const noexcept override
    {
        if (outputTypes == nullptr || inputTypes == nullptr
            || nbOutputs != 1 || nbInputs != 1
            || !supportedType(inputTypes[0]))
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
        if (inputs == nullptr || outputs == nullptr || nbInputs != 1
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
        if (inOut == nullptr || nbInputs != 1 || nbOutputs != 1
            || position < 0 || position >= 2)
        {
            return false;
        }
        if (position == 0)
        {
            return inOut[0].desc.format == nvinfer1::TensorFormat::kLINEAR
                && supportedType(inOut[0].desc.type);
        }
        return inOut[1].desc.format == inOut[0].desc.format
            && inOut[1].desc.type == inOut[0].desc.type;
    }

    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* inputs,
        int32_t nbInputs, nvinfer1::DynamicPluginTensorDesc const* outputs,
        int32_t nbOutputs) noexcept override
    {
        if (inputs == nullptr || outputs == nullptr || nbInputs != 1
            || nbOutputs != 1)
        {
            return 1;
        }
        return inputs[0].desc.format == nvinfer1::TensorFormat::kLINEAR
                && outputs[0].desc.format == inputs[0].desc.format
                && outputs[0].desc.type == inputs[0].desc.type
                && supportedType(inputs[0].desc.type)
            ? 0 : 1;
    }

    std::size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const*,
        int32_t, nvinfer1::DynamicPluginTensorDesc const*,
        int32_t) const noexcept override
    {
        return 0;
    }

    int32_t getFormatCombinationLimit() noexcept override { return 1; }

    int32_t getAliasedInput(int32_t outputIndex) noexcept override
    {
        // Keep this as a real graph barrier. TensorRT 11.2's preview aliasing
        // optimizer rejects some DRFT tensors that Myelin marks as modified,
        // so enqueue() must receive distinct buffers and perform the copy.
        static_cast<void>(outputIndex);
        return -1;
    }

    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* inputs,
        int32_t nbInputs, nvinfer1::PluginTensorDesc const* outputs,
        int32_t nbOutputs) noexcept override
    {
        if (inputs == nullptr || outputs == nullptr || nbInputs != 1
            || nbOutputs != 1 || inputs[0].dims.nbDims != outputs[0].dims.nbDims
            || inputs[0].type != outputs[0].type
            || inputs[0].format != outputs[0].format)
        {
            return 1;
        }
        for (int32_t i = 0; i < inputs[0].dims.nbDims; ++i)
        {
            if (inputs[0].dims.d[i] != outputs[0].dims.d[i])
            {
                return 1;
            }
        }
        return 0;
    }

    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
        nvinfer1::PluginTensorDesc const*, void const* const* inputs,
        void* const* outputs, void*, cudaStream_t stream) noexcept override
    {
        if (inputDesc == nullptr || inputs == nullptr || outputs == nullptr
            || inputs[0] == nullptr || outputs[0] == nullptr)
        {
            return 1;
        }
        if (inputs[0] == outputs[0])
        {
            return 0;
        }
        std::size_t bytes = elementSize(inputDesc[0].type);
        if (bytes == 0)
        {
            return 1;
        }
        for (int32_t i = 0; i < inputDesc[0].dims.nbDims; ++i)
        {
            int64_t const extent = inputDesc[0].dims.d[i];
            if (extent < 0)
            {
                return 1;
            }
            if (extent == 0)
            {
                return 0;
            }
            auto const unsignedExtent = static_cast<std::size_t>(extent);
            if (bytes > std::numeric_limits<std::size_t>::max() / unsignedExtent)
            {
                return 1;
            }
            bytes *= unsignedExtent;
        }
        return cudaMemcpyAsync(outputs[0], inputs[0], bytes,
                   cudaMemcpyDeviceToDevice, stream) == cudaSuccess
            ? 0 : 1;
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
    nvinfer1::PluginFieldCollection mEmptyFields{0, nullptr};
};

class DrftAliasedIdentityBarrierPluginCreator final
    : public nvinfer1::IPluginCreatorV3One
{
public:
    nvinfer1::IPluginV3* createPlugin(char const*,
        nvinfer1::PluginFieldCollection const*,
        nvinfer1::TensorRTPhase) noexcept override
    {
        return new (std::nothrow) DrftAliasedIdentityBarrierPlugin();
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

extern "C" nvinfer1::IPluginCreatorInterface*
drftGetIdentityBarrierPluginCreator()
{
    static DrftAliasedIdentityBarrierPluginCreator creator;
    return &creator;
}
