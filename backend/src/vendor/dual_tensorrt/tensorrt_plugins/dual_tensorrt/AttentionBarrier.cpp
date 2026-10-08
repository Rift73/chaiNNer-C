#include "Export.h"
// Deployment-only bitwise copy. Prevents TensorRT commoning equal-temperature
// XR/XG logits/softmax into a long-lived quadratic tensor with two consumers.
// No arithmetic, changed temperature, attention approximation or model parameter.
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>
#include <new>

namespace {
constexpr char kName[] = "DualAttentionBarrier_TRT";
bool valid(nvinfer1::PluginTensorDesc const& a,
           nvinfer1::PluginTensorDesc const& b) {
    using namespace nvinfer1;
    if (a.type != DataType::kBF16 || b.type != a.type
        || a.format != TensorFormat::kLINEAR || b.format != a.format
        || a.dims.nbDims != 4 || b.dims.nbDims != 4) return false;
    for (int i=0; i<4; ++i)
        if (a.dims.d[i] == 0 || a.dims.d[i] < -1
            || a.dims.d[i] != b.dims.d[i]) return false;
    return true;
}
class Plugin final : public nvinfer1::IPluginV3, public nvinfer1::IPluginV3OneCore,
    public nvinfer1::IPluginV3OneBuild, public nvinfer1::IPluginV3OneRuntime {
public:
    nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType t) noexcept override {
        using namespace nvinfer1;
        switch(t) {
        case PluginCapabilityType::kCORE: return static_cast<IPluginV3OneCore*>(this);
        case PluginCapabilityType::kBUILD: return static_cast<IPluginV3OneBuild*>(this);
        case PluginCapabilityType::kRUNTIME: return static_cast<IPluginV3OneRuntime*>(this);
        default: return nullptr;
        }
    }
    nvinfer1::IPluginV3* clone() noexcept override { return new(std::nothrow) Plugin(); }
    char const* getPluginName() const noexcept override { return kName; }
    char const* getPluginVersion() const noexcept override { return "1"; }
    char const* getPluginNamespace() const noexcept override { return ""; }
    int32_t getNbOutputs() const noexcept override { return 1; }
    int32_t getOutputDataTypes(nvinfer1::DataType* o, int32_t no,
        nvinfer1::DataType const* i, int32_t ni) const noexcept override {
        if (!o || !i || ni!=1 || no!=1 || i[0]!=nvinfer1::DataType::kBF16) return 1;
        o[0]=i[0]; return 0;
    }
    int32_t getOutputShapes(nvinfer1::DimsExprs const* i, int32_t ni,
        nvinfer1::DimsExprs const*, int32_t ns, nvinfer1::DimsExprs* o,
        int32_t no, nvinfer1::IExprBuilder&) noexcept override {
        if (!i || !o || ni!=1 || no!=1 || ns!=0 || i[0].nbDims!=4) return 1;
        o[0]=i[0]; return 0;
    }
    bool supportsFormatCombination(int32_t p, nvinfer1::DynamicPluginTensorDesc const* io,
        int32_t ni, int32_t no) noexcept override {
        return io && ni==1 && no==1 && p>=0 && p<2
            && io[p].desc.format==nvinfer1::TensorFormat::kLINEAR
            && io[p].desc.type==nvinfer1::DataType::kBF16;
    }
    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* i, int32_t ni,
        nvinfer1::DynamicPluginTensorDesc const* o, int32_t no) noexcept override {
        return i && o && ni==1 && no==1 && valid(i[0].desc,o[0].desc) ? 0:1;
    }
    size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
        nvinfer1::DynamicPluginTensorDesc const*, int32_t) const noexcept override { return 0; }
    char const* getTimingCacheID() noexcept override { return "bf16_bitcopy_v1"; }
    char const* getMetadataString() noexcept override { return "{\"operation\":\"bitwise_identity\",\"purpose\":\"separate_attention_fusion\"}"; }
    int32_t getFormatCombinationLimit() noexcept override { return 1; }
    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* i, int32_t ni,
        nvinfer1::PluginTensorDesc const* o, int32_t no) noexcept override {
        return i && o && ni==1 && no==1 && valid(i[0],o[0]) ? 0:1;
    }
    int32_t enqueue(nvinfer1::PluginTensorDesc const* d,
        nvinfer1::PluginTensorDesc const*, void const* const* i, void* const* o,
        void*, cudaStream_t stream) noexcept override {
        if (!d || !i || !o || !i[0] || !o[0] || d[0].dims.nbDims!=4) return 1;
        size_t bytes=2;
        for(int k=0;k<4;++k) {
            auto dim=d[0].dims.d[k];
            if(dim<1 || static_cast<uint64_t>(dim)>std::numeric_limits<size_t>::max()/bytes) return 1;
            bytes*=static_cast<size_t>(dim);
        }
        return cudaMemcpyAsync(o[0],i[0],bytes,cudaMemcpyDeviceToDevice,stream)==cudaSuccess ? 0:1;
    }
    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext*) noexcept override { return clone(); }
    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override { return &fields; }
private:
    nvinfer1::PluginFieldCollection fields{0,nullptr};
};
class Creator final : public nvinfer1::IPluginCreatorV3One {
public:
    nvinfer1::IPluginV3* createPlugin(char const*, nvinfer1::PluginFieldCollection const* f,
        nvinfer1::TensorRTPhase) noexcept override {
        return f && f->nbFields==0 ? new(std::nothrow) Plugin():nullptr;
    }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fields; }
    char const* getPluginName() const noexcept override { return kName; }
    char const* getPluginVersion() const noexcept override { return "1"; }
    char const* getPluginNamespace() const noexcept override { return ""; }
private:
    nvinfer1::PluginFieldCollection fields{0,nullptr};
};
}
extern "C" DUAL_EXPORT nvinfer1::IPluginCreatorInterface* const* getCreators(int32_t& n) {
    static Creator creator;
    static nvinfer1::IPluginCreatorInterface* const creators[]={&creator};
    n=1; return creators;
}
extern "C" DUAL_EXPORT void setLoggerFinder(nvinfer1::ILoggerFinder*) {}
