#include "Export.h"
#define STR1(x) #x
#define STR(x) STR1(x)
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda.h>
#include <array>
#include <mutex>
#include <new>

struct Blob { int warps, shared; const char* name; const unsigned char* data; };
#include DUAL_PROJECT_HEADER

namespace {
using namespace nvinfer1;
constexpr char NAME[] = "DualProject_" DUAL_KEY "_TRT";
class Plugin final : public IPluginV3, public IPluginV3OneCore,
    public IPluginV3OneBuild, public IPluginV3OneRuntime {
    CUcontext context{};
    CUmodule module{};
    CUfunction function{};
    std::mutex mutex;
    PluginFieldCollection fields{};
public:
    ~Plugin() override {
        if (context && module && cuCtxPushCurrent(context) == CUDA_SUCCESS) {
            cuModuleUnload(module);
            CUcontext previous{}; cuCtxPopCurrent(&previous);
        }
    }
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override {
        switch (type) {
        case PluginCapabilityType::kCORE: return static_cast<IPluginV3OneCore*>(this);
        case PluginCapabilityType::kBUILD: return static_cast<IPluginV3OneBuild*>(this);
        case PluginCapabilityType::kRUNTIME: return static_cast<IPluginV3OneRuntime*>(this);
        default: return nullptr;
        }
    }
    IPluginV3* clone() noexcept override { return new(std::nothrow) Plugin; }
    char const* getPluginName() const noexcept override { return NAME; }
    char const* getPluginVersion() const noexcept override { return "1"; }
    char const* getPluginNamespace() const noexcept override { return ""; }
    int32_t getNbOutputs() const noexcept override { return 1; }
    int32_t getOutputDataTypes(DataType* out, int32_t no, DataType const* in, int32_t ni) const noexcept override {
        if (!out || !in || no != 1 || ni != 3) return 1;
        out[0] = DataType::kBF16; return 0;
    }
    int32_t getOutputShapes(DimsExprs const* in, int32_t ni, DimsExprs const*, int32_t ns,
        DimsExprs* out, int32_t no, IExprBuilder&) noexcept override {
        if (!in || !out || ni != 3 || no != 1 || ns || in[0].nbDims != 4) return 1;
        out[0] = in[0]; return 0;
    }
    bool supportsFormatCombination(int32_t p, DynamicPluginTensorDesc const* io, int32_t ni, int32_t no) noexcept override {
        return io && ni == 3 && no == 1 && p >= 0 && p < 4 && io[p].desc.format == TensorFormat::kLINEAR && io[p].desc.type == DataType::kBF16;
    }
    bool valid(PluginTensorDesc const* in, int ni, PluginTensorDesc const* out, int no) const {
        if (!in || !out || ni != 3 || no != 1) return false;
        auto shape = [](Dims const& d) { return d.nbDims == 4 && d.d[0] == 1 && d.d[1] == DUAL_H && d.d[2] == DUAL_W && d.d[3] == DUAL_C; };
        return shape(in[0].dims) && shape(in[2].dims) && shape(out[0].dims) && in[1].dims.nbDims == 2 &&
            in[1].dims.d[0] == DUAL_C && in[1].dims.d[1] == DUAL_C && in[0].type == DataType::kBF16 &&
            in[1].type == DataType::kBF16 && in[2].type == DataType::kBF16 && out[0].type == DataType::kBF16;
    }
    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t ni, DynamicPluginTensorDesc const* out, int32_t no) noexcept override {
        if (!in || !out || ni != 3 || no != 1) return 1;
        std::array<PluginTensorDesc,3> desc; for (int i=0;i<3;++i) desc[i]=in[i].desc;
        return valid(desc.data(),ni,&out[0].desc,no) ? 0 : 1;
    }
    size_t getWorkspaceSize(DynamicPluginTensorDesc const*,int32_t,DynamicPluginTensorDesc const*,int32_t) const noexcept override { return 0; }
    char const* getTimingCacheID() noexcept override { return "dual_project_v1_" DUAL_KEY "_sm" STR(DUAL_SM); }
    char const* getMetadataString() noexcept override { return "{\"projection_round_before_add\":\"bf16\"}"; }
    int32_t onShapeChange(PluginTensorDesc const* in,int32_t ni,PluginTensorDesc const* out,int32_t no) noexcept override { return valid(in,ni,out,no) ? 0 : 1; }
    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }
    PluginFieldCollection const* getFieldsToSerialize() noexcept override { return &fields; }
    int32_t enqueue(PluginTensorDesc const*,PluginTensorDesc const*,void const* const* in,void* const* out,void*,cudaStream_t stream) noexcept override {
        if (!in || !out) return 1;
        CUcontext current{}; auto error=cuCtxGetCurrent(&current); if(error!=CUDA_SUCCESS || !current) return 1;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (context && context != current) return 1;
            context=current;
            if (!module) {
                error=cuModuleLoadData(&module,project_kernel.data); if(error!=CUDA_SUCCESS) return error;
                error=cuModuleGetFunction(&function,module,project_kernel.name);
                if(error!=CUDA_SUCCESS) { cuModuleUnload(module); module=nullptr; return error; }
                if(project_kernel.shared>49152) { error=cuFuncSetAttribute(function,CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,project_kernel.shared); if(error!=CUDA_SUCCESS) return error; }
            }
        }
        std::array<void const*,6> values{in[0],in[1],in[2],out[0],nullptr,nullptr};
        std::array<void*,6> args{}; for(int i=0;i<6;++i) args[i]=&values[i];
        return cuLaunchKernel(function,(DUAL_H*DUAL_W+PROJECT_BM-1)/PROJECT_BM,1,1,32*project_kernel.warps,1,1,project_kernel.shared,reinterpret_cast<CUstream>(stream),args.data(),nullptr);
    }
};
class Creator final : public IPluginCreatorV3One {
    PluginFieldCollection fields{};
public:
    IPluginV3* createPlugin(char const*,PluginFieldCollection const* f,TensorRTPhase) noexcept override { if(!f || f->nbFields) return nullptr; return new(std::nothrow) Plugin; }
    PluginFieldCollection const* getFieldNames() noexcept override { return &fields; }
    char const* getPluginName() const noexcept override { return NAME; }
    char const* getPluginVersion() const noexcept override { return "1"; }
    char const* getPluginNamespace() const noexcept override { return ""; }
};
}
extern "C" DUAL_EXPORT nvinfer1::IPluginCreatorInterface* const* getCreators(int32_t& count) {
    static Creator creator; static nvinfer1::IPluginCreatorInterface* const creators[]={&creator}; count=1; return creators;
}
extern "C" DUAL_EXPORT void setLoggerFinder(nvinfer1::ILoggerFinder*) {}
