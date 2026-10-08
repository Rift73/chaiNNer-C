#include "Export.h"
#define STR1(x) #x
#define STR(x) STR1(x)
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda.h>
#include <array>
#include <mutex>
#include <new>
#include DUAL_AOT_HEADER

namespace {
using namespace nvinfer1;
constexpr char NAME[]="DualCore_" DUAL_KEY "_TRT";
constexpr int CELLS=((DUAL_H+15)/16)*((DUAL_W+15)/16);
constexpr int HEADS=DUAL_C/32;
constexpr int REGIONS=((DUAL_H+31)/32+(DUAL_H+47)/32)*((DUAL_W+31)/32+(DUAL_W+47)/32);
constexpr int INPUTS=3;
constexpr char VERSION[]="1";
constexpr size_t COUNT=sizeof(blobs)/sizeof(blobs[0]);
class Plugin final: public IPluginV3, public IPluginV3OneCore,
    public IPluginV3OneBuild, public IPluginV3OneRuntime {
    CUcontext context{};
    std::array<CUmodule,COUNT> modules{};
    std::array<CUfunction,COUNT> functions{};
    std::mutex mutex;
    PluginFieldCollection fields{};
public:
    ~Plugin() override {
        if(context && cuCtxPushCurrent(context)==CUDA_SUCCESS){
            for(auto m:modules)if(m)cuModuleUnload(m);
            CUcontext old{};cuCtxPopCurrent(&old);
        }
    }
    IPluginCapability* getCapabilityInterface(PluginCapabilityType t)noexcept override {
        switch(t){case PluginCapabilityType::kCORE:return static_cast<IPluginV3OneCore*>(this);
        case PluginCapabilityType::kBUILD:return static_cast<IPluginV3OneBuild*>(this);
        case PluginCapabilityType::kRUNTIME:return static_cast<IPluginV3OneRuntime*>(this);default:return nullptr;}
    }
    IPluginV3* clone()noexcept override{return new(std::nothrow)Plugin;}
    char const* getPluginName()const noexcept override{return NAME;}
    char const* getPluginVersion()const noexcept override{return VERSION;}
    char const* getPluginNamespace()const noexcept override{return "";}
    int32_t getNbOutputs()const noexcept override{return 1;}
    int32_t getOutputDataTypes(DataType* o,int32_t no,DataType const* i,int32_t ni)const noexcept override{
        if(!o||!i||no!=1||ni!=INPUTS)return 1;o[0]=DataType::kBF16;return 0;}
    int32_t getOutputShapes(DimsExprs const* i,int32_t ni,DimsExprs const*,int32_t ns,
        DimsExprs* o,int32_t no,IExprBuilder& b)noexcept override{
        if(!i||!o||ni!=INPUTS||no!=1||ns||i[0].nbDims!=4)return 1;o[0]=i[0];
        o[0].d[3]=b.constant(DUAL_C);
        return 0;}
    bool supportsFormatCombination(int32_t p,DynamicPluginTensorDesc const* io,int32_t ni,int32_t no)noexcept override{
        return io&&ni==INPUTS&&no==1&&p>=0&&p<INPUTS+1&&io[p].desc.format==TensorFormat::kLINEAR&&
            io[p].desc.type==(p==INPUTS-1?DataType::kFLOAT:DataType::kBF16);}
    bool valid(PluginTensorDesc const* i,int ni,PluginTensorDesc const* o,int no)const{
        if(!i||!o||ni!=INPUTS||no!=1)return false;
        auto a=i[0].dims,b=o[0].dims;
        return a.nbDims==4&&a.d[0]==1&&a.d[1]==DUAL_H&&a.d[2]==DUAL_W&&a.d[3]==3*DUAL_C&&
          b.nbDims==4&&b.d[0]==1&&b.d[1]==DUAL_H&&b.d[2]==DUAL_W&&b.d[3]==DUAL_C&&
          i[1].dims.nbDims==3&&i[1].dims.d[0]==3&&i[1].dims.d[1]==3&&i[1].dims.d[2]==3*DUAL_C&&
          i[2].dims.nbDims==1&&i[2].dims.d[0]==HEADS&&i[0].type==DataType::kBF16&&
          i[1].type==DataType::kBF16&&i[2].type==DataType::kFLOAT&&o[0].type==DataType::kBF16;
    }
    int32_t configurePlugin(DynamicPluginTensorDesc const* i,int32_t ni,DynamicPluginTensorDesc const* o,int32_t no)noexcept override{
        if(!i||!o||ni!=INPUTS||no!=1)return 1;std::array<PluginTensorDesc,INPUTS>a;for(int j=0;j<INPUTS;++j)a[j]=i[j].desc;return valid(a.data(),ni,&o[0].desc,no)?0:1;}
    size_t getWorkspaceSize(DynamicPluginTensorDesc const*,int32_t,DynamicPluginTensorDesc const*,int32_t)const noexcept override{
        return size_t(CELLS)*HEADS*1088*4+size_t(REGIONS)*HEADS*1024*2;}
    char const* getTimingCacheID()noexcept override{return "dual_core_v1_" DUAL_KEY "_sm" STR(DUAL_SM);}
    char const* getMetadataString()noexcept override{return "{\"gram\":\"bf16_products_fp32_accumulator\",\"AV\":\"four_bf16_rounds\",\"blend\":\"fp32_canonical\"}";}
    int32_t onShapeChange(PluginTensorDesc const* i,int32_t ni,PluginTensorDesc const* o,int32_t no)noexcept override{return valid(i,ni,o,no)?0:1;}
    IPluginV3* attachToContext(IPluginResourceContext*)noexcept override{return clone();}
    PluginFieldCollection const* getFieldsToSerialize()noexcept override{return &fields;}
    int32_t enqueue(PluginTensorDesc const*,PluginTensorDesc const*,void const* const* in,void* const* out,void* workspace,cudaStream_t stream)noexcept override{
        if(!in||!out||!workspace)return 1;
        CUcontext current{};auto e=cuCtxGetCurrent(&current);if(e!=CUDA_SUCCESS||!current)return 1;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if(context&&context!=current)return 1;context=current;
            for(size_t i=0;i<COUNT;++i)if(!modules[i]){
                e=cuModuleLoadData(&modules[i],blobs[i].data);if(e!=CUDA_SUCCESS)return e;
                e=cuModuleGetFunction(&functions[i],modules[i],blobs[i].name);
                if(e!=CUDA_SUCCESS){cuModuleUnload(modules[i]);modules[i]=nullptr;return e;}
                if(blobs[i].shared>49152){e=cuFuncSetAttribute(functions[i],CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,blobs[i].shared);if(e!=CUDA_SUCCESS)return e;}
            }
        }
        // Installed Triton ABI: three tensor pointers and two null scratch pointers.
        void* matrices=static_cast<char*>(workspace)+size_t(CELLS)*HEADS*1088*4;
        std::array<void const*,5> d{in[0],in[1],workspace,nullptr,nullptr};
        std::array<void*,5> args{};for(int i=0;i<5;++i)args[i]=&d[i];
        e=cuLaunchKernel(functions[0],CELLS,HEADS,1,32*blobs[0].warps,1,1,blobs[0].shared,reinterpret_cast<CUstream>(stream),args.data(),nullptr);
        if(e!=CUDA_SUCCESS)return e;
        std::array<void const*,5> g{workspace,in[2],matrices,nullptr,nullptr};for(int i=0;i<5;++i)args[i]=&g[i];
        e=cuLaunchKernel(functions[1],REGIONS,HEADS,1,32*blobs[1].warps,1,1,blobs[1].shared,reinterpret_cast<CUstream>(stream),args.data(),nullptr);
        if(e!=CUDA_SUCCESS)return e;
        std::array<void const*,6> a{in[0],in[1],matrices,out[0],nullptr,nullptr};
        std::array<void*,6> applyArgs{};for(int i=0;i<6;++i)applyArgs[i]=&a[i];
        return cuLaunchKernel(functions[2],CELLS,HEADS,256/APPLY_BT,32*blobs[2].warps,1,1,blobs[2].shared,reinterpret_cast<CUstream>(stream),applyArgs.data(),nullptr);
    }
};
class Creator final: public IPluginCreatorV3One {
    PluginFieldCollection fields{};
public:
    IPluginV3* createPlugin(char const*,PluginFieldCollection const* f,TensorRTPhase)noexcept override{
        if(!f||f->nbFields)return nullptr;return new(std::nothrow)Plugin;}
    PluginFieldCollection const* getFieldNames()noexcept override{return &fields;}
    char const* getPluginName()const noexcept override{return NAME;}
    char const* getPluginVersion()const noexcept override{return VERSION;}
    char const* getPluginNamespace()const noexcept override{return "";}
};
}
extern "C" DUAL_EXPORT nvinfer1::IPluginCreatorInterface* const* getCreators(int32_t& n){static Creator c;static nvinfer1::IPluginCreatorInterface* const a[]={&c};n=1;return a;}
extern "C" DUAL_EXPORT void setLoggerFinder(nvinfer1::ILoggerFinder*){}
