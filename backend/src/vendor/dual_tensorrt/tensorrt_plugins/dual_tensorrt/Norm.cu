#include "Export.h"
#define STR1(x) #x
#define STR(x) STR1(x)
// Width-specialized centered NHWC normalization. DUAL export selects residual=0.
// Two centered FP32 passes, no E[x^2]-E[x]^2, no fast math or FP16.
#include <NvInfer.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <new>

namespace {
constexpr char NAME[] = "DualNorm_" DUAL_KEY "_TRT";
__device__ float warp_sum(float value) {
    for (int d=16; d>0; d/=2) value += __shfl_down_sync(0xffffffffU,value,d);
    return __shfl_sync(0xffffffffU,value,0);
}
template<bool Residual>
__global__ void normalize(__nv_bfloat16 const* x, float const* gamma, float const* beta,
    __nv_bfloat16 const* update, float const* prior_sigma, float const* scale,
    __nv_bfloat16* y, float* sigma, __nv_bfloat16* sum_out, int pixels, float eps) {
    int pixel=(blockIdx.x*blockDim.x+threadIdx.x)/32, lane=threadIdx.x%32;
    if(pixel>=pixels) return;
    float values[DUAL_C/32], mean=0.0F;
    #pragma unroll
    for(int k=0;k<DUAL_C/32;++k) {
        int channel=lane+k*32;
        int64_t index=static_cast<int64_t>(pixel)*DUAL_C+channel;
        float v=__bfloat162float(x[index]);
        if constexpr(Residual) {
            float delta=__fmul_rn(__fmul_rn(scale[channel],prior_sigma[pixel]),__bfloat162float(update[index]));
            __nv_bfloat16 rounded=__float2bfloat16_rn(__fadd_rn(v,__bfloat162float(__float2bfloat16_rn(delta))));
            sum_out[index]=rounded;
            v=__bfloat162float(rounded);
        }
        values[k]=v; mean+=v;
    }
    mean=warp_sum(mean)/float(DUAL_C);
    float variance=0.0F;
    #pragma unroll
    for(int k=0;k<DUAL_C/32;++k) {values[k]-=mean; variance+=values[k]*values[k];}
    float v=warp_sum(variance)/float(DUAL_C)+eps;
    float inv=rsqrtf(v);
    if(lane==0) sigma[pixel]=sqrtf(v);
    #pragma unroll
    for(int k=0;k<DUAL_C/32;++k) {
        int channel=lane+k*32;
        float result=__fadd_rn(__fmul_rn(__fmul_rn(gamma[channel],values[k]),inv),beta[channel]);
        y[static_cast<int64_t>(pixel)*DUAL_C+channel]=__float2bfloat16_rn(result);
    }
}
bool valid(nvinfer1::PluginTensorDesc const* in,int ni,nvinfer1::PluginTensorDesc const* out,int no,bool residual) {
    using namespace nvinfer1;
    if(!in||!out||ni!=(residual?6:3)||no!=(residual?3:2)) return false;
    if(in[0].dims.nbDims!=4||in[0].dims.d[3]!=DUAL_C) return false;
    for(int j=1;j<=2;++j) if(in[j].dims.nbDims!=1||in[j].dims.d[0]!=DUAL_C) return false;
    if(out[0].dims.nbDims!=4||out[1].dims.nbDims!=4||out[1].dims.d[3]!=1) return false;
    if(residual && (in[3].dims.nbDims!=4||in[4].dims.nbDims!=4||in[4].dims.d[3]!=1||in[5].dims.nbDims!=1||in[5].dims.d[0]!=DUAL_C||out[2].dims.nbDims!=4)) return false;
    for(int j=0;j<4;++j) {
        int d=in[0].dims.d[j];
        if(d==0||d< -1||out[0].dims.d[j]!=d||(j<3&&out[1].dims.d[j]!=d)) return false;
        if(residual && (in[3].dims.d[j]!=d||out[2].dims.d[j]!=d||(j<3&&in[4].dims.d[j]!=d))) return false;
    }
    for(int j=0;j<ni;++j) if(in[j].format!=TensorFormat::kLINEAR||in[j].type!=((j==0||j==3)?DataType::kBF16:DataType::kFLOAT)) return false;
    for(int j=0;j<no;++j) if(out[j].format!=TensorFormat::kLINEAR||out[j].type!=(j==1?DataType::kFLOAT:DataType::kBF16)) return false;
    return true;
}
class Plugin final:public nvinfer1::IPluginV3,public nvinfer1::IPluginV3OneCore,
    public nvinfer1::IPluginV3OneBuild,public nvinfer1::IPluginV3OneRuntime {
public:
    Plugin(int residual,float eps):residual_(residual),eps_(eps){}
    nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType t) noexcept override {
        using namespace nvinfer1;
        switch(t){case PluginCapabilityType::kCORE:return static_cast<IPluginV3OneCore*>(this);
        case PluginCapabilityType::kBUILD:return static_cast<IPluginV3OneBuild*>(this);
        case PluginCapabilityType::kRUNTIME:return static_cast<IPluginV3OneRuntime*>(this);default:return nullptr;}
    }
    nvinfer1::IPluginV3* clone() noexcept override{return new(std::nothrow)Plugin(residual_,eps_);}
    char const* getPluginName() const noexcept override{return NAME;}
    char const* getPluginVersion() const noexcept override{return "1";}
    char const* getPluginNamespace() const noexcept override{return "";}
    int32_t getNbOutputs() const noexcept override{return residual_?3:2;}
    int32_t getOutputDataTypes(nvinfer1::DataType* out,int32_t no,nvinfer1::DataType const* in,int32_t ni) const noexcept override {
        if(!out||!in||no!=getNbOutputs()||ni!=(residual_?6:3))return 1;
        out[0]=nvinfer1::DataType::kBF16;out[1]=nvinfer1::DataType::kFLOAT;
        if(residual_)out[2]=nvinfer1::DataType::kBF16;return 0;
    }
    int32_t getOutputShapes(nvinfer1::DimsExprs const* in,int32_t ni,nvinfer1::DimsExprs const*,int32_t ns,nvinfer1::DimsExprs* out,int32_t no,nvinfer1::IExprBuilder& builder) noexcept override {
        if(!in||!out||ni!=(residual_?6:3)||no!=getNbOutputs()||ns||in[0].nbDims!=4)return 1;
        out[0]=out[1]=in[0];out[1].d[3]=builder.constant(1);if(residual_)out[2]=in[0];return 0;
    }
    bool supportsFormatCombination(int32_t p,nvinfer1::DynamicPluginTensorDesc const* io,int32_t ni,int32_t no) noexcept override {
        using namespace nvinfer1;
        if(!io||ni!=(residual_?6:3)||no!=getNbOutputs()||p<0||p>=ni+no)return false;
        bool fp32=p<ni?(p!=0&&p!=3):(p-ni==1);
        return io[p].desc.format==TensorFormat::kLINEAR&&io[p].desc.type==(fp32?DataType::kFLOAT:DataType::kBF16);
    }
    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in,int32_t ni,nvinfer1::DynamicPluginTensorDesc const* out,int32_t no) noexcept override {
        if(!in||!out||ni!=(residual_?6:3)||no!=getNbOutputs())return 1;
        std::array<nvinfer1::PluginTensorDesc,6>a{};std::array<nvinfer1::PluginTensorDesc,3>b{};
        for(int i=0;i<ni;++i)a[i]=in[i].desc;for(int i=0;i<no;++i)b[i]=out[i].desc;
        return valid(a.data(),ni,b.data(),no,residual_)?0:1;
    }
    size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const*,int32_t,nvinfer1::DynamicPluginTensorDesc const*,int32_t) const noexcept override{return 0;}
    char const* getTimingCacheID() noexcept override{return residual_?"dual_norm_rs_v1_" DUAL_KEY : "dual_norm_v1_" DUAL_KEY;}
    char const* getMetadataString() noexcept override{return "{\"moments\":\"two_pass_centered_FP32\",\"residual\":\"BF16_round_update_then_sum\"}";}
    int32_t getFormatCombinationLimit() noexcept override{return 1;}
    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in,int32_t ni,nvinfer1::PluginTensorDesc const* out,int32_t no) noexcept override{return valid(in,ni,out,no,residual_)?0:1;}
    int32_t enqueue(nvinfer1::PluginTensorDesc const* desc,nvinfer1::PluginTensorDesc const*,void const* const* in,void* const* out,void*,cudaStream_t stream) noexcept override {
        if(!desc||!in||!out)return 1;
        auto d=desc[0].dims;int64_t n=static_cast<int64_t>(d.d[0])*d.d[1]*d.d[2];
        if(d.d[0]<1||d.d[1]<1||d.d[2]<1||d.d[3]!=DUAL_C||n>2147483647LL)return 1;
        auto x=static_cast<__nv_bfloat16 const*>(in[0]);auto g=static_cast<float const*>(in[1]);auto b=static_cast<float const*>(in[2]);
        auto y=static_cast<__nv_bfloat16*>(out[0]);auto sigma=static_cast<float*>(out[1]);
        if(residual_)normalize<true><<<static_cast<unsigned>((n+7)/8),256,0,stream>>>(x,g,b,static_cast<__nv_bfloat16 const*>(in[3]),static_cast<float const*>(in[4]),static_cast<float const*>(in[5]),y,sigma,static_cast<__nv_bfloat16*>(out[2]),n,eps_);
        else normalize<false><<<static_cast<unsigned>((n+7)/8),256,0,stream>>>(x,g,b,nullptr,nullptr,nullptr,y,sigma,nullptr,n,eps_);
        return cudaGetLastError()==cudaSuccess?0:1;
    }
    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext*) noexcept override{return clone();}
    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override {
        using namespace nvinfer1;fields_={PluginField{"residual",&residual_,PluginFieldType::kINT32,1},PluginField{"epsilon",&eps_,PluginFieldType::kFLOAT32,1}};
        collection_={2,fields_.data()};return &collection_;
    }
private:
    int32_t residual_;float eps_;
    std::array<nvinfer1::PluginField,2>fields_{};nvinfer1::PluginFieldCollection collection_{};
};
class Creator final:public nvinfer1::IPluginCreatorV3One {
public:
    Creator(){using namespace nvinfer1;fields_={PluginField{"residual",nullptr,PluginFieldType::kINT32,1},PluginField{"epsilon",nullptr,PluginFieldType::kFLOAT32,1}};collection_={2,fields_.data()};}
    nvinfer1::IPluginV3* createPlugin(char const*,nvinfer1::PluginFieldCollection const* fields,nvinfer1::TensorRTPhase) noexcept override {
        int r=-1;float e=0;if(!fields)return nullptr;
        for(int i=0;i<fields->nbFields;++i){auto f=fields->fields[i];if(!f.name||!f.data||f.length!=1)continue;
            if(!std::strcmp(f.name,"residual")&&f.type==nvinfer1::PluginFieldType::kINT32)r=*static_cast<int const*>(f.data);
            if(!std::strcmp(f.name,"epsilon")&&f.type==nvinfer1::PluginFieldType::kFLOAT32)e=*static_cast<float const*>(f.data);}
        return (r==0||r==1)&&std::isfinite(e)&&e>0?new(std::nothrow)Plugin(r,e):nullptr;
    }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override{return &collection_;}
    char const* getPluginName() const noexcept override{return NAME;}
    char const* getPluginVersion() const noexcept override{return "1";}
    char const* getPluginNamespace() const noexcept override{return "";}
private:std::array<nvinfer1::PluginField,2>fields_{};nvinfer1::PluginFieldCollection collection_{};
};
}
extern "C" DUAL_EXPORT nvinfer1::IPluginCreatorInterface* const* getCreators(int32_t& n){static Creator creator;static nvinfer1::IPluginCreatorInterface* const creators[]={&creator};n=1;return creators;}
extern "C" DUAL_EXPORT void setLoggerFinder(nvinfer1::ILoggerFinder*){}
