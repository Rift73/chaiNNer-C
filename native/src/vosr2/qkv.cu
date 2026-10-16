#include <cuda_bf16.h>
#include <cuda_runtime.h>

__device__ float rounded(float value) { return __bfloat162float(__float2bfloat16_rn(value)); }
__device__ float inverse(float a, float b) {
    float sum=a*a+b*b;
    for(int s=16;s;s/=2)sum+=__shfl_xor_sync(0xffffffffU,sum,s);
    return rsqrtf(sum*(1.0F/64.0F)+1e-6F);
}
__device__ float rope(float x, float cosine, float sine, int lane) {
    float other=__shfl_xor_sync(0xffffffffU,x,1);
    if((lane&1)==0)other=-other;
    return rounded(rounded(x*cosine)+rounded(other*sine));
}
__global__ void pack_kernel(__nv_bfloat16 const* a,__nv_bfloat16 const* b,__nv_bfloat16 const* c,
    __nv_bfloat16 const* d,__nv_bfloat16 const* e,__nv_bfloat16* oq,__nv_bfloat16* ok,__nv_bfloat16* ov,int nq,int nk) {
    int const lane=static_cast<int>(threadIdx.x)&31;
    int const warp=static_cast<int>(blockIdx.x)*8+static_cast<int>(threadIdx.x)/32;
    int const token=warp/24;
    int const head=warp%24;
    if(token>=nq)return;
    if(token<nq){
        int const base=(token*4608)+head*64;
        float q0=__bfloat162float(a[base+lane]),q1=__bfloat162float(a[base+lane+32]);
        float inv=inverse(q0,q1);
        auto qw=b;
        q0=rounded(rounded(q0*inv)*__bfloat162float(qw[lane]));
        q1=rounded(rounded(q1*inv)*__bfloat162float(qw[lane+32]));
        {
            q0=rope(q0,__bfloat162float(d[token*64+lane]),__bfloat162float(e[token*64+lane]),lane);
            q1=rope(q1,__bfloat162float(d[token*64+lane+32]),__bfloat162float(e[token*64+lane+32]),lane);
        }
        int const target=(head*nq+token)*64+lane;
        oq[target]=__float2bfloat16_rn(q0);oq[target+32]=__float2bfloat16_rn(q1);
    }
    if(token<nk){
        int const base=(token*4608+1536)+head*64;
        auto key=a;auto kw=c;
        float k0=__bfloat162float(key[base+lane]),k1=__bfloat162float(key[base+lane+32]);
        float inv=inverse(k0,k1);
        k0=rounded(rounded(k0*inv)*__bfloat162float(kw[lane]));
        k1=rounded(rounded(k1*inv)*__bfloat162float(kw[lane+32]));
        {
            k0=rope(k0,__bfloat162float(d[token*64+lane]),__bfloat162float(e[token*64+lane]),lane);
            k1=rope(k1,__bfloat162float(d[token*64+lane+32]),__bfloat162float(e[token*64+lane+32]),lane);
        }
        int const target=(head*nk+token)*64+lane;
        ok[target]=__float2bfloat16_rn(k0);ok[target+32]=__float2bfloat16_rn(k1);
        int const vb=(token*4608+3072)+head*64+lane;
        auto value=a;
        ov[target]=value[vb];ov[target+32]=value[vb+32];
    }
}
extern "C" int launchPack(void const* const* input,void* const* output,int nq,cudaStream_t stream) {
    auto a=static_cast<__nv_bfloat16 const*>(input[0]);auto b=static_cast<__nv_bfloat16 const*>(input[1]);
    auto c=static_cast<__nv_bfloat16 const*>(input[2]);auto d=static_cast<__nv_bfloat16 const*>(input[3]);
    auto e=static_cast<__nv_bfloat16 const*>(input[4]);
    auto q=static_cast<__nv_bfloat16*>(output[0]);auto k=static_cast<__nv_bfloat16*>(output[1]);auto v=static_cast<__nv_bfloat16*>(output[2]);
    int blocks=(nq*24+7)/8;
    pack_kernel<<<blocks,256,0,stream>>>(a,b,c,d,e,q,k,v,nq,nq);
    return static_cast<int>(cudaPeekAtLastError());
}
