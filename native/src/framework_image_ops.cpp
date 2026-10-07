/* Model image preparation, mask postprocessing and semantic segmentation.
 * Array arithmetic/layout/masking is local; inference remains in its native
 * framework. NumPy polynomial attribution accompanies framework_math_f32.h. */
#include "parallel.h"
#include "numeric.h"
#include "framework_math_f32.h"
#include <atomic>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <memory>
#include <new>

namespace {
struct Bounds { uintptr_t start, end; };
cn_status bounds(const void* p, size_t n, size_t alignment, Bounds& out) noexcept {
    auto a = reinterpret_cast<uintptr_t>(p);
    if (!p || a % alignment) return CN_INVALID_ARGUMENT;
    if (a > UINTPTR_MAX - n) return CN_SIZE_OVERFLOW;
    out = {a, a + n}; return CN_OK;
}
bool overlap(Bounds a, Bounds b) noexcept { return a.start < b.end && b.start < a.end; }
cn_status pair(const void* src, size_t input_bytes, void* dst, size_t output_bytes,
               size_t input_alignment, size_t output_alignment, bool inplace = false) noexcept {
    Bounds a{}, b{};
    cn_status status = bounds(src, input_bytes, input_alignment, a);
    if (status == CN_OK) status = bounds(dst, output_bytes, output_alignment, b);
    if (status != CN_OK) return status;
    return overlap(a,b) && !(inplace && src == dst && input_bytes == output_bytes)
        ? CN_INVALID_ARGUMENT : CN_OK;
}
int fp_events() noexcept {
    int e = std::fetestexcept(FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW | FE_INVALID);
    return ((e & FE_DIVBYZERO) ? 1 : 0) | ((e & FE_OVERFLOW) ? 2 : 0)
        | ((e & FE_UNDERFLOW) ? 4 : 0) | ((e & FE_INVALID) ? 8 : 0);
}
cn_status events_bounds(int* events, const void* src, size_t in_bytes,
                        const void* out, size_t out_bytes, size_t count) noexcept {
    Bounds a{}, b{}, e{};
    cn_status status = bounds(events, count * sizeof(int), alignof(int), e);
    if (status != CN_OK) return status;
    bounds(src,in_bytes,1,a); bounds(out,out_bytes,1,b);
    return overlap(e,a) || overlap(e,b) ? CN_INVALID_ARGUMENT : CN_OK;
}
size_t reflect(size_t p, size_t n) noexcept {
    if (n == 1) return 0;
    size_t period = 2 * (n - 1), q = p % period;
    return q < n ? q : period - q;
}
struct PadJob { const unsigned char* src; unsigned char* out; size_t h,w,c,oh,ow,item; bool fortran; };
void pad_rows(void* opaque, size_t begin, size_t end) noexcept {
    const auto& j = *static_cast<PadJob*>(opaque);
    for (size_t y = begin; y < end; ++y)
        for (size_t x = 0; x < j.ow; ++x)
            for (size_t c = 0; c < j.c; ++c) {
                size_t a = (reflect(y,j.h)*j.w+reflect(x,j.w))*j.c+c;
                size_t b = j.fortran ? y + j.oh*(x+j.ow*c) : (y*j.ow+x)*j.c+c;
                std::memcpy(j.out+b*j.item,j.src+a*j.item,j.item);
            }
}
struct PlanarJob { const uint8_t* src; float* out; size_t pixels,channels; };
void planar_range(void* opaque, size_t begin, size_t end) noexcept {
    const auto& j = *static_cast<PlanarJob*>(opaque);
    constexpr float scale = static_cast<float>(1.0 / 255.0);
    for (size_t i=begin;i<end;++i)
        for (size_t c=0;c<j.channels;++c)
            j.out[c*j.pixels+i]=static_cast<float>(j.src[i*j.channels+c])*scale;
}
struct AffineJob { const float* src; float* out; size_t channels,channel; float scalar; int divide; std::atomic<int> events{0}; };
void affine_range(void* opaque,size_t begin,size_t end) noexcept {
    auto& j=*static_cast<AffineJob*>(opaque);
    std::fenv_t environment;std::fegetenv(&environment);std::feclearexcept(FE_ALL_EXCEPT);
    for(size_t i=begin;i<end;++i) {
        size_t k=i*j.channels+j.channel;
        j.out[k]=j.divide ? j.src[k]/j.scalar : j.src[k]-j.scalar;
    }
    int flags=fp_events();std::fesetenv(&environment);j.events.fetch_or(flags,std::memory_order_relaxed);
}
struct MaskJob { const float* src; float* out; };
void threshold_range(void* opaque,size_t begin,size_t end) noexcept {
    const auto& j=*static_cast<MaskJob*>(opaque);
    for(size_t i=begin;i<end;++i) j.out[i]=j.src[i]<0.5f ? 0.f : 1.f;
}
struct TrimapJob { const float* mask; const size_t* fg; const size_t* bg; double* out; size_t h,w,size; float ft,bt; };
size_t area(const size_t* integral,size_t stride,size_t y0,size_t x0,size_t y1,size_t x1) noexcept {
    return integral[y1*stride+x1]-integral[y0*stride+x1]-integral[y1*stride+x0]+integral[y0*stride+x0];
}
void trimap_range(void* opaque,size_t begin,size_t end) noexcept {
    const auto& j=*static_cast<TrimapJob*>(opaque);
    for(size_t y=begin;y<end;++y) for(size_t x=0;x<j.w;++x) {
        bool fg=true,bg=true;
        if(j.size==0) {
            constexpr int dy[]={0,-1,0,1,0},dx[]={0,0,-1,0,1};
            for(size_t t=0;t<5;++t) {
                ptrdiff_t yy=static_cast<ptrdiff_t>(y)+dy[t],xx=static_cast<ptrdiff_t>(x)+dx[t];
                if(yy<0||xx<0||static_cast<size_t>(yy)>=j.h||static_cast<size_t>(xx)>=j.w) fg=false;
                else { float v=j.mask[static_cast<size_t>(yy)*j.w+static_cast<size_t>(xx)];fg=fg&&(v>j.ft);bg=bg&&(v<j.bt); }
            }
        } else {
            size_t before=j.size/2,after=j.size-before;
            size_t y0=y>before?y-before:0,x0=x>before?x-before:0;
            size_t y1=after>j.h-y?j.h:y+after,x1=after>j.w-x?j.w:x+after;
            size_t pixels=(y1-y0)*(x1-x0);
            fg=y>=before&&x>=before&&after<=j.h-y&&after<=j.w-x&&area(j.fg,j.w+1,y0,x0,y1,x1)==pixels;
            bg=area(j.bg,j.w+1,y0,x0,y1,x1)==pixels;
        }
        j.out[y*j.w+x]=bg ? 0.0 : fg ? 1.0 : 0.5;
    }
}
struct ClothJob {
    const float* src; float* tmp; float* exponential; float* logsum; uint8_t* out;
    size_t pixels,classes; int fused,phase; std::atomic<int> events{0};
};
void cloth_range(void* opaque,size_t begin,size_t end) noexcept {
    auto& j=*static_cast<ClothJob*>(opaque);
    std::fenv_t environment;std::fegetenv(&environment);std::feclearexcept(FE_ALL_EXCEPT);
    for(size_t i=begin;i<end;++i) {
        if(j.phase==0) {
            float maximum=j.src[i];
            for(size_t c=1;c<j.classes;++c) { float v=j.src[c*j.pixels+i];if(std::isnan(v)||(!std::isnan(maximum)&&v>maximum))maximum=v; }
            if(!std::isfinite(maximum))maximum=0;
            for(size_t c=0;c<j.classes;++c)j.tmp[c*j.pixels+i]=j.src[c*j.pixels+i]-maximum;
        } else if(j.phase==1) {
            j.exponential[i]=cn_framework_exp_f32(j.tmp[i],j.fused);
        } else if(j.phase==2) {
            float total=0;
            // scipy's np.sum(exp, axis=1) of one pixel: a fresh contiguous operand,
            // so NumPy 2.5.3 reduces it in one pairwise tree (block 0).
            if(j.pixels==1)total=cn_numpy_sum_f32(j.exponential,j.classes,1,0,0);
            else for(size_t c=0;c<j.classes;++c)total+=j.exponential[c*j.pixels+i];
            j.logsum[i]=cn_framework_log_f32(total,j.fused);
        } else {
            float best=j.tmp[i]-j.logsum[i];size_t index=0;
            for(size_t c=1;c<j.classes;++c) {
                float value=j.tmp[c*j.pixels+i]-j.logsum[i];
                if(!std::isnan(best)&&(std::isnan(value)||value>best)) {best=value;index=c;}
            }
            j.out[i]=static_cast<uint8_t>(index);
        }
    }
    int flags=fp_events();std::fesetenv(&environment);
    if(j.phase==2)flags&=~1; // scipy.log_softmax suppresses log(0) diagnostics.
    j.events.fetch_or(flags,std::memory_order_relaxed);
}
struct PaletteJob { const uint8_t* src;float* out;size_t pixels; };
void palette_range(void* opaque,size_t begin,size_t end) noexcept {
    const auto& j=*static_cast<PaletteJob*>(opaque);
    // Pillow 12.3's putpalette unpacks the four supplied colours into a palette that
    // ImagingPaletteNew filled with black, so every other index converts to 0.
    for(size_t i=begin;i<end;++i)for(size_t c=0;c<3;++c)
        j.out[c*j.pixels+i]=j.src[i]==c+1?1.f:0.f;
}
}

extern "C" CN_EXPORT cn_status cn_framework_pad(const void* src,void* dst,size_t h,size_t w,size_t c,size_t oh,size_t ow,size_t item,int fortran) noexcept {
    if(!h||!w||!c||oh<h||ow<w||(item!=2&&item!=4&&item!=8)||(fortran!=0&&fortran!=1))return CN_INVALID_ARGUMENT;
    if(oh>PTRDIFF_MAX/ow||oh*ow>PTRDIFF_MAX/c/item)return CN_SIZE_OVERFLOW;
    auto status=pair(src,h*w*c*item,dst,oh*ow*c*item,item,item);if(status!=CN_OK)return status;
    PadJob j{static_cast<const unsigned char*>(src),static_cast<unsigned char*>(dst),h,w,c,oh,ow,item,fortran!=0};
    return cn_parallel_for(oh,1,pad_rows,&j);
}
extern "C" CN_EXPORT cn_status cn_framework_ncnn(const uint8_t* src,float* dst,size_t pixels,size_t channels) noexcept {
    if(!pixels||(channels!=1&&channels!=3&&channels!=4))return CN_INVALID_ARGUMENT;
    if(pixels>SIZE_MAX/channels/sizeof(float))return CN_SIZE_OVERFLOW;
    auto status=pair(src,pixels*channels,dst,pixels*channels*sizeof(float),1,alignof(float));if(status!=CN_OK)return status;
    PlanarJob j{src,dst,pixels,channels};return cn_parallel_for(pixels,65536,planar_range,&j);
}
extern "C" CN_EXPORT cn_status cn_framework_affine(const float* src,float* dst,size_t pixels,size_t channels,size_t channel,float scalar,int divide,int* events) noexcept {
    if(!pixels||!channels||channel>=channels||(divide!=0&&divide!=1))return CN_INVALID_ARGUMENT;
    if(pixels>SIZE_MAX/channels/sizeof(float))return CN_SIZE_OVERFLOW;
    size_t bytes=pixels*channels*sizeof(float);
    auto status=pair(src,bytes,dst,bytes,alignof(float),alignof(float),true);if(status!=CN_OK)return status;
    status=events_bounds(events,src,bytes,dst,bytes,1);if(status!=CN_OK)return status;
    AffineJob j{src,dst,channels,channel,scalar,divide};status=cn_parallel_for(pixels,65536,affine_range,&j);
    *events=j.events.load(std::memory_order_relaxed);return status;
}
extern "C" CN_EXPORT cn_status cn_framework_mask_threshold(const float* src,float* dst,size_t n) noexcept {
    if(!n){return CN_INVALID_ARGUMENT;}if(n>SIZE_MAX/sizeof(float))return CN_SIZE_OVERFLOW;
    auto status=pair(src,n*sizeof(float),dst,n*sizeof(float),alignof(float),alignof(float));if(status!=CN_OK)return status;
    MaskJob j{src,dst};return cn_parallel_for(n,65536,threshold_range,&j);
}
extern "C" CN_EXPORT cn_status cn_framework_trimap(const float* src,double* dst,size_t h,size_t w,float foreground,float background,size_t size) noexcept {
    if(!h||!w)return CN_INVALID_ARGUMENT;
    if(h>PTRDIFF_MAX/w/sizeof(double)||h==SIZE_MAX||w==SIZE_MAX||(h+1)>SIZE_MAX/(w+1)/sizeof(size_t))return CN_SIZE_OVERFLOW;
    auto status=pair(src,h*w*sizeof(float),dst,h*w*sizeof(double),alignof(float),alignof(double));if(status!=CN_OK)return status;
    try {
        size_t n=(h+1)*(w+1);std::unique_ptr<size_t[]> fg,bg;
        if(size) {
            fg=std::make_unique<size_t[]>(n);bg=std::make_unique<size_t[]>(n);
            for(size_t y=0;y<h;++y){size_t f=0,b=0;for(size_t x=0;x<w;++x){float v=src[y*w+x];f+=v>foreground;b+=v<background;fg[(y+1)*(w+1)+x+1]=fg[y*(w+1)+x+1]+f;bg[(y+1)*(w+1)+x+1]=bg[y*(w+1)+x+1]+b;}}
        }
        TrimapJob j{src,fg.get(),bg.get(),dst,h,w,size,foreground,background};return cn_parallel_for(h,1,trimap_range,&j);
    }catch(const std::bad_alloc&){return CN_ALLOCATION_FAILED;}
}
extern "C" CN_EXPORT cn_status cn_framework_cloth_labels(const float* src,uint8_t* dst,size_t pixels,size_t classes,int fused,int* events) noexcept {
    if(!pixels||!classes||(fused!=0&&fused!=1))return CN_INVALID_ARGUMENT;
    if(pixels>SIZE_MAX/classes/sizeof(float))return CN_SIZE_OVERFLOW;
    size_t values=pixels*classes,bytes=values*sizeof(float);
    auto status=pair(src,bytes,dst,pixels,alignof(float),1);if(status!=CN_OK)return status;
    status=events_bounds(events,src,bytes,dst,pixels,4);if(status!=CN_OK)return status;
    try {
        auto temporary=std::make_unique<float[]>(values),exponential=std::make_unique<float[]>(values),logsum=std::make_unique<float[]>(pixels);
        ClothJob j{src,temporary.get(),exponential.get(),logsum.get(),dst,pixels,classes,fused,0};
        for(int phase=0;phase<4;++phase){j.phase=phase;j.events=0;status=cn_parallel_for(phase==1?values:pixels,16384,cloth_range,&j);events[phase]=j.events.load(std::memory_order_relaxed);if(status!=CN_OK)return status;}
        return CN_OK;
    }catch(const std::bad_alloc&){return CN_ALLOCATION_FAILED;}
}
extern "C" CN_EXPORT cn_status cn_framework_cloth_masks(const uint8_t* src,float* dst,size_t pixels) noexcept {
    if(!pixels){return CN_INVALID_ARGUMENT;}if(pixels>SIZE_MAX/3/sizeof(float))return CN_SIZE_OVERFLOW;
    auto status=pair(src,pixels,dst,pixels*3*sizeof(float),1,alignof(float));if(status!=CN_OK)return status;
    PaletteJob j{src,dst,pixels};return cn_parallel_for(pixels,65536,palette_range,&j);
}
