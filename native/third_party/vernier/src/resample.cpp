#include "resample.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <iomanip>
#include <numbers>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace native {
int Kernel::radius() const {
    switch(family) {
    case Family::Bilinear: return 1;
    case Family::Bicubic: case Family::Spline16: return 2;
    case Family::Spline36: return 3;
    case Family::Spline64: return 4;
    case Family::Lanczos: return taps;
    }
    throw std::invalid_argument("Unknown kernel");
}
std::string Kernel::name() const {
    std::ostringstream s;
    switch(family) {
    case Family::Bilinear: return "Bilinear";
    case Family::Bicubic: s << "Bicubic B=" << std::setprecision(4) << b << " C=" << c; break;
    case Family::Lanczos: s << "Lanczos " << taps; break;
    case Family::Spline16: return "Spline16";
    case Family::Spline36: return "Spline36";
    case Family::Spline64: return "Spline64";
    }
    return s.str();
}
double Kernel::value(double x) const {
    x = std::abs(x);
    switch(family) {
    case Family::Bilinear: return std::max(0.0, 1-x);
    case Family::Bicubic:
        if(x < 1) return ((12-9*b-6*c)*x*x*x + (-18+12*b+6*c)*x*x + 6-2*b)/6;
        if(x < 2) return ((-b-6*c)*x*x*x + (6*b+30*c)*x*x + (-12*b-48*c)*x + 8*b+24*c)/6;
        return 0;
    case Family::Lanczos:
        if(x < 1e-12) return 1;
        if(x >= taps) return 0;
        { const double p = std::numbers::pi*x; return std::sin(p)*std::sin(p/taps)/(p*p/taps); }
    case Family::Spline16:
        if(x < 1) return ((x-9.0/5)*x-1.0/5)*x+1;
        if(x < 2) { x-=1; return ((-x/3+4.0/5)*x-7.0/15)*x; }
        return 0;
    case Family::Spline36:
        if(x < 1) return ((13.0/11*x-453.0/209)*x-3.0/209)*x+1;
        if(x < 2) { x-=1; return ((-6.0/11*x+270.0/209)*x-156.0/209)*x; }
        if(x < 3) { x-=2; return ((x/11-45.0/209)*x+26.0/209)*x; }
        return 0;
    case Family::Spline64:
        if(x < 1) return ((49.0/41*x-6387.0/2911)*x-3.0/2911)*x+1;
        if(x < 2) { x-=1; return ((-24.0/41*x+4032.0/2911)*x-2328.0/2911)*x; }
        if(x < 3) { x-=2; return ((6.0/41*x-1008.0/2911)*x+582.0/2911)*x; }
        if(x < 4) { x-=3; return ((-x/41+168.0/2911)*x-97.0/2911)*x; }
        return 0;
    }
    return 0;
}
std::vector<Kernel> commonKernels() {
    return {{Family::Bilinear}, {Family::Bicubic,0,.5}, {Family::Bicubic,1.0/3,1.0/3},
            {Family::Bicubic,0,.75}, {Family::Bicubic,1,0}, {Family::Bicubic,.5,0},
            {Family::Bicubic,0,1}, {Family::Lanczos,0,0,2}, {Family::Lanczos,0,0,3},
            {Family::Lanczos,0,0,4}, {Family::Spline16}, {Family::Spline36}, {Family::Spline64}};
}
Resampler::Resampler(int n, int m, Kernel k, double extent, double shift, int holdout)
    : n_(n), m_(m), band_(2*k.radius()-1), holdout_(holdout) {
    if(n<2 || m<n || m>32768 || k.radius()<1 || k.radius()>6 || holdout<0 || holdout==1)
        throw std::invalid_argument("Invalid resampling dimensions, kernel or holdout");
    if(extent==0) extent=n;
    if(!std::isfinite(extent) || !std::isfinite(shift) || extent<=0 || std::abs(extent-n)>2 || std::abs(shift)>2)
        throw std::invalid_argument("Invalid fractional grid");
    band_=std::min(band_,n-1);
    rows_.resize(m);
    lower_.assign(size_t(n)*(band_+1),0);
    for(int j=0;j<m;++j) {
        const double pos=(j+.5)*extent/m-.5+(n-extent)*.5+shift;
        const int first=int(std::floor(pos))-k.radius()+1;
        const int last=first+2*k.radius()-1;
        Row& r=rows_[j];
        r.start=std::clamp(first,0,n-1);
        r.count=std::clamp(last,0,n-1)-r.start+1;
        double sum=0;
        for(int i=first;i<=last;++i) {
            const double w=k.value(pos-i);
            r.weight[std::clamp(i,0,n-1)-r.start]+=w;
            sum+=w;
        }
        if(std::abs(sum)<1e-12) throw std::runtime_error("Degenerate kernel normalization");
        for(int a=0;a<r.count;++a) r.weight[a]/=sum;
        if(holdout_ && j%holdout_==holdout_/2) continue;
        for(int a=0;a<r.count;++a)
            for(int b=0;b<=a;++b)
                l(r.start+a,r.start+b)+=r.weight[a]*r.weight[b];
    }
    double smallest=1e300, largest=0;
    for(int i=0;i<n;++i) {
        for(int j=std::max(0,i-band_);j<=i;++j) {
            double v=l(i,j);
            for(int p=std::max(0,i-band_);p<j;++p) v-=l(i,p)*l(j,p);
            if(i==j) {
                if(!std::isfinite(v) || v<1e-12) throw SingularOperator("Unidentifiable or singular resampling operator");
                smallest=std::min(smallest,v); largest=std::max(largest,v);
                l(i,j)=std::sqrt(v);
            } else l(i,j)=v/l(j,j);
        }
    }
    pivotRatio_=smallest/largest;
}
void Resampler::solve(std::vector<double>& v) const {
    for(int i=0;i<n_;++i) {
        for(int j=std::max(0,i-band_);j<i;++j) v[i]-=l(i,j)*v[j];
        v[i]/=l(i,i);
    }
    for(int i=n_-1;i>=0;--i) {
        for(int j=i+1;j<=std::min(n_-1,i+band_);++j) v[i]-=l(j,i)*v[j];
        v[i]/=l(i,i);
    }
}
std::vector<double> Resampler::forward(const std::vector<double>& x) const {
    if(x.size()!=size_t(n_)) throw std::invalid_argument("Forward sample count mismatch");
    std::vector<double> y(m_,0);
    for(int j=0;j<m_;++j) {
        const auto& r=rows_[j];
        for(int a=0;a<r.count;++a) y[j]+=r.weight[a]*x[r.start+a];
    }
    return y;
}
std::vector<double> Resampler::inverse(const std::vector<double>& y) const {
    if(y.size()!=size_t(m_)) throw std::invalid_argument("Inverse sample count mismatch");
    std::vector<double> x(n_,0);
    for(int j=0;j<m_;++j) {
        if(holdout_ && j%holdout_==holdout_/2) continue;
        const auto& r=rows_[j];
        for(int a=0;a<r.count;++a) x[r.start+a]+=r.weight[a]*y[j];
    }
    solve(x);
    return x;
}
void Resampler::project(const std::vector<double>& y,std::vector<double>& x,std::vector<double>& fitted) const {
    x=inverse(y); fitted=forward(x);
}
double Resampler::predictionNoiseGain(int index) const {
    if(index<0 || index>=m_)throw std::invalid_argument("Invalid prediction index");
    const auto& row=rows_[index];std::vector<double> rhs(n_,0);
    for(int i=0;i<row.count;++i)rhs[row.start+i]=row.weight[i];
    solve(rhs);double gain=1;
    for(int i=0;i<row.count;++i)gain+=row.weight[i]*rhs[row.start+i];
    return gain;
}
std::vector<double> Resampler::projectBatch(const std::vector<double>& y,int signals) const {
    return forwardBatch(inverseBatch(y,signals),signals);
}
std::vector<double> Resampler::inverseBatch(const std::vector<double>& y,int signals) const {
    if(signals<1 || y.size()!=size_t(m_)*signals) throw std::invalid_argument("Batch sample count mismatch");
    std::vector<double> x(size_t(n_)*signals,0);
    for(int j=0;j<m_;++j) {
        if(holdout_ && j%holdout_==holdout_/2) continue;
        const auto& r=rows_[j];const double* src=y.data()+size_t(j)*signals;
        for(int a=0;a<r.count;++a) {
            double* dst=x.data()+size_t(r.start+a)*signals;const double w=r.weight[a];
            for(int s=0;s<signals;++s) dst[s]+=w*src[s];
        }
    }
    for(int i=0;i<n_;++i) {
        double* dst=x.data()+size_t(i)*signals;
        for(int j=std::max(0,i-band_);j<i;++j) {
            const double* src=x.data()+size_t(j)*signals;const double w=l(i,j);
            for(int s=0;s<signals;++s) dst[s]-=w*src[s];
        }
        const double d=l(i,i);for(int s=0;s<signals;++s) dst[s]/=d;
    }
    for(int i=n_-1;i>=0;--i) {
        double* dst=x.data()+size_t(i)*signals;
        for(int j=i+1;j<=std::min(n_-1,i+band_);++j) {
            const double* src=x.data()+size_t(j)*signals;const double w=l(j,i);
            for(int s=0;s<signals;++s) dst[s]-=w*src[s];
        }
        const double d=l(i,i);for(int s=0;s<signals;++s) dst[s]/=d;
    }
    return x;
}
std::vector<double> Resampler::forwardBatch(const std::vector<double>& x,int signals) const {
    if(signals<1 || x.size()!=size_t(n_)*signals)throw std::invalid_argument("Forward batch sample count mismatch");
    std::vector<double> fitted(size_t(m_)*signals,0);
    for(int j=0;j<m_;++j) {
        double* dst=fitted.data()+size_t(j)*signals;const auto& r=rows_[j];
        for(int a=0;a<r.count;++a) {
            const double* src=x.data()+size_t(r.start+a)*signals;const double w=r.weight[a];
            for(int s=0;s<signals;++s) dst[s]+=w*src[s];
        }
    }
    return fitted;
}
static void validateImage(const Image& im) {
    if(im.width<2 || im.height<2 || im.width>16384 || im.height>16384 || im.pixels.size()!=size_t(im.width)*im.height)
        throw std::invalid_argument("Invalid image dimensions");
    if(!std::all_of(im.pixels.begin(),im.pixels.end(),[](double v){return std::isfinite(v);}))
        throw std::invalid_argument("Image contains non-finite samples");
}
static Image transformAxis(const Image& im,const Resampler& op,bool horizontal,bool inverse,int threads) {
    const int input=horizontal?im.width:im.height,across=horizontal?im.height:im.width;
    const int output=inverse?op.nativeSize():op.observedSize();
    Image out{horizontal?output:im.width,horizontal?im.height:output,{}};out.pixels.resize(size_t(out.width)*out.height);
    std::atomic_int next=0;std::atomic_bool failed=false;std::exception_ptr error;std::mutex mutex;
    auto worker=[&]{try{for(;;){
        if(failed)return;const int start=next.fetch_add(32);if(start>=across)return;const int count=std::min(32,across-start);
        std::vector<double> batch(size_t(input)*count);
        for(int j=0;j<input;++j)for(int s=0;s<count;++s)batch[size_t(j)*count+s]=horizontal?im.at(j,start+s):im.at(start+s,j);
        auto values=inverse?op.inverseBatch(batch,count):op.forwardBatch(batch,count);
        for(int j=0;j<output;++j)for(int s=0;s<count;++s)out.pixels[horizontal?size_t(start+s)*output+j:size_t(j)*across+start+s]=values[size_t(j)*count+s];
    }}catch(...){std::lock_guard lock(mutex);if(!error)error=std::current_exception();failed=true;}};
    const int count=std::clamp(threads,1,std::min(64,(across+31)/32));
    if(count==1)worker();else{std::vector<std::thread> workers;for(int t=0;t<count;++t)workers.emplace_back(worker);for(auto& t:workers)t.join();}
    if(error)std::rethrow_exception(error);
    return out;
}
Image reconstruct(const Image& im,int w,int h,const Kernel& k,double ex,double ey,double sx,double sy,int threads) {
    validateImage(im);
    Resampler ax(im.width,w,k,ex,sx),ay(im.height,h,k,ey,sy);
    return transformAxis(transformAxis(im,ax,true,false,threads),ay,false,false,threads);
}
Image resize(const Image& im,int w,int h,const Kernel& k) { return reconstruct(im,w,h,k); }
Image descale(const Image& im,int w,int h,const Kernel& k,double ex,double ey,double sx,double sy,int threads) {
    validateImage(im);
    Resampler ax(w,im.width,k,ex,sx),ay(h,im.height,k,ey,sy);
    return transformAxis(transformAxis(im,ax,true,true,threads),ay,false,true,threads);
}
}
