#include "spectrum.hpp"
#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <thread>
#include <exception>
#include <mutex>

namespace native {
using Complex=std::complex<double>;
static void fft(std::vector<Complex>& data,bool inverse=false) {
    const int n=int(data.size());
    for(int i=1,j=0;i<n;++i) {
        int bit=n>>1;
        for(;j&bit;bit>>=1) j^=bit;
        j^=bit;
        if(i<j) std::swap(data[i],data[j]);
    }
    for(int len=2;len<=n;len<<=1) {
        Complex root=std::polar(1.0,(inverse?2:-2)*std::numbers::pi/len);
        for(int i=0;i<n;i+=len) {
            Complex w=1;
            for(int j=0;j<len/2;++j) {
                Complex a=data[i+j],b=data[i+j+len/2]*w;
                data[i+j]=a+b; data[i+j+len/2]=a-b; w*=root;
            }
        }
    }
    if(inverse) for(auto& z:data) z/=n;
}
SpectrumPlan::SpectrumPlan(int length):length_(length),padded_(1) {
    if(length<2 || length>32768) throw std::invalid_argument("Invalid FFT length");
    while(padded_<2*length-1) padded_*=2;
    chirp_.resize(length); filter_.assign(padded_,0);
    for(int i=0;i<length;++i) {
        chirp_[i]=std::polar(1.0,-std::numbers::pi*double(i)*i/length);
        filter_[i]=std::conj(chirp_[i]);
        if(i) filter_[padded_-i]=filter_[i];
    }
    fft(filter_);
}
std::vector<Complex> SpectrumPlan::transform(const std::vector<double>& data) const {
    if(data.size()!=size_t(length_)) throw std::invalid_argument("FFT input length mismatch");
    std::vector<Complex> a(padded_);
    for(int i=0;i<length_;++i) a[i]=data[i]*chirp_[i];
    fft(a);
    for(int i=0;i<padded_;++i) a[i]*=filter_[i];
    fft(a,true); a.resize(length_);
    for(int i=0;i<length_;++i) a[i]*=chirp_[i];
    return a;
}
std::vector<FoldPoint> foldScan(const Image& im,bool horizontal,int minimum,int maximum,int count,std::atomic_bool* cancel,int threads) {
    const int m=horizontal?im.width:im.height,across=horizontal?im.height:im.width;
    count=std::min(count,across-24);
    SpectrumPlan plan(2*m);
    std::vector<double> real(size_t(m+1)*count),imag(size_t(m+1)*count);
    std::vector<double> energy(m+1,0);
    auto parallel=[&](int tasks,auto operation) {
        std::atomic_int next=0;std::atomic_bool failed=false;std::exception_ptr error;std::mutex mutex;
        auto worker=[&]{try{for(;;){if(failed)return;if(cancel && cancel->load())throw std::runtime_error("Analysis cancelled");int i=next.fetch_add(1);if(i>=tasks)return;operation(i);}}catch(...){std::lock_guard lock(mutex);if(!error)error=std::current_exception();failed=true;}};
        std::vector<std::thread> workers;for(int t=0;t<std::clamp(threads,1,64);++t)workers.emplace_back(worker);for(auto& t:workers)t.join();if(error)std::rethrow_exception(error);
    };
    parallel(count,[&](int s) {
        const int p=12+int((s+.5)*(across-24)/count);
        std::vector<double> line(2*m,0);
        double mean=0,weightSum=0;
        for(int i=0;i<m;++i) {line[i]=horizontal?im.at(i,p):im.at(p,i);const double w=.5-.5*std::cos(2*std::numbers::pi*i/(m-1));mean+=w*line[i];weightSum+=w;}
        mean/=weightSum;
        for(int i=0;i<m;++i) line[i]=(line[i]-mean)*(.5-.5*std::cos(2*std::numbers::pi*i/(m-1)));
        auto spectrum=plan.transform(line);
        for(int i=0;i<=m;++i) {real[size_t(i)*count+s]=spectrum[i].real();imag[size_t(i)*count+s]=spectrum[i].imag();}
    });
    for(int i=0;i<=m;++i)for(int s=0;s<count;++s){auto j=size_t(i)*count+s;energy[i]+=real[j]*real[j]+imag[j]*imag[j];}
    std::vector<FoldPoint> points(size_t(2*(maximum-minimum)+1));
    parallel(int(points.size()),[&](int index) {
        int f=2*minimum+index;
        double weighted=0,shuffled=0,weightSum=0,high=0,positive=0,squared=0; int bins=0;
        for(int b=f/2+5;b<=std::min(m-4,f-4);++b) {
            const int low=f-b;
            if(low<=0 || low>m) continue;
            const double a=energy[b],d=energy[low];
            if(a*d<=1e-30)continue;
            auto correlation=[&](int shift) {
                // Eight independent accumulators expose SIMD without fast-math
                // reassociation. The fixed reduction tree is thread independent.
                double rr[8]{},ii[8]{};
                const double* ar=real.data()+size_t(b)*count;const double* ai=imag.data()+size_t(b)*count;
                const double* br=real.data()+size_t(low)*count;const double* bi=imag.data()+size_t(low)*count;
                auto accumulate=[&](int begin,int end,int offset){
                    int s=begin;
                    for(;s+8<=end;s+=8)for(int lane=0;lane<8;++lane){int a=s+lane,t=a+offset;rr[lane]+=ar[a]*br[t]-ai[a]*bi[t];ii[lane]+=ar[a]*bi[t]+ai[a]*br[t];}
                    for(int lane=0;s<end;++s,++lane){int t=s+offset;rr[lane]+=ar[s]*br[t]-ai[s]*bi[t];ii[lane]+=ar[s]*bi[t]+ai[s]*br[t];}
                };
                accumulate(0,count-shift,shift);accumulate(count-shift,count,shift-count);
                double cr=0,ci=0;for(int lane=0;lane<8;++lane){cr+=rr[lane];ci+=ii[lane];}
                return std::clamp((cr*cr+ci*ci)/(a*d),0.0,1.0);
            };
            const double coherence=correlation(0);
            double floor=0;for(int p=1;p<=4;++p)floor+=correlation(p*(count/5))*.25;
            const double weight=std::min(a,d),e=weight*std::max(0.0,coherence-floor);
            weighted+=weight*coherence;shuffled+=weight*floor;weightSum+=weight;high+=a;positive+=e;squared+=e*e;++bins;
        }
        auto& point=points[index];point.extent=f*.5;point.bins=bins;point.highEnergy=high;point.sampleLines=count;
        if(weightSum>1e-10){point.coherence=weighted/weightSum;point.permuted=shuffled/weightSum;point.excess=point.coherence-point.permuted;point.participation=squared>0?positive*positive/(squared*bins):0;}
    });
    auto med=[](std::vector<double> v){if(v.empty())return 0.;size_t mid=v.size()/2;std::nth_element(v.begin(),v.begin()+mid,v.end());return v[mid];};
    for(size_t i=0;i<points.size();++i){std::vector<double> neighbours;for(int d=-40;d<=40;++d)if(std::abs(d)>=7 && int(i)+d>=0 && int(i)+d<int(points.size()))neighbours.push_back(points[int(i)+d].coherence);double centre=med(neighbours);for(auto& v:neighbours)v=std::abs(v-centre);double mad=med(neighbours);points[i].prominence=(points[i].coherence-centre)/std::max(1e-8,1.4826*mad);}
    return points;
}
}
