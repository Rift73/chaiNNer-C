#include "analyze.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <limits>
#include <mutex>
#include <memory>
#include <optional>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace native {
int defaultWorkers(){return std::clamp(int(std::thread::hardware_concurrency()/2),1,16);}
Candidate exportGeometry(Candidate c) {
    if(!c.x.fractionalSupported){c.x.extent=c.x.size;c.x.shift=0;}
    if(!c.y.fractionalSupported){c.y.extent=c.y.size;c.y.shift=0;}
    return c;
}
namespace {
void cancelled(std::atomic_bool* c) { if(c && c->load()) throw std::runtime_error("Analysis cancelled"); }
double median(std::vector<double> v) {
    if(v.empty()) return std::numeric_limits<double>::infinity();
    const auto n=v.size()/2;
    std::nth_element(v.begin(),v.begin()+n,v.end());
    double a=v[n];
    if(v.size()%2==0) a=(a+*std::max_element(v.begin(),v.begin()+n))*.5;
    return a;
}
struct Signals {
    int length=0,count=0;
    std::vector<double> values;
    std::vector<unsigned char> mask;
    std::shared_ptr<std::atomic_int> rejected=std::make_shared<std::atomic_int>(0);
};
Signals samples(const Image& im,bool horizontal,int count) {
    const int length=horizontal?im.width:im.height;
    const int across=horizontal?im.height:im.width;
    count=std::min(count,across-24);
    Signals out{length,count,std::vector<double>(size_t(length)*count),std::vector<unsigned char>(size_t(length)*count)};
    for(int s=0;s<count;++s) {
        const int p=12+int((s+.5)*(across-24)/count);
        for(int i=0;i<length;++i) out.values[size_t(i)*count+s]=horizontal?im.at(i,p):im.at(p,i);
    }
    std::vector<double> curvature;
    curvature.reserve(size_t(length)*count);
    for(int i=12;i<length-12;++i)for(int s=0;s<count;++s)curvature.push_back(std::abs(out.values[size_t(i-1)*count+s]-2*out.values[size_t(i)*count+s]+out.values[size_t(i+1)*count+s]));
    const double madThreshold=3*median(curvature)/.6744897501960817;
    auto q=curvature.begin()+curvature.size()*9/10;
    std::nth_element(curvature.begin(),q,curvature.end());
    // Dense texture contaminates a noise-only MAD estimate. Retain its strongest
    // decile; the independent fold/null checks still decide whether it is scaled.
    const double threshold=std::max(1e-4,std::min(madThreshold,*q));
    for(int i=12;i<length-12;++i)for(int s=0;s<count;++s)if(std::abs(out.values[size_t(i-1)*count+s]-2*out.values[size_t(i)*count+s]+out.values[size_t(i+1)*count+s])>threshold)
        for(int d=-2;d<=2;++d)out.mask[size_t(i+d)*count+s]=1;
    return out;
}
Signals lineSplit(const Signals& data,int parity) {
    const int count=(data.count+1-parity)/2;
    Signals out{data.length,count,std::vector<double>(size_t(data.length)*count),std::vector<unsigned char>(size_t(data.length)*count)};
    out.rejected=data.rejected;
    for(int i=0;i<data.length;++i)for(int s=0;s<count;++s){out.values[size_t(i)*count+s]=data.values[size_t(i)*data.count+2*s+parity];out.mask[size_t(i)*count+s]=data.mask[size_t(i)*data.count+2*s+parity];}
    return out;
}
AxisFit fit(const Signals& data,int n,Kernel k,double extent=0,double shift=0,int holdout=0) {
    const int m=data.length,count=data.count;
    AxisFit f; f.size=n; f.extent=extent?extent:n; f.shift=shift; f.kernel=k;
    std::optional<Resampler> model;
    try{model.emplace(n,m,k,extent,shift,holdout);}catch(const SingularOperator&){++*data.rejected;f.identifiable=false;f.score=f.mse=std::numeric_limits<double>::infinity();return f;}
    const auto& op=*model;
    f.pivotRatio=op.minPivotRatio();
    double error=0,signalError=0,baselineError=0,scaledError=0; size_t total=0,informative=0;
    const auto fitted=op.projectBatch(data.values,count);
    // Fixed input curvature mask; pool edge evidence instead of flat-block noise.
    for(int i=12;i<m-12;++i) {
        if(holdout && i%holdout!=holdout/2)continue;
        const double noiseGain=holdout?op.predictionNoiseGain(i):1;
        for(int s=0;s<count;++s) {
            const size_t idx=size_t(i)*count+s;
            const double e=data.values[idx]-fitted[idx];error+=e*e;++total;
            if(data.mask[idx]) {
                signalError+=e*e;++informative;
                scaledError+=e*e/noiseGain;
                if(holdout) {
                    const double prediction=-(data.values[idx-2*count]+data.values[idx+2*count])/6+2*(data.values[idx-count]+data.values[idx+count])/3;
                    const double residual=data.values[idx]-prediction;baselineError+=residual*residual/(1+1.0/18+8.0/9);
                }
            }
        }
    }
    f.informative=int(informative);
    f.mse=total?error/total:0;
    const double remaining=1-double(n)/m;
    f.score=informative<16?std::numeric_limits<double>::infinity():signalError/informative/(holdout?1:remaining);
    if(holdout) f.predictive=baselineError>1e-20?scaledError/baselineError:-1;
    return f;
}
AxisFit refineFit(const Signals& data,AxisFit best,std::atomic_bool* cancel) {
    // A joint coarse grid prevents phase from masquerading as fractional extent.
    const int centre=best.size;
    const double broadProminence=best.prominence;
    for(int e=-2;e<=2;++e) for(int p=-2;p<=2;++p) {
        cancelled(cancel);
        const double extent=centre+e*.25, shift=p*.125;
        auto f=fit(data,centre,best.kernel,extent,shift);
        if(f.score<best.score) best=f;
    }
    for(double step:{.05,.01,.002}) for(int iteration=0;iteration<6;++iteration) {
        const AxisFit base=best;
        for(int e=-2;e<=2;++e) for(int p=-2;p<=2;++p) {
            cancelled(cancel);
            const double extent=base.extent+e*step,shift=base.shift+p*step;
            if(std::abs(extent-centre)>1.4 || std::abs(shift)>.5) continue;
            auto f=fit(data,centre,base.kernel,extent,shift);
            if(f.score<best.score) best=f;
        }
        if(best.extent==base.extent && best.shift==base.shift) break;
    }
    best.prominence=broadProminence;
    return best;
}
const FoldPoint& nearFold(const std::vector<FoldPoint>& points,double extent) {
    return *std::min_element(points.begin(),points.end(),[&](const auto& a,const auto& b){return std::abs(a.extent-extent)<std::abs(b.extent-extent);});
}
bool foldEvidence(const FoldPoint& f) {
    // Conservative diagnostic thresholds, not a statistical significance test.
    return f.excess>.06 && f.prominence>6 && f.participation>.5 && f.bins>=24;
}
std::vector<AxisFit> shortlist(const std::vector<AxisFit>& all,size_t offset,int count,const std::vector<FoldPoint>& folds) {
    std::vector<FoldPoint> peaks;
    for(size_t i=0;i<folds.size();++i) if((i==0 || folds[i].excess>=folds[i-1].excess) && (i+1==folds.size() || folds[i].excess>=folds[i+1].excess))peaks.push_back(folds[i]);
    std::sort(peaks.begin(),peaks.end(),[](const auto& a,const auto& b){return a.excess>b.excess;});
    std::vector<AxisFit> out;
    auto add=[&](AxisFit f){if(std::none_of(out.begin(),out.end(),[&](const auto& a){return a.size==f.size;}))out.push_back(f);};
    std::vector<double> chosen;
    for(const auto& peak:peaks) {
        if(std::any_of(chosen.begin(),chosen.end(),[&](double x){return std::abs(x-peak.extent)<2;}))continue;
        chosen.push_back(peak.extent);
        auto best=all[offset];best.score=std::numeric_limits<double>::infinity();
        for(int i=0;i<count;++i)if(std::abs(all[offset+i].extent-peak.extent)<=1 && all[offset+i].score<best.score)best=all[offset+i];
        if(std::isfinite(best.score))add(best);if(chosen.size()==5)break;
    }
    // Retain spatial notches as a fallback when the spectral assumptions fail.
    std::vector<AxisFit> notches;
    for(int i=0;i<count;++i)if((i==0 || all[offset+i].score<=all[offset+i-1].score) && (i+1==count || all[offset+i].score<=all[offset+i+1].score))notches.push_back(all[offset+i]);
    std::sort(notches.begin(),notches.end(),[](const auto& a,const auto& b){return a.prominence>b.prominence;});
    for(size_t i=0;i<std::min(size_t(3),notches.size());++i)add(notches[i]);
    if(out.empty())out.push_back(all[offset]);
    const double maxExcess=std::max_element(folds.begin(),folds.end(),[](const auto& a,const auto& b){return a.excess<b.excess;})->excess;
    auto supported=[&](const AxisFit& a){const auto& f=nearFold(folds,a.extent);return foldEvidence(f) && f.excess>=.5*maxExcess;};
    std::stable_sort(out.begin(),out.end(),[&](const auto& a,const auto& b){if(supported(a)!=supported(b))return supported(a);return a.score<b.score;});
    return out;
}
Candidate atGeometry(const Signals& xData,const Signals& yData,const Candidate& geometry,Kernel kernel) {
    auto x=fit(xData,geometry.x.size,kernel,geometry.x.extent,geometry.x.shift);
    auto y=fit(yData,geometry.y.size,kernel,geometry.y.extent,geometry.y.shift);
    x.prominence=geometry.x.prominence;y.prominence=geometry.y.prominence;
    return {x,y,std::sqrt(x.score*y.score)};
}
Candidate refineCubic(const Signals& xData,const Signals& yData,Candidate best,std::atomic_bool* cancel) {
    for(int b=0;b<=4;++b)for(int c=0;c<=4;++c){cancelled(cancel);auto f=atGeometry(xData,yData,best,{Family::Bicubic,b*.25,c*.25});if(f.score<best.score)best=f;}
    for(double step:{.05,.01,.002})for(int iteration=0;iteration<8;++iteration) {
        const auto base=best;
        for(int b=-1;b<=1;++b)for(int c=-1;c<=1;++c){
            cancelled(cancel);double bb=base.x.kernel.b+b*step,cc=base.x.kernel.c+c*step;
            if(bb<0 || bb>1 || cc<0 || cc>1)continue;
            auto f=atGeometry(xData,yData,base,{Family::Bicubic,bb,cc});if(f.score<best.score)best=f;
        }
        if(best.x.kernel.b==base.x.kernel.b && best.x.kernel.c==base.x.kernel.c)break;
    }
    return best;
}
double fullCheck(const Image& im,Candidate& c,int threads) {
    auto latent=descale(im,c.x.size,c.y.size,c.x.kernel,c.x.extent,c.y.extent,c.x.shift,c.y.shift,threads);
    auto fitted=reconstruct(latent,im.width,im.height,c.x.kernel,c.x.extent,c.y.extent,c.x.shift,c.y.shift,threads);
    double err=0; size_t count=0;
    int supported=0, informative=0;
    for(int by=0;by<6;++by) for(int bx=0;bx<8;++bx) {
        double blockError=0,energy=0; int pixels=0;
        const int y0=12+(im.height-24)*by/6,y1=12+(im.height-24)*(by+1)/6;
        const int x0=12+(im.width-24)*bx/8,x1=12+(im.width-24)*(bx+1)/8;
        for(int y=y0;y<y1;++y) for(int x=x0;x<x1;++x) {
            const double e=im.at(x,y)-fitted.at(x,y);
            const double dx=im.at(x,y)-im.at(x-1,y),dy=im.at(x,y)-im.at(x,y-1);
            blockError+=e*e; energy+=dx*dx+dy*dy; ++pixels;
        }
        err+=blockError; count+=pixels;
        if(energy/pixels>2e-7) { ++informative; if(blockError/energy<.015) ++supported; }
    }
    c.rmse=std::sqrt(err/count);
    c.spatialSupport=informative?double(supported)/informative:0;
    return err/count;
}
}
std::vector<AxisFit> scanAxis(const Image& im,bool horizontal,int minimum,int maximum,
                             const Options& options,std::atomic_bool* cancel,Progress progress) {
    auto data=samples(im,horizontal,options.lines);
    if(data.count==0) throw std::invalid_argument("Image is too small to analyze");
    const int span=maximum-minimum+1;
    std::vector<AxisFit> fits(size_t(span)*options.kernels.size());
    std::atomic_size_t next=0,done=0;
    std::atomic_bool failed=false;
    std::exception_ptr error;
    std::mutex guard;
    auto worker=[&] {
        try {
            for(;;) {
                cancelled(cancel);
                if(failed.load()) break;
                const size_t idx=next.fetch_add(1);
                if(idx>=fits.size()) break;
                fits[idx]=fit(data,minimum+int(idx%span),options.kernels[idx/span]);
                const size_t d=done.fetch_add(1)+1;
                if(progress && d%128==0) { std::lock_guard lock(guard); progress(double(d)/fits.size(),horizontal?"Scanning horizontal grids":"Scanning vertical grids"); }
            }
        } catch(...) { std::lock_guard lock(guard); if(!error) error=std::current_exception(); failed=true; }
    };
    std::vector<std::thread> workers;
    for(int i=0;i<std::clamp(options.threads,1,64);++i) workers.emplace_back(worker);
    for(auto& t:workers) t.join();
    if(error) std::rethrow_exception(error);
    for(size_t ki=0;ki<options.kernels.size();++ki) for(int i=0;i<span;++i) {
        std::vector<double> neighbours;
        for(int delta:{-12,-8,-4,-3,3,4,8,12}) if(i+delta>=0 && i+delta<span) neighbours.push_back(fits[ki*span+i+delta].score);
        auto& f=fits[ki*span+i];
        f.prominence=median(neighbours)/std::max(f.score,1e-20);
    }
    return fits;
}
Result analyze(const Image& im,const Options& o,std::atomic_bool* cancel,Progress progress) {
    const auto start=std::chrono::steady_clock::now();
    if(im.width<48 || im.height<48 || im.width>8192 || im.height>8192 || im.pixels.size()!=size_t(im.width)*im.height)
        throw std::invalid_argument("Analysis requires a 48..8192 pixel image with valid storage");
    for(double p:im.pixels) if(!std::isfinite(p)) throw std::invalid_argument("Non-finite input sample");
    if(o.lines<4 || o.lines>512 || o.refinementLines<4 || o.refinementLines>512 || o.threads<1 || o.threads>64 || o.kernels.empty() || o.kernels.size()>64)
        throw std::invalid_argument("Invalid sample, thread or kernel count");
    int minX=o.minWidth?o.minWidth:int(std::floor(double(o.minHeight)*im.width/im.height));
    int maxX=o.maxWidth?o.maxWidth:int(std::ceil(double(o.maxHeight)*im.width/im.height));
    if(o.minHeight<16 || o.maxHeight>=im.height || o.minHeight>o.maxHeight || minX<16 || maxX>=im.width || minX>maxX)
        throw std::invalid_argument("Search bounds must be below the source size, ordered, and at least 16 pixels");
    Result out; out.width=im.width; out.height=im.height; out.options=o;
    if(progress) progress(0,"Preparing unchanged source samples");
    out.foldX=foldScan(im,true,minX,maxX,std::max(256,o.lines),cancel,o.threads);
    out.foldY=foldScan(im,false,o.minHeight,o.maxHeight,std::max(256,o.lines),cancel,o.threads);
    out.vertical=scanAxis(im,false,o.minHeight,o.maxHeight,o,cancel,[&](double p,const std::string& s){if(progress) progress(p*.30,s);});
    out.horizontal=scanAxis(im,true,minX,maxX,o,cancel,[&](double p,const std::string& s){if(progress) progress(.30+p*.45,s);});
    const int finalLines=o.refine?std::max(o.lines,o.refinementLines):o.lines;
    const auto xData=samples(im,true,finalLines),yData=samples(im,false,finalLines);
    const int nx=maxX-minX+1,ny=o.maxHeight-o.minHeight+1;
    for(size_t k=0;k<o.kernels.size();++k) {
        const auto xs=shortlist(out.horizontal,k*nx,nx,out.foldX),ys=shortlist(out.vertical,k*ny,ny,out.foldY);
        out.alternativesX.insert(out.alternativesX.end(),xs.begin(),xs.end());out.alternativesY.insert(out.alternativesY.end(),ys.begin(),ys.end());
        AxisFit x=xs.front(),y=ys.front();
        Candidate c{x,y,std::sqrt(x.score*y.score)};
        out.candidates.push_back(o.refine?atGeometry(xData,yData,c,x.kernel):c);
    }
    std::sort(out.candidates.begin(),out.candidates.end(),[](const auto& a,const auto& b){return a.score<b.score;});
    if(!std::isfinite(out.candidates.front().score)) {
        out.verdict="Inconclusive"; out.explanation="Too little edge detail to identify a scaling grid. Flat or nearly flat images fit many operators.";
        out.candidates.clear();
    } else {
        const auto xEven=lineSplit(xData,0),xOdd=lineSplit(xData,1),yEven=lineSplit(yData,0),yOdd=lineSplit(yData,1);
        if(o.refine) {
            if(progress)progress(.75,"Refining every preset on denser samples");
            std::atomic_size_t next=0;std::atomic_bool failed=false;std::exception_ptr error;std::mutex mutex;
            auto worker=[&]{try{for(;;){cancelled(cancel);if(failed)return;const size_t i=next.fetch_add(1);if(i>=out.candidates.size())return;auto& c=out.candidates[i];c.x=refineFit(xData,c.x,cancel);c.y=refineFit(yData,c.y,cancel);c.score=std::sqrt(c.x.score*c.y.score);}}catch(...){std::lock_guard lock(mutex);if(!error)error=std::current_exception();failed=true;}};
            std::vector<std::thread> workers;for(int i=0;i<std::min(o.threads,int(out.candidates.size()));++i)workers.emplace_back(worker);for(auto& t:workers)t.join();if(error)std::rethrow_exception(error);
            std::sort(out.candidates.begin(),out.candidates.end(),[](const auto& a,const auto& b){return a.score<b.score;});
        }
        if(o.refine && std::any_of(o.kernels.begin(),o.kernels.end(),[](const auto& k){return k.family==Family::Bicubic;})) {
            if(progress)progress(.87,"Fitting cubic coefficients on even lines; checking odd lines");
            auto geometry=atGeometry(xEven,yEven,out.candidates.front(),out.candidates.front().x.kernel);
            auto cubic=atGeometry(xEven,yEven,geometry,{Family::Bicubic,0,.5});
            for(int iteration=0;iteration<3;++iteration){
                cubic=refineCubic(xEven,yEven,cubic,cancel);
                if(cubic.score>=geometry.score)break;
                cubic.x=refineFit(xEven,cubic.x,cancel);cubic.y=refineFit(yEven,cubic.y,cancel);cubic.score=std::sqrt(cubic.x.score*cubic.y.score);
                geometry=cubic;
            }
            auto candidate=atGeometry(xData,yData,cubic,cubic.x.kernel);candidate.continuous=true;out.candidates.push_back(candidate);
        }
        std::sort(out.candidates.begin(),out.candidates.end(),[](const auto& a,const auto& b){return a.score<b.score;});
        // Shared geometry remains available as a secondary diagnostic; it must
        // not determine the ranking because the anchor favours its own kernel.
        const auto anchor=out.candidates.front();
        std::vector<Candidate> even,odd;
        for(auto& c:out.candidates){
            cancelled(cancel);
            c.sharedGeometryScore=atGeometry(xData,yData,anchor,c.x.kernel).score;
            even.push_back(atGeometry(xEven,yEven,c,c.x.kernel));odd.push_back(atGeometry(xOdd,yOdd,c,c.x.kernel));
            c.evenScore=even.back().score;c.oddScore=odd.back().score;
            auto validateGeometry=[&](AxisFit& f,const Signals& all,const Signals& a,const Signals& b){
                const auto integer=fit(all,f.size,f.kernel);f.refinementGain=1-f.score/std::max(integer.score,1e-30);
                const auto fractionalA=fit(a,f.size,f.kernel,f.extent,f.shift),fractionalB=fit(b,f.size,f.kernel,f.extent,f.shift);
                f.fractionalSupported=f.refinementGain>.01 && fractionalA.score<.99*fit(a,f.size,f.kernel).score && fractionalB.score<.99*fit(b,f.size,f.kernel).score;
            };
            validateGeometry(c.x,xData,xEven,xOdd);validateGeometry(c.y,yData,yEven,yOdd);
        }
        out.kernelConsistent=std::all_of(out.candidates.begin(),out.candidates.end(),[&](const auto& c){return c.x.score>=anchor.x.score && c.y.score>=anchor.y.score;});
        for(const auto* split:{&even,&odd})out.kernelConsistent&=std::all_of(split->begin(),split->end(),[&](const auto& c){return c.x.score>=split->front().x.score && c.y.score>=split->front().y.score;});
        const size_t checked=std::min(size_t(4),out.candidates.size());
        for(size_t i=0;i<checked;++i) {
            cancelled(cancel);
            if(progress) progress(.90+.09*i/checked,"Checking full-frame residuals and withheld pixels");
            auto& c=out.candidates[i];
            fullCheck(im,c,o.threads);
            if(double(c.x.size)/im.width<.93) c.x.predictive=fit(xData,c.x.size,c.x.kernel,c.x.extent,c.x.shift,17).predictive;
            if(double(c.y.size)/im.height<.93) c.y.predictive=fit(yData,c.y.size,c.y.kernel,c.y.extent,c.y.shift,17).predictive;
        }
        const auto& winner=out.candidates.front();
        const auto fx=nearFold(out.foldX,winner.x.extent),fy=nearFold(out.foldY,winner.y.extent);
        auto competing=[](const auto& folds,const auto& best){
            const double leading=nearFold(folds,best.extent).excess;
            for(size_t i=1;i+1<folds.size();++i){const auto& f=folds[i];if(std::abs(f.extent-best.extent)>3 && f.excess>=folds[i-1].excess && f.excess>=folds[i+1].excess && f.excess>.5*leading && foldEvidence(f))return true;}
            return false;
        };
        out.competingGrids=competing(out.foldX,winner.x) || competing(out.foldY,winner.y);
        for(size_t i=1;i<out.candidates.size();++i) {
            const auto& k=out.candidates[i].x.kernel;const auto& w=winner.x.kernel;
            if(k.family==Family::Bicubic && w.family==Family::Bicubic && std::hypot(k.b-w.b,k.c-w.c)<.08)continue;
            out.separation=(out.candidates[i].score-winner.score)/std::max(out.candidates[i].score,1e-20);break;
        }
        out.kernelStatus=out.separation>=.08 && out.kernelConsistent && !winner.continuous?"supported_model":"unresolved";
        if(out.kernelStatus=="unresolved") {
            // Among close preset fits, prefer the smallest support. This is an
            // explicit parsimony choice for consumption, not an identification.
            int radius=100;
            for(size_t i=0;i<out.candidates.size();++i){const auto& c=out.candidates[i];if(!c.continuous && c.x.size==winner.x.size && c.y.size==winner.y.size && c.score<=winner.score/0.92 && c.x.kernel.radius()<radius){out.recommendedCandidate=i;radius=c.x.kernel.radius();}}
        }
        if(winner.x.informative<12 || winner.y.informative<12 || winner.x.prominence<1.3 || winner.y.prominence<1.3 || winner.spatialSupport<.4 || !foldEvidence(fx) || !foldEvidence(fy)) {
            out.verdict="Inconclusive";
            out.explanation="The leading tested operator is not supported consistently by both axes and the full-frame residual. Native-sized, processed, mixed-resolution, or weak-detail material may behave this way.";
        } else if(out.competingGrids) {
            out.verdict="Ambiguous grid";out.explanation="More than one separated grid has spectral and spatial evidence. Inspect regions and independent scenes before selecting dimensions.";
        } else if(out.kernelStatus=="unresolved") {
            out.verdict="Ambiguous kernel";
            out.explanation="A grid is visible, but the kernel is unresolved. Close fits, disagreement between axes/line splits, or a continuous approximation prevent identification. Post-processing can mimic another kernel.";
        } else {
            out.verdict="Supported candidate";
            out.explanation="Independent frequency-fold peaks and spatial fits support both axes within the tested models. Verify on independent scenes; this is not proof of a unique production pipeline.";
        }
        if(winner.x.size==minX || winner.x.size==maxX || winner.y.size==o.minHeight || winner.y.size==o.maxHeight) {
            out.verdict="Search boundary";
            out.explanation="The leading candidate touches a search boundary. Widen the range before interpreting the result.";
        }
        out.gridStatus=out.verdict=="Supported candidate" || out.verdict=="Ambiguous kernel"?"supported":out.verdict=="Ambiguous grid"?"ambiguous":"inconclusive";
        if(out.gridStatus!="supported")out.kernelStatus="unresolved";
        auto& recommended=out.candidates[out.recommendedCandidate];if(recommended.rmse<0)fullCheck(im,recommended,o.threads);
    }
    out.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    out.rejectedSingularFits=xData.rejected->load()+yData.rejected->load();
    for(const auto* scan:{&out.horizontal,&out.vertical})for(const auto& f:*scan)if(!f.identifiable)++out.rejectedSingularFits;
    if(progress) progress(1,"Analysis complete");
    return out;
}
}
