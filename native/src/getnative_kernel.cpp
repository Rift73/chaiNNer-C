#include "getnative_kernel.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iterator>
#include <limits>
#include <locale>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace native {
namespace {
const double unmeasured=std::numeric_limits<double>::infinity();
constexpr int minimumFrames=7;
constexpr double binomialLevel=.01;      // one-sided P(X >= wins | frames, 1/2)
constexpr double scoreMargin=.005;       // median paired margin, log least-squares score
constexpr double separation=.08;         // Vernier's per-frame separation; also its 8 % parsimony band
// A fold peak on the axis taken for unscaled means it has a grid too, from this share of the scaled axis's excess...
constexpr double foldShare=.4;
constexpr double notchedFoldShare=.1;    // ...or from this share where the fit at that size notches the score curve.

double medianOf(std::vector<double> v) {
    if(v.empty()) return std::numeric_limits<double>::infinity();
    const auto n=v.size()/2;
    std::nth_element(v.begin(),v.begin()+std::ptrdiff_t(n),v.end());
    double a=v[n];
    if(v.size()%2==0) a=(a+*std::max_element(v.begin(),v.begin()+std::ptrdiff_t(n)))*.5;
    return a;
}

// Vernier's sampled lines, curvature mask and least-squares score (analyze.cpp),
// repeated here to score a preset at the leader's geometry.
struct Lines { int length=0, count=0; std::vector<double> values; std::vector<unsigned char> mask; };
Lines sampleLines(const Image& im,bool horizontal,int count) {
    const int length=horizontal?im.width:im.height, across=horizontal?im.height:im.width;
    count=std::min(count,across-24);
    Lines out{length,count,std::vector<double>(size_t(length)*size_t(count)),std::vector<unsigned char>(size_t(length)*size_t(count))};
    for(int s=0;s<count;++s) {
        const int p=12+int((s+.5)*(across-24)/count);
        for(int i=0;i<length;++i) out.values[size_t(i)*size_t(count)+size_t(s)]=horizontal?im.at(i,p):im.at(p,i);
    }
    auto curvature=[&](int i,int s) {
        const double* v=out.values.data()+size_t(s);
        return std::abs(v[size_t(i-1)*size_t(count)]-2*v[size_t(i)*size_t(count)]+v[size_t(i+1)*size_t(count)]);
    };
    std::vector<double> all;
    all.reserve(size_t(length)*size_t(count));
    for(int i=12;i<length-12;++i) for(int s=0;s<count;++s) all.push_back(curvature(i,s));
    const double madThreshold=3*medianOf(all)/.6744897501960817;
    auto q=all.begin()+std::ptrdiff_t(all.size()*9/10);
    std::nth_element(all.begin(),q,all.end());
    const double threshold=std::max(1e-4,std::min(madThreshold,*q));
    for(int i=12;i<length-12;++i) for(int s=0;s<count;++s) if(curvature(i,s)>threshold)
        for(int d=-2;d<=2;++d) out.mask[size_t(i+d)*size_t(count)+size_t(s)]=1;
    return out;
}
Lines splitLines(const Lines& data,int parity) {
    const int count=(data.count+1-parity)/2;
    Lines out{data.length,count,std::vector<double>(size_t(data.length)*size_t(count)),std::vector<unsigned char>(size_t(data.length)*size_t(count))};
    for(int i=0;i<data.length;++i) for(int s=0;s<count;++s) {
        const size_t from=size_t(i)*size_t(data.count)+size_t(2*s+parity), to=size_t(i)*size_t(count)+size_t(s);
        out.values[to]=data.values[from]; out.mask[to]=data.mask[from];
    }
    return out;
}
double lineScore(const Lines& data,int n,const Kernel& kernel,double extent,double shift) {
    std::optional<Resampler> op;
    try { op.emplace(n,data.length,kernel,extent,shift); } catch(const SingularOperator&) { return unmeasured; }
    const auto fitted=op->projectBatch(data.values,data.count);
    double error=0; size_t informative=0;
    for(int i=12;i<data.length-12;++i) for(int s=0;s<data.count;++s) {
        const size_t idx=size_t(i)*size_t(data.count)+size_t(s);
        if(data.mask[idx]) { const double e=data.values[idx]-fitted[idx]; error+=e*e; ++informative; }
    }
    return informative<16?unmeasured:error/double(informative)/(1-double(n)/data.length);
}

// p beats q on one frame only when all lines, both line halves and both axes agree.
bool favours(const KernelSplit& p,const KernelSplit& q) {
    return p.full<q.full && p.even<q.even && p.odd<q.odd && p.x<q.x && p.y<q.y;
}
// Cross-frame domination of p over q: enough frames, a one-sided sign test on the
// paired wins (non-agreeing frames count as ties) and a median paired margin on all lines.
bool dominates(const std::vector<const KernelEvidence*>& frames,size_t p,size_t q) {
    int wins=0;
    std::vector<double> margins;
    for(const auto* frame:frames) {
        const auto& a=frame->presets[p].logScore; const auto& b=frame->presets[q].logScore;
        const double margin=b.full-a.full;
        margins.push_back(std::isnan(margin)?0:margin);
        if(favours(a,b)) ++wins;
    }
    const int count=int(frames.size());
    double probability=std::pow(.5,count), tail=0;
    for(int i=0;i<=count;++i) { if(i>=wins) tail+=probability; probability=probability*(count-i)/(i+1); }
    return count>=minimumFrames && tail<=binomialLevel && medianOf(margins)>=scoreMargin;
}
bool dominatesSingle(const KernelEvidence& frame,size_t p,size_t q) {
    const auto& a=frame.presets[p].logScore; const auto& b=frame.presets[q].logScore;
    return favours(a,b) && 1-std::exp(a.full-b.full)>=separation;
}

void writeNumber(std::ostringstream& s,double v) { if(std::isfinite(v)) s<<v; else s<<"null"; }
void writeString(std::ostringstream& s,const std::string& text) {
    s<<'"';
    for(const char c:text) {
        if(c=='"' || c=='\\') s<<'\\'<<c;
        else if(static_cast<unsigned char>(c)<0x20) { const char* hex="0123456789abcdef"; s<<"\\u00"<<hex[(c>>4)&15]<<hex[c&15]; }
        else s<<c;
    }
    s<<'"';
}
void writeSplit(std::ostringstream& s,const KernelSplit& v) {
    s<<"{\"full\":"; writeNumber(s,v.full); s<<",\"even\":"; writeNumber(s,v.even); s<<",\"odd\":"; writeNumber(s,v.odd);
    s<<",\"x\":"; writeNumber(s,v.x); s<<",\"y\":"; writeNumber(s,v.y); s<<'}';
}
std::ostringstream jsonStream() { std::ostringstream s; s.imbue(std::locale::classic()); s.precision(17); return s; }

template<class Operation> void forEachParallel(size_t count,int threads,std::atomic_bool* cancel,const Operation& operation) {
    std::atomic_size_t next=0; std::atomic_bool failed=false; std::exception_ptr error; std::mutex mutex;
    auto worker=[&] { try { for(;;) {
        if(failed) return;
        if(cancel && cancel->load()) throw std::runtime_error("Analysis cancelled");
        const size_t index=next.fetch_add(1);
        if(index>=count) return;
        operation(index);
    } } catch(...) { std::lock_guard lock(mutex); if(!error) error=std::current_exception(); failed=true; } };
    std::vector<std::thread> workers;
    for(int t=0;t<std::clamp(threads,1,int(count));++t) workers.emplace_back(worker);
    for(auto& t:workers) t.join();
    if(error) std::rethrow_exception(error);
}

// Vernier's nearFold, foldEvidence and full-frame check (analyze.cpp), repeated here to judge one axis alone.
const FoldPoint& nearestFold(const std::vector<FoldPoint>& folds,double extent) {
    return *std::min_element(folds.begin(),folds.end(),[&](const auto& a,const auto& b){return std::abs(a.extent-extent)<std::abs(b.extent-extent);});
}
bool foldEvidence(const FoldPoint& f) { return f.excess>.06 && f.prominence>6 && f.participation>.5 && f.bins>=24; }
void fullFrameCheck(const Image& im,Candidate& c,int threads) {
    const auto latent=descale(im,c.x.size,c.y.size,c.x.kernel,c.x.extent,c.y.extent,c.x.shift,c.y.shift,threads);
    const auto fitted=reconstruct(latent,im.width,im.height,c.x.kernel,c.x.extent,c.y.extent,c.x.shift,c.y.shift,threads);
    double total=0; size_t count=0;
    int supported=0, informative=0;
    for(int by=0;by<6;++by) for(int bx=0;bx<8;++bx) {
        double blockError=0, energy=0; int pixels=0;
        const int y0=12+(im.height-24)*by/6, y1=12+(im.height-24)*(by+1)/6;
        const int x0=12+(im.width-24)*bx/8, x1=12+(im.width-24)*(bx+1)/8;
        for(int y=y0;y<y1;++y) for(int x=x0;x<x1;++x) {
            const double e=im.at(x,y)-fitted.at(x,y), dx=im.at(x,y)-im.at(x-1,y), dy=im.at(x,y)-im.at(x,y-1);
            blockError+=e*e; energy+=dx*dx+dy*dy; ++pixels;
        }
        total+=blockError; count+=size_t(pixels);
        if(energy/pixels>2e-7) { ++informative; if(blockError/energy<.015) ++supported; }
    }
    c.rmse=std::sqrt(total/double(count));
    c.spatialSupport=informative?double(supported)/informative:0;
}
// Vernier's bars on one axis's leading fit, but for the fold's participation, which Lanczos upscales miss on both axes alike.
bool gridVisible(const FoldPoint& fold,const AxisFit& fit) {
    return fold.excess>.06 && fold.prominence>6 && fold.bins>=24 && fit.informative>=12 && fit.prominence>=1.3;
}
bool anyGrid(const std::vector<FoldPoint>& folds,const std::vector<AxisFit>& scan,double excess) {
    for(size_t i=0;i<folds.size();++i) {
        const auto& f=folds[i];
        if((i>0 && f.excess<folds[i-1].excess) || (i+1<folds.size() && f.excess<folds[i+1].excess)) continue;
        if(f.prominence<=6 || f.bins<24 || f.excess<notchedFoldShare*excess) continue;
        if(f.excess>=foldShare*excess || std::any_of(scan.begin(),scan.end(),[&](const AxisFit& a){return std::abs(a.size-f.extent)<=1 && a.prominence>=1.3;})) return true;
    }
    return false;
}
// One axis's leading fit shows a grid, and the other axis shows none anywhere in its scan.
bool showsGridAlone(const std::vector<FoldPoint>& folds,const AxisFit& fit,const std::vector<FoldPoint>& otherFolds,const std::vector<AxisFit>& otherScan) {
    const auto& fold=nearestFold(folds,fit.extent);
    return gridVisible(fold,fit) && !anyGrid(otherFolds,otherScan,fold.excess);
}
}

void searchUnscaledAxis(const Image& im,Result& r,const Options& o,std::atomic_bool* cancel) {
    if(r.gridStatus!="inconclusive" || r.candidates.empty()) return;
    const bool width=showsGridAlone(r.foldX,r.candidates.front().x,r.foldY,r.vertical), height=showsGridAlone(r.foldY,r.candidates.front().y,r.foldX,r.horizontal);
    if(width==height) return;
    AxisFit Candidate::* const scaled=width?&Candidate::x:&Candidate::y;
    AxisFit Candidate::* const unscaled=width?&Candidate::y:&Candidate::x;
    const auto& folds=width?r.foldX:r.foldY;
    const Lines lines=sampleLines(im,width,o.refine?std::max(o.lines,o.refinementLines):o.lines);
    const std::array<Lines,2> halves{splitLines(lines,0),splitLines(lines,1)};
    auto candidates=r.candidates;
    forEachParallel(candidates.size(),o.threads,cancel,[&](size_t index) {
        auto& c=candidates[index];
        const AxisFit& f=c.*scaled;
        AxisFit same;
        same.size=width?im.height:im.width; same.extent=same.size; same.kernel=f.kernel; same.pivotRatio=1;
        c.*unscaled=same;
        c.score=f.score; c.rmse=-1; c.spatialSupport=-1; c.sharedGeometryScore=unmeasured;
        c.evenScore=lineScore(halves[0],f.size,f.kernel,f.extent,f.shift);
        c.oddScore=lineScore(halves[1],f.size,f.kernel,f.extent,f.shift);
    });
    std::stable_sort(candidates.begin(),candidates.end(),[](const auto& a,const auto& b){return a.score<b.score;});
    auto& winner=candidates.front();
    const AxisFit& fit=winner.*scaled;
    const auto& fold=nearestFold(folds,fit.extent);
    if(!gridVisible(fold,fit)) return;
    // What the core's search left out of the unscaled axis, up to where a fold still has its 24 bins, must show no grid
    // either. A quarter of the core's 256 lines: the excess needs no more, and they are most of this search's cost.
    const auto& otherFolds=width?r.foldY:r.foldX;
    const int searched=int(otherFolds.back().extent), reach=(width?im.height:im.width)-32;
    if(searched<reach) {
        auto rest=foldScan(im,!width,std::max(int(otherFolds.front().extent),searched-20),reach,64,cancel,o.threads);
        rest.erase(rest.begin(),std::find_if(rest.begin(),rest.end(),[&](const FoldPoint& f){return f.extent>searched;}));
        if(anyGrid(rest,{},fold.excess)) return;
    }
    fullFrameCheck(im,winner,o.threads);
    if(winner.spatialSupport<.4) return;
    bool consistent=true;
    for(const auto& c:candidates) consistent=consistent && c.evenScore>=winner.evenScore && c.oddScore>=winner.oddScore;
    double gap=0;
    for(size_t i=1;i<candidates.size();++i) {
        const auto& k=candidates[i].x.kernel; const auto& w=winner.x.kernel;
        if(k.family==Family::Bicubic && w.family==Family::Bicubic && std::hypot(k.b-w.b,k.c-w.c)<.08) continue;
        gap=(candidates[i].score-winner.score)/std::max(candidates[i].score,1e-20);
        break;
    }
    const bool resolved=gap>=separation && consistent && !winner.continuous;
    size_t recommended=0;
    if(!resolved) {
        int radius=100;
        for(size_t i=0;i<candidates.size();++i) {
            const auto& c=candidates[i];
            if(!c.continuous && (c.*scaled).size==fit.size && c.score<=winner.score/.92 && c.x.kernel.radius()<radius) { recommended=i; radius=c.x.kernel.radius(); }
        }
    }
    const std::string axes=std::string("One axis scaled: the ")+(width?"width":"height")+" has a grid; the "+(width?"height":"width")+" shows none and stays at the source's size. ";
    r.competingGrids=false;
    for(size_t i=1;i+1<folds.size();++i) {
        const auto& f=folds[i];
        if(std::abs(f.extent-fit.extent)>3 && f.excess>=folds[i-1].excess && f.excess>=folds[i+1].excess && f.excess>.5*fold.excess && foldEvidence(f)) r.competingGrids=true;
    }
    if(!foldEvidence(fold)) {
        r.verdict="Inconclusive";
        r.explanation=axes+"The grid's frequency-fold peak does not pass every check, so the leading tested operator is not fully supported.";
    } else if(r.competingGrids) {
        r.verdict="Ambiguous grid";
        r.explanation=axes+"More than one separated grid has spectral and spatial evidence. Inspect regions and independent scenes before selecting dimensions.";
    } else if(!resolved) {
        r.verdict="Ambiguous kernel";
        r.explanation=axes+"The kernel is unresolved. Close fits, disagreement between line splits, or a continuous approximation prevent identification. Post-processing can mimic another kernel.";
    } else {
        r.verdict="Supported candidate";
        r.explanation=axes+"Independent frequency-fold peaks and spatial fits support that grid within the tested models. Verify on independent scenes; this is not proof of a unique production pipeline.";
    }
    if(fit.size==int(folds.front().extent) || fit.size==int(folds.back().extent)) {
        r.verdict="Search boundary";
        r.explanation=axes+"The leading candidate touches a search boundary. Widen the range before interpreting the result.";
    }
    r.gridStatus=r.verdict=="Supported candidate" || r.verdict=="Ambiguous kernel"?"supported":r.verdict=="Ambiguous grid"?"ambiguous":"inconclusive";
    r.kernelStatus=resolved && r.gridStatus=="supported"?"supported_model":"unresolved";
    r.kernelConsistent=consistent;
    r.separation=gap;
    r.recommendedCandidate=recommended;
    r.candidates=std::move(candidates);
}

std::shared_ptr<KernelEvidence> measureKernelEvidence(const Image& im,const Result& r,const Options& o,std::atomic_bool* cancel) {
    auto out=std::make_shared<KernelEvidence>();
    out->observedWidth=im.width; out->observedHeight=im.height;
    if(r.gridStatus!="supported" || r.candidates.empty()) { out->reason=r.verdict; return out; }
    const Candidate& leader=r.candidates.front();
    out->width=leader.x.size; out->height=leader.y.size;
    const int lineCount=o.refine?std::max(o.lines,o.refinementLines):o.lines;
    const auto xLines=sampleLines(im,true,lineCount), yLines=sampleLines(im,false,lineCount);
    const std::array<Lines,2> xHalves{splitLines(xLines,0),splitLines(xLines,1)}, yHalves{splitLines(yLines,0),splitLines(yLines,1)};
    out->presets.resize(o.kernels.size());
    forEachParallel(out->presets.size(),o.threads,cancel,[&](size_t index) {
        auto& p=out->presets[index];
        p.kernel=o.kernels[index];
        const auto own=std::find_if(r.candidates.begin(),r.candidates.end(),[&](const Candidate& c){return !c.continuous && c.x.kernel.name()==p.kernel.name();});
        p.ownGeometry=own!=r.candidates.end() && own->x.size==leader.x.size && own->y.size==leader.y.size;
        const Candidate& geometry=p.ownGeometry?*own:leader;
        p.extentX=geometry.x.extent; p.shiftX=geometry.x.shift; p.extentY=geometry.y.extent; p.shiftY=geometry.y.shift;
        // An axis at the source's size has no fit to score: it takes the other axis's score, so it favours no preset.
        auto score=[&](const Lines& x,const Lines& y,double& sx,double& sy) {
            sx=out->width<im.width?lineScore(x,out->width,p.kernel,p.extentX,p.shiftX):lineScore(y,out->height,p.kernel,p.extentY,p.shiftY);
            sy=out->height<im.height && out->width<im.width?lineScore(y,out->height,p.kernel,p.extentY,p.shiftY):sx;
            return std::log(std::sqrt(sx*sy));
        };
        double sx=0, sy=0, unused=0;
        p.logScore.full=score(xLines,yLines,sx,sy); p.logScore.x=std::log(sx); p.logScore.y=std::log(sy);
        p.logScore.even=score(xHalves[0],yHalves[0],unused,unused);
        p.logScore.odd=score(xHalves[1],yHalves[1],unused,unused);
    });
    out->measured=true;
    return out;
}

KernelDecision decideKernel(const std::vector<std::shared_ptr<KernelEvidence>>& evidence,int consensusWidth,int consensusHeight) {
    KernelDecision out;
    std::vector<const KernelEvidence*> frames;
    for(const auto& e:evidence) if(e && e->measured && e->width==consensusWidth && e->height==consensusHeight
                                   && (frames.empty() || e->presets.size()==frames.front()->presets.size())) frames.push_back(e.get());
    out.frames=int(frames.size());
    if(frames.empty() || frames.front()->presets.empty()) { out.status="not_measured"; return out; }
    const auto& presets=frames.front()->presets;
    const size_t count=presets.size();
    std::vector<double> mean(count,0);  // log geometric mean of the least-squares scores
    for(const auto* f:frames) for(size_t p=0;p<count;++p) mean[p]+=f->presets[p].logScore.full/double(frames.size());
    const double best=*std::min_element(mean.begin(),mean.end());
    auto smallest=[&](const std::vector<size_t>& set) {
        size_t chosen=set.front();
        for(const size_t p:set) {
            const int a=presets[p].kernel.radius(), b=presets[chosen].kernel.radius();
            if(a<b || (a==b && mean[p]<mean[chosen])) chosen=p;
        }
        return chosen;
    };
    std::vector<size_t> parsimony;
    for(size_t p=0;p<count;++p) if(mean[p]<=best-std::log(1-separation)) parsimony.push_back(p);
    auto finish=[&](std::string status,std::string rule,const std::vector<size_t>& band,size_t carried) {
        out.status=std::move(status); out.rule=std::move(rule); out.carried=presets[carried].kernel.name();
        for(const size_t p:band) out.band.push_back(presets[p].kernel.name());
        return out;
    };
    if(count==1) return finish("unresolved","single_preset",parsimony,0);  // nothing to tell it apart from
    if(frames.size()>1 && frames.size()<size_t(minimumFrames)) return finish("unresolved","parsimony",parsimony,smallest(parsimony));
    std::vector<size_t> survivors;
    for(size_t q=0;q<count;++q) {
        bool dominated=false;
        for(size_t p=0;p<count && !dominated;++p) if(p!=q) dominated=frames.size()==1?dominatesSingle(*frames.front(),p,q):dominates(frames,p,q);
        if(!dominated) survivors.push_back(q);
    }
    if(survivors.empty()) return finish("statistics_disagree","parsimony",parsimony,smallest(parsimony));
    if(survivors.size()==1) return finish("supported_model",frames.size()==1?"single_frame":"cross_frame",survivors,survivors.front());
    if(frames.size()>1) return finish("ambiguous_band","cross_frame_band",survivors,smallest(survivors));
    std::vector<size_t> band;
    std::set_intersection(survivors.begin(),survivors.end(),parsimony.begin(),parsimony.end(),std::back_inserter(band));
    if(band.empty()) band=survivors;
    return finish("unresolved","parsimony",band,smallest(band));
}

std::string kernelEvidenceJson(const KernelEvidence& e) {
    auto s=jsonStream();
    s<<"{\"measured\":"<<(e.measured?"true":"false")<<",\"reason\":"; writeString(s,e.reason);
    s<<",\"width\":"<<e.width<<",\"height\":"<<e.height<<",\"observed_width\":"<<e.observedWidth<<",\"observed_height\":"<<e.observedHeight<<",\"presets\":[";
    for(size_t i=0;i<e.presets.size();++i) {
        const auto& p=e.presets[i];
        if(i) s<<',';
        s<<"{\"name\":"; writeString(s,p.kernel.name()); s<<",\"radius\":"<<p.kernel.radius()<<",\"geometry\":"<<(p.ownGeometry?"\"own\"":"\"leader\"");
        s<<",\"extent_x\":"; writeNumber(s,p.extentX); s<<",\"shift_x\":"; writeNumber(s,p.shiftX);
        s<<",\"extent_y\":"; writeNumber(s,p.extentY); s<<",\"shift_y\":"; writeNumber(s,p.shiftY);
        s<<",\"log_score\":"; writeSplit(s,p.logScore); s<<'}';
    }
    s<<"]}";
    return s.str();
}

std::string kernelDecisionJson(const KernelDecision& d) {
    auto s=jsonStream();
    s<<"{\"status\":"; writeString(s,d.status); s<<",\"band\":[";
    for(size_t i=0;i<d.band.size();++i) { if(i) s<<','; writeString(s,d.band[i]); }
    s<<"],\"carried\":"; writeString(s,d.carried); s<<",\"rule\":"; writeString(s,d.rule); s<<",\"frames\":"<<d.frames<<'}';
    return s.str();
}
}
