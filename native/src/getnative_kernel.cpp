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
    std::atomic_size_t next=0; std::atomic_bool failed=false; std::exception_ptr error; std::mutex mutex;
    auto worker=[&] { try { for(;;) {
        if(failed) return;
        if(cancel && cancel->load()) throw std::runtime_error("Analysis cancelled");
        const size_t index=next.fetch_add(1);
        if(index>=out->presets.size()) return;
        auto& p=out->presets[index];
        p.kernel=o.kernels[index];
        const auto own=std::find_if(r.candidates.begin(),r.candidates.end(),[&](const Candidate& c){return !c.continuous && c.x.kernel.name()==p.kernel.name();});
        p.ownGeometry=own!=r.candidates.end() && own->x.size==leader.x.size && own->y.size==leader.y.size;
        const Candidate& geometry=p.ownGeometry?*own:leader;
        p.extentX=geometry.x.extent; p.shiftX=geometry.x.shift; p.extentY=geometry.y.extent; p.shiftY=geometry.y.shift;
        auto score=[&](const Lines& x,const Lines& y,double& sx,double& sy) {
            sx=lineScore(x,out->width,p.kernel,p.extentX,p.shiftX); sy=lineScore(y,out->height,p.kernel,p.extentY,p.shiftY);
            return std::log(std::sqrt(sx*sy));
        };
        double sx=0, sy=0, unused=0;
        p.logScore.full=score(xLines,yLines,sx,sy); p.logScore.x=std::log(sx); p.logScore.y=std::log(sy);
        p.logScore.even=score(xHalves[0],yHalves[0],unused,unused);
        p.logScore.odd=score(xHalves[1],yHalves[1],unused,unused);
    } } catch(...) { std::lock_guard lock(mutex); if(!error) error=std::current_exception(); failed=true; } };
    std::vector<std::thread> workers;
    for(int t=0;t<std::clamp(o.threads,1,int(out->presets.size()));++t) workers.emplace_back(worker);
    for(auto& t:workers) t.join();
    if(error) std::rethrow_exception(error);
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
