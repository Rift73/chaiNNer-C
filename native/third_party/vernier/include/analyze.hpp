#pragma once
#include "resample.hpp"
#include "spectrum.hpp"
#include <atomic>
#include <functional>
#include <string>
#include <vector>
#include <array>

namespace native {
int defaultWorkers();
struct Options {
    int minHeight=500, maxHeight=1000, minWidth=0, maxWidth=0;
    int lines=24, refinementLines=128, threads=defaultWorkers();
    bool refine=true;
    std::vector<Kernel> kernels=commonKernels();
};
struct AxisFit {
    int size=0;
    double extent=0, shift=0;
    Kernel kernel;
    double score=0, mse=0, prominence=1, predictive=-1;
    double pivotRatio=0;
    double refinementGain=0;
    bool fractionalSupported=false;
    bool identifiable=true;
    int informative=0;
};
struct Candidate {
    AxisFit x,y;
    double score=0;
    double rmse=-1;
    double spatialSupport=-1;
    double sharedGeometryScore=0;
    double evenScore=0,oddScore=0;
    bool continuous=false;
};
struct Result {
    int width=0,height=0;
    Options options;
    std::vector<AxisFit> horizontal,vertical;
    std::vector<AxisFit> alternativesX,alternativesY;
    std::vector<FoldPoint> foldX,foldY;
    std::vector<Candidate> candidates;
    std::string verdict, explanation;
    double seconds=0, separation=0;
    bool competingGrids=false;
    bool kernelConsistent=false;
    std::string gridStatus="inconclusive",kernelStatus="unresolved";
    size_t recommendedCandidate=0;
    int rejectedSingularFits=0;
    std::string lightDomain="encoded",colorAssumptions;
    std::array<double,3> linearWeights{.2126,.7152,.0722};
};
Candidate exportGeometry(Candidate);
using Progress=std::function<void(double,const std::string&)>;
Result analyze(const Image&,const Options&,std::atomic_bool* cancel=nullptr,Progress progress={});
std::vector<AxisFit> scanAxis(const Image&,bool horizontal,int minimum,int maximum,
                             const Options&,std::atomic_bool*,Progress={});
}
