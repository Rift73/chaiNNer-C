#pragma once
#include "resample.hpp"
#include <atomic>
#include <complex>
#include <vector>

namespace native {
struct FoldPoint {
    double extent=0,coherence=0;
    int bins=0;
    double highEnergy=0,permuted=0,excess=0,participation=0,prominence=0;
    int sampleLines=0;
};
// Bluestein transforms arbitrary image dimensions through a radix-2 FFT.
class SpectrumPlan {
public:
    explicit SpectrumPlan(int length);
    std::vector<std::complex<double>> transform(const std::vector<double>&) const;
private:
    int length_, padded_;
    std::vector<std::complex<double>> chirp_,filter_;
};
std::vector<FoldPoint> foldScan(const Image&,bool horizontal,int minimum,int maximum,
                                int lines,std::atomic_bool* cancel=nullptr,int threads=4);
}
