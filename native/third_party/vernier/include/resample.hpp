#pragma once
#include <array>
#include <string>
#include <stdexcept>
#include <vector>

namespace native {
class SingularOperator:public std::runtime_error {public:using std::runtime_error::runtime_error;};
enum class Family { Bilinear, Bicubic, Lanczos, Spline16, Spline36, Spline64 };
struct Kernel {
    Family family = Family::Bilinear;
    double b = 0, c = .5;
    int taps = 3;
    std::string name() const;
    int radius() const;
    double value(double x) const;
};
std::vector<Kernel> commonKernels();

// A maps native samples onto observed pixel centres. All arithmetic is double.
// extent is the native-coordinate span; shift is relative to a centred crop.
// A clamped boundary is explicit. Scoring excludes a 12-pixel boundary.
class Resampler {
public:
    Resampler(int nativeSize, int observedSize, Kernel kernel,
              double extent = 0, double shift = 0, int holdoutPeriod = 0);
    std::vector<double> forward(const std::vector<double>& x) const;
    std::vector<double> inverse(const std::vector<double>& y) const;
    void project(const std::vector<double>& y, std::vector<double>& x,
                 std::vector<double>& fitted) const;
    // Pixel-major, signal-interleaved batch: independent signals occupy SIMD
    // lanes. The recurrence order for each signal remains unchanged.
    std::vector<double> projectBatch(const std::vector<double>& y,int signals) const;
    std::vector<double> inverseBatch(const std::vector<double>& y,int signals) const;
    std::vector<double> forwardBatch(const std::vector<double>& x,int signals) const;
    double predictionNoiseGain(int observedIndex) const;
    int nativeSize() const { return n_; }
    int observedSize() const { return m_; }
    double minPivotRatio() const { return pivotRatio_; }
private:
    struct Row { int start = 0, count = 0; std::array<double, 12> weight{}; };
    int n_, m_, band_, holdout_;
    std::vector<Row> rows_;
    std::vector<double> lower_;
    double pivotRatio_ = 0;
    double& l(int i, int j) { return lower_[size_t(i) * (band_ + 1) + i - j]; }
    double l(int i, int j) const { return lower_[size_t(i) * (band_ + 1) + i - j]; }
    void solve(std::vector<double>& rhs) const;
};

struct Image {
    int width = 0, height = 0;
    std::vector<double> pixels;
    double at(int x, int y) const { return pixels[size_t(y) * width + x]; }
};
Image resize(const Image& image, int width, int height, const Kernel& kernel);
Image descale(const Image& image, int width, int height, const Kernel& kernel,
              double extentX = 0, double extentY = 0, double shiftX = 0, double shiftY = 0,int threads=4);
Image reconstruct(const Image& latent, int width, int height, const Kernel& kernel,
                  double extentX = 0, double extentY = 0, double shiftX = 0, double shiftY = 0,int threads=4);
}
