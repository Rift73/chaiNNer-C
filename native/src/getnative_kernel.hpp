/* Kernel identification for Get Native Resolution, layered on the vendored
 * Vernier core. Each frame scores every preset by least squares on the frame's
 * leading grid (all lines, both line halves, both axes); a pre-registered paired
 * rule turns the scores into one decision across the frames that share the
 * consensus grid. The free bicubic is a nuisance model, never a candidate. Also the
 * search of an unscaled axis, which the core's bounds keep it from reaching. */
#pragma once
#include "analyze.hpp"
#include <atomic>
#include <memory>
#include <string>
#include <vector>

namespace native {
// Log least-squares score on all sampled lines, on the even and odd line halves,
// and per axis. full/even/odd combine both axes.
struct KernelSplit { double full=0, even=0, odd=0, x=0, y=0; };
struct PresetEvidence {
    Kernel kernel;
    bool ownGeometry=false;      // the preset's fitted geometry; else the leader's
    double extentX=0, shiftX=0, extentY=0, shiftY=0;
    KernelSplit logScore;
};
struct KernelEvidence {
    bool measured=false;
    std::string reason;
    int width=0, height=0, observedWidth=0, observedHeight=0;
    std::vector<PresetEvidence> presets;
};
struct KernelDecision {
    std::string status;  // "supported_model" | "ambiguous_band" | "unresolved" | "statistics_disagree" | "not_measured"
    std::vector<std::string> band; std::string carried, rule; int frames = 0;
};
// A frame the core left without a supported grid, where one axis alone shows a grid: the other axis goes to the source's
// size, which the core's search cannot reach, and the frame is ranked and judged on the scaled axis alone. Any other
// result stays as it is.
void searchUnscaledAxis(const Image&, Result&, const Options&, std::atomic_bool* cancel);
std::shared_ptr<KernelEvidence> measureKernelEvidence(const Image&, const Result&, const Options&, std::atomic_bool* cancel);
KernelDecision decideKernel(const std::vector<std::shared_ptr<KernelEvidence>>&, int consensusWidth, int consensusHeight);
std::string kernelEvidenceJson(const KernelEvidence&);
std::string kernelDecisionJson(const KernelDecision&);
}
