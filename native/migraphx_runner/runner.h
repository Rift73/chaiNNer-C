// The MIGraphX runner, the TensorRT runner's counterpart: one program loaded `streams` times, each copy with its own
// HIP stream, device buffers and pinned staging buffers. A frame's planes go up as stored, are converted, tiled and
// blended on the GPU (kernels.hip) and come back in the same sample format; the tiles, blend weights and channel rules
// are the TensorRT runner's (trt_runner/plan.h and frame.h), on the program's fixed input size.
#pragma once

#include "frame.h"
#include "plan.h"

#include <cstddef>
#include <memory>

namespace cn_migx {

using cn_trt::Bounds;
using cn_trt::Element;
using cn_trt::FrameJob;
using cn_trt::HostPlane;

class Runner {
public:
    // A saved GPU program (Program::save) of one NCHW input of a fixed size and one output, its integer upscale.
    // Checks the arguments before any HIP call.
    Runner(void const* program, std::size_t size, int device, int streams);
    ~Runner();
    Runner(Runner const&) = delete;
    Runner& operator=(Runner const&) = delete;

    int input_channels() const;
    int output_channels() const;
    int scale() const;
    Bounds const& bounds() const;
    Element input_element() const;
    Element output_element() const;

    // Upscales `job` (describe_frame with this program's bounds, scale and channels) from `planes` into `result`. Up to
    // `streams` calls run at once; more wait for a free slot.
    void upscale(FrameJob const& job, HostPlane const* planes, HostPlane const* result, bool clip);

    // Frees everything, every step even after a HIP error, and throws the first error. The destructor frees what
    // close() has not, without reporting.
    void close();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cn_migx
