// The TensorRT runner: one deserialized engine and `streams` slots, each an execution
// context with its own CUDA stream, device buffers, pinned staging buffers and CUDA graphs
// (one per input shape). A frame's planes go up as stored, are converted, tiled and blended
// on the GPU (kernels.cu) and come back in the same sample format.
#pragma once

#include "frame.h"
#include "plan.h"

#include <cstddef>
#include <memory>
#include <string>

namespace cn_trt {

class Runner {
public:
    // Checks the arguments before any CUDA call; nvinfer_11.dll and nvinfer_plugin_11.dll (on
    // Linux libnvinfer.so.11 and libnvinfer_plugin.so.11) are loaded here (Python's `import
    // tensorrt` has usually loaded them already).
    Runner(void const* engine, std::size_t size, int device, int streams);
    ~Runner();
    Runner(Runner const&) = delete;
    Runner& operator=(Runner const&) = delete;

    int input_channels() const;
    int output_channels() const;
    int scale() const;
    Bounds const& bounds() const;
    Element input_element() const;
    Element output_element() const;

    // Upscales `job` (describe_frame with this engine's bounds, scale and channels) from
    // `planes` into `result`. Up to `streams` calls run at once; more wait for a free slot.
    void upscale(FrameJob const& job, HostPlane const* planes, HostPlane const* result, bool clip);

    // Why a slot runs without CUDA graphs (the first failed capture); empty when none failed.
    std::string capture_failure() const;

    // Frees everything, every step even after a CUDA error, and throws the first error. The
    // destructor frees what close() has not, without reporting (a destructor cannot throw).
    void close();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cn_trt
