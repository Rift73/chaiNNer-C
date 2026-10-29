// A MIGraphX program as chaiNNer-C keeps it: compiled from ONNX (with the custom ops of loaded libraries), for a GPU
// or MIGraphX's CPU reference target, run on device (or, for ref, host) memory, saved as a header (format version,
// MIGraphX version, target and gfx, compute type, custom ops) and MIGraphX's own value of the program, in msgpack.
#pragma once

#include <hip/hip_runtime_api.h>
#include <migraphx/argument.hpp>
#include <migraphx/program.hpp>
#include <migraphx/shape.hpp>

#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace cn_migx {

// MIGraphX's version, built against.
std::string migraphx_version();
// The gfx name of a HIP device ("gfx1100"), or std::runtime_error.
std::string device_gfx(int device);
// A HIP error as std::runtime_error.
void check_hip(hipError_t error, char const* call);
// NumPy-style names of MIGraphX's types: "float32", "float16", "bfloat16", "fp8e4m3fnuz", ...
std::string type_name(migraphx::shape::type_t type);

struct CompileOptions {
    int device = 0;              // HIP device, for the gpu target
    std::string target = "gpu";  // "gpu", or "ref" (MIGraphX's CPU reference)
    bool fp16 = false;
    bool fp8 = false;
    bool exhaustive_tune = false;
    std::map<std::string, std::vector<std::size_t>> input_dims;
    std::vector<std::map<std::string, migraphx::argument>> calibration;  // host inputs, for fp8
};

class Program {
public:
    // From an ONNX model's bytes, or (from_file) from the file at `model`, whose external data lies beside it.
    static Program compile(std::string const& model, bool from_file, CompileOptions const& options);
    // From save()'s bytes, on a HIP device (ignored for a ref program).
    static Program load(char const* data, std::size_t size, int device);

    Program(Program&&) noexcept;
    Program& operator=(Program&&) noexcept;
    ~Program();

    std::vector<char> save() const;

    std::string const& target() const { return target_; }
    std::string const& gfx() const { return gfx_; }
    std::string const& compute_type() const { return compute_type_; }
    std::vector<std::string> const& custom_ops() const { return custom_ops_; }
    int device() const { return device_; }
    std::vector<std::string> const& input_names() const { return input_names_; }
    std::vector<migraphx::shape> const& input_shapes() const { return input_shapes_; }
    std::vector<migraphx::shape> const& output_shapes() const { return output_shapes_; }

    // Runs once on these buffers (device memory, ordered on `stream`; host memory for a ref program, which returns
    // when done): every input by name, every output in order. One run at a time per program.
    void run(std::map<std::string, void*> const& inputs, std::vector<void*> const& outputs, hipStream_t stream);

    // Frees the program (its weights and scratch) now; the destructor does what close() has not.
    void close();

private:
    Program() = default;
    void describe();

    migraphx::program program_;
    std::string target_, gfx_, compute_type_;
    std::vector<std::string> custom_ops_;
    int device_ = -1;
    bool closed_ = false;
    std::vector<std::string> input_names_;
    std::vector<migraphx::shape> input_shapes_, output_shapes_;
    bool output_parameters_ = false;  // outputs written into main:#output_<i> parameters
    std::unique_ptr<std::mutex> mutex_ = std::make_unique<std::mutex>();
};

}  // namespace cn_migx
