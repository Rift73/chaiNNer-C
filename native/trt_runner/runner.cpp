#include "runner.h"

#include "kernels.h"

#include <cuda_runtime_api.h>
#include <NvInferPlugin.h>
#include <NvInferRuntime.h>
#include <NvInferVersion.h>
#include <windows.h>

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <exception>
#include <list>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static_assert(NV_TENSORRT_MAJOR == 11, "The TensorRT runner requires TensorRT 11.");

namespace cn_trt {
namespace {

// TensorRTSession's MAX_GRAPHS and MAX_SEEN (nodes/impl/tensorrt/inference.py).
constexpr std::size_t max_graphs = 4;
constexpr std::size_t max_seen = 32;
// dual.ALIGNED_INPUT: a DUAL engine's input name carries the multiple its sizes must be.
constexpr char aligned_input[] = "input_aligned_";

std::string cuda_message(cudaError_t error, char const* call) {
    return std::string("CUDA error ") + cudaGetErrorName(error) + " in " + call + ": " + cudaGetErrorString(error);
}

void check(cudaError_t error, char const* call) {
    if (error != cudaSuccess) throw std::runtime_error(cuda_message(error, call));
}

// Keeps TensorRT's last error message for the exception that reports the failed call.
class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, nvinfer1::AsciiChar const* message) noexcept override {
        if (severity > Severity::kERROR) return;
        std::lock_guard<std::mutex> lock(mutex_);
        std::snprintf(last_, sizeof last_, "%s", message);
    }
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        last_[0] = '\0';
    }
    [[noreturn]] void fail(std::string what) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (last_[0]) what += std::string(": ") + last_;
        last_[0] = '\0';
        throw std::runtime_error(what + ".");
    }

private:
    std::mutex mutex_;
    char last_[1024] = {};
};

// nvinfer_11.dll and nvinfer_plugin_11.dll are delay-loaded, so the module imports and its
// CPU functions run without TensorRT. They are loaded before the first call into them:
// already loaded (`import tensorrt` loads tensorrt_libs' copies), else from the default and
// added DLL directories, else the PATH. A missing one is this error, not the delay-load
// helper's crash.
void load_tensorrt() {
    for (wchar_t const* name : {L"nvinfer_11.dll", L"nvinfer_plugin_11.dll"}) {
        if (GetModuleHandleW(name) || LoadLibraryExW(name, nullptr, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS) || LoadLibraryW(name))
            continue;
        throw std::runtime_error(
            "TensorRT 11 (nvinfer_11.dll, nvinfer_plugin_11.dll) was not found: install chaiNNer's TensorRT "
            "package, or put TensorRT's DLLs on the PATH.");
    }
}

Element element_of(nvinfer1::DataType type, char const* name) {
    switch (type) {
    case nvinfer1::DataType::kFLOAT: return Element::F32;
    case nvinfer1::DataType::kHALF: return Element::F16;
    case nvinfer1::DataType::kBF16: return Element::BF16;
    default:
        throw std::invalid_argument(std::string("The TensorRT runner takes float32, float16 or bfloat16 tensors; ") +
                                    name + " is not one.");
    }
}

int alignment_of(std::string const& name) {
    std::size_t const prefix = sizeof aligned_input - 1;
    if (name.compare(0, prefix, aligned_input) != 0 || name.size() == prefix ||
        !std::all_of(name.begin() + prefix, name.end(), [](char c) { return c >= '0' && c <= '9'; }))
        return tile_alignment;
    return std::stoi(name.substr(prefix));
}

// A grow-only device or pinned host buffer.
template <cudaError_t (*Allocate)(void**, std::size_t), cudaError_t (*Free)(void*)>
class Buffer {
public:
    Buffer() = default;
    Buffer(Buffer const&) = delete;
    Buffer& operator=(Buffer const&) = delete;
    ~Buffer() { release(); }
    bool fits(std::size_t bytes) const { return bytes <= size_; }
    void grow(std::size_t bytes) {
        if (fits(bytes)) return;
        check(release(), "freeing a buffer");
        check(Allocate(&ptr_, bytes), "allocating a buffer");
        size_ = bytes;
    }
    cudaError_t release() noexcept {
        cudaError_t const error = ptr_ ? Free(ptr_) : cudaSuccess;
        ptr_ = nullptr;
        size_ = 0;
        return error;
    }
    void* get() const { return ptr_; }
    std::size_t size() const { return size_; }

private:
    void* ptr_ = nullptr;
    std::size_t size_ = 0;
};

cudaError_t device_allocate(void** ptr, std::size_t bytes) { return cudaMalloc(ptr, bytes); }
cudaError_t host_allocate(void** ptr, std::size_t bytes) { return cudaMallocHost(ptr, bytes); }
using DeviceBuffer = Buffer<device_allocate, cudaFree>;
using PinnedBuffer = Buffer<host_allocate, cudaFreeHost>;

struct Graph {
    int height;
    int width;
    cudaGraph_t graph;
    cudaGraphExec_t exec;
};

// One execution context with its stream and buffers; a frame holds it from upload to download.
struct Slot {
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t stream = nullptr;
    DeviceBuffer source, input, output, activation, accumulator, result, weights;
    PinnedBuffer host_source, host_result, host_weights;
    std::vector<float> uploaded;      // the blend weights on the device
    std::list<Graph> graphs;          // most recently used first
    std::list<std::pair<int, int>> seen;  // input shapes run, oldest first
    bool graph_failed = false;
    int shape_h = 0;  // the context's input shape and tensor addresses are set for this size
    int shape_w = 0;
};

}  // namespace

struct Runner::Impl {
    Logger logger;
    int device = 0;
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    std::string input_name, output_name;
    Element input_element = Element::F32, output_element = Element::F32;
    int input_channels = 0, output_channels = 0, scale = 0;
    Bounds bounds{};
    std::vector<std::unique_ptr<Slot>> slots;
    std::mutex mutex;
    std::condition_variable freed;
    std::vector<Slot*> idle;
    std::string capture_failure;

    ~Impl() { teardown(nullptr); }

    // Frees everything: every step runs, the contexts before their memory and the engine
    // before its runtime; the first error goes to `first` (a failed construction passes none,
    // so a teardown error cannot hide its own).
    void teardown(std::exception_ptr* first) noexcept {
        auto attempt = [first](cudaError_t error, char const* call) {
            if (error != cudaSuccess && first && !*first)
                *first = std::make_exception_ptr(std::runtime_error(cuda_message(error, call)));
        };
        if (!engine && !runtime && slots.empty()) return;
        attempt(cudaSetDevice(device), "cudaSetDevice");
        for (auto& slot : slots)
            if (slot->stream) attempt(cudaStreamSynchronize(slot->stream), "cudaStreamSynchronize");
        for (auto& slot : slots) {
            attempt(drop_graphs(*slot), "cudaGraphDestroy");
            for (PinnedBuffer* buffer : {&slot->host_source, &slot->host_result, &slot->host_weights})
                attempt(buffer->release(), "cudaFreeHost");
            delete slot->context;
            slot->context = nullptr;
            for (DeviceBuffer* buffer : {&slot->source, &slot->input, &slot->output, &slot->activation,
                                         &slot->accumulator, &slot->result, &slot->weights})
                attempt(buffer->release(), "cudaFree");
            if (slot->stream) attempt(cudaStreamDestroy(slot->stream), "cudaStreamDestroy");
            slot->stream = nullptr;
        }
        slots.clear();
        idle.clear();
        delete engine;
        engine = nullptr;
        delete runtime;
        runtime = nullptr;
    }

    static cudaError_t drop_graphs(Slot& slot) noexcept {
        cudaError_t first = cudaSuccess;
        for (Graph const& graph : slot.graphs)
            for (cudaError_t error : {cudaGraphExecDestroy(graph.exec), cudaGraphDestroy(graph.graph)})
                if (first == cudaSuccess) first = error;
        slot.graphs.clear();
        return first;
    }

    Slot& acquire() {
        std::unique_lock<std::mutex> lock(mutex);
        if (!engine) throw std::runtime_error("The TensorRT runner is closed.");
        freed.wait(lock, [this] { return !idle.empty(); });
        Slot* slot = idle.back();
        idle.pop_back();
        return *slot;
    }

    void release(Slot& slot) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            idle.push_back(&slot);
        }
        freed.notify_one();
    }

    // Sets the context's input shape, sizes the tensors and activation memory for it (a
    // growth waits for the stream and drops the graphs, which hold the old addresses) and
    // points the context at them.
    void prepare(Slot& s, int height, int width) {
        if (s.shape_h == height && s.shape_w == width) return;
        nvinfer1::Dims4 const dims{1, input_channels, height, width};
        if (!s.context->setInputShape(input_name.c_str(), dims))
            logger.fail("Tile shape (1, " + std::to_string(input_channels) + ", " + std::to_string(height) + ", " +
                        std::to_string(width) + ") is outside the engine's optimization profile");
        nvinfer1::Dims const out = s.context->getTensorShape(output_name.c_str());
        if (out.nbDims != 4 || out.d[0] != 1 || out.d[1] != output_channels ||
            out.d[2] != static_cast<int64_t>(height) * scale || out.d[3] != static_cast<int64_t>(width) * scale)
            throw std::runtime_error("The engine's output shape for a " + std::to_string(width) + "x" +
                                     std::to_string(height) + " tile is not its " + std::to_string(scale) + "x upscale.");
        std::size_t const pixels = static_cast<std::size_t>(height) * width;
        std::size_t const in_bytes = input_channels * pixels * element_bytes(input_element);
        std::size_t const out_bytes = output_channels * pixels * scale * scale * element_bytes(output_element);
        std::size_t const act_bytes = s.context->updateDeviceMemorySizeForShapes();
        if (!s.input.fits(in_bytes) || !s.output.fits(out_bytes) || !s.activation.fits(act_bytes)) {
            check(cudaStreamSynchronize(s.stream), "cudaStreamSynchronize");
            check(drop_graphs(s), "cudaGraphDestroy");
            s.input.grow(in_bytes);
            s.output.grow(out_bytes);
            s.activation.grow(act_bytes);
        }
        if (act_bytes) s.context->setDeviceMemoryV2(s.activation.get(), static_cast<int64_t>(s.activation.size()));
        if (!s.context->setTensorAddress(input_name.c_str(), s.input.get()) ||
            !s.context->setTensorAddress(output_name.c_str(), s.output.get()))
            logger.fail("TensorRT did not take the tensor addresses");
        s.shape_h = height;
        s.shape_w = width;
    }

    // Enqueues the engine: the shape's captured graph when there is one, else TensorRT's
    // enqueue, after which a shape run before is captured (TensorRTSession._execute).
    void execute(Slot& s) {
        for (auto it = s.graphs.begin(); it != s.graphs.end(); ++it) {
            if (it->height != s.shape_h || it->width != s.shape_w) continue;
            s.graphs.splice(s.graphs.begin(), s.graphs, it);
            check(cudaGraphLaunch(s.graphs.front().exec, s.stream), "cudaGraphLaunch");
            return;
        }
        if (!s.context->enqueueV3(s.stream)) logger.fail("TensorRT failed to enqueue inference");
        std::pair<int, int> const shape{s.shape_h, s.shape_w};
        if (std::find(s.seen.begin(), s.seen.end(), shape) == s.seen.end()) {
            s.seen.push_back(shape);
            if (s.seen.size() > max_seen) s.seen.pop_front();
        } else if (!s.graph_failed) {
            capture(s);
        }
    }

    // Captures one enqueue (recorded, not run). A failed capture leaves the slot on
    // TensorRT's enqueue, its reason kept for capture_failure(); the error it left behind
    // is taken so that the next launch does not report it.
    void capture(Slot& s) {
        check(cudaStreamSynchronize(s.stream), "cudaStreamSynchronize");
        cudaGraph_t graph = nullptr;
        bool enqueued = false;
        cudaError_t error = cudaStreamBeginCapture(s.stream, cudaStreamCaptureModeThreadLocal);
        if (error == cudaSuccess) {
            enqueued = s.context->enqueueV3(s.stream);
            error = cudaStreamEndCapture(s.stream, &graph);
            if (error == cudaSuccess) {
                if (enqueued) {
                    cudaGraphExec_t exec = nullptr;
                    error = cudaGraphInstantiate(&exec, graph, 0);
                    if (error == cudaSuccess) {
                        s.graphs.push_front({s.shape_h, s.shape_w, graph, exec});
                        if (s.graphs.size() > max_graphs) {
                            Graph const oldest = s.graphs.back();
                            s.graphs.pop_back();
                            check(cudaGraphExecDestroy(oldest.exec), "cudaGraphExecDestroy");
                            check(cudaGraphDestroy(oldest.graph), "cudaGraphDestroy");
                        }
                        return;
                    }
                }
                check(cudaGraphDestroy(graph), "cudaGraphDestroy");
            }
        }
        cudaError_t const left = cudaGetLastError();
        logger.clear();
        check(drop_graphs(s), "cudaGraphDestroy");
        s.graph_failed = true;
        std::lock_guard<std::mutex> lock(mutex);
        if (capture_failure.empty())
            capture_failure = "TensorRT engine not captured in a CUDA graph (enqueued " +
                              std::string(enqueued ? "True" : "False") + ", " + cudaGetErrorName(error) + ", " +
                              cudaGetErrorName(left) + "); it runs without one";
    }
};

Runner::Runner(void const* engine, std::size_t size, int device, int streams) : impl_(std::make_unique<Impl>()) {
    check_runner_arguments(size, device, streams);
    load_tensorrt();
    Impl& m = *impl_;
    m.device = device;
    check(cudaSetDevice(device), "cudaSetDevice");
    // As deserialize_engine: TensorRT's standard plugins first.
    if (!initLibNvInferPlugins(&m.logger, "")) m.logger.fail("TensorRT could not register its plugins");
    m.runtime = nvinfer1::createInferRuntime(m.logger);
    if (!m.runtime) m.logger.fail("TensorRT could not create its runtime");
    m.engine = m.runtime->deserializeCudaEngine(engine, size);
    if (!m.engine)
        m.logger.fail("Failed to deserialize the TensorRT engine. It may have been built with an incompatible "
                      "TensorRT version or for a different GPU architecture");

    std::vector<std::string> inputs, outputs;
    for (int i = 0; i < m.engine->getNbIOTensors(); ++i) {
        char const* name = m.engine->getIOTensorName(i);
        (m.engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT ? inputs : outputs).emplace_back(name);
    }
    if (inputs.size() != 1 || outputs.size() != 1)
        throw std::invalid_argument("Expected one input and one output tensor, got " + std::to_string(inputs.size()) +
                                    " and " + std::to_string(outputs.size()) + ".");
    m.input_name = inputs[0];
    m.output_name = outputs[0];
    nvinfer1::Dims const shape = m.engine->getTensorShape(m.input_name.c_str());
    if (shape.nbDims != 4) throw std::invalid_argument("Expected an NCHW input tensor.");
    m.input_element = element_of(m.engine->getTensorDataType(m.input_name.c_str()), m.input_name.c_str());
    m.output_element = element_of(m.engine->getTensorDataType(m.output_name.c_str()), m.output_name.c_str());
    nvinfer1::Dims const min = m.engine->getProfileShape(m.input_name.c_str(), 0, nvinfer1::OptProfileSelector::kMIN);
    nvinfer1::Dims const max = m.engine->getProfileShape(m.input_name.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
    m.input_channels = static_cast<int>(shape.d[1]);
    m.bounds = {static_cast<int>(min.d[2]), static_cast<int>(min.d[3]), static_cast<int>(max.d[2]),
                static_cast<int>(max.d[3]), alignment_of(m.input_name)};

    for (int i = 0; i < streams; ++i) {
        auto slot = std::make_unique<Slot>();
        // Activation memory is sized for each tile shape, not the profile's maximum (prepare).
        slot->context = m.engine->createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED);
        if (!slot->context) m.logger.fail("Failed to create a TensorRT execution context");
        check(cudaStreamCreateWithFlags(&slot->stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
        m.idle.push_back(slot.get());
        m.slots.push_back(std::move(slot));
    }

    // The scale from TensorRT's shape inference at the profile's minimum (read_engine_info).
    nvinfer1::IExecutionContext& probe = *m.slots[0]->context;
    if (!probe.setInputShape(m.input_name.c_str(), min)) m.logger.fail("Failed to infer the TensorRT engine's output shape");
    nvinfer1::Dims const out = probe.getTensorShape(m.output_name.c_str());
    if (out.nbDims != 4 || min.d[2] <= 0 || min.d[3] <= 0 || out.d[2] % min.d[2] || out.d[3] % min.d[3] ||
        out.d[2] / min.d[2] != out.d[3] / min.d[3])
        throw std::invalid_argument("The engine's output is not a uniform integer upscale of its input.");
    m.scale = static_cast<int>(out.d[2] / min.d[2]);
    m.output_channels = static_cast<int>(out.d[1]);
    check_engine_channels(m.input_channels, m.output_channels);
}

Runner::~Runner() = default;

int Runner::input_channels() const { return impl_->input_channels; }
int Runner::output_channels() const { return impl_->output_channels; }
int Runner::scale() const { return impl_->scale; }
Bounds const& Runner::bounds() const { return impl_->bounds; }
Element Runner::input_element() const { return impl_->input_element; }
Element Runner::output_element() const { return impl_->output_element; }

std::string Runner::capture_failure() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->capture_failure;
}

void Runner::close() {
    std::exception_ptr first;
    impl_->teardown(&first);
    if (first) std::rethrow_exception(first);
}

void Runner::upscale(FrameJob const& job, HostPlane const* planes, HostPlane const* result, bool clip) {
    Impl& m = *impl_;
    if (job.scale != m.scale) throw std::invalid_argument("The frame was described for another engine.");
    Slot& s = m.acquire();
    // A frame that failed may leave work in flight: it ends before another frame reuses the
    // slot's buffers (its error is the one being raised, or a sticky one the next frame reports).
    struct Lease {
        Impl& m;
        Slot& s;
        ~Lease() {
            if (std::uncaught_exceptions()) cudaStreamSynchronize(s.stream);
            m.release(s);
        }
    } const lease{m, s};
    check(cudaSetDevice(m.device), "cudaSetDevice");

    int const out_w = job.width * job.scale, out_h = job.height * job.scale;
    int64_t const out_plane = static_cast<int64_t>(out_w) * out_h;
    bool const tiled = !job.places.empty();
    // The stream is idle here: every frame ends with a synchronize.
    s.source.grow(job.source.bytes);
    s.host_source.grow(job.source.bytes);
    s.result.grow(job.result.bytes);
    s.host_result.grow(job.result.bytes);
    if (tiled) {
        s.accumulator.grow(static_cast<std::size_t>(m.output_channels * out_plane) * sizeof(float));
        if (s.uploaded != job.weights) {
            std::size_t const bytes = job.weights.size() * sizeof(float);
            s.host_weights.grow(bytes);
            s.weights.grow(bytes);
            std::memcpy(s.host_weights.get(), job.weights.data(), bytes);
            check(cudaMemcpyAsync(s.weights.get(), s.host_weights.get(), bytes, cudaMemcpyHostToDevice, s.stream),
                  "cudaMemcpyAsync");
            s.uploaded = job.weights;
        }
    }
    pack(planes, job.planes, job.width, job.height, job.format.sample, job.source, static_cast<char*>(s.host_source.get()));
    check(cudaMemcpyAsync(s.source.get(), s.host_source.get(), job.source.bytes, cudaMemcpyHostToDevice, s.stream),
          "cudaMemcpyAsync");
    float* acc = static_cast<float*>(s.accumulator.get());
    if (tiled)
        check(cudaMemsetAsync(acc, 0, static_cast<std::size_t>(m.output_channels * out_plane) * sizeof(float), s.stream),
              "cudaMemsetAsync");

    Source const source{s.source.get(), job.format.sample, job.planes, job.width, job.height, job.source.layout, job.codes};
    Result const target{s.result.get(), job.format.sample, job.result_planes, out_w, out_h, job.result.layout, job.codes};
    float const* weights = static_cast<float const*>(s.weights.get());
    for (std::size_t i = 0; i < job.tiles.size(); ++i) {
        TileIn const& tile = job.tiles[i];
        m.prepare(s, tile.height, tile.width);
        check(launch_to_tensor(source, tile, s.input.get(), m.input_element, m.input_channels, clip, s.stream),
              "the input kernel");
        m.execute(s);
        int const tile_out_w = tile.width * job.scale;
        Planar const out{s.output.get(), m.output_element, m.output_channels, tile_out_w,
                         static_cast<int64_t>(tile_out_w) * tile.height * job.scale};
        if (tiled) {
            Place const& place = job.places[i];
            check(launch_accumulate(out, acc, out_w, out_plane, place.x0, place.y0, weights + place.wy, place.ny,
                                    weights + place.wx, place.nx, s.stream),
                  "the blend kernel");
        } else {
            check(launch_store(out, target, clip, s.stream), "the output kernel");
        }
    }
    if (tiled)
        check(launch_store({acc, Element::F32, m.output_channels, out_w, out_plane}, target, clip, s.stream),
              "the output kernel");
    check(cudaMemcpyAsync(s.host_result.get(), s.result.get(), job.result.bytes, cudaMemcpyDeviceToHost, s.stream),
          "cudaMemcpyAsync");
    check(cudaStreamSynchronize(s.stream), "cudaStreamSynchronize");
    unpack(static_cast<char const*>(s.host_result.get()), job.result, result, job.result_planes, out_w, out_h,
           job.format.sample);
}

}  // namespace cn_trt
