#include "runner.h"

#include "kernels.h"
#include "program.h"

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace cn_migx {
namespace {

// dual.ALIGNED_INPUT: a DUAL model's input name carries the multiple its sizes must be (as the TensorRT runner reads
// it from an engine).
constexpr char aligned_input[] = "input_aligned_";

std::string hip_message(hipError_t error, char const* call) {
    return std::string("HIP error ") + hipGetErrorName(error) + " in " + call + ": " + hipGetErrorString(error);
}

int alignment_of(std::string const& name) {
    std::size_t const prefix = sizeof aligned_input - 1;
    if (name.compare(0, prefix, aligned_input) != 0 || name.size() == prefix ||
        !std::all_of(name.begin() + prefix, name.end(), [](char c) { return c >= '0' && c <= '9'; }))
        return cn_trt::tile_alignment;
    return std::stoi(name.substr(prefix));
}

Element element_of(migraphx::shape::type_t type) {
    switch (type) {
    case migraphx::shape::float_type: return Element::F32;
    case migraphx::shape::half_type: return Element::F16;
    case migraphx::shape::bf16_type: return Element::BF16;
    default:
        throw std::invalid_argument("The MIGraphX runner takes float32, float16 or bfloat16 tensors, not " +
                                    type_name(type) + ".");
    }
}

// A grow-only device or pinned host buffer.
template <hipError_t (*Allocate)(void**, std::size_t), hipError_t (*Free)(void*)>
class Buffer {
public:
    Buffer() = default;
    Buffer(Buffer const&) = delete;
    Buffer& operator=(Buffer const&) = delete;
    ~Buffer() { (void)release(); }
    void grow(std::size_t bytes) {
        if (bytes <= size_) return;
        check_hip(release(), "freeing a buffer");
        check_hip(Allocate(&ptr_, bytes), "allocating a buffer");
        size_ = bytes;
    }
    hipError_t release() noexcept {
        hipError_t const error = ptr_ ? Free(ptr_) : hipSuccess;
        ptr_ = nullptr;
        size_ = 0;
        return error;
    }
    void* get() const { return ptr_; }

private:
    void* ptr_ = nullptr;
    std::size_t size_ = 0;
};

hipError_t device_allocate(void** ptr, std::size_t bytes) { return hipMalloc(ptr, bytes); }
hipError_t device_free(void* ptr) { return hipFree(ptr); }
hipError_t host_allocate(void** ptr, std::size_t bytes) { return hipHostMalloc(ptr, bytes, hipHostMallocDefault); }
hipError_t host_free(void* ptr) { return hipHostFree(ptr); }
using DeviceBuffer = Buffer<device_allocate, device_free>;
using PinnedBuffer = Buffer<host_allocate, host_free>;

// One copy of the program with its stream and buffers; a frame holds it from upload to download.
struct Slot {
    std::optional<Program> program;
    hipStream_t stream = nullptr;
    DeviceBuffer source, input, output, accumulator, result, weights;
    PinnedBuffer host_source, host_result, host_weights;
    std::vector<float> uploaded;  // the blend weights on the device
};

}  // namespace

struct Runner::Impl {
    int device = 0;
    std::string input_name;
    Element input_element = Element::F32, output_element = Element::F32;
    int input_channels = 0, output_channels = 0, scale = 0;
    Bounds bounds{};
    std::size_t input_bytes = 0, output_bytes = 0;
    std::vector<std::unique_ptr<Slot>> slots;
    std::mutex mutex;
    std::condition_variable freed;
    std::vector<Slot*> idle;
    bool closed = false;

    ~Impl() { teardown(nullptr); }

    // Frees everything: every step runs; the first error goes to `first` (a failed construction passes none, so a
    // teardown error cannot hide its own).
    void teardown(std::exception_ptr* first) noexcept {
        auto attempt = [first](auto&& step) {
            try {
                step();
            } catch (...) {
                if (first && !*first) *first = std::current_exception();
            }
        };
        auto hip = [&](hipError_t error, char const* call) {
            attempt([&] {
                if (error != hipSuccess) throw std::runtime_error(hip_message(error, call));
            });
        };
        if (slots.empty()) return;
        hip(hipSetDevice(device), "hipSetDevice");
        for (auto& slot : slots)
            if (slot->stream) hip(hipStreamSynchronize(slot->stream), "hipStreamSynchronize");
        for (auto& slot : slots) {
            if (slot->program) attempt([&] { slot->program->close(); });
            slot->program.reset();
            for (PinnedBuffer* buffer : {&slot->host_source, &slot->host_result, &slot->host_weights})
                hip(buffer->release(), "hipHostFree");
            for (DeviceBuffer* buffer :
                 {&slot->source, &slot->input, &slot->output, &slot->accumulator, &slot->result, &slot->weights})
                hip(buffer->release(), "hipFree");
            if (slot->stream) hip(hipStreamDestroy(slot->stream), "hipStreamDestroy");
            slot->stream = nullptr;
        }
        slots.clear();
        idle.clear();
    }

    Slot& acquire() {
        std::unique_lock<std::mutex> lock(mutex);
        if (closed) throw std::runtime_error("The MIGraphX runner is closed.");
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
};

Runner::Runner(void const* program, std::size_t size, int device, int streams) : impl_(std::make_unique<Impl>()) {
    if (size == 0) throw std::invalid_argument("The MIGraphX program is empty.");
    if (device < 0) throw std::invalid_argument("The GPU index must be 0 or more, not " + std::to_string(device) + ".");
    if (streams < 1 || streams > cn_trt::max_streams)
        throw std::invalid_argument("The stream count must be 1 to " + std::to_string(cn_trt::max_streams) + ", not " +
                                    std::to_string(streams) + ".");
    Impl& m = *impl_;
    m.device = device;
    for (int i = 0; i < streams; ++i) {
        auto slot = std::make_unique<Slot>();
        slot->program.emplace(Program::load(static_cast<char const*>(program), size, device));
        m.slots.push_back(std::move(slot));
    }
    Program const& p = *m.slots[0]->program;
    if (p.target() != "gpu") throw std::invalid_argument("The MIGraphX runner takes GPU programs, not " + p.target() + " ones.");
    if (p.input_names().size() != 1 || p.output_shapes().size() != 1)
        throw std::invalid_argument("Expected one input and one output tensor, got " +
                                    std::to_string(p.input_names().size()) + " and " +
                                    std::to_string(p.output_shapes().size()) + ".");
    migraphx::shape const& in = p.input_shapes()[0];
    migraphx::shape const& out = p.output_shapes()[0];
    if (in.dynamic() || out.dynamic())
        throw std::invalid_argument("The MIGraphX runner takes programs of a fixed input size: compile the model "
                                    "with input_dims.");
    if (in.lens().size() != 4 || out.lens().size() != 4 || in.lens()[0] != 1 || out.lens()[0] != 1)
        throw std::invalid_argument("Expected NCHW input and output tensors of one image.");
    m.input_name = p.input_names()[0];
    m.input_element = element_of(in.type());
    m.output_element = element_of(out.type());
    m.input_channels = static_cast<int>(in.lens()[1]);
    m.output_channels = static_cast<int>(out.lens()[1]);
    std::size_t const h = in.lens()[2], w = in.lens()[3];
    if (out.lens()[2] % h || out.lens()[3] % w || out.lens()[2] / h != out.lens()[3] / w)
        throw std::invalid_argument("The program's output is not a uniform integer upscale of its input.");
    m.scale = static_cast<int>(out.lens()[2] / h);
    if ((m.input_channels != 1 && m.input_channels != 3) || (m.output_channels != 1 && m.output_channels != 3))
        throw std::invalid_argument("The MIGraphX runner takes programs with 1 or 3 input and output channels, not " +
                                    std::to_string(m.input_channels) + " -> " + std::to_string(m.output_channels) +
                                    ".");
    m.bounds = {static_cast<int>(h), static_cast<int>(w), static_cast<int>(h), static_cast<int>(w),
                alignment_of(m.input_name)};
    m.input_bytes = in.bytes();
    m.output_bytes = out.bytes();
    check_hip(hipSetDevice(device), "hipSetDevice");
    for (auto& slot : m.slots) {
        check_hip(hipStreamCreateWithFlags(&slot->stream, hipStreamNonBlocking), "hipStreamCreateWithFlags");
        slot->input.grow(m.input_bytes);
        slot->output.grow(m.output_bytes);
        m.idle.push_back(slot.get());
    }
}

Runner::~Runner() = default;

int Runner::input_channels() const { return impl_->input_channels; }
int Runner::output_channels() const { return impl_->output_channels; }
int Runner::scale() const { return impl_->scale; }
Bounds const& Runner::bounds() const { return impl_->bounds; }
Element Runner::input_element() const { return impl_->input_element; }
Element Runner::output_element() const { return impl_->output_element; }

void Runner::close() {
    std::exception_ptr first;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->closed = true;
    }
    impl_->teardown(&first);
    if (first) std::rethrow_exception(first);
}

void Runner::upscale(FrameJob const& job, HostPlane const* planes, HostPlane const* result, bool clip) {
    Impl& m = *impl_;
    if (job.scale != m.scale) throw std::invalid_argument("The frame was described for another program.");
    Slot& s = m.acquire();
    // A frame that failed may leave work in flight: it ends before another frame reuses the slot's buffers.
    struct Lease {
        Impl& m;
        Slot& s;
        ~Lease() {
            if (std::uncaught_exceptions()) (void)hipStreamSynchronize(s.stream);
            m.release(s);
        }
    } const lease{m, s};
    check_hip(hipSetDevice(m.device), "hipSetDevice");

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
            check_hip(hipMemcpyAsync(s.weights.get(), s.host_weights.get(), bytes, hipMemcpyHostToDevice, s.stream),
                      "hipMemcpyAsync");
            s.uploaded = job.weights;
        }
    }
    cn_trt::pack(planes, job.planes, job.width, job.height, job.format.sample, job.source,
                 static_cast<char*>(s.host_source.get()));
    check_hip(hipMemcpyAsync(s.source.get(), s.host_source.get(), job.source.bytes, hipMemcpyHostToDevice, s.stream),
              "hipMemcpyAsync");
    float* acc = static_cast<float*>(s.accumulator.get());
    if (tiled)
        check_hip(hipMemsetAsync(acc, 0, static_cast<std::size_t>(m.output_channels * out_plane) * sizeof(float),
                                 s.stream),
                  "hipMemsetAsync");

    cn_trt::Source const source{s.source.get(), job.format.sample, job.planes, job.width, job.height,
                                job.source.layout, job.codes};
    cn_trt::Result const target{s.result.get(), job.format.sample, job.result_planes, out_w, out_h,
                                job.result.layout, job.codes};
    float const* weights = static_cast<float const*>(s.weights.get());
    std::map<std::string, void*> const inputs = {{m.input_name, s.input.get()}};
    std::vector<void*> const outputs = {s.output.get()};
    for (std::size_t i = 0; i < job.tiles.size(); ++i) {
        cn_trt::TileIn const& tile = job.tiles[i];
        if (tile.height != m.bounds.max_h || tile.width != m.bounds.max_w)
            throw std::logic_error("A tile does not have the program's input size.");
        check_hip(launch_to_tensor(source, tile, s.input.get(), m.input_element, m.input_channels, clip, s.stream),
                  "the input kernel");
        // The program waits for the stream and the stream for the program (MIGraphX's asynchronous run).
        s.program->run(inputs, outputs, s.stream);
        int const tile_out_w = tile.width * job.scale;
        cn_trt::Planar const out{s.output.get(), m.output_element, m.output_channels, tile_out_w,
                                 static_cast<int64_t>(tile_out_w) * tile.height * job.scale};
        if (tiled) {
            cn_trt::Place const& place = job.places[i];
            check_hip(launch_accumulate(out, acc, out_w, out_plane, place.x0, place.y0, weights + place.wy, place.ny,
                                        weights + place.wx, place.nx, s.stream),
                      "the blend kernel");
        } else {
            check_hip(launch_store(out, target, clip, s.stream), "the output kernel");
        }
    }
    if (tiled)
        check_hip(launch_store({acc, Element::F32, m.output_channels, out_w, out_plane}, target, clip, s.stream),
                  "the output kernel");
    check_hip(hipMemcpyAsync(s.host_result.get(), s.result.get(), job.result.bytes, hipMemcpyDeviceToHost, s.stream),
              "hipMemcpyAsync");
    check_hip(hipStreamSynchronize(s.stream), "hipStreamSynchronize");
    cn_trt::unpack(static_cast<char const*>(s.host_result.get()), job.result, result, job.result_planes, out_w, out_h,
                   job.format.sample);
}

}  // namespace cn_migx
