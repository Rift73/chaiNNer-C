// runner.cpp on the CPU over stand-ins of CUDA and TensorRT that fail on demand. Each scenario counts what a GPU would
// punish: an enqueue on a context whose shape or addresses are not the tile's, a freed buffer in use, work after a
// failure, a leaked context.
#include "runner.h"

#include "kernels.h"

#include <NvInferRuntime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using namespace cn_trt;

namespace {
struct Fake {
    std::size_t fail_bytes = 0;
    bool fail_enqueue = false;
    bool fail_sync = false;
    bool fail_stream_create = false;
    cudaError_t last = cudaSuccess;
    std::map<char*, std::size_t> live;
    int enqueues = 0;
    int live_contexts = 0;
    int tile_h = 0, tile_w = 0;
    int event_waits = 0;
    int capturing = 0;
    std::vector<std::string> bad;
} fake;

void bad(std::string const& what) {
    if (fake.bad.size() < 4) fake.bad.push_back(what);
}

bool is_live(void const* p) {
    auto const it = fake.live.upper_bound(static_cast<char*>(const_cast<void*>(p)));
    if (it == fake.live.begin()) return false;
    auto const block = std::prev(it);
    return static_cast<char const*>(p) < block->first + block->second;
}

class Context final : public nvinfer1::IExecutionContext {
public:
    Context() { ++fake.live_contexts; }
    ~Context() override { --fake.live_contexts; }
    bool setInputShape(char const*, nvinfer1::Dims const& d) override {
        h = static_cast<int>(d.d[2]);
        w = static_cast<int>(d.d[3]);
        return true;
    }
    nvinfer1::Dims getTensorShape(char const*) const override {
        return nvinfer1::Dims4(1, 3, h * 2, w * 2);
    }
    std::size_t updateDeviceMemorySizeForShapes() override { return static_cast<std::size_t>(h) * w * 7 + 13; }
    void setDeviceMemoryV2(void* p, int64_t size) override {
        activation = p;
        activation_size = size;
    }
    bool setTensorAddress(char const* name, void* p) override {
        (std::strcmp(name, "input") == 0 ? input : output) = p;
        return true;
    }
    bool enqueueV3(cudaStream_t) override {
        ++fake.enqueues;
        if (fake.fail_enqueue) {
            fake.fail_enqueue = false;
            return false;
        }
        if (h != fake.tile_h || w != fake.tile_w)
            bad("enqueue at shape " + std::to_string(w) + "x" + std::to_string(h) + " for a tile of " +
                std::to_string(fake.tile_w) + "x" + std::to_string(fake.tile_h));
        if (!is_live(input) || !is_live(output)) bad("enqueue with a freed tensor address");
        if (!is_live(activation)) bad("enqueue with freed activation memory");
        return true;
    }
    int h = 0, w = 0;
    void* input = nullptr;
    void* output = nullptr;
    void* activation = nullptr;
    int64_t activation_size = 0;
};

class Engine final : public nvinfer1::ICudaEngine {
public:
    int getNbIOTensors() const override { return 2; }
    char const* getIOTensorName(int i) const override { return i ? "output" : "input"; }
    nvinfer1::TensorIOMode getTensorIOMode(char const* name) const override {
        return std::strcmp(name, "input") == 0 ? nvinfer1::TensorIOMode::kINPUT : nvinfer1::TensorIOMode::kOUTPUT;
    }
    nvinfer1::Dims getTensorShape(char const*) const override { return nvinfer1::Dims4(1, 3, -1, -1); }
    nvinfer1::DataType getTensorDataType(char const*) const override { return nvinfer1::DataType::kFLOAT; }
    nvinfer1::Dims getProfileShape(char const*, int, nvinfer1::OptProfileSelector s) const override {
        int const size = s == nvinfer1::OptProfileSelector::kMIN ? 64 : 128;
        return nvinfer1::Dims4(1, 3, size, size);
    }
    nvinfer1::IExecutionContext* createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy) override {
        return new Context;
    }
};

class Runtime final : public nvinfer1::IRuntime {
public:
    nvinfer1::ICudaEngine* deserializeCudaEngine(void const*, std::size_t) override { return new Engine; }
};

cudaError_t fail(cudaError_t error) {
    fake.last = error;
    return error;
}

template <typename T>
T token() {
    static std::uintptr_t next = 16;
    next += 16;
    return reinterpret_cast<T>(next);
}
}  // namespace

nvinfer1::IRuntime* nvinfer1::createInferRuntime(ILogger&) { return new Runtime; }
bool initLibNvInferPlugins(void*, char const*) { return true; }
int32_t getInferLibVersion() { return 110400; }
char const* cudaGetErrorName(cudaError_t e) { return e == cudaErrorMemoryAllocation ? "cudaErrorMemoryAllocation" : e ? "cudaErrorUnknown" : "cudaSuccess"; }
char const* cudaGetErrorString(cudaError_t) { return "fake"; }
cudaError_t cudaGetLastError() {
    cudaError_t const e = fake.last;
    fake.last = cudaSuccess;
    return e;
}
cudaError_t cudaSetDevice(int) { return cudaSuccess; }
cudaError_t cudaMalloc(void** p, std::size_t bytes) {
    if (bytes == fake.fail_bytes) return fail(cudaErrorMemoryAllocation);
    *p = std::malloc(bytes);
    fake.live[static_cast<char*>(*p)] = bytes;
    return cudaSuccess;
}
cudaError_t cudaFree(void* p) {
    fake.live.erase(static_cast<char*>(p));
    std::free(p);
    return cudaSuccess;
}
cudaError_t cudaMallocHost(void** p, std::size_t bytes) {
    *p = std::malloc(bytes);
    return cudaSuccess;
}
cudaError_t cudaFreeHost(void* p) {
    std::free(p);
    return cudaSuccess;
}
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* s, unsigned) {
    if (fake.fail_stream_create) return fail(cudaErrorUnknown);
    *s = token<cudaStream_t>();
    return cudaSuccess;
}
cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
cudaError_t cudaStreamSynchronize(cudaStream_t) {
    if (fake.fail_sync) {
        fake.fail_sync = false;
        return fail(cudaErrorUnknown);
    }
    return cudaSuccess;
}
cudaError_t cudaMemcpyAsync(void*, void const*, std::size_t, cudaMemcpyKind, cudaStream_t) { return cudaSuccess; }
cudaError_t cudaMemsetAsync(void*, int, std::size_t, cudaStream_t) { return cudaSuccess; }
cudaError_t cudaGraphLaunch(cudaGraphExec_t, cudaStream_t) { return cudaSuccess; }
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t) { return cudaSuccess; }
cudaError_t cudaGraphDestroy(cudaGraph_t) { return cudaSuccess; }
cudaError_t cudaGraphInstantiate(cudaGraphExec_t* e, cudaGraph_t, unsigned) {
    *e = token<cudaGraphExec_t>();
    return cudaSuccess;
}
cudaError_t cudaStreamBeginCapture(cudaStream_t, cudaStreamCaptureMode) { return cudaSuccess; }
cudaError_t cudaStreamEndCapture(cudaStream_t, cudaGraph_t* g) {
    *g = token<cudaGraph_t>();
    return cudaSuccess;
}
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e, unsigned) {
    *e = token<cudaEvent_t>();
    return cudaSuccess;
}
cudaError_t cudaEventDestroy(cudaEvent_t) { return cudaSuccess; }
cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t) { return cudaSuccess; }
cudaError_t cudaEventSynchronize(cudaEvent_t) {
    ++fake.event_waits;
    return cudaSuccess;
}

namespace cn_trt {
cudaError_t launch_to_tensor(Source const& source, TileIn const& tile, void* tensor, Element, int, bool, cudaStream_t) {
    fake.tile_h = tile.height;
    fake.tile_w = tile.width;
    if (!is_live(tensor) || !is_live(source.data)) bad("input kernel on a freed buffer");
    return cudaGetLastError();
}
cudaError_t launch_accumulate(Planar const& tile, float* acc, int, int64_t, int, int, float const* wy, int, float const* wx, int, cudaStream_t) {
    if (!is_live(tile.data) || !is_live(acc)) bad("blend kernel on a freed buffer");
    if (!is_live(wy) || !is_live(wx)) bad("blend kernel on freed weights");
    return cudaGetLastError();
}
cudaError_t launch_store(Planar const& source, Result const& result, bool, cudaStream_t) {
    if (!is_live(source.data) || !is_live(result.data)) bad("output kernel on a freed buffer");
    return cudaGetLastError();
}
}  // namespace cn_trt

namespace {
struct Frame {
    std::vector<float> in, out;
    HostPlane pin[3], pout[3];
    FrameJob job;
    Frame(Runner& r, int w, int h) : in(3u * w * h, 0.5f), out(12u * w * h, 0.0f) {
        for (int p = 0; p < 3; ++p) {
            pin[p] = {in.data() + p * w * h, static_cast<std::ptrdiff_t>(w) * 4, 4};
            pout[p] = {out.data() + p * 4 * w * h, static_cast<std::ptrdiff_t>(w) * 8, 4};
        }
        job = describe_frame(pin, 3, pout, 3, w, h, Format{Sample::F32, 32, true}, r.bounds(), r.scale(), 3, 3);
    }
    void run(Runner& r) { r.upscale(job, pin, pout, false); }
};

std::string attempt(Runner& r, Frame& f) {
    try {
        f.run(r);
        return "";
    } catch (std::exception const& e) {
        return e.what();
    }
}

int failures = 0;
void verdict(char const* name, std::string const& detail = "") {
    bool const ok = fake.bad.empty() && detail.empty();
    failures += !ok;
    std::printf("%s %s", ok ? "PASS" : "FAIL", name);
    if (!detail.empty()) std::printf(": %s", detail.c_str());
    for (auto const& b : fake.bad) std::printf(" [%s]", b.c_str());
    std::printf("\n");
    fake = Fake{};
}
char const engine_bytes[] = "engine";
}  // namespace

int main() {
    {  // A failed allocation's error stays on the thread; the next frame is good.
        Runner r(engine_bytes, 6, 0, 1);
        Frame f(r, 64, 64);
        fake.fail_bytes = f.job.source.bytes;
        std::string const first = attempt(r, f);
        fake.fail_bytes = 0;
        std::string const second = attempt(r, f);
        verdict("the frame after a failed allocation is good", first.empty() ? "no failure" : second);
    }
    fake = Fake{};
    {  // A growth fails inside prepare; a frame of the shape before it runs.
        Runner r(engine_bytes, 6, 0, 1);
        Frame a(r, 64, 64), b(r, 128, 128);
        std::string detail = attempt(r, a);
        fake.fail_bytes = 128u * 128u * 7u + 13u;
        if (attempt(r, b).empty()) detail += "no failure";
        fake.fail_bytes = 0;
        detail += attempt(r, a);
        verdict("after a growth that failed, the shape before it is set again", detail);
    }
    fake = Fake{};
    {  // The weights' growth fails; a frame of the weights it freed runs.
        Runner r(engine_bytes, 6, 0, 1);
        Frame a(r, 300, 100), b(r, 600, 100);
        std::string detail = attempt(r, a);
        fake.fail_bytes = b.job.weights.size() * sizeof(float);
        if (attempt(r, b).empty()) detail += "no failure";
        fake.fail_bytes = 0;
        detail += attempt(r, a);
        verdict("after a weights growth that failed, the weights go up again", detail);
    }
    fake = Fake{};
    {  // An enqueue fails: nothing more goes on the context.
        Runner r(engine_bytes, 6, 0, 1);
        Frame f(r, 64, 64);
        fake.fail_enqueue = true;
        std::string const first = attempt(r, f);
        int const enqueues = fake.enqueues;
        std::string const second = attempt(r, f);
        verdict("no enqueue after a failed enqueue",
                first.empty() || fake.enqueues != enqueues
                    ? "enqueues " + std::to_string(enqueues) + " -> " + std::to_string(fake.enqueues)
                    : "");
    }
    fake = Fake{};
    {  // A wait fails at the frame's end: nothing more goes on the stream.
        Runner r(engine_bytes, 6, 0, 1);
        Frame f(r, 64, 64);
        fake.fail_sync = true;
        std::string const first = attempt(r, f);
        int const enqueues = fake.enqueues;
        std::string const second = attempt(r, f);
        verdict("no enqueue after a failed wait",
                first.empty() || fake.enqueues != enqueues
                    ? "enqueues " + std::to_string(enqueues) + " -> " + std::to_string(fake.enqueues)
                    : "");
    }
    fake = Fake{};
    {  // An allocation that fails is no failure of the stream: the session goes on.
        Runner r(engine_bytes, 6, 0, 1);
        Frame f(r, 64, 64);
        fake.fail_bytes = f.job.result.bytes;
        attempt(r, f);
        fake.fail_bytes = 0;
        verdict("a failed allocation does not end the session", attempt(r, f));
    }
    fake = Fake{};
    {  // A runner that fails while it is made frees its context.
        fake.fail_stream_create = true;
        bool threw = false;
        try {
            Runner r(engine_bytes, 6, 0, 1);
        } catch (std::exception const&) {
            threw = true;
        }
        verdict("a runner that fails while it is made leaves no context",
                !threw ? "no failure" : fake.live_contexts ? std::to_string(fake.live_contexts) + " context(s) leaked" : "");
    }
    fake = Fake{};

    {  // The hook sees each tile, at most two ahead; what it throws ends the frame, and the next frame is good.
        Runner r(engine_bytes, 6, 0, 1);
        Frame f(r, 300, 200);
        std::size_t const tiles = f.job.tiles.size();
        std::vector<std::size_t> seen;
        r.upscale(f.job, f.pin, f.pout, false, [&](std::size_t done, std::size_t total) {
            if (total != tiles) bad("wrong tile count");
            seen.push_back(done);
        });
        std::string detail;
        if (seen.size() != tiles) detail += "hook calls " + std::to_string(seen.size());
        if (fake.event_waits != static_cast<int>(tiles) - 2) detail += " event waits " + std::to_string(fake.event_waits);
        try {
            r.upscale(f.job, f.pin, f.pout, false, [](std::size_t done, std::size_t) {
                if (done == 1) throw std::domain_error("stopped");
            });
            detail += " not stopped";
        } catch (std::domain_error const&) {
        }
        detail += attempt(r, f);
        verdict("the hook sees each tile and can stop the frame", detail);
    }

    std::printf("%d failing\n", failures);
    return failures != 0;
}
