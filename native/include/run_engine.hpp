/* The native run engine's node ABI: the contract between the engine (native/src/run/), native node entry points
 * (native/src/run/nodes/) and the native codecs. A node with an entry point here runs on an engine thread without
 * the GIL; every other node runs its Python body through the engine's Python boundary.
 *
 * Values reach an entry point enforced exactly as the node's Python body receives them (the engine implements each
 * input kind's enforce once). An entry point must produce the same outputs, bit for bit, and the same error text and
 * Python exception type as the node's Python body; image outputs are normalized by the engine exactly as
 * ImageOutput.enforce does. */
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace cn::run {

class ImageStorage; // engine-owned, pooled storage shared by Images and the NumPy views handed to Python

/* A published, read-only float32 image: height x width x channels, channels in {1, 3, 4}, rows contiguous
 * (row stride = width * channels floats). Python sees channels == 1 as a 2-D (height, width) array. */
struct Image {
    std::shared_ptr<const ImageStorage> storage;
    const float *data = nullptr;
    int32_t height = 0;
    int32_t width = 0;
    int32_t channels = 0;

    size_t values() const noexcept { return size_t(height) * size_t(width) * size_t(channels); }
};

/* An image being written. allocate() returns uninitialised pooled memory; publish() freezes it. */
class MutableImage {
public:
    static MutableImage allocate(int32_t height, int32_t width, int32_t channels);
    float *data() noexcept { return data_; }
    int32_t height() const noexcept { return height_; }
    int32_t width() const noexcept { return width_; }
    int32_t channels() const noexcept { return channels_; }
    Image publish() &&;

private:
    std::shared_ptr<ImageStorage> storage_;
    float *data_ = nullptr;
    int32_t height_ = 0, width_ = 0, channels_ = 0;
};

/* A Python object the engine keeps alive (touching it needs the GIL; native entry points only pass it on). */
struct PyRef {
    std::shared_ptr<void> handle;
};

struct Path {
    std::filesystem::path value;
};

/* A colour input (chaiNNer's Color): its channel values in the order the Python object stores them. */
struct Color {
    std::vector<double> values;
};

/* A seed input (nodes.utils.seed.Seed) whose value fits int64; a larger seed stays a PyRef. */
struct Seed {
    int64_t value;
};

class LazyValue;

/* One input or output value. Enum (dropdown) inputs arrive as their option value (int64_t or std::string). */
using Value = std::variant<std::monostate, int64_t, double, bool, std::string, Path, Color, Seed, Image, PyRef,
                           std::shared_ptr<LazyValue>>;

/* A lazy input: value() computes the upstream on first read (blocking without the GIL) and may throw its error. */
class LazyValue {
public:
    virtual ~LazyValue() = default;
    virtual const Value &value() = 0;
};

/* A node failure: message and Python exception type name become the NodeExecutionError cause, as the Python body
 * would raise them (e.g. {"ValueError", "..."}). */
struct NodeError : std::runtime_error {
    NodeError(std::string python_type, const std::string &message)
        : std::runtime_error(message), python_type(std::move(python_type)) {}
    std::string python_type;
};

/* Raised by NodeContext::check_aborted after Stop. */
struct Aborted {};

/* Thrown by an entry point, before it produces any output, when its Python body would take a path the entry point
 * does not mirror (an OpenCV/IPP or NumPy primitive, an integer beyond int64, an output size beyond int32). The
 * engine then runs the node's Python body on the same enforced inputs and counts the decline per schema;
 * run_native raises _chainner_graph.NativeDeclined. */
struct RunPython {};

class NodeContext {
public:
    virtual ~NodeContext() = default;
    virtual bool aborted() const noexcept = 0;
    virtual void check_aborted() const = 0; // throws Aborted
    virtual std::string_view node_id() const noexcept = 0;
    virtual const std::filesystem::path &storage_dir() const noexcept = 0;
    /* The node's package setting for this run, by key; nullptr when unset. */
    virtual const Value *setting(std::string_view key) const noexcept = 0;
};

/* A regular node: outputs in schema order (the engine checks the count). */
using NativeRun = void (*)(NodeContext &context, std::span<const Value> inputs, std::vector<Value> &outputs);

/* A generator's item source. describe() runs serially in item order and returns false at the end;
 * materialize() is const and thread-safe, may run on any engine thread, and fills the item's iterated outputs in
 * schema order. A NodeError thrown by materialize() is that item's error (deferred unless the generator fails
 * fast). */
class ItemSource {
public:
    virtual ~ItemSource() = default;
    virtual int64_t expected_length() const noexcept = 0;
    virtual bool fail_fast() const noexcept { return true; }
    virtual bool describe(int64_t &token) = 0;
    virtual void materialize(int64_t token, std::vector<Value> &item) const = 0;
    /* The files each item reads (the read-ahead check); false when unknown. */
    virtual bool source_paths(std::vector<std::filesystem::path> &paths) const { (void)paths; return false; }
};

/* A generator node: returns its item source and fills its static (non-iterated) outputs in schema order. */
using NativeGeneratorRun = std::unique_ptr<ItemSource> (*)(NodeContext &context, std::span<const Value> inputs,
                                                           std::vector<Value> &static_outputs);

class Registry {
public:
    virtual ~Registry() = default;
    virtual void add(std::string_view schema_id, NativeRun run) = 0;
    virtual void add_generator(std::string_view schema_id, NativeGeneratorRun run) = 0;
};

} // namespace cn::run

/* Each file native/src/run/nodes/<group>.cpp defines `void cn_run_nodes_<group>(cn::run::Registry &)` and adds one
 * line `CN_RUN_NODES(<group>)` to native/src/run/nodes/all_nodes.inc. */
#define CN_RUN_NODES_DECLARE(group) void cn_run_nodes_##group(cn::run::Registry &registry)
