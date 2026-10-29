#include "ops.h"

#include <hip/hip_runtime_api.h>
#include <migraphx/argument.hpp>
#include <migraphx/context.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/reflect.hpp>
#include <migraphx/register_op.hpp>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace cn_migx {
namespace {

std::mutex registry_mutex;
// Loaded ops by name, and each loaded library's canonical path with its ops' names.
std::map<std::string, chainner_migx_op const*> loaded_ops;
std::map<std::string, std::vector<std::string>> loaded_libraries;

constexpr std::pair<migraphx::shape::type_t, chainner_migx_type> types[] = {
    {migraphx::shape::float_type, CHAINNER_MIGX_FLOAT},
    {migraphx::shape::half_type, CHAINNER_MIGX_HALF},
    {migraphx::shape::bf16_type, CHAINNER_MIGX_BF16},
    {migraphx::shape::double_type, CHAINNER_MIGX_DOUBLE},
    {migraphx::shape::int8_type, CHAINNER_MIGX_INT8},
    {migraphx::shape::uint8_type, CHAINNER_MIGX_UINT8},
    {migraphx::shape::int16_type, CHAINNER_MIGX_INT16},
    {migraphx::shape::uint16_type, CHAINNER_MIGX_UINT16},
    {migraphx::shape::int32_type, CHAINNER_MIGX_INT32},
    {migraphx::shape::uint32_type, CHAINNER_MIGX_UINT32},
    {migraphx::shape::int64_type, CHAINNER_MIGX_INT64},
    {migraphx::shape::uint64_type, CHAINNER_MIGX_UINT64},
    {migraphx::shape::bool_type, CHAINNER_MIGX_BOOL},
    {migraphx::shape::fp8e4m3fnuz_type, CHAINNER_MIGX_FP8E4M3FNUZ},
    {migraphx::shape::fp8e4m3fn_type, CHAINNER_MIGX_FP8E4M3FN},
    {migraphx::shape::fp8e5m2_type, CHAINNER_MIGX_FP8E5M2},
    {migraphx::shape::fp8e5m2fnuz_type, CHAINNER_MIGX_FP8E5M2FNUZ},
};

chainner_migx_shape shape_of(migraphx::shape const& s, std::string const& op) {
    if (s.dynamic()) throw std::runtime_error(op + " takes inputs of fixed sizes: compile the model with input_dims.");
    if (!s.standard()) throw std::runtime_error(op + " takes packed inputs.");
    auto const type = std::find_if(std::begin(types), std::end(types), [&](auto const& t) { return t.first == s.type(); });
    if (type == std::end(types)) throw std::runtime_error(op + " does not take " + s.type_string() + " tensors.");
    if (s.lens().size() > CHAINNER_MIGX_MAX_RANK)
        throw std::runtime_error(op + " takes tensors of rank " + std::to_string(CHAINNER_MIGX_MAX_RANK) +
                                 " at most, not " + std::to_string(s.lens().size()) + ".");
    chainner_migx_shape result{};
    result.type = type->second;
    result.rank = static_cast<int32_t>(s.lens().size());
    for (std::size_t i = 0; i < s.lens().size(); ++i) result.dims[i] = static_cast<int64_t>(s.lens()[i]);
    return result;
}

migraphx::shape migraphx_shape(chainner_migx_shape const& s, std::string const& op) {
    auto const type = std::find_if(std::begin(types), std::end(types), [&](auto const& t) { return t.second == s.type; });
    if (type == std::end(types) || s.rank < 0 || s.rank > CHAINNER_MIGX_MAX_RANK)
        throw std::runtime_error(op + " gave an output shape of an unknown type or rank.");
    std::vector<std::size_t> lens;
    for (int32_t i = 0; i < s.rank; ++i) {
        if (s.dims[i] < 1) throw std::runtime_error(op + " gave an empty output dimension.");
        lens.push_back(static_cast<std::size_t>(s.dims[i]));
    }
    return {type->first, lens};
}

// The encoding: per attribute its name's length (uint32, the NUL included), the name, its kind (int32), its count
// (int64), then the values (int64 or float) or the bytes and a NUL, every part starting 8-byte aligned, so the
// attributes are read in place.
constexpr std::size_t align8(std::size_t n) { return (n + 7) / 8 * 8; }

void put(migraphx::value::binary& out, void const* data, std::size_t bytes) {
    std::size_t const at = out.size();
    out.resize(align8(at + bytes));
    if (bytes) std::memcpy(out.data() + at, data, bytes);
}

// The attributes of one encoding, pointing into it: valid while it lives.
std::vector<chainner_migx_attribute> attribute_views(migraphx::value::binary const& encoded) {
    std::vector<chainner_migx_attribute> views;
    std::size_t at = 0;
    auto take = [&](std::size_t bytes) {
        if (encoded.size() - at < bytes) throw std::runtime_error("A custom op's encoded attributes are damaged.");
        unsigned char const* data = encoded.data() + at;
        at = align8(at + bytes);
        return data;
    };
    while (at < encoded.size()) {
        chainner_migx_attribute view{};
        uint32_t name_bytes;
        std::memcpy(&name_bytes, take(sizeof name_bytes), sizeof name_bytes);
        view.name = reinterpret_cast<char const*>(take(name_bytes));
        int32_t kind_count[2];
        std::memcpy(kind_count, take(sizeof kind_count), sizeof kind_count);
        view.kind = kind_count[0];
        std::memcpy(&view.count, take(sizeof view.count), sizeof view.count);
        if (view.count < 0) throw std::runtime_error("A custom op's encoded attributes are damaged.");
        auto const count = static_cast<std::size_t>(view.count);
        if (view.kind == CHAINNER_MIGX_ATTRIBUTE_INT || view.kind == CHAINNER_MIGX_ATTRIBUTE_INTS)
            view.ints = reinterpret_cast<int64_t const*>(take(count * sizeof(int64_t)));
        else if (view.kind == CHAINNER_MIGX_ATTRIBUTE_FLOAT || view.kind == CHAINNER_MIGX_ATTRIBUTE_FLOATS)
            view.floats = reinterpret_cast<float const*>(take(count * sizeof(float)));
        else
            view.bytes = reinterpret_cast<char const*>(take(count + 1));
        views.push_back(view);
    }
    return views;
}

std::vector<chainner_migx_shape> shapes_of(std::vector<migraphx::shape> const& inputs, std::string const& op) {
    std::vector<chainner_migx_shape> shapes;
    for (auto const& s : inputs) shapes.push_back(shape_of(s, op));
    return shapes;
}

chainner_migx_op const& loaded(std::string const& name) {
    chainner_migx_op const* op = find_op(name);
    if (!op)
        throw std::runtime_error("The custom op " + name +
                                 " is not loaded: load its library (load_op_library) before compiling or loading the "
                                 "program.");
    return *op;
}

// A buffer of shape s, on the host or the current device, for an output or scratch the program did not allocate
// (before finalize_custom_ops, while quantization evaluates the program to calibrate it); freed with its argument.
migraphx::argument buffer(migraphx::shape const& s, bool host) {
    if (host) return migraphx::argument{s};
    void* data = nullptr;
    hipError_t const error = hipMalloc(&data, s.bytes());
    if (error != hipSuccess)
        throw std::runtime_error(std::string("HIP error ") + hipGetErrorName(error) + " in hipMalloc: " +
                                 hipGetErrorString(error));
    std::shared_ptr<char> owner(static_cast<char*>(data), [](char* p) { (void)hipFree(p); });
    return {s, owner};
}

// A custom op on one node: the loaded op of this name, the node's encoded attributes, and once the program's types are
// final (finalize_custom_ops) its output allocation as its last input and its scratch (`workspace` bytes) before it.
struct custom_op {
    std::string op;
    migraphx::value node_attributes = migraphx::value::binary{};
    bool allocated = false;
    std::size_t workspace = 0;

    template <class Self, class F>
    static auto reflect(Self& self, F f) {
        return migraphx::pack(f(self.op, "op"), f(self.node_attributes, "node_attributes"),
                              f(self.allocated, "allocated"), f(self.workspace, "workspace"));
    }

    std::string name() const { return op; }

    migraphx::value attributes() const {
        bool const host = (loaded(op).flags & CHAINNER_MIGX_HOST) != 0;
        return {{"custom_op", true}, {"target", host ? "cpu" : "gpu"}};
    }

    std::size_t extra_inputs() const { return allocated ? (workspace ? 2 : 1) : 0; }

    migraphx::shape compute_shape(std::vector<migraphx::shape> inputs) const {
        std::size_t const extra = extra_inputs();
        if (inputs.size() < extra) throw std::runtime_error(op + ": its output allocation is missing.");
        std::vector<migraphx::shape> const data(inputs.begin(), inputs.end() - static_cast<std::ptrdiff_t>(extra));
        std::size_t needed = 0;
        migraphx::shape const output =
            custom_output_shape(loaded(op), node_attributes.get_binary(), data, &needed);
        if (!allocated) return output;
        if (inputs.back().type() != output.type() || inputs.back().lens() != output.lens() || needed != workspace)
            throw std::runtime_error(op + ": its output allocation no longer fits its inputs.");
        return inputs.back();
    }

    std::vector<std::size_t> output_alias(std::vector<migraphx::shape> const& inputs) const {
        if (!allocated) return {};
        return {inputs.size() - 1};
    }

    migraphx::argument compute(migraphx::context& ctx, migraphx::shape const& output,
                               std::vector<migraphx::argument> const& args) const {
        chainner_migx_op const& library = loaded(op);
        bool const host = (library.flags & CHAINNER_MIGX_HOST) != 0;
        std::size_t const data = args.size() - extra_inputs();
        std::vector<chainner_migx_tensor> inputs;
        for (std::size_t i = 0; i < data; ++i) inputs.push_back({args[i].data(), shape_of(args[i].get_shape(), op)});
        migraphx::argument result, scratch;
        std::size_t scratch_bytes = workspace;
        if (allocated) {
            result = args.back();
            if (workspace) scratch = args[args.size() - 2];
        } else {
            std::vector<migraphx::shape> shapes;
            for (std::size_t i = 0; i < data; ++i) shapes.push_back(args[i].get_shape());
            custom_output_shape(library, node_attributes.get_binary(), shapes, &scratch_bytes);
            result = buffer(output, host);
            if (scratch_bytes) scratch = buffer({migraphx::shape::uint8_type, {scratch_bytes}}, host);
        }
        chainner_migx_tensor const out{result.data(), shape_of(output, op)};
        void* stream = host ? nullptr : ctx.get_queue().unsafe_get();
        auto const views = attribute_views(node_attributes.get_binary());
        char error[1024] = {};
        if (library.compute(stream, inputs.data(), static_cast<int32_t>(inputs.size()), views.data(),
                            static_cast<int32_t>(views.size()), &out, scratch_bytes ? scratch.data() : nullptr,
                            scratch_bytes, error, sizeof error) != 0)
            throw std::runtime_error(op + ": " + (error[0] ? error : "failed"));
        return result;
    }
};

void check_op(chainner_migx_op const& op, std::string const& path) {
    if (!op.name || !is_custom_op_name(op.name) || !op.compute_shape || !op.compute ||
        (op.flags & ~(CHAINNER_MIGX_HOST | CHAINNER_MIGX_FIXED_TYPES)))
        throw std::runtime_error(path + " lists an op that breaks custom_op.h's contract (" +
                                 (op.name ? std::string(op.name) : std::string("no name")) + ").");
}

#ifdef _WIN32
// Windows' text for a system error code, in UTF-8, after its number.
std::string system_message(DWORD code) {
    std::string message = "error " + std::to_string(code);
    wchar_t* text = nullptr;
    DWORD length = FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                                      FORMAT_MESSAGE_IGNORE_INSERTS,
                                  nullptr, code, 0, reinterpret_cast<wchar_t*>(&text), 0, nullptr);
    if (length == 0) return message;
    while (length > 0 && (text[length - 1] == L'\r' || text[length - 1] == L'\n' || text[length - 1] == L'.'))
        --length;
    int const size = WideCharToMultiByte(CP_UTF8, 0, text, static_cast<int>(length), nullptr, 0, nullptr, nullptr);
    std::string utf8(static_cast<size_t>(size), '\0');
    WideCharToMultiByte(CP_UTF8, 0, text, static_cast<int>(length), utf8.data(), size, nullptr, nullptr);
    LocalFree(text);
    return message + ": " + utf8;
}
#endif

}  // namespace

bool is_custom_op_name(std::string_view name) {
    return name.size() > op_prefix.size() && name.substr(0, op_prefix.size()) == op_prefix;
}

std::vector<std::string> load_op_library(std::string const& path) {
    std::error_code error;
#ifdef _WIN32
    std::filesystem::path const file = std::filesystem::canonical(utf8_path(path), error);
    std::string const canonical = file.u8string();
#else
    std::string const canonical = std::filesystem::canonical(path, error).string();
#endif
    if (error) throw std::runtime_error("The custom-op library " + path + " was not found: " + error.message() + ".");
    std::lock_guard<std::mutex> lock(registry_mutex);
    if (auto const known = loaded_libraries.find(canonical); known != loaded_libraries.end()) return known->second;
#ifdef _WIN32
    // Its imports from its own folder, then the process's DLL search path (HIP is loaded already; the MIGraphX
    // package adds the ROCm folders, nodes/impl/migraphx/runner.py).
    HMODULE const library =
        LoadLibraryExW(file.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!library)
        throw std::runtime_error("The custom-op library " + path + " did not load: " + system_message(GetLastError()) +
                                 ".");
    // Freed again unless every op registers; on success it stays loaded, as MIGraphX keeps its ops.
    std::unique_ptr<std::remove_pointer_t<HMODULE>, decltype(&FreeLibrary)> owner(library, FreeLibrary);
    auto const entry = reinterpret_cast<chainner_migx_ops_function>(
        reinterpret_cast<void*>(GetProcAddress(library, "chainner_migx_ops")));
#else
    void* library = dlopen(canonical.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!library) throw std::runtime_error("The custom-op library " + path + " did not load: " + dlerror() + ".");
    // Closed again unless every op registers; on success it stays loaded, as MIGraphX keeps its ops.
    std::unique_ptr<void, int (*)(void*)> owner(library, dlclose);
    auto const entry = reinterpret_cast<chainner_migx_ops_function>(dlsym(library, "chainner_migx_ops"));
#endif
    if (!entry) throw std::runtime_error(path + " is not a custom-op library: it does not export chainner_migx_ops.");
    chainner_migx_op const* ops = nullptr;
    int32_t count = 0;
    if (entry(CHAINNER_MIGX_ABI, &ops, &count) != 0 || count < 0 || (count && !ops))
        throw std::runtime_error(path + " was built for another custom_op.h version than this runner's (" +
                                 std::to_string(CHAINNER_MIGX_ABI) + ").");
    std::vector<std::string> names;
    for (int32_t i = 0; i < count; ++i) {
        check_op(ops[i], path);
        if (loaded_ops.count(ops[i].name) || std::find(names.begin(), names.end(), ops[i].name) != names.end())
            throw std::runtime_error("The custom op " + std::string(ops[i].name) + " of " + path +
                                     " is already loaded from another library.");
        names.emplace_back(ops[i].name);
    }
    for (int32_t i = 0; i < count; ++i) {
        loaded_ops.emplace(ops[i].name, &ops[i]);
        migraphx::register_op(migraphx::operation{custom_op{ops[i].name}});
    }
    owner.release();
    loaded_libraries.emplace(canonical, names);
    return names;
}

chainner_migx_op const* find_op(std::string const& name) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    auto const op = loaded_ops.find(name);
    return op == loaded_ops.end() ? nullptr : op->second;
}

migraphx::value::binary encode_attributes(std::vector<Attribute> const& attributes) {
    migraphx::value::binary out;
    for (Attribute const& a : attributes) {
        auto const name_bytes = static_cast<uint32_t>(a.name.size() + 1);
        put(out, &name_bytes, sizeof name_bytes);
        put(out, a.name.c_str(), name_bytes);
        int32_t const kind_count[2] = {a.kind, 0};
        put(out, kind_count, sizeof kind_count);
        int64_t count;
        if (a.kind == CHAINNER_MIGX_ATTRIBUTE_INT || a.kind == CHAINNER_MIGX_ATTRIBUTE_INTS) {
            count = static_cast<int64_t>(a.ints.size());
            put(out, &count, sizeof count);
            put(out, a.ints.data(), a.ints.size() * sizeof(int64_t));
        } else if (a.kind == CHAINNER_MIGX_ATTRIBUTE_FLOAT || a.kind == CHAINNER_MIGX_ATTRIBUTE_FLOATS) {
            count = static_cast<int64_t>(a.floats.size());
            put(out, &count, sizeof count);
            put(out, a.floats.data(), a.floats.size() * sizeof(float));
        } else {
            count = static_cast<int64_t>(a.bytes.size());
            put(out, &count, sizeof count);
            put(out, a.bytes.c_str(), a.bytes.size() + 1);
        }
    }
    return out;
}

migraphx::shape custom_output_shape(chainner_migx_op const& op, migraphx::value::binary const& attributes,
                                    std::vector<migraphx::shape> const& inputs, std::size_t* workspace_bytes) {
    std::string const name = op.name;
    auto const shapes = shapes_of(inputs, name);
    auto const views = attribute_views(attributes);
    chainner_migx_shape output{};
    std::size_t workspace = 0;
    char error[1024] = {};
    if (op.compute_shape(shapes.data(), static_cast<int32_t>(shapes.size()), views.data(),
                         static_cast<int32_t>(views.size()), &output, &workspace, error, sizeof error) != 0)
        throw std::runtime_error(name + ": " + (error[0] ? error : "it refuses these inputs"));
    *workspace_bytes = workspace;
    return migraphx_shape(output, name);
}

migraphx::operation make_custom_op(chainner_migx_op const& op, migraphx::value::binary attributes) {
    return migraphx::operation{custom_op{op.name, migraphx::value(std::move(attributes))}};
}

bool keeps_types(migraphx::operation const& op) {
    if (!is_custom_op_name(op.name())) return false;
    chainner_migx_op const* library = find_op(op.name());
    return library && (library->flags & CHAINNER_MIGX_FIXED_TYPES);
}

void finalize_custom_ops(migraphx::module& m) {
    for (auto ins : migraphx::iterator_for(m)) {
        auto const* op = migraphx::any_cast<custom_op>(&ins->get_operator());
        if (!op || op->allocated) continue;
        std::vector<migraphx::instruction_ref> inputs;
        std::vector<migraphx::shape> shapes;
        for (auto input : ins->inputs()) {
            if (!input->get_shape().standard())
                input = m.insert_instruction(ins, migraphx::make_op("contiguous"), input);
            inputs.push_back(input);
            shapes.push_back(input->get_shape());
        }
        custom_op allocated = *op;
        allocated.allocated = true;
        migraphx::shape const output =
            custom_output_shape(loaded(op->op), op->node_attributes.get_binary(), shapes, &allocated.workspace);
        if (allocated.workspace)
            inputs.push_back(m.insert_instruction(
                ins, migraphx::make_op("allocate", {{"shape", migraphx::to_value(migraphx::shape{
                                                                  migraphx::shape::uint8_type, {allocated.workspace}})}})));
        inputs.push_back(
            m.insert_instruction(ins, migraphx::make_op("allocate", {{"shape", migraphx::to_value(output)}})));
        m.replace_instruction(ins, migraphx::operation{allocated}, inputs);
    }
}

std::vector<std::string> custom_op_names(migraphx::module const& m) {
    std::set<std::string> names;
    for (auto ins : migraphx::iterator_for(m))
        if (migraphx::any_cast<custom_op>(&ins->get_operator())) names.insert(ins->name());
    return {names.begin(), names.end()};
}

}  // namespace cn_migx
