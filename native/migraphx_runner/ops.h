// Custom-op libraries (custom_op.h) and the MIGraphX operation that runs their ops. A loaded op is registered with
// MIGraphX under its name, so programs that use it load (make_op); each instruction carries its node's attributes,
// encoded in one binary value that survives saving.
#pragma once

#include "custom_op.h"

#include <migraphx/module.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/value.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace cn_migx {

// A file's path from Python's text, which arrives in UTF-8: Windows reads a narrow path in the ANSI code page.
inline std::filesystem::path utf8_path(std::string const& text) {
#ifdef _WIN32
    return std::filesystem::u8path(text);
#else
    return text;
#endif
}

// The prefix of every custom op's name, and the ONNX domain of its nodes.
constexpr std::string_view op_prefix = "chainner_";
constexpr std::string_view onnx_domain = "chainner";

// One ONNX attribute of a node, owned.
struct Attribute {
    std::string name;
    int32_t kind = CHAINNER_MIGX_ATTRIBUTE_INT;
    std::vector<int64_t> ints;
    std::vector<float> floats;
    std::string bytes;
};

// Opens the library at path and registers its ops; the ops' names. A library already loaded gives its names again.
std::vector<std::string> load_op_library(std::string const& path);
// The loaded op of this name, or nullptr.
chainner_migx_op const* find_op(std::string const& name);
// Whether a name is a custom op's (its prefix), loaded or not.
bool is_custom_op_name(std::string_view name);

migraphx::value::binary encode_attributes(std::vector<Attribute> const& attributes);

// A loaded op's output shape for inputs of these types and sizes (packed), and its scratch bytes; the op's message
// as std::runtime_error when it refuses them.
migraphx::shape custom_output_shape(chainner_migx_op const& op, migraphx::value::binary const& attributes,
                                    std::vector<migraphx::shape> const& inputs, std::size_t* workspace_bytes);

// The MIGraphX operation of a loaded op on one node's attributes, without its output allocation: finalize_custom_ops
// adds it once the program's types are final (after quantization).
migraphx::operation make_custom_op(chainner_migx_op const& op, migraphx::value::binary attributes);
// Whether an instruction's operation is a custom op that keeps its types under quantization
// (CHAINNER_MIGX_FIXED_TYPES).
bool keeps_types(migraphx::operation const& op);
// Gives each custom op of the module packed inputs, its output allocation and its scratch, as MIGraphX's own custom
// ops take them (the output an allocation aliased by the op's result, last).
void finalize_custom_ops(migraphx::module& m);
// The custom ops a module uses, sorted, each once.
std::vector<std::string> custom_op_names(migraphx::module const& m);

}  // namespace cn_migx
