// What compile_onnx reads and changes in an ONNX model itself, on protobuf's wire format: the main graph's nodes of
// domain "chainner" with their inputs and attributes, and the same model with a shape carrier before chosen ones of
// them, whole or up to a node.
//
// MIGraphX's ONNX parser has no custom ops. compile_onnx parses chainner nodes as placeholders, which MIGraphX gives
// the shape of their first input, and puts the custom op in each one's place afterwards. Every shape after a
// placeholder is right only when the op's output has that shape, so each node whose output differs gets a carrier as
// its first input: an Expand of a zero to the op's output type and size (a broadcast, nothing allocated), dropped when
// the op takes the node's place. compile_onnx learns each node's output shape in graph order, from the model up to the
// node with the carriers found so far (a wrong placeholder can make a later node refuse to parse).
#pragma once

#include "ops.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace cn_migx {

struct CustomNode {
    std::size_t index;  // among the main graph's nodes
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<Attribute> attributes;
};

// The main graph's nodes of domain "chainner", in graph order; std::invalid_argument for a damaged model, a node
// without exactly one output, or one with an empty (omitted) input.
std::vector<CustomNode> custom_nodes(std::string_view model);

// A carrier's element type (an ONNX TensorProto data type) and dimensions.
struct Carrier {
    int32_t data_type;
    std::vector<int64_t> dims;
};

// The model with a carrier before each node of `carriers` (keyed by node index), as that node's first input; with
// `outputs`, only its nodes before `end`, whose outputs are those tensors, and the initializers they use.
std::string with_carriers(std::string_view model, std::map<std::size_t, Carrier> const& carriers,
                          std::size_t end = std::numeric_limits<std::size_t>::max(),
                          std::vector<std::string> const* outputs = nullptr);

}  // namespace cn_migx
