#include "onnx_model.h"

#include <cstring>
#include <set>
#include <stdexcept>

namespace cn_migx {
namespace {

// Protobuf's wire types (groups, 3 and 4, are not in ONNX).
constexpr uint32_t varint_wire = 0, fixed64_wire = 1, length_wire = 2, fixed32_wire = 5;

// onnx.proto's field numbers.
constexpr uint32_t model_graph = 7;
constexpr uint32_t graph_node = 1, graph_initializer = 5, graph_input = 11, graph_output = 12;
constexpr uint32_t node_input = 1, node_output = 2, node_name = 3, node_op_type = 4, node_attribute = 5,
                   node_domain = 7;
constexpr uint32_t attribute_name = 1, attribute_f = 2, attribute_i = 3, attribute_s = 4, attribute_g = 6,
                   attribute_floats = 7, attribute_ints = 8, attribute_graphs = 11, attribute_type = 20,
                   attribute_doc_string = 13;
constexpr uint32_t tensor_dims = 1, tensor_data_type = 2, tensor_name = 8, tensor_raw_data = 9;
constexpr uint32_t value_info_name = 1;
// AttributeProto.AttributeType
constexpr uint64_t type_float = 1, type_int = 2, type_string = 3, type_floats = 6, type_ints = 7;
// TensorProto.DataType
constexpr int32_t onnx_int64 = 7;

// Carriers' names, unlikely in a model: <prefix><node index>, with _zero and _dims for their two initializers.
constexpr char carrier_prefix[] = "chainner_migraphx_carrier_";

[[noreturn]] void damaged() { throw std::invalid_argument("The ONNX model is damaged (its protobuf does not parse)."); }

struct Field {
    uint32_t number;
    uint32_t wire;
    uint64_t value;          // a varint or fixed field's value
    std::string_view bytes;  // a length-delimited field's payload
    std::string_view whole;  // the whole field, key included
};

class Reader {
public:
    explicit Reader(std::string_view data) : data_(data) {}
    bool done() const { return at_ == data_.size(); }

    uint64_t varint() {
        uint64_t result = 0;
        for (int shift = 0; shift < 64; shift += 7) {
            if (at_ >= data_.size()) damaged();
            auto const byte = static_cast<unsigned char>(data_[at_++]);
            result |= static_cast<uint64_t>(byte & 0x7Fu) << shift;
            if (!(byte & 0x80u)) return result;
        }
        damaged();
    }

    Field next() {
        std::size_t const start = at_;
        uint64_t const key = varint();
        Field field{static_cast<uint32_t>(key >> 3), static_cast<uint32_t>(key & 7u), 0, {}, {}};
        if (field.wire == varint_wire) {
            field.value = varint();
        } else if (field.wire == fixed64_wire || field.wire == fixed32_wire) {
            std::size_t const n = field.wire == fixed64_wire ? 8 : 4;
            if (data_.size() - at_ < n) damaged();
            std::memcpy(&field.value, data_.data() + at_, n);  // little-endian, as x86-64
            at_ += n;
        } else if (field.wire == length_wire) {
            uint64_t const n = varint();
            if (n > data_.size() - at_) damaged();
            field.bytes = data_.substr(at_, n);
            at_ += n;
        } else {
            damaged();
        }
        field.whole = data_.substr(start, at_ - start);
        return field;
    }

private:
    std::string_view data_;
    std::size_t at_ = 0;
};

void put_varint(std::string& out, uint64_t value) {
    while (value >= 0x80u) {
        out += static_cast<char>(value | 0x80u);
        value >>= 7;
    }
    out += static_cast<char>(value);
}

void put_field(std::string& out, uint32_t number, std::string_view bytes) {
    put_varint(out, uint64_t{number} << 3 | length_wire);
    put_varint(out, bytes.size());
    out.append(bytes);
}

void put_field(std::string& out, uint32_t number, uint64_t value) {
    put_varint(out, uint64_t{number} << 3 | varint_wire);
    put_varint(out, value);
}

std::string_view graph_of(std::string_view model) {
    std::string_view graph;
    bool found = false;
    for (Reader r(model); !r.done();) {
        Field const f = r.next();
        if (f.number == model_graph && f.wire == length_wire) {
            graph = f.bytes;
            found = true;
        }
    }
    if (!found) throw std::invalid_argument("The ONNX model has no graph.");
    return graph;
}

float float_of(uint64_t bits) {
    auto const narrow = static_cast<uint32_t>(bits);
    float value;
    std::memcpy(&value, &narrow, sizeof value);
    return value;
}

Attribute attribute_of(std::string_view bytes, std::string const& op_type) {
    Attribute a;
    uint64_t type = 0;
    bool single = false, text = false;
    for (Reader r(bytes); !r.done();) {
        Field const f = r.next();
        switch (f.number) {
        case attribute_name: a.name = f.bytes; break;
        case attribute_type: type = f.value; break;
        case attribute_f:
            a.floats = {float_of(f.value)};
            single = true;
            break;
        case attribute_i:
            a.ints = {static_cast<int64_t>(f.value)};
            single = true;
            break;
        case attribute_s:
            a.bytes = f.bytes;
            text = true;
            break;
        case attribute_floats:
            if (f.wire == length_wire) {
                if (f.bytes.size() % 4) damaged();
                for (std::size_t i = 0; i < f.bytes.size(); i += 4) {
                    uint32_t bits;
                    std::memcpy(&bits, f.bytes.data() + i, 4);
                    a.floats.push_back(float_of(bits));
                }
            } else {
                a.floats.push_back(float_of(f.value));
            }
            break;
        case attribute_ints:
            if (f.wire == length_wire) {
                for (Reader packed(f.bytes); !packed.done();) a.ints.push_back(static_cast<int64_t>(packed.varint()));
            } else {
                a.ints.push_back(static_cast<int64_t>(f.value));
            }
            break;
        case attribute_doc_string: break;
        default:
            throw std::invalid_argument("The attribute " + a.name + " of " + op_type +
                                        " is not an int, float, string, ints or floats attribute.");
        }
    }
    // Before the type field (IR version 2), the field set gives the type.
    if (type == 0) type = text ? type_string : (!a.floats.empty() ? (single ? type_float : type_floats) : (single ? type_int : type_ints));
    if (type == type_float) {
        a.kind = CHAINNER_MIGX_ATTRIBUTE_FLOAT;
        a.floats.resize(1);
    } else if (type == type_int) {
        a.kind = CHAINNER_MIGX_ATTRIBUTE_INT;
        a.ints.resize(1);
    } else if (type == type_string) {
        a.kind = CHAINNER_MIGX_ATTRIBUTE_STRING;
    } else if (type == type_floats) {
        a.kind = CHAINNER_MIGX_ATTRIBUTE_FLOATS;
    } else if (type == type_ints) {
        a.kind = CHAINNER_MIGX_ATTRIBUTE_INTS;
    } else {
        throw std::invalid_argument("The attribute " + a.name + " of " + op_type +
                                    " is not an int, float, string, ints or floats attribute.");
    }
    return a;
}

std::size_t element_bytes(int32_t data_type) {
    switch (data_type) {
    case 1: case 6: case 12: return 4;           // float, int32, uint32
    case 7: case 11: case 13: return 8;          // int64, double, uint64
    case 4: case 5: case 10: case 16: return 2;  // uint16, int16, float16, bfloat16
    default: return 1;                           // uint8, int8, bool, the FP8 types
    }
}

// The string field `number` of a message (its last occurrence), or "".
std::string_view string_field(std::string_view message, uint32_t number) {
    std::string_view value;
    for (Reader r(message); !r.done();) {
        Field const f = r.next();
        if (f.number == number && f.wire == length_wire) value = f.bytes;
    }
    return value;
}

// Whether a node holds a subgraph (If, Loop, Scan), which may use any of the graph's initializers.
bool holds_subgraph(std::string_view node) {
    for (Reader r(node); !r.done();) {
        Field const f = r.next();
        if (f.number != node_attribute || f.wire != length_wire) continue;
        for (Reader a(f.bytes); !a.done();) {
            uint32_t const number = a.next().number;
            if (number == attribute_g || number == attribute_graphs) return true;
        }
    }
    return false;
}

std::string tensor(std::string const& name, int32_t data_type, std::vector<int64_t> const& dims, std::string_view raw) {
    std::string t;
    if (!dims.empty()) {
        std::string packed;
        for (int64_t d : dims) put_varint(packed, static_cast<uint64_t>(d));
        put_field(t, tensor_dims, packed);
    }
    put_field(t, tensor_data_type, static_cast<uint64_t>(data_type));
    put_field(t, tensor_name, name);
    put_field(t, tensor_raw_data, raw);
    return t;
}

}  // namespace

std::vector<CustomNode> custom_nodes(std::string_view model) {
    std::vector<CustomNode> nodes;
    std::size_t index = 0;
    for (Reader graph(graph_of(model)); !graph.done();) {
        Field const f = graph.next();
        if (f.number != graph_node || f.wire != length_wire) continue;
        std::string op_type, domain;
        int outputs = 0;
        bool omitted = false;
        std::vector<std::string> inputs;
        std::vector<std::string_view> attributes;
        for (Reader node(f.bytes); !node.done();) {
            Field const g = node.next();
            if (g.number == node_input) {
                omitted = omitted || g.bytes.empty();
                inputs.emplace_back(g.bytes);
            }
            else if (g.number == node_output) ++outputs;
            else if (g.number == node_op_type) op_type = g.bytes;
            else if (g.number == node_domain) domain = g.bytes;
            else if (g.number == node_attribute) attributes.push_back(g.bytes);
        }
        if (domain == onnx_domain) {
            if (outputs != 1)
                throw std::invalid_argument("The node " + op_type + " (domain chainner) has " + std::to_string(outputs) +
                                            " outputs; a custom op has one.");
            if (omitted)
                throw std::invalid_argument("The node " + op_type + " (domain chainner) omits an input; a custom op "
                                            "takes all of its inputs.");
            CustomNode custom{index, op_type, std::move(inputs), {}};
            for (std::string_view a : attributes) custom.attributes.push_back(attribute_of(a, op_type));
            nodes.push_back(std::move(custom));
        }
        ++index;
    }
    return nodes;
}

std::string with_carriers(std::string_view model, std::map<std::size_t, Carrier> const& carriers, std::size_t end,
                          std::vector<std::string> const* outputs) {
    std::string_view const source = graph_of(model);
    // A prefix keeps the initializers its nodes and outputs use (all when a node holds a subgraph), and drops the
    // others from the graph's inputs too, where older models list them.
    std::set<std::string_view> used;
    bool keep_all = outputs == nullptr;
    if (outputs) {
        used.insert(outputs->begin(), outputs->end());
        std::size_t index = 0;
        for (Reader r(source); !r.done() && index < end;) {
            Field const f = r.next();
            if (f.number != graph_node || f.wire != length_wire) continue;
            ++index;
            keep_all = keep_all || holds_subgraph(f.bytes);
            for (Reader node(f.bytes); !node.done();) {
                Field const g = node.next();
                if (g.number == node_input) used.insert(g.bytes);
            }
        }
    }
    std::set<std::string_view> dropped;
    if (!keep_all)
        for (Reader r(source); !r.done();) {
            Field const f = r.next();
            if (f.number == graph_initializer && f.wire == length_wire) {
                std::string_view const name = string_field(f.bytes, tensor_name);
                if (!used.count(name)) dropped.insert(name);
            }
        }

    std::string graph;
    std::size_t index = 0;
    for (Reader r(source); !r.done();) {
        Field const f = r.next();
        if (f.number == graph_node && f.wire == length_wire) {
            std::size_t const i = index++;
            if (i >= end) continue;
            auto const carrier = carriers.find(i);
            if (carrier != carriers.end()) {
                std::string const name = carrier_prefix + std::to_string(carrier->first);
                std::string expand;
                put_field(expand, node_input, name + "_zero");
                put_field(expand, node_input, name + "_dims");
                put_field(expand, node_output, name);
                put_field(expand, node_name, name);
                put_field(expand, node_op_type, std::string_view("Expand"));
                put_field(graph, graph_node, expand);
                std::string node;
                put_field(node, node_input, name);
                node.append(f.bytes);
                put_field(graph, graph_node, node);
                continue;
            }
        } else if (f.wire == length_wire &&
                   ((f.number == graph_initializer && dropped.count(string_field(f.bytes, tensor_name))) ||
                    (f.number == graph_input && dropped.count(string_field(f.bytes, value_info_name))) ||
                    (f.number == graph_output && outputs))) {
            continue;
        }
        graph.append(f.whole);
    }
    for (auto const& [node_index, carrier] : carriers) {
        if (node_index >= end) continue;
        std::string const name = carrier_prefix + std::to_string(node_index);
        put_field(graph, graph_initializer,
                  tensor(name + "_zero", carrier.data_type, {}, std::string(element_bytes(carrier.data_type), '\0')));
        std::string dims(carrier.dims.size() * sizeof(int64_t), '\0');
        if (!carrier.dims.empty()) std::memcpy(dims.data(), carrier.dims.data(), dims.size());
        put_field(graph, graph_initializer,
                  tensor(name + "_dims", onnx_int64, {static_cast<int64_t>(carrier.dims.size())}, dims));
    }
    if (outputs)
        for (std::string const& name : *outputs) {
            std::string info;
            put_field(info, value_info_name, name);
            put_field(graph, graph_output, info);
        }
    std::string out;
    for (Reader r(model); !r.done();) {
        Field const f = r.next();
        if (f.number == model_graph && f.wire == length_wire) put_field(out, model_graph, graph);
        else out.append(f.whole);
    }
    return out;
}

}  // namespace cn_migx
