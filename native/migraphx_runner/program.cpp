#include "program.h"

#include "onnx_model.h"
#include "ops.h"

#include <migraphx/compile_options.hpp>
#include <migraphx/execution_environment.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/msgpack.hpp>
#include <migraphx/onnx.hpp>
#include <migraphx/param_utils.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/version.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <stdexcept>
#include <utility>

namespace cn_migx {
namespace {

// The saved form: this magic, then msgpack of the header fields and MIGraphX's value of the program.
constexpr char magic[8] = {'C', 'N', 'M', 'I', 'G', 'X', '0', '1'};
constexpr char format_name[] = "chaiNNer-C MIGraphX program";
constexpr int format_version = 1;

std::string read_file(std::string const& path) {
    std::ifstream file(utf8_path(path), std::ios::binary);
    if (!file) throw std::invalid_argument("The ONNX model " + path + " cannot be opened.");
    std::string data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (file.bad()) throw std::runtime_error("The ONNX model " + path + " could not be read.");
    return data;
}

// ONNX's TensorProto data type of a MIGraphX type, for carriers.
int32_t onnx_type(migraphx::shape::type_t type) {
    switch (type) {
    case migraphx::shape::float_type: return 1;
    case migraphx::shape::uint8_type: return 2;
    case migraphx::shape::int8_type: return 3;
    case migraphx::shape::uint16_type: return 4;
    case migraphx::shape::int16_type: return 5;
    case migraphx::shape::int32_type: return 6;
    case migraphx::shape::int64_type: return 7;
    case migraphx::shape::bool_type: return 9;
    case migraphx::shape::half_type: return 10;
    case migraphx::shape::double_type: return 11;
    case migraphx::shape::uint32_type: return 12;
    case migraphx::shape::uint64_type: return 13;
    case migraphx::shape::bf16_type: return 16;
    case migraphx::shape::fp8e4m3fn_type: return 17;
    case migraphx::shape::fp8e4m3fnuz_type: return 18;
    case migraphx::shape::fp8e5m2_type: return 19;
    case migraphx::shape::fp8e5m2fnuz_type: return 20;
    default: throw std::runtime_error("A custom op's output type has no ONNX counterpart.");
    }
}

// The model as a program with every chainner node's custom op in place (onnx_model.h). Each node's output shape comes,
// in graph order, from the model up to the node with the carriers found so far, whose shapes are therefore right; the
// whole model then parses with a carrier before each node whose output differs from its placeholder's.
migraphx::program parse_model(std::string const& model, migraphx::onnx_options options) {
    std::vector<CustomNode> const nodes = custom_nodes(model);
    if (nodes.empty()) return migraphx::parse_onnx_buffer(model, options);
    std::vector<chainner_migx_op const*> ops;
    std::vector<migraphx::value::binary> attributes;
    for (CustomNode const& node : nodes) {
        chainner_migx_op const* op = find_op(node.op_type);
        if (!op)
            throw std::invalid_argument("The model uses the custom op " + node.op_type +
                                        ", which is not loaded: load its library with load_op_library first.");
        ops.push_back(op);
        attributes.push_back(encode_attributes(node.attributes));
    }
    options.skip_unknown_operators = true;
    std::map<std::size_t, Carrier> carriers;
    std::vector<migraphx::shape> outputs;
    for (std::size_t k = 0; k < nodes.size(); ++k) {
        migraphx::program const prefix =
            migraphx::parse_onnx_buffer(with_carriers(model, carriers, nodes[k].index, &nodes[k].inputs), options);
        std::vector<migraphx::shape> shapes;
        for (auto input : prefix.get_main_module()->get_returns()) {
            migraphx::shape const& s = input->get_shape();
            if (s.dynamic())
                throw std::invalid_argument(nodes[k].op_type +
                                            " takes inputs of fixed sizes: compile the model with input_dims.");
            shapes.emplace_back(s.type(), s.lens());
        }
        if (shapes.size() != nodes[k].inputs.size())
            throw std::runtime_error("MIGraphX did not give " + nodes[k].op_type + " all of its inputs.");
        std::size_t workspace = 0;
        outputs.push_back(custom_output_shape(*ops[k], attributes[k], shapes, &workspace));
        migraphx::shape const& output = outputs.back();
        if (shapes.empty() || shapes[0].type() != output.type() || shapes[0].lens() != output.lens())
            carriers[nodes[k].index] = {onnx_type(output.type()),
                                        std::vector<int64_t>(output.lens().begin(), output.lens().end())};
    }

    migraphx::program program =
        migraphx::parse_onnx_buffer(carriers.empty() ? model : with_carriers(model, carriers), options);
    migraphx::module& m = *program.get_main_module();
    std::vector<migraphx::instruction_ref> placeholders;
    for (auto ins : migraphx::iterator_for(m)) {
        if (ins->name().rfind("unknown:", 0) != 0) continue;
        std::string const op_type = ins->name().substr(8);
        if (placeholders.size() >= nodes.size() || nodes[placeholders.size()].op_type != op_type)
            throw std::invalid_argument("MIGraphX does not know the ONNX operator " + op_type + ".");
        placeholders.push_back(ins);
    }
    if (placeholders.size() != nodes.size())
        throw std::invalid_argument("A chainner node is inside a subgraph, where custom ops cannot be.");
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        std::vector<migraphx::instruction_ref> inputs = placeholders[i]->inputs();
        if (carriers.count(nodes[i].index)) inputs.erase(inputs.begin());
        for (auto& input : inputs)
            if (!input->get_shape().standard())
                input = m.insert_instruction(placeholders[i], migraphx::make_op("contiguous"), input);
        m.replace_instruction(placeholders[i], make_custom_op(*ops[i], attributes[i]), inputs);
        if (placeholders[i]->get_shape().lens() != outputs[i].lens())
            throw std::runtime_error("MIGraphX gave " + nodes[i].op_type + " other inputs than its model's prefix.");
    }
    return program;
}

// fp16 quantization's operations: all, or every one but the custom ops that keep their types.
std::vector<std::string> quantized_ops(migraphx::module const& m) {
    std::set<std::string> names;
    bool kept = false;
    for (auto ins : migraphx::iterator_for(m)) {
        if (keeps_types(ins->get_operator())) kept = true;
        else if (ins->name().front() != '@') names.insert(ins->name());
    }
    if (!kept) return {"all"};
    return {names.begin(), names.end()};
}

// Every output packed row-major, as callers allocate them (a model may end in a transpose).
void pack_outputs(migraphx::module& m) {
    auto const last = std::prev(m.end());
    if (last->name() != "@return") return;
    std::vector<migraphx::instruction_ref> outputs = last->inputs();
    bool packed = true;
    for (auto& output : outputs) {
        if (output->get_shape().dynamic() || output->get_shape().standard()) continue;
        output = m.insert_instruction(last, migraphx::make_op("contiguous"), output);
        packed = false;
    }
    if (!packed) m.replace_return(outputs);
}

// The element type of the convolutions and matrix products ("mixed" when they differ, "" without any), the type a
// dequantization feeds them from when there is one.
std::string product_type(migraphx::module const& m) {
    std::set<std::string> types;
    for (auto ins : migraphx::iterator_for(m)) {
        static std::set<std::string> const products = {"convolution", "convolution_backwards", "quant_convolution",
                                                       "dot", "quant_dot"};
        if (!products.count(ins->name()) || ins->inputs().empty()) continue;
        auto input = ins->inputs().front();
        if (input->name() == "dequantizelinear") input = input->inputs().front();
        types.insert(type_name(input->get_shape().type()));
    }
    if (types.empty()) return "";
    return types.size() == 1 ? *types.begin() : "mixed";
}

}  // namespace

std::string migraphx_version() {
    return std::to_string(MIGRAPHX_VERSION_MAJOR) + "." + std::to_string(MIGRAPHX_VERSION_MINOR) + "." +
           std::to_string(MIGRAPHX_VERSION_PATCH);
}

void check_hip(hipError_t error, char const* call) {
    if (error != hipSuccess)
        throw std::runtime_error(std::string("HIP error ") + hipGetErrorName(error) + " in " + call + ": " +
                                 hipGetErrorString(error));
}

std::string device_gfx(int device) {
    hipDeviceProp_t properties{};
    check_hip(hipGetDeviceProperties(&properties, device), "hipGetDeviceProperties");
    std::string name = properties.gcnArchName;
    return name.substr(0, name.find(':'));
}

std::string type_name(migraphx::shape::type_t type) {
    switch (type) {
    case migraphx::shape::float_type: return "float32";
    case migraphx::shape::half_type: return "float16";
    case migraphx::shape::bf16_type: return "bfloat16";
    case migraphx::shape::double_type: return "float64";
    case migraphx::shape::int8_type: return "int8";
    case migraphx::shape::uint8_type: return "uint8";
    case migraphx::shape::int16_type: return "int16";
    case migraphx::shape::uint16_type: return "uint16";
    case migraphx::shape::int32_type: return "int32";
    case migraphx::shape::uint32_type: return "uint32";
    case migraphx::shape::int64_type: return "int64";
    case migraphx::shape::uint64_type: return "uint64";
    case migraphx::shape::bool_type: return "bool";
    case migraphx::shape::fp8e4m3fnuz_type: return "fp8e4m3fnuz";
    case migraphx::shape::fp8e4m3fn_type: return "fp8e4m3fn";
    case migraphx::shape::fp8e5m2_type: return "fp8e5m2";
    case migraphx::shape::fp8e5m2fnuz_type: return "fp8e5m2fnuz";
    default: return migraphx::shape{type}.type_string();
    }
}

Program::Program(Program&&) noexcept = default;
Program& Program::operator=(Program&&) noexcept = default;
Program::~Program() = default;

Program Program::compile(std::string const& model, bool from_file, CompileOptions const& options) {
    if (options.target != "gpu" && options.target != "ref")
        throw std::invalid_argument("The target is gpu or ref, not " + options.target + ".");
    bool const gpu = options.target == "gpu";
    if (gpu && options.device < 0)
        throw std::invalid_argument("The GPU index must be 0 or more, not " + std::to_string(options.device) + ".");
    if (options.fp16 && options.fp8) throw std::invalid_argument("A program is quantized to fp16 or fp8, not both.");
    if (options.fp8 && options.calibration.empty())
        throw std::invalid_argument("FP8 quantization needs calibration inputs (a few typical inputs of the model).");

    Program p;
    p.target_ = options.target;
    if (gpu) {
        p.device_ = options.device;
        check_hip(hipSetDevice(options.device), "hipSetDevice");
        p.gfx_ = device_gfx(options.device);
    } else {
        p.gfx_ = "ref";
    }
    migraphx::target const target = migraphx::make_target(options.target);

    migraphx::onnx_options onnx;
    for (auto const& [name, dims] : options.input_dims) onnx.map_input_dims[name] = dims;
    if (from_file) onnx.external_data_path = utf8_path(model).parent_path().string();
    p.program_ = parse_model(from_file ? read_file(model) : model, onnx);

    migraphx::module& m = *p.program_.get_main_module();
    if (options.fp16) migraphx::quantize_fp16(p.program_, quantized_ops(m));
    if (options.fp8) {
        std::vector<migraphx::parameter_map> calibration;
        for (auto const& inputs : options.calibration)
            calibration.emplace_back(inputs.begin(), inputs.end());
        migraphx::quantize_fp8(p.program_, target, calibration);
    }
    finalize_custom_ops(*p.program_.get_main_module());
    pack_outputs(*p.program_.get_main_module());
    p.compute_type_ = product_type(*p.program_.get_main_module());
    p.custom_ops_ = custom_op_names(*p.program_.get_main_module());

    // MIGraphX's default fast math trades accuracy for speed (as ONNX Runtime's MIGraphX provider, off).
    migraphx::compile_options compile;
    compile.offload_copy = false;
    compile.fast_math = false;
    compile.exhaustive_tune = options.exhaustive_tune;
    p.program_.compile(target, compile);
    p.describe();
    return p;
}

Program Program::load(char const* data, std::size_t size, int device) {
    if (size < sizeof magic || std::memcmp(data, magic, sizeof magic) != 0)
        throw std::invalid_argument("This is not a chaiNNer-C MIGraphX program.");
    migraphx::value const v = migraphx::from_msgpack(data + sizeof magic, size - sizeof magic);
    if (!v.contains("format") || v.at("format").to<std::string>() != format_name ||
        v.at("version").to<int>() != format_version)
        throw std::invalid_argument("This MIGraphX program was saved by another chaiNNer-C version: build it again "
                                    "from its ONNX model.");
    std::string const built = v.at("migraphx").to<std::string>();
    if (built != migraphx_version())
        throw std::invalid_argument("This program was built with MIGraphX " + built + ", and this chaiNNer-C has " +
                                    migraphx_version() + ": build it again from its ONNX model.");
    for (auto const& op : v.at("custom_ops"))
        if (!find_op(op.to<std::string>()))
            throw std::invalid_argument("This program uses the custom op " + op.to<std::string>() +
                                        ", which is not loaded: load its library with load_op_library first.");
    Program p;
    p.target_ = v.at("target").to<std::string>();
    p.gfx_ = v.at("gfx").to<std::string>();
    p.compute_type_ = v.at("compute_type").to<std::string>();
    p.custom_ops_ = migraphx::from_value<std::vector<std::string>>(v.at("custom_ops"));
    if (p.target_ == "gpu") {
        if (device < 0) throw std::invalid_argument("The GPU index must be 0 or more, not " + std::to_string(device) + ".");
        check_hip(hipSetDevice(device), "hipSetDevice");
        std::string const gfx = device_gfx(device);
        if (gfx != p.gfx_)
            throw std::invalid_argument("This program was built for " + p.gfx_ + ", and GPU " + std::to_string(device) +
                                        " is " + gfx + ": build it again from its ONNX model for this GPU.");
        p.device_ = device;
    }
    p.program_.from_value(v.at("program"));
    p.describe();
    return p;
}

void Program::describe() {
    auto const shapes = program_.get_parameter_shapes();
    input_names_.clear();
    input_shapes_.clear();
    output_parameters_ = false;
    for (std::string const& name : program_.get_parameter_names()) {
        if (name.find("#output_") != std::string::npos) {
            output_parameters_ = true;
            continue;
        }
        input_names_.push_back(name);
        input_shapes_.push_back(shapes.at(name));
    }
    output_shapes_ = program_.get_output_shapes();
}

std::vector<char> Program::save() const {
    std::lock_guard<std::mutex> lock(*mutex_);
    if (closed_) throw std::runtime_error("The MIGraphX program is closed.");
    migraphx::value v = {{"format", format_name},
                         {"version", format_version},
                         {"migraphx", migraphx_version()},
                         {"target", target_},
                         {"gfx", gfx_},
                         {"compute_type", compute_type_}};
    v["custom_ops"] = migraphx::to_value(custom_ops_);
    v["program"] = program_.to_value();
    std::vector<char> body = migraphx::to_msgpack(v);
    std::vector<char> out(std::begin(magic), std::end(magic));
    out.insert(out.end(), body.begin(), body.end());
    return out;
}

void Program::run(std::map<std::string, void*> const& inputs, std::vector<void*> const& outputs, hipStream_t stream) {
    std::lock_guard<std::mutex> lock(*mutex_);
    if (closed_) throw std::runtime_error("The MIGraphX program is closed.");
    if (outputs.size() != output_shapes_.size())
        throw std::invalid_argument("The program has " + std::to_string(output_shapes_.size()) + " outputs, not " +
                                    std::to_string(outputs.size()) + ".");
    migraphx::parameter_map parameters;
    for (std::size_t i = 0; i < input_names_.size(); ++i) {
        auto const input = inputs.find(input_names_[i]);
        if (input == inputs.end()) throw std::invalid_argument("The program's input " + input_names_[i] + " is missing.");
        parameters[input_names_[i]] = migraphx::argument(input_shapes_[i], static_cast<char*>(input->second));
    }
    if (inputs.size() != input_names_.size())
        throw std::invalid_argument("The program takes " + std::to_string(input_names_.size()) + " inputs, not " +
                                    std::to_string(inputs.size()) + ".");
    if (output_parameters_)
        for (std::size_t i = 0; i < outputs.size(); ++i)
            parameters[migraphx::param_name(i, "main:#output_")] =
                migraphx::argument(output_shapes_[i], static_cast<char*>(outputs[i]));
    std::vector<migraphx::argument> results;
    if (target_ == "gpu") check_hip(hipSetDevice(device_), "hipSetDevice");
    // On a stream: MIGraphX's own stream waits for it and it for MIGraphX's (asynchronous); without one, the run ends
    // before returning.
    if (target_ == "gpu" && stream)
        results = program_.eval(parameters, migraphx::execution_environment{migraphx::any_ptr(stream), true});
    else
        results = program_.eval(parameters);
    if (output_parameters_) return;
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        std::size_t const bytes = output_shapes_[i].bytes();
        if (target_ == "gpu")
            check_hip(hipMemcpyAsync(outputs[i], results[i].data(), bytes, hipMemcpyDeviceToDevice, stream),
                      "hipMemcpyAsync");
        else
            std::memcpy(outputs[i], results[i].data(), bytes);
    }
}

void Program::close() {
    std::lock_guard<std::mutex> lock(*mutex_);
    if (closed_) return;
    if (device_ >= 0) check_hip(hipSetDevice(device_), "hipSetDevice");
    program_ = migraphx::program{};
    closed_ = true;
}

}  // namespace cn_migx
