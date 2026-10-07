/* ONNX graph metadata and dimension algorithms, ported from the installed
 * chaiNNer helpers. The protobuf storage/checker/shape inference engines remain
 * their original native dependencies through public Python object interfaces.
 * update_inputs_outputs_dims derives from ONNX Project Contributors,
 * SPDX-License-Identifier: Apache-2.0, with chaiNNer's documented modifications.
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <array>
#include <string>
#include <vector>

namespace py = pybind11;
namespace {
py::object own(py::handle value) { return py::reinterpret_borrow<py::object>(value); }
py::object item(py::handle object, Py_ssize_t index) {
    py::int_ key(index);
    PyObject* result = PyObject_GetItem(object.ptr(), key.ptr());
    if (!result) throw py::error_already_set();
    return py::reinterpret_steal<py::object>(result);
}
bool compare(py::handle a, py::handle b, int op) {
    int value = PyObject_RichCompareBool(a.ptr(), b.ptr(), op);
    if (value < 0) throw py::error_already_set();
    return value != 0;
}
bool truth(py::handle value) {
    int result = PyObject_IsTrue(value.ptr());
    if (result < 0) throw py::error_already_set();
    return result != 0;
}
py::object as_int(py::handle value) {
    return PyLong_Check(value.ptr()) ? own(value) : py::none();
}
py::object or_else(py::handle value, py::handle fallback) {
    return value.is_none() ? own(fallback) : own(value);
}
py::tuple parse_shape(py::object shape) {
    py::object second = item(shape, 1);
    if (PyLong_Check(second.ptr()) && compare(second, py::int_(4), Py_LE))
        return py::make_tuple("BCHW", second, as_int(item(shape, 3)), as_int(item(shape, 2)));
    py::object fourth = item(shape, 3);
    if (PyLong_Check(fourth.ptr()) && compare(fourth, py::int_(4), Py_LE))
        return py::make_tuple("BHWC", fourth, as_int(item(shape, 2)), as_int(second));
    return py::make_tuple("BCHW", 3, as_int(fourth), as_int(item(shape, 2)));
}
// A range-for extends the lifetime of its range's last temporary only: an earlier
// accessor in an .attr() chain would die and free the protobuf wrapper the range
// still points into. Each chain is therefore named before it is iterated.
py::tuple tensor_shape(py::object tensor) {
    py::list result;
    py::object dims = tensor.attr("shape").attr("dim");
    for (py::handle h : dims) {
        py::object dim = own(h);
        result.append(truth(dim.attr("HasField")("dim_param")) ? dim.attr("dim_param") : dim.attr("dim_value"));
    }
    if (py::len(result) != 4)
        throw py::value_error("Expected 4 dimensions, got " + std::to_string(py::len(result)));
    return py::tuple(result);
}
bool tensor_input(py::object input) { return truth(input.attr("type").attr("HasField")("tensor_type")); }
bool image_to_image(py::object model) {
    py::object graph = model.attr("graph"), inputs = graph.attr("input"), outputs = graph.attr("output");
    return py::len(inputs) == 1 && py::len(outputs) == 1 && tensor_input(item(inputs, 0)) && tensor_input(item(outputs, 0));
}
std::string fp_type(py::object model) {
    py::object graph = model.attr("graph");
    for (const char* key : {"input", "output"}) {
        py::object values = graph.attr(key);
        for (py::handle value : values) {
            py::object input = own(value);
            if (!tensor_input(input)) continue;
            py::object type = input.attr("type").attr("tensor_type").attr("elem_type");
            for (const auto& entry : std::array<std::pair<int, const char*>, 4>{{{10,"fp16"},{1,"fp32"},{11,"fp64"},{16,"bf16"}}})
                if (compare(type, py::int_(entry.first), Py_EQ)) return entry.second;
        }
    }
    return "fp32";
}
py::object opset(py::object model) {
    for (py::handle h : model.attr("opset_import")) {
        py::object entry = own(h);
        if (compare(entry.attr("domain"), py::str(""), Py_EQ)) return entry.attr("version");
    }
    return py::int_(-1);
}
py::object update_dims(py::object model, py::object inputs, py::object outputs) {
    py::set symbols;
    py::object graph = model.attr("graph");
    for (const char* key : {"input", "output", "value_info"})
        for (py::handle h : graph.attr(key)) {
            py::object dims = own(h).attr("type").attr("tensor_type").attr("shape").attr("dim");
            for (py::handle d : dims) {
                py::object dim = own(d);
                if (truth(dim.attr("HasField")("dim_param"))) symbols.add(dim.attr("dim_param"));
            }
        }
    for (const auto& group : {std::make_pair("input", inputs), std::make_pair("output", outputs)})
        for (py::handle h : graph.attr(group.first)) {
            py::object tensor = own(h), name = tensor.attr("name");
            py::object values = group.second[name];
            Py_ssize_t index = 0;
            for (py::handle value : values) {
                py::object dim = item(tensor.attr("type").attr("tensor_type").attr("shape").attr("dim"), index);
                if (PyLong_Check(value.ptr())) {
                    if (compare(value, py::int_(0), Py_GE)) dim.attr("dim_value") = value;
                    else {
                        py::str generated(py::str(name).cast<std::string>() + "_" + std::to_string(index));
                        if (symbols.contains(generated))
                            throw py::value_error("Unable to generate unique dim_param for axis " + std::to_string(index) + " of " + py::str(name).cast<std::string>() + ". Please manually provide a dim_param value.");
                        dim.attr("dim_param") = generated;
                    }
                } else dim.attr("dim_param") = value;
                ++index;
            }
        }
    py::module_::import("onnx.checker").attr("check_model")(model);
    return model;
}
void infer_init(py::object self, py::object model) {
    self.attr("model") = model;
    py::object input = item(model.attr("graph").attr("input"), 0);
    py::object output = item(model.attr("graph").attr("output"), 0);
    if (!tensor_input(input) || !tensor_input(output)) throw py::value_error("Expected tensor inputs and outputs");
    py::tuple in_shape = tensor_shape(input.attr("type").attr("tensor_type"));
    py::tuple out_shape = tensor_shape(output.attr("type").attr("tensor_type"));
    self.attr("input_shape") = in_shape; self.attr("output_shape") = out_shape;
    py::tuple parsed = parse_shape(in_shape);
    self.attr("tensor_format") = parsed[0]; self.attr("input_channels") = parsed[1];
    self.attr("fixed_input_width") = parsed[2]; self.attr("fixed_input_height") = parsed[3];
    self.attr("output_channels") = as_int(out_shape[compare(parsed[0], py::str("BCHW"), Py_EQ) ? 1 : 3]);
}
py::tuple infer_shape(py::object self, py::object size) {
    py::object b = or_else(as_int(item(self.attr("input_shape"), 0)), py::int_(1));
    py::object c = self.attr("input_channels");
    py::object h = or_else(self.attr("fixed_input_height"), item(size, 1));
    py::object w = or_else(self.attr("fixed_input_width"), item(size, 0));
    std::string format = self.attr("tensor_format").cast<std::string>();
    py::list dims;
    if (format == "BCHW") { dims.append(b); dims.append(c); dims.append(h); dims.append(w); }
    else if (format == "BHWC") { dims.append(b); dims.append(h); dims.append(w); dims.append(c); }
    else throw py::value_error("Unknown tensor format: " + format);
    py::object model = self.attr("model");
    py::object input = item(model.attr("graph").attr("input"), 0), output = item(model.attr("graph").attr("output"), 0);
    py::dict inputs, outputs;
    inputs[input.attr("name")] = dims; outputs[output.attr("name")] = py::list(self.attr("output_shape"));
    update_dims(model, inputs, outputs);
    py::object inferred = py::module_::import("onnx.shape_inference").attr("infer_shapes")(model, py::arg("strict_mode") = true);
    py::tuple a = tensor_shape(item(inferred.attr("graph").attr("input"), 0).attr("type").attr("tensor_type"));
    py::tuple z = tensor_shape(item(inferred.attr("graph").attr("output"), 0).attr("type").attr("tensor_type"));
    // Preserve the installed width/height ordering, including its BCHW reversal.
    py::tuple aa(3), zz(3);
    for (size_t index = 0; index < 3; ++index) {
        size_t source = format == "BCHW" ? 3 - index : index + 1;
        aa[index] = as_int(a[source]); zz[index] = as_int(z[source]);
    }
    return py::make_tuple(aa, zz);
}
bool sequence_match(const std::string& bytes, const std::vector<std::string>& parts) {
    size_t offset = 0;
    for (size_t i = 0; i < parts.size(); ++i) {
        size_t found = bytes.find(parts[i], offset);
        if (found == std::string::npos) return false;
        offset = found + parts[i].size();
        if (i + 1 < parts.size()) {
            if (offset == bytes.size()) return false;
            ++offset; // RE2's dot-all .+ requires at least one byte between names.
        }
    }
    return true;
}
int classify(py::bytes data) {
    std::string bytes = data;
    auto suffix = [&](size_t count) { return bytes.substr(bytes.size() > count ? bytes.size() - count : 0); };
    if (sequence_match(suffix(1000), {"1959","1960","1961","1962","1963","1964","1965"}) ||
        sequence_match(suffix(600), {"1808","1827","1828","2296","1831","1850","1958"}) ||
        sequence_match(bytes.substr(0,10000), {"/stage1/rebnconvin/conv_s1/Conv","/stage1/rebnconvin/relu_s1/Relu"})) return 1;
    if (sequence_match(suffix(1000), {"output","d1","Concat_1876","Concat_1896","Concat_1916","Concat_1936","Concat_1956"})) return 2;
    return 0;
}
}

void cn_bind_onnx_graph(py::module_& module) {
    module.def("onnx_parse_shape", &parse_shape);
    module.def("onnx_tensor_shape", &tensor_shape);
    module.def("onnx_tensor_input", &tensor_input);
    module.def("onnx_image_to_image", &image_to_image);
    module.def("onnx_fp_type", &fp_type);
    module.def("onnx_opset", &opset);
    module.def("onnx_update_dims", &update_dims);
    module.def("onnx_infer_init", &infer_init);
    module.def("onnx_infer_shape", &infer_shape);
    module.def("onnx_classify", &classify);
}
