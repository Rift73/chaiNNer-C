/* chaiNNer ONNX application model decisions, ported from its GPL-3.0 source.
 * Public ONNX/protobuf decoding, checking, optimization and inference engines
 * remain dependencies. Probe/control flow, pairing, validation, rebuilding and
 * metadata decisions are native; no original Python algorithm is evaluated.
 */
#include "graph_python.hpp"
#include "graph_exception.hpp"
#include "graph_unpack.hpp"
#include <pybind11/numpy.h>
#include <array>

namespace py = pybind11;
namespace {
using graphpy::O;
O own(py::handle value) { return py::reinterpret_borrow<O>(value); }
O item(const O &value, Py_ssize_t index) { return graphpy::item(value, py::int_(index)); }
O subtract(const O &a, const O &b) {
    PyObject *result = PyNumber_Subtract(a.ptr(), b.ptr());
    if (!result) throw py::error_already_set();
    return py::reinterpret_steal<O>(result);
}
O divide(const O &a, const O &b) {
    PyObject *result = PyNumber_TrueDivide(a.ptr(), b.ptr());
    if (!result) throw py::error_already_set();
    return py::reinterpret_steal<O>(result);
}

// The original's tuple unpacking; CPython unpacks, so its errors are the running
// interpreter's own.
using graphpy::unpack;

py::tuple detect_size(const O &infer, const O &size_req, const O &logger) {
    if (!infer.attr("fixed_input_width").is_none() && !infer.attr("fixed_input_height").is_none()) {
        O callback = infer.attr("infer_shape");
        O width = infer.attr("fixed_input_width");
        O height = infer.attr("fixed_input_height");
        O shape = callback(py::make_tuple(width, height));
        return py::make_tuple(size_req(), shape);
    }
    for (int size : std::array<int, 4>{16, 64, 256, 512}) {
        try {
            O shape = infer.attr("infer_shape")(py::make_tuple(size, size));
            py::tuple out = unpack(item(shape, 1), 3);
            if (!out[0].is_none() && !out[1].is_none())
                return py::make_tuple(size_req(py::arg("multiple_of") = size), shape);
        } catch (const py::error_already_set &error) {
            if (!error.matches(PyExc_Exception)) throw;
            GraphHandledException context(error);
            logger.attr("exception")("Failed to infer shape for size %s", size);
        }
    }
    return py::make_tuple(size_req(), py::none());
}

O scale(const O &input, const O &output) {
    if (input.is_none() || output.is_none()) return py::none();
    PyObject *raw = PyNumber_Remainder(output.ptr(), input.ptr());
    if (!raw) throw py::error_already_set();
    O remainder = py::reinterpret_steal<O>(raw);
    if (graphpy::compare(remainder, py::int_(0), Py_NE)) return py::none();
    raw = PyNumber_FloorDivide(output.ptr(), input.ptr());
    if (!raw) throw py::error_already_set();
    return py::reinterpret_steal<O>(raw);
}

O load_model(const O &source, const py::dict &context) {
    O onnx = context["onnx"], model, bytes;
    int instance = PyObject_IsInstance(source.ptr(), onnx.attr("ModelProto").ptr());
    if (instance < 0) throw py::error_already_set();
    if (instance != 0) { model = source; bytes = model.attr("SerializeToString")(); }
    else { bytes = source; model = onnx.attr("load_model_from_string")(source); }
    O opset = context["get_opset"](model);
    O dtype = context["get_tensor_fp_datatype"](model);
    O info = context["OnnxInfo"](py::arg("opset") = opset, py::arg("dtype") = dtype);
    const int classification = py::cast<int>(context["graph"]().attr("onnx_classify")(bytes));
    if (classification == 1 || classification == 2) {
        info.attr("scale_width") = 1;
        info.attr("scale_height") = classification == 1 ? 1 : 3;
        return context["OnnxRemBg"](bytes, info);
    }
    try {
        O infer = context["ModelShapeInference"](model);
        for (const char *key : {"fixed_input_width", "fixed_input_height", "input_channels", "output_channels"})
            info.attr(key) = infer.attr(key);
        py::tuple detected = unpack(context["_detect_size_req"](infer), 2);
        O requirement = own(detected[0]), shape = own(detected[1]);
        info.attr("size_req") = requirement;
        if (graphpy::truth(shape)) {
            py::tuple pair = unpack(shape, 2);
            py::tuple input = unpack(own(pair[0]), 3);
            py::tuple output = unpack(own(pair[1]), 3);
            info.attr("scale_width") = scale(own(input[1]), own(output[1]));
            info.attr("scale_height") = scale(own(input[0]), own(output[0]));
            info.attr("output_channels") = output[2];
        }
    } catch (const py::error_already_set &error) {
        if (!error.matches(PyExc_Exception)) throw;
    }
    return context["OnnxGeneric"](bytes, info);
}

py::list perform_interp(const O &weights_a, const O &weights_b, const O &amount,
                        const O &decode, const O &encode, const O &interpolate, const O &cast) {
    O amount_b = divide(amount, py::int_(100));
    O amount_a = subtract(py::int_(1), amount_b);
    py::list result;
    // Match zip(..., strict=False), including one extra A iterator advance when
    // B ends first. Iterators and field access order remain observable.
    O first = py::iter(weights_a), second = py::iter(weights_b);
    while (true) {
        PyObject *raw = PyIter_Next(first.ptr());
        if (!raw) { if (PyErr_Occurred()) throw py::error_already_set(); break; }
        O a = py::reinterpret_steal<O>(raw);
        raw = PyIter_Next(second.ptr());
        if (!raw) { if (PyErr_Occurred()) throw py::error_already_set(); break; }
        O b = py::reinterpret_steal<O>(raw);
        O name = b.attr("name");
        O aa = decode(a);
        O bb = decode(b);
        O shape_a = aa.attr("shape");
        O shape_b = bb.attr("shape");
        graphpy::require(graphpy::compare(shape_a, shape_b, Py_EQ),
                         py::str("Weights must have same size and shape"));
        O data = interpolate(aa, bb, amount_a, amount_b);
        data = cast(data, aa.attr("dtype"));
        result.append(encode(data, name));
    }
    return result;
}

O will_upscale(const O &node_context, const O &model, const py::dict &context) {
    if (graphpy::compare(model.attr("sub_type"), py::str("Generic"), Py_NE)) return py::bool_(true);
    py::array_t<float> image({3, 3, 3}, {4, 12, 36});
    for (size_t i = 0; i < 27; ++i) image.mutable_data()[i] = 1.0F;
    O result = context["upscale_image_node"](node_context, image, model, context["NO_TILING"], 0, false);
    O mean;
    if (graphpy::compare(result.attr("dtype"), context["np"].attr("float32"), Py_EQ))
        mean = context["native_mean"](result);
    else mean = context["np"].attr("mean")(result);
    result = py::none();
    PyObject *raw = PyObject_RichCompare(mean.ptr(), py::float_(0.5).ptr(), Py_GT);
    if (!raw) throw py::error_already_set();
    return py::reinterpret_steal<O>(raw);
}

py::tuple interpolate_models(const O &node_context, const O &a, const O &b,
                              const O &amount, const py::dict &context) {
    if (graphpy::compare(amount, py::int_(0), Py_EQ)) return py::make_tuple(a, 100, 0);
    if (graphpy::compare(amount, py::int_(100), Py_EQ)) return py::make_tuple(b, 0, 100);
    O model_a = context["onnx"].attr("load_from_string")(a.attr("bytes"));
    model_a = context["safely_optimize_onnx_model"](model_a);
    O weights_a = model_a.attr("graph").attr("initializer");
    O model_b = context["onnx"].attr("load_from_string")(b.attr("bytes"));
    model_b = context["safely_optimize_onnx_model"](model_b);
    O weights_b = model_b.attr("graph").attr("initializer");
    const size_t count_a = py::len(weights_a);
    const size_t count_b = py::len(weights_b);
    graphpy::require(count_a == count_b, py::str("Models must have same number of weights"));
    context["logger"].attr("debug")("Interpolating models...");
    O weights = context["perform_interp"](weights_a, weights_b, amount);
    O model = context["deepcopy"](model_b);
    const size_t count = py::len(model.attr("graph").attr("initializer"));
    for (size_t i = 0; i < count; ++i) model.attr("graph").attr("initializer").attr("pop")();
    model.attr("graph").attr("initializer").attr("extend")(weights);
    O loaded = context["load_onnx_model"](model);
    if (!graphpy::truth(context["check_will_upscale"](node_context, loaded)))
        throw py::value_error("These models are not compatible and not able to be interpolated together");
    return py::make_tuple(loaded, subtract(py::int_(100), amount), amount);
}
}

void cn_bind_onnx_model_application(py::module_ &module) {
    module.def("onnx_detect_size_req", &detect_size);
    module.def("onnx_load_model", &load_model);
    module.def("onnx_perform_interp", &perform_interp);
    module.def("onnx_check_will_upscale", &will_upscale);
    module.def("onnx_interpolate_models", &interpolate_models);
}
