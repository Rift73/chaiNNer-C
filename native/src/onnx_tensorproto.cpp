/* Application TensorProto helpers ported from chaiNNer (GPL-3.0).
 * ONNX numpy_helper.to_array remains the public tensor storage/format decoder.
 * Attribute search, scalar decisions, list shape rules, allocation and element
 * copies run here. In particular, this preserves the original setter's INTS
 * attribute with a floats field and the INT64 row-comparison errors.
 */
#include "graph_python.hpp"
#include <pybind11/numpy.h>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace py = pybind11;
namespace {
using graphpy::O;
constexpr int Float = 1, Int32 = 6, Int64 = 7, Float16 = 10, Double = 11;

O clamp_i64(O value) {
    const py::int_ upper((std::numeric_limits<int64_t>::max)());
    const py::int_ lower((std::numeric_limits<int64_t>::min)());
    // Match max(min(value, upper), lower), including ndarray truth errors and
    // NumPy's empty-array truth warnings. Do not flatten multidimensional rows.
    if (graphpy::compare(upper, value, Py_LT)) value = upper;
    if (graphpy::compare(lower, value, Py_GT)) value = lower;
    return value;
}

O find_attribute(const O &node, const O &key) {
    for (py::handle handle : node.attr("attribute")) {
        O attribute = py::reinterpret_borrow<O>(handle);
        if (graphpy::compare(attribute.attr("name"), key, Py_EQ)) return attribute;
    }
    return py::none();
}

py::array attr_ai(const O &node, const O &key) {
    O attribute = find_attribute(node, key);
    if (attribute.is_none()) return py::array_t<int32_t>(0);
    O values = attribute.attr("ints");
    py::array_t<int64_t> result(static_cast<py::ssize_t>(py::len(values)));
    int64_t *output = result.mutable_data();
    py::ssize_t index = 0;
    for (py::handle value : values)
        output[index++] = py::cast<int64_t>(clamp_i64(py::reinterpret_borrow<O>(value)));
    return result;
}

void set_attr_ai(const O &node, const O &key, const O &value, const O &constructor) {
    // Public protobuf owns field validation. Build before appending so errors
    // leave the node intact. AttributeProto.INTS is 7, despite the floats field.
    O attribute = constructor(py::arg("name") = key, py::arg("floats") = value,
                              py::arg("type") = 7);
    node.attr("attribute").attr("append")(attribute);
}

py::array attr_af(const O &node, const O &key) {
    O attribute = find_attribute(node, key);
    if (attribute.is_none()) return py::array_t<float>(0);
    O values = attribute.attr("floats");
    py::array_t<float> result(static_cast<py::ssize_t>(py::len(values)));
    float *output = result.mutable_data();
    py::ssize_t index = 0;
    for (py::handle value : values) output[index++] = py::cast<float>(value);
    return result;
}

O attr_scalar(const O &node, const O &key, const O &fallback, int kind) {
    O attribute = find_attribute(node, key);
    if (attribute.is_none()) return fallback;
    if (kind == 0) return clamp_i64(attribute.attr("i"));
    if (kind == 1) return attribute.attr("f");
    return attribute.attr("s").attr("decode")("ascii");
}

O attr_tensor(const O &node, const O &key, const O &constructor) {
    O attribute = find_attribute(node, key);
    return attribute.is_none() ? constructor() : O(attribute.attr("t"));
}

O input_f(const O &tensor, const O &decode) {
    O data = decode(tensor); // Decoder errors precede unsupported-type errors.
    const int type = py::cast<int>(tensor.attr("data_type"));
    if (type == Float || type == Float16 || type == Double || type == Int32)
        return data.attr("item")(0);
    if (type == Int64) return clamp_i64(data.attr("item")(0));
    throw py::type_error("Unknown data type " + std::to_string(type));
}

py::array copy_elements(const py::array &source, const std::vector<py::ssize_t> &shape) {
    py::array result(source.dtype(), shape);
    if (source.size() == 0) return result;
    const auto *input = static_cast<const unsigned char *>(source.data());
    auto *output = static_cast<unsigned char *>(result.mutable_data());
    const size_t bytes = static_cast<size_t>(source.itemsize());
    for (py::ssize_t index = 0; index < source.size(); ++index) {
        py::ssize_t remaining = index, offset = 0;
        for (py::ssize_t dim = source.ndim(); dim > 0;) {
            --dim;
            offset += (remaining % source.shape(dim)) * source.strides(dim);
            remaining /= source.shape(dim);
        }
        std::memcpy(output + static_cast<size_t>(index) * bytes, input + offset, bytes);
    }
    return result;
}

py::array list_copy(const py::array &source) {
    if (source.ndim() == 0) throw py::type_error("iteration over a 0-d array");
    // Rebuilding an empty list loses trailing dimensions. Nonempty lists of
    // rows preserve shape, including rows that themselves contain zero values.
    std::vector<py::ssize_t> shape;
    if (source.shape(0) == 0) shape.push_back(0);
    else for (py::ssize_t dim = 0; dim < source.ndim(); ++dim) shape.push_back(source.shape(dim));
    return copy_elements(source, shape);
}

py::array input_ai(const O &tensor, const O &decode, const O &logger) {
    const int type = py::cast<int>(tensor.attr("data_type"));
    if (type != Int32 && type != Int64) {
        logger.attr("error")("Unknown data type %s", type);
        return py::array_t<int32_t>(0);
    }
    py::array data = py::cast<py::array>(decode(tensor));
    if (data.size() == 1) return copy_elements(data, {1});
    if (type == Int64) {
        for (py::handle row : data) (void)clamp_i64(py::reinterpret_borrow<O>(row));
    }
    return list_copy(data);
}

py::array input_af(const O &tensor, const O &decode, const O &logger) {
    const int type = py::cast<int>(tensor.attr("data_type"));
    if (type != Float && type != Float16 && type != Double) {
        logger.attr("error")("Unknown data type %s", type);
        return py::array_t<float>(0);
    }
    return list_copy(py::cast<py::array>(decode(tensor)));
}

py::int_ data_size(const O &tensor, const O &mode) {
    O raw = tensor.attr("raw_data");
    if (graphpy::truth(raw)) {
        const size_t divisor = graphpy::compare(mode, py::int_(Float16), Py_EQ) ? 2 : 4;
        return py::int_(py::len(raw) / divisor);
    }
    const int type = py::cast<int>(tensor.attr("data_type"));
    if (type == Float || type == Float16) return py::int_(py::len(tensor.attr("float_data")));
    return py::int_(0);
}
}

void cn_bind_onnx_tensorproto(py::module_ &module) {
    module.def("onnx_get_node_attr_ai", &attr_ai);
    module.def("onnx_set_node_attr_ai", &set_attr_ai);
    module.def("onnx_get_node_attr_af", &attr_af);
    module.def("onnx_get_node_attr_i", [](const O &node, const O &key, const O &fallback) {
        return attr_scalar(node, key, fallback, 0);
    });
    module.def("onnx_get_node_attr_f", [](const O &node, const O &key, const O &fallback) {
        return attr_scalar(node, key, fallback, 1);
    });
    module.def("onnx_get_node_attr_s", [](const O &node, const O &key, const O &fallback) {
        return attr_scalar(node, key, fallback, 2);
    });
    module.def("onnx_get_node_attr_tensor", &attr_tensor);
    module.def("onnx_get_node_attr_from_input_f", &input_f);
    module.def("onnx_get_node_attr_from_input_ai", &input_ai);
    module.def("onnx_get_node_attr_from_input_af", &input_af);
    module.def("onnx_get_tensor_proto_data_size", &data_size);
}
