/* Public CPython object adapters for native application graph algorithms.
 * Existing protobuf/model objects retain their identities and unknown fields;
 * graph traversal and decisions live in the C++ implementation units. */
#include <pybind11/pybind11.h>

void cn_bind_onnx_graph(pybind11::module_&);
void cn_bind_ncnn(pybind11::module_&);
void cn_bind_onnx_converter(pybind11::module_&);
void cn_bind_onnx_tensorproto(pybind11::module_&);
void cn_bind_onnx_model_application(pybind11::module_&);
void cn_bind_tiling(pybind11::module_&);
void cn_bind_utility_scalar(pybind11::module_&);
void cn_bind_utility_random(pybind11::module_&);
void cn_bind_utility_text(pybind11::module_&);
void cn_bind_image_io(pybind11::module_&);
void cn_bind_execution_ops(pybind11::module_&);
void cn_bind_file_sequence(pybind11::module_&);
void cn_bind_video_io(pybind11::module_&);
void cn_bind_utility_clipboard(pybind11::module_&);
void cn_bind_numpy_pool(pybind11::module_&);

PYBIND11_MODULE(_chainner_graph, module) {
    module.doc() = "Native chaiNNer graph algorithms with public object adapters";
    module.attr("abi_version") = 1;
    cn_bind_onnx_graph(module);
    cn_bind_ncnn(module);
    cn_bind_onnx_converter(module);
    cn_bind_onnx_tensorproto(module);
    cn_bind_onnx_model_application(module);
    cn_bind_tiling(module);
    cn_bind_utility_scalar(module);
    cn_bind_utility_random(module);
    cn_bind_utility_text(module);
    cn_bind_image_io(module);
    cn_bind_execution_ops(module);
    cn_bind_file_sequence(module);
    cn_bind_video_io(module);
    cn_bind_utility_clipboard(module);
    cn_bind_numpy_pool(module);
}
