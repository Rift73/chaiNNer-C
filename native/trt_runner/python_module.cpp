/* _chainner_trt: the TensorRT runner for chaiNNer's TensorRT Upscale node, with its tile
 * planner, frame description and CPU reference for the tests. Planes are 2-D NumPy arrays
 * (uint8, uint16, float16 or float32) of any strides; a BGR image passes its three channel
 * views in R, G, B order and is uploaded interleaved, as one block. */
#include "frame.h"
#include "plan.h"
#include "reference.h"
#include "runner.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace cn_trt;

namespace {

Sample sample_of(py::dtype const& dtype) {
    char const kind = dtype.kind();
    auto const size = dtype.itemsize();
    if (kind == 'u' && size == 1) return Sample::U8;
    if (kind == 'u' && size == 2) return Sample::U16;
    if (kind == 'f' && size == 2) return Sample::F16;
    if (kind == 'f' && size == 4) return Sample::F32;
    throw std::invalid_argument("Planes are uint8, uint16, float16 or float32, not " + std::string(py::str(dtype)) + ".");
}

Element element_of(std::string const& name) {
    if (name == "fp32") return Element::F32;
    if (name == "fp16") return Element::F16;
    if (name == "bf16") return Element::BF16;
    throw std::invalid_argument("Tensor precision is fp32, fp16 or bf16, not " + name + ".");
}

char const* precision_of(Element element) {
    return element == Element::F32 ? "fp32" : (element == Element::F16 ? "fp16" : "bf16");
}

// NumPy's dtype holding an engine tensor's elements (bfloat16 as its uint16 bits).
py::dtype element_dtype(Element element) {
    if (element == Element::F32) return py::dtype::of<float>();
    return element == Element::F16 ? py::dtype("float16") : py::dtype::of<uint16_t>();
}

// The planes of one frame, kept alive while the runner reads or writes them.
struct Planes {
    std::vector<py::array> arrays;
    std::vector<HostPlane> planes;
};

Planes planes_of(py::sequence const& items, bool writable, char const* what) {
    Planes result;
    for (py::handle item : items) {
        if (!py::isinstance<py::array>(item)) throw py::type_error(std::string(what) + " are NumPy arrays.");
        auto array = py::reinterpret_borrow<py::array>(item);
        if (array.ndim() != 2) throw std::invalid_argument(std::string(what) + " are 2-D arrays, one per plane.");
        if (!result.arrays.empty() && (!array.dtype().is(result.arrays[0].dtype()) ||
                                       array.shape(0) != result.arrays[0].shape(0) ||
                                       array.shape(1) != result.arrays[0].shape(1)))
            throw std::invalid_argument(std::string(what) + " share one dtype and one size.");
        if (writable && !array.writeable()) throw std::invalid_argument(std::string(what) + " are writable arrays.");
        result.planes.push_back({const_cast<void*>(array.data()), array.strides(0), array.strides(1)});
        result.arrays.push_back(std::move(array));
    }
    if (result.arrays.empty()) throw std::invalid_argument(std::string(what) + " hold at least one plane.");
    return result;
}

struct Frame {
    Planes inputs;
    Planes outputs;
    FrameJob job;
};

Frame frame_of(py::sequence const& inputs, py::sequence const& outputs, int scale, Bounds const& bounds,
               int input_channels, int output_channels, int bits, bool full_range) {
    Frame frame{planes_of(inputs, false, "Input planes"), planes_of(outputs, true, "Result planes"), {}};
    py::array const& in = frame.inputs.arrays[0];
    py::array const& out = frame.outputs.arrays[0];
    if (!out.dtype().is(in.dtype())) throw std::invalid_argument("Result planes have the input planes' dtype.");
    if (out.shape(0) != in.shape(0) * scale || out.shape(1) != in.shape(1) * scale)
        throw std::invalid_argument("Result planes are " + std::to_string(in.shape(1) * scale) + "x" +
                                    std::to_string(in.shape(0) * scale) + ", the input's size times " +
                                    std::to_string(scale) + ".");
    Format const format{sample_of(in.dtype()), bits, full_range};
    frame.job = describe_frame(frame.inputs.planes.data(), static_cast<int>(frame.inputs.planes.size()),
                               frame.outputs.planes.data(), static_cast<int>(frame.outputs.planes.size()),
                               static_cast<int>(in.shape(1)), static_cast<int>(in.shape(0)), format, bounds, scale,
                               input_channels, output_channels);
    return frame;
}

Bounds bounds_of(py::tuple const& t) {
    if (t.size() != 5) throw std::invalid_argument("Bounds are (min_h, min_w, max_h, max_w, alignment).");
    return {t[0].cast<int>(), t[1].cast<int>(), t[2].cast<int>(), t[3].cast<int>(), t[4].cast<int>()};
}

py::list weights_list(std::vector<std::vector<float>> const& weights) {
    py::list list;
    for (auto const& w : weights) {
        py::array_t<float> array(static_cast<py::ssize_t>(w.size()));
        std::memcpy(array.mutable_data(), w.data(), w.size() * sizeof(float));
        list.append(array);
    }
    return list;
}

}  // namespace

PYBIND11_MODULE(_chainner_trt, module) {
    module.doc() = "chaiNNer-C's TensorRT runner (native/trt_runner)";
    module.attr("MAX_STREAMS") = max_streams;

    module.def(
        "plan_tiles",
        [](int height, int width, int scale, py::tuple bounds, int overlap) {
            Plan const plan = plan_tiles(height, width, scale, bounds_of(bounds), overlap);
            py::dict result;
            result["tile_h"] = plan.tile_h;
            result["tile_w"] = plan.tile_w;
            result["ys"] = plan.ys;
            result["xs"] = plan.xs;
            result["wy"] = weights_list(plan.wy);
            result["wx"] = weights_list(plan.wx);
            return result;
        },
        "tiled_upscale's tiles and blend weights for a frame (empty weights for one tile).", py::arg("height"),
        py::arg("width"), py::arg("scale"), py::arg("bounds"), py::arg("overlap") = tile_overlap);

    module.def(
        "describe",
        [](py::sequence inputs, py::sequence outputs, int scale, py::tuple bounds, int input_channels,
           int output_channels, int bits, bool full_range) {
            Frame const frame =
                frame_of(inputs, outputs, scale, bounds_of(bounds), input_channels, output_channels, bits, full_range);
            py::list tiles;
            for (TileIn const& t : frame.job.tiles)
                tiles.append(py::make_tuple(t.y0, t.x0, t.real_h, t.real_w, t.height, t.width, t.edge));
            py::dict result;
            result["tiles"] = tiles;
            result["blended"] = !frame.job.places.empty();
            result["result_planes"] = frame.job.result_planes;
            result["source_interleaved"] = frame.job.source.interleaved;
            result["result_interleaved"] = frame.job.result.interleaved;
            return result;
        },
        "Checks a frame's arguments as Runner.upscale does; its tiles (y0, x0, real_h, real_w, height, width, "
        "edge) and packing.",
        py::arg("inputs"), py::arg("outputs"), py::arg("scale"), py::arg("bounds"), py::arg("input_channels"),
        py::arg("output_channels"), py::kw_only(), py::arg("bits") = 8, py::arg("full_range") = true);

    module.def(
        "reference_upscale",
        [](py::sequence inputs, py::sequence outputs, int scale, py::tuple bounds, int input_channels,
           int output_channels, std::string const& input_precision, std::string const& output_precision,
           py::function engine, int bits, bool full_range, bool clip) {
            Frame const frame =
                frame_of(inputs, outputs, scale, bounds_of(bounds), input_channels, output_channels, bits, full_range);
            Element const in = element_of(input_precision), out = element_of(output_precision);
            Engine const run = [&](TileIn const& tile, void const* input, void* output) {
                py::array tensor(element_dtype(in), {input_channels, tile.height, tile.width});
                std::memcpy(tensor.mutable_data(), input, static_cast<std::size_t>(tensor.nbytes()));
                py::array const made = py::array::ensure(engine(tensor), py::array::c_style);
                py::array const expected(element_dtype(out), {output_channels, tile.height * scale, tile.width * scale});
                if (!made || !made.dtype().is(expected.dtype()) || made.ndim() != 3 ||
                    made.shape(0) != expected.shape(0) || made.shape(1) != expected.shape(1) ||
                    made.shape(2) != expected.shape(2))
                    throw std::invalid_argument("The engine stand-in returns a C-contiguous " +
                                                std::string(py::str(expected.dtype())) + " array of shape (" +
                                                std::to_string(output_channels) + ", " +
                                                std::to_string(tile.height * scale) + ", " +
                                                std::to_string(tile.width * scale) + ").");
                std::memcpy(output, made.data(), static_cast<std::size_t>(made.nbytes()));
            };
            reference_upscale(frame.job, frame.inputs.planes.data(), frame.outputs.planes.data(), input_channels,
                              output_channels, in, out, run, clip);
        },
        "The runner's frame loop on the CPU, with `engine` (a function from a tile's input tensor to its "
        "output tensor) for TensorRT; writes the result planes.",
        py::arg("inputs"), py::arg("outputs"), py::arg("scale"), py::arg("bounds"), py::arg("input_channels"),
        py::arg("output_channels"), py::arg("input_precision"), py::arg("output_precision"), py::arg("engine"),
        py::kw_only(), py::arg("bits") = 8, py::arg("full_range") = true, py::arg("clip") = false);

    py::class_<Runner>(module, "Runner",
                       "A deserialized engine with `streams` execution contexts; Python decompresses compressed "
                       "engines first (nodes/impl/compressed_file.py).")
        .def(py::init([](py::bytes const& engine, int device, int streams) {
                 char* data = nullptr;
                 Py_ssize_t size = 0;
                 if (PyBytes_AsStringAndSize(engine.ptr(), &data, &size) != 0) throw py::error_already_set();
                 py::gil_scoped_release release;
                 return std::make_unique<Runner>(data, static_cast<std::size_t>(size), device, streams);
             }),
             py::arg("engine"), py::arg("device"), py::arg("streams"))
        .def_property_readonly("input_channels", &Runner::input_channels)
        .def_property_readonly("output_channels", &Runner::output_channels)
        .def_property_readonly("scale", &Runner::scale)
        .def_property_readonly("input_precision", [](Runner const& r) { return precision_of(r.input_element()); })
        .def_property_readonly("output_precision", [](Runner const& r) { return precision_of(r.output_element()); })
        .def_property_readonly("bounds",
                               [](Runner const& r) {
                                   Bounds const& b = r.bounds();
                                   return py::make_tuple(b.min_h, b.min_w, b.max_h, b.max_w, b.alignment);
                               })
        .def(
            "upscale",
            [](Runner& r, py::sequence inputs, py::sequence outputs, int bits, bool full_range, bool clip) {
                Frame const frame = frame_of(inputs, outputs, r.scale(), r.bounds(), r.input_channels(),
                                             r.output_channels(), bits, full_range);
                py::gil_scoped_release release;
                r.upscale(frame.job, frame.inputs.planes.data(), frame.outputs.planes.data(), clip);
            },
            "Upscales one frame's planes into the result planes (the input's size times the scale).",
            py::arg("inputs"), py::arg("outputs"), py::kw_only(), py::arg("bits") = 8, py::arg("full_range") = true,
            py::arg("clip") = false)
        .def("capture_failure", &Runner::capture_failure)
        .def("close", [](Runner& r) {
            py::gil_scoped_release release;
            r.close();
        });
}
