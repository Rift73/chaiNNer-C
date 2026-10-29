/* _chainner_migraphx: the MIGraphX runner for chaiNNer's MIGraphX nodes, _chainner_trt's counterpart (the same tile
 * planner, frame description and CPU reference for the tests, and a Runner on a saved program), plus what MIGraphX
 * adds: compiling ONNX models (with the custom ops of loaded libraries, custom_op.h), saved programs, and running a
 * program on device pointers. Planes are 2-D NumPy arrays (uint8, uint16, float16 or float32) of any strides; a BGR
 * image passes its three channel views in R, G, B order and is uploaded interleaved, as one block. */
#include "frame.h"
#include "ops.h"
#include "plan.h"
#include "program.h"
#include "reference.h"
#include "runner.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace cn_trt;
using cn_migx::Program;
using cn_migx::Runner;

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

// NumPy's dtype holding a program tensor's elements (bfloat16 as its uint16 bits).
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

// A str or os.PathLike argument as a path, or nullopt for bytes.
std::optional<std::string> path_of(py::handle source) {
    if (py::isinstance<py::bytes>(source)) return std::nullopt;
    return py::str(py::module_::import("os").attr("fspath")(source)).cast<std::string>();
}

std::string read_bytes(std::string const& path) {
    std::ifstream file(cn_migx::utf8_path(path), std::ios::binary);
    if (!file) throw std::invalid_argument("The file " + path + " cannot be opened.");
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

migraphx::shape::type_t migraphx_type(py::dtype const& dtype) {
    static std::map<std::string, migraphx::shape::type_t> const types = {
        {"float32", migraphx::shape::float_type}, {"float16", migraphx::shape::half_type},
        {"float64", migraphx::shape::double_type}, {"int8", migraphx::shape::int8_type},
        {"uint8", migraphx::shape::uint8_type},   {"int16", migraphx::shape::int16_type},
        {"uint16", migraphx::shape::uint16_type}, {"int32", migraphx::shape::int32_type},
        {"uint32", migraphx::shape::uint32_type}, {"int64", migraphx::shape::int64_type},
        {"uint64", migraphx::shape::uint64_type}, {"bool", migraphx::shape::bool_type},
        {"bfloat16", migraphx::shape::bf16_type}};
    auto const type = types.find(py::str(dtype).cast<std::string>());
    if (type == types.end())
        throw std::invalid_argument("Calibration inputs are NumPy arrays of a standard type or bfloat16, not " +
                                    py::str(dtype).cast<std::string>() + ".");
    return type->second;
}

// A NumPy array copied into a host argument.
migraphx::argument argument_of(py::handle item) {
    auto const array = py::array::ensure(item, py::array::c_style);
    if (!array) throw py::type_error("Calibration inputs are NumPy arrays.");
    std::vector<std::size_t> lens(array.shape(), array.shape() + array.ndim());
    migraphx::argument argument{migraphx::shape{migraphx_type(array.dtype()), lens}};
    std::memcpy(argument.data(), array.data(), static_cast<std::size_t>(array.nbytes()));
    return argument;
}

py::tuple shape_tuple(migraphx::shape const& s) {
    std::vector<std::size_t> const lens = s.dynamic() ? s.max_lens() : s.lens();
    py::tuple t(lens.size());
    for (std::size_t i = 0; i < lens.size(); ++i) t[i] = lens[i];
    return t;
}

}  // namespace

PYBIND11_MODULE(_chainner_migraphx, module) {
    module.doc() = "chaiNNer-C's MIGraphX runner (native/migraphx_runner)";
    module.attr("MAX_STREAMS") = max_streams;
    module.attr("MIGRAPHX_VERSION") = cn_migx::migraphx_version();

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
                    throw std::invalid_argument("The program stand-in returns a C-contiguous " +
                                                std::string(py::str(expected.dtype())) + " array of shape (" +
                                                std::to_string(output_channels) + ", " +
                                                std::to_string(tile.height * scale) + ", " +
                                                std::to_string(tile.width * scale) + ").");
                std::memcpy(output, made.data(), static_cast<std::size_t>(made.nbytes()));
            };
            reference_upscale(frame.job, frame.inputs.planes.data(), frame.outputs.planes.data(), input_channels,
                              output_channels, in, out, run, clip);
        },
        "The runner's frame loop on the CPU, with `engine` (a function from a tile's input tensor to its output "
        "tensor) for MIGraphX; writes the result planes.",
        py::arg("inputs"), py::arg("outputs"), py::arg("scale"), py::arg("bounds"), py::arg("input_channels"),
        py::arg("output_channels"), py::arg("input_precision"), py::arg("output_precision"), py::arg("engine"),
        py::kw_only(), py::arg("bits") = 8, py::arg("full_range") = true, py::arg("clip") = false);

    module.def(
        "load_op_library", [](py::object path) { return cn_migx::load_op_library(*path_of(path)); },
        "Loads a custom-op library (custom_op.h) and registers its ops with MIGraphX; their names.", py::arg("path"));

    py::class_<Program>(module, "Program",
                        "A compiled MIGraphX program: its inputs and outputs, and runs on device pointers.")
        .def_property_readonly("target", &Program::target)
        .def_property_readonly("gfx", &Program::gfx)
        .def_property_readonly("device", &Program::device)
        .def_property_readonly("compute_type", &Program::compute_type)
        .def_property_readonly("custom_ops", &Program::custom_ops)
        .def_property_readonly("input_names", &Program::input_names)
        .def_property_readonly("input_shapes",
                               [](Program const& p) {
                                   py::list shapes;
                                   for (auto const& s : p.input_shapes()) shapes.append(shape_tuple(s));
                                   return shapes;
                               })
        .def_property_readonly("input_types",
                               [](Program const& p) {
                                   std::vector<std::string> types;
                                   for (auto const& s : p.input_shapes()) types.push_back(cn_migx::type_name(s.type()));
                                   return types;
                               })
        .def_property_readonly("output_shapes",
                               [](Program const& p) {
                                   py::list shapes;
                                   for (auto const& s : p.output_shapes()) shapes.append(shape_tuple(s));
                                   return shapes;
                               })
        .def_property_readonly("output_types",
                               [](Program const& p) {
                                   std::vector<std::string> types;
                                   for (auto const& s : p.output_shapes())
                                       types.push_back(cn_migx::type_name(s.type()));
                                   return types;
                               })
        .def(
            "to_bytes",
            [](Program const& p) {
                std::vector<char> data;
                {
                    py::gil_scoped_release release;
                    data = p.save();
                }
                return py::bytes(data.data(), data.size());
            },
            "The program as save() writes it.")
        .def(
            "save",
            [](Program const& p, py::object path) {
                std::string const target = *path_of(path);
                py::gil_scoped_release release;
                std::vector<char> const data = p.save();
                std::ofstream file(cn_migx::utf8_path(target), std::ios::binary);
                file.write(data.data(), static_cast<std::streamsize>(data.size()));
                file.close();
                if (!file) throw std::runtime_error("The program could not be written to " + target + ".");
            },
            py::arg("path"))
        .def(
            "run",
            [](Program& p, std::map<std::string, std::uintptr_t> const& inputs, std::vector<std::uintptr_t> const& outputs,
               std::uintptr_t stream) {
                std::map<std::string, void*> in;
                for (auto const& [name, pointer] : inputs) in[name] = reinterpret_cast<void*>(pointer);
                std::vector<void*> out;
                for (std::uintptr_t pointer : outputs) out.push_back(reinterpret_cast<void*>(pointer));
                py::gil_scoped_release release;
                p.run(in, out, reinterpret_cast<hipStream_t>(stream));
            },
            "Runs once: inputs by name and outputs in order as device pointers, ordered on the HIP stream `stream` "
            "(host pointers for a ref program, which returns when done).",
            py::arg("inputs"), py::arg("outputs"), py::arg("stream") = 0)
        .def("close", [](Program& p) {
            py::gil_scoped_release release;
            p.close();
        });

    module.def(
        "compile_onnx",
        [](py::object model, int device, bool fp16, bool fp8, std::optional<std::map<std::string, std::vector<std::size_t>>> input_dims,
           bool exhaustive_tune, std::string target, py::sequence calibration) {
            cn_migx::CompileOptions options;
            options.device = device;
            options.target = std::move(target);
            options.fp16 = fp16;
            options.fp8 = fp8;
            options.exhaustive_tune = exhaustive_tune;
            if (input_dims) options.input_dims = *input_dims;
            for (py::handle inputs : calibration) {
                std::map<std::string, migraphx::argument> arguments;
                for (auto const& [name, array] : py::reinterpret_borrow<py::dict>(inputs))
                    arguments[name.cast<std::string>()] = argument_of(array);
                options.calibration.push_back(std::move(arguments));
            }
            std::optional<std::string> const path = path_of(model);
            std::string const bytes = path ? *path : model.cast<std::string>();
            py::gil_scoped_release release;
            return Program::compile(bytes, path.has_value(), options);
        },
        "Compiles an ONNX model (bytes, or a path whose external data lies beside it) for the GPU `device`, or for "
        "target 'ref' (MIGraphX's CPU reference). Nodes of domain "
        "chainner run the custom ops of loaded libraries. fp16 or fp8 quantize it (fp8 calibrates with "
        "`calibration`, a few dicts of typical inputs); input_dims fixes inputs' sizes.",
        py::arg("model"), py::kw_only(), py::arg("device") = 0, py::arg("fp16") = false, py::arg("fp8") = false,
        py::arg("input_dims") = py::none(), py::arg("exhaustive_tune") = false, py::arg("target") = "gpu",
        py::arg("calibration") = py::tuple());

    module.def(
        "load_program",
        [](py::object source, int device) {
            std::optional<std::string> const path = path_of(source);
            std::string const data = path ? read_bytes(*path) : source.cast<std::string>();
            py::gil_scoped_release release;
            return Program::load(data.data(), data.size(), device);
        },
        "A saved program (Program.save's file or Program.to_bytes' bytes) on a HIP device.", py::arg("source"),
        py::arg("device"));

    py::class_<Runner>(module, "Runner",
                       "A saved GPU program loaded `streams` times; Python decompresses compressed programs first.")
        .def(py::init([](py::bytes const& program, int device, int streams) {
                 char* data = nullptr;
                 Py_ssize_t size = 0;
                 if (PyBytes_AsStringAndSize(program.ptr(), &data, &size) != 0) throw py::error_already_set();
                 py::gil_scoped_release release;
                 return std::make_unique<Runner>(data, static_cast<std::size_t>(size), device, streams);
             }),
             py::arg("program"), py::arg("device"), py::arg("streams"))
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
        .def(
            "capture_failure", [](Runner const&) { return std::string(); },
            "Always empty: the runner does not capture HIP graphs (TensorRT's runner reports a failed CUDA graph "
            "capture here).")
        .def("close", [](Runner& r) {
            py::gil_scoped_release release;
            r.close();
        });
}
