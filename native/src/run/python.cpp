/* The run engine's Python boundary: Value <-> Python conversion, NodeError/Aborted <-> Python exceptions, and the
 * test entries run_native and native_schema_ids (native/tests/native_differential.py). */
#include "engine.hpp"
#include "kernel_status.hpp"

#include <pybind11/gil_safe_call_once.h>
#include <pybind11/numpy.h>

#include <exception>
#include <limits>
#include <mutex>

namespace py = pybind11;

namespace cn::run {
namespace {

constexpr const char *image_capsule = "chainner_c.run.image";

/* The Python names the conversion uses, imported at the first conversion. */
struct Names {
    py::object pure_path, path, enumeration, color, lazy, aborted, normalized_readonly, normalize, require;
};

const Names &names()
{
    PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<Names> storage;
    return storage
        .call_once_and_store_result([] {
            py::module_ pathlib = py::module_::import("pathlib");
            py::module_ api = py::module_::import("api");
            py::module_ buffers = py::module_::import("nodes.impl.native_buffers");
            return Names{pathlib.attr("PurePath"),
                         pathlib.attr("Path"),
                         py::module_::import("enum").attr("Enum"),
                         py::module_::import("nodes.impl.color.color").attr("Color"),
                         api.attr("Lazy"),
                         api.attr("Aborted"),
                         buffers.attr("normalized_readonly"),
                         py::module_::import("nodes.impl.image_utils").attr("normalize"),
                         py::module_::import("numpy").attr("require")};
        })
        .get_stored();
}

bool is_instance(py::handle object, const py::object &type)
{
    int result = PyObject_IsInstance(object.ptr(), type.ptr());
    if (result < 0) throw py::error_already_set();
    return result != 0;
}

void release_capsule(PyObject *capsule)
{
    delete static_cast<std::shared_ptr<const ImageStorage> *>(PyCapsule_GetPointer(capsule, image_capsule));
}

/* The engine storage an array views (its base is an engine image capsule); null for any other array. */
std::shared_ptr<const ImageStorage> engine_storage(const py::array &array)
{
    py::object base = array.base();
    if (!base || !PyCapsule_IsValid(base.ptr(), image_capsule)) return nullptr;
    return *static_cast<std::shared_ptr<const ImageStorage> *>(PyCapsule_GetPointer(base.ptr(), image_capsule));
}

/* An array's image shape; a TypeError for anything the engine's Image cannot hold. */
struct Shape {
    int32_t height, width, channels;
};

Shape image_shape(const py::array &array)
{
    py::ssize_t channels = array.ndim() == 3 ? array.shape(2) : 1;
    if ((array.ndim() != 2 && array.ndim() != 3) || (channels != 1 && channels != 3 && channels != 4) ||
        array.shape(0) > std::numeric_limits<int32_t>::max() || array.shape(1) > std::numeric_limits<int32_t>::max())
        throw py::type_error(py::str("The native engine takes images of 1, 3 or 4 channels, not an array of shape {}")
                                 .format(array.attr("shape")));
    return {int32_t(array.shape(0)), int32_t(array.shape(1)), int32_t(channels)};
}

bool c_contiguous_f32(const py::array &array)
{
    return py::isinstance<py::array_t<float>>(array) && (array.flags() & py::array::c_style) &&
           (array.flags() & py::detail::npy_api::NPY_ARRAY_ALIGNED_);
}

Image borrow(const py::array &array, const Shape &shape)
{
    auto *data = static_cast<float *>(const_cast<void *>(array.data()));
    std::shared_ptr<const ImageStorage> storage = engine_storage(array);
    if (!storage) storage = std::make_shared<ImageStorage>(data, py_ref(array));
    return Image{std::move(storage), data, shape.height, shape.width, shape.channels};
}

/* array itself when it is C-contiguous and aligned, else the copy np.require(array, None, "CA") makes (as
 * native.f32 does), with its dtype and bits, read-only: only the engine holds it. */
py::array c_aligned(const py::array &array)
{
    if ((array.flags() & py::array::c_style) && (array.flags() & py::detail::npy_api::NPY_ARRAY_ALIGNED_)) return array;
    py::array copy = names().require(array, py::none(), "CA");
    copy.attr("setflags")(py::arg("write") = false);
    return copy;
}

/* cn_pixels_convert_checked's type code of a native-order dtype (native_buffers._TYPES); -1 for any other. */
int pixel_type(const py::dtype &dtype)
{
    if (dtype.byteorder() != '=' && dtype.byteorder() != '|') return -1;
    switch (dtype.kind()) {
    case 'f': return dtype.itemsize() == 4 ? 0 : dtype.itemsize() == 8 ? 1 : -1;
    case 'u': return dtype.itemsize() == 1 ? 2 : dtype.itemsize() == 2 ? 3 : dtype.itemsize() == 4 ? 7 : dtype.itemsize() == 8 ? 9 : -1;
    case 'i': return dtype.itemsize() == 1 ? 4 : dtype.itemsize() == 2 ? 5 : dtype.itemsize() == 4 ? 6 : dtype.itemsize() == 8 ? 8 : -1;
    case 'b': return dtype.itemsize() == 1 ? 10 : -1;
    default: return -1;
    }
}

/* normalize(array) (image_utils): converted_pixels' clip to float32, in C order. */
Image normalized_copy(const py::array &array, const Shape &shape, int type)
{
    py::array source = c_aligned(array);
    MutableImage image = MutableImage::allocate(shape.height, shape.width, shape.channels);
    cn_status status;
    {
        py::gil_scoped_release unlocked;
        status = cn_pixels_convert_checked(source.data(), image.data(), size_t(source.size()), type, 0, 1, nullptr);
    }
    check(status);
    return std::move(image).publish();
}

Image image_from_array(const py::array &array, ImageMode mode)
{
    Shape shape = image_shape(array);
    if (mode == ImageMode::enforced) {
        if (!py::isinstance<py::array_t<float>>(array))
            throw py::type_error(py::str("An enforced image is float32, not {}").format(array.dtype()));
        return borrow(c_aligned(array), shape);
    }
    if (c_contiguous_f32(array)) {
        bool same;
        if (std::shared_ptr<const ImageStorage> storage = engine_storage(array)) {
            int normalized = 0;
            check(cn_pixels_normalized_f32(static_cast<const float *>(array.data()), size_t(array.size()), &normalized));
            same = normalized != 0;
        } else {
            same = names().normalized_readonly(array).is(array);
        }
        if (same) return borrow(array, shape);
    }
    int type = pixel_type(array.dtype());
    if (type >= 0) return normalized_copy(array, shape, type);
    return image_from_array(py::array(names().normalize(array)), ImageMode::enforced);
}

/* The whole of a borrowed array: that array itself. */
bool whole(py::handle owner, const Image &image)
{
    if (!py::isinstance<py::array>(owner)) return false;
    auto array = py::reinterpret_borrow<py::array>(owner);
    return array.data() == image.data && array.shape(0) == image.height && array.shape(1) == image.width &&
           array.size() == py::ssize_t(image.values());
}

py::object image_to_python(const Image &image)
{
    if (const PyRef &owner = image.storage->owner(); owner.handle) {
        py::handle object(static_cast<PyObject *>(owner.handle.get()));
        if (whole(object, image)) return py::reinterpret_borrow<py::object>(object);
    }
    std::vector<py::ssize_t> shape{image.height, image.width};
    std::vector<py::ssize_t> strides{py::ssize_t(sizeof(float)) * image.width * image.channels,
                                     py::ssize_t(sizeof(float)) * image.channels};
    if (image.channels != 1) {
        shape.push_back(image.channels);
        strides.push_back(sizeof(float));
    }
    py::array array;
    if (image.values() == 0) {
        array = py::array_t<float>(shape);
    } else {
        auto *kept = new std::shared_ptr<const ImageStorage>(image.storage);
        PyObject *capsule = PyCapsule_New(kept, image_capsule, release_capsule);
        if (!capsule) {
            delete kept;
            throw py::error_already_set();
        }
        array = py::array(py::dtype::of<float>(), shape, strides, image.data, py::reinterpret_steal<py::object>(capsule));
    }
    array.attr("setflags")(py::arg("write") = false);
    return array;
}

/* A LazyValue reading an api.Lazy (with the GIL, once). */
class PythonLazy final : public LazyValue {
public:
    PythonLazy(PyRef lazy, ImageMode mode) : lazy_(std::move(lazy)), mode_(mode) {}

    const Value &value() override
    {
        std::call_once(once_, [this] {
            py::gil_scoped_acquire held;
            try {
                try {
                    value_ = to_value(py::handle(object()).attr("value"), mode_);
                } catch (py::error_already_set &error) {
                    rethrow_python_error(error);
                }
            } catch (...) {
                error_ = std::current_exception();
            }
        });
        if (error_) std::rethrow_exception(error_);
        return value_;
    }

    PyObject *object() const noexcept { return static_cast<PyObject *>(lazy_.handle.get()); }

private:
    PyRef lazy_;
    ImageMode mode_;
    std::once_flag once_;
    Value value_;
    std::exception_ptr error_;
};

py::object lazy_to_python(const std::shared_ptr<LazyValue> &lazy)
{
    if (auto *python = dynamic_cast<PythonLazy *>(lazy.get())) return py::reinterpret_borrow<py::object>(python->object());
    return names().lazy(py::cpp_function([lazy]() {
        const Value *value;
        {
            py::gil_scoped_release unlocked;
            value = &lazy->value();
        }
        return to_python(*value);
    }));
}

py::object exception_type(const std::string &name)
{
    size_t dot = name.rfind('.');
    py::object type = dot == std::string::npos
                          ? py::module_::import("builtins").attr(name.c_str())
                          : py::module_::import(name.substr(0, dot).c_str()).attr(name.substr(dot + 1).c_str());
    if (!PyType_Check(type.ptr()) || !PyType_IsSubtype(reinterpret_cast<PyTypeObject *>(type.ptr()),
                                                       reinterpret_cast<PyTypeObject *>(PyExc_Exception)))
        throw py::type_error("Not an exception type: " + name);
    return type;
}

/* A NodeError as the Python exception it names (RuntimeError "<type>: <message>" when the name is no exception
 * type); Aborted as api's Aborted. */
void translate(std::exception_ptr pending)
{
    try {
        std::rethrow_exception(pending);
    } catch (const NodeError &error) {
        try {
            PyErr_SetObject(exception_type(error.python_type).ptr(), py::str(error.what()).ptr());
        } catch (py::error_already_set &) {
            PyErr_SetString(PyExc_RuntimeError, (error.python_type + ": " + error.what()).c_str());
        } catch (const py::type_error &) {
            PyErr_SetString(PyExc_RuntimeError, (error.python_type + ": " + error.what()).c_str());
        }
    } catch (const Aborted &) {
        PyErr_SetNone(names().aborted.ptr());
    }
}

/* run_native's context: never aborted, no node id, storage directory or settings. */
class DirectContext final : public NodeContext {
public:
    bool aborted() const noexcept override { return false; }
    void check_aborted() const override {}
    std::string_view node_id() const noexcept override { return {}; }
    const std::filesystem::path &storage_dir() const noexcept override { return storage_dir_; }
    const Value *setting(std::string_view) const noexcept override { return nullptr; }

private:
    std::filesystem::path storage_dir_;
};

py::list run_native(const std::string &schema_id, const py::list &inputs)
{
    const Entry *entry = find_entry(schema_id);
    if (!entry) throw py::key_error("No native entry point for " + schema_id);
    if (!entry->run) throw py::type_error(schema_id + " is a generator; run_native runs regular nodes");
    std::vector<Value> values;
    values.reserve(inputs.size());
    for (py::handle input : inputs) values.push_back(to_value(input, ImageMode::enforced));
    std::vector<Value> outputs;
    {
        py::gil_scoped_release unlocked;
        DirectContext context;
        entry->run(context, values, outputs);
    }
    py::list result;
    for (const Value &output : outputs) result.append(to_python(output));
    return result;
}

py::list native_schema_ids()
{
    py::list ids;
    for (const auto &[id, entry] : entries()) ids.append(id);
    return ids;
}

}  // namespace

PyRef py_ref(py::handle object)
{
    return PyRef{std::shared_ptr<void>(object.inc_ref().ptr(), [](void *held) {
        if (Py_IsFinalizing()) return;
        PyGILState_STATE state = PyGILState_Ensure();
        Py_DECREF(static_cast<PyObject *>(held));
        PyGILState_Release(state);
    })};
}

Value to_value(py::handle object, ImageMode mode)
{
    if (object.is_none()) return std::monostate{};
    if (PyBool_Check(object.ptr())) return object.ptr() == Py_True;
    const Names &python = names();
    if (is_instance(object, python.enumeration)) return to_value(object.attr("value"), mode);
    if (PyLong_Check(object.ptr())) {
        int overflow = 0;
        long long number = PyLong_AsLongLongAndOverflow(object.ptr(), &overflow);
        if (number == -1 && PyErr_Occurred()) throw py::error_already_set();
        if (overflow) return py_ref(object);
        return int64_t(number);
    }
    if (PyFloat_Check(object.ptr())) return PyFloat_AsDouble(object.ptr());
    if (PyUnicode_Check(object.ptr())) {
        Py_ssize_t size = 0;
        const char *text = PyUnicode_AsUTF8AndSize(object.ptr(), &size);
        if (!text) {
            PyErr_Clear();  // a lone surrogate: kept as the Python string
            return py_ref(object);
        }
        return std::string(text, size_t(size));
    }
    if (is_instance(object, python.pure_path)) {
        py::str text(object);
        Py_ssize_t size = 0;
        wchar_t *wide = PyUnicode_AsWideCharString(text.ptr(), &size);
        if (!wide) throw py::error_already_set();
        Path path{std::filesystem::path(std::wstring(wide, size_t(size)))};
        PyMem_Free(wide);
        return path;
    }
    if (is_instance(object, python.color)) {
        Color color;
        for (py::handle channel : object.attr("value")) color.values.push_back(channel.cast<double>());
        return color;
    }
    if (py::isinstance<py::array>(object)) return image_from_array(py::reinterpret_borrow<py::array>(object), mode);
    if (is_instance(object, python.lazy)) return std::shared_ptr<LazyValue>(std::make_shared<PythonLazy>(py_ref(object), mode));
    return py_ref(object);
}

py::object to_python(const Value &value)
{
    return std::visit(
        [](const auto &v) -> py::object {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, std::monostate>) return py::none();
            else if constexpr (std::is_same_v<T, int64_t>) return py::int_(v);
            else if constexpr (std::is_same_v<T, double>) return py::float_(v);
            else if constexpr (std::is_same_v<T, bool>) return py::bool_(v);
            else if constexpr (std::is_same_v<T, std::string>) return py::str(v);
            else if constexpr (std::is_same_v<T, Path>) {
                const std::wstring &native = v.value.native();
                PyObject *text = PyUnicode_FromWideChar(native.c_str(), Py_ssize_t(native.size()));
                if (!text) throw py::error_already_set();
                return names().path(py::reinterpret_steal<py::object>(text));
            } else if constexpr (std::is_same_v<T, Color>) {
                py::tuple channels(v.values.size());
                for (size_t i = 0; i < v.values.size(); ++i) channels[i] = py::float_(v.values[i]);
                return names().color(channels);
            } else if constexpr (std::is_same_v<T, Image>) return image_to_python(v);
            else if constexpr (std::is_same_v<T, PyRef>) return py::reinterpret_borrow<py::object>(static_cast<PyObject *>(v.handle.get()));
            else return lazy_to_python(v);
        },
        value);
}

void rethrow_python_error(py::error_already_set &error)
{
    if (error.matches(names().aborted)) throw Aborted{};
    py::object type = error.type();
    std::string module = py::str(type.attr("__module__"));
    std::string name = py::str(type.attr("__qualname__"));
    throw NodeError(module == "builtins" ? name : module + "." + name, py::str(error.value()));
}

}  // namespace cn::run

void cn_bind_run_engine(py::module_ &module)
{
    py::register_local_exception_translator(cn::run::translate);
    module.def("run_native", &cn::run::run_native, py::arg("schema_id"), py::arg("inputs"),
               "Run a node's native entry point on its enforced inputs (the values process.enforce_inputs passes); "
               "returns its outputs before output enforcement, images as read-only float32 arrays.");
    module.def("native_schema_ids", &cn::run::native_schema_ids, "The schema ids with a native entry point, sorted.");
    module.def(
        "run_engine_value",
        [](py::handle value, bool normalize) {
            return cn::run::to_python(
                cn::run::to_value(value, normalize ? cn::run::ImageMode::normalized : cn::run::ImageMode::enforced));
        },
        py::arg("value"), py::arg("normalize") = false,
        "A value through the engine's conversion and back (images enforced, or normalized as ImageOutput.enforce).");
}
