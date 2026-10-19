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
    py::object pure_path, path, enumeration, color, seed, lazy, aborted, generator, normalized_readonly, normalize,
        require;
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
                         py::module_::import("nodes.utils.seed").attr("Seed"),
                         api.attr("Lazy"),
                         api.attr("Aborted"),
                         api.attr("Generator"),
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

/* The exception a NodeError names, with its message (RuntimeError "<type>: <message>" when the name is no
 * exception type). */
py::object exception_object(const NodeError &error)
{
    try {
        return exception_type(error.python_type)(py::str(error.what()));
    } catch (py::error_already_set &) {
    } catch (const py::type_error &) {
    }
    return py::reinterpret_borrow<py::object>(PyExc_RuntimeError)(py::str(error.python_type + ": " + error.what()));
}

/* A generator item's iterated outputs as its supplier yields them: one value, or a tuple of several. */
py::object item_to_python(const std::vector<Value> &item)
{
    if (item.size() == 1) return to_python(item[0]);
    py::tuple values(item.size());
    for (size_t i = 0; i < item.size(); ++i) values[i] = to_python(item[i]);
    return values;
}

/* A NodeError as the Python exception it names; Aborted as api's Aborted. */
void translate(std::exception_ptr pending)
{
    try {
        std::rethrow_exception(pending);
    } catch (const NodeError &error) {
        py::object exception = exception_object(error);
        PyErr_SetObject(reinterpret_cast<PyObject *>(Py_TYPE(exception.ptr())), exception.ptr());
    } catch (const Aborted &) {
        PyErr_SetNone(names().aborted.ptr());
    }
}

/* run_native's context: never aborted; the given node id, storage directory and package settings. */
class RunContext final : public NodeContext {
public:
    RunContext(std::map<std::string, Value, std::less<>> settings, std::string node_id, std::filesystem::path storage_dir)
        : settings_(std::move(settings)), node_id_(std::move(node_id)), storage_dir_(std::move(storage_dir))
    {
    }
    bool aborted() const noexcept override { return false; }
    void check_aborted() const override {}
    std::string_view node_id() const noexcept override { return node_id_; }
    const std::filesystem::path &storage_dir() const noexcept override { return storage_dir_; }
    const Value *setting(std::string_view key) const noexcept override
    {
        auto found = settings_.find(key);
        return found == settings_.end() ? nullptr : &found->second;
    }

private:
    std::map<std::string, Value, std::less<>> settings_;
    std::string node_id_;
    std::filesystem::path storage_dir_;
};

/* A native generator's items, as the iterable its Generator's supplier returns: each next() describes and
 * materializes one item with the GIL released; an item's NodeError is yielded as its exception (an item error, as
 * Generator.from_list yields them). It keeps the context and inputs the source was created with. */
class NativeItems {
public:
    NativeItems(std::shared_ptr<RunContext> context, std::vector<Value> inputs, std::unique_ptr<ItemSource> source)
        : context_(std::move(context)), inputs_(std::move(inputs)), source_(std::move(source))
    {
    }

    py::object next()
    {
        std::vector<Value> item;
        std::exception_ptr failure;
        bool more;
        {
            py::gil_scoped_release unlocked;
            int64_t token = 0;
            more = source_->describe(token);
            if (more) {
                try {
                    source_->materialize(token, item);
                } catch (const NodeError &) {
                    failure = std::current_exception();
                }
            }
        }
        if (!more) throw py::stop_iteration();
        if (failure) {
            try {
                std::rethrow_exception(failure);
            } catch (const NodeError &error) {
                return exception_object(error);
            }
        }
        return item_to_python(item);
    }

    const ItemSource &source() const noexcept { return *source_; }

private:
    std::shared_ptr<RunContext> context_;
    std::vector<Value> inputs_;
    std::unique_ptr<ItemSource> source_;
};

/* An api.Generator over a native item source: expected_length, fail_fast and source_paths from the source; its
 * supplier returns the items once (the executor calls it once per run). */
py::object native_generator(const std::shared_ptr<NativeItems> &items)
{
    const ItemSource &source = items->source();
    std::vector<std::filesystem::path> paths;
    py::object source_paths = py::none();
    if (source.source_paths(paths)) {
        py::tuple tuple(paths.size());
        for (size_t i = 0; i < paths.size(); ++i) tuple[i] = to_python(Path{paths[i]});
        source_paths = tuple;
    }
    auto pending = std::make_shared<std::shared_ptr<NativeItems>>(items);
    py::cpp_function supplier([pending]() -> py::object {
        if (!*pending) throw py::value_error("A native generator's items can be read once");
        py::object iterable = py::cast(*pending);
        pending->reset();
        return iterable;
    });
    return names().generator(supplier, source.expected_length(), source.fail_fast(), py::none(), source_paths);
}

/* _chainner_graph.NativeDeclined, set when the module is bound. */
PyObject *native_declined = nullptr;

/* Runs a node's native entry point on enforced inputs. A regular node returns its outputs; a generator returns
 * [api.Generator over its items, *static outputs]. An entry point that declines to Python raises NativeDeclined. */
py::list run_native(const std::string &schema_id, const py::list &inputs, const py::object &settings,
                    const std::string &node_id, const py::object &storage_dir)
{
    const Entry *entry = find_entry(schema_id);
    if (!entry) throw py::key_error("No native entry point for " + schema_id);
    std::vector<Value> values;
    values.reserve(inputs.size());
    for (py::handle input : inputs) values.push_back(to_value(input, ImageMode::enforced));
    std::map<std::string, Value, std::less<>> parsed;
    if (!settings.is_none()) {
        for (auto [key, value] : py::reinterpret_borrow<py::dict>(settings))
            parsed.emplace(py::str(key), to_value(value, ImageMode::enforced));
    }
    std::filesystem::path directory;
    if (!storage_dir.is_none()) directory = std::get<Path>(to_value(storage_dir, ImageMode::enforced)).value;
    auto context = std::make_shared<RunContext>(std::move(parsed), node_id, std::move(directory));
    std::vector<Value> outputs;
    py::list result;
    std::unique_ptr<ItemSource> source;
    try {
        py::gil_scoped_release unlocked;
        if (entry->run)
            entry->run(*context, values, outputs);
        else
            source = entry->generator(*context, values, outputs);
    } catch (const RunPython &) {
        PyErr_SetString(native_declined, schema_id.c_str());
        throw py::error_already_set();
    }
    if (!entry->run) {
        if (!source) throw py::value_error(schema_id + " returned no item source");
        result.append(native_generator(std::make_shared<NativeItems>(context, std::move(values), std::move(source))));
    }
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

namespace {

/* References dropped by threads without the GIL, released in a batch by the next thread that takes it (DESIGN.md
 * amendment A2: no GIL taken only to drop a reference). */
std::mutex deferred_lock;
std::vector<PyObject *> deferred;

}  // namespace

PyRef py_ref(py::handle object)
{
    return PyRef{std::shared_ptr<void>(object.inc_ref().ptr(), [](void *held) {
        if (Py_IsFinalizing()) return;
        if (PyGILState_Check()) {
            Py_DECREF(static_cast<PyObject *>(held));
            return;
        }
        std::lock_guard<std::mutex> locked(deferred_lock);
        deferred.push_back(static_cast<PyObject *>(held));
    })};
}

void release_deferred()
{
    std::vector<PyObject *> pending;
    {
        std::lock_guard<std::mutex> locked(deferred_lock);
        pending.swap(deferred);
    }
    for (PyObject *object : pending) Py_DECREF(object);
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
        // Channels are floats, or ints where Color's _norm kept one (Color.gray(0)); anything else stays itself.
        Color color;
        bool any_integral = false;
        std::vector<bool> integral;
        for (py::handle channel : object.attr("value")) {
            if (PyFloat_CheckExact(channel.ptr())) {
                color.values.push_back(PyFloat_AS_DOUBLE(channel.ptr()));
                integral.push_back(false);
                continue;
            }
            if (!PyLong_CheckExact(channel.ptr())) return py_ref(object);
            int overflow = 0;
            long long whole = PyLong_AsLongLongAndOverflow(channel.ptr(), &overflow);
            if (whole == -1 && PyErr_Occurred()) throw py::error_already_set();
            // Beyond 2**53 a double would not keep the int exactly
            if (overflow != 0 || whole > (1LL << 53) || whole < -(1LL << 53)) return py_ref(object);
            color.values.push_back(double(whole));
            integral.push_back(true);
            any_integral = true;
        }
        if (any_integral) color.integral = std::move(integral);
        return color;
    }
    if (is_instance(object, python.seed)) {
        int overflow = 0;
        long long seed = PyLong_AsLongLongAndOverflow(object.attr("value").ptr(), &overflow);
        if (seed == -1 && PyErr_Occurred()) throw py::error_already_set();
        if (overflow == 0) return Seed{seed};
        return py_ref(object);  // beyond int64: an entry point declines (RunPython)
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
                for (size_t i = 0; i < v.values.size(); ++i) {
                    if (i < v.integral.size() && v.integral[i])
                        channels[i] = py::int_(static_cast<long long>(v.values[i]));
                    else
                        channels[i] = py::float_(v.values[i]);
                }
                return names().color(channels);
            } else if constexpr (std::is_same_v<T, Seed>) return names().seed(py::int_(v.value));
            else if constexpr (std::is_same_v<T, Image>) return image_to_python(v);
            else if constexpr (std::is_same_v<T, PyRef>) return py::reinterpret_borrow<py::object>(static_cast<PyObject *>(v.handle.get()));
            else return lazy_to_python(v);
        },
        value);
}

py::object python_exception(const NodeError &error)
{
    return exception_object(error);
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
    cn::run::native_declined = PyErr_NewExceptionWithDoc(
        "_chainner_graph.NativeDeclined",
        "A native entry point declined to its node's Python body (run_engine.hpp RunPython); the message is the "
        "schema id.",
        nullptr, nullptr);
    if (!cn::run::native_declined) throw py::error_already_set();
    module.add_object("NativeDeclined", py::reinterpret_borrow<py::object>(cn::run::native_declined));
    py::class_<cn::run::NativeItems, std::shared_ptr<cn::run::NativeItems>>(module, "_NativeItems")
        .def("__iter__", [](py::object self) { return self; })
        .def("__next__", &cn::run::NativeItems::next);
    module.def("run_native", &cn::run::run_native, py::arg("schema_id"), py::arg("inputs"), py::kw_only(),
               py::arg("settings") = py::none(), py::arg("node_id") = "", py::arg("storage_dir") = py::none(),
               "Run a node's native entry point on its enforced inputs (the values process.enforce_inputs passes) "
               "with a context of the given package settings (a dict), node id and storage directory (a Path); "
               "returns its outputs before output enforcement, images as read-only float32 arrays. A generator "
               "returns [an api.Generator over its native items, *static outputs].");
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
