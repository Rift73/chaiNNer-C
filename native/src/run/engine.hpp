/* The run engine's internal interface (native/src/run): image storage, the registry of native entry points and the
 * Python boundary's value conversion. Native node entry points include run_engine.hpp (and nodes/dispatch.hpp, the
 * one nodes file that calls Python, for the bridges' cached dispatch facts), never this header. */
#pragma once

#include "run_engine.hpp"

#include <pybind11/pybind11.h>

#include <map>
#include <string>
#include <string_view>

namespace cn::run {

/* Image memory: a block of the NumPy pool (numpy_pool.hpp) the storage owns, or a Python buffer it keeps alive
 * (a borrowed NumPy array; the owner's last reference is dropped with the GIL). */
class ImageStorage {
public:
    // pooled, uninitialised, never null; std::bad_alloc when out of memory (MutableImage::allocate raises it as
    // NumPy's MemoryError text)
    explicit ImageStorage(size_t bytes);
    ImageStorage(float *data, PyRef owner) noexcept : data_(data), owner_(std::move(owner)) {}
    ~ImageStorage();
    ImageStorage(const ImageStorage &) = delete;
    ImageStorage &operator=(const ImageStorage &) = delete;

    float *data() const noexcept { return data_; }
    /* The borrowed Python object; empty for pooled storage. */
    const PyRef &owner() const noexcept { return owner_; }
    /* A pooled block's size (what every view of it keeps alive); 0 for borrowed storage. */
    size_t bytes() const noexcept { return bytes_; }

private:
    float *data_ = nullptr;
    size_t bytes_ = 0;
    bool pooled_ = false;
    PyRef owner_;
};

/* A registered schema: exactly one of the two entry points is set. */
struct Entry {
    NativeRun run = nullptr;
    NativeGeneratorRun generator = nullptr;
};

/* Every native entry point (native/src/run/nodes/all_nodes.inc), built at first use; read-only afterwards. */
const std::map<std::string, Entry, std::less<>> &entries();
const Entry *find_entry(std::string_view schema_id);

/* A Python object as a PyRef (a new reference). The last copy dropped on a thread holding the GIL releases it; on
 * any other thread it waits for release_deferred. */
PyRef py_ref(pybind11::handle object);
/* Releases the references dropped without the GIL (GIL held). */
void release_deferred();

/* Python -> Value (GIL held): None, bool, int (int64 range, else a PyRef), float, str, the platform's pathlib.Path
 * type (exact types: a subclass such as np.float64, np.str_ or PurePosixPath is a PyRef), chaiNNer's Color, an
 * Enum member (as its value), a NumPy array of 1, 3 or 4 channels (Image), api.Lazy (a LazyValue
 * reading it); anything else is a PyRef. An array is a value that has passed the input's enforce (float32; a
 * TypeError otherwise): borrowed when it is C-contiguous and aligned, else its C-ordered copy (np.require, as
 * native.f32 makes it; the same bits) is. */
Value to_value(pybind11::handle object);
/* ImageOutput.enforce of an engine image output (Run::enforce_outputs, when the output does not assume it
 * normalized): the image itself when normalize's clip to [0, 1] changes no bit, else normalize's float32 copy. */
Image normalized_output(const Image &image);
/* Value -> Python (GIL held). An Image becomes a read-only float32 array over its storage (2-D for one channel);
 * the whole of a borrowed array is that array itself. */
pybind11::object to_python(const Value &value);

/* A Python exception as a NodeError (GIL held): its type's name (qualified unless a builtin), str() of it, and the
 * exception object itself. */
NodeError node_error(pybind11::error_already_set &error);
/* The error of an item source's materialize() that handed back to Python, which it must not (run_engine.hpp
 * ItemSource): a RuntimeError naming the bug. */
NodeError materialize_handed_back();
/* Raises Aborted for api's Aborted, else node_error's NodeError (GIL held). */
[[noreturn]] void rethrow_python_error(pybind11::error_already_set &error);
/* The Python exception of a NodeError (GIL held), the one mapping of both: the original exception object when it
 * has one, else the type it names (an Exception subclass, builtin or module.Name) called with its message, else
 * RuntimeError("<type>: <message>"). */
pybind11::object python_exception(const NodeError &error);

}  // namespace cn::run
