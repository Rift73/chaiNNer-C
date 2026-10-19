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
    explicit ImageStorage(size_t bytes);  // pooled, uninitialised, never null; NodeError("MemoryError") when out of memory
    ImageStorage(float *data, PyRef owner) noexcept : data_(data), owner_(std::move(owner)) {}
    ~ImageStorage();
    ImageStorage(const ImageStorage &) = delete;
    ImageStorage &operator=(const ImageStorage &) = delete;

    float *data() const noexcept { return data_; }
    /* The borrowed Python object; empty for pooled storage. */
    const PyRef &owner() const noexcept { return owner_; }

private:
    float *data_ = nullptr;
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

/* How a NumPy array becomes an Image.
 * - Enforced: a value that has passed the input's enforce (float32): borrowed when it is C-contiguous and aligned,
 *   else its C-ordered copy (np.require, as native.f32 makes it; the same bits) is.
 * - Normalized: an output, as ImageOutput.enforce normalizes it: borrowed when it is C-contiguous and
 *   native_buffers.normalized_readonly accepts it (an engine image when cn_pixels_normalized_f32 proves it), else
 *   converted by cn_pixels_convert_checked as normalize does (a float32 output reports no events). */
enum class ImageMode { enforced, normalized };

/* Python -> Value (GIL held): None, bool, int (int64 range, else a PyRef), float, str, pathlib.Path, chaiNNer's
 * Color, an Enum member (as its value), a NumPy array of 1, 3 or 4 channels (Image), api.Lazy (a LazyValue
 * reading it); anything else is a PyRef. */
Value to_value(pybind11::handle object, ImageMode mode);
/* Value -> Python (GIL held). An Image becomes a read-only float32 array over its storage (2-D for one channel);
 * the whole of a borrowed array is that array itself. */
pybind11::object to_python(const Value &value);

/* Raises a NodeError or Aborted from the pending Python exception (GIL held). */
[[noreturn]] void rethrow_python_error(pybind11::error_already_set &error);
/* The Python exception a NodeError names, with its message (GIL held). */
pybind11::object python_exception(const NodeError &error);

}  // namespace cn::run
