/* Atomic file writes for Python (GPL-3.0; native/include/atomic_file.hpp): a FileError becomes the OSError Python's
 * own open() or write() raises for the final path, and file I/O runs without the GIL. */
#include "atomic_file.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl/filesystem.h>

#include <cstring>
#include <string>

namespace py = pybind11;
namespace {

using cn::files::AtomicTarget;

// OSError(errno, strerror(errno), path) as open() and write() raise it, the text from this module's CRT; a Windows
// error the CRT does not map: OSError with winerror and FormatMessage's text.
void raise_file_error(const cn::files::FileError &error) {
    const std::wstring &name = error.path().native();
    PyObject *path = PyUnicode_FromWideChar(name.c_str(), Py_ssize_t(name.size()));
    if (!path) return;
    if (error.windows()) {
        PyErr_SetExcFromWindowsErrWithFilenameObject(PyExc_OSError, error.error(), path);
    } else {
        char message[128];
        strerror_s(message, sizeof message, error.error());
        if (PyObject *exception = PyObject_CallFunction(PyExc_OSError, "isO", error.error(), message, path)) {
            PyErr_SetObject(reinterpret_cast<PyObject *>(Py_TYPE(exception)), exception);
            Py_DECREF(exception);
        }
    }
    Py_DECREF(path);
}

bool atomic_write(const std::filesystem::path &path, const py::buffer &data) {
    Py_buffer view;
    if (PyObject_GetBuffer(data.ptr(), &view, PyBUF_SIMPLE) != 0) throw py::error_already_set();
    // Released with the GIL held, after the write: the exporter cannot resize the bytes meanwhile.
    struct Release {
        Py_buffer &view;
        ~Release() { PyBuffer_Release(&view); }
    } release_view{view};
    py::gil_scoped_release release;
    return cn::files::write_atomic(path, {static_cast<const uint8_t *>(view.buf), size_t(view.len)});
}

bool atomic_move(const std::filesystem::path &source, const std::filesystem::path &path) {
    py::gil_scoped_release release;
    return cn::files::move_atomic(source, path);
}

} // namespace

void cn_bind_atomic_file(py::module_ &module) {
    py::register_local_exception_translator([](std::exception_ptr pointer) {
        try {
            if (pointer) std::rethrow_exception(pointer);
        } catch (const cn::files::FileError &error) {
            raise_file_error(error);
        }
    });
    module.def("atomic_write", &atomic_write, py::arg("path"), py::arg("data"),
               "Writes data to path through a temporary file in its folder; False when the file already held these "
               "bytes (left untouched).");
    module.def("atomic_move", &atomic_move, py::arg("source"), py::arg("path"),
               "Moves the finished file source to path through a temporary file in path's folder; False when path "
               "already held the same bytes (left untouched; source stays).");
    py::class_<AtomicTarget>(module, "AtomicTarget",
                             "The temporary file in path's folder that a writer fills before commit() renames it onto "
                             "path; as a context manager it commits on success and abandons on an exception.")
        .def(py::init<std::filesystem::path>(), py::arg("path"))
        .def_property_readonly("temp", [](const AtomicTarget &target) { return target.temp().native(); })
        .def("commit",
             [](AtomicTarget &target) {
                 py::gil_scoped_release release;
                 return target.commit();
             })
        .def("abandon",
             [](AtomicTarget &target) {
                 py::gil_scoped_release release;
                 target.abandon();
             })
        .def("__enter__", [](py::object self) { return self; })
        .def("__exit__", [](AtomicTarget &target, const py::object &type, const py::object &, const py::object &) {
            const bool failed = !type.is_none();
            py::gil_scoped_release release;
            if (failed) target.abandon();
            else if (target.open()) target.commit();
            return false;
        });
}
