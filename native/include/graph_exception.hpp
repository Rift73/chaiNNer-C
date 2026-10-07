/* Public CPython handled-exception state for native application catch blocks.
 * This is separate from the active error indicator: callbacks inside a catch
 * must see sys.exc_info() and automatic __context__ chaining, then the caller's
 * previous handled state must be restored on every exit path. */
#pragma once
#include <pybind11/pybind11.h>

class GraphHandledException {
    PyObject *type_ = nullptr, *value_ = nullptr, *trace_ = nullptr;
public:
    explicit GraphHandledException(const pybind11::error_already_set &error) {
        PyErr_GetExcInfo(&type_, &value_, &trace_);
        PyErr_SetExcInfo(Py_XNewRef(error.type().ptr()), Py_XNewRef(error.value().ptr()),
                         Py_XNewRef(error.trace().ptr()));
    }
    ~GraphHandledException() { PyErr_SetExcInfo(type_, value_, trace_); }
    GraphHandledException(const GraphHandledException &) = delete;
    GraphHandledException &operator=(const GraphHandledException &) = delete;
};
