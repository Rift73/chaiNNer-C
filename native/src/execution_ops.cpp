/* chaiNNer execution/path application control, adapted from its GPL-3.0 source.
 * Native state preserves Python object identity and ordered arbitrary-precision
 * arithmetic. CPython objects/range and pathlib host resolution remain engines.
 * Note and Execution Number have no substantive backend algorithm.
 */
#include "graph_python.hpp"
#include <memory>

namespace py = pybind11;
namespace {
using graphpy::O;

struct ActiveIteration {
    bool &running;
    explicit ActiveIteration(bool &value) : running(value) {
        if (running) throw py::value_error("generator already executing");
        running = true;
    }
    ~ActiveIteration() { running = false; }
};

O checked(PyObject *value) {
    if (!value) throw py::error_already_set();
    return py::reinterpret_steal<O>(value);
}
[[noreturn]] void exhausted() {
    PyErr_SetNone(PyExc_StopIteration);
    throw py::error_already_set();
}
O name(const py::dict &globals, const char *key) {
    py::str k(key);
    if (globals.contains(k)) return globals[k];
    return graphpy::builtin(key);
}
bool comparison(const O &a, const O &b, int op) {
    return graphpy::truth(checked(PyObject_RichCompare(a.ptr(), b.ptr(), op)));
}
bool option(const py::dict &globals, const O &value, const char *type, const char *member) {
    O expected = name(globals, type).attr(member);
    return comparison(value, expected, Py_EQ);
}
[[noreturn]] void not_implemented() {
    PyErr_SetNone(PyExc_NotImplementedError);
    throw py::error_already_set();
}

O directory_into(const O &directory, const py::tuple &folders) {
    O result = directory;
    for (py::handle item : folders) {
        if (!item.is_none()) {
            O path = checked(PyNumber_TrueDivide(result.ptr(), item.ptr()));
            result = path.attr("resolve")();
        }
    }
    return result;
}

O neutral(const py::dict &globals, const O &operation) {
    if (option(globals, operation, "Operation", "SUM")) return py::int_(0);
    if (option(globals, operation, "Operation", "PRODUCT")) return py::int_(1);
    if (option(globals, operation, "Operation", "MAXIMUM")) return name(globals, "float")("-inf");
    if (option(globals, operation, "Operation", "MINIMUM")) return name(globals, "float")("inf");
    not_implemented();
}

O reduce(const py::dict &globals, const O &operation, const O &a, const O &b) {
    if (option(globals, operation, "Operation", "SUM")) return checked(PyNumber_Add(a.ptr(), b.ptr()));
    if (option(globals, operation, "Operation", "PRODUCT")) return checked(PyNumber_Multiply(a.ptr(), b.ptr()));
    if (option(globals, operation, "Operation", "MAXIMUM")) return name(globals, "max")(a, b);
    if (option(globals, operation, "Operation", "MINIMUM")) return name(globals, "min")(a, b);
    not_implemented();
}

struct Accumulator {
    O operation;
    O value;
    explicit Accumulator(const O &op) : operation(op), value(op.attr("neutral")) {}
    void iterate(const O &item) {
        // Resolve the bound method before reading the current state, as Python does.
        O callback = operation.attr("reduce");
        O previous = value;
        O next = callback(previous, item);
        value = next; // An exception leaves the previous result untouched.
    }
};

O accumulate(const py::dict &globals, const O &operation) {
    auto state = std::make_shared<Accumulator>(operation);
    py::cpp_function iterate([state](const O &item) { state->iterate(item); });
    py::cpp_function complete([state]() { return state->value; });
    return name(globals, "Collector")(
        py::arg("on_iterate") = iterate, py::arg("on_complete") = complete);
}

O logic(const py::dict &globals, const O &operation, const O &a, const O &b) {
    if (option(globals, operation, "LogicOperation", "AND"))
        return graphpy::truth(a) ? O(b.attr("value")) : a;
    if (option(globals, operation, "LogicOperation", "OR"))
        return graphpy::truth(a) ? a : O(b.attr("value"));
    if (option(globals, operation, "LogicOperation", "XOR")) {
        O value = b.attr("value");
        return checked(PyObject_RichCompare(a.ptr(), value.ptr(), Py_NE));
    }
    if (option(globals, operation, "LogicOperation", "NOT")) return py::bool_(!graphpy::truth(a));
    return py::none();
}

O conditional(const O &condition, const O &yes, const O &no) {
    return graphpy::truth(condition) ? O(yes.attr("value")) : O(no.attr("value"));
}

class RangeIterator {
    O start_;
    O count_;
    O cursor_ = py::none();
    bool closed_ = false;
    bool running_ = false;
public:
    RangeIterator(const O &start, const O &count) : start_(start), count_(count) {}
    O next() {
        ActiveIteration active(running_);
        if (closed_) exhausted();
        O index;
        try {
            if (cursor_.is_none()) {
                O range = graphpy::builtin("range")(count_);
                cursor_ = checked(PyObject_GetIter(range.ptr()));
            }
            PyObject *raw = PyIter_Next(cursor_.ptr());
            if (!raw) {
                closed_ = true;
                if (PyErr_Occurred()) throw py::error_already_set();
                exhausted();
            }
            index = py::reinterpret_steal<O>(raw);
        } catch (...) {
            closed_ = true;
            throw;
        }
        try {
            return checked(PyNumber_Add(start_.ptr(), index.ptr()));
        } catch (const py::error_already_set &error) {
            if (error.matches(PyExc_Exception)) return error.value();
            closed_ = true;
            throw;
        }
    }
    void close() {
        if (running_) throw py::value_error("generator already executing");
        closed_ = true;
        cursor_ = py::none();
    }
};

O range_node(const py::dict &globals, O start, const O &start_inclusive,
             O end, const O &end_inclusive) {
    const O one = py::int_(1);
    if (!graphpy::truth(start_inclusive)) start = checked(PyNumber_InPlaceAdd(start.ptr(), one.ptr()));
    if (graphpy::truth(end_inclusive)) end = checked(PyNumber_InPlaceAdd(end.ptr(), one.ptr()));
    O count = checked(PyNumber_Subtract(end.ptr(), start.ptr()));
    if (!comparison(count, py::int_(0), Py_GE)) {
        PyErr_SetNone(PyExc_AssertionError);
        throw py::error_already_set();
    }
    py::cpp_function supplier([start, count]() {
        return std::make_shared<RangeIterator>(start, count);
    });
    O generator = name(globals, "Generator")(supplier, count);
    generator.attr("source_paths") = py::tuple();  // Numbers only.
    return generator;
}
}

void cn_bind_execution_ops(py::module_ &module) {
    py::class_<RangeIterator, std::shared_ptr<RangeIterator>>(module, "_ExecutionRangeIterator")
        .def("__iter__", [](RangeIterator &self) -> RangeIterator & { return self; },
             py::return_value_policy::reference_internal)
        .def("__next__", &RangeIterator::next)
        .def("close", &RangeIterator::close);
    module.def("execution_directory_into", &directory_into);
    module.def("execution_accumulate_neutral", &neutral);
    module.def("execution_accumulate_reduce", &reduce);
    module.def("execution_accumulate", &accumulate);
    module.def("execution_logic", &logic);
    module.def("execution_conditional", &conditional);
    module.def("execution_range", &range_node);
    module.def("execution_number", [](const O &number) { return number; });
    module.def("execution_note", [](const O &, const O &) { return py::none(); });
}
