/* chaiNNer scalar/path application control, adapted from its GPL-3.0 source.
 * CPython owns arbitrary-precision arithmetic, numeric parsing, and objects;
 * its compiled math primitives and pathlib's host semantics remain dependencies.
 * Branches, expression ordering, rounding composition and exception decisions
 * are implemented here, without evaluating original Python node bodies.
 */
#include "graph_python.hpp"
#include "graph_exception.hpp"

namespace py = pybind11;
namespace {
using graphpy::O;

O checked(PyObject *result) {
    if (!result) throw py::error_already_set();
    return py::reinterpret_steal<O>(result);
}

O name(const py::dict &globals, const char *key) {
    py::str lookup(key);
    if (globals.contains(lookup)) return globals[lookup];
    return graphpy::builtin(key);
}

bool equal(const O &first, const O &second) {
    // Python's == expression invokes rich comparison even for identical objects.
    return graphpy::truth(checked(PyObject_RichCompare(first.ptr(), second.ptr(), Py_EQ)));
}

bool option(const py::dict &globals, const O &value, const char *type, const char *member) {
    O expected = name(globals, type).attr(member);
    return equal(value, expected);
}

O add(const O &a, const O &b) { return checked(PyNumber_Add(a.ptr(), b.ptr())); }
O subtract(const O &a, const O &b) { return checked(PyNumber_Subtract(a.ptr(), b.ptr())); }
O multiply(const O &a, const O &b) { return checked(PyNumber_Multiply(a.ptr(), b.ptr())); }
O divide(const O &a, const O &b) { return checked(PyNumber_TrueDivide(a.ptr(), b.ptr())); }

O format(const O &value) {
    // f-string fields use __format__(""), which need not equal __str__().
    return checked(PyObject_Format(value.ptr(), nullptr));
}

O concat(const O &a, const O &b) { return checked(PyUnicode_Concat(a.ptr(), b.ptr())); }

[[noreturn]] void raise_object(const O &error) {
    if (!PyExceptionInstance_Check(error.ptr()))
        graphpy::raise(PyExc_TypeError, py::str("exceptions must derive from BaseException"));
    PyErr_SetObject(reinterpret_cast<PyObject *>(Py_TYPE(error.ptr())), error.ptr());
    throw py::error_already_set();
}

[[noreturn]] void unknown(const py::dict &globals, const char *type, const char *prefix, const O &value) {
    O constructor = name(globals, type);
    O text = concat(py::str(prefix), format(value));
    raise_object(constructor(text));
}

[[noreturn]] void invalid_power(const py::dict &globals, const O &a, const O &b, const O &cause) {
    O constructor = name(globals, "ValueError");
    O first = format(a);
    O second = format(b);
    O message = concat(concat(concat(first, py::str("^")), second),
                       py::str(" is not defined for real numbers."));
    O error = constructor(message);
    if (!cause.is_none() && PyExceptionInstance_Check(error.ptr()))
        PyException_SetCause(error.ptr(), Py_NewRef(cause.ptr()));
    raise_object(error);
}

O directory_up(const py::dict &globals, const O &directory, const O &amount) {
    O result = directory;
    O indices = name(globals, "range")(amount);
    for (py::handle index : py::reinterpret_borrow<py::iterable>(indices)) {
        (void)index;
        result = result.attr("parent");
    }
    return result;
}

O math_node(const py::dict &globals, const O &op, const O &a, const O &b) {
    if (option(globals, op, "MathOperation", "ADD")) return add(a, b);
    if (option(globals, op, "MathOperation", "SUBTRACT")) return subtract(a, b);
    if (option(globals, op, "MathOperation", "MULTIPLY")) return multiply(a, b);
    if (option(globals, op, "MathOperation", "DIVIDE")) return divide(a, b);
    if (option(globals, op, "MathOperation", "POWER")) {
        O result;
        try {
            result = name(globals, "pow")(a, b);
        } catch (const py::error_already_set &error) {
            O exception = name(globals, "Exception");
            if (!error.matches(exception.ptr())) throw;
            GraphHandledException context(error);
            invalid_power(globals, a, b, error.value());
        }
        O instance = name(globals, "isinstance");
        O integer = name(globals, "int");
        O floating = name(globals, "float");
        if (graphpy::truth(instance(result, py::make_tuple(integer, floating)))) return result;
        invalid_power(globals, a, b, py::none());
    }
    if (option(globals, op, "MathOperation", "LOG")) return name(globals, "math").attr("log")(b, a);
    if (option(globals, op, "MathOperation", "MAXIMUM")) return name(globals, "max")(a, b);
    if (option(globals, op, "MathOperation", "MINIMUM")) return name(globals, "min")(a, b);
    if (option(globals, op, "MathOperation", "MODULO")) {
        bool special = graphpy::contains(name(globals, "_special_mod_numbers"), a);
        if (!special) special = graphpy::contains(name(globals, "_special_mod_numbers"), b);
        if (special) {
            O floor = name(globals, "math").attr("floor");
            O quotient = divide(a, b);
            O rounded = floor(quotient);
            O product = multiply(b, rounded);
            return subtract(a, product);
        }
        return checked(PyNumber_Remainder(a.ptr(), b.ptr()));
    }
    if (option(globals, op, "MathOperation", "PERCENT")) {
        O product = multiply(a, b);
        return divide(product, py::int_(100));
    }
    unknown(globals, "RuntimeError", "Unknown operator ", op);
}

O half_up(const py::dict &globals, const O &value) {
    O floor = name(globals, "math").attr("floor");
    O shifted = add(value, py::float_(0.5));
    return floor(shifted);
}

O round_node(const py::dict &globals, const O &a, const O &operation,
             const O &scale, const O &multiple, const O &power) {
    O callback;
    bool half = false;
    if (option(globals, operation, "RoundOperation", "FLOOR"))
        callback = name(globals, "math").attr("floor");
    else if (option(globals, operation, "RoundOperation", "CEILING"))
        callback = name(globals, "math").attr("ceil");
    else if (option(globals, operation, "RoundOperation", "ROUND")) half = true;
    else unknown(globals, "RuntimeError", "Unknown operation ", operation);

    auto apply = [&](const O &value) { return half ? half_up(globals, value) : callback(value); };
    if (option(globals, scale, "RoundScale", "UNIT")) return apply(a);
    if (option(globals, scale, "RoundScale", "MULTIPLE")) {
        O quotient = divide(a, multiple);
        O rounded = apply(quotient);
        return multiply(rounded, multiple);
    }
    if (option(globals, scale, "RoundScale", "POWER")) {
        O logarithm = name(globals, "math").attr("log")(a, power);
        O exponent = apply(logarithm);
        return checked(PyNumber_Power(power.ptr(), exponent.ptr(), Py_None));
    }
    unknown(globals, "RuntimeError", "Unknown scale ", scale);
}

O parse_number(const py::dict &globals, const O &text, const O &base) {
    // int's compiled constructor retains Unicode digits, arbitrary precision,
    // base/__index__ rules, configured digit limits, and exact diagnostic text.
    return name(globals, "int")(text, base);
}
}

void cn_bind_utility_scalar(py::module_ &module) {
    module.def("utility_directory_up", &directory_up);
    module.def("utility_math", &math_node);
    module.def("utility_round", &round_node);
    module.def("utility_parse_number", &parse_number);
}
