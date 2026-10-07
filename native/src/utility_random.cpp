/* Native application RNG/seed control, pinned to CPython 3.14.8 random.py.
 * Range validation, byte-seed preparation and rejection sampling are adapted
 * from that PSF-licensed source (frozen hash in reference_utility_random).
 * Retained public compiled primitives: CPython arbitrary integers/Unicode,
 * SHA-256/SHA-512, and _random.Random's MT seed/getrandbits. No Python Random
 * seed/randrange/randint implementation or process-global RNG is invoked.
 */
#include <pybind11/pybind11.h>
#include <cstring>
#include <limits>
#include "graph_python.hpp"

namespace {
namespace py = pybind11;
using O = py::object;
using graphpy::compare;

O owned(PyObject *value) {
    if (!value) throw py::error_already_set();
    return py::reinterpret_steal<O>(value);
}
O add(const O &a, const O &b) { return owned(PyNumber_Add(a.ptr(), b.ptr())); }
O sub(const O &a, const O &b) { return owned(PyNumber_Subtract(a.ptr(), b.ptr())); }
O as_int(const O &value) {
    return owned(PyObject_CallOneArg(reinterpret_cast<PyObject *>(&PyLong_Type), value.ptr()));
}
bool instance(const O &value, const O &type) {
    const int result = PyObject_IsInstance(value.ptr(), type.ptr());
    if (result < 0) throw py::error_already_set();
    return result != 0;
}
O type_object(PyTypeObject &type) {
    return py::reinterpret_borrow<O>(reinterpret_cast<PyObject *>(&type));
}

O source_bytes(O source, const O &seed_type) {
    if (instance(source, type_object(PyUnicode_Type))) {
        // Preserve an explicit str subclass's encode protocol as in the source.
        if (!PyUnicode_CheckExact(source.ptr()))
            return source.attr("encode")(py::arg("errors") = "backslashreplace");
        return owned(PyUnicode_AsEncodedString(source.ptr(), "utf-8", "backslashreplace"));
    }
    if (instance(source, seed_type)) source = source.attr("value");
    const O integer = as_int(source);
    if (instance(source, type_object(PyLong_Type)) || compare(source, integer, Py_EQ)) {
        const O bits = integer.attr("bit_length")();
        const O bytes = add(owned(PyNumber_FloorDivide(bits.ptr(), py::int_(8).ptr())), py::int_(1));
        // Python's public bigint primitive packs unbounded signed two's-complement
        // integers without narrowing to an implementation-sized C integer.
        return integer.attr("to_bytes")(bytes, py::arg("byteorder") = "big", py::arg("signed") = true);
    }
    // struct.pack('d', source) uses the public double conversion and native byte
    // order. MSVC's pinned ABI has IEEE binary64; preserve its bits verbatim.
    static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559);
    const double value = PyFloat_AsDouble(source.ptr());
    if (PyErr_Occurred()) {
        // CPython struct's native double packer replaces every failed conversion,
        // including user __float__ exceptions, with this struct.error.
        PyErr_Clear();
        graphpy::raise(py::module_::import("struct").attr("error").ptr(), py::str("required argument is not a float"));
    }
    char packed[sizeof(double)];
    std::memcpy(packed, &value, sizeof(value));
    return py::bytes(packed, sizeof(packed));
}

// random.seed's SHA-512 on CPython 3.12+: `from _sha2 import sha512`, or hashlib's
// when that import fails (3.11 imported _sha512, which 3.12 removed).
O seed_sha512() {
    try {
        return py::module_::import("_sha2").attr("sha512");
    } catch (const py::error_already_set &error) {
        if (!error.matches(PyExc_ImportError) && !error.matches(PyExc_AttributeError)) throw;
        return py::module_::import("hashlib").attr("sha512");
    }
}

O prepare_seed(O value) {
    const O byte_types = py::make_tuple(type_object(PyUnicode_Type), type_object(PyBytes_Type), type_object(PyByteArray_Type));
    if (instance(value, byte_types)) {
        if (instance(value, type_object(PyUnicode_Type))) value = value.attr("encode")();
        const O hash = seed_sha512()(value).attr("digest")();
        const O combined = add(value, hash);
        // Public native bigint decoder, big-endian and unsigned by default.
        return py::reinterpret_borrow<O>(reinterpret_cast<PyObject *>(&PyLong_Type)).attr("from_bytes")(combined);
    }
    const O allowed_types = py::make_tuple(py::type::of(py::none()), type_object(PyLong_Type),
        type_object(PyFloat_Type), type_object(PyUnicode_Type), type_object(PyBytes_Type), type_object(PyByteArray_Type));
    if (!instance(value, allowed_types))
        throw py::type_error("The only supported seed types are:\nNone, int, float, str, bytes, and bytearray.");
    return value;
}

O core(const O &seed) {
    // This is _random's C constructor, not random.Random's Python wrappers.
    return py::module_::import("_random").attr("Random")(prepare_seed(seed));
}

O below(const O &generator, const O &width) {
    const O getrandbits = generator.attr("getrandbits");
    const O bits = width.attr("bit_length")();
    O value = getrandbits(bits);
    while (compare(value, width, Py_GE)) value = getrandbits(bits);
    return value;
}

O inclusive(const O &generator, const O &minimum, const O &maximum) {
    // randint: a and b through operator.index (a float raises TypeError and nothing
    // warns), then the empty-range check on those ints, then a + _randbelow(b - a + 1).
    const O first = owned(PyNumber_Index(minimum.ptr()));
    const O last = owned(PyNumber_Index(maximum.ptr()));
    if (compare(last, first, Py_LT))
        graphpy::raise(PyExc_ValueError, py::str("empty range in randint({}, {})").attr("format")(first, last));
    return add(first, below(generator, add(sub(last, first), py::int_(1))));
}

O random_number(const O &minimum, const O &maximum, const O &seed) {
    const O generator = core(seed.attr("value"));
    return inclusive(generator, minimum, maximum);
}

O derive(const O &seed, const py::tuple &sources, const O &seed_type) {
    bool present = false;
    for (const auto &source : sources) if (!source.is_none()) { present = true; break; }
    if (!present) return seed;
    const O hash = py::module_::import("hashlib").attr("sha256")();
    hash.attr("update")(source_bytes(seed, seed_type));
    for (const auto &source : sources)
        if (!source.is_none()) hash.attr("update")(source_bytes(py::reinterpret_borrow<O>(source), seed_type));
    const O generator = core(hash.attr("digest")());
    return seed_type(inclusive(generator, py::int_(0), py::int_(4294967295ULL)));
}
} // namespace

void cn_bind_utility_random(pybind11::module_ &module) {
    module.def("utility_seed_to_bytes", &source_bytes);
    module.def("utility_derive_seed", &derive);
    module.def("utility_random_number", &random_number);
    // Same primitives exposed for exact core-state and rejection-trajectory tests.
    // Neither export retains state; callers explicitly own the supplied core.
    module.def("utility_random_core", &core);
    module.def("utility_random_inclusive", &inclusive);
}
