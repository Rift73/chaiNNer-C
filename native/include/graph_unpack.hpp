/* CPython's bounded iterable unpacking, `a, b = value` or `a, b, c = value`. CPython
 * itself unpacks, in nodes.impl.unpacking, so the iterator consumption and the errors
 * are the running interpreter's own (3.14 added ", got N" to some messages). Copies
 * references only; source objects preserve their identity. */
#pragma once
#include "graph_python.hpp"
#include <stdexcept>

namespace graphpy {
inline O unpack(const O &value,py::ssize_t count) {
    if(count!=2 && count!=3)throw std::logic_error("graphpy::unpack takes 2 or 3 targets");
    return py::module_::import("nodes.impl.unpacking").attr(count==2?"unpack2":"unpack3")(value);
}
}
