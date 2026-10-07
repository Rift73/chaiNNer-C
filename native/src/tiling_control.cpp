/* Native tile geometry and restart state machines. Public callbacks, image
 * views, progress objects and already-native tile blending remain adapters.
 * Source variants preserve both installed and checkout host contracts. */
#include "graph_python.hpp"
#include "graph_unpack.hpp"
#include "graph_numeric.hpp"
#include "graph_exception.hpp"
#include <string>

namespace {
using namespace graphpy;
enum class Op { add,sub,mul,div,floordiv,pow,mod };
O binary(const O &a,const O &b,Op op) { return cn_graph_binary(a,b,static_cast<int>(op)); }
O inplace(const O &a,const O &b,Op op) { return cn_graph_inplace(a,b,static_cast<int>(op)); }
const O &local(const O &value,const char *name) {
    if(!value)raise(PyExc_UnboundLocalError,py::str(std::string("cannot access local variable '")+name+"' where it is not associated with a value"));
    return value;
}
O rich_compare(const O &a,const O &b,int op) {
    PyObject *result=PyObject_RichCompare(a.ptr(),b.ptr(),op);
    if(!result){throw py::error_already_set();}return py::reinterpret_steal<O>(result);
}
O concat_text(std::initializer_list<O> values) {
    O result=py::str("");
    for(const auto &v:values) {
        PyObject *next=PyUnicode_Concat(result.ptr(),v.ptr());
        if(!next){throw py::error_already_set();}result=py::reinterpret_steal<O>(next);
    }
    return result;
}
[[noreturn]] void raise_class(const O &type) {
    PyErr_SetNone(type.ptr());throw py::error_already_set();
}
#include "tiling_control.hpp"
}

void cn_bind_tiling(pybind11::module_ &module) { bind_tiling(module); }
