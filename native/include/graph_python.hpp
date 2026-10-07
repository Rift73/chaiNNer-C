/* CPython object/metadata primitives shared by fixed graph algorithms.
 * No runtime eval, source execution, graph algorithm or NumPy arithmetic. */
#pragma once
#include <pybind11/pybind11.h>
#include <initializer_list>
namespace graphpy {
namespace py = pybind11;
using O = py::object;
inline O builtin(const char *name) { return py::module_::import("builtins").attr(name); }
inline O attr(const O &object,const char *name) { return object.attr(name); }
inline O item(const O &object,const O &key) { return object[key]; }
inline void set_attr(const O &object,const char *name,const O &value) { object.attr(name)=value; }
inline void set_item(const O &object,const O &key,const O &value) { object[key]=value; }
inline bool truth(const O &object) {
    int result=PyObject_IsTrue(object.ptr());
    if(result<0)throw py::error_already_set();
    return result!=0;
}
inline bool compare(const O &a,const O &b,int comparison) {
    int result=PyObject_RichCompareBool(a.ptr(),b.ptr(),comparison);
    if(result<0)throw py::error_already_set();
    return result!=0;
}
inline bool contains(const O &collection,const O &value) {
    int result=PySequence_Contains(collection.ptr(),value.ptr());
    if(result<0)throw py::error_already_set();
    return result!=0;
}
inline O make_list(std::initializer_list<O> values) {
    py::list result; for(const auto &v:values)result.append(v);return result;
}
inline O make_tuple(std::initializer_list<O> values) {
    py::tuple result(values.size());size_t i=0;
    for(const auto &v:values){result[i++]=v;}return result;
}
[[noreturn]] inline void raise(PyObject *type,const O &message) {
    PyErr_SetObject(type,message.ptr());throw py::error_already_set();
}
inline void require(bool condition,const O &message) {
    if(!condition)raise(PyExc_AssertionError,message);
}
inline O checked(const O &type,const O &value) {
    int ok=PyObject_IsInstance(value.ptr(),type.ptr());
    if(ok<0)throw py::error_already_set();
    if(!ok)raise(PyExc_AssertionError,py::str("Value is {}, must be type {}").attr("format")(py::type::of(value),type));
    return value;
}
inline O negative(const O &value) {
    PyObject *result=PyNumber_Negative(value.ptr());
    if(!result){throw py::error_already_set();}return py::reinterpret_steal<O>(result);
}
}
