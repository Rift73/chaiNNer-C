/* Native graph array helpers. Dtype/allocation metadata is resolved by NumPy;
 * numeric elements are computed by C/C++ kernels. */
#pragma once
#include <pybind11/pybind11.h>
// add=0, sub=1, mul=2, div=3, floordiv=4, pow=5, mod=6.
pybind11::object cn_graph_binary(const pybind11::object&,const pybind11::object&,int operation);
pybind11::object cn_graph_inplace(const pybind11::object&,const pybind11::object&,int operation);
pybind11::object cn_graph_array_cast(const pybind11::object&,const pybind11::object& dtype);
void cn_graph_report_fp(int events,const char *operation);
// A binary ufunc's operands as NumPy 2.5 prepares them for loop dtype `dtype`:
// a weak Python int/float/complex is stored by that dtype's setitem (NEP 50),
// and check_for_trivial_loop's up-front casts report "cast" FP events. The
// events of the casts left to the iterator are returned in `deferred`; they
// belong to the ufunc loop when it runs (a non-empty result).
struct cn_graph_operands { pybind11::object left,right; int deferred; };
cn_graph_operands cn_graph_ufunc_operands(const pybind11::object&,const pybind11::object&,const pybind11::object& dtype);
