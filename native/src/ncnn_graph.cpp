/* NCNN application graph algorithms, translated from chaiNNer's installed
 * GPL-3.0 source. The NCNN inference engine itself is not reimplemented here.
 * CPython owns objects, exceptions and scalar semantics; this translation unit
 * owns graph control flow and numeric weight operations. NumPy is retained for
 * dtype/shape/allocation metadata and its native scientific scalar formatter. */
#include "cn_crt_math.h"
#include "graph_python.hpp"
#include "graph_unpack.hpp"
#include "graph_numeric.hpp"
#include "ncnn_tensor_abi.hpp"
#include "ncnn_complex_math.hpp"
#include <pybind11/numpy.h>
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <initializer_list>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

// The FMA3 intrinsics run only when NumPy itself dispatched its FMA3 loops.
#if defined(__GNUC__) || defined(__clang__)
#define CN_FMA_TARGET __attribute__((target("fma")))
#else
#define CN_FMA_TARGET
#endif

namespace {
using namespace graphpy;
enum class Op { add,sub,mul,div,floordiv,pow,mod };
O np() { return py::module_::import("numpy"); }
O result(PyObject *value) { if(!value)throw py::error_already_set();return py::reinterpret_steal<O>(value); }
void status(int code) {
    if(code==0)return;
    if(code==3)throw std::bad_alloc();
    if(code==2)raise(PyExc_OverflowError,py::str("NCNN tensor buffer size overflow"));
    raise(PyExc_ValueError,py::str("Invalid NCNN tensor buffer"));
}
void report_fp(int events,const char *operation) {
    if(!events)return;
    O policy=np().attr("geterr")();
    const int bits[]={1,2,4,8};
    const char *keys[]={"divide","over","under","invalid"};
    const char *names[]={"divide by zero","overflow","underflow","invalid value"};
    for(int i=0;i<4;++i)if(events & bits[i]) {
        std::string mode=py::str(policy[py::str(keys[i])]);
        std::string message=std::string(names[i])+" encountered in "+operation;
        if(mode=="warn") { if(PyErr_WarnEx(PyExc_RuntimeWarning,message.c_str(),2)<0)throw py::error_already_set(); }
        else if(mode=="raise")raise(PyExc_FloatingPointError,py::str(message));
        else if(mode=="print")PySys_WriteStderr("Warning: %s\n",message.c_str());
        else if(mode=="call" || mode=="log") {
            O callback=np().attr("geterrcall")();
            if(callback.is_none())raise(PyExc_NameError,py::str(std::string("python callback specified for ")+names[i]+" (in "+operation+") but no function found."));
            if(mode=="call")callback(py::str(names[i]),py::int_(events));
            else callback.attr("write")(py::str("Warning: "+message+"\n"));
        }
    }
}
int fp_events() {
    int flags=std::fetestexcept(FE_DIVBYZERO|FE_OVERFLOW|FE_UNDERFLOW|FE_INVALID);
    return ((flags&FE_DIVBYZERO)?1:0)|((flags&FE_OVERFLOW)?2:0)|((flags&FE_UNDERFLOW)?4:0)|((flags&FE_INVALID)?8:0);
}
int type_code(const py::dtype &dtype) {
    char kind=dtype.kind();auto size=dtype.itemsize();
    if(kind=='f')return size==2?0:size==4?1:2;
    if(kind=='c')return size==8?3:4;
    if(kind=='b')return 5;
    if(kind=='u')return size==1?6:size==2?7:size==4?8:9;
    if(kind=='i')return size==1?10:size==2?11:size==4?12:13;
    raise(PyExc_TypeError,py::str("NCNN weight dtype is not numeric"));
}
struct Tensor {
    py::array array;
    std::vector<size_t> shape;
    std::vector<ptrdiff_t> strides;
    cn_ncnn_tensor_view view;
    explicit Tensor(const O &object):array(py::reinterpret_borrow<py::array>(object)),view{} {
        for(py::ssize_t i=0;i<array.ndim();++i){shape.push_back(static_cast<size_t>(array.shape(i)));strides.push_back(array.strides(i));}
        view={const_cast<void*>(array.data()),static_cast<size_t>(array.size()),shape.size(),shape.data(),strides.data(),type_code(array.dtype())};
    }
};
O asarray(const O &value) { return np().attr("asarray")(value); }
O native_byte_order(const O &value) {
    py::array source=py::reinterpret_borrow<py::array>(value);
    if(truth(source.dtype().attr("isnative")))return value;
    O dtype=source.dtype().attr("newbyteorder")(py::str("="));
    O output=np().attr("empty_like")(source,py::arg("dtype")=dtype,py::arg("order")="K",py::arg("subok")=false);
    Tensor a(value),out(output);
    size_t width=static_cast<size_t>(source.itemsize()),parts=source.dtype().kind()=='c'?2:1,part_width=width/parts;
    for(size_t i=0;i<a.view.count;++i){
        ptrdiff_t ai=0,oi=0;size_t index=i;
        for(size_t d=a.shape.size();d>0;--d){size_t coordinate=index%a.shape[d-1];index/=a.shape[d-1];ai+=static_cast<ptrdiff_t>(coordinate)*a.strides[d-1];oi+=static_cast<ptrdiff_t>(coordinate)*out.strides[d-1];}
        const char *p=static_cast<const char*>(source.data())+ai;char *q=static_cast<char*>(out.array.mutable_data())+oi;
        for(size_t part=0;part<parts;++part)for(size_t b=0;b<part_width;++b)q[part*part_width+b]=p[part*part_width+part_width-1-b];
    }
    return output;
}
O array_allocate(const O &a,const O &b,const O &dtype) {
    // Only asks NumPy's iterator for K-order allocation metadata. No iteration.
    O it=np().attr("nditer")(make_tuple({a,b,py::none()}),
        py::arg("flags")=make_list({py::str("zerosize_ok")}),
        py::arg("op_flags")=make_list({make_list({py::str("readonly")}),make_list({py::str("readonly")}),make_list({py::str("writeonly"),py::str("allocate")})}),
        py::arg("op_dtypes")=make_list({py::none(),py::none(),dtype}),py::arg("casting")="unsafe");
    O output=item(it.attr("operands"),py::int_(2));it.attr("close")();return output;
}
O array_cast(const O &value,const O &dtype,bool warnings=true,int *raised=nullptr) {
    O original=asarray(value),source=native_byte_order(original),target=np().attr("dtype")(dtype);
    bool swap_output=!truth(target.attr("isnative"));
    O native_target=swap_output?O(target.attr("newbyteorder")(py::str("="))):target;
    O output=np().attr("empty_like")(source,py::arg("dtype")=native_target,py::arg("order")="K",py::arg("subok")=false);
    Tensor a(source),out(output);int events=0;
    if(a.array.dtype().kind()=='c' && out.array.dtype().kind()!='c' && out.array.dtype().kind()!='b') {
        O warning=np().attr("exceptions").attr("ComplexWarning");
        if(PyErr_WarnEx(warning.ptr(),"Casting complex values to real discards the imaginary part",2)<0)throw py::error_already_set();
    }
    status(cn_tensor_cast_typed(&a.view,&out.view,&events));
    if(raised)*raised|=events;
    if(warnings)report_fp(events,"cast");
    if(swap_output){
        // Reuse the endian copier with a metadata-only opposite-endian view.
        O opposite=output.attr("view")(target);O swapped=native_byte_order(opposite);
        return swapped.attr("view")(target);
    }
    return output;
}
// NumPy 2.5 (NEP 50) treats a Python int, float or complex operand as weak: it
// takes the loop's dtype through that dtype's setitem (arraytypes.c.src).
bool weak_literal(const O &value) {
    PyObject *object=value.ptr();
    return PyLong_CheckExact(object) || PyFloat_CheckExact(object) || PyComplex_CheckExact(object);
}
template<class T> T converted(T value) {
    if(value==static_cast<T>(-1) && PyErr_Occurred())throw py::error_already_set();
    return value;
}
uint64_t weak_integer(const O &value,const py::dtype &dtype) {
    // @TYPE@_safe_pyint_setitem: C long parses a type up to its width (C long
    // long beyond), unsigned int/long parse unsigned and retry signed, and a
    // wrapped or narrowed value is out of bounds. CPython's own errors propagate.
    O number=result(PyNumber_Long(value.ptr()));
    PyObject *object=number.ptr();
    const auto bits=static_cast<size_t>(dtype.itemsize())*8;
    uint64_t stored;bool wrapped=false,narrowed=false;
    if(dtype.kind()=='i' && bits<=sizeof(long)*8) {
        const long parsed=converted(PyLong_AsLong(object));
        stored=static_cast<uint64_t>(static_cast<int64_t>(parsed));
        if(bits<sizeof(long)*8)narrowed=parsed<-(1L<<(bits-1)) || parsed>(1L<<(bits-1))-1;
    } else if(dtype.kind()=='i') {
        stored=static_cast<uint64_t>(converted(PyLong_AsLongLong(object)));
    } else if(bits<32) {
        const long parsed=converted(PyLong_AsLong(object));
        stored=static_cast<uint64_t>(static_cast<int64_t>(parsed));
        narrowed=parsed<0 || parsed>(1L<<bits)-1;
    } else if(bits<=sizeof(unsigned long)*8) {
        unsigned long parsed=PyLong_AsUnsignedLong(object);
        if(PyErr_Occurred()) {
            PyErr_Clear();wrapped=true;
            parsed=static_cast<unsigned long>(converted(PyLong_AsLong(object)));
        }
        stored=parsed;
        if(bits<sizeof(unsigned long)*8)narrowed=parsed>(1UL<<bits)-1;
    } else {
        unsigned long long parsed=PyLong_AsUnsignedLongLong(object);
        if(PyErr_Occurred()) {
            PyErr_Clear();wrapped=true;
            parsed=static_cast<unsigned long long>(converted(PyLong_AsLongLong(object)));
        }
        stored=parsed;
    }
    if(wrapped || narrowed)raise(PyExc_OverflowError,py::str("Python integer {} out of bounds for {}").attr("format")(builtin("repr")(value),builtin("str")(dtype)));
    return stored;
}
O weak_operand(const O &value,const py::dtype &dtype) {
    O output=np().attr("empty")(py::tuple(),py::arg("dtype")=dtype);
    char *data=static_cast<char*>(py::reinterpret_borrow<py::array>(output).mutable_data());
    const char kind=dtype.kind();const auto size=dtype.itemsize();
    // A real or complex part that a finite double cannot keep is reported as a
    // "cast" overflow (MyPyFloat_AsFloat, MyPyFloat_AsHalf, CFLOAT_setitem).
    bool overflow=false;
    if(kind=='i' || kind=='u') {
        const uint64_t stored=weak_integer(value,dtype);
        std::memcpy(data,&stored,static_cast<size_t>(size));
    } else if(kind=='f') {
        O number=result(PyNumber_Float(value.ptr()));
        double parsed=PyFloat_AS_DOUBLE(number.ptr());
        if(size==2) {
            uint16_t half=0;int ignored=0;
            cn_ncnn_tensor_view source={&parsed,1,0,nullptr,nullptr,2},target={&half,1,0,nullptr,nullptr,0};
            status(cn_tensor_cast_typed(&source,&target,&ignored));
            overflow=(half&0x7fffU)==0x7c00U && !std::isinf(parsed);
            std::memcpy(data,&half,sizeof(half));
        } else if(size==4) {
            const float narrow=static_cast<float>(parsed);
            overflow=std::isinf(narrow) && !std::isinf(parsed);
            std::memcpy(data,&narrow,sizeof(narrow));
        } else if(size==8)std::memcpy(data,&parsed,sizeof(parsed));
        else raise(PyExc_TypeError,py::str("NCNN weight dtype is not numeric"));
    } else if(kind=='c') {
        const Py_complex parsed=PyComplex_AsCComplex(value.ptr());
        if(parsed.real==-1.0 && PyErr_Occurred())throw py::error_already_set();
        if(size==8) {
            const float parts[2]={static_cast<float>(parsed.real),static_cast<float>(parsed.imag)};
            overflow=(std::isinf(parts[0]) && !std::isinf(parsed.real)) || (std::isinf(parts[1]) && !std::isinf(parsed.imag));
            std::memcpy(data,parts,sizeof(parts));
        } else if(size==16) {
            const double parts[2]={parsed.real,parsed.imag};
            std::memcpy(data,parts,sizeof(parts));
        } else raise(PyExc_TypeError,py::str("NCNN weight dtype is not numeric"));
    } else raise(PyExc_TypeError,py::str("NCNN weight dtype is not numeric"));
    if(overflow)report_fp(2,"cast");
    return output;
}
struct UfuncOperands {
    O left,right;int deferred;
    // What NumPy's loop sees: whether check_for_trivial_loop finished its scan,
    // and each operand as handed on (an iterator-cast one before its cast).
    bool trivial;O iterated[2];bool buffered[2];
};
UfuncOperands ufunc_operands(const O &left,const O &right,const O &dtype_object) {
    // NumPy 2.5 ufunc_object.c: weak literals are resolved first. Then
    // check_for_trivial_loop casts, in operand order, each operand that needs a
    // cast (or is unaligned) up front if it is 0-d or 1-d within the buffer size,
    // reporting "cast" events. The first other such operand ends that scan; the
    // iterator casts it and the rest, and their events join the ufunc loop's.
    const py::dtype dtype=py::reinterpret_borrow<py::dtype>(np().attr("dtype")(dtype_object));
    O operands[2]={weak_literal(left)?left:asarray(left),weak_literal(right)?right:asarray(right)};
    for(O &operand:operands)if(weak_literal(operand))operand=weak_operand(operand,dtype);
    const auto buffer=np().attr("getbufsize")().cast<py::ssize_t>();
    int deferred=0;bool scanning=true;
    O iterated[2];bool buffered[2]={false,false};
    for(int i=0;i<2;++i) {
        O &operand=operands[i];
        const py::array array=py::reinterpret_borrow<py::array>(operand);
        const bool cast=!compare(array.dtype(),dtype,Py_EQ);
        if(cast || !truth(array.attr("flags").attr("aligned"))) {
            if(scanning && (array.ndim()==0 || (array.ndim()==1 && array.shape(0)<=buffer)))operand=array_cast(operand,dtype);
            else {
                scanning=false;
                if(cast) {
                    iterated[i]=operand;buffered[i]=true;
                    operand=array_cast(operand,dtype,false,&deferred);
                }
            }
        }
        if(!buffered[i])iterated[i]=operand;
    }
    return {operands[0],operands[1],deferred,scanning,{iterated[0],iterated[1]},{buffered[0],buffered[1]}};
}
// The inner-loop stride NumPy 2.5 gives each input of a binary ufunc whose
// output it allocates. try_trivial_single_output_loop takes 0-d operands with
// stride 0, 1-d ones with their own stride and N-d ones contiguous in a single
// order. Otherwise the iterator (which never negates strides when allocating)
// orders the axes with npyiter_find_best_axis_ordering, and the innermost axis
// longer than one is the loop's; a buffered operand is read from its buffer.
std::vector<ptrdiff_t> inner_strides(const UfuncOperands &operands,const O &shape_object) {
    std::vector<py::array> inputs;
    for(const O &operand:operands.iterated)inputs.push_back(py::reinterpret_borrow<py::array>(operand));
    bool trivial=operands.trivial;py::ssize_t order=0;const py::array *first=nullptr;
    for(const py::array &input:inputs) {
        if(!trivial || input.ndim()==0)continue;
        if(!first)first=&input;
        else if(input.ndim()!=first->ndim() || !std::equal(input.shape(),input.shape()+input.ndim(),first->shape()))trivial=false;
        if(input.ndim()>1) {
            const py::ssize_t layout=(truth(input.attr("flags").attr("c_contiguous"))?1:0)|(truth(input.attr("flags").attr("f_contiguous"))?2:0);
            if(layout==0 || (order!=0 && order!=layout))trivial=false;
            else order=layout;
        }
    }
    std::vector<ptrdiff_t> strides;
    if(trivial) {
        for(const py::array &input:inputs)strides.push_back(input.ndim()==0?0:input.ndim()==1?input.strides(0):input.itemsize());
        return strides;
    }
    // Axis data in NumPy's initial (reversed C) order; a length-1 axis has stride 0.
    const py::tuple shape(shape_object);
    const size_t ndim=shape.size();
    std::vector<py::ssize_t> lengths(ndim);std::vector<std::vector<ptrdiff_t>> axes(ndim);
    for(size_t axis=0;axis<ndim;++axis) {
        const size_t source=ndim-1-axis;lengths[axis]=shape[source].cast<py::ssize_t>();
        for(const py::array &input:inputs) {
            const py::ssize_t offset=static_cast<py::ssize_t>(ndim)-input.ndim(),dimension=static_cast<py::ssize_t>(source)-offset;
            axes[axis].push_back(dimension<0 || input.shape(dimension)==1 ? 0 : input.strides(dimension));
        }
    }
    std::vector<size_t> permutation(ndim);
    for(size_t axis=0;axis<ndim;++axis)permutation[axis]=axis;
    for(size_t i0=1;i0<ndim;++i0) {
        size_t position=i0;const size_t j0=permutation[i0];
        for(size_t i1=i0;i1-->0;) {
            bool ambiguous=true,swap=false;
            for(size_t op=0;op<inputs.size();++op) {
                const ptrdiff_t s0=axes[j0][op],s1=axes[permutation[i1]][op];
                if(s0!=0 && s1!=0) {
                    if(std::abs(s1)<=std::abs(s0))swap=false;
                    else if(ambiguous)swap=true;
                    ambiguous=false;
                }
            }
            if(!ambiguous) {
                if(swap)position=i1;
                else break;
            }
        }
        for(size_t i1=i0;i1>position;--i1)permutation[i1]=permutation[i1-1];
        permutation[position]=j0;
    }
    strides.assign(inputs.size(),0);
    for(size_t axis:permutation)if(lengths[axis]>1) {
        strides=axes[axis];break;
    }
    const O cast[2]={operands.left,operands.right};
    for(size_t op=0;op<inputs.size();++op)
        if(operands.buffered[op] && strides[op]!=0)strides[op]=py::reinterpret_borrow<py::array>(cast[op]).itemsize();
    return strides;
}
O array_empty(const O &shape,const O &dtype=py::dtype::of<double>()) { return np().attr("empty")(shape,py::arg("dtype")=dtype); }
O array_zeros(const O &shape,const O &dtype=py::dtype::of<double>()) {
    O output=array_empty(shape,dtype);auto a=py::reinterpret_borrow<py::array>(output);
    if(a.nbytes()){std::memset(a.mutable_data(),0,static_cast<size_t>(a.nbytes()));}return output;
}
O array_broadcast(const O &value,const O &shape_object) {
    py::array source=py::reinterpret_borrow<py::array>(asarray(value));
    py::tuple shape=py::tuple(shape_object);
    std::vector<py::ssize_t> dims,strides;
    auto invalid=[&](){
        // Error-only use of the original shape primitive preserves its exact
        // diagnostic. It cannot process elements on these rejected shapes.
        np().attr("broadcast_to")(value,shape_object);
        throw std::logic_error("Unexpected accepted NCNN broadcast shape");
    };
    if(static_cast<py::ssize_t>(shape.size())<source.ndim())invalid();
    py::ssize_t offset=static_cast<py::ssize_t>(shape.size())-source.ndim();
    for(py::ssize_t i=0;i<static_cast<py::ssize_t>(shape.size());++i) {
        auto d=shape[i].cast<py::ssize_t>();
        if(d<0)invalid();
        dims.push_back(d);
        if(i<offset)strides.push_back(0);
        else { auto k=i-offset;
            if(source.shape(k)!=1 && source.shape(k)!=d)invalid();
            strides.push_back(source.shape(k)==1?0:source.strides(k));
        }
    }
    py::array output(source.dtype(),dims,strides,source.data(),source);
    output.attr("setflags")(py::arg("write")=false);return output;
}
O array_transpose(const O &array,const O &axes) { return array.attr("transpose")(axes); }
O array_reshape(const O &array,const O &shape) { return array.attr("reshape")(shape); }
ptrdiff_t byte_offset(const Tensor &tensor,size_t index) {
    ptrdiff_t offset=0;
    for(size_t k=tensor.shape.size();k>0;--k){size_t d=tensor.shape[k-1];offset+=static_cast<ptrdiff_t>(index%d)*tensor.strides[k-1];index/=d;}
    return offset;
}
template<class T> T load(const Tensor &a,size_t index) {
    T value;std::memcpy(&value,static_cast<const char*>(a.array.data())+byte_offset(a,index),sizeof(T));return value;
}
template<class T> void store(Tensor &a,size_t index,T value) {
    std::memcpy(static_cast<char*>(a.array.mutable_data())+byte_offset(a,index),&value,sizeof(T));
}
template<class T> void arithmetic(Tensor &a,Tensor &b,Tensor &out,Op op) {
    for(size_t i=0;i<out.view.count;++i){T x=load<T>(a,i),y=load<T>(b,i),z;
        switch(op){case Op::add:z=x+y;break;case Op::sub:z=x-y;break;case Op::mul:z=x*y;break;case Op::div:z=x/y;break;
            case Op::pow:if constexpr(std::is_same_v<T,float>)z=cn_crt.powf(x,y);else z=cn_crt.pow(x,y);break;
            case Op::mod:{z=std::fmod(x,y);if(z!=0 && ((y<0)!=(z<0)))z+=y;else if(z==0)z=std::copysign(T(0),y);break;}
            case Op::floordiv:{T mod=std::fmod(x,y);T div=(x-mod)/y;if(mod!=0 && ((y<0)!=(mod<0)))div-=T(1);if(div!=0){z=std::floor(div);if(div-z>T(0.5))z+=T(1);}else z=std::copysign(T(0),x/y);break;}
            default:raise(PyExc_TypeError,py::str("Unsupported array operation in NCNN graph"));}
        store<T>(out,i,z);
    }
}
template<class U> void integer_arithmetic(Tensor &a,Tensor &b,Tensor &out,Op op) {
    // NumPy integer add/subtract/multiply wrap at the dtype width. Unsigned
    // arithmetic also avoids C++ signed overflow for manually supplied weights.
    for(size_t i=0;i<out.view.count;++i){uint64_t x=load<U>(a,i),y=load<U>(b,i),z;
        if(out.view.type==5){
            if(op==Op::sub)raise(PyExc_TypeError,py::str("numpy boolean subtract, the `-` operator, is not supported, use the bitwise_xor, the `^` operator, or the logical_xor function instead."));
            z=op==Op::add?static_cast<uint64_t>(x!=0 || y!=0):static_cast<uint64_t>(x!=0 && y!=0);
        }else switch(op){case Op::add:z=x+y;break;case Op::sub:z=x-y;break;case Op::mul:z=x*y;break;
            default:raise(PyExc_TypeError,py::str("Unsupported integer operation in NCNN graph"));}
        store<U>(out,i,static_cast<U>(z));
    }
}
bool numpy_fused_complex_multiply(const py::dtype &dtype) {
    // loops_arithm_fp.dispatch.c.src fuses (FMA3 muladdsub) on the targets with
    // FMA3, else multiplies, then addsubs. Ask for the dtype's own multiply loop
    // (FFF or DDD), as every NumPy mirror does (native_numpy_simd).
    O simd=py::module_::import("nodes.impl.native_numpy_simd");
    O target=simd.attr("loop_target")(py::str("multiply"),py::str(std::string(3,dtype.char_())));
    return truth(simd.attr("fma3")(target));
}
template<class T> T quiet(T value) {
    // x86 returns a NaN operand with its quiet bit set.
    using Bits=std::conditional_t<sizeof(T)==4,uint32_t,uint64_t>;
    Bits bits;std::memcpy(&bits,&value,sizeof(bits));
    bits|=Bits(1)<<(std::numeric_limits<T>::digits-2);
    std::memcpy(&value,&bits,sizeof(bits));return value;
}
template<class T> T executed(T value) {
    // Keeps a hardware result that a NaN operand overrides, for its FP flags.
    volatile T kept=value;
    return kept;
}
template<class T> T first_nan(std::initializer_list<T> operands,T computed) {
    for(T operand:operands)if(std::isnan(operand))return quiet(operand);
    return computed;
}
CN_FMA_TARGET inline float fused_multiply_add(float a,float b,float c) { return _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(a),_mm_set_ss(b),_mm_set_ss(c))); }
CN_FMA_TARGET inline double fused_multiply_add(double a,double b,double c) { return _mm_cvtsd_f64(_mm_fmadd_sd(_mm_set_sd(a),_mm_set_sd(b),_mm_set_sd(c))); }
CN_FMA_TARGET inline float fused_multiply_subtract(float a,float b,float c) { return _mm_cvtss_f32(_mm_fmsub_ss(_mm_set_ss(a),_mm_set_ss(b),_mm_set_ss(c))); }
CN_FMA_TARGET inline double fused_multiply_subtract(double a,double b,double c) { return _mm_cvtsd_f64(_mm_fmsub_sd(_mm_set_sd(a),_mm_set_sd(b),_mm_set_sd(c))); }
// How NumPy 2.5 multiplies a complex pair. CLONGDOUBLE uses loops.c.src's
// expression. CFLOAT/CDOUBLE use loops_arithm_fp.dispatch.c.src: simd_cmul
// forms (a.imag*b.imag, a.imag*b.real), then muladdsub(a.real, b, that), split
// (multiply, then addsub) on the X86_V2 baseline and fused on X86_V3. That
// target's scalar fallback is its compiler's contraction of the expression:
// fma(a.real, b.real, -(a.imag*b.imag)) and fma(a.imag, b.real, a.real*b.imag).
enum class ComplexMultiply { expression,split,fused,contracted };
template<class T> cn_ncnn_complex<T> complex_multiply(cn_ncnn_complex<T> a,cn_ncnn_complex<T> b,ComplexMultiply kind) {
    // Every hardware operation runs for its FP flags; a NaN operand propagates
    // in the order measured on the pinned wheel.
    if(kind==ComplexMultiply::expression)return {a.real*b.real-a.imag*b.imag,a.real*b.imag+a.imag*b.real};
    if(kind==ComplexMultiply::contracted) {
        const T ii=first_nan({b.imag,a.imag},executed(a.imag*b.imag)),ri=first_nan({b.imag,a.real},executed(a.real*b.imag));
        return {first_nan({b.real,a.real,ii},executed(fused_multiply_subtract(a.real,b.real,ii))),
                first_nan({b.real,a.imag,ri},executed(fused_multiply_add(a.imag,b.real,ri)))};
    }
    const T ii=first_nan({a.imag,b.imag},executed(a.imag*b.imag)),ir=first_nan({a.imag,b.real},executed(a.imag*b.real));
    if(kind==ComplexMultiply::fused)return {first_nan({a.real,b.real,ii},executed(fused_multiply_subtract(a.real,b.real,ii))),
                                            first_nan({a.real,b.imag,ir},executed(fused_multiply_add(a.real,b.imag,ir)))};
    const T rr=first_nan({a.real,b.real},executed(a.real*b.real)),ri=first_nan({a.real,b.imag},executed(a.real*b.imag));
    return {first_nan({rr,ii},executed(rr-ii)),first_nan({ri,ir},executed(ri+ir))};
}
ComplexMultiply complex_multiply_kind(const UfuncOperands &operands,const O &shape,const py::dtype &dtype) {
    if(dtype.char_()=='G')return ComplexMultiply::expression;
    if(!numpy_fused_complex_multiply(dtype))return ComplexMultiply::split;
    // X86_V3's npyv_loadable_stride_f32 divides the byte stride by sizeof(float)
    // as an unsigned value, so a negative stride exceeds NPY_SIMD_MAXLOAD_STRIDE32
    // and the loop falls back to scalar code. Doubles have no load limit there.
    if(dtype.itemsize()==8)for(ptrdiff_t stride:inner_strides(operands,shape))
        if(std::llabs(static_cast<long long>(static_cast<unsigned long long>(stride)/sizeof(float)))>0x7fffffff/8)return ComplexMultiply::contracted;
    return ComplexMultiply::fused;
}
template<class T> void complex_arithmetic(Tensor &a,Tensor &b,Tensor &out,Op op,ComplexMultiply multiply) {
    for(size_t i=0;i<out.view.count;++i){auto x=load<cn_ncnn_complex<T>>(a,i),y=load<cn_ncnn_complex<T>>(b,i);cn_ncnn_complex<T> z;
        switch(op){
        case Op::add:z={x.real+y.real,x.imag+y.imag};break;
        case Op::sub:z={x.real-y.real,x.imag-y.imag};break;
        case Op::mul:z=complex_multiply(x,y,multiply);break;
        case Op::div:z=cn_ncnn_complex_divide(x,y);break;
        default:raise(PyExc_TypeError,py::str("Unsupported complex operation in NCNN graph"));}
        store(out,i,z);
    }
}
O ufunc_result(const O &output) {
    // NumPy ufuncs return a NumPy scalar for zero-dimensional results, even
    // when both inputs were zero-dimensional ndarrays. astype keeps its array.
    if(py::reinterpret_borrow<py::array>(output).ndim()==0)return item(output,py::tuple());
    return output;
}
O array_binary(const O &left,const O &right,Op op,bool warnings=true,int *raised=nullptr) {
    O dtype=np().attr("result_type")(left,right);
    char kind=py::reinterpret_borrow<py::dtype>(dtype).kind();
    if(op==Op::div && kind!='f' && kind!='c')dtype=py::dtype::of<double>();
    // Input promotion and every ufunc result are separate rounding boundaries.
    UfuncOperands operands=ufunc_operands(left,right,dtype);
    O output=array_allocate(operands.left,operands.right,dtype),shape=output.attr("shape");
    int events=py::reinterpret_borrow<py::array>(output).size()?operands.deferred:0;
    O ba=array_broadcast(operands.left,shape),bb=array_broadcast(operands.right,shape);
    Tensor aa(ba),ab(bb),out(output);
    if(op==Op::add && out.view.type<5){int operation_events=0;status(cn_tensor_add_typed(&aa.view,&ab.view,&out.view,&operation_events));events|=operation_events;}
    else {
        bool half=out.view.type==0;
        O compute_dtype=half?O(py::dtype::of<float>()):dtype;
        O wa=half?array_cast(ba,compute_dtype,false,&events):ba;
        O wb=half?array_cast(bb,compute_dtype,false,&events):bb;
        O work=half?array_allocate(wa,wb,compute_dtype):output;
        Tensor ta(wa),tb(wb),tw(work);
        std::feclearexcept(FE_ALL_EXCEPT);
        if(tw.view.type==1)arithmetic<float>(ta,tb,tw,op);
        else if(tw.view.type==2)arithmetic<double>(ta,tb,tw,op);
        else if(tw.view.type==3 || tw.view.type==4) {
            const ComplexMultiply multiply=op==Op::mul?complex_multiply_kind(operands,shape,py::reinterpret_borrow<py::dtype>(dtype)):ComplexMultiply::expression;
            std::feclearexcept(FE_ALL_EXCEPT);
            if(tw.view.type==3)complex_arithmetic<float>(ta,tb,tw,op,multiply);
            else complex_arithmetic<double>(ta,tb,tw,op,multiply);
        }
        else switch(tw.array.itemsize()){
            case 1:integer_arithmetic<uint8_t>(ta,tb,tw,op);break;
            case 2:integer_arithmetic<uint16_t>(ta,tb,tw,op);break;
            case 4:integer_arithmetic<uint32_t>(ta,tb,tw,op);break;
            case 8:integer_arithmetic<uint64_t>(ta,tb,tw,op);break;
            default:raise(PyExc_TypeError,py::str("Unsupported NCNN integer dtype"));}
        events|=fp_events();
        if(half){int cast_events=0;status(cn_tensor_cast_typed(&tw.view,&out.view,&cast_events));events|=cast_events;}
    }
    const char *name=op==Op::add?"add":op==Op::sub?"subtract":op==Op::mul?"multiply":op==Op::pow?"power":op==Op::mod?"remainder":op==Op::floordiv?"floor_divide":"divide";
    if(raised)*raised|=events;
    if(warnings)report_fp(events,name);
    return ufunc_result(output);
}
O array_sqrt(const O &input) {
    O source=asarray(input),dtype=item(np().attr("sqrt").attr("resolve_dtypes")(make_tuple({source.attr("dtype"),py::none()})),py::int_(1));
    bool half=py::reinterpret_borrow<py::dtype>(dtype).itemsize()==2;
    O compute_dtype=half?O(py::dtype::of<float>()):dtype;
    int events=0;
    O work=array_cast(source,compute_dtype,false,&events);
    O output=np().attr("empty_like")(work,py::arg("order")="K",py::arg("subok")=false);
    Tensor a(work),out(output);std::feclearexcept(FE_ALL_EXCEPT);
    // NumPy 2.5.3's compiled npy_csqrt signals invalid for a NaN component in
    // every case but its first (an infinite imaginary part), the special-case
    // returns included (measured on the pinned wheel).
    bool nan_component=false;
    auto complex_sqrt=[&nan_component](auto z){
        nan_component=nan_component || ((std::isnan(z.real) || std::isnan(z.imag)) && !std::isinf(z.imag));
        return cn_ncnn_complex_sqrt(z);
    };
    for(size_t i=0;i<out.view.count;++i){
        if(out.view.type==1)store<float>(out,i,std::sqrt(load<float>(a,i)));
        else if(out.view.type==2)store<double>(out,i,std::sqrt(load<double>(a,i)));
        else if(out.view.type==3)store(out,i,complex_sqrt(load<cn_ncnn_complex<float>>(a,i)));
        else store(out,i,complex_sqrt(load<cn_ncnn_complex<double>>(a,i)));
    }
    events|=fp_events()|(nan_component?8:0);
    if(half)output=array_cast(output,dtype,false,&events);
    report_fp(events,"sqrt");return ufunc_result(output);
}
O binary(const O &a,const O &b,Op op) {
    if(py::isinstance<py::array>(a)||py::isinstance<py::array>(b))return array_binary(a,b,op);
    switch(op){
    case Op::add:return result(PyNumber_Add(a.ptr(),b.ptr()));
    case Op::sub:return result(PyNumber_Subtract(a.ptr(),b.ptr()));
    case Op::mul:return result(PyNumber_Multiply(a.ptr(),b.ptr()));
    case Op::div:return result(PyNumber_TrueDivide(a.ptr(),b.ptr()));
    case Op::floordiv:return result(PyNumber_FloorDivide(a.ptr(),b.ptr()));
    case Op::pow:return result(PyNumber_Power(a.ptr(),b.ptr(),Py_None));
    case Op::mod:return result(PyNumber_Remainder(a.ptr(),b.ptr()));
    }throw std::logic_error("Unknown NCNN scalar operation");
}
O inplace(const O &a,const O &b,Op op) {
    if(py::isinstance<py::array>(a)) {
        py::array dest=py::reinterpret_borrow<py::array>(a);
        if(!dest.writeable())raise(PyExc_ValueError,py::str("output array is read-only"));
        O rhs=asarray(b),ufunc=np().attr(op==Op::add?"add":"subtract");
        // Resolve dtype and output-shape contracts before doing arithmetic,
        // exactly as NumPy does. These primitives do not iterate any elements.
        // A weak Python literal resolves as its type (NEP 50).
        O rhs_dtype=weak_literal(b)?O(py::type::of(b)):O(rhs.attr("dtype"));
        ufunc.attr("resolve_dtypes")(make_tuple({dest.dtype(),rhs_dtype,dest.dtype()}));
        O iterator=np().attr("nditer")(make_tuple({a,rhs,a}),py::arg("flags")=make_list({py::str("zerosize_ok")}),
            py::arg("op_flags")=make_list({make_list({py::str("readonly")}),make_list({py::str("readonly")}),make_list({py::str("readwrite"),py::str("no_broadcast")})}),
            py::arg("casting")="unsafe");
        iterator.attr("close")();
        int events=0;
        O computed=array_binary(a,b,op,false,&events);
        if(!truth(np().attr("can_cast")(computed.attr("dtype"),dest.dtype(),py::arg("casting")="same_kind")))raise(PyExc_TypeError,py::str("Cannot cast ufunc result to the output dtype"));
        O converted=array_cast(computed,dest.dtype(),false,&events);Tensor source(converted),out(a);
        if(source.shape!=out.shape)raise(PyExc_ValueError,py::str("non-broadcastable output operand"));
        size_t width=static_cast<size_t>(dest.itemsize());
        for(size_t i=0;i<out.view.count;++i)std::memcpy(static_cast<char*>(dest.mutable_data())+byte_offset(out,i),static_cast<const char*>(source.array.data())+byte_offset(source,i),width);
        report_fp(events,op==Op::add?"add":"subtract");
        return a;
    }
    if(op==Op::add)return result(PyNumber_InPlaceAdd(a.ptr(),b.ptr()));
    return result(PyNumber_InPlaceSubtract(a.ptr(),b.ptr()));
}

#include "ncnn_optimizer_passes.hpp"

O parameter(const O &collection,const O &pid,const O &types) {
    O dictionary=collection.attr("param_dict"),missing=py::none();
    try{return dictionary[pid];}catch(py::error_already_set &error){if(!error.matches(PyExc_KeyError))throw;missing=error.value();}
    try {
    O key=builtin("str")(pid),schema=types.attr("param_schema")[collection.attr("op")],spec;
    try{spec=schema[key];}catch(py::error_already_set &error){
        if(error.matches(PyExc_KeyError))types.attr("logger").attr("error")(py::str("Op {} does not have param {}, please report").attr("format")(collection.attr("op"),pid));
        throw;
    }
    O def=spec[py::str("defaultValue")],value=def;
    if(py::isinstance<py::str>(value)) {
        py::list entries=builtin("list")(schema.attr("items")());bool found=false;
        for(py::ssize_t i=0;i+1<static_cast<py::ssize_t>(entries.size());++i){O entry=entries[i],candidate=item(entry,py::int_(1));
            if(compare(value,candidate[py::str("paramPhase")],Py_EQ)){
                O ref=builtin("int")(item(entry,py::int_(0)));
                try{value=dictionary[ref].attr("value");}catch(py::error_already_set &e){if(!e.matches(PyExc_KeyError))throw;value=candidate[py::str("defaultValue")];}
                def=candidate[py::str("defaultValue")];found=true;break;
            }
        }
        if(!found){
            O exception=builtin("KeyError")(py::str("Op {} does not have param {}, please report").attr("format")(collection.attr("op"),value));
            PyException_SetCause(exception.ptr(),Py_NewRef(missing.ptr()));
            raise(PyExc_KeyError,exception);
        }
    }
    return types.attr("NcnnParam")(key,spec[py::str("paramPhase")],value,def);
    }catch(py::error_already_set &error){
        PyException_SetContext(error.value().ptr(),Py_NewRef(missing.ptr()));throw;
    }
}
void parameter_set(const O &collection,const O &pid,const O &value,const O &types) {
    O key=builtin("str")(pid),schema=types.attr("param_schema")[collection.attr("op")],spec;
    try{spec=schema[key];}catch(py::error_already_set &error){
        if(error.matches(PyExc_KeyError))types.attr("logger").attr("error")(py::str("Op {} does not have param {}, please report").attr("format")(collection.attr("op"),key));
        throw;
    }
    collection.attr("param_dict")[pid]=types.attr("NcnnParam")(key,spec[py::str("paramPhase")],value,spec[py::str("defaultValue")]);
}
O scalar_text(const O &value) {
    if(py::isinstance<py::float_>(value))return np().attr("format_float_scientific")(value,6,false,py::arg("exp_digits")=2);
    return builtin("str")(value);
}
O parameter_string(const O &collection,const O &types) {
    O schema=types.attr("param_schema")[collection.attr("op")];
    // Sorting and reconstruction are metadata primitives; the serializer's
    // omission and formatting decisions are made here in C++.
    O dictionary=builtin("dict")(builtin("sorted")(collection.attr("param_dict").attr("items")()));
    collection.attr("param_dict")=dictionary;
    O text=py::str("");
    for(py::handle handle:dictionary.attr("values")()) {
        O param=py::reinterpret_borrow<O>(handle),value=param.attr("value"),def=param.attr("default");
        if(compare(value,def,Py_EQ))continue;
        if(py::isinstance<py::str>(def) && !contains(def,py::str("FLT_MAX"))) {
            O pid=py::none();py::list entries=builtin("list")(schema.attr("items")());
            for(py::ssize_t i=0;i+1<static_cast<py::ssize_t>(entries.size());++i){O entry=entries[i];
                if(compare(def,item(entry,py::int_(1))[py::str("paramPhase")],Py_EQ)){pid=builtin("int")(item(entry,py::int_(0)));break;}
            }
            if(pid.is_none())raise(PyExc_KeyError,py::str("Op {} does not have param {}, please report").attr("format")(collection.attr("op"),def));
            O related=dictionary[pid];
            if(contains(make_tuple({related.attr("value"),related.attr("default")}),value))continue;
        }
        bool list=py::isinstance<py::list>(value);
        text=binary(text,list?binary(py::str(" -233"),param.attr("id").attr("zfill")(2),Op::add):binary(py::str(" "),param.attr("id"),Op::add),Op::add);
        text=binary(text,py::str("="),Op::add);
        if(list){py::list parts;for(py::handle v:value)parts.append(scalar_text(py::reinterpret_borrow<O>(v)));text=binary(text,py::str(",").attr("join")(parts),Op::add);}
        else text=binary(text,scalar_text(value),Op::add);
    }
    return text;
}
O parse_number(const O &text) {
    return (contains(text,py::str("."))||contains(text,py::str("e")))?builtin("float")(text):builtin("int")(text);
}
O parse_layer(const O &line,const O &types) {
    O tokens=line.attr("strip")().attr("split")();
    O prefix=unpack(item(tokens,py::slice(py::int_(0),py::int_(2),py::none())),2);
    O op=item(prefix,py::int_(0)),name=item(prefix,py::int_(1));
    require(!compare(op,py::str("MemoryData"),Py_EQ),py::str("This NCNN param file contains invalid layers"));
    O ni=builtin("int")(item(tokens,py::int_(2))),no=builtin("int")(item(tokens,py::int_(3)));
    O end=binary(py::int_(4),ni,Op::add),last=binary(end,no,Op::add);
    O inputs=builtin("list")(item(tokens,py::slice(py::int_(4),end,py::none()))),outputs=builtin("list")(item(tokens,py::slice(end,last,py::none())));
    O dictionary=py::dict();
    for(py::handle h:item(tokens,py::slice(last,py::none(),py::none()))) {
        O token=py::reinterpret_borrow<O>(h),parts=unpack(token.attr("split")(py::str("=")),2);
        O key=item(parts,py::int_(0)),data=item(parts,py::int_(1)),pid=builtin("int")(key),value;
        if(compare(pid,py::int_(0),Py_LT)) {
            py::list values;for(py::handle number:data.attr("split")(py::str(",")))values.append(parse_number(py::reinterpret_borrow<O>(number)));
            value=values;pid=builtin("abs")(binary(pid,py::int_(23300),Op::add));key=builtin("str")(pid);
        }else value=parse_number(data);
        O spec=types.attr("param_schema")[op][key];
        dictionary[pid]=types.attr("NcnnParam")(key,spec[py::str("paramPhase")],value,spec[py::str("defaultValue")]);
    }
    O params=types.attr("NcnnParamCollection")(op,dictionary);
    return make_tuple({op,types.attr("NcnnLayer")(op,name,ni,no,inputs,outputs,params)});
}
O array_bytes(const O &object) {
    if(!py::isinstance<py::array>(object))raise(PyExc_TypeError,py::str("descriptor 'tobytes' for 'numpy.ndarray' objects doesn't apply to a '{}' object").attr("format")(Py_TYPE(object.ptr())->tp_name));
    py::array array=py::reinterpret_borrow<py::array>(object);
    py::bytes output(nullptr,static_cast<py::ssize_t>(array.nbytes()));
    char *destination=PyBytes_AS_STRING(output.ptr());size_t itemsize=static_cast<size_t>(array.itemsize());
    for(size_t i=0;i<static_cast<size_t>(array.size());++i){
        size_t index=i;ptrdiff_t offset=0;
        for(py::ssize_t d=array.ndim();d>0;--d){auto dimension=static_cast<size_t>(array.shape(d-1));offset+=static_cast<ptrdiff_t>(index%dimension)*array.strides(d-1);index/=dimension;}
        std::memcpy(destination+i*itemsize,static_cast<const char*>(array.data())+offset,itemsize);
    }
    return output;
}
O layer_add_weight(const O &layer,const O &name,const O &data,const O &tag,const O &types) {
    O source=data;
    if(py::isinstance<py::float_>(data))source=np().attr("array")(data,py::dtype::of<float>());
    else if(py::isinstance<py::int_>(data))source=np().attr("array")(data,py::dtype::of<int32_t>());
    else {O method=data.attr("astype");(void)method;} // Force the lazy accessor; preserve missing-method errors.
    O dtype=compare(tag,types.attr("DTYPE_FP16"),Py_EQ)?O(np().attr("float16")):O(np().attr("float32"));
    O array=array_cast(source,dtype);
    if(!py::isinstance<py::array>(source) && PyObject_IsInstance(source.ptr(),np().attr("generic").ptr())==1)array=array[py::tuple()];
    layer.attr("weight_data")[name]=types.attr("NcnnWeight")(array,tag);
    return binary(py::int_(py::len(tag)),py::int_(py::len(array_bytes(asarray(array)))),Op::add);
}
O serialize_weights(const O &model) {
    py::list pieces;
    for(py::handle h:model.attr("layers")){
        O layer=py::reinterpret_borrow<O>(h),weights=layer.attr("weight_data");
        if(!truth(weights)||compare(layer.attr("op_type"),py::str("ncnnfused"),Py_EQ))continue;
        for(py::handle w:weights.attr("values")()){O weight=py::reinterpret_borrow<O>(w);pieces.append(weight.attr("quantize_tag"));pieces.append(array_bytes(weight.attr("weight")));}
    }
    return py::bytes("").attr("join")(pieces);
}
O interp_layers(const O &a,const O &b,const O &alpha,const O &types) {
    O weights_a=a.attr("weight_data"),weights_b=b.attr("weight_data");
    py::dict weights;O encoded=py::bytes("");
    if(truth(weights_a)){
        require(py::len(weights_a)==py::len(weights_b),py::str("All corresponding nodes must have same number of weights"));
        py::list pieces;
        for(py::handle entry:weights_a.attr("items")()){
            O pair=py::reinterpret_borrow<O>(entry),name=item(pair,py::int_(0)),wa=item(pair,py::int_(1)),wb;
            try{wb=item(weights_b,name);}catch(py::error_already_set &error){
                if(error.matches(PyExc_KeyError))types.attr("logger").attr("error")(py::str("Weights in node {} and {} do not correspond").attr("format")(a.attr("name"),b.attr("name")));
                throw;
            }
            require(compare(wa.attr("shape"),wb.attr("shape"),Py_EQ),py::str("Corresponding weights must have the same size and shape"));
            require(py::len(wa.attr("quantize_tag"))==py::len(wb.attr("quantize_tag")),py::str("Weights must either both have or both not have a quantize tag"));
            O convert=py::none();
            if(compare(wa.attr("quantize_tag"),types.attr("DTYPE_FP16"),Py_EQ) && compare(wb.attr("quantize_tag"),types.attr("DTYPE_FP32"),Py_EQ))convert=wb;
            else if(compare(wa.attr("quantize_tag"),types.attr("DTYPE_FP32"),Py_EQ) && compare(wb.attr("quantize_tag"),types.attr("DTYPE_FP16"),Py_EQ))convert=wa;
            if(!convert.is_none()){
                convert.attr("quantize_tag")=types.attr("DTYPE_FP16");
                O source=convert.attr("weight"),cast=array_cast(source,np().attr("float16"));
                if(!py::isinstance<py::array>(source))cast=ufunc_result(cast);
                convert.attr("weight")=cast;
            }
            // Separate statements retain Python's left-to-right evaluation and
            // each ufunc's dtype/rounding/error boundary.
            O left=binary(wa.attr("weight"),alpha,Op::mul);
            O complement=binary(py::int_(1),alpha,Op::sub);
            O right=binary(wb.attr("weight"),complement,Op::mul);
            O data=binary(left,right,Op::add),weight=types.attr("NcnnWeight")(data,wa.attr("quantize_tag"));
            pieces.append(binary(weight.attr("quantize_tag"),array_bytes(asarray(weight.attr("weight"))),Op::add));
            weights[name]=weight;
        }
        encoded=py::bytes("").attr("join")(pieces);
    }
    O layer=types.attr("NcnnLayer")(a.attr("op_type"),a.attr("name"),a.attr("num_inputs"),a.attr("num_outputs"),a.attr("inputs"),a.attr("outputs"),a.attr("params"),weights);
    return make_tuple({layer,encoded});
}
O interpolate_model(const O &a,const O &b,const O &alpha,const O &types) {
    // deepcopy is the retained host object protocol, including user-defined
    // attributes/cycles/__deepcopy__. All graph pairing and weight work below
    // is owned by this translation unit.
    O output=py::module_::import("copy").attr("deepcopy")(a);
    py::list left,right;py::ssize_t i=0;
    for(py::handle handle:a.attr("layers")){O layer=py::reinterpret_borrow<O>(handle);if(truth(layer.attr("weight_data")))left.append(make_tuple({py::int_(i),layer}));++i;}
    i=0;
    for(py::handle handle:b.attr("layers")){O layer=py::reinterpret_borrow<O>(handle);if(truth(layer.attr("weight_data")))right.append(make_tuple({py::int_(i),layer}));++i;}
    require(left.size()==right.size(),py::str("Models must have same number of layers containing weights"));
    py::list encoded;
    for(py::size_t index=0;index<left.size();++index){
        O lhs=left[index],rhs=right[index];
        O pair=interp_layers(item(lhs,py::int_(1)),item(rhs,py::int_(1)),alpha,types);
        set_item(output.attr("layers"),item(lhs,py::int_(0)),item(pair,py::int_(0)));
        encoded.append(item(pair,py::int_(1)));
    }
    return output;
}
O write_param(const O &model,const O &types) {
    O text=py::str("{}\n{} {}\n").attr("format")(model.attr("magic"),model.attr("node_count"),model.attr("blob_count"));
    for(py::handle h:model.attr("layers")){
        O layer=py::reinterpret_borrow<O>(h);
        if(compare(layer.attr("op_type"),py::str("ncnnfused"),Py_EQ))continue;
        text=binary(text,py::str("{:<16} {:<24} {} {}").attr("format")(layer.attr("op_type"),layer.attr("name"),layer.attr("num_inputs"),layer.attr("num_outputs")),Op::add);
        for(const char *field:{"inputs","outputs"})if(truth(layer.attr(field)))text=binary(text,binary(py::str(" "),py::str(" ").attr("join")(layer.attr(field)),Op::add),Op::add);
        O params=layer.attr("params");
        if(truth(params.attr("param_dict")))text=binary(text,parameter_string(params,types),Op::add);
        text=binary(text,py::str("\n"),Op::add);
    }
    return text;
}

O param_value(const O &layer,int id) {return layer.attr("params")[py::int_(id)].attr("value");}
O read_array(const O &stream,const O &bytes,const O &dtype) {
    O raw=stream.attr("read")(bytes);py::buffer buffer=py::reinterpret_borrow<py::buffer>(raw);
    auto data=buffer.request();py::dtype dt=np().attr("dtype")(dtype);
    py::ssize_t length=data.size*data.itemsize;
    if(length%dt.itemsize())raise(PyExc_ValueError,py::str("buffer size must be a multiple of element size"));
    py::array array(dt,{length/dt.itemsize()},{dt.itemsize()},data.ptr,raw);
    if(data.readonly)array.attr("setflags")(py::arg("write")=false);
    return array;
}
O load_weights(const O &stream,const O &op,const O &layer,const O &types) {
    py::dict weights;O integer=builtin("int");
    auto is=[&](const char *name){return compare(op,py::str(name),Py_EQ);};
    auto count=[&](int id){return checked(integer,param_value(layer,id));};
    auto mul=[](const O &a,int b){return binary(a,py::int_(b),Op::mul);};
    auto add=[&](const char *name,const O &array,const O &tag=py::bytes("")){weights[py::str(name)]=types.attr("NcnnWeight")(array,tag);};
    if(is("BatchNorm")) {
        O bytes=mul(count(0),4);
        for(const char *name:{"slope","mean","variance","bias"})add(name,read_array(stream,bytes,py::dtype::of<float>()));
    }else if(is("Convolution")||is("ConvolutionDepthWise")||is("Deconvolution")) {
        O tag=stream.attr("read")(4),dtype=types.attr("DTYPE_DICT")[tag],length=count(6);
        O bytes=mul(length,compare(tag,types.attr("DTYPE_FP16"),Py_EQ)?2:4);
        O has_bias=param_value(layer,5),filters=count(0),kw=count(1),kh=count(11),inputs,shape;
        if(is("ConvolutionDepthWise")) {
            O group=count(7),per_group=binary(filters,group,Op::floordiv);
            inputs=binary(binary(binary(length,per_group,Op::floordiv),kw,Op::floordiv),kh,Op::floordiv);
            shape=make_tuple({group,per_group,binary(inputs,group,Op::floordiv),kh,kw});
        }else {
            inputs=binary(binary(binary(length,filters,Op::floordiv),kw,Op::floordiv),kh,Op::floordiv);
            shape=make_tuple({filters,inputs,kh,kw});
        }
        add("weight",array_reshape(read_array(stream,bytes,dtype),shape),tag);
        if(truth(has_bias))add("bias",read_array(stream,mul(filters,4),py::dtype::of<float>()));
    }else if(is("InnerProduct")) {
        O tag=stream.attr("read")(4),dtype=types.attr("DTYPE_DICT")[tag],length=param_value(layer,2);
        require(py::isinstance<py::int_>(length),py::str("Weight data size must be int"));
        O bytes=mul(length,compare(tag,types.attr("DTYPE_FP16"),Py_EQ)?2:4);
        O data=read_array(stream,bytes,dtype),outputs=param_value(layer,0);
        require(py::isinstance<py::int_>(outputs),py::str("Num output must be int"));
        O inputs=binary(length,outputs,Op::floordiv);
        add("weight",array_reshape(data,make_tuple({inputs,outputs})),tag);
        if(compare(param_value(layer,1),py::int_(1),Py_EQ))add("bias",read_array(stream,mul(outputs,4),py::dtype::of<float>()));
    }else if(is("PReLU")) {
        O count_=param_value(layer,0);require(py::isinstance<py::int_>(count_),py::str("Num slopes must be int"));
        add("slope",read_array(stream,mul(count_,4),py::dtype::of<float>()));
    }else if(is("Scale")) {
        O length=param_value(layer,0);require(py::isinstance<py::int_>(length),py::str("Scale data size must be int"));
        if(!compare(length,py::int_(-233),Py_EQ)) {
            O tag=stream.attr("read")(4),dtype=types.attr("DTYPE_DICT")[tag];
            O bytes=mul(length,compare(tag,types.attr("DTYPE_FP16"),Py_EQ)?2:4);
            add("weight",read_array(stream,bytes,dtype),tag);
            if(compare(param_value(layer,1),py::int_(1),Py_EQ))add("bias",read_array(stream,mul(length,4),py::dtype::of<float>()));
        }
    }else if(py::len(layer.attr("params").attr("weight_order"))!=0) {
        raise(PyExc_ValueError,py::str("Load weights not added for {} yet, please report").attr("format")(op));
    }
    return weights;
}
O read_model(const O &parameters,const O &weights,const O &types) {
    O model=types.attr("NcnnModel")();parameters.attr("readline")();
    O counts=parameters.attr("readline")().attr("strip")().attr("split")(py::str(" "));
    model.attr("node_count")=builtin("int")(item(counts,py::int_(0)));
    model.attr("blob_count")=builtin("int")(item(counts,py::int_(1)));
    for(py::handle line:parameters){O parsed=parse_layer(py::reinterpret_borrow<O>(line),types),layer=item(parsed,py::int_(1));
        layer.attr("weight_data")=load_weights(weights,item(parsed,py::int_(0)),layer,types);
        model.attr("layers").attr("append")(layer);
    }
    weights.attr("seek")(0,2);model.attr("bin_length")=weights.attr("tell")();return model;
}
O get_nf(const O &layer) {
    O nf=param_value(layer,0),kw=param_value(layer,1),kh;
    try{kh=param_value(layer,11);}catch(py::error_already_set &error){if(!error.matches(PyExc_KeyError))throw;kh=kw;}
    O count=param_value(layer,6);
    require(py::isinstance<py::int_>(nf)&&py::isinstance<py::int_>(kw)&&py::isinstance<py::int_>(kh)&&py::isinstance<py::int_>(count),py::str("Out nc, kernel width and height, and weight data size must all be ints"));
    return make_tuple({nf,binary(binary(binary(count,nf,Op::floordiv),kw,Op::floordiv),kh,Op::floordiv)});
}
O broadcast_data(const O &model,const O &types) {
    O scale=py::float_(1.0),inputs=py::int_(0),outputs=py::int_(0),nf=py::int_(0),fp=py::str("fp32"),shuffle=py::int_(1),current=py::none();
    bool found=false;py::ssize_t i=0;O layers=model.attr("layers");
    for(py::handle h:layers){O layer=py::reinterpret_borrow<O>(h),op=layer.attr("op_type");
        if(compare(op,py::str("Interp"),Py_EQ)){
            try{O next=item(layers,py::int_(i+1));
                if(!compare(next.attr("op_type"),py::str("BinaryOp"),Py_EQ)&&!compare(param_value(next,0),py::int_(0),Py_EQ))scale=binary(scale,checked(builtin("float"),param_value(layer,1)),Op::mul);
            }catch(py::error_already_set &error){if(!error.matches(PyExc_IndexError))throw;scale=binary(scale,checked(builtin("float"),param_value(layer,1)),Op::mul);}
        }else if(compare(op,py::str("PixelShuffle"),Py_EQ)){
            scale=binary(scale,checked(builtin("int"),param_value(layer,0)),Op::mul);shuffle=binary(shuffle,checked(builtin("int"),param_value(layer,0)),Op::mul);
        }else if(contains(make_tuple({py::str("Convolution"),py::str("Convolution1D"),py::str("ConvolutionDepthWise")}),op)){
            if(!found){O dims=get_nf(layer);nf=item(dims,py::int_(0));inputs=item(dims,py::int_(1));
                if(compare(layer.attr("weight_data")[py::str("weight")].attr("quantize_tag"),types.attr("DTYPE_FP16"),Py_EQ)){fp=py::str("fp16");}found=true;}
            scale=binary(scale,checked(builtin("int"),param_value(layer,3)),Op::div);current=layer;
        }else if(contains(make_tuple({py::str("Deconvolution"),py::str("DeconvolutionDepthWise")}),op)){
            if(!found){O dims=get_nf(layer);nf=item(dims,py::int_(0));inputs=item(dims,py::int_(1));found=true;}
            scale=binary(scale,checked(builtin("int"),param_value(layer,3)),Op::mul);current=layer;
        }
        ++i;
    }
    require(!current.is_none(),py::str("Cannot broadcast; model has no Convolution layers"));
    outputs=binary(checked(builtin("int"),param_value(current,0)),binary(shuffle,py::int_(2),Op::pow),Op::floordiv);
    require(compare(scale,py::int_(1),Py_GE),py::str("Models with scale less than 1x not supported"));
    require(compare(binary(scale,py::int_(1),Op::mod),py::int_(0),Py_EQ),py::str("Model not supported, scale {} is not an integer").attr("format")(scale));
    return make_tuple({builtin("int")(scale),inputs,outputs,nf,fp});
}
}

pybind11::object cn_graph_binary(const pybind11::object &a,const pybind11::object &b,int op) {
    if(op<0 || op>6)throw pybind11::value_error("Unknown native graph numeric operation");
    return binary(a,b,static_cast<Op>(op));
}
pybind11::object cn_graph_inplace(const pybind11::object &a,const pybind11::object &b,int op) {
    if(op<0 || op>6)throw pybind11::value_error("Unknown native graph numeric operation");
    return inplace(a,b,static_cast<Op>(op));
}
pybind11::object cn_graph_array_cast(const pybind11::object &a,const pybind11::object &dtype) {
    return array_cast(a,dtype);
}
void cn_graph_report_fp(int events,const char *operation) { report_fp(events,operation); }
cn_graph_operands cn_graph_ufunc_operands(const pybind11::object &a,const pybind11::object &b,const pybind11::object &dtype) {
    UfuncOperands operands=ufunc_operands(a,b,dtype);
    return {operands.left,operands.right,operands.deferred};
}

void cn_bind_ncnn(pybind11::module_ &module) {
    module.def("ncnn_optimize",pass_optimize);
    module.def("ncnn_optimizer_pass",[](const O &model,const O &types,const std::string &name){
        using Pass=void(*)(const O&,const O&);
        const std::pair<const char*,Pass> passes[]={
#define NCNN_PASS(name) {#name,pass_##name}
            NCNN_PASS(fuse_batchnorm_scale),NCNN_PASS(fuse_x_batchnorm),NCNN_PASS(fuse_x_mul),NCNN_PASS(fuse_x_add),NCNN_PASS(fuse_innerproduct_dropout),
            NCNN_PASS(replace_reduction_with_global_pooling),NCNN_PASS(replace_prelu_with_leaky_relu),NCNN_PASS(fuse_x_activation),NCNN_PASS(fuse_memorydata_binaryop),NCNN_PASS(fuse_binaryop_eltwise),
            NCNN_PASS(eliminate_dropout),NCNN_PASS(eliminate_pooling1x1),NCNN_PASS(eliminate_noop),NCNN_PASS(eliminate_split),NCNN_PASS(eliminate_flatten_after_global_pooling),NCNN_PASS(eliminate_reshape_after_global_pooling),NCNN_PASS(eliminate_reshape_before_binaryop),
            NCNN_PASS(replace_convolution_with_innerproduct_after_global_pooling),NCNN_PASS(replace_convolution_with_innerproduct_after_innerproduct),NCNN_PASS(eliminate_flatten_after_innerproduct),NCNN_PASS(eliminate_orphaned_memorydata)
#undef NCNN_PASS
        };
        for(const auto &entry:passes)if(name==entry.first){entry.second(model,types);return;}
        throw py::value_error("Unknown NCNN optimizer pass");
    });
    module.def("ncnn_param_get",parameter);
    module.def("ncnn_param_set",parameter_set);
    module.def("ncnn_param_string",parameter_string);
    module.def("ncnn_parse_layer",parse_layer);
    module.def("ncnn_add_weight",layer_add_weight);
    module.def("ncnn_serialize_weights",serialize_weights);
    module.def("ncnn_interp_layers",interp_layers);
    module.def("ncnn_interpolate",interpolate_model);
    module.def("ncnn_write_param",write_param);
    module.def("ncnn_load_weights",load_weights);
    module.def("ncnn_read_model",read_model);
    module.def("ncnn_get_nf",get_nf);
    module.def("ncnn_broadcast_data",broadcast_data);
}
