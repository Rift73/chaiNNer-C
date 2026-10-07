/* Fixed native ONNX-to-NCNN application conversion. Public protobuf objects
 * retain unknown fields, identity and partial mutations. Container/scalar and
 * tensor-format adapters remain public dependencies; graph algorithms do not.
 * Derived from chaiNNer, GPL-3.0. See the pinned generator/source manifest. */
#include "graph_python.hpp"
#include "graph_unpack.hpp"
#include "graph_numeric.hpp"
#include "ncnn_tensor_abi.hpp"
#include <pybind11/numpy.h>
#include <algorithm>
#include <cfenv>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {
using namespace graphpy;
enum class Op { add, sub, mul, div, floordiv, pow, mod };
O np() { return py::module_::import("numpy"); }
const O &local(const O &value,const char *name) {
    if(!value)raise(PyExc_UnboundLocalError,py::str(std::string("cannot access local variable '")+name+"' where it is not associated with a value"));
    return value;
}
O binary(const O &a,const O &b,Op op) { return cn_graph_binary(a,b,static_cast<int>(op)); }
O inplace(const O &a,const O &b,Op op) { return cn_graph_inplace(a,b,static_cast<int>(op)); }
O make_dict(std::initializer_list<std::pair<O,O>> values) {
    py::dict result;
    for(const auto &v:values)result[v.first]=v.second;
    return result;
}
O concat_text(std::initializer_list<O> values) {
    O result=py::str("");
    for(const auto &v:values) {
        PyObject *next=PyUnicode_Concat(result.ptr(),v.ptr());
        if(!next)throw py::error_already_set();
        result=py::reinterpret_steal<O>(next);
    }
    return result;
}
O array_construct(const O &values,const O &dtype=py::none()) {
    // Public allocation/sequence-to-storage conversion, not graph arithmetic.
    return np().attr("array")(values,py::arg("dtype")=dtype);
}
O array_empty(const O &shape,const O &dtype) { return np().attr("empty")(shape,py::arg("dtype")=dtype); }
O scalar_compare(const O &a,const O &b,int op) {
    PyObject *value=PyObject_RichCompare(a.ptr(),b.ptr(),op);
    if(!value)throw py::error_already_set();
    return py::reinterpret_steal<O>(value);
}
#include "onnx_converter_compare.hpp"
O rich_compare(const O &a,const O &b,int op) {
    if(!py::isinstance<py::array>(a) && !py::isinstance<py::array>(b))return scalar_compare(a,b,op);
    return converter_compare::compare(a,b,op);
}
O array_reduce_bool(const O &value,bool all) {
    O array=np().attr("asarray")(value),flat=array.attr("flat");
    auto size=array.attr("size").cast<py::ssize_t>();
    bool result=all;
    for(py::ssize_t i=0;i<size;++i) {
        bool element=truth(item(flat,py::int_(i)));
        if(element!=all){result=element;break;}
    }
    return np().attr("bool_")(py::bool_(result));
}
O array_any(const O &value) { return array_reduce_bool(value,false); }
O array_all(const O &value) { return array_reduce_bool(value,true); }
ptrdiff_t offset(const py::array &array,py::ssize_t index) {
    ptrdiff_t result=0;
    for(py::ssize_t d=array.ndim();d>0;--d) {
        auto n=array.shape(d-1);
        result+=(index%n)*array.strides(d-1);index/=n;
    }
    return result;
}
template<class T> T read(const char *source,ptrdiff_t stride,size_t index) {
    T result;std::memcpy(&result,source+static_cast<ptrdiff_t>(index)*stride,sizeof(T));return result;
}
template<class T> T pair_sum(const char *source,size_t n,ptrdiff_t stride) {
    if(n<8) { T result=static_cast<T>(-0.0);for(size_t i=0;i<n;++i)result+=read<T>(source,stride,i);return result; }
    if(n<=128) {
        T partial[8];for(size_t i=0;i<8;++i)partial[i]=read<T>(source,stride,i);
        size_t i=8;
        for(;i<n-n%8;i+=8)for(size_t j=0;j<8;++j)partial[j]+=read<T>(source,stride,i+j);
        T result=((partial[0]+partial[1])+(partial[2]+partial[3]))+((partial[4]+partial[5])+(partial[6]+partial[7]));
        for(;i<n;++i)result+=read<T>(source,stride,i);
        return result;
    }
    size_t left=n/2;left-=left%8;
    return pair_sum<T>(source,left,stride)+pair_sum<T>(source+static_cast<ptrdiff_t>(left)*stride,n-left,stride);
}
ptrdiff_t reduction_offset(const py::array &a,py::ssize_t index) {
    ptrdiff_t result=0;
    for(py::ssize_t d=a.ndim();d>0;--d)if(d!=2) {
        auto n=a.shape(d-1);result+=(index%n)*a.strides(d-1);index/=n;
    }
    return result;
}
bool pairwise_axis(const py::array &a) {
    auto stride=std::abs(a.strides(1));
    for(py::ssize_t d=0;d<a.ndim();++d)if(d!=1 && a.shape(d)>1 && std::abs(a.strides(d))<stride)return false;
    return true;
}
size_t reduction_block(const py::array &a) {
    // NumPy 2.5's iterator (nditer_constr.c, npyiter_find_buffering_setup)
    // gives an unbuffered GROWINNER reduction one inner loop per whole row.
    // Only a buffered (byte-swapped or unaligned) operand is copied in chunks
    // of the ufunc buffer size, which stop at the end of the row.
    const bool buffered=!truth(a.dtype().attr("isnative")) || !truth(a.attr("flags").attr("aligned"));
    if(!buffered)return std::numeric_limits<size_t>::max();
    return np().attr("getbufsize")().cast<size_t>();
}
template<class T> void sum_rows(const py::array &a,py::array &out,bool pairwise,size_t block) {
    auto n=static_cast<size_t>(a.shape(1));
    for(py::ssize_t i=0;i<out.size();++i) {
        const char *row=static_cast<const char*>(a.data())+reduction_offset(a,i);
        T result=0;
        if(pairwise) {
            for(size_t start=0;start<n;) {
                auto count=std::min<size_t>(n-start,block);
                result+=pair_sum<T>(row+static_cast<ptrdiff_t>(start)*a.strides(1),count,a.strides(1));
                start+=count;
            }
        } else for(size_t j=0;j<n;++j)result+=read<T>(row,a.strides(1),j);
        std::memcpy(static_cast<char*>(out.mutable_data())+i*static_cast<py::ssize_t>(sizeof(T)),&result,sizeof(T));
    }
}
template<class T> struct Complex { T real,imag; };
template<class T> T complex_component_add(T a,T b) {
    T result=a+b;
    // The pinned complex reduction loop gives the right operand precedence
    // when both operands are NaNs. Make that independent of register reuse.
    return std::isnan(a) && std::isnan(b) ? b+b : result;
}
template<class T> Complex<T> complex_add(Complex<T> a,Complex<T> b) {
    return {complex_component_add(a.real,b.real),complex_component_add(a.imag,b.imag)};
}
template<class T> Complex<T> complex_pair_sum(const char *row,size_t n,ptrdiff_t stride) {
    using C=Complex<T>;
    if(n<4) { C result{-T(0),-T(0)};for(size_t i=0;i<n;++i)result=complex_add(result,read<C>(row,stride,i));return result; }
    if(n<=64) {
        C sums[4];for(size_t i=0;i<4;++i)sums[i]=read<C>(row,stride,i);
        size_t i=4;for(;i<n-n%4;i+=4)for(size_t j=0;j<4;++j)sums[j]=complex_add(sums[j],read<C>(row,stride,i+j));
        C result=complex_add(complex_add(sums[0],sums[1]),complex_add(sums[2],sums[3]));
        for(;i<n;++i){result=complex_add(result,read<C>(row,stride,i));}return result;
    }
    size_t left=n/2;left-=left%4;
    return complex_add(complex_pair_sum<T>(row,left,stride),complex_pair_sum<T>(row+static_cast<ptrdiff_t>(left)*stride,n-left,stride));
}
template<class T> void sum_complex_rows(const py::array &a,py::array &out,bool pairwise,size_t block) {
    using C=Complex<T>;auto n=static_cast<size_t>(a.shape(1));
    for(py::ssize_t i=0;i<out.size();++i) {
        const char *row=static_cast<const char*>(a.data())+reduction_offset(a,i);
        C result{0,0};
        if(pairwise)for(size_t start=0;start<n;) {
            auto count=std::min<size_t>(n-start,block);
            result=complex_add(result,complex_pair_sum<T>(row+static_cast<ptrdiff_t>(start)*a.strides(1),count,a.strides(1)));start+=count;
        } else for(size_t j=0;j<n;++j)result=complex_add(result,read<C>(row,a.strides(1),j));
        std::memcpy(static_cast<char*>(out.mutable_data())+i*static_cast<py::ssize_t>(sizeof(C)),&result,sizeof(C));
    }
}
O array_sum(const O &value,const O &axis) {
    auto a=py::reinterpret_borrow<py::array>(np().attr("asarray")(value));
    // Normalize the axis through NumPy's public metadata-only validation (NumPy 2's
    // home for it; numpy.core is a deprecated alias of numpy._core).
    int selected=py::module_::import("numpy.lib.array_utils").attr("normalize_axis_index")(axis,py::int_(a.ndim())).cast<int>();
    if(selected!=1)raise(PyExc_ValueError,py::str("Converter bias reduction requires axis 1"));
    O dtype=a.dtype();
    char kind=a.dtype().kind();
    if(kind=='f' && a.itemsize()==2) {
        bool pairwise=pairwise_axis(a);
        size_t inner=reduction_block(a);
        O promoted=cn_graph_array_cast(a,py::dtype::of<float>());
        auto work=py::reinterpret_borrow<py::array>(promoted);
        py::list shape;for(py::ssize_t d=0;d<a.ndim();++d)if(d!=1)shape.append(py::int_(a.shape(d)));
        auto output=py::reinterpret_borrow<py::array>(array_empty(shape,np().attr("dtype")("float16")));
        int events=0;
        // In a strided multi-output reduction, NumPy's half loop writes each
        // addition back to half. A contiguous reduction writes once per inner
        // loop. Reuse the established checked cast ABI for those roundings.
        for(py::ssize_t i=0;i<output.size();++i) {
            const char *row=static_cast<const char*>(work.data())+reduction_offset(work,i);
            uint16_t half=0;float total=0;
            cn_ncnn_tensor_view f={&total,1,0,nullptr,nullptr,1};
            cn_ncnn_tensor_view h={&half,1,0,nullptr,nullptr,0};
            auto count=static_cast<size_t>(work.shape(1));
            for(size_t start=0;start<count;) {
                auto block=pairwise?std::min<size_t>(count-start,inner):size_t(1);
                int raised=0;
                if(cn_tensor_cast_typed(&h,&f,&raised)!=0)throw std::logic_error("Invalid scalar half accumulator");
                events|=raised;std::feclearexcept(FE_ALL_EXCEPT);
                total+=pair_sum<float>(row+static_cast<ptrdiff_t>(start)*work.strides(1),block,work.strides(1));
                int flags=std::fetestexcept(FE_OVERFLOW|FE_INVALID|FE_UNDERFLOW);
                events|=((flags&FE_OVERFLOW)?2:0)|((flags&FE_UNDERFLOW)?4:0)|((flags&FE_INVALID)?8:0);
                if(cn_tensor_cast_typed(&f,&h,&raised)!=0)throw std::logic_error("Invalid scalar half accumulator");
                events|=raised;start+=block;
            }
            std::memcpy(static_cast<char*>(output.mutable_data())+i*2,&half,2);
        }
        cn_graph_report_fp(events,"reduce");return output;
    }
    size_t block=reduction_block(a);
    if(!truth(a.dtype().attr("isnative")))a=py::reinterpret_borrow<py::array>(cn_graph_array_cast(a,a.dtype().attr("newbyteorder")(py::str("="))));
    bool pairwise=pairwise_axis(a);
    if(kind=='i' || kind=='u' || kind=='b') {
        O native=np().attr("dtype")(np().attr(kind=='u'?"uint":"int_"));
        if(kind=='b' || a.itemsize()<native.attr("itemsize").cast<py::ssize_t>())a=py::reinterpret_borrow<py::array>(cn_graph_array_cast(a,native));
    }
    py::list shape;for(py::ssize_t d=0;d<a.ndim();++d)if(d!=1)shape.append(py::int_(a.shape(d)));
    O result=array_empty(shape,a.dtype());
    auto out=py::reinterpret_borrow<py::array>(result);
    std::feclearexcept(FE_ALL_EXCEPT);
    if(kind=='f') {
        if(a.itemsize()==4)sum_rows<float>(a,out,pairwise,block);else sum_rows<double>(a,out,pairwise,block);
    } else if(kind=='c') {
        if(a.itemsize()==8)sum_complex_rows<float>(a,out,pairwise,block);else sum_complex_rows<double>(a,out,pairwise,block);
    } else if(kind=='i' || kind=='u' || kind=='b') {
        if(a.itemsize()==4)sum_rows<uint32_t>(a,out,false,block);else sum_rows<uint64_t>(a,out,false,block);
    } else raise(PyExc_TypeError,py::str("Converter bias tensor must have numeric storage"));
    int flags=std::fetestexcept(FE_OVERFLOW|FE_INVALID|FE_UNDERFLOW);
    cn_graph_report_fp(((flags&FE_OVERFLOW)?2:0)|((flags&FE_UNDERFLOW)?4:0)|((flags&FE_INVALID)?8:0),"reduce");
    return result;
}
O array_delete(const O &value,const O &index) {
    auto a=py::reinterpret_borrow<py::array>(np().attr("asarray")(value));
    auto i=index.cast<py::ssize_t>(),n=a.size();
    if(i<0)i+=n;
    if(i<0 || i>=n)raise(PyExc_IndexError,py::str("index "+py::str(index).cast<std::string>()+" is out of bounds for axis 0 with size "+std::to_string(n)));
    O result=array_empty(py::int_(n-1),a.dtype());
    auto out=py::reinterpret_borrow<py::array>(result);
    auto width=static_cast<size_t>(a.itemsize());
    if(a.dtype().has_fields() || truth(a.dtype().attr("hasobject"))) {
        O source=a.attr("flat"),destination=out.attr("flat");
        for(py::ssize_t j=0,k=0;j<n;++j)if(j!=i)set_item(destination,py::int_(k++),item(source,py::int_(j)));
        return result;
    }
    for(py::ssize_t j=0,k=0;j<n;++j)if(j!=i) {
        std::memcpy(static_cast<char*>(out.mutable_data())+k*out.itemsize(),static_cast<const char*>(a.data())+offset(a,j),width);++k;
    }
    return result;
}

#include "onnx_converter_passes.hpp"
} // namespace

void cn_bind_onnx_converter(py::module_ &module) {
    bind_converter(module);
    module.def("onnx_converter_sum_axis", &array_sum);
    module.def("onnx_converter_compare", &rich_compare);
}
