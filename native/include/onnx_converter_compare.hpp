/* Native numeric comparison for the NumPy 2.5.3 ndarray contract. Complex
 * predicate order, the Python-int comparisons and the operand preparation are
 * altered adaptations of numpy/_core/src/umath/loops.c.src,
 * special_integer_comparisons.cpp, ufunc_object.c and
 * numpy/_core/src/multiarray/arrayobject.c. Allocation/promotion remain public
 * metadata operations; element comparisons and traversal are native here.
 *
 * Copyright (c) 2005-2022, NumPy Developers. All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * * Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of the NumPy Developers nor the names of any contributors
 *   may be used to endorse or promote products derived from this software
 *   without specific prior written permission.
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#pragma once

namespace converter_compare {
template<class T> struct FloatingBits;
template<> struct FloatingBits<float> {
    using U = uint32_t;
    static constexpr U sign = 0x80000000U, infinity = 0x7f800000U, quiet = 0x00400000U;
};
template<> struct FloatingBits<double> {
    using U = uint64_t;
    static constexpr U sign = UINT64_C(0x8000000000000000),
        infinity = UINT64_C(0x7ff0000000000000), quiet = UINT64_C(0x0008000000000000);
};
template<class T> int nan_kind(T value) noexcept {
    using Bits = FloatingBits<T>;
    typename Bits::U bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & ~Bits::sign) <= Bits::infinity ? 0 : (bits & Bits::quiet) ? 1 : 2;
}
template<class T> bool ordinary(T left, T right, int op) noexcept {
    switch(op) {
    case Py_LT: return left < right;
    case Py_LE: return left <= right;
    case Py_EQ: return left == right;
    case Py_NE: return left != right;
    case Py_GT: return left > right;
    case Py_GE: return left >= right;
    default: return false;
    }
}
template<class T> bool floating(T left, T right, int op) noexcept {
    // Real comparison loops clear their FP flags, including signaling NaNs.
    if(nan_kind(left) || nan_kind(right)) return op == Py_NE;
    return ordinary(left, right, op);
}
bool half(uint16_t left, uint16_t right, int op) noexcept {
    const auto a = left & 0x7fffU, b = right & 0x7fffU;
    if(a > 0x7c00U || b > 0x7c00U) return op == Py_NE;
    const bool equal = left == right || (a == 0 && b == 0);
    bool less = false;
    if(!equal) less = (left ^ right) & 0x8000U ? (left & 0x8000U) != 0
        : (left & 0x8000U) ? left > right : left < right;
    switch(op) {
    case Py_LT: return less;
    case Py_LE: return less || equal;
    case Py_EQ: return equal;
    case Py_NE: return !equal;
    case Py_GT: return !less && !equal;
    case Py_GE: return !less;
    default: return false;
    }
}
template<class T> struct ComplexValue { T real, imaginary; };
template<class T> bool complex(ComplexValue<T> left, ComplexValue<T> right,
                             int op, int &events) noexcept {
    const int lr = nan_kind(left.real), rr = nan_kind(right.real);
    const bool equal_real = !lr && !rr && left.real == right.real;
    if(op == Py_EQ || op == Py_NE) {
        if(lr == 2 || rr == 2) events |= 8;
        bool equal = false;
        if(equal_real) {
            const int li = nan_kind(left.imaginary), ri = nan_kind(right.imaginary);
            if(li == 2 || ri == 2) events |= 8;
            equal = !li && !ri && left.imaginary == right.imaginary;
        }
        return op == Py_EQ ? equal : !equal;
    }
    if(lr || rr) { events |= 8; return false; }
    const bool ordered_real = (op == Py_LT || op == Py_LE)
        ? left.real < right.real : left.real > right.real;
    if(ordered_real) {
        const int li = nan_kind(left.imaginary);
        if(li == 2) events |= 8;
        if(li) return false;
        const int ri = nan_kind(right.imaginary);
        if(ri == 2) events |= 8;
        return !ri;
    }
    if(!equal_real) return false;
    if(nan_kind(left.imaginary) || nan_kind(right.imaginary)) { events |= 8; return false; }
    return ordinary(left.imaginary, right.imaginary, op);
}
ptrdiff_t position(const py::array &array, py::ssize_t index) noexcept {
    ptrdiff_t result = 0;
    for(py::ssize_t d = array.ndim(); d > 0; --d) {
        result += (index % array.shape(d - 1)) * array.strides(d - 1);
        index /= array.shape(d - 1);
    }
    return result;
}
template<class T, class Compare> void traverse(const py::array &left,
        const py::array &right, py::array &output, Compare compare) {
    for(py::ssize_t i = 0; i < output.size(); ++i) {
        T a, b;
        std::memcpy(&a, static_cast<const char*>(left.data()) + position(left, i), sizeof(T));
        std::memcpy(&b, static_cast<const char*>(right.data()) + position(right, i), sizeof(T));
        const bool value = compare(a, b);
        std::memcpy(static_cast<char*>(output.mutable_data()) + position(output, i), &value, 1);
    }
}
O allocation(const O &left, const O &right) {
    O iterator = np().attr("nditer")(make_tuple({left, right, py::none()}),
        py::arg("flags") = make_list({py::str("zerosize_ok"), py::str("refs_ok")}),
        py::arg("op_flags") = make_list({make_list({py::str("readonly")}),
            make_list({py::str("readonly")}), make_list({py::str("writeonly"), py::str("allocate")})}),
        py::arg("op_dtypes") = make_list({py::none(), py::none(), py::dtype::of<bool>()}));
    O output = item(iterator.attr("operands"), py::int_(2));
    iterator.attr("close")();
    return output;
}
O object_operand(const O &value) {
    O array = np().attr("asarray")(value);
    auto source = py::reinterpret_borrow<py::array>(array);
    // Object promotion represents real/complex values with CPython's double
    // storage. Do these numeric casts natively, retaining cast warnings.
    if(source.dtype().kind() == 'f' && source.itemsize() < 8)
        return cn_graph_array_cast(array, py::dtype::of<double>());
    if(source.dtype().kind() == 'c' && source.itemsize() < 16)
        return cn_graph_array_cast(array, np().attr("dtype")("complex128"));
    return array;
}
struct ObjectFloatingState {
    std::fenv_t previous;
    ObjectFloatingState() { std::fegetenv(&previous); std::feclearexcept(FE_ALL_EXCEPT); }
    ~ObjectFloatingState() { std::fesetenv(&previous); }
    int events() const noexcept {
        const int state = std::fetestexcept(FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW | FE_INVALID);
        return ((state & FE_DIVBYZERO) ? 1 : 0) | ((state & FE_OVERFLOW) ? 2 : 0)
            | ((state & FE_UNDERFLOW) ? 4 : 0) | ((state & FE_INVALID) ? 8 : 0);
    }
};
O objects(const O &left, const O &right, const O &output_object, int op) {
    O a = object_operand(left), b = object_operand(right);
    O views = np().attr("broadcast_arrays")(a,b);
    auto av = py::reinterpret_borrow<py::array>(item(views,py::int_(0)));
    auto bv = py::reinterpret_borrow<py::array>(item(views,py::int_(1)));
    auto output = py::reinterpret_borrow<py::array>(output_object);
    O af = av.attr("flat"), bf = bv.attr("flat");
    int events = 0;
    for(py::ssize_t i = 0; i < output.size(); ++i) {
        O x = item(af,py::int_(i)), y = item(bf,py::int_(i));
        if(av.dtype().kind() != 'O') x = x.attr("item")();
        if(bv.dtype().kind() != 'O') y = y.attr("item")();
        bool value;
        {
            ObjectFloatingState flags;
            value = truth(scalar_compare(x,y,op));
            events |= flags.events();
        }
        std::memcpy(static_cast<char*>(output.mutable_data()) + position(output,i),&value,1);
    }
    const char *names[] = {"less", "less_equal", "equal", "not_equal", "greater", "greater_equal"};
    cn_graph_report_fp(events,names[op]);
    return output.ndim() ? output_object : item(output_object,py::tuple());
}
int integer_range(const O &value, const py::dtype &dtype) {
    // get_value_range: -1 below the dtype's range, 0 inside, 1 above.
    const auto bits = dtype.itemsize() * 8;
    const bool is_signed = dtype.kind() == 'i';
    const long long minimum = !is_signed ? 0 : bits == 64 ? LLONG_MIN : -(1LL << (bits - 1));
    const unsigned long long maximum = is_signed ? (bits == 64 ? LLONG_MAX : (1ULL << (bits - 1)) - 1)
        : bits == 64 ? ULLONG_MAX : (1ULL << bits) - 1;
    int overflow = 0;
    const long long parsed = PyLong_AsLongLongAndOverflow(value.ptr(), &overflow);
    if(parsed == -1 && overflow == 0 && PyErr_Occurred()) throw py::error_already_set();
    if(overflow == 0)
        return parsed < minimum ? -1 : parsed > 0 && static_cast<unsigned long long>(parsed) > maximum ? 1 : 0;
    if(overflow < 0) return -1;
    if(maximum <= static_cast<unsigned long long>(LLONG_MAX)) return 1;
    return scalar_compare(value, py::int_(maximum), Py_GT).ptr() == Py_True ? 1 : 0;
}
O numeric(const O &left, const O &right, int op) {
    const py::dtype left_dtype = py::reinterpret_borrow<py::array>(left).dtype();
    if(PyLong_CheckExact(right.ptr()) && (left_dtype.kind() == 'i' || left_dtype.kind() == 'u')) {
        // special_integer_comparisons.cpp: a Python int outside the integer
        // dtype's range gives every element the same answer, without FP events.
        const int range = integer_range(right, left_dtype);
        if(range != 0) {
            O output_object = allocation(left, py::bool_(false));
            auto output = py::reinterpret_borrow<py::array>(output_object);
            const bool value = range < 0 ? (op == Py_NE || op == Py_GT || op == Py_GE)
                : (op == Py_NE || op == Py_LT || op == Py_LE);
            for(py::ssize_t i = 0; i < output.size(); ++i)
                std::memcpy(static_cast<char*>(output.mutable_data()) + position(output, i), &value, 1);
            return output.ndim() ? output_object : item(output_object, py::tuple());
        }
    }
    O dtype = np().attr("result_type")(left, right);
    const auto descriptor = py::reinterpret_borrow<py::dtype>(dtype);
    const char kind = descriptor.kind();
    if(kind != 'f' && kind != 'c' && kind != 'b' && kind != 'i' && kind != 'u' && kind != 'O')
        raise(PyExc_TypeError, py::str("Converter comparison requires numeric storage"));
    if(kind == 'O') return objects(left, right, allocation(left, right), op);
    // Weak literals and the up-front casts come before the broadcast check.
    // Same-dtype copies preserve sNaNs.
    const cn_graph_operands operands = cn_graph_ufunc_operands(left, right, dtype);
    O output_object = allocation(operands.left, operands.right);
    O views = np().attr("broadcast_arrays")(operands.left, operands.right);
    auto av = py::reinterpret_borrow<py::array>(item(views, py::int_(0)));
    auto bv = py::reinterpret_borrow<py::array>(item(views, py::int_(1)));
    auto output = py::reinterpret_borrow<py::array>(output_object);
    int events = 0;
    if(kind == 'f') {
        if(descriptor.itemsize() == 2) traverse<uint16_t>(av, bv, output, [op](auto x, auto y) { return half(x,y,op); });
        else if(descriptor.itemsize() == 4) traverse<float>(av,bv,output,[op](auto x,auto y) { return floating(x,y,op); });
        else traverse<double>(av,bv,output,[op](auto x,auto y) { return floating(x,y,op); });
    } else if(kind == 'c') {
        // The real loops clear the FP status after comparing, so only a complex
        // loop reports the events of the iterator's casts, with its own.
        if(descriptor.itemsize() == 8) traverse<ComplexValue<float>>(av,bv,output,[&](auto x,auto y) { return complex(x,y,op,events); });
        else traverse<ComplexValue<double>>(av,bv,output,[&](auto x,auto y) { return complex(x,y,op,events); });
        if(output.size()) events |= operands.deferred;
    } else if(kind == 'b') {
        traverse<uint8_t>(av,bv,output,[op](auto x,auto y) { return ordinary(x != 0,y != 0,op); });
    } else if(kind == 'u') {
        switch(descriptor.itemsize()) {
        case 1: traverse<uint8_t>(av,bv,output,[op](auto x,auto y) { return ordinary(x,y,op); }); break;
        case 2: traverse<uint16_t>(av,bv,output,[op](auto x,auto y) { return ordinary(x,y,op); }); break;
        case 4: traverse<uint32_t>(av,bv,output,[op](auto x,auto y) { return ordinary(x,y,op); }); break;
        default: traverse<uint64_t>(av,bv,output,[op](auto x,auto y) { return ordinary(x,y,op); }); break;
        }
    } else {
        switch(descriptor.itemsize()) {
        case 1: traverse<int8_t>(av,bv,output,[op](auto x,auto y) { return ordinary(x,y,op); }); break;
        case 2: traverse<int16_t>(av,bv,output,[op](auto x,auto y) { return ordinary(x,y,op); }); break;
        case 4: traverse<int32_t>(av,bv,output,[op](auto x,auto y) { return ordinary(x,y,op); }); break;
        default: traverse<int64_t>(av,bv,output,[op](auto x,auto y) { return ordinary(x,y,op); }); break;
        }
    }
    const char *names[] = {"less", "less_equal", "equal", "not_equal", "greater", "greater_equal"};
    cn_graph_report_fp(events, names[op]);
    return output.ndim() ? output_object : item(output_object, py::tuple());
}
O compare(const O &left, const O &right, int op) {
    if(op < Py_LT || op > Py_GE) raise(PyExc_ValueError,py::str("Invalid comparison operation"));
    // A NumPy scalar facing an array compares through its zero-dimensional
    // array (gentype_richcompare).
    if(!py::isinstance<py::array>(left) && py::isinstance(left, np().attr("generic")))
        return compare(np().attr("asarray")(left), right, op);
    if(!py::isinstance<py::array>(left) && py::isinstance<py::array>(right)) {
        const int reflected[] = {Py_GT,Py_GE,Py_EQ,Py_NE,Py_LT,Py_LE};
        return compare(right,left,reflected[op]);
    }
    // NumPy 2 array_richcompare: a failed numeric EQ/NE comparison raises; it
    // no longer falls back to an identity result.
    return numeric(left, right, op);
}
} // namespace converter_compare
