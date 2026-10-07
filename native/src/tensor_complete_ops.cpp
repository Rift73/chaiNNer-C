/* Typed CPU tensor arithmetic. Python owns dtype/shape resolution and framework
 * objects; this translation unit owns every numeric element operation. Separate
 * stages deliberately round at the same ufunc/TensorIterator boundaries.
 * No framework headers, private runtime symbols, model execution or GPU access. */
#include "parallel.h"
#include <atomic>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {
enum class Kind : int {
    half, single, real, complex_single, complex_real, boolean, u8, u16, u32,
    u64, i8, i16, i32, i64, bfloat, complex_half
};
struct View {
    void *data;
    size_t count, dimensions;
    const size_t *shape;
    const ptrdiff_t *strides;
    int type;
};
constexpr size_t widths[] = {2,4,8,8,16,1,1,2,4,8,1,2,4,8,2,4};

template<class T> T read(const char *p) noexcept {
    T value; std::memcpy(&value, p, sizeof(T)); return value;
}
template<class T> void write(char *p, T value) noexcept {
    std::memcpy(p, &value, sizeof(T));
}
float half_value(uint16_t h) noexcept {
    uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    uint32_t fraction = h & 0x3ffu, exponent = (h >> 10) & 31u, bits;
    if (!exponent) {
        if (!fraction) bits = sign;
        else {
            int e = -14;
            while (!(fraction & 0x400u)) { fraction <<= 1; --e; }
            bits = sign | (static_cast<uint32_t>(e + 127) << 23) | ((fraction & 0x3ffu) << 13);
        }
    } else if (exponent == 31) bits = sign | 0x7f800000u | (fraction << 13);
    else bits = sign | ((exponent + 112u) << 23) | (fraction << 13);
    return std::bit_cast<float>(bits);
}
double half_double(uint16_t h) noexcept {
    uint32_t bits=std::bit_cast<uint32_t>(half_value(h));
    if ((h & 0x7c00u)==0x7c00u) {
        uint64_t sign=static_cast<uint64_t>(h & 0x8000u)<<48;
        uint64_t fraction=static_cast<uint64_t>(h & 0x3ffu)<<42;
        return std::bit_cast<double>(sign | UINT64_C(0x7ff0000000000000) | fraction);
    }
    return static_cast<double>(std::bit_cast<float>(bits));
}
double widen_float(float value) noexcept {
    uint32_t bits=std::bit_cast<uint32_t>(value);
    if ((bits & 0x7f800000u)==0x7f800000u) {
        uint64_t sign=static_cast<uint64_t>(bits & 0x80000000u)<<32;
        uint64_t fraction=static_cast<uint64_t>(bits & 0x7fffffu)<<29;
        return std::bit_cast<double>(sign | UINT64_C(0x7ff0000000000000) | fraction);
    }
    return static_cast<double>(value);
}
/* Direct double -> half rounding avoids the double-rounding of a float detour. */
uint16_t half_bits(double value) noexcept {
    uint64_t bits = std::bit_cast<uint64_t>(value), fraction = bits & UINT64_C(0xfffffffffffff);
    uint16_t sign = static_cast<uint16_t>((bits >> 48) & 0x8000u);
    unsigned exponent = static_cast<unsigned>((bits >> 52) & 2047u);
    if (exponent == 2047u) {
        uint16_t payload = static_cast<uint16_t>(fraction >> 42);
        return static_cast<uint16_t>(sign | 0x7c00u | (fraction ? (payload ? payload : 1u) : 0u));
    }
    int e = static_cast<int>(exponent) - 1023 + 15;
    if (e >= 31) { std::feraiseexcept(FE_OVERFLOW); return static_cast<uint16_t>(sign | 0x7c00u); }
    if (e < -10) {
        if (value != 0) std::feraiseexcept(FE_UNDERFLOW);
        return sign;
    }
    unsigned shift = 42;
    if (e <= 0) { fraction |= UINT64_C(0x10000000000000); shift += static_cast<unsigned>(1-e); }
    uint64_t result = fraction >> shift;
    uint64_t remainder = fraction & ((UINT64_C(1) << shift) - 1);
    uint64_t halfway = UINT64_C(1) << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (result & 1))) ++result;
    if (e > 0) result += static_cast<uint64_t>(e) << 10;
    else if (remainder) std::feraiseexcept(FE_UNDERFLOW);
    if (result >= 0x7c00u) std::feraiseexcept(FE_OVERFLOW);
    return static_cast<uint16_t>(sign | result);
}
float bfloat_value(uint16_t b) noexcept {
    return std::bit_cast<float>(static_cast<uint32_t>(b) << 16);
}
uint16_t bfloat_bits(float value) noexcept {
    uint32_t bits = std::bit_cast<uint32_t>(value);
    if (std::isnan(value)) return 0x7fc0u; // PyTorch's canonical BFloat16 NaN.
    return static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
bool complex_kind(int type) noexcept { return type == 3 || type == 4 || type == 15; }
bool integer_kind(int type) noexcept { return type >= 5 && type <= 13; }
template<class T> T real_value(const char *p, int type) noexcept {
    switch (static_cast<Kind>(type)) {
    case Kind::half: case Kind::complex_half:
        if constexpr (sizeof(T)==sizeof(double)) return half_double(read<uint16_t>(p));
        else return half_value(read<uint16_t>(p));
    case Kind::single: case Kind::complex_single: return static_cast<T>(read<float>(p));
    case Kind::real: case Kind::complex_real: return static_cast<T>(read<double>(p));
    case Kind::bfloat: return static_cast<T>(bfloat_value(read<uint16_t>(p)));
    case Kind::boolean: return static_cast<T>(read<uint8_t>(p) != 0);
    case Kind::u8: return static_cast<T>(read<uint8_t>(p));
    case Kind::u16: return static_cast<T>(read<uint16_t>(p));
    case Kind::u32: return static_cast<T>(read<uint32_t>(p));
    case Kind::u64: return static_cast<T>(read<uint64_t>(p));
    case Kind::i8: return static_cast<T>(read<int8_t>(p));
    case Kind::i16: return static_cast<T>(read<int16_t>(p));
    case Kind::i32: return static_cast<T>(read<int32_t>(p));
    case Kind::i64: return static_cast<T>(read<int64_t>(p));
    }
    return T(0);
}
template<class T> T imag_value(const char *p, int type) noexcept {
    return complex_kind(type) ? real_value<T>(p + widths[type]/2, type) : T(0);
}
template<class T> T operand_value(const char *p, int input_type, int output_type) noexcept {
    if (output_type == 0 || output_type == 15)
        return static_cast<T>(half_value(half_bits(real_value<double>(p,input_type))));
    if (output_type == 14)
        return static_cast<T>(bfloat_value(bfloat_bits(real_value<float>(p,input_type))));
    return real_value<T>(p,input_type);
}
template<class T> T operand_imag(const char *p, int input_type, int output_type) noexcept {
    return complex_kind(input_type) ? operand_value<T>(p+widths[input_type]/2,input_type,output_type) : T(0);
}
template<class T> void store_real(char *p, int type, T value) noexcept {
    if (type == 0 || type == 15) write(p, half_bits(static_cast<double>(value)));
    else if (type == 14) write(p, bfloat_bits(static_cast<float>(value)));
    else if (type == 1 || type == 3) write(p, static_cast<float>(value));
    else write(p, static_cast<double>(value));
}

struct Bounds { uintptr_t low, high; };
cn_status object_bounds(const void *pointer, size_t bytes, size_t alignment, Bounds &bounds) noexcept {
    uintptr_t address=reinterpret_cast<uintptr_t>(pointer);
    if (!pointer || address % alignment) return CN_INVALID_ARGUMENT;
    if (bytes > UINTPTR_MAX-address) return CN_SIZE_OVERFLOW;
    bounds={address,address+bytes};
    return CN_OK;
}
cn_status validate(const View *v, Bounds &bounds) noexcept {
    Bounds metadata{};
    cn_status status=object_bounds(v,sizeof(View),alignof(View),metadata);
    if (status != CN_OK) return status;
    if (!v->data || v->type < 0 || v->type >= 16 || v->dimensions > 32 ||
        (v->dimensions && (!v->shape || !v->strides))) return CN_INVALID_ARGUMENT;
    if (v->dimensions) {
        status=object_bounds(v->shape,v->dimensions*sizeof(size_t),alignof(size_t),metadata);
        if (status != CN_OK) return status;
        status=object_bounds(v->strides,v->dimensions*sizeof(ptrdiff_t),alignof(ptrdiff_t),metadata);
        if (status != CN_OK) return status;
    }
    size_t count = 1, negative = 0, positive = 0;
    for (size_t d = 0; d < v->dimensions; ++d) {
        size_t length = v->shape[d];
        if (length && count > SIZE_MAX / length) return CN_SIZE_OVERFLOW;
        count *= length;
        if (length <= 1) continue;
        ptrdiff_t stride = v->strides[d];
        if (stride == PTRDIFF_MIN) return CN_SIZE_OVERFLOW;
        size_t magnitude = static_cast<size_t>(stride < 0 ? -stride : stride);
        if (magnitude && length-1 > static_cast<size_t>(PTRDIFF_MAX) / magnitude) return CN_SIZE_OVERFLOW;
        size_t span = (length-1) * magnitude;
        size_t &total = stride < 0 ? negative : positive;
        if (total > static_cast<size_t>(PTRDIFF_MAX)-span) return CN_SIZE_OVERFLOW;
        total += span;
    }
    if (count != v->count) return CN_INVALID_ARGUMENT;
    if (count > SIZE_MAX/widths[v->type]) return CN_SIZE_OVERFLOW;
    uintptr_t address = reinterpret_cast<uintptr_t>(v->data);
    if (negative > address || positive > UINTPTR_MAX-address ||
        widths[v->type] > UINTPTR_MAX-address-positive) return CN_SIZE_OVERFLOW;
    bounds = {address-negative, address+positive+(count ? widths[v->type] : 0)};
    return CN_OK;
}
bool same_shape(const View &a, const View &b) noexcept {
    if (a.dimensions != b.dimensions || a.count != b.count) return false;
    for (size_t d=0; d<a.dimensions; ++d) if (a.shape[d] != b.shape[d]) return false;
    return true;
}
bool overlap(Bounds a, Bounds b) noexcept { return a.low < b.high && b.low < a.high; }
char *at(const View &v, size_t index) noexcept {
    ptrdiff_t offset = 0;
    for (size_t d=v.dimensions; d>0; --d) {
        size_t coordinate = index % v.shape[d-1]; index /= v.shape[d-1];
        offset += static_cast<ptrdiff_t>(coordinate) * v.strides[d-1];
    }
    return static_cast<char *>(v.data) + offset;
}
uint64_t integer_value(const char *p, int type) noexcept {
    switch (static_cast<Kind>(type)) {
    case Kind::boolean: return read<uint8_t>(p) != 0;
    case Kind::u8: return read<uint8_t>(p);
    case Kind::u16: return read<uint16_t>(p);
    case Kind::u32: return read<uint32_t>(p);
    case Kind::u64: return read<uint64_t>(p);
    case Kind::i8: return static_cast<uint64_t>(read<int8_t>(p));
    case Kind::i16: return static_cast<uint64_t>(read<int16_t>(p));
    case Kind::i32: return static_cast<uint64_t>(read<int32_t>(p));
    case Kind::i64: return static_cast<uint64_t>(read<int64_t>(p));
    default: return 0;
    }
}
/* Explicit Windows NumPy conversion sentinels, rather than undefined C++
 * floating-to-integer casts for exceptional model coefficients. */
uint64_t cast_integer(double value, int type) noexcept {
    if (type == 9) {
        double shifted = value >= 9223372036854775808.0 ? value-9223372036854775808.0 : value;
        if (!(shifted >= -9223372036854775808.0 && shifted < 9223372036854775808.0)) {
            std::feraiseexcept(FE_INVALID);
            return UINT64_C(0x8000000000000000);
        }
        uint64_t result = static_cast<uint64_t>(static_cast<int64_t>(shifted));
        return value >= 9223372036854775808.0 ? result ^ UINT64_C(0x8000000000000000) : result;
    }
    if (type == 8 || type == 13) {
        if (!(value >= -9223372036854775808.0 && value < 9223372036854775808.0)) {
            std::feraiseexcept(FE_INVALID); return UINT64_C(0x8000000000000000);
        }
        return static_cast<uint64_t>(static_cast<int64_t>(value));
    }
    if (!(value >= -2147483648.0 && value < 2147483648.0)) {
        std::feraiseexcept(FE_INVALID); return UINT64_C(0xffffffff80000000);
    }
    return static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(value)));
}
void store_integer(char *p, int type, uint64_t value) noexcept {
    if (widths[type] == 1) write(p, static_cast<uint8_t>(value));
    else if (widths[type] == 2) write(p, static_cast<uint16_t>(value));
    else if (widths[type] == 4) write(p, static_cast<uint32_t>(value));
    else write(p, value);
}
struct Job {
    const View *a, *b, *out;
    double amount;
    int operation, framework;
    std::atomic<int> events{0};
};
template<class T> void arithmetic(Job &j, const char *a, const char *b, char *out) noexcept {
    int type = j.out->type;
    T ar = j.operation==2 ? real_value<T>(a,j.a->type) : operand_value<T>(a,j.a->type,type);
    T ai = complex_kind(type) ? (j.operation==2 ? imag_value<T>(a,j.a->type) : operand_imag<T>(a,j.a->type,type)) : T(0);
    if (j.operation == 0) {
        T coefficient = static_cast<T>(j.amount);
        if (type == 15 || (!j.framework && type == 0)) coefficient = static_cast<T>(half_value(half_bits(j.amount)));
        T real = ar * coefficient;
        if (complex_kind(type)) {
            T imag_zero = ai * T(0), real_zero = ar * T(0);
            T imag = ai * coefficient;
            real -= imag_zero;
            imag += real_zero;
            store_real(out+widths[type]/2, type, imag);
        }
        store_real(out, type, real);
    } else if (j.operation == 1) {
        T br = operand_value<T>(b,j.b->type,type), bi = operand_imag<T>(b,j.b->type,type);
        store_real(out, type, ar + br);
        if (complex_kind(type)) store_real(out+widths[type]/2, type, ai+bi);
    } else {
        store_real(out, type, ar);
        if (complex_kind(type)) store_real(out+widths[type]/2, type, ai);
    }
}
void run_range(void *opaque, size_t begin, size_t end) noexcept {
    auto &j = *static_cast<Job *>(opaque);
    std::fenv_t environment;
    std::fegetenv(&environment); std::feclearexcept(FE_ALL_EXCEPT);
    for (size_t i=begin; i<end; ++i) {
        const char *a = at(*j.a,i), *b = j.b ? at(*j.b,i) : nullptr;
        char *out = at(*j.out,i);
        int type = j.out->type;
        if (j.operation == 2 && type == j.a->type) {
            std::memcpy(out,a,widths[type]);
        } else if (j.operation == 2 && integer_kind(type)) {
            uint64_t value;
            if (type == 5 && j.a->type==0) value=(read<uint16_t>(a)&0x7fffu)!=0;
            else if (type == 5) value = real_value<double>(a,j.a->type) != 0 || imag_value<double>(a,j.a->type) != 0;
            else if (integer_kind(j.a->type)) value = integer_value(a,j.a->type);
            else value = cast_integer(real_value<double>(a,j.a->type),type);
            store_integer(out,type,value);
        } else if (j.operation==2 && type==0 && (j.a->type==1 || j.a->type==3)) {
            write(out,half_bits(widen_float(read<float>(a))));
        } else if (type == 2 || type == 4 || (j.operation == 2 && (type == 0 || type == 15))) {
            arithmetic<double>(j,a,b,out);
        } else arithmetic<float>(j,a,b,out);
    }
    int raised = std::fetestexcept(FE_OVERFLOW|FE_UNDERFLOW|FE_INVALID);
    int flags = ((raised & FE_OVERFLOW) ? 2 : 0) | ((raised & FE_UNDERFLOW) ? 4 : 0) | ((raised & FE_INVALID) ? 8 : 0);
    std::fesetenv(&environment);
    j.events.fetch_or(flags,std::memory_order_relaxed);
}
cn_status execute(const View *a, const View *b, const View *out, double amount,
                  int operation, int framework, int *events) noexcept {
    if (operation == 1 && !b) return CN_INVALID_ARGUMENT;
    Bounds ba{}, bb{}, bo{};
    cn_status status = validate(a,ba); if (status != CN_OK) return status;
    status = validate(out,bo); if (status != CN_OK) return status;
    if (!same_shape(*a,*out) || (a->count && overlap(ba,bo))) return CN_INVALID_ARGUMENT;
    if (b) {
        status = validate(b,bb); if (status != CN_OK) return status;
        if (!same_shape(*b,*out) || (b->count && overlap(bb,bo))) return CN_INVALID_ARGUMENT;
    }
    if ((operation != 2 && integer_kind(out->type)) || framework < 0 || framework > 1) return CN_INVALID_ARGUMENT;
    Bounds event_bounds{};
    if (events) {
        status=object_bounds(events,sizeof(int),alignof(int),event_bounds);
        if (status != CN_OK) return status;
        if (overlap(event_bounds,ba) || overlap(event_bounds,bo) ||
            (b && overlap(event_bounds,bb))) return CN_INVALID_ARGUMENT;
    }
    const View *views[]={a,b,out};
    for (const View *view : views) {
        if (!view) continue;
        Bounds metadata[3]{};
        (void)object_bounds(view,sizeof(View),alignof(View),metadata[0]);
        if (view->dimensions) {
            (void)object_bounds(view->shape,view->dimensions*sizeof(size_t),alignof(size_t),metadata[1]);
            (void)object_bounds(view->strides,view->dimensions*sizeof(ptrdiff_t),alignof(ptrdiff_t),metadata[2]);
        }
        for (Bounds region : metadata) {
            if (overlap(bo,region) || (events && overlap(event_bounds,region))) return CN_INVALID_ARGUMENT;
        }
    }
    if (!out->count) { if (events) *events=0; return CN_OK; }
    // Output axes must describe disjoint elements, including foreign layouts.
    size_t span = widths[out->type];
    size_t last = 0;
    for (size_t pass=0; pass<out->dimensions; ++pass) {
        size_t next=SIZE_MAX, dimension=out->dimensions;
        for (size_t d=0; d<out->dimensions; ++d) {
            if (out->shape[d] <= 1) continue;
            if (out->strides[d] <= 0) return CN_INVALID_ARGUMENT;
            size_t stride=static_cast<size_t>(out->strides[d]);
            if (stride > last && stride < next) { next=stride; dimension=d; }
        }
        if (dimension==out->dimensions) break;
        if (next < span) return CN_INVALID_ARGUMENT;
        for (size_t d=0; d<out->dimensions; ++d)
            if (d!=dimension && out->shape[d]>1 && out->strides[d]==static_cast<ptrdiff_t>(next)) return CN_INVALID_ARGUMENT;
        span=next*(out->shape[dimension]-1)+span;
        last=next;
    }
    Job job{a,b,out,amount,operation,framework};
    status=cn_parallel_for(out->count,65536,run_range,&job);
    if (events) *events=job.events.load(std::memory_order_relaxed);
    return status;
}
} // namespace

extern "C" CN_EXPORT cn_status cn_tensor_scale_typed(const View *a, const View *out,
        double amount, int framework, int *events) noexcept {
    return execute(a,nullptr,out,amount,0,framework,events);
}
extern "C" CN_EXPORT cn_status cn_tensor_add_typed(const View *a, const View *b,
        const View *out, int *events) noexcept {
    return execute(a,b,out,0,1,0,events);
}
extern "C" CN_EXPORT cn_status cn_tensor_cast_typed(const View *a, const View *out,
        int *events) noexcept {
    return execute(a,nullptr,out,0,2,0,events);
}
