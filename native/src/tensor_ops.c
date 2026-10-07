/* CPU weight interpolation. Half operations round after each multiply and
 * addition, matching NumPy's float16 ufunc stages. No model execution occurs. */
#include "parallel.h"
#include <string.h>

static float half_to_float(uint16_t half) {
    uint32_t sign = ((uint32_t)half & 0x8000u) << 16;
    uint32_t fraction = (uint32_t)half & 0x03ffu;
    uint32_t exponent = ((uint32_t)half >> 10) & 31u;
    uint32_t bits;
    if (!exponent) {
        if (!fraction) bits = sign;
        else {
            int e = -14;
            while (!(fraction & 0x0400u)) { fraction <<= 1; --e; }
            bits = sign | ((uint32_t)(e + 127) << 23) | ((fraction & 0x03ffu) << 13);
        }
    } else if (exponent == 31u) bits = sign | 0x7f800000u | (fraction << 13);
    else bits = sign | ((exponent + 112u) << 23) | (fraction << 13);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint16_t float_to_half(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    uint32_t fraction = bits & 0x007fffffu;
    uint32_t exponent = (bits >> 23) & 255u;
    if (exponent == 255u)
        return (uint16_t)(sign | 0x7c00u | (fraction ? (fraction >> 13) | 0x0200u : 0));
    int e = (int)exponent - 127 + 15;
    if (e >= 31) return (uint16_t)(sign | 0x7c00u);
    if (e <= 0) {
        if (e < -10) return (uint16_t)sign;
        fraction |= 0x00800000u;
        unsigned int shift = (unsigned int)(14 - e);
        uint32_t result = fraction >> shift;
        uint32_t remainder = fraction & ((1u << shift) - 1u);
        uint32_t halfway = 1u << (shift - 1u);
        if (remainder > halfway || (remainder == halfway && (result & 1u))) ++result;
        return (uint16_t)(sign | result);
    }
    uint32_t result = ((uint32_t)e << 10) | (fraction >> 13);
    uint32_t remainder = fraction & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (result & 1u))) ++result;
    return (uint16_t)(sign | result);
}

typedef struct {
    const void *a, *b;
    void *out;
    double amount_a, amount_b;
    int type;
} interpolation_job;

static void interpolation_range(void *opaque, size_t begin, size_t end) {
    interpolation_job *j = opaque;
    if (j->type == 2) {
        const double *a = j->a, *b = j->b;
        double *out = j->out;
        for (size_t i = begin; i < end; ++i) {
            double left = a[i] * j->amount_a;
            double right = b[i] * j->amount_b;
            out[i] = left + right;
        }
    } else if (j->type == 1) {
        const float *a = j->a, *b = j->b;
        float *out = j->out;
        float ca = (float)j->amount_a, cb = (float)j->amount_b;
        for (size_t i = begin; i < end; ++i) {
            float left = a[i] * ca;
            float right = b[i] * cb;
            out[i] = left + right;
        }
    } else {
        const uint16_t *a = j->a, *b = j->b;
        uint16_t *out = j->out;
        float ca = (float)j->amount_a, cb = (float)j->amount_b;
        for (size_t i = begin; i < end; ++i) {
            float left = half_to_float(float_to_half(half_to_float(a[i]) * ca));
            float right = half_to_float(float_to_half(half_to_float(b[i]) * cb));
            out[i] = float_to_half(left + right);
        }
    }
}

CN_EXPORT cn_status cn_tensor_interpolate(const void *a, const void *b, void *out,
        size_t count, int type, double amount_a, double amount_b) {
    if (!a || !b || !out || type < 0 || type > 2) return CN_INVALID_ARGUMENT;
    size_t element_size = type == 2 ? sizeof(double) : type == 1 ? sizeof(float) : sizeof(uint16_t);
    if (count > SIZE_MAX / element_size) return CN_SIZE_OVERFLOW;
    interpolation_job job = {a, b, out, amount_a, amount_b, type};
    return cn_parallel_for(count, 65536, interpolation_range, &job);
}
