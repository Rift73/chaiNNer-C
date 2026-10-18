// Per-sample conversions of the TensorRT runner, shared by the GPU kernels (kernels.cu) and
// their CPU references (reference.cpp), so both compute the same IEEE float operations:
// nvcc builds with --fmad=false and IEEE division, clang-cl with /fp:strict, so neither
// contracts or reorders. Each function mirrors the Python it replaces, named beside it.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#ifdef __CUDACC__
#define CN_TRT_HD __host__ __device__
#else
#define CN_TRT_HD
#endif

namespace cn_trt {

// How a plane's samples are stored: integer codes (8-bit as uint8, 9-16 bits as uint16)
// or float16/float32 plane values.
enum class Sample : int32_t { U8, U16, F16, F32 };
// An engine tensor's element type (TensorRT's kFLOAT, kHALF, kBF16).
enum class Element : int32_t { F32, F16, BF16 };

// Integer codes as plane values: value = (code - offset) / scale, colour_info's _scale for
// a luma or RGB plane (the runner never sees chroma: engines take RGB or gray).
struct Codes {
    float offset;
    float scale;
    float top;  // the largest code, 2^bits - 1
};

inline Codes codes_for(int bits, bool full_range) {
    float const top = static_cast<float>((1u << bits) - 1u);
    if (full_range) return {0.0f, top, top};
    float const step = bits >= 8 ? static_cast<float>(1u << (bits - 8))
                                 : 1.0f / static_cast<float>(1u << (8 - bits));
    return {16.0f * step, 219.0f * step, top};
}

CN_TRT_HD inline uint32_t float_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof bits);
    return bits;
}

CN_TRT_HD inline float bits_float(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof value);
    return value;
}

// colour_info.code_to_float: (float32(code) - offset) / scale.
CN_TRT_HD inline float decode_code(uint32_t code, Codes c) {
    return (static_cast<float>(code) - c.offset) / c.scale;
}

// colour_info.float_to_code: rint(value * scale + offset), clipped to 0..top. NaN gives 0.
CN_TRT_HD inline uint32_t encode_code(float value, Codes c) {
    float code = rintf(value * c.scale + c.offset);
    if (!(code >= 0.0f)) code = 0.0f;
    if (code > c.top) code = c.top;
    return static_cast<uint32_t>(code);
}

// IEEE binary16, round to nearest even (NumPy's float32 -> float16 cast); a NaN becomes the
// quiet NaN with its sign.
CN_TRT_HD inline uint16_t float_to_half(float value) {
    uint32_t const f = float_bits(value);
    uint32_t const sign = (f >> 16) & 0x8000u;
    uint32_t const a = f & 0x7FFFFFFFu;
    if (a > 0x7F800000u) return static_cast<uint16_t>(sign | 0x7E00u);
    if (a >= 0x477FF000u) return static_cast<uint16_t>(sign | 0x7C00u);  // 65520 and up: Inf
    if (a < 0x38800000u) {                                                 // below 2^-14: subnormal
        int const shift = 126 - static_cast<int>(a >> 23);
        if (shift > 24) return static_cast<uint16_t>(sign);
        uint32_t const mantissa = (a & 0x7FFFFFu) | 0x800000u;
        uint32_t half = mantissa >> shift;
        uint32_t const rest = mantissa & ((1u << shift) - 1u);
        uint32_t const midpoint = 1u << (shift - 1);
        if (rest > midpoint || (rest == midpoint && (half & 1u))) ++half;
        return static_cast<uint16_t>(sign | half);
    }
    uint32_t const rebased = a - 0x38000000u;  // exponent bias 127 -> 15
    return static_cast<uint16_t>(sign | ((rebased + 0x0FFFu + ((rebased >> 13) & 1u)) >> 13));
}

CN_TRT_HD inline float half_to_float(uint16_t half) {
    uint32_t const sign = static_cast<uint32_t>(half & 0x8000u) << 16;
    uint32_t const exponent = (half >> 10) & 0x1Fu;
    uint32_t const mantissa = half & 0x3FFu;
    if (exponent == 0x1Fu) return bits_float(sign | 0x7F800000u | (mantissa << 13));
    if (exponent == 0) {
        // m * 2^-24, exact in float32
        float const magnitude = static_cast<float>(mantissa) * 5.9604644775390625e-8f;
        return bits_float(sign | float_bits(magnitude));
    }
    return bits_float(sign | ((exponent + 112u) << 23) | (mantissa << 13));
}

// tensorrt/bfloat16.py float32_to_bf16_bits: round to nearest even; a NaN becomes 0x7FC0
// with its sign.
CN_TRT_HD inline uint16_t float_to_bf16(float value) {
    uint32_t const bits = float_bits(value);
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u) return (bits >> 31) ? 0xFFC0u : 0x7FC0u;
    return static_cast<uint16_t>((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
}

CN_TRT_HD inline float bf16_to_float(uint16_t bits) {
    return bits_float(static_cast<uint32_t>(bits) << 16);
}

// np.clip(value, 0, 1): NaN stays NaN.
CN_TRT_HD inline float clip01(float value) {
    return value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
}

// cv2.COLOR_BGR2GRAY's BT.601 weights in OpenCV's scalar order (as_target_channels' narrowing).
CN_TRT_HD inline float luma(float r, float g, float b) {
    return (b * 0.114f + g * 0.587f) + r * 0.299f;
}

// tiling._pad_tile's np.pad source index for padded position i of n real pixels: "reflect"
// (periodic for pads longer than n - 1, as NumPy repeats it) or "edge".
CN_TRT_HD inline int pad_index(int i, int n, bool edge) {
    if (i < n) return i;
    if (edge || n == 1) return n - 1;
    int const period = 2 * (n - 1);
    int const m = i % period;
    return m < n ? m : period - m;
}

CN_TRT_HD inline float load_sample(void const* data, Sample sample, int64_t index, Codes c) {
    switch (sample) {
    case Sample::U8: return decode_code(static_cast<uint8_t const*>(data)[index], c);
    case Sample::U16: return decode_code(static_cast<uint16_t const*>(data)[index], c);
    case Sample::F16: return half_to_float(static_cast<uint16_t const*>(data)[index]);
    default: return static_cast<float const*>(data)[index];
    }
}

CN_TRT_HD inline void store_sample(void* data, Sample sample, int64_t index, float value, Codes c) {
    switch (sample) {
    case Sample::U8: static_cast<uint8_t*>(data)[index] = static_cast<uint8_t>(encode_code(value, c)); break;
    case Sample::U16: static_cast<uint16_t*>(data)[index] = static_cast<uint16_t>(encode_code(value, c)); break;
    case Sample::F16: static_cast<uint16_t*>(data)[index] = float_to_half(value); break;
    default: static_cast<float*>(data)[index] = value; break;
    }
}

CN_TRT_HD inline float load_element(void const* data, Element element, int64_t index) {
    switch (element) {
    case Element::F16: return half_to_float(static_cast<uint16_t const*>(data)[index]);
    case Element::BF16: return bf16_to_float(static_cast<uint16_t const*>(data)[index]);
    default: return static_cast<float const*>(data)[index];
    }
}

CN_TRT_HD inline void store_element(void* data, Element element, int64_t index, float value) {
    switch (element) {
    case Element::F16: static_cast<uint16_t*>(data)[index] = float_to_half(value); break;
    case Element::BF16: static_cast<uint16_t*>(data)[index] = float_to_bf16(value); break;
    default: static_cast<float*>(data)[index] = value; break;
    }
}

inline int sample_bytes(Sample sample) {
    return sample == Sample::U8 ? 1 : (sample == Sample::F32 ? 4 : 2);
}

inline int element_bytes(Element element) { return element == Element::F32 ? 4 : 2; }

// Where a frame's planes lie in a device block, in samples: plane p's pixel (y, x) is at
// offset[p] + y * pitch + x * step. Packed planar (offset p * width * height, pitch width,
// step 1) or packed interleaved (offset of the channel, pitch width * planes, step planes).
struct Layout {
    int64_t offset[3];
    int64_t pitch;
    int step;
};

// The frame's planes on the device.
struct Source {
    void const* data;
    Sample sample;
    int planes;  // 1 (gray) or 3 (R, G, B)
    int width;
    int height;
    Layout layout;
    Codes codes;  // integer samples only
};

// One engine input tile: `real_w` x `real_h` frame pixels from (x0, y0), padded (np.pad
// reflect, or edge) to the engine input `width` x `height`.
struct TileIn {
    int x0;
    int y0;
    int real_w;
    int real_h;
    int width;
    int height;
    bool edge;
};

// Channel-planar float data read by accumulate and store: an engine output tensor (rows of
// `pitch` elements, `plane` elements per channel) or the float32 accumulator.
struct Planar {
    void const* data;
    Element element;
    int channels;
    int pitch;
    int64_t plane;
};

// The result planes on the device.
struct Result {
    void* data;
    Sample sample;
    int planes;
    int width;
    int height;
    Layout layout;
    Codes codes;
};

CN_TRT_HD inline int64_t layout_index(Layout const& l, int p, int64_t y, int64_t x) {
    return l.offset[p] + y * l.pitch + x * l.step;
}

// Pixel (y, x)'s `channels` engine values (convenient_upscale's channel rules: planes as
// they are, gray replicated, or RGB narrowed to gray), clipped first when asked.
CN_TRT_HD inline void tensor_pixel(Source const& s, int64_t y, int64_t x, int channels, bool clip, float* out) {
    float v[3];
    for (int p = 0; p < s.planes; ++p) {
        v[p] = load_sample(s.data, s.sample, layout_index(s.layout, p, y, x), s.codes);
        if (clip) v[p] = clip01(v[p]);
    }
    if (s.planes == channels) {
        for (int c = 0; c < channels; ++c) out[c] = v[c];
    } else if (s.planes == 1) {
        for (int c = 0; c < channels; ++c) out[c] = v[0];
    } else {
        out[0] = luma(v[0], v[1], v[2]);
    }
}

// Result pixel (y, x) from `src` (rows of src.pitch): its channels as `dst.planes` planes,
// clipped first when asked, as convenient_upscale clips the model's output before narrowing.
CN_TRT_HD inline void result_pixel(Planar const& src, Result const& dst, int64_t y, int64_t x, bool clip) {
    int64_t const at = y * src.pitch + x;
    float v[3];
    for (int c = 0; c < src.channels; ++c) {
        v[c] = load_element(src.data, src.element, c * src.plane + at);
        if (clip) v[c] = clip01(v[c]);
    }
    if (dst.planes == src.channels || src.channels == 1) {
        for (int p = 0; p < dst.planes; ++p)
            store_sample(dst.data, dst.sample, layout_index(dst.layout, p, y, x), v[dst.planes == src.channels ? p : 0], dst.codes);
    } else {
        store_sample(dst.data, dst.sample, layout_index(dst.layout, 0, y, x), luma(v[0], v[1], v[2]), dst.codes);
    }
}

}  // namespace cn_trt
