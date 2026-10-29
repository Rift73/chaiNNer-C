// texconv.exe's WIC steps, measured against texconv.exe 2026.5.8.1 on Windows 11 (exhaustive or random inputs):
// - conversions: DirectXTex's Convert hands a pair of WIC formats to WIC's format converter (UseWICConversion,
//   DirectXTexConvert.cpp); the pairs chaiNNer's intermediates meet are rebuilt here, every other pair keeps
//   DirectXTex's own conversion, which gives WIC's bytes for them;
// - mip levels and the size cap: for formats of at most 8 bits per channel DirectXTex scales each level from the top
//   level with WIC's bitmap scaler (UseWICFiltering, DirectXTexMipmaps.cpp and DirectXTexResize.cpp). Its Fant mode
//   (texconv's default filter, BOX and FANT) is Wolberg's Fant resampling, its nearest neighbour (POINT) takes the
//   last input pixel each output covers and its linear mode (LINEAR) is fixed-point bilinear sampling, all
//   premultiplied on BGRA; rebuilt here exactly on WIC's own 8-bit formats (a Fant tie in 20,000 premultiplied
//   samples aside), Fant and POINT within a few steps on the formats it scales as float. Its cubic mode, and linear
//   through float, are not rebuilt and keep DirectXTex's own filters.
// The steps' DirectXTex code is otherwise unchanged.
// The standard headers come first (texconv.cpp): DirectXTexP.h includes <thread> after the DirectX-Headers adapters,
// whose SAL macros break libstdc++'s C++20 headers.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#include <DirectXTexP.h>

#include "wic.h"

using namespace DirectX;

namespace wic {
namespace {

// The WIC pixel formats of DirectXTex's DXGIToWIC (DirectXTexUtil.cpp); sRGB: the integer formats, which WIC takes
// as sRGB (g_WICFormats).
enum class Pixel { none, rgba32, bgra32, bgr32, bgra5551, bgr565, gray8, alpha8, rgba64, gray16, linear, srgb };

Pixel pixel(DXGI_FORMAT format, bool ignore_rgb_vs_bgr) {
    switch (format) {
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return ignore_rgb_vs_bgr ? Pixel::bgra32 : Pixel::rgba32;
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: return Pixel::bgra32;
    case DXGI_FORMAT_B8G8R8X8_UNORM:
    case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB: return Pixel::bgr32;
    case DXGI_FORMAT_B5G5R5A1_UNORM: return Pixel::bgra5551;
    case DXGI_FORMAT_B5G6R5_UNORM: return Pixel::bgr565;
    case DXGI_FORMAT_R8_UNORM: return Pixel::gray8;
    case DXGI_FORMAT_A8_UNORM: return Pixel::alpha8;
    case DXGI_FORMAT_R16G16B16A16_UNORM: return Pixel::rgba64;
    case DXGI_FORMAT_R16_UNORM:
    case DXGI_FORMAT_D16_UNORM: return Pixel::gray16;
    case DXGI_FORMAT_R10G10B10A2_UNORM:
    case DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM: return Pixel::srgb;
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
    case DXGI_FORMAT_R32G32B32_FLOAT: // WIC2's 96bppRGBFloat
    case DXGI_FORMAT_R32_FLOAT:
    case DXGI_FORMAT_D32_FLOAT:
    case DXGI_FORMAT_R16_FLOAT:
    case DXGI_FORMAT_R1_UNORM: return Pixel::linear;
    default: return Pixel::none;
    }
}

bool srgb(Pixel p) { return p != Pixel::none && p != Pixel::alpha8 && p != Pixel::linear; }

// DirectXTex's UseWICConversion.
bool converts_with_wic(TEX_FILTER_FLAGS filter, DXGI_FORMAT from, DXGI_FORMAT to) {
    const Pixel source = pixel(from, false), target = pixel(to, false);
    if ((filter & TEX_FILTER_FORCE_NON_WIC) || source == Pixel::none || target == Pixel::none) return false;
    if (filter & TEX_FILTER_FORCE_WIC) return true;
    if (filter & (TEX_FILTER_SEPARATE_ALPHA | TEX_FILTER_FLOAT_X2BIAS)) return false;
    const bool half = to == DXGI_FORMAT_R16G16B16A16_FLOAT || to == DXGI_FORMAT_R16G16_FLOAT || to == DXGI_FORMAT_R16_FLOAT;
    switch (from) {
    case DXGI_FORMAT_R32G32_FLOAT:
    case DXGI_FORMAT_R32_FLOAT:
    case DXGI_FORMAT_D32_FLOAT:
        if (half || to == DXGI_FORMAT_A8_UNORM) return false;
        break;
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
    case DXGI_FORMAT_R32G32B32_FLOAT:
        if (half) return false;
        [[fallthrough]];
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
        if (to == DXGI_FORMAT_R16_FLOAT || to == DXGI_FORMAT_R32_FLOAT || to == DXGI_FORMAT_D32_FLOAT ||
            to == DXGI_FORMAT_A8_UNORM)
            return false;
        break;
    case DXGI_FORMAT_R16_FLOAT:
        if (to == DXGI_FORMAT_R32_FLOAT || to == DXGI_FORMAT_D32_FLOAT || to == DXGI_FORMAT_A8_UNORM) return false;
        break;
    case DXGI_FORMAT_A8_UNORM: return false;
    default:
        if (to == DXGI_FORMAT_A8_UNORM) return false;
    }
    uint32_t asked = filter & TEX_FILTER_SRGB;
    if (IsSRGB(from)) asked |= TEX_FILTER_SRGB_IN;
    if (IsSRGB(to)) asked |= TEX_FILTER_SRGB_OUT;
    if (asked == TEX_FILTER_SRGB) asked = 0;
    uint32_t implied = (srgb(source) ? TEX_FILTER_SRGB_IN : 0u) | (srgb(target) ? TEX_FILTER_SRGB_OUT : 0u);
    if (implied == TEX_FILTER_SRGB) implied = 0;
    return asked == implied; // else WIC would change the colour space unasked
}

// sRGB to linear light and back on 0..1.
double to_linear(double v) { return v <= 0.04045 ? v / 12.92 : std::pow((v + 0.055) / 1.055, 2.4); }
double to_srgb(double v) { return v <= 0.0031308 ? v * 12.92 : 1.055 * std::pow(v, 1 / 2.4) - 0.055; }

// WIC's grey of an sRGB colour: BT.709 luminance in linear light, back to sRGB. WIC's own rounding is not rebuilt:
// at 8 bits 97.5 % of all colours give its value and the rest are one step off; at 16 bits up to 3 steps.
uint32_t grey(uint32_t r, uint32_t g, uint32_t b, double top) {
    const double y = 0.2126 * to_linear(r / top) + 0.7152 * to_linear(g / top) + 0.0722 * to_linear(b / top);
    return uint32_t(std::floor(to_srgb(y) * top + 0.5));
}

// One pixel of the WIC pairs rebuilt here, as 8-bit B, G, R, A (or 16-bit grey) in and out.
struct Bgra {
    uint32_t b, g, r, a;
};

Bgra load(Pixel p, const uint8_t* s) {
    switch (p) {
    case Pixel::rgba32: return {s[2], s[1], s[0], s[3]};
    case Pixel::bgra32: return {s[0], s[1], s[2], s[3]};
    case Pixel::bgr32: return {s[0], s[1], s[2], 255};
    case Pixel::gray8: return {s[0], s[0], s[0], 255};
    case Pixel::bgr565:
    case Pixel::bgra5551: {
        // 5 and 6 bits widen by repeating their top bits
        const uint32_t v = uint32_t(s[0]) | uint32_t(s[1]) << 8;
        const auto wide5 = [](uint32_t x) { return x << 3 | x >> 2; };
        if (p == Pixel::bgr565) return {wide5(v & 31), (v >> 5 & 63) << 2 | (v >> 5 & 63) >> 4, wide5(v >> 11), 255};
        return {wide5(v & 31), wide5(v >> 5 & 31), wide5(v >> 10 & 31), (v >> 15) * 255};
    }
    default: return {};
    }
}

void store(Pixel p, const Bgra& c, uint8_t* d) {
    switch (p) {
    case Pixel::rgba32: d[0] = uint8_t(c.r), d[1] = uint8_t(c.g), d[2] = uint8_t(c.b), d[3] = uint8_t(c.a); break;
    case Pixel::bgra32: d[0] = uint8_t(c.b), d[1] = uint8_t(c.g), d[2] = uint8_t(c.r), d[3] = uint8_t(c.a); break;
    case Pixel::bgr32: // the 4th byte keeps the source's alpha
        d[0] = uint8_t(c.b), d[1] = uint8_t(c.g), d[2] = uint8_t(c.r), d[3] = uint8_t(c.a);
        break;
    case Pixel::gray8: d[0] = uint8_t(grey(c.r, c.g, c.b, 255)); break;
    case Pixel::bgr565:
    case Pixel::bgra5551: { // narrowing keeps the top bits
        const uint32_t v = p == Pixel::bgr565 ? (c.r >> 3) << 11 | (c.g >> 2) << 5 | c.b >> 3
                                              : (c.a >> 7) << 15 | (c.r >> 3) << 10 | (c.g >> 3) << 5 | c.b >> 3;
        d[0] = uint8_t(v), d[1] = uint8_t(v >> 8);
        break;
    }
    default: break;
    }
}

bool rebuilt(Pixel from, Pixel to) {
    const auto rgb8 = [](Pixel p) { return p == Pixel::rgba32 || p == Pixel::bgra32 || p == Pixel::bgr32; };
    const auto packed = [](Pixel p) { return p == Pixel::bgr565 || p == Pixel::bgra5551; };
    if (rgb8(from)) return rgb8(to) || packed(to) || to == Pixel::gray8;
    if (from == Pixel::gray8) return packed(to);
    if (packed(from)) return rgb8(to);
    return from == Pixel::rgba64 && to == Pixel::gray16;
}

HRESULT convert_like_wic(const Image* images, size_t count, const TexMetadata& metadata, DXGI_FORMAT format,
    ScratchImage& result) {
    const Pixel from = pixel(metadata.format, false), to = pixel(format, false);
    TexMetadata target = metadata;
    target.format = format;
    HRESULT hr = result.Initialize(target);
    if (FAILED(hr)) return hr;
    if (count != result.GetImageCount()) return E_FAIL;
    const size_t in_bytes = BitsPerPixel(metadata.format) / 8, out_bytes = BitsPerPixel(format) / 8;
    for (size_t i = 0; i < count; ++i) {
        const Image& source = images[i];
        const Image& destination = result.GetImages()[i];
        for (size_t y = 0; y < source.height; ++y) {
            const uint8_t* s = source.pixels + y * source.rowPitch;
            uint8_t* d = destination.pixels + y * destination.rowPitch;
            for (size_t x = 0; x < source.width; ++x, s += in_bytes, d += out_bytes) {
                if (from == Pixel::rgba64) {
                    uint16_t c[4];
                    std::memcpy(c, s, 8);
                    const auto v = uint16_t(grey(c[0], c[1], c[2], 65535));
                    std::memcpy(d, &v, 2);
                } else
                    store(to, load(from, s), d);
            }
        }
    }
    return S_OK;
}

// DirectXTex's UseWICFiltering (both copies) with the conditions of its callers: not premultiplied, and a format
// outside WIC's within the size WIC takes.
bool scales_with_wic(const TexMetadata& metadata, TEX_FILTER_FLAGS filter, size_t width, size_t height) {
    const DXGI_FORMAT format = metadata.format;
    if (metadata.IsPMAlpha() || (filter & TEX_FILTER_FORCE_NON_WIC)) return false;
    if (filter & TEX_FILTER_FORCE_WIC) return true;
    if (IsSRGB(format) || (filter & TEX_FILTER_SRGB)) return false;
    switch (filter & TEX_FILTER_MODE_MASK) {
    case TEX_FILTER_LINEAR:
        if (filter & TEX_FILTER_WRAP) return false;
        break;
    case TEX_FILTER_CUBIC:
        if (filter & (TEX_FILTER_WRAP | TEX_FILTER_MIRROR)) return false;
        break;
    case TEX_FILTER_TRIANGLE: return false;
    default: break;
    }
    if (BitsPerColor(format) > 8) return false;
    if (pixel(format, true) == Pixel::none) {
        constexpr uint64_t limit = UINT32_MAX;
        if (uint64_t(width) * height * 16 > limit || uint64_t(metadata.width) * metadata.height * 16 > limit) return false;
    }
    return true;
}

enum class Mode { fant, point, linear, other };

Mode mode(TEX_FILTER_FLAGS filter) {
    switch (filter & TEX_FILTER_MODE_MASK) {
    case 0:
    case TEX_FILTER_FANT: return Mode::fant;
    case TEX_FILTER_POINT: return Mode::point;
    case TEX_FILTER_LINEAR: return Mode::linear;
    default: return Mode::other;
    }
}

// The scaler modes rebuilt: Fant (the default filter, BOX and FANT) and nearest neighbour (POINT) on every format,
// linear on WIC's own 8-bit formats.
bool mode_rebuilt(DXGI_FORMAT format, TEX_FILTER_FLAGS filter) {
    const Pixel p = pixel(format, true);
    switch (mode(filter)) {
    case Mode::fant:
    case Mode::point: return true;
    case Mode::linear:
        return p == Pixel::bgra32 || p == Pixel::bgr32 || p == Pixel::gray8 || p == Pixel::alpha8 ||
            p == Pixel::bgr565 || p == Pixel::bgra5551;
    default: return false;
    }
}

struct Tap {
    size_t index;
    uint64_t weight;
};
using Taps = std::vector<std::vector<Tap>>;

// Wolberg's Fant resampling of a line of n_in samples to n_out (Digital Image Warping, 7.6: each output pixel
// gathers the input it covers, its pixels read linearly between neighbours), in integers: output x is
// sum(weight * input) / (n_in * n_out). Positions count in units of 1 / n_out of an input pixel.
Taps fant(size_t n_in, size_t n_out) {
    Taps taps(n_out);
    const auto add = [&](size_t x, size_t u, uint64_t weight) {
        if (!weight) return;
        u = std::min(u, n_in - 1); // past the last input: the last input
        auto& list = taps[x];
        if (!list.empty() && list.back().index == u) list.back().weight += weight;
        else list.push_back({u, weight});
    };
    uint64_t inseg = n_out, outseg = n_in;
    for (size_t x = 0, u = 0; x < n_out;) {
        const uint64_t segment = std::min(inseg, outseg);
        add(x, u, inseg * segment);
        add(x, u + 1, (n_out - inseg) * segment);
        if (inseg < outseg) {
            outseg -= inseg;
            inseg = n_out;
            ++u;
        } else {
            inseg -= outseg;
            outseg = n_in;
            ++x;
        }
    }
    return taps;
}

// WIC's nearest neighbour as taps: output x takes input ((x + 1) * n_in - 1) / n_out, the last of those it covers.
Taps nearest(size_t n_in, size_t n_out) {
    Taps taps(n_out);
    for (size_t x = 0; x < n_out; ++x) taps[x].push_back({((x + 1) * n_in - 1) / n_out, uint64_t(n_in) * n_out});
    return taps;
}

Taps line_taps(size_t n_in, size_t n_out, Mode m) { return m == Mode::point ? nearest(n_in, n_out) : fant(n_in, n_out); }

// WIC's linear mode on a line (texconv only shrinks): output x reads the two inputs around (x + 0.5) * n_in / n_out
// - 0.5, in 16.16 fixed point from a truncated step, weighted in 8 bits.
struct Linear {
    size_t first, second;
    uint32_t weight;
};

std::vector<Linear> bilinear(size_t n_in, size_t n_out) {
    std::vector<Linear> taps(n_out);
    const uint64_t step = (uint64_t(n_in) << 16) / n_out;
    for (size_t x = 0; x < n_out; ++x) {
        const uint64_t position = x * step + step / 2 - 0x8000;
        const size_t i = size_t(position >> 16);
        taps[x] = {std::min(i, n_in - 1), std::min(i + 1, n_in - 1), uint32_t(position & 0xFFFF) >> 8};
    }
    return taps;
}

struct Plane {
    const uint8_t* pixels;
    size_t row_pitch, bytes, width, height;
};

// One 8-bit channel of `source` scaled into the same channel of `target` (tw x th), the exact value rounded half up,
// as WIC's Fant gives it (one tie in 19,750 samples of the probes rounds down there, on premultiplied colour). A
// column's sum stays below 255 * h * th < 2^54 (a DDS side is at most 2^32, an output side at most 16384); the 2D sums
// and their denominator reach 2^92 where the 16384 cap scales a larger input, so they are 128-bit.
void scale_channel(const Plane& source, const Taps& tx, const Taps& ty, uint8_t* target, size_t target_pitch,
    size_t channel) {
    using Wide = unsigned __int128;
    const Wide den = Wide(uint64_t(source.width) * tx.size()) * (uint64_t(source.height) * ty.size());
    std::vector<uint64_t> row(source.width);
    for (size_t y = 0; y < ty.size(); ++y) {
        std::fill(row.begin(), row.end(), 0);
        for (const Tap& t : ty[y]) {
            const uint8_t* s = source.pixels + t.index * source.row_pitch + channel;
            for (size_t u = 0; u < source.width; ++u) row[u] += t.weight * s[u * source.bytes];
        }
        uint8_t* d = target + y * target_pitch + channel;
        for (size_t x = 0; x < tx.size(); ++x) {
            Wide sum = 0;
            for (const Tap& t : tx[x]) sum += Wide(t.weight) * row[t.index];
            d[x * source.bytes] = uint8_t((2 * sum + den) / (2 * den));
        }
    }
}

// One 8-bit channel scaled as WIC's linear mode gives it: rows first, each pass truncated to 8 bits.
void linear_channel(const Plane& source, const Image& target, size_t channel) {
    const auto tx = bilinear(source.width, target.width), ty = bilinear(source.height, target.height);
    std::vector<uint8_t> rows(target.width * source.height);
    for (size_t y = 0; y < source.height; ++y) {
        const uint8_t* s = source.pixels + y * source.row_pitch + channel;
        for (size_t x = 0; x < target.width; ++x) {
            const Linear& t = tx[x];
            rows[y * target.width + x] =
                uint8_t((s[t.first * source.bytes] * (256 - t.weight) + s[t.second * source.bytes] * t.weight) >> 8);
        }
    }
    for (size_t y = 0; y < target.height; ++y) {
        const Linear& t = ty[y];
        uint8_t* d = target.pixels + y * target.rowPitch + channel;
        for (size_t x = 0; x < target.width; ++x)
            d[x * source.bytes] = uint8_t((rows[t.first * target.width + x] * (256 - t.weight) +
                                              rows[t.second * target.width + x] * t.weight) >> 8);
    }
}

// An 8-bit image of `bytes` channels scaled as WIC's scaler gives it: plain, or (`premultiply`, 32-bit BGRA) as
// premultiplied BGRA, 8-bit premultiplication and back through a 16-bit reciprocal of alpha.
void scale_image(const Image& source, const Image& target, size_t bytes, bool premultiply, Mode m) {
    const Taps tx = m == Mode::linear ? Taps() : line_taps(source.width, target.width, m);
    const Taps ty = m == Mode::linear ? Taps() : line_taps(source.height, target.height, m);
    const auto channel = [&](const Plane& plane, size_t c) {
        if (m == Mode::linear) linear_channel(plane, target, c);
        else scale_channel(plane, tx, ty, target.pixels, target.rowPitch, c);
    };
    const Plane plain{source.pixels, source.rowPitch, bytes, source.width, source.height};
    if (!premultiply) {
        for (size_t c = 0; c < bytes; ++c) channel(plain, c);
        return;
    }
    channel(plain, 3);
    std::vector<uint8_t> premultiplied(source.width * source.height * 4);
    for (size_t y = 0; y < source.height; ++y) {
        const uint8_t* s = source.pixels + y * source.rowPitch;
        uint8_t* p = premultiplied.data() + y * source.width * 4;
        for (size_t x = 0; x < source.width * 4; x += 4) {
            for (size_t c = 0; c < 3; ++c) p[x + c] = uint8_t((uint32_t(s[x + c]) * s[x + 3] + 127) / 255);
            p[x + 3] = s[x + 3];
        }
    }
    const Plane colour{premultiplied.data(), source.width * 4, 4, source.width, source.height};
    for (size_t c = 0; c < 3; ++c) channel(colour, c);
    for (size_t y = 0; y < target.height; ++y) {
        uint8_t* d = target.pixels + y * target.rowPitch;
        for (size_t x = 0; x < target.width * 4; x += 4) {
            const uint32_t a = d[x + 3], reciprocal = a ? (255u << 16) / a : 0;
            for (size_t c = 0; c < 3; ++c) d[x + c] = uint8_t(std::min<uint32_t>(255, (d[x + c] * reciprocal) >> 16));
        }
    }
}

// Wolberg's loop in float over a line of `n_in` samples `in_stride` floats apart.
void fant_line(const float* in, size_t in_stride, size_t n_in, float* out, size_t out_stride, size_t n_out) {
    const float sizfac = float(n_out) / float(n_in), insfac = 1.0f / sizfac;
    float outseg = insfac, inseg = 1.0f, acc = 0.0f;
    for (size_t x = 0, u = 0; x < n_out;) {
        const float intensity = inseg * in[std::min(u, n_in - 1) * in_stride] +
            (1.0f - inseg) * in[std::min(u + 1, n_in - 1) * in_stride];
        if (inseg < outseg) {
            acc += intensity * inseg;
            outseg -= inseg;
            inseg = 1.0f;
            ++u;
        } else {
            acc += intensity * outseg;
            out[x * out_stride] = acc * sizfac;
            acc = 0.0f;
            inseg -= outseg;
            outseg = insfac;
            ++x;
        }
    }
}

// An R32G32B32A32_FLOAT image scaled as WIC's scaler gives 128bppRGBAFloat, rows first, then columns: premultiplied
// by alpha and back, or with `premultiply` false (DirectXTex's separate colour) with negative colour at 0.
void scale_float(const Image& source, const Image& target, bool premultiply, Mode m) {
    const auto line = [m](const float* in, size_t in_stride, size_t n_in, float* out, size_t out_stride, size_t n_out) {
        if (m != Mode::point) return fant_line(in, in_stride, n_in, out, out_stride, n_out);
        for (size_t x = 0; x < n_out; ++x) out[x * out_stride] = in[((x + 1) * n_in - 1) / n_out * in_stride];
    };
    const size_t w = source.width, h = source.height, tw = target.width, th = target.height;
    std::vector<float> in(w * h * 4), rows(tw * h * 4);
    for (size_t y = 0; y < h; ++y) {
        float* row = in.data() + y * w * 4;
        std::memcpy(row, source.pixels + y * source.rowPitch, w * 16);
        for (size_t x = 0; x < w * 4; x += 4)
            for (size_t c = 0; c < 3; ++c) row[x + c] = premultiply ? row[x + c] * row[x + 3] : std::max(row[x + c], 0.0f);
        for (size_t c = 0; c < 4; ++c) line(row + c, 4, w, rows.data() + y * tw * 4 + c, 4, tw);
    }
    for (size_t x = 0; x < tw; ++x)
        for (size_t c = 0; c < 4; ++c)
            line(rows.data() + x * 4 + c, tw * 4, h, reinterpret_cast<float*>(target.pixels) + x * 4 + c,
                target.rowPitch / 4, th);
    for (size_t y = 0; premultiply && y < th; ++y) {
        auto* row = reinterpret_cast<float*>(target.pixels + y * target.rowPitch);
        for (size_t x = 0; x < tw * 4; x += 4)
            for (size_t c = 0; c < 3; ++c) row[x + c] = row[x + 3] != 0.0f ? row[x + c] / row[x + 3] : 0.0f;
    }
}

// One level (or the capped size) scaled as WIC gives it for the format: WIC's own 8-bit formats directly (32-bit
// BGRA premultiplied unless `separate`, DirectXTex's ResizeSeparateColorAndAlpha; BGRX, grey and alpha as they are);
// 5:6:5 and 5:5:5:1 widened to BGRX and BGRA and narrowed back; every other format through R32G32B32A32_FLOAT.
HRESULT scale_level(const Image& source, const Image& target, bool separate, Mode m) {
    const Pixel p = pixel(source.format, true);
    switch (p) {
    case Pixel::bgra32: scale_image(source, target, 4, !separate, m); return S_OK;
    case Pixel::bgr32: scale_image(source, target, 4, false, m); return S_OK;
    case Pixel::gray8:
    case Pixel::alpha8: scale_image(source, target, 1, false, m); return S_OK;
    case Pixel::bgr565:
    case Pixel::bgra5551: {
        ScratchImage wide, scaled;
        HRESULT hr = wide.Initialize2D(DXGI_FORMAT_B8G8R8A8_UNORM, source.width, source.height, 1, 1);
        if (SUCCEEDED(hr)) hr = scaled.Initialize2D(DXGI_FORMAT_B8G8R8A8_UNORM, target.width, target.height, 1, 1);
        if (FAILED(hr)) return hr;
        const Image& from = *wide.GetImage(0, 0, 0);
        const Image& to = *scaled.GetImage(0, 0, 0);
        for (size_t y = 0; y < source.height; ++y)
            for (size_t x = 0; x < source.width; ++x)
                store(Pixel::bgra32, load(p, source.pixels + y * source.rowPitch + x * 2),
                    from.pixels + y * from.rowPitch + x * 4);
        scale_image(from, to, 4, p == Pixel::bgra5551 && !separate, m);
        for (size_t y = 0; y < target.height; ++y)
            for (size_t x = 0; x < target.width; ++x)
                store(p, load(Pixel::bgra32, to.pixels + y * to.rowPitch + x * 4),
                    target.pixels + y * target.rowPitch + x * 2);
        return S_OK;
    }
    default: {
        ScratchImage wide, scaled;
        HRESULT hr = Internal::ConvertToR32G32B32A32(source, wide);
        if (SUCCEEDED(hr)) hr = scaled.Initialize2D(DXGI_FORMAT_R32G32B32A32_FLOAT, target.width, target.height, 1, 1);
        if (FAILED(hr)) return hr;
        scale_float(*wide.GetImage(0, 0, 0), *scaled.GetImage(0, 0, 0), !separate, m);
        return Internal::ConvertFromR32G32B32A32(*scaled.GetImage(0, 0, 0), target);
    }
    }
}

} // namespace

HRESULT Resize(const Image* images, size_t count, const TexMetadata& metadata, size_t width, size_t height,
    TEX_FILTER_FLAGS filter, ScratchImage& result) {
    if (metadata.dimension == TEX_DIMENSION_TEXTURE3D || !scales_with_wic(metadata, filter, width, height) ||
        !mode_rebuilt(metadata.format, filter))
        return DirectX::Resize(images, count, metadata, width, height, filter, result);
    TexMetadata target = metadata;
    target.width = width;
    target.height = height;
    target.mipLevels = 1;
    HRESULT hr = result.Initialize(target);
    for (size_t item = 0; SUCCEEDED(hr) && item < metadata.arraySize; ++item) {
        const size_t index = metadata.ComputeIndex(0, item, 0);
        hr = index < count
            ? scale_level(images[index], *result.GetImage(0, item, 0), filter & TEX_FILTER_SEPARATE_ALPHA, mode(filter))
            : E_FAIL;
    }
    return hr;
}

HRESULT Convert(const Image* images, size_t count, const TexMetadata& metadata, DXGI_FORMAT format,
    TEX_FILTER_FLAGS filter, float threshold, ScratchImage& result) {
    if (converts_with_wic(filter, metadata.format, format) &&
        rebuilt(pixel(metadata.format, false), pixel(format, false)))
        return convert_like_wic(images, count, metadata, format, result);
    return DirectX::Convert(images, count, metadata, format, filter, threshold, result);
}

HRESULT GenerateMipMaps(const Image* images, size_t count, const TexMetadata& metadata, TEX_FILTER_FLAGS filter,
    size_t levels, ScratchImage& result) {
    if (!scales_with_wic(metadata, filter, std::max<size_t>(1, metadata.width >> 1),
            std::max<size_t>(1, metadata.height >> 1)) ||
        !mode_rebuilt(metadata.format, filter))
        return DirectX::GenerateMipMaps(images, count, metadata, filter, levels, result);
    // DirectXTex's checks: a whole chain for 0 levels, at most a whole chain, more than one level.
    size_t chain = 1;
    for (size_t w = metadata.width, h = metadata.height; w > 1 || h > 1;
         w = std::max<size_t>(1, w >> 1), h = std::max<size_t>(1, h >> 1))
        ++chain;
    if (!levels) levels = chain;
    if (levels > chain || levels <= 1) return E_INVALIDARG;
    TexMetadata target = metadata;
    target.mipLevels = levels;
    HRESULT hr = result.Initialize(target);
    // Every level is the top level scaled (GenerateMipMapsUsingWIC), not the level above.
    for (size_t item = 0; SUCCEEDED(hr) && item < metadata.arraySize; ++item) {
        const size_t index = metadata.ComputeIndex(0, item, 0);
        if (index >= count) return E_FAIL;
        const Image& base = images[index];
        const Image& top = *result.GetImage(0, item, 0);
        for (size_t y = 0; y < base.height; ++y)
            std::memcpy(top.pixels + y * top.rowPitch, base.pixels + y * base.rowPitch,
                std::min(top.rowPitch, base.rowPitch));
        for (size_t level = 1; SUCCEEDED(hr) && level < levels; ++level)
            hr = scale_level(base, *result.GetImage(level, item, 0), filter & TEX_FILTER_SEPARATE_ALPHA, mode(filter));
    }
    return hr;
}

} // namespace wic
