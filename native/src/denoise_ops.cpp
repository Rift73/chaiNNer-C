/* Altered C++20 implementation of OpenCV 4.8.0 non-local means and linear
 * BGR <-> Lab byte conversion. Sources: modules/photo/src/denoising.cpp,
 * fast_nlmeans_denoising_invoker{,_commons}.hpp; imgproc/src/color_lab.cpp;
 * core/src/softfloat.cpp. The mathematical integer-distance/weight contract
 * is retained; bounded row tiles and owned vectors replace cv::Mat/invokers.
 * No OpenCV implementation or framework is called by these kernels.
 * Copyright (C) 2000, Intel Corporation, all rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * * Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * The name of Intel Corporation may not be used to endorse or promote products
 *   derived from this software without specific prior written permission.
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE INTEL CORPORATION OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * The Lab code is Apache-2.0; its positive finite cbrt expression derives from
 * OpenCV SoftFloat (Regents of the University of California, 2011-2017).
 * Full additional OpenCV/SoftFloat notices accompany chainner_native.LICENSE.txt.
 */
#include "parallel.h"
#include <algorithm>
#include <array>
#include <bit>
#include <climits>
#include <cmath>
#include <cstring>
#include <new>
#include <stdexcept>
#include <vector>

namespace {
using Byte = uint8_t;
cn_status image_span(const Byte *src, Byte *out, size_t height, size_t width, size_t channels) {
    if (!src || !out || !height || !width || !channels || channels > 4) return CN_INVALID_ARGUMENT;
    if (width > static_cast<size_t>(INT_MAX - 120) || height > static_cast<size_t>(INT_MAX - 120) ||
        height > SIZE_MAX / width || height * width > SIZE_MAX / channels) return CN_SIZE_OVERFLOW;
    size_t bytes = height * width * channels;
    if (bytes > static_cast<size_t>(PTRDIFF_MAX)) return CN_SIZE_OVERFLOW;
    uintptr_t a = reinterpret_cast<uintptr_t>(src), b = reinterpret_cast<uintptr_t>(out);
    if (a > UINTPTR_MAX - bytes || b > UINTPTR_MAX - bytes) return CN_SIZE_OVERFLOW;
    if (a < b + bytes && b < a + bytes) return CN_INVALID_ARGUMENT;
    return CN_OK;
}
int reflect(int64_t index, int length) {
    if (length == 1) return 0;
    int64_t period = 2 * (static_cast<int64_t>(length) - 1);
    index %= period;
    if (index < 0) index += period;
    return static_cast<int>(index < length ? index : period - index);
}
int rounded(double value) { return static_cast<int>(std::nearbyint(value)); }
Byte saturated(int value) { return static_cast<Byte>(std::clamp(value, 0, 255)); }

/* OpenCV's positive finite f32_cbrt, preserving its explicit truncation. */
float lab_cbrt(float value) {
    uint32_t bits = std::bit_cast<uint32_t>(value);
    int exponent = static_cast<int>((bits >> 23) & 255u) - 127;
    int remainder = exponent % 3;
    remainder -= remainder >= 0 ? 3 : 0;
    exponent = (exponent - remainder) / 3 - 1;
    double x = std::bit_cast<double>((static_cast<uint64_t>(remainder + 1023) << 52) |
        (static_cast<uint64_t>(bits & 0x7fffffu) << 29));
    constexpr uint64_t raw[] = {0x4046a09e6653ba70,0x406808f46c6116e0,0x405dca97439cae14,
        0x402add70d2827500,0x3fc4f15f83f55d2d,0x402d9e20660edb21,0x4062ff15c0285815,
        0x406510d06a8112ce,0x4040fecbc9e2c375};
    double a[9];
    for (size_t i = 0; i < 9; ++i) a[i] = std::bit_cast<double>(raw[i]);
    x = ((((a[0]*x+a[1])*x+a[2])*x+a[3])*x+a[4]) /
        ((((a[5]*x+a[6])*x+a[7])*x+a[8])*x+1.0);
    uint32_t result = (static_cast<uint32_t>(exponent + 127) << 23) |
        static_cast<uint32_t>((std::bit_cast<uint64_t>(x) & UINT64_C(0xfffffffffffff)) >> 29);
    return std::bit_cast<float>(bits ? result : 0u);
}
struct LabTables {
    std::array<int, 3072> cube{};
    std::array<int, 512> yf{};
    LabTables() {
        float scale = 1.0f / (255.0f * 8.0f);
        float threshold = 216.0f / 24389.0f, linear = 841.0f / 108.0f, bias = 16.0f / 116.0f;
        for (int i = 0; i < 3072; ++i) {
            float x = scale * static_cast<float>(i);
            float f = x < threshold ? std::fma(x, linear, bias) : lab_cbrt(x);
            cube[static_cast<size_t>(i)] = rounded(32768.0f * f);
        }
        for (int i = 0; i < 256; ++i) {
            float y, fy;
            if (i <= 20) {
                y = static_cast<float>(i*16384*20*9) / static_cast<float>(17*29*29*29);
                fy = 16384.0f * (16.0f/116.0f + static_cast<float>(i*5)/static_cast<float>(3*17*29));
            } else {
                fy = static_cast<float>(i*100*16384) / static_cast<float>(255*116) + 262144.0f/116.0f;
                y = ((fy*fy)*fy) / 268435456.0f;
            }
            yf[static_cast<size_t>(i)*2] = rounded(y);
            yf[static_cast<size_t>(i)*2+1] = rounded(fy);
        }
    }
};
const LabTables &lab_tables() { static const LabTables value; return value; }
void to_lab(const Byte *src, Byte *dst, const LabTables &table) {
    constexpr int coeff[9] = {778,1541,1777,296,2929,871,3575,448,73};
    int f[3];
    for (size_t i = 0; i < 3; ++i) {
        int index = (8*(src[0]*coeff[i*3] + src[1]*coeff[i*3+1] + src[2]*coeff[i*3+2]) + 2048) >> 12;
        f[i] = table.cube[static_cast<size_t>(index)];
    }
    dst[0] = saturated((296*f[1] - 1336934 + 16384) >> 15);
    dst[1] = saturated((500*(f[0]-f[1]) + 128*32768 + 16384) >> 15);
    dst[2] = saturated((200*(f[1]-f[2]) + 128*32768 + 16384) >> 15);
}
int ab_xyz(int value) {
    return value <= 3390 ? value*108/841 - (16384*16/116)*108/841 : (value*value/16384)*value/16384;
}
void from_lab(const Byte *src, Byte *dst, const LabTables &table) {
    int y = table.yf[src[0]*2u], fy = table.yf[src[0]*2u+1];
    int adiv = ((5*src[1]*53687+128) >> 13) - 128*16384/500;
    int bdiv = ((src[2]*41943+16) >> 9) - 128*16384/200 + 1;
    int x = ab_xyz(fy+adiv), z = ab_xyz(fy-bdiv);
    constexpr int coeff[9] = {217,-836,4715,-3773,7684,185,12615,-6296,-2223};
    for (size_t c = 0; c < 3; ++c) {
        int linear = (coeff[c*3]*x + coeff[c*3+1]*y + coeff[c*3+2]*z + 8192) >> 14;
        linear = std::clamp(linear, 0, 4095);
        dst[c] = static_cast<Byte>((linear*255) >> 12);
    }
}
struct ColorJob { const Byte *src; Byte *out; size_t channels; const LabTables *table; int inverse; };
void color_range(void *opaque, size_t begin, size_t end) {
    const auto &j = *static_cast<ColorJob *>(opaque);
    for (size_t p = begin; p < end; ++p) {
        if (j.inverse) from_lab(j.src+p*j.channels, j.out+p*j.channels, *j.table);
        else to_lab(j.src+p*j.channels, j.out+p*j.channels, *j.table);
        if (j.channels == 4) j.out[p*4+3] = j.src[p*4+3];
    }
}
struct Scratch { std::vector<int> columns; std::vector<uint32_t> sums; };
struct NlmJob {
    const Byte *padded;
    Byte *out;
    size_t height, width, padded_width, channels, tile_rows;
    int patch, search, border, shift;
    const int *weights;
    Scratch *scratch;
};
int distance(const Byte *a, const Byte *b, size_t channels) {
    int sum = 0;
    for (size_t c = 0; c < channels; ++c) { int d = a[c]-b[c]; sum += d*d; }
    return sum;
}
void nlm_tiles(void *opaque, size_t begin, size_t end) {
    const auto &j = *static_cast<NlmJob *>(opaque);
    size_t stride = j.padded_width*j.channels;
    size_t columns = j.width+static_cast<size_t>(j.patch)*2;
    size_t patch_size = static_cast<size_t>(j.patch)*2+1;
    for (size_t tile = begin; tile < end; ++tile) {
        size_t first = tile*j.tile_rows, last = std::min(first+j.tile_rows, j.height);
        auto &scratch = j.scratch[tile];
        for (int dy = -j.search; dy <= j.search; ++dy) {
            for (int dx = -j.search; dx <= j.search; ++dx) {
                ptrdiff_t displacement = static_cast<ptrdiff_t>(dy)*static_cast<ptrdiff_t>(stride) + dx*static_cast<ptrdiff_t>(j.channels);
                std::fill(scratch.columns.begin(), scratch.columns.end(), 0);
                for (size_t py = 0; py < patch_size; ++py) {
                    const Byte *row = j.padded + (first+static_cast<size_t>(j.border-j.patch)+py)*stride + static_cast<size_t>(j.border-j.patch)*j.channels;
                    for (size_t x = 0; x < columns; ++x) {
                        const Byte *a = row+x*j.channels;
                        scratch.columns[x] += distance(a, a+displacement, j.channels);
                    }
                }
                for (size_t y = first; y < last; ++y) {
                    if (y != first) {
                        const Byte *up = j.padded+(y+static_cast<size_t>(j.border-j.patch)-1)*stride+static_cast<size_t>(j.border-j.patch)*j.channels;
                        const Byte *down = up+patch_size*stride;
                        for (size_t x = 0; x < columns; ++x) {
                            const Byte *a = up+x*j.channels, *b = down+x*j.channels;
                            scratch.columns[x] += distance(b,b+displacement,j.channels)-distance(a,a+displacement,j.channels);
                        }
                    }
                    int patch_distance = 0;
                    for (size_t x = 0; x < patch_size; ++x) patch_distance += scratch.columns[x];
                    const Byte *candidate = j.padded+(y+static_cast<size_t>(j.border))*stride+static_cast<size_t>(j.border)*j.channels+displacement;
                    for (size_t x = 0; x < j.width; ++x) {
                        uint32_t weight = static_cast<uint32_t>(j.weights[patch_distance >> j.shift]);
                        uint32_t *sum = scratch.sums.data()+((y-first)*j.width+x)*(j.channels+1);
                        for (size_t c = 0; c < j.channels; ++c) sum[c] += weight*candidate[x*j.channels+c];
                        sum[j.channels] += weight;
                        if (x+1 < j.width) patch_distance += scratch.columns[x+patch_size]-scratch.columns[x];
                    }
                }
            }
        }
        for (size_t p = 0; p < (last-first)*j.width; ++p) {
            const uint32_t *sum = scratch.sums.data()+p*(j.channels+1);
            uint32_t divisor = sum[j.channels];
            for (size_t c = 0; c < j.channels; ++c)
                j.out[(first*j.width+p)*j.channels+c] = static_cast<Byte>((sum[c]+divisor/2)/divisor);
        }
    }
}
cn_status nlm(const Byte *src, Byte *out, size_t height, size_t width, size_t channels, float strength, int patch, int search) {
    int border = patch+search, patch_size = patch*2+1, search_size = search*2+1;
    size_t pw = width+static_cast<size_t>(border)*2, ph = height+static_cast<size_t>(border)*2;
    if (pw > SIZE_MAX/ph || pw*ph > static_cast<size_t>(PTRDIFF_MAX)/channels) return CN_SIZE_OVERFLOW;
    std::vector<Byte> padded(pw*ph*channels);
    for (size_t y = 0; y < ph; ++y) {
        size_t sy = static_cast<size_t>(reflect(static_cast<int64_t>(y)-border, static_cast<int>(height)));
        for (size_t x = 0; x < pw; ++x) {
            size_t sx = static_cast<size_t>(reflect(static_cast<int64_t>(x)-border, static_cast<int>(width)));
            std::memcpy(padded.data()+(y*pw+x)*channels, src+(sy*width+sx)*channels, channels);
        }
    }
    int shift = 0;
    while ((1 << shift) < patch_size*patch_size) ++shift;
    double multiplier = static_cast<double>(1 << shift)/(patch_size*patch_size);
    int fixed = INT_MAX/(search_size*search_size*255);
    size_t max_distance = static_cast<size_t>(65025.0*static_cast<double>(channels)/multiplier+1);
    std::vector<int> weights(max_distance);
    float denominator = (strength*strength)*static_cast<float>(channels);
    for (size_t d = 0; d < max_distance; ++d) {
        double weight = std::exp(-(static_cast<double>(d)*multiplier)/denominator);
        if (std::isnan(weight)) weight = 1;
        int integer = rounded(fixed*weight);
        weights[d] = integer < .001*fixed ? 0 : integer;
    }
    constexpr size_t tile_rows = 16;
    size_t tiles = (height+tile_rows-1)/tile_rows;
    if (width > SIZE_MAX/(tile_rows*(channels+1)*sizeof(uint32_t))) return CN_SIZE_OVERFLOW;
    std::vector<Scratch> scratch(tiles);
    for (size_t i = 0; i < tiles; ++i) {
        scratch[i].columns.resize(width+static_cast<size_t>(patch)*2);
        scratch[i].sums.resize(std::min(tile_rows,height-i*tile_rows)*width*(channels+1));
    }
    NlmJob job{padded.data(),out,height,width,pw,channels,tile_rows,patch,search,border,shift,weights.data(),scratch.data()};
    return cn_parallel_for(tiles, 1, nlm_tiles, &job);
}
}

extern "C" CN_EXPORT cn_status cn_denoise_lab_u8(const uint8_t *src, uint8_t *out,
    size_t height, size_t width, size_t channels, int inverse) {
    if ((channels != 3 && channels != 4) || (inverse != 0 && inverse != 1)) return CN_INVALID_ARGUMENT;
    cn_status status = image_span(src,out,height,width,channels);
    if (status != CN_OK) return status;
    ColorJob job{src,out,channels,&lab_tables(),inverse};
    return cn_parallel_for(height*width,65536,color_range,&job);
}
extern "C" CN_EXPORT cn_status cn_denoise_nlm_u8(const uint8_t *src, uint8_t *out,
    size_t height, size_t width, size_t channels, float strength, int patch, int search) {
    cn_status status = image_span(src,out,height,width,channels);
    if (status != CN_OK) return status;
    if (patch < 1 || patch > 30 || search < 1 || search > 30) return CN_INVALID_ARGUMENT;
    try { return nlm(src,out,height,width,channels,strength,patch,search); }
    catch (const std::bad_alloc &) { return CN_ALLOCATION_FAILED; }
    catch (const std::length_error &) { return CN_SIZE_OVERFLOW; }
}
extern "C" CN_EXPORT cn_status cn_denoise_u8(const uint8_t *src, uint8_t *out,
    size_t height, size_t width, size_t channels, float strength, float color_strength, int patch, int search) {
    cn_status status = image_span(src,out,height,width,channels);
    if (status != CN_OK) return status;
    if ((channels != 1 && channels != 3 && channels != 4) || patch < 1 || patch > 30 || search < 1 || search > 30) return CN_INVALID_ARGUMENT;
    try {
        if (channels == 1) return nlm(src,out,height,width,channels,strength,patch,search);
        size_t count = height*width;
        std::vector<Byte> l(count), ab(count*2), dl(count), dab(count*2);
        const LabTables &table = lab_tables();
        for (size_t p = 0; p < count; ++p) {
            Byte lab[3]; to_lab(src+p*channels,lab,table);
            l[p] = lab[0]; ab[p*2] = lab[1]; ab[p*2+1] = lab[2];
        }
        status = nlm(l.data(),dl.data(),height,width,1,strength,patch,search);
        if (status != CN_OK) return status;
        status = nlm(ab.data(),dab.data(),height,width,2,color_strength,patch,search);
        if (status != CN_OK) return status;
        for (size_t p = 0; p < count; ++p) {
            Byte lab[3] = {dl[p],dab[p*2],dab[p*2+1]};
            from_lab(lab,out+p*channels,table);
            if (channels == 4) out[p*4+3] = src[p*4+3];
        }
        return CN_OK;
    } catch (const std::bad_alloc &) { return CN_ALLOCATION_FAILED; }
      catch (const std::length_error &) { return CN_SIZE_OVERFLOW; }
}
