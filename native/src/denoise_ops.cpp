/* Altered C++20 implementation of OpenCV 4.8.0 non-local means and linear
 * BGR <-> Lab byte conversion. Sources: modules/photo/src/denoising.cpp,
 * fast_nlmeans_denoising_invoker{,_commons}.hpp; imgproc/src/color_lab.cpp;
 * core/src/softfloat.cpp. The mathematical integer-distance/weight contract
 * is retained; blocks of row tiles and owned buffers replace cv::Mat/invokers.
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
#include "denoise_shared.h"
#include "isa.h"
#include "parallel.h"
#include "run_kernels.h"
#include <algorithm>
#include <array>
#include <bit>
#include <climits>
#include <cmath>
#include <cstring>
#include <memory>
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
/* The scalar level's row functions: denoise_block_shared.h says what each one does. */
template <size_t C> void column_add(uint32_t *columns, const Byte *row, ptrdiff_t displacement, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        const Byte *a = row+i*C;
        for (size_t c = 0; c < C; ++c) { int d = a[c]-a[displacement+static_cast<ptrdiff_t>(c)]; columns[i] += static_cast<uint32_t>(d*d); }
    }
}
template <size_t C> void column_slide(uint32_t *columns, const Byte *down, const Byte *up, ptrdiff_t displacement, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        const Byte *a = down+i*C, *b = up+i*C;
        for (size_t c = 0; c < C; ++c) {
            int in = a[c]-a[displacement+static_cast<ptrdiff_t>(c)], out = b[c]-b[displacement+static_cast<ptrdiff_t>(c)];
            columns[i] += static_cast<uint32_t>(in*in-out*out);
        }
    }
}
template <size_t C> void accumulate(const nlm_block *b, size_t count, uint32_t *a_sums, const Byte *a_pixels, uint32_t *b_sums, const Byte *b_pixels) {
    size_t plane = b->rows*b->row_lanes;
    for (size_t x = 0; x < count; ++x) {
        uint32_t distance = b->distances[x];
        if (distance >= b->limit) continue;
        uint32_t weight = static_cast<uint32_t>(b->weights[distance >> b->shift]);
        if (a_sums) {
            a_sums[x] += weight;
            for (size_t c = 0; c < C; ++c) a_sums[(c+1)*plane+x] += weight*a_pixels[x*C+c];
        }
        if (b_sums) {
            b_sums[x] += weight;
            for (size_t c = 0; c < C; ++c) b_sums[(c+1)*plane+x] += weight*b_pixels[x*C+c];
        }
    }
}
void nlm_column_add(uint32_t *columns, const Byte *row, ptrdiff_t displacement, size_t count, size_t channels) {
    switch (channels) {
        case 1: column_add<1>(columns,row,displacement,count); break;
        case 2: column_add<2>(columns,row,displacement,count); break;
        case 3: column_add<3>(columns,row,displacement,count); break;
        default: column_add<4>(columns,row,displacement,count);
    }
}
void nlm_column_slide(uint32_t *columns, const Byte *down, const Byte *up, ptrdiff_t displacement, size_t count, size_t channels) {
    switch (channels) {
        case 1: column_slide<1>(columns,down,up,displacement,count); break;
        case 2: column_slide<2>(columns,down,up,displacement,count); break;
        case 3: column_slide<3>(columns,down,up,displacement,count); break;
        default: column_slide<4>(columns,down,up,displacement,count);
    }
}
void nlm_window_pass(uint32_t *out, const uint32_t *in, const uint32_t *columns, size_t count, size_t span, int odd) {
    if (odd) for (size_t i = 0; i < count; ++i) out[i] = in[i]+in[i+span]+columns[i+2*span];
    else for (size_t i = 0; i < count; ++i) out[i] = in[i]+in[i+span];
}
void nlm_accumulate(const nlm_block *b, size_t count, uint32_t *a_sums, const Byte *a_pixels, uint32_t *b_sums, const Byte *b_pixels) {
    switch (b->channels) {
        case 1: accumulate<1>(b,count,a_sums,a_pixels,b_sums,b_pixels); break;
        case 2: accumulate<2>(b,count,a_sums,a_pixels,b_sums,b_pixels); break;
        case 3: accumulate<3>(b,count,a_sums,a_pixels,b_sums,b_pixels); break;
        default: accumulate<4>(b,count,a_sums,a_pixels,b_sums,b_pixels);
    }
}
#include "denoise_block_shared.h"

struct NlmJob {
    nlm_block block;
    Byte *out;
    size_t height, width, block_width, tile_rows;
    size_t tiles_per_chunk, extra_tiles;
    void (*block_sums)(const nlm_block *);
    uint32_t *workspace;
    size_t column_lanes, chunk_lanes;
};
void nlm_tiles(void *opaque, size_t begin, size_t end) noexcept {
    const auto &j = *static_cast<NlmJob *>(opaque);
    nlm_block block = j.block;
    block.columns = j.workspace+begin*j.chunk_lanes;
    block.distances = block.columns+j.column_lanes;
    block.sums = block.distances+j.column_lanes;
    std::fill_n(block.distances, j.column_lanes, 0u);
    size_t channels = block.channels;
    size_t first_tile = begin*j.tiles_per_chunk+std::min(begin,j.extra_tiles);
    size_t last_tile = end*j.tiles_per_chunk+std::min(end,j.extra_tiles);
    for (size_t tile = first_tile; tile < last_tile; ++tile) {
        block.first = tile*j.tile_rows;
        block.rows = std::min(block.first+j.tile_rows, j.height)-block.first;
        size_t plane = block.rows*block.row_lanes;
        for (block.column = 0; block.column < j.width; block.column += j.block_width) {
            block.width = std::min(j.block_width, j.width-block.column);
            std::fill_n(block.sums, plane*(channels+1), 0u);
            j.block_sums(&block);
            for (size_t y = 0; y < block.rows; ++y) {
                const uint32_t *sums = block.sums+y*block.row_lanes+block.search;
                Byte *row = j.out+((block.first+y)*j.width+block.column)*channels;
                for (size_t x = 0; x < block.width; ++x) {
                    uint32_t divisor = sums[x];
                    for (size_t c = 0; c < channels; ++c)
                        row[x*channels+c] = static_cast<Byte>((sums[(c+1)*plane+x]+divisor/2)/divisor);
                }
            }
        }
    }
}
void nlm_scalar(const nlm_block *block) { nlm_block_sums(block); }
void (*block_function(cn_isa_level level, size_t channels))(const nlm_block *) {
#if CN_ISA_UNITS()
    if (channels <= 2 && level >= CN_ISA_AVX512) return cn_nlm_block_avx512;
    if (channels <= 2 && level >= CN_ISA_AVX2) return cn_nlm_block_avx2;
#else
    (void)level; (void)channels;
#endif
    return nlm_scalar;
}
cn_status nlm(const Byte *src, Byte *out, size_t height, size_t width, size_t channels, float strength, int patch, int search, cn_isa_level level) {
    int border = patch+search, patch_size = patch*2+1, search_size = search*2+1;
    size_t pw = width+static_cast<size_t>(border)*2, ph = height+static_cast<size_t>(border)*2;
    if (pw > SIZE_MAX/ph || pw*ph > (static_cast<size_t>(PTRDIFF_MAX)-NLM_PADDED_SLACK)/channels) return CN_SIZE_OVERFLOW;
    size_t padded_bytes = pw*ph*channels;
    auto padded = std::make_unique_for_overwrite<Byte[]>(padded_bytes+NLM_PADDED_SLACK);
    std::memset(padded.get()+padded_bytes, 0, NLM_PADDED_SLACK);
    std::vector<size_t> sources(pw);
    for (size_t x = 0; x < pw; ++x) sources[x] = static_cast<size_t>(reflect(static_cast<int64_t>(x)-border, static_cast<int>(width)));
    for (size_t y = 0; y < ph; ++y) {
        const Byte *row = src+static_cast<size_t>(reflect(static_cast<int64_t>(y)-border, static_cast<int>(height)))*width*channels;
        Byte *to = padded.get()+y*pw*channels;
        for (size_t x = 0; x < static_cast<size_t>(border); ++x) {
            std::memcpy(to+x*channels, row+sources[x]*channels, channels);
            std::memcpy(to+(static_cast<size_t>(border)+width+x)*channels, row+sources[static_cast<size_t>(border)+width+x]*channels, channels);
        }
        std::memcpy(to+static_cast<size_t>(border)*channels, row, width*channels);
    }
    int shift = 0;
    while ((1 << shift) < patch_size*patch_size) ++shift;
    double multiplier = static_cast<double>(1 << shift)/(patch_size*patch_size);
    int fixed = INT_MAX/(search_size*search_size*255);
    size_t max_distance = static_cast<size_t>(65025.0*static_cast<double>(channels)/multiplier+1);
    // The weight falls with the distance, so the table ends at the first 0: every later distance weighs 0 too.
    std::vector<int32_t> weights;
    float denominator = (strength*strength)*static_cast<float>(channels);
    for (size_t d = 0; d < max_distance; ++d) {
        double weight = std::exp(-(static_cast<double>(d)*multiplier)/denominator);
        if (std::isnan(weight)) weight = 1;
        int integer = rounded(fixed*weight);
        if (integer < .001*fixed) break;
        weights.push_back(integer);
    }
    constexpr size_t tile_rows = 16, block_lanes = 512;
    size_t tiles = (height+tile_rows-1)/tile_rows;
    // A block's sums stay in the second cache level on wide images. Without a sideways displacement its columns end on
    // a whole number of the widest vector step, so no row function runs past them for nothing.
    size_t reach = static_cast<size_t>(patch)*2;
    size_t block_width = std::min((block_lanes+reach+31)/32*32-reach, width);
    size_t row_lanes = (block_width+static_cast<size_t>(search)*2+2*NLM_ROW_ROUND-1)/NLM_ROW_ROUND*NLM_ROW_ROUND;
    // At most 128 work items gives each workspace its own cn_parallel_for callback.
    size_t chunks = std::min(tiles,size_t{128});
    // Each chunk's lanes are first touched on the thread that runs it. A displacement's columns reach sideways by up to search.
    size_t column_lanes = block_width+static_cast<size_t>(search)+reach+NLM_COLUMN_SLACK;
    size_t chunk_lanes = column_lanes*2+std::min(tile_rows,height)*row_lanes*(channels+1);
    auto workspace = std::make_unique_for_overwrite<uint32_t[]>(chunks*chunk_lanes);
    nlm_block block{};
    block.padded = padded.get();
    block.stride = pw*channels;
    block.channels = channels;
    block.row_lanes = row_lanes;
    block.patch = patch;
    block.search = search;
    block.border = border;
    block.shift = shift;
    block.limit = static_cast<uint32_t>(weights.size() << shift);
    block.weights = weights.data();
    NlmJob job{block,out,height,width,block_width,tile_rows,tiles/chunks,tiles%chunks,block_function(level,channels),workspace.get(),column_lanes,chunk_lanes};
    return cn_parallel_for(chunks, 1, nlm_tiles, &job);
}
struct SplitJob { const Byte *src; Byte *l, *ab; size_t channels; const LabTables *table; };
void split_range(void *opaque, size_t begin, size_t end) {
    const auto &j = *static_cast<SplitJob *>(opaque);
    for (size_t p = begin; p < end; ++p) {
        Byte lab[3]; to_lab(j.src+p*j.channels,lab,*j.table);
        j.l[p] = lab[0]; j.ab[p*2] = lab[1]; j.ab[p*2+1] = lab[2];
    }
}
struct MergeJob { const Byte *l, *ab, *src; Byte *out; size_t channels; const LabTables *table; };
void merge_range(void *opaque, size_t begin, size_t end) {
    const auto &j = *static_cast<MergeJob *>(opaque);
    for (size_t p = begin; p < end; ++p) {
        Byte lab[3] = {j.l[p],j.ab[p*2],j.ab[p*2+1]};
        from_lab(lab,j.out+p*j.channels,*j.table);
        if (j.channels == 4) j.out[p*4+3] = j.src[p*4+3];
    }
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
    try { return nlm(src,out,height,width,channels,strength,patch,search,cn_isa_current()); }
    catch (const std::bad_alloc &) { return CN_ALLOCATION_FAILED; }
    catch (const std::length_error &) { return CN_SIZE_OVERFLOW; }
}
extern "C" CN_EXPORT cn_status cn_denoise_u8(const uint8_t *src, uint8_t *out,
    size_t height, size_t width, size_t channels, float strength, float color_strength, int patch, int search) {
    cn_status status = image_span(src,out,height,width,channels);
    if (status != CN_OK) return status;
    if ((channels != 1 && channels != 3 && channels != 4) || patch < 1 || patch > 30 || search < 1 || search > 30) return CN_INVALID_ARGUMENT;
    try {
        cn_isa_level level = cn_isa_current();
        if (channels == 1) return nlm(src,out,height,width,channels,strength,patch,search,level);
        size_t count = height*width;
        auto planes = std::make_unique_for_overwrite<Byte[]>(count*6);
        Byte *l = planes.get(), *ab = l+count, *dl = ab+count*2, *dab = dl+count;
        const LabTables &table = lab_tables();
        SplitJob split{src,l,ab,channels,&table};
        status = cn_parallel_for(count,65536,split_range,&split);
        if (status != CN_OK) return status;
        status = nlm(l,dl,height,width,1,strength,patch,search,level);
        if (status != CN_OK) return status;
        status = nlm(ab,dab,height,width,2,color_strength,patch,search,level);
        if (status != CN_OK) return status;
        MergeJob merge{dl,dab,src,out,channels,&table};
        status = cn_parallel_for(count,65536,merge_range,&merge);
        if (status != CN_OK) return status;
        return CN_OK;
    } catch (const std::bad_alloc &) { return CN_ALLOCATION_FAILED; }
      catch (const std::length_error &) { return CN_SIZE_OVERFLOW; }
}
