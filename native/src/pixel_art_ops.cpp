/* Pixel-art kernels adapted from chaiNNer-rs 0.3.10's pinned implementation.
 * C++ value types preserve scalar/RGB/RGBA expression ordering without a foreign
 * image engine. HQ decisions are the complete generated, reviewed tables.
 * See pixel_art_tables.inc and chainner_native.LICENSE.txt for attribution. */
#include "chainner_ext_kernels.h"
#include "parallel.h"
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <new>

namespace {
template<size_t N> struct Yuv {
    float data[N]{};
    bool operator==(const Yuv& other) const noexcept {
        constexpr float thresholds[] = {3.f / 255.f, 7.f / 255.f, 6.f / 255.f, 1.f / 255.f};
        for (size_t i = 0; i < N; ++i)
            if (!(std::fabs(data[i] - other.data[i]) <= thresholds[i])) return false;
        return true;
    }
};

template<size_t N> struct Pixel {
    float data[N]{};
    bool operator==(const Pixel& other) const noexcept {
        for (size_t i = 0; i < N; ++i) if (data[i] != other.data[i]) return false;
        return true;
    }
    Pixel operator+(const Pixel& other) const noexcept {
        Pixel result;
        for (size_t i = 0; i < N; ++i) result.data[i] = data[i] + other.data[i];
        return result;
    }
    Pixel operator*(float value) const noexcept {
        Pixel result;
        for (size_t i = 0; i < N; ++i) result.data[i] = data[i] * value;
        return result;
    }
    Yuv<N> into_yuv() const noexcept {
        Yuv<N> result;
        if constexpr (N == 1) result.data[0] = data[0];
        else {
            /* The pinned binding treats its first stored channel as r here;
             * preserve that existing convention even though node images are BGR. */
            float r = data[0], g = data[1], b = data[2];
            result.data[0] = 0.299f * r + 0.587f * g + 0.114f * b;
            result.data[1] = -0.169f * r - 0.331f * g + 0.5f * b + 0.5f;
            result.data[2] = 0.5f * r - 0.419f * g - 0.081f * b + 0.5f;
            if constexpr (N == 4) result.data[3] = data[3];
        }
        return result;
    }
};

template<class T> T avg2(T a, T b) noexcept { return (a + b) * 0.5f; }
template<class T> T avg4(T a, T b, T c, T d) noexcept { return (a + b + c + d) * 0.25f; }
template<class T> int get_result_1(T a, T b, T c, T d) noexcept {
    int x = 0, y = 0;
    if (a == c) ++x; else if (b == c) ++y;
    if (a == d) ++x; else if (b == d) ++y;
    return (x <= 1 ? 1 : 0) - (y <= 1 ? 1 : 0);
}
template<class T> int get_result_2(T a, T b, T c, T d) noexcept {
    return -get_result_1(a, b, c, d);
}
template<class T> T interp1(T a, T b) noexcept { return (a * 3.f + b) * 0.25f; }
template<class T> T interp2(T a, T b, T c) noexcept { return (a * 2.f + b + c) * 0.25f; }
template<class T> T interp3(T a, T b) noexcept { return (a * 7.f + b) * 0.125f; }
template<class T> T interp4(T a, T b, T c) noexcept { return (a * 2.f + (b + c) * 7.f) * 0.0625f; }
template<class T> T interp5(T a, T b) noexcept { return (a + b) * 0.5f; }
template<class T> T interp6(T a, T b, T c) noexcept { return (a * 5.f + b * 2.f + c) * 0.125f; }
template<class T> T interp7(T a, T b, T c) noexcept { return (a * 6.f + b + c) * 0.125f; }
template<class T> T interp8(T a, T b) noexcept { return (a * 5.f + b * 3.f) * 0.125f; }
template<class T> T interp9(T a, T b, T c) noexcept { return (a * 2.f + (b + c) * 3.f) * 0.125f; }
template<class T> T interp10(T a, T b, T c) noexcept { return (a * 14.f + b + c) * 0.0625f; }

#include "pixel_art_tables.inc"

constexpr size_t scales[] = {2, 3, 4, 2, 3, 2, 2, 2, 2, 3, 4};
struct Job { const float* src; float* dst; size_t height, width; int mode; };

template<size_t N, size_t S>
void store_block(const Job& job, size_t x, size_t y,
                 const std::array<Pixel<N>, S * S>& result) noexcept {
    for (size_t row = 0; row < S; ++row)
        for (size_t column = 0; column < S; ++column) {
            size_t index = ((y * S + row) * (job.width * S) + x * S + column) * N;
            std::memcpy(job.dst + index, result[row * S + column].data, N * sizeof(float));
        }
}

template<size_t N> void rows(void* opaque, size_t begin, size_t end) noexcept {
    const auto& job = *static_cast<Job*>(opaque);
    for (size_t y = begin; y < end; ++y) {
        size_t ys[] = {y ? y - 1 : 0, y,
            y + 1 < job.height ? y + 1 : job.height - 1,
            y + 2 < job.height ? y + 2 : job.height - 1};
        for (size_t x = 0; x < job.width; ++x) {
            size_t xs[] = {x ? x - 1 : 0, x,
                x + 1 < job.width ? x + 1 : job.width - 1,
                x + 2 < job.width ? x + 2 : job.width - 1};
            std::array<Pixel<N>, 16> v;
            for (size_t yy = 0; yy < 4; ++yy)
                for (size_t xx = 0; xx < 4; ++xx)
                    std::memcpy(v[yy * 4 + xx].data,
                        job.src + (ys[yy] * job.width + xs[xx]) * N, N * sizeof(float));
            switch (job.mode) {
            case 0: store_block<N, 2>(job, x, y, adv_mame_2x(v)); break;
            case 1: store_block<N, 3>(job, x, y, adv_mame_3x(v)); break;
            case 3: store_block<N, 2>(job, x, y, eagle_2x(v)); break;
            case 4: store_block<N, 3>(job, x, y, eagle_3x(v)); break;
            case 5: store_block<N, 2>(job, x, y, super_eagle_2x(v)); break;
            case 6: store_block<N, 2>(job, x, y, sai_2x(v)); break;
            case 7: store_block<N, 2>(job, x, y, super_sai_2x(v)); break;
            default: {
                std::array<Pixel<N>, 10> w{};
                for (size_t yy = 0; yy < 3; ++yy)
                    for (size_t xx = 0; xx < 3; ++xx) w[1 + yy * 3 + xx] = v[yy * 4 + xx];
                if (job.mode == 8) store_block<N, 2>(job, x, y, hq2x_pixel(w));
                else if (job.mode == 9) store_block<N, 3>(job, x, y, hq3x_pixel(w));
                else store_block<N, 4>(job, x, y, hq4x_pixel(w));
            }
            }
        }
    }
}

cn_status run(const float* src, float* dst, size_t height, size_t width,
              size_t channels, int mode) noexcept {
    Job job{src, dst, height, width, mode};
    size_t grain = 32768 / (width * channels * scales[mode] * scales[mode]);
    if (!grain) grain = 1;
    if (channels == 1) return cn_parallel_for(height, grain, rows<1>, &job);
    if (channels == 3) return cn_parallel_for(height, grain, rows<3>, &job);
    return cn_parallel_for(height, grain, rows<4>, &job);
}
}

extern "C" CN_EXPORT cn_status cn_pixel_art_f32(const float* src, float* dst,
    size_t height, size_t width, size_t channels, int mode) noexcept {
    if (!src || !dst || !height || !width || (channels != 1 && channels != 3 && channels != 4)
        || mode < 0 || mode > 10 || reinterpret_cast<uintptr_t>(src) % alignof(float)
        || reinterpret_cast<uintptr_t>(dst) % alignof(float)) return CN_INVALID_ARGUMENT;
    size_t scale = scales[mode];
    if (height > SIZE_MAX / width || height * width > SIZE_MAX / channels)
        return CN_SIZE_OVERFLOW;
    size_t count = height * width * channels;
    if (count > SIZE_MAX / sizeof(float) / scale / scale) return CN_SIZE_OVERFLOW;
    size_t input_bytes = count * sizeof(float), output_bytes = input_bytes * scale * scale;
    uintptr_t a = reinterpret_cast<uintptr_t>(src), b = reinterpret_cast<uintptr_t>(dst);
    if (a > UINTPTR_MAX - input_bytes || b > UINTPTR_MAX - output_bytes) return CN_SIZE_OVERFLOW;
    if (a < b + output_bytes && b < a + input_bytes) return CN_INVALID_ARGUMENT;
    if (mode != 2) return run(src, dst, height, width, channels, mode);
    try {
        std::unique_ptr<float[]> intermediate(new float[count * 4]);
        cn_status status = run(src, intermediate.get(), height, width, channels, 0);
        if (status != CN_OK) return status;
        return run(intermediate.get(), dst, height * 2, width * 2, channels, 0);
    } catch (const std::bad_alloc&) { return CN_ALLOCATION_FAILED; }
}
