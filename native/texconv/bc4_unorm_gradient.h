// BC4 and BC5 UNORM encoding: the 8 values a block's indices choose among, as texconv.exe computes them. DirectXTex
// (BC4BC5.cpp, FindClosestUNORM) takes them from BC4_UNORM::DecodeFromIndex: red_0 / 255 and red_1 / 255, then
// (red_0 / 255 * (7 - i) + red_1 / 255 * i) / 7 for i = 1..6, or the 5-step ramp with 0 and 1. texconv.exe is built
// with MSVC's /fp:fast, which unrolls the 8 values and folds the constants in float arithmetic, each value in its own
// shape (read from its code). A texel exactly between two values picks the first, so the rounding decides it.
// CMakeLists.txt makes FindClosestUNORM call this function.
#pragma once

#include <cstddef>
#include <cstdint>

inline float texconv_unorm_gradient(uint8_t red_0, uint8_t red_1, size_t index) {
    constexpr float unit = 1.0f / 255.0f;
    constexpr auto c7 = [](float k) { return k * unit * (1.0f / 7.0f); }; // MSVC's (k * (1/255)) * (1/7)
    constexpr auto c5 = [](float k) { return k * unit * (1.0f / 5.0f); };
    const float r0 = float(red_0), r1 = float(red_1), f0 = r0 * unit, f1 = r1 * unit;
    if (index == 0) return f0;
    if (index == 1) return f1;
    if (red_0 > red_1) {
        switch (index) {
        case 2: return r0 * c7(6) + r1 * c7(1);
        case 3: return f1 * (2.0f / 7.0f) + r0 * c7(5);
        case 4: return r0 * c7(4) + r1 * c7(3);
        case 5: return r0 * c7(3) + r1 * c7(4);
        case 6: return r0 * c7(2) + f1 * (5.0f / 7.0f);
        default: return r1 * c7(6) + f0 * (1.0f / 7.0f);
        }
    }
    switch (index) {
    case 2: return r1 * c5(1) + r0 * c5(4);
    case 3: return r0 * c5(3) + r1 * c5(2);
    case 4: return r1 * c5(3) + r0 * c5(2);
    case 5: return r1 * c5(4) + r0 * c5(1);
    case 6: return 0.0f;
    default: return 1.0f;
    }
}
