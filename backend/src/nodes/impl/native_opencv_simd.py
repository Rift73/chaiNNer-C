"""Which SIMD path OpenCV dispatches on this CPU, for the C kernels that mirror it.

Every fact is read from OpenCV's own dispatch at run time, never from CPU features
alone. The mapping comes from OpenCV 5.0.0's sources:

- modules/{core,imgproc}/CMakeLists.txt, ocv_add_dispatched_file: each kernel source
  lists its own dispatch modes. FAMILIES holds their x86 modes, in source order.
- cmake/OpenCVCompilerOptimizations.cmake: a source is compiled for those of its modes
  that the build dispatches. getCPUFeaturesLine() marks the build's modes with "*";
  this wheel has SSE4.1, SSE4.2, AVX, FP16, AVX2 and AVX512-SKX over an SSE3 baseline.
- core/cv_cpu_dispatch.h and cv_cpu_helper.h, CV_CPU_DISPATCH: the compiled modes run
  highest first, each when checkHardwareSupport(CV_CPU_<mode>) holds, else the
  baseline. setUseOptimized(False) and OPENCV_CPU_DISABLE turn that check off.
- core/hal/intrin.hpp: a kernel's wide vectors are 512 bits under AVX512_SKX, 256
  under AVX2 and 128 otherwise. No core or imgproc source forces a width.
"""

from __future__ import annotations

from functools import lru_cache

import cv2

FAMILIES: dict[str, tuple[str, ...]] = {
    # modules/core/CMakeLists.txt
    "mathfuncs_core": ("SSE2", "AVX", "AVX2"),
    "stat": ("SSE4_2", "AVX2"),
    "arithm": ("SSE2", "SSE4_1", "AVX2"),
    "convert": ("SSE2", "AVX2"),
    "convert_scale": ("SSE2", "AVX2"),
    "count_non_zero": ("SSE2", "AVX2"),
    "has_non_zero": ("SSE2", "AVX2"),
    "matmul": ("SSE2", "SSE4_1", "AVX2", "AVX512_SKX"),
    "mean": ("SSE2", "AVX2"),
    "merge": ("SSE2", "AVX2"),
    "minmax": ("SSE2", "SSE4_1", "AVX2"),
    "nan_mask": ("SSE2", "AVX2"),
    "split": ("SSE2", "AVX2"),
    "sum": ("SSE2", "AVX2"),
    "reduce": ("SSE2", "SSSE3", "AVX2"),
    "norm": ("SSE2", "SSE4_1", "AVX", "AVX2"),
    "transpose": ("AVX", "AVX2"),
    # modules/imgproc/CMakeLists.txt
    "accum": ("SSE4_1", "AVX", "AVX2"),
    "bilateral_filter": ("SSE2", "AVX2", "AVX512_SKX", "AVX512_ICL"),
    "box_filter": ("SSE2", "SSE4_1", "AVX2", "AVX512_SKX"),
    "filter": ("SSE2", "SSE4_1", "AVX2"),
    "color_hsv": ("SSE2", "SSE4_1", "AVX2"),
    "color_rgb": ("SSE2", "SSE4_1", "AVX2"),
    "color_yuv": ("SSE2", "SSE4_1", "AVX2"),
    "median_blur": ("SSE2", "SSE4_1", "AVX2", "AVX512_SKX", "AVX512_ICL"),
    "morph": ("SSE2", "SSE4_1", "AVX2"),
    "smooth": ("SSE2", "SSE4_1", "AVX2", "AVX512_ICL"),
    "sumpixels": ("SSE2", "AVX2", "AVX512_SKX"),
    "warp_kernels": ("SSE2", "SSE4_1", "AVX2"),
    "undistort": ("SSE2", "AVX2"),
}

# CV_CPU_<mode> (core/cvdef.h) for checkHardwareSupport.
_CPU_ID = {
    "SSE2": 2,
    "SSSE3": 5,
    "SSE4_1": 6,
    "SSE4_2": 7,
    "FP16": 9,
    "AVX": 10,
    "AVX2": 11,
    "AVX512_SKX": 256,
    "AVX512_ICL": 262,
}

# float32 lanes of a wide vector under each mode (core/hal/intrin.hpp).
_LANES = {
    "BASELINE": 4,
    "SSE2": 4,
    "SSSE3": 4,
    "SSE4_1": 4,
    "SSE4_2": 4,
    "FP16": 4,
    "AVX": 4,
    "AVX2": 8,
    "AVX512_SKX": 16,
    "AVX512_ICL": 16,
}


@lru_cache(maxsize=1)
def dispatched_modes() -> frozenset[str]:
    """The modes the build dispatches: getCPUFeaturesLine()'s "*" entries, named as
    CMake names them (SSE4.1 -> SSE4_1, AVX512-SKX -> AVX512_SKX). The line appends
    "?" to a feature checkHardwareSupport() denies now (core/src/system.cpp), as
    setUseOptimized(False) does to all; stripping it keeps the set, and the cache,
    independent of that state."""
    return frozenset(
        name[1:].rstrip("?").replace(".", "_").replace("-", "_")
        for name in cv2.getCPUFeaturesLine().split()
        if name.startswith("*")
    )


def target(family: str) -> str:
    """The mode CV_CPU_DISPATCH runs the family's kernels on now, or "BASELINE".
    Read on every call: setUseOptimized() changes it."""
    for mode in reversed(FAMILIES[family]):
        if mode in dispatched_modes() and cv2.checkHardwareSupport(_CPU_ID[mode]):
            return mode
    return "BASELINE"


def float32_lanes(mode: str) -> int:
    """float32 lanes of the mode's wide vector: 4, 8 or 16."""
    return _LANES[mode]
