"""native_opencv_simd: OpenCV's dispatch, whatever setUseOptimized was at first use."""

import cv2

from nodes.impl import native_opencv_simd as simd

CV_CPU_AVX2 = 11
CV_CPU_AVX512_SKX = 256


def test_dispatched_modes_first_read_with_optimization_off():
    was = cv2.useOptimized()
    try:
        simd.dispatched_modes.cache_clear()
        cv2.setUseOptimized(False)
        first = simd.dispatched_modes()
        assert {simd.target(family) for family in simd.FAMILIES} == {"BASELINE"}
        cv2.setUseOptimized(True)
        simd.dispatched_modes.cache_clear()
        assert simd.dispatched_modes() == first
        assert all(simd.float32_lanes(mode) in (4, 8, 16) for mode in first)
        if "AVX2" in first and cv2.checkHardwareSupport(CV_CPU_AVX2):
            assert simd.target("filter") == "AVX2"
        if "AVX512_SKX" in first and cv2.checkHardwareSupport(CV_CPU_AVX512_SKX):
            assert simd.target("median_blur") == "AVX512_SKX"
    finally:
        cv2.setUseOptimized(was)
