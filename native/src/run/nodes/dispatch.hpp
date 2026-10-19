/* Process facts that native node entry points pass to their kernels where the Python bridges read them from OpenCV
 * or NumPy at run time: OpenCV's SIMD dispatch per kernel family, NumPy's ufunc loop targets, the tested-OpenCV
 * gate, OpenCV's IPP state, CPU feature line and thread count (cv2.getNumThreads, which Canny's IPP policy reads).
 *
 * Each fact is computed once per process by calling the very Python function the bridge calls (dispatch.cpp, the
 * one file under nodes/ that touches Python: it takes the GIL for that first call, never under a lock), then read
 * without the GIL or a lock. The backend never calls cv2.setUseOptimized, cv2.setNumThreads or the IPP setters (only
 * tests do), so these are constants of the process. A Python error while computing a fact is raised as the bridge would raise it. */
#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>

namespace cn::run::dispatch {

/* module.function(*arguments), function a dotted attribute path, its result an int or a bool. */
int64_t number(std::string_view module, std::string_view function, std::initializer_list<std::string_view> arguments);
/* module.function(*arguments), its result a str. */
std::string text(std::string_view module, std::string_view function, std::initializer_list<std::string_view> arguments);

/* native_opencv_simd.float32_lanes(native_opencv_simd.target(family)): OpenCV's float32 vector lanes (4, 8 or 16)
 * for the kernel family ("filter", "morph", ...). */
size_t opencv_lanes(std::string_view family);
/* native_opencv_simd.target(family): the mode OpenCV dispatches the family on ("AVX2", ..., "BASELINE"). */
std::string opencv_target(std::string_view family);
/* native_versions.cv2_is_tested(gate), gate the bridge's module name (it logs the gate's first fallback). */
bool cv2_tested(std::string_view gate);
/* cv2.ipp.useIPP() and cv2.ipp.useIPP_NotExact(). */
bool cv2_use_ipp();
bool cv2_use_ipp_not_exact();
/* Whether cv2.getCPUFeaturesLine() lists SSE3 (as "SSE3" or "*SSE3"). */
bool cv2_sse3();

}  // namespace cn::run::dispatch
