/* Compiles HIP kernel sources with nvcc for an NVIDIA GPU, so a kernel ported to HIP can be checked against its
 * CUDA original on NVIDIA hardware. HIP's own NVIDIA headers (hipother's nvidia_detail) map the runtime API, __half and
 * __hip_bfloat16; this header adds what they lack under CUDA 13. Force-include it on the NVIDIA side only:
 *   nvcc -x cu -D__HIP_PLATFORM_NVIDIA__ -I<hipother>/hipnv/include -I<rocm>/include -include hip_nv_shim.h ...
 * The AMD build never sees it.
 * SPDX-License-Identifier: GPL-3.0-only
 */
#ifndef CHAINNER_HIP_NV_SHIM_H
#define CHAINNER_HIP_NV_SHIM_H
#if !defined(__HIP_PLATFORM_NVIDIA__) || !defined(__CUDACC__)
#error "hip_nv_shim.h is for nvcc builds of HIP sources (__HIP_PLATFORM_NVIDIA__)"
#endif

#include <hip/hip_runtime.h>
#include <cuda_fp8.h>

/* HIP's mask-free warp functions act on the whole wavefront; CUDA 13 rejects them from compute_70 on and has only the
 * *_sync forms. The full mask is the same contract: every lane of the warp reaches the call. CUDA still declares
 * __ballot, __any and __all (unsigned int __ballot), so those three are macros; HIP's __ballot returns 64 bits
 * (wave64 on CDNA). */
#define CHAINNER_HIP_NV_FULL_WARP 0xffffffffu

template <typename T>
__device__ __forceinline__ T __shfl(T var, int src_lane, int width = warpSize) {
    return __shfl_sync(CHAINNER_HIP_NV_FULL_WARP, var, src_lane, width);
}
template <typename T>
__device__ __forceinline__ T __shfl_up(T var, unsigned int delta, int width = warpSize) {
    return __shfl_up_sync(CHAINNER_HIP_NV_FULL_WARP, var, delta, width);
}
template <typename T>
__device__ __forceinline__ T __shfl_down(T var, unsigned int delta, int width = warpSize) {
    return __shfl_down_sync(CHAINNER_HIP_NV_FULL_WARP, var, delta, width);
}
template <typename T>
__device__ __forceinline__ T __shfl_xor(T var, int lane_mask, int width = warpSize) {
    return __shfl_xor_sync(CHAINNER_HIP_NV_FULL_WARP, var, lane_mask, width);
}
#define __ballot(predicate) ((unsigned long long)__ballot_sync(CHAINNER_HIP_NV_FULL_WARP, (predicate)))
#define __any(predicate) __any_sync(CHAINNER_HIP_NV_FULL_WARP, (predicate))
#define __all(predicate) __all_sync(CHAINNER_HIP_NV_FULL_WARP, (predicate))

/* hip_fp8.h is empty on NVIDIA. HIP's OCP FP8 types (device code on gfx950, gfx12xx and the targets without FP8
 * hardware) mirror CUDA's; the FNUZ types of gfx942 have no NVIDIA counterpart, so kernels using them are checked on
 * AMD only. */
typedef __nv_fp8_e4m3 __hip_fp8_e4m3;
typedef __nv_fp8_e5m2 __hip_fp8_e5m2;
typedef __nv_fp8x2_e4m3 __hip_fp8x2_e4m3;
typedef __nv_fp8x2_e5m2 __hip_fp8x2_e5m2;
typedef __nv_fp8x4_e4m3 __hip_fp8x4_e4m3;
typedef __nv_fp8x4_e5m2 __hip_fp8x4_e5m2;
typedef __nv_fp8_storage_t __hip_fp8_storage_t;
typedef __nv_fp8x2_storage_t __hip_fp8x2_storage_t;
typedef __nv_saturation_t __hip_saturation_t;
typedef __nv_fp8_interpretation_t __hip_fp8_interpretation_t;
#define __HIP_NOSAT __NV_NOSAT
#define __HIP_SATFINITE __NV_SATFINITE
#define __HIP_E4M3 __NV_E4M3
#define __HIP_E5M2 __NV_E5M2
#define __hip_cvt_float_to_fp8 __nv_cvt_float_to_fp8
#define __hip_cvt_float2_to_fp8x2 __nv_cvt_float2_to_fp8x2
#define __hip_cvt_halfraw_to_fp8 __nv_cvt_halfraw_to_fp8
#define __hip_cvt_fp8_to_halfraw __nv_cvt_fp8_to_halfraw
#define __hip_cvt_fp8x2_to_halfraw2 __nv_cvt_fp8x2_to_halfraw2

#endif
