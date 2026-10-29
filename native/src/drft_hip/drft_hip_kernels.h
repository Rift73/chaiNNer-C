// The HIP kernels of legacy DRFT's MIGraphX ops (drft_migx_ops.cpp), ported from the TensorRT
// plugins of src/drft: the same maths and layouts, kernels that assume no wavefront size (wave32 RDNA,
// wave64 CDNA) and BF16 converted bit by bit as CUDA's __float2bfloat16_rn rounds. Each launcher
// checks its arguments, enqueues on `stream` and returns the launch's hipError_t.
#pragma once

#include <hip/hip_runtime_api.h>

#include <cstdint>

namespace drft_hip
{
enum class ElementType : int32_t
{
    kFloat32 = 0,
    kFloat16 = 1,
    kBFloat16 = 2,
};

// ACT attention's rank-factored bias (hnr,hmr->hnm): query factor (heads, queryTokens, rank), key
// factor (heads, keyTokens, rank), output (heads, queryTokens, keyTokens), all FP16 or all BF16,
// accumulated in FP32 (DrftWindowNeuralBias_TRT).
hipError_t launchWindowNeuralBias(void const* queryFactor, void const* keyFactor, void* output,
    ElementType type, int32_t heads, int32_t queryTokens, int32_t keyTokens, int32_t rank,
    hipStream_t stream) noexcept;

// OCAB's dense relative bias (1, heads, q*q, k*k) from its compact table (span*span, heads),
// span = q + k - 1, for q 32 and k 40 or 48, FP32, FP16 or BF16 (DrftOCABRelativeBiasSafe_TRT).
hipError_t launchOCABRelativeBias(void const* table, void* output, ElementType type,
    int32_t heads, int32_t queryWindow, int32_t keyWindow, hipStream_t stream) noexcept;

// One of OCAB's q, k or v window packs from qkv (batch, height, width, 3, channels), token major:
// (window*window, windows, heads, channels / heads), windows = batch * (height / queryWindow) *
// (width / queryWindow), each window centred on a query window and zero outside the image
// (DrftOCABQKVPackTokenMajor_TRT's outputs: q is component 0 at the query window, k and v
// components 1 and 2 at the key window). FP16 or BF16, moved as bits.
hipError_t launchOCABWindowPack(void const* qkv, void* output, int32_t batch, int32_t height,
    int32_t width, int32_t channels, int32_t heads, int32_t queryWindow, int32_t window,
    int32_t component, hipStream_t stream) noexcept;

// The output tail for img_range 1: the body's output (batch, 3, sourceHeight, sourceWidth) in FP16
// or BF16 plus the mean, cropped to (outputHeight, outputWidth), FP32 (DrftFinalAffineCrop_TRT).
hipError_t launchFinalAffineCrop(void const* input, float* output, ElementType type,
    int32_t batch, int32_t channels, int32_t sourceHeight, int32_t sourceWidth,
    int32_t outputHeight, int32_t outputWidth, float const (&mean)[3],
    hipStream_t stream) noexcept;

// OCAB attention with its relative bias from the compact table, online softmax, BF16 in and out:
// q (batch, heads, q*q, 32), k transposed (batch, heads, 32, k*k), v (batch, heads, k*k, 32), table
// (span*span, heads); products accumulated in FP32, probabilities rounded to BF16 before they
// weight v, as DrftOCABAttentionSafe_TRT's WMMA kernel (whose tiles' summation order differs).
hipError_t launchOCABAttention(void const* query, void const* keyTransposed, void const* value,
    void const* relativeBiasTable, void* output, int32_t batch, int32_t heads,
    int32_t queryWindow, int32_t keyWindow, hipStream_t stream) noexcept;
} // namespace drft_hip
