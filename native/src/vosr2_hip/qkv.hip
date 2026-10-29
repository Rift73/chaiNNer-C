#include "qkv.h"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cstddef>

// The CUDA kernel's arithmetic in its order: compile without contraction (-ffp-contract=off; nvcc --fmad=false on
// NVIDIA), so a*a+b*b and the rotary products round as the original's do.
namespace {

// One group of 32 lanes per 64-channel head; lane i holds channels i and i + 32. A wave32 GPU (RDNA) runs one group
// per wavefront, a wave64 GPU (CDNA) two: every shuffle names its width, 32, so it stays inside its group.
constexpr int group_lanes = 32;
constexpr int heads = 24;
constexpr int head_width = 64;
constexpr int projection_width = 3 * heads * head_width;
constexpr int block_threads = 256;
constexpr int block_groups = block_threads / group_lanes;

__device__ float rounded(float value) { return __bfloat162float(__float2bfloat16(value)); }

__device__ float inverse(float a, float b) {
    float sum = a * a + b * b;
    for (int s = group_lanes / 2; s; s /= 2) sum += __shfl_xor(sum, s, group_lanes);
    return rsqrtf(sum * (1.0F / head_width) + 1e-6F);
}

// Adjacent-pair rotation: an even lane pairs with the next lane's value negated.
__device__ float rope(float x, float cosine, float sine, int lane) {
    float other = __shfl_xor(x, 1, group_lanes);
    if ((lane & 1) == 0) other = -other;
    return rounded(rounded(x * cosine) + rounded(other * sine));
}

// RMS normalization, weight, BF16 rounding and rotation of one head's 64 values at `source`, written to `target`.
__device__ void normalize(__hip_bfloat16 const* source, __hip_bfloat16 const* weight, __hip_bfloat16 const* cosine,
                          __hip_bfloat16 const* sine, __hip_bfloat16* target, int lane) {
    float first = __bfloat162float(source[lane]);
    float second = __bfloat162float(source[lane + group_lanes]);
    float const scale = inverse(first, second);
    first = rounded(rounded(first * scale) * __bfloat162float(weight[lane]));
    second = rounded(rounded(second * scale) * __bfloat162float(weight[lane + group_lanes]));
    first = rope(first, __bfloat162float(cosine[lane]), __bfloat162float(sine[lane]), lane);
    second = rope(second, __bfloat162float(cosine[lane + group_lanes]), __bfloat162float(sine[lane + group_lanes]),
                  lane);
    target[lane] = __float2bfloat16(first);
    target[lane + group_lanes] = __float2bfloat16(second);
}

__global__ void pack(__hip_bfloat16 const* x, __hip_bfloat16 const* q_weight, __hip_bfloat16 const* k_weight,
                     __hip_bfloat16 const* cosine, __hip_bfloat16 const* sine, __hip_bfloat16* q, __hip_bfloat16* k,
                     __hip_bfloat16* v, int tokens) {
    int const lane = static_cast<int>(threadIdx.x) % group_lanes;
    int const group = static_cast<int>(blockIdx.x) * block_groups + static_cast<int>(threadIdx.x) / group_lanes;
    int const token = group / heads;
    int const head = group % heads;
    // A whole group leaves together: its shuffles never read a lane that left.
    if (token >= tokens) return;
    int const source = token * projection_width + head * head_width;
    int const target = (head * tokens + token) * head_width;
    int const table = token * head_width;
    normalize(x + source, q_weight, cosine + table, sine + table, q + target, lane);
    normalize(x + source + heads * head_width, k_weight, cosine + table, sine + table, k + target, lane);
    int const value = source + 2 * heads * head_width + lane;
    v[target + lane] = x[value];
    v[target + lane + group_lanes] = x[value + group_lanes];
}

}  // namespace

hipError_t vosr2_qkv_pack(void const* x, void const* q_weight, void const* k_weight, void const* cosine,
                          void const* sine, void* packed, int tokens, hipStream_t stream) {
    auto* const q = static_cast<__hip_bfloat16*>(packed);
    std::size_t const plane = static_cast<std::size_t>(heads) * static_cast<std::size_t>(tokens) * head_width;
    int const blocks = (tokens * heads + block_groups - 1) / block_groups;
    hipLaunchKernelGGL(pack, dim3(blocks), dim3(block_threads), 0, stream, static_cast<__hip_bfloat16 const*>(x),
                       static_cast<__hip_bfloat16 const*>(q_weight), static_cast<__hip_bfloat16 const*>(k_weight),
                       static_cast<__hip_bfloat16 const*>(cosine), static_cast<__hip_bfloat16 const*>(sine), q,
                       q + plane, q + 2 * plane, tokens);
    return hipPeekAtLastError();
}
