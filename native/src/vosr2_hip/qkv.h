#pragma once

#include <hip/hip_runtime_api.h>

// VOSR's Q/K/V packing (src/vosr2/qkv.cu, the VosrQkvPack TensorRT plugin's kernel) in HIP. `x` is the DiT
// block's QKV projection (1, tokens, 4608), `q_weight` and `k_weight` the 64 RMS normalization weights, `cosine`
// and `sine` the rotary tables (tokens, 64), all BF16. `packed` receives Q, K and V one after another, each
// (1, 24, tokens, 64) BF16. Returns the launch's hipError_t.
hipError_t vosr2_qkv_pack(void const* x, void const* q_weight, void const* k_weight, void const* cosine,
                          void const* sine, void* packed, int tokens, hipStream_t stream);
