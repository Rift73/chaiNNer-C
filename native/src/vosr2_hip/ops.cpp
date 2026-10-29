// vosr2_migx_ops: VOSR's custom MIGraphX op (migraphx_runner/custom_op.h), the counterpart of the VosrQkvPack
// TensorRT plugin (src/vosr2/qkv_plugin.cpp). chainner_vosr2_qkv takes the plugin's five BF16 inputs and writes
// its three outputs as one tensor (custom ops have one output): Q, K and V, (3, 1, 24, tokens, 64).
#include "custom_op.h"
#include "qkv.h"

#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <cstdio>

namespace {

constexpr int64_t heads = 24;
constexpr int64_t head_width = 64;
constexpr int64_t projection_width = 3 * heads * head_width;
// The plugin's limit: 1152 tokens, a 64 x 72 latent window.
constexpr int64_t max_tokens = 1152;

int fail(char* error, size_t error_size, char const* message) {
    std::snprintf(error, error_size, "chainner_vosr2_qkv: %s", message);
    return 1;
}

bool bf16(chainner_migx_shape const& shape, int32_t rank) {
    return shape.type == CHAINNER_MIGX_BF16 && shape.rank == rank;
}

bool table(chainner_migx_shape const& shape, int64_t tokens) {
    return bf16(shape, 2) && shape.dims[0] == tokens && shape.dims[1] == head_width;
}

int compute_shape(chainner_migx_shape const* inputs, int32_t input_count, chainner_migx_attribute const*,
                  int32_t attribute_count, chainner_migx_shape* output, size_t*, char* error, size_t error_size) {
    if (input_count != 5 || attribute_count != 0)
        return fail(error, error_size, "expects 5 inputs and no attributes");
    chainner_migx_shape const& x = inputs[0];
    if (!bf16(x, 3) || x.dims[0] != 1 || x.dims[1] < 1 || x.dims[1] > max_tokens || x.dims[2] != projection_width)
        return fail(error, error_size, "x must be BF16 (1, 1 to 1152, 4608)");
    if (!bf16(inputs[1], 1) || inputs[1].dims[0] != head_width || !bf16(inputs[2], 1) ||
        inputs[2].dims[0] != head_width)
        return fail(error, error_size, "the normalization weights must be BF16 (64)");
    if (!table(inputs[3], x.dims[1]) || !table(inputs[4], x.dims[1]))
        return fail(error, error_size, "the rotary tables must be BF16 (tokens, 64)");
    output->type = CHAINNER_MIGX_BF16;
    output->rank = 5;
    int64_t const dims[] = {3, 1, heads, x.dims[1], head_width};
    for (int32_t i = 0; i < output->rank; ++i) output->dims[i] = dims[i];
    return 0;
}

int compute(void* stream, chainner_migx_tensor const* inputs, int32_t, chainner_migx_attribute const*, int32_t,
            chainner_migx_tensor const* output, void*, size_t, char* error, size_t error_size) {
    hipError_t const status =
        vosr2_qkv_pack(inputs[0].data, inputs[1].data, inputs[2].data, inputs[3].data, inputs[4].data, output->data,
                       static_cast<int>(inputs[0].shape.dims[1]), static_cast<hipStream_t>(stream));
    if (status != hipSuccess) return fail(error, error_size, hipGetErrorString(status));
    return 0;
}

// FIXED_TYPES: the kernel computes in BF16 whatever precision the program is compiled with.
chainner_migx_op const ops[] = {{"chainner_vosr2_qkv", CHAINNER_MIGX_FIXED_TYPES, compute_shape, compute}};

}  // namespace

extern "C" CHAINNER_MIGX_EXPORT int chainner_migx_ops(uint32_t abi, chainner_migx_op const** list,
                                                      int32_t* count) {
    if (abi != CHAINNER_MIGX_ABI) return 1;
    *list = ops;
    *count = 1;
    return 0;
}
