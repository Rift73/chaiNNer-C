// drft_migx_ops: legacy DRFT's MIGraphX ops (migraphx_runner/custom_op.h), the counterparts of the
// TensorRT plugins of src/drft with the same attributes, computed by drft_hip_kernels.hip on the
// program's stream. Every op keeps the ONNX's types (FP16 body, BF16 attention, FP32 output tail).
//   chainner_drft_window_neural_bias  (heads, N, r), (heads, M, r) -> (heads, N, M); FP16 or BF16
//   chainner_drft_ocab_relative_bias  table (span^2, heads) -> (1, heads, q^2, k^2); FP32, FP16, BF16
//   chainner_drft_ocab_window_pack    qkv (B, H, W, 3, C) -> (window^2, windows, heads, C / heads);
//                                     FP16 or BF16; one of the q/k/v pack's outputs (component)
//   chainner_drft_final_affine_crop   x (B, 3, h, w) FP16 or BF16, image (B, 3, H, W)
//                                     -> (B, 3, H * scale, W * scale) FP32
//   chainner_drft_ocab_attention      q, k transposed, v, table -> like q; BF16
#include "custom_op.h"
#include "drft_hip_kernels.h"

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace
{
using drft_hip::ElementType;

int fail(char* error, size_t errorSize, char const* format, ...)
{
    if (error != nullptr && errorSize > 0)
    {
        va_list arguments;
        va_start(arguments, format);
        std::vsnprintf(error, errorSize, format, arguments);
        va_end(arguments);
    }
    return 1;
}

int launched(hipError_t status, char const* op, char* error, size_t errorSize)
{
    return status == hipSuccess
        ? 0
        : fail(error, errorSize, "%s: %s", op, hipGetErrorString(status));
}

chainner_migx_attribute const* findAttribute(chainner_migx_attribute const* attributes,
    int32_t count, char const* name)
{
    for (int32_t index = 0; index < count; ++index)
    {
        if (attributes[index].name != nullptr && std::strcmp(attributes[index].name, name) == 0)
        {
            return &attributes[index];
        }
    }
    return nullptr;
}

bool intAttribute(chainner_migx_attribute const* attributes, int32_t count, char const* name,
    int32_t& value)
{
    chainner_migx_attribute const* attribute = findAttribute(attributes, count, name);
    if (attribute == nullptr || attribute->kind != CHAINNER_MIGX_ATTRIBUTE_INT
        || attribute->ints == nullptr || attribute->ints[0] < INT32_MIN
        || attribute->ints[0] > INT32_MAX)
    {
        return false;
    }
    value = static_cast<int32_t>(attribute->ints[0]);
    return true;
}

bool floatAttribute(chainner_migx_attribute const* attributes, int32_t count, char const* name,
    float& value)
{
    chainner_migx_attribute const* attribute = findAttribute(attributes, count, name);
    if (attribute == nullptr || attribute->kind != CHAINNER_MIGX_ATTRIBUTE_FLOAT
        || attribute->floats == nullptr)
    {
        return false;
    }
    value = attribute->floats[0];
    return true;
}

bool isSixteenBit(int32_t type)
{
    return type == CHAINNER_MIGX_HALF || type == CHAINNER_MIGX_BF16;
}

ElementType elementType(int32_t type)
{
    switch (type)
    {
    case CHAINNER_MIGX_HALF:
        return ElementType::kFloat16;
    case CHAINNER_MIGX_BF16:
        return ElementType::kBFloat16;
    default:
        return ElementType::kFloat32;
    }
}

bool fitsInt32(chainner_migx_shape const& shape)
{
    for (int32_t axis = 0; axis < shape.rank; ++axis)
    {
        if (shape.dims[axis] <= 0 || shape.dims[axis] > INT32_MAX)
        {
            return false;
        }
    }
    return true;
}

void setShape(chainner_migx_shape* output, int32_t type, int32_t rank, int64_t const* dims)
{
    output->type = type;
    output->rank = rank;
    for (int32_t axis = 0; axis < rank; ++axis)
    {
        output->dims[axis] = dims[axis];
    }
}

// The OCAB windows the relative bias and attention kernels take.
bool isOCABWindow(int32_t queryWindow, int32_t keyWindow)
{
    return queryWindow == 32 && (keyWindow == 40 || keyWindow == 48);
}

constexpr char kWindowNeuralBias[] = "chainner_drft_window_neural_bias";

int windowNeuralBiasShape(chainner_migx_shape const* inputs, int32_t inputCount,
    chainner_migx_attribute const*, int32_t, chainner_migx_shape* output, size_t*, char* error,
    size_t errorSize)
{
    if (inputCount != 2 || inputs[0].rank != 3 || inputs[1].rank != 3
        || !isSixteenBit(inputs[0].type) || inputs[1].type != inputs[0].type
        || !fitsInt32(inputs[0]) || !fitsInt32(inputs[1])
        || inputs[1].dims[0] != inputs[0].dims[0] || inputs[1].dims[2] != inputs[0].dims[2])
    {
        return fail(error, errorSize,
            "%s takes two FP16 or BF16 factors (heads, tokens, rank) of the same heads, rank and"
            " type",
            kWindowNeuralBias);
    }
    int64_t const dims[3] = {inputs[0].dims[0], inputs[0].dims[1], inputs[1].dims[1]};
    setShape(output, inputs[0].type, 3, dims);
    return 0;
}

int windowNeuralBiasCompute(void* stream, chainner_migx_tensor const* inputs, int32_t,
    chainner_migx_attribute const*, int32_t, chainner_migx_tensor const* output, void*, size_t,
    char* error, size_t errorSize)
{
    chainner_migx_shape const& query = inputs[0].shape;
    return launched(drft_hip::launchWindowNeuralBias(inputs[0].data, inputs[1].data,
                        output->data, elementType(query.type),
                        static_cast<int32_t>(query.dims[0]), static_cast<int32_t>(query.dims[1]),
                        static_cast<int32_t>(inputs[1].shape.dims[1]),
                        static_cast<int32_t>(query.dims[2]), static_cast<hipStream_t>(stream)),
        kWindowNeuralBias, error, errorSize);
}

constexpr char kRelativeBias[] = "chainner_drft_ocab_relative_bias";

bool relativeBiasWindows(chainner_migx_attribute const* attributes, int32_t attributeCount,
    int32_t& queryWindow, int32_t& keyWindow)
{
    return intAttribute(attributes, attributeCount, "q_window_size", queryWindow)
        && intAttribute(attributes, attributeCount, "kv_window_size", keyWindow)
        && isOCABWindow(queryWindow, keyWindow);
}

int relativeBiasShape(chainner_migx_shape const* inputs, int32_t inputCount,
    chainner_migx_attribute const* attributes, int32_t attributeCount,
    chainner_migx_shape* output, size_t*, char* error, size_t errorSize)
{
    int32_t queryWindow = 0;
    int32_t keyWindow = 0;
    if (!relativeBiasWindows(attributes, attributeCount, queryWindow, keyWindow))
    {
        return fail(error, errorSize,
            "%s takes q_window_size 32 and kv_window_size 40 or 48", kRelativeBias);
    }
    int64_t const span = queryWindow + keyWindow - 1;
    if (inputCount != 1 || inputs[0].rank != 2 || !fitsInt32(inputs[0])
        || inputs[0].dims[0] != span * span
        || (inputs[0].type != CHAINNER_MIGX_FLOAT && !isSixteenBit(inputs[0].type)))
    {
        return fail(error, errorSize,
            "%s takes one FP32, FP16 or BF16 table (%lld, heads)", kRelativeBias,
            static_cast<long long>(span * span));
    }
    int64_t const dims[4] = {1, inputs[0].dims[1],
        static_cast<int64_t>(queryWindow) * queryWindow,
        static_cast<int64_t>(keyWindow) * keyWindow};
    setShape(output, inputs[0].type, 4, dims);
    return 0;
}

int relativeBiasCompute(void* stream, chainner_migx_tensor const* inputs, int32_t,
    chainner_migx_attribute const* attributes, int32_t attributeCount,
    chainner_migx_tensor const* output, void*, size_t, char* error, size_t errorSize)
{
    int32_t queryWindow = 0;
    int32_t keyWindow = 0;
    if (!relativeBiasWindows(attributes, attributeCount, queryWindow, keyWindow))
    {
        return fail(error, errorSize, "%s: bad window attributes", kRelativeBias);
    }
    return launched(drft_hip::launchOCABRelativeBias(inputs[0].data, output->data,
                        elementType(inputs[0].shape.type),
                        static_cast<int32_t>(inputs[0].shape.dims[1]), queryWindow, keyWindow,
                        static_cast<hipStream_t>(stream)),
        kRelativeBias, error, errorSize);
}

constexpr char kWindowPack[] = "chainner_drft_ocab_window_pack";

struct WindowPack
{
    int32_t heads;
    int32_t queryWindow;
    int32_t window;
    int32_t component;
};

bool windowPackAttributes(chainner_migx_attribute const* attributes, int32_t attributeCount,
    WindowPack& pack)
{
    return intAttribute(attributes, attributeCount, "heads", pack.heads)
        && intAttribute(attributes, attributeCount, "query_window_size", pack.queryWindow)
        && intAttribute(attributes, attributeCount, "window_size", pack.window)
        && intAttribute(attributes, attributeCount, "component", pack.component)
        && pack.heads > 0 && pack.queryWindow > 0 && pack.window >= pack.queryWindow
        && (pack.window - pack.queryWindow) % 2 == 0 && pack.component >= 0
        && pack.component <= 2;
}

int windowPackShape(chainner_migx_shape const* inputs, int32_t inputCount,
    chainner_migx_attribute const* attributes, int32_t attributeCount,
    chainner_migx_shape* output, size_t*, char* error, size_t errorSize)
{
    WindowPack pack{};
    if (!windowPackAttributes(attributes, attributeCount, pack))
    {
        return fail(error, errorSize,
            "%s takes heads, query_window_size, window_size (at least the query window, an even"
            " number more) and component 0, 1 or 2",
            kWindowPack);
    }
    if (inputCount != 1 || inputs[0].rank != 5 || !isSixteenBit(inputs[0].type)
        || !fitsInt32(inputs[0]) || inputs[0].dims[3] != 3)
    {
        return fail(error, errorSize, "%s takes one FP16 or BF16 qkv (B, H, W, 3, C)",
            kWindowPack);
    }
    int64_t const* dims = inputs[0].dims;
    int64_t const channels = dims[4];
    if (channels % pack.heads != 0 || (channels / pack.heads) % 8 != 0
        || dims[1] % pack.queryWindow != 0 || dims[2] % pack.queryWindow != 0)
    {
        return fail(error, errorSize,
            "%s: %lld channels in %d heads (a multiple of 8 per head) and a %lldx%lld grid of"
            " %d px windows are needed",
            kWindowPack, static_cast<long long>(channels), pack.heads,
            static_cast<long long>(dims[1]), static_cast<long long>(dims[2]), pack.queryWindow);
    }
    int64_t const outputDims[4] = {static_cast<int64_t>(pack.window) * pack.window,
        dims[0] * (dims[1] / pack.queryWindow) * (dims[2] / pack.queryWindow), pack.heads,
        channels / pack.heads};
    setShape(output, inputs[0].type, 4, outputDims);
    return 0;
}

int windowPackCompute(void* stream, chainner_migx_tensor const* inputs, int32_t,
    chainner_migx_attribute const* attributes, int32_t attributeCount,
    chainner_migx_tensor const* output, void*, size_t, char* error, size_t errorSize)
{
    WindowPack pack{};
    if (!windowPackAttributes(attributes, attributeCount, pack))
    {
        return fail(error, errorSize, "%s: bad attributes", kWindowPack);
    }
    int64_t const* dims = inputs[0].shape.dims;
    return launched(drft_hip::launchOCABWindowPack(inputs[0].data, output->data,
                        static_cast<int32_t>(dims[0]), static_cast<int32_t>(dims[1]),
                        static_cast<int32_t>(dims[2]), static_cast<int32_t>(dims[4]), pack.heads,
                        pack.queryWindow, pack.window, pack.component,
                        static_cast<hipStream_t>(stream)),
        kWindowPack, error, errorSize);
}

constexpr char kFinalAffineCrop[] = "chainner_drft_final_affine_crop";

struct FinalAffineCrop
{
    int32_t scale;
    float mean[3];
};

bool finalAffineCropAttributes(chainner_migx_attribute const* attributes,
    int32_t attributeCount, FinalAffineCrop& crop)
{
    return intAttribute(attributes, attributeCount, "scale", crop.scale)
        && floatAttribute(attributes, attributeCount, "mean_r", crop.mean[0])
        && floatAttribute(attributes, attributeCount, "mean_g", crop.mean[1])
        && floatAttribute(attributes, attributeCount, "mean_b", crop.mean[2]) && crop.scale > 0;
}

int finalAffineCropShape(chainner_migx_shape const* inputs, int32_t inputCount,
    chainner_migx_attribute const* attributes, int32_t attributeCount,
    chainner_migx_shape* output, size_t*, char* error, size_t errorSize)
{
    FinalAffineCrop crop{};
    if (!finalAffineCropAttributes(attributes, attributeCount, crop))
    {
        return fail(error, errorSize, "%s takes a positive scale and mean_r, mean_g, mean_b",
            kFinalAffineCrop);
    }
    if (inputCount != 2 || inputs[0].rank != 4 || inputs[1].rank != 4
        || !isSixteenBit(inputs[0].type) || !fitsInt32(inputs[0]) || !fitsInt32(inputs[1])
        || inputs[0].dims[1] != 3 || inputs[1].dims[0] != inputs[0].dims[0]
        || inputs[1].dims[2] * crop.scale > inputs[0].dims[2]
        || inputs[1].dims[3] * crop.scale > inputs[0].dims[3])
    {
        return fail(error, errorSize,
            "%s takes an FP16 or BF16 (B, 3, h, w) and an image (B, C, H, W) with H * scale <= h"
            " and W * scale <= w",
            kFinalAffineCrop);
    }
    int64_t const dims[4] = {inputs[0].dims[0], 3, inputs[1].dims[2] * crop.scale,
        inputs[1].dims[3] * crop.scale};
    setShape(output, CHAINNER_MIGX_FLOAT, 4, dims);
    return 0;
}

int finalAffineCropCompute(void* stream, chainner_migx_tensor const* inputs, int32_t,
    chainner_migx_attribute const* attributes, int32_t attributeCount,
    chainner_migx_tensor const* output, void*, size_t, char* error, size_t errorSize)
{
    FinalAffineCrop crop{};
    if (!finalAffineCropAttributes(attributes, attributeCount, crop))
    {
        return fail(error, errorSize, "%s: bad attributes", kFinalAffineCrop);
    }
    chainner_migx_shape const& source = inputs[0].shape;
    chainner_migx_shape const& target = output->shape;
    return launched(drft_hip::launchFinalAffineCrop(inputs[0].data,
                        static_cast<float*>(output->data), elementType(source.type),
                        static_cast<int32_t>(source.dims[0]), 3,
                        static_cast<int32_t>(source.dims[2]), static_cast<int32_t>(source.dims[3]),
                        static_cast<int32_t>(target.dims[2]), static_cast<int32_t>(target.dims[3]),
                        crop.mean, static_cast<hipStream_t>(stream)),
        kFinalAffineCrop, error, errorSize);
}

constexpr char kAttention[] = "chainner_drft_ocab_attention";

int attentionShape(chainner_migx_shape const* inputs, int32_t inputCount,
    chainner_migx_attribute const* attributes, int32_t attributeCount,
    chainner_migx_shape* output, size_t*, char* error, size_t errorSize)
{
    int32_t queryWindow = 0;
    int32_t keyWindow = 0;
    if (!relativeBiasWindows(attributes, attributeCount, queryWindow, keyWindow))
    {
        return fail(error, errorSize,
            "%s takes q_window_size 32 and kv_window_size 40 or 48", kAttention);
    }
    int64_t const queryTokens = static_cast<int64_t>(queryWindow) * queryWindow;
    int64_t const keyTokens = static_cast<int64_t>(keyWindow) * keyWindow;
    int64_t const span = queryWindow + keyWindow - 1;
    bool valid = inputCount == 4;
    for (int32_t index = 0; valid && index < inputCount; ++index)
    {
        valid = inputs[index].type == CHAINNER_MIGX_BF16 && fitsInt32(inputs[index])
            && inputs[index].rank == (index == 3 ? 2 : 4);
    }
    if (valid)
    {
        int64_t const* q = inputs[0].dims;
        int64_t const* k = inputs[1].dims;
        int64_t const* v = inputs[2].dims;
        int64_t const* table = inputs[3].dims;
        valid = q[2] == queryTokens && q[3] == 32 && k[0] == q[0] && k[1] == q[1] && k[2] == 32
            && k[3] == keyTokens && v[0] == q[0] && v[1] == q[1] && v[2] == keyTokens
            && v[3] == 32 && table[0] == span * span && table[1] == q[1];
    }
    if (!valid)
    {
        return fail(error, errorSize,
            "%s takes BF16 q (B, heads, %lld, 32), k transposed (B, heads, 32, %lld), v (B, heads,"
            " %lld, 32) and table (%lld, heads)",
            kAttention, static_cast<long long>(queryTokens), static_cast<long long>(keyTokens),
            static_cast<long long>(keyTokens), static_cast<long long>(span * span));
    }
    *output = inputs[0];
    return 0;
}

int attentionCompute(void* stream, chainner_migx_tensor const* inputs, int32_t,
    chainner_migx_attribute const* attributes, int32_t attributeCount,
    chainner_migx_tensor const* output, void*, size_t, char* error, size_t errorSize)
{
    int32_t queryWindow = 0;
    int32_t keyWindow = 0;
    if (!relativeBiasWindows(attributes, attributeCount, queryWindow, keyWindow))
    {
        return fail(error, errorSize, "%s: bad window attributes", kAttention);
    }
    chainner_migx_shape const& query = inputs[0].shape;
    return launched(drft_hip::launchOCABAttention(inputs[0].data, inputs[1].data, inputs[2].data,
                        inputs[3].data, output->data, static_cast<int32_t>(query.dims[0]),
                        static_cast<int32_t>(query.dims[1]), queryWindow, keyWindow,
                        static_cast<hipStream_t>(stream)),
        kAttention, error, errorSize);
}

constexpr chainner_migx_op kOps[] = {
    {kWindowNeuralBias, CHAINNER_MIGX_FIXED_TYPES, windowNeuralBiasShape,
        windowNeuralBiasCompute},
    {kRelativeBias, CHAINNER_MIGX_FIXED_TYPES, relativeBiasShape, relativeBiasCompute},
    {kWindowPack, CHAINNER_MIGX_FIXED_TYPES, windowPackShape, windowPackCompute},
    {kFinalAffineCrop, CHAINNER_MIGX_FIXED_TYPES, finalAffineCropShape, finalAffineCropCompute},
    {kAttention, CHAINNER_MIGX_FIXED_TYPES, attentionShape, attentionCompute},
};
} // namespace

extern "C" CHAINNER_MIGX_EXPORT int chainner_migx_ops(
    uint32_t abi, chainner_migx_op const** ops, int32_t* count)
{
    if (abi != CHAINNER_MIGX_ABI || ops == nullptr || count == nullptr)
    {
        return 1;
    }
    *ops = kOps;
    *count = static_cast<int32_t>(sizeof(kOps) / sizeof(kOps[0]));
    return 0;
}
