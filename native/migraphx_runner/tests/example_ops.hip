// A custom-op library for the MIGraphX runner's tests (custom_op.h), with one op of each kind:
// - chainner_example_repeat (on the host, for MIGraphX's ref target): x [N, C, H, W] float32 or float16 to
//   y [N, C, H * factor, W * factor] of x's type, y[n, c, i, j] = x[n, order[c], i / factor, j / factor] * gain + bias[c]
//   in float32, with the attributes factor (int), gain (float), order (ints, a permutation of the channels), bias
//   (floats, one per channel) and tag (string, "chain\0ner", which arrives byte for byte);
// - chainner_example_repeat_float (on the host): the same for float32 only, keeping its types under quantization;
// - chainner_example_repeat_hip (on the GPU): the same for float32 with at most 8 channels, keeping its types under
//   quantization and asking for 256 bytes of scratch, which it clears.
#include <hip/hip_runtime.h>

#include "custom_op.h"

#include <cstdio>
#include <cstring>

namespace {

constexpr int max_device_channels = 8;
constexpr size_t device_scratch = 256;
constexpr char tag[] = "chain\0ner";

struct Parameters {
    int64_t factor = 0;
    float gain = 0.0f;
    int64_t const* order = nullptr;
    float const* bias = nullptr;
};

int fail(char* error, size_t error_size, char const* message) {
    std::snprintf(error, error_size, "%s", message);
    return 1;
}

int read(chainner_migx_attribute const* attributes, int32_t count, int64_t channels, Parameters& p, char* error,
         size_t error_size) {
    bool tagged = false, gained = false;
    for (int32_t i = 0; i < count; ++i) {
        chainner_migx_attribute const& a = attributes[i];
        if (!std::strcmp(a.name, "factor") && a.kind == CHAINNER_MIGX_ATTRIBUTE_INT) {
            p.factor = a.ints[0];
        } else if (!std::strcmp(a.name, "gain") && a.kind == CHAINNER_MIGX_ATTRIBUTE_FLOAT) {
            p.gain = a.floats[0];
            gained = true;
        } else if (!std::strcmp(a.name, "order") && a.kind == CHAINNER_MIGX_ATTRIBUTE_INTS && a.count == channels) {
            p.order = a.ints;
        } else if (!std::strcmp(a.name, "bias") && a.kind == CHAINNER_MIGX_ATTRIBUTE_FLOATS && a.count == channels) {
            p.bias = a.floats;
        } else if (!std::strcmp(a.name, "tag") && a.kind == CHAINNER_MIGX_ATTRIBUTE_STRING &&
                   a.count == sizeof tag - 1 && !std::memcmp(a.bytes, tag, sizeof tag)) {
            tagged = true;
        } else {
            return fail(error, error_size, "unexpected attribute");
        }
    }
    if (p.factor < 1 || !gained || !p.order || !p.bias || !tagged) return fail(error, error_size, "missing attribute");
    for (int64_t c = 0; c < channels; ++c)
        if (p.order[c] < 0 || p.order[c] >= channels) return fail(error, error_size, "order is not a permutation");
    return 0;
}

int repeat_shape(chainner_migx_shape const* inputs, int32_t input_count, chainner_migx_attribute const* attributes,
                 int32_t attribute_count, chainner_migx_shape* output, char* error, size_t error_size, bool device,
                 bool float32) {
    if (input_count != 1 || inputs[0].rank != 4) return fail(error, error_size, "takes one NCHW tensor");
    if (inputs[0].type != CHAINNER_MIGX_FLOAT && (float32 || inputs[0].type != CHAINNER_MIGX_HALF))
        return fail(error, error_size, float32 ? "takes float32" : "takes float32 or float16");
    if (device && inputs[0].dims[1] > max_device_channels) return fail(error, error_size, "takes 8 channels at most");
    Parameters p;
    if (read(attributes, attribute_count, inputs[0].dims[1], p, error, error_size)) return 1;
    *output = inputs[0];
    output->dims[2] *= p.factor;
    output->dims[3] *= p.factor;
    return 0;
}

int host_shape(chainner_migx_shape const* inputs, int32_t input_count, chainner_migx_attribute const* attributes,
               int32_t attribute_count, chainner_migx_shape* output, size_t*, char* error, size_t error_size) {
    return repeat_shape(inputs, input_count, attributes, attribute_count, output, error, error_size, false, false);
}

int float_shape(chainner_migx_shape const* inputs, int32_t input_count, chainner_migx_attribute const* attributes,
                int32_t attribute_count, chainner_migx_shape* output, size_t*, char* error, size_t error_size) {
    return repeat_shape(inputs, input_count, attributes, attribute_count, output, error, error_size, false, true);
}

int host_compute(void* stream, chainner_migx_tensor const* inputs, int32_t, chainner_migx_attribute const* attributes,
                 int32_t attribute_count, chainner_migx_tensor const* output, void*, size_t, char* error,
                 size_t error_size) {
    if (stream) return fail(error, error_size, "a host op has no stream");
    chainner_migx_shape const& s = inputs[0].shape;
    int64_t const n = s.dims[0], c = s.dims[1], h = s.dims[2], w = s.dims[3];
    Parameters p;
    if (read(attributes, attribute_count, c, p, error, error_size)) return 1;
    int64_t const oh = h * p.factor, ow = w * p.factor;
    bool const half = s.type == CHAINNER_MIGX_HALF;
    for (int64_t b = 0; b < n; ++b)
        for (int64_t k = 0; k < c; ++k)
            for (int64_t i = 0; i < oh; ++i)
                for (int64_t j = 0; j < ow; ++j) {
                    int64_t const from = ((b * c + p.order[k]) * h + i / p.factor) * w + j / p.factor;
                    int64_t const to = ((b * c + k) * oh + i) * ow + j;
                    float const x = half ? static_cast<float>(static_cast<_Float16 const*>(inputs[0].data)[from])
                                         : static_cast<float const*>(inputs[0].data)[from];
                    float const y = x * p.gain + p.bias[k];
                    if (half) static_cast<_Float16*>(output->data)[to] = static_cast<_Float16>(y);
                    else static_cast<float*>(output->data)[to] = y;
                }
    return 0;
}

struct Channels {
    int64_t order[max_device_channels];
    float bias[max_device_channels];
};

__global__ void repeat_kernel(float const* x, float* y, int64_t n, int64_t c, int64_t h, int64_t w, int64_t factor,
                              float gain, Channels channels) {
    int64_t const oh = h * factor, ow = w * factor;
    int64_t const index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= n * c * oh * ow) return;
    int64_t const j = index % ow, i = index / ow % oh, k = index / (ow * oh) % c, b = index / (ow * oh * c);
    float const v = x[((b * c + channels.order[k]) * h + i / factor) * w + j / factor];
    y[index] = v * gain + channels.bias[k];
}

int device_shape(chainner_migx_shape const* inputs, int32_t input_count, chainner_migx_attribute const* attributes,
                 int32_t attribute_count, chainner_migx_shape* output, size_t* workspace_bytes, char* error,
                 size_t error_size) {
    *workspace_bytes = device_scratch;
    return repeat_shape(inputs, input_count, attributes, attribute_count, output, error, error_size, true, true);
}

int device_compute(void* stream, chainner_migx_tensor const* inputs, int32_t, chainner_migx_attribute const* attributes,
                   int32_t attribute_count, chainner_migx_tensor const* output, void* workspace, size_t workspace_bytes,
                   char* error, size_t error_size) {
    chainner_migx_shape const& s = inputs[0].shape;
    Parameters p;
    if (read(attributes, attribute_count, s.dims[1], p, error, error_size)) return 1;
    if (!workspace || workspace_bytes != device_scratch) return fail(error, error_size, "no scratch");
    Channels channels{};
    for (int64_t k = 0; k < s.dims[1]; ++k) {
        channels.order[k] = p.order[k];
        channels.bias[k] = p.bias[k];
    }
    auto const queue = static_cast<hipStream_t>(stream);
    hipError_t status = hipMemsetAsync(workspace, 0, workspace_bytes, queue);
    int64_t const elements = s.dims[0] * s.dims[1] * s.dims[2] * s.dims[3] * p.factor * p.factor;
    if (status == hipSuccess) {
        repeat_kernel<<<static_cast<unsigned>((elements + 255) / 256), 256, 0, queue>>>(
            static_cast<float const*>(inputs[0].data), static_cast<float*>(output->data), s.dims[0], s.dims[1],
            s.dims[2], s.dims[3], p.factor, p.gain, channels);
        status = hipGetLastError();
    }
    if (status != hipSuccess) return fail(error, error_size, hipGetErrorString(status));
    return 0;
}

chainner_migx_op const ops[] = {
    {"chainner_example_repeat", CHAINNER_MIGX_HOST, host_shape, host_compute},
    {"chainner_example_repeat_float", CHAINNER_MIGX_HOST | CHAINNER_MIGX_FIXED_TYPES, float_shape, host_compute},
    {"chainner_example_repeat_hip", CHAINNER_MIGX_FIXED_TYPES, device_shape, device_compute},
};

}  // namespace

extern "C" CHAINNER_MIGX_EXPORT int chainner_migx_ops(uint32_t abi, chainner_migx_op const** list, int32_t* count) {
    if (abi != CHAINNER_MIGX_ABI) return 1;
    *list = ops;
    *count = static_cast<int32_t>(sizeof ops / sizeof ops[0]);
    return 0;
}
