/* DUAL's MIGraphX custom ops (dual_migx_ops; contract: native/migraphx_runner/custom_op.h): each
 * launches one of the Triton kernels of DUAL's TensorRT plugins (backend
 * nodes/impl/tensorrt/dual_dynamic_kernels.py, dual_na_kernel.py), which Build Program compiles
 * for the GPU and the op's node carries (nodes/impl/migraphx/dual_hip.py lowers the exporter's
 * ONNX): the code object (attribute "code": HSACO on AMD, PTX through HIP on NVIDIA), its
 * SHA-256 ("code_id"), the function ("kernel"), threads per block ("block") and shared memory
 * ("shared"), plus the tile sizes the grids take ("rows", "parts"). The library computes each
 * grid and the trunk's run-time sizes from the input shapes as tensorrt/dual_aot.py's dynamic
 * plugins do, and holds no kernel code of its own: only HIP's host API.
 *
 * Triton's kernels take their non-constexpr arguments in order, then its global and profile
 * scratch pointers (null: dual_hip.py refuses kernels that need either). They are compiled for
 * 16-byte aligned buffers (Norm's sigma scratch excepted), which every launch checks. All tensors are batch 1, BF16 except
 * Norm's affine and sigma, Core's temperatures and statistics (FP32).
 *
 * chainner_dual_norm     (x [..., C], gamma [C], beta [C]) -> y like x; sigma in scratch
 * chainner_dual_cell     (qkv [1,H,W,3C], weight [3,3,3C]) -> statistics [cells * heads * 1088]
 * chainner_dual_region   (statistics, tau [heads], qkv) -> matrices [regions * heads * 1024]
 * chainner_dual_apply    (qkv, weight, matrices) -> [1,H,W,C]
 * chainner_dual_project  (x [..., C], weight [C,C], residual like x) -> like x
 * chainner_dual_na       (qkv, bq [heads,(M+15)^2,16], bk like bq; window M) -> [1,H,W,C]
 * chainner_dual_barrier  (x) -> a copy of x
 * heads = C / 32, cells = ceil(H/16) * ceil(W/16), regions = the four shifted 32x32 region
 * grids' counts (dual_aot.Geometry). */

#include "custom_op.h"

#include <hip/hip_runtime_api.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace {

int fail(char* error, size_t error_size, char const* format, ...) {
    va_list arguments;
    va_start(arguments, format);
    std::vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
    return 1;
}

struct Attributes {
    chainner_migx_attribute const* items;
    int32_t count;

    chainner_migx_attribute const* find(char const* name, int32_t kind) const {
        for (int32_t i = 0; i < count; ++i) {
            if (std::strcmp(items[i].name, name) == 0 && items[i].kind == kind) {
                return &items[i];
            }
        }
        return nullptr;
    }

    /* An INT attribute, or -1 when the node has none. */
    int64_t integer(char const* name) const {
        auto const* attribute = find(name, CHAINNER_MIGX_ATTRIBUTE_INT);
        return attribute == nullptr ? -1 : attribute->ints[0];
    }
};

int64_t elements(chainner_migx_shape const& shape) {
    int64_t count = 1;
    for (int32_t i = 0; i < shape.rank; ++i) {
        count *= shape.dims[i];
    }
    return count;
}

int64_t cdiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

/* The trunk a [1,H,W,3C] QKV tensor describes, with its RCA layout. */
struct Trunk {
    int64_t height, width, channels;
    int64_t counts[4];

    int64_t heads() const { return channels / 32; }
    int64_t cells() const { return cdiv(height, 16) * cdiv(width, 16); }
    int64_t regions() const { return counts[0] + counts[1] + counts[2] + counts[3]; }
};

int trunk_of(chainner_migx_shape const& qkv, Trunk* trunk, char* error, size_t error_size) {
    if (qkv.type != CHAINNER_MIGX_BF16 || qkv.rank != 4 || qkv.dims[0] != 1 || qkv.dims[3] % 96 != 0 ||
        qkv.dims[1] < 1 || qkv.dims[2] < 1) {
        return fail(error, error_size, "expects a BF16 [1,H,W,3C] QKV with C a multiple of 32");
    }
    *trunk = Trunk{qkv.dims[1], qkv.dims[2], qkv.dims[3] / 3, {0, 0, 0, 0}};
    // The kernels index QKV with 32-bit offsets (dual.check_profile's bound).
    if (qkv.dims[1] * qkv.dims[2] * qkv.dims[3] >= (int64_t{1} << 31)) {
        return fail(error, error_size, "a %lldx%lld trunk is too large for DUAL's kernels' 32-bit indexing",
                    static_cast<long long>(qkv.dims[1]), static_cast<long long>(qkv.dims[2]));
    }
    int const shifts[4][2] = {{0, 0}, {0, 16}, {16, 0}, {16, 16}};
    for (int i = 0; i < 4; ++i) {
        trunk->counts[i] = cdiv(trunk->height + shifts[i][0], 32) * cdiv(trunk->width + shifts[i][1], 32);
    }
    return 0;
}

bool same_dims(chainner_migx_shape const& shape, std::initializer_list<int64_t> dims) {
    if (shape.rank != static_cast<int32_t>(dims.size())) {
        return false;
    }
    int32_t i = 0;
    for (int64_t dim : dims) {
        if (shape.dims[i++] != dim) {
            return false;
        }
    }
    return true;
}

chainner_migx_shape shape_of(int32_t type, std::initializer_list<int64_t> dims) {
    chainner_migx_shape shape{};
    shape.type = type;
    for (int64_t dim : dims) {
        shape.dims[shape.rank++] = dim;
    }
    return shape;
}

/* The code objects loaded on each device, keyed by device and code_id; each entry keeps the
 * code it was loaded from, which every use compares, so a code_id never stands for other code. */
struct Loaded {
    std::string code;
    hipModule_t module;
    std::map<std::string, hipFunction_t> functions;
};

std::mutex loaded_lock;
std::map<std::pair<int, std::string>, Loaded> loaded;

int function_of(Attributes const& attributes, hipFunction_t* function, char* error, size_t error_size) {
    auto const* code = attributes.find("code", CHAINNER_MIGX_ATTRIBUTE_STRING);
    auto const* code_id = attributes.find("code_id", CHAINNER_MIGX_ATTRIBUTE_STRING);
    auto const* kernel = attributes.find("kernel", CHAINNER_MIGX_ATTRIBUTE_STRING);
    if (code == nullptr || code_id == nullptr || kernel == nullptr) {
        return fail(error, error_size, "the node carries no kernel (code, code_id, kernel)");
    }
    int device = 0;
    hipError_t status = hipGetDevice(&device);
    if (status != hipSuccess) {
        return fail(error, error_size, "hipGetDevice failed: %s", hipGetErrorString(status));
    }
    std::lock_guard<std::mutex> guard(loaded_lock);
    auto key = std::make_pair(device, std::string(code_id->bytes, static_cast<size_t>(code_id->count)));
    auto entry = loaded.find(key);
    if (entry == loaded.end()) {
        hipModule_t module = nullptr;
        status = hipModuleLoadData(&module, code->bytes);
        if (status != hipSuccess) {
            return fail(error, error_size, "loading kernel %s on device %d failed: %s", kernel->bytes, device,
                        hipGetErrorString(status));
        }
        entry = loaded.emplace(key, Loaded{std::string(code->bytes, static_cast<size_t>(code->count)), module, {}})
                    .first;
    } else if (entry->second.code.size() != static_cast<size_t>(code->count) ||
               std::memcmp(entry->second.code.data(), code->bytes, entry->second.code.size()) != 0) {
        return fail(error, error_size, "kernel %s: code_id %s names different code objects", kernel->bytes,
                    key.second.c_str());
    }
    auto& functions = entry->second.functions;
    auto found = functions.find(kernel->bytes);
    if (found == functions.end()) {
        hipFunction_t loaded_function = nullptr;
        status = hipModuleGetFunction(&loaded_function, entry->second.module, kernel->bytes);
        if (status != hipSuccess) {
            return fail(error, error_size, "kernel %s is not in its code object: %s", kernel->bytes,
                        hipGetErrorString(status));
        }
        found = functions.emplace(kernel->bytes, loaded_function).first;
    }
    *function = found->second;
    return 0;
}

/* A Triton kernel's argument list: pointers and 32-bit integers in order, then the two scratch
 * pointers. */
class Arguments {
public:
    int pointer(void* value, char const* what, char* error, size_t error_size, bool aligned = true) {
        if (aligned && reinterpret_cast<uintptr_t>(value) % 16 != 0) {
            return fail(error, error_size, "DUAL's kernels need 16-byte aligned buffers; %s is at %p", what, value);
        }
        values_.push_back(reinterpret_cast<uintptr_t>(value));
        return 0;
    }
    void integer(int64_t value) {
        int32_t narrowed = static_cast<int32_t>(value);
        uintptr_t slot = 0;
        std::memcpy(&slot, &narrowed, sizeof narrowed);
        values_.push_back(slot);
    }
    void** finish() {
        values_.push_back(0);  // global scratch
        values_.push_back(0);  // profile scratch
        for (auto& value : values_) {
            pointers_.push_back(&value);
        }
        return pointers_.data();
    }

private:
    std::vector<uintptr_t> values_;
    std::vector<void*> pointers_;
};

int launch(Attributes const& attributes, void* stream, Arguments& arguments, int64_t grid_x, int64_t grid_y,
           int64_t grid_z, char* error, size_t error_size) {
    int64_t const block = attributes.integer("block");
    int64_t const shared = attributes.integer("shared");
    if (block < 1 || shared < 0) {
        return fail(error, error_size, "the node carries no launch size (block, shared)");
    }
    hipFunction_t function = nullptr;
    if (int failed = function_of(attributes, &function, error, error_size)) {
        return failed;
    }
    void** parameters = arguments.finish();
    hipError_t status = hipModuleLaunchKernel(
        function, static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y), static_cast<unsigned>(grid_z),
        static_cast<unsigned>(block), 1, 1, static_cast<unsigned>(shared), static_cast<hipStream_t>(stream),
        parameters, nullptr);
    if (status != hipSuccess) {
        return fail(error, error_size, "launching %s failed: %s",
                    attributes.find("kernel", CHAINNER_MIGX_ATTRIBUTE_STRING)->bytes, hipGetErrorString(status));
    }
    return 0;
}

/* Every op: shapes in, attributes, output, scratch bytes */
#define SHAPE_ARGUMENTS                                                                                   \
    chainner_migx_shape const *inputs, int32_t input_count, chainner_migx_attribute const *attribute_items, \
        int32_t attribute_count, chainner_migx_shape *output, size_t *workspace_bytes, char *error,       \
        size_t error_size
#define COMPUTE_ARGUMENTS                                                                                   \
    void *stream, chainner_migx_tensor const *inputs, int32_t input_count,                                  \
        chainner_migx_attribute const *attribute_items, int32_t attribute_count,                            \
        chainner_migx_tensor const *output, void *workspace, size_t workspace_bytes, char *error,           \
        size_t error_size

int check_count(int32_t input_count, int32_t expected, char* error, size_t error_size) {
    return input_count == expected ? 0 : fail(error, error_size, "takes %d inputs, not %d", expected, input_count);
}

/* Norm and Project: a [..., C] BF16 tensor's width and pixel count. */
int rows_of(chainner_migx_shape const& x, int64_t* channels, int64_t* pixels, char* error, size_t error_size) {
    if (x.type != CHAINNER_MIGX_BF16 || x.rank < 1 || x.dims[x.rank - 1] % 32 != 0 || elements(x) < 1 ||
        elements(x) >= (int64_t{1} << 31)) {
        return fail(error, error_size, "expects a BF16 [..., C] tensor with C a multiple of 32");
    }
    *channels = x.dims[x.rank - 1];
    *pixels = elements(x) / *channels;
    return 0;
}

int norm_shape(SHAPE_ARGUMENTS) {
    (void)attribute_items;
    (void)attribute_count;
    int64_t channels = 0, pixels = 0;
    if (int failed = check_count(input_count, 3, error, error_size)) {
        return failed;
    }
    if (int failed = rows_of(inputs[0], &channels, &pixels, error, error_size)) {
        return failed;
    }
    for (int i = 1; i < 3; ++i) {
        if (inputs[i].type != CHAINNER_MIGX_FLOAT || !same_dims(inputs[i], {channels})) {
            return fail(error, error_size, "expects FP32 gamma and beta of the input's width");
        }
    }
    *output = inputs[0];
    *workspace_bytes = static_cast<size_t>(pixels) * sizeof(float);
    return 0;
}

int norm_compute(COMPUTE_ARGUMENTS) {
    (void)input_count;
    (void)workspace_bytes;
    Attributes attributes{attribute_items, attribute_count};
    int64_t channels = 0, pixels = 0;
    if (int failed = rows_of(inputs[0].shape, &channels, &pixels, error, error_size)) {
        return failed;
    }
    int64_t const rows = attributes.integer("rows");
    if (rows < 1) {
        return fail(error, error_size, "the node carries no tile size (rows)");
    }
    Arguments arguments;
    int failed = arguments.pointer(inputs[0].data, "x", error, error_size);
    failed = failed ? failed : arguments.pointer(inputs[1].data, "gamma", error, error_size);
    failed = failed ? failed : arguments.pointer(inputs[2].data, "beta", error, error_size);
    if (failed) {
        return failed;
    }
    arguments.integer(pixels);
    failed = arguments.pointer(output->data, "y", error, error_size);
    // dual_hip.py compiles sigma's pointer without the alignment promise: scratch memory has none.
    failed = failed ? failed : arguments.pointer(workspace, "sigma", error, error_size, false);
    if (failed) {
        return failed;
    }
    return launch(attributes, stream, arguments, cdiv(pixels, rows), 1, 1, error, error_size);
}

/* Cell, region, apply and NA read the trunk from QKV (input `index`); every other input
 * of theirs is checked against it. */
int core_weight(chainner_migx_shape const& weight, Trunk const& trunk, char* error, size_t error_size) {
    if (weight.type != CHAINNER_MIGX_BF16 || !same_dims(weight, {3, 3, 3 * trunk.channels})) {
        return fail(error, error_size, "expects a BF16 [3,3,3C] depthwise weight");
    }
    return 0;
}

int cell_shape(SHAPE_ARGUMENTS) {
    (void)attribute_items;
    (void)attribute_count;
    (void)workspace_bytes;
    Trunk trunk{};
    if (int failed = check_count(input_count, 2, error, error_size)) {
        return failed;
    }
    if (int failed = trunk_of(inputs[0], &trunk, error, error_size)) {
        return failed;
    }
    if (int failed = core_weight(inputs[1], trunk, error, error_size)) {
        return failed;
    }
    *output = shape_of(CHAINNER_MIGX_FLOAT, {trunk.cells() * trunk.heads() * 1088});
    return 0;
}

int cell_compute(COMPUTE_ARGUMENTS) {
    (void)input_count;
    (void)workspace;
    (void)workspace_bytes;
    Attributes attributes{attribute_items, attribute_count};
    Trunk trunk{};
    if (int failed = trunk_of(inputs[0].shape, &trunk, error, error_size)) {
        return failed;
    }
    Arguments arguments;
    int failed = arguments.pointer(inputs[0].data, "qkv", error, error_size);
    failed = failed ? failed : arguments.pointer(inputs[1].data, "weight", error, error_size);
    if (failed) {
        return failed;
    }
    arguments.integer(trunk.height);
    arguments.integer(trunk.width);
    if (int pointer_failed = arguments.pointer(output->data, "statistics", error, error_size)) {
        return pointer_failed;
    }
    return launch(attributes, stream, arguments, trunk.cells(), trunk.heads(), 1, error, error_size);
}

int region_shape(SHAPE_ARGUMENTS) {
    (void)attribute_items;
    (void)attribute_count;
    (void)workspace_bytes;
    Trunk trunk{};
    if (int failed = check_count(input_count, 3, error, error_size)) {
        return failed;
    }
    if (int failed = trunk_of(inputs[2], &trunk, error, error_size)) {
        return failed;
    }
    if (inputs[0].type != CHAINNER_MIGX_FLOAT || !same_dims(inputs[0], {trunk.cells() * trunk.heads() * 1088})) {
        return fail(error, error_size, "expects the cell statistics of its QKV's trunk");
    }
    if (inputs[1].type != CHAINNER_MIGX_FLOAT || !same_dims(inputs[1], {trunk.heads()})) {
        return fail(error, error_size, "expects FP32 temperatures, one per head");
    }
    *output = shape_of(CHAINNER_MIGX_BF16, {trunk.regions() * trunk.heads() * 1024});
    return 0;
}

int region_compute(COMPUTE_ARGUMENTS) {
    (void)input_count;
    (void)workspace;
    (void)workspace_bytes;
    Attributes attributes{attribute_items, attribute_count};
    Trunk trunk{};
    if (int failed = trunk_of(inputs[2].shape, &trunk, error, error_size)) {
        return failed;
    }
    Arguments arguments;
    int failed = arguments.pointer(inputs[0].data, "statistics", error, error_size);
    failed = failed ? failed : arguments.pointer(inputs[1].data, "tau", error, error_size);
    failed = failed ? failed : arguments.pointer(inputs[2].data, "qkv", error, error_size);
    if (failed) {
        return failed;
    }
    arguments.integer(trunk.height);
    arguments.integer(trunk.width);
    for (int i = 0; i < 3; ++i) {
        arguments.integer(trunk.counts[i]);
    }
    if (int pointer_failed = arguments.pointer(output->data, "matrices", error, error_size)) {
        return pointer_failed;
    }
    return launch(attributes, stream, arguments, trunk.regions(), trunk.heads(), 1, error, error_size);
}

int apply_shape(SHAPE_ARGUMENTS) {
    (void)attribute_items;
    (void)attribute_count;
    (void)workspace_bytes;
    Trunk trunk{};
    if (int failed = check_count(input_count, 3, error, error_size)) {
        return failed;
    }
    if (int failed = trunk_of(inputs[0], &trunk, error, error_size)) {
        return failed;
    }
    if (int failed = core_weight(inputs[1], trunk, error, error_size)) {
        return failed;
    }
    if (inputs[2].type != CHAINNER_MIGX_BF16 || !same_dims(inputs[2], {trunk.regions() * trunk.heads() * 1024})) {
        return fail(error, error_size, "expects the region matrices of its QKV's trunk");
    }
    *output = shape_of(CHAINNER_MIGX_BF16, {1, trunk.height, trunk.width, trunk.channels});
    return 0;
}

int apply_compute(COMPUTE_ARGUMENTS) {
    (void)input_count;
    (void)workspace;
    (void)workspace_bytes;
    Attributes attributes{attribute_items, attribute_count};
    Trunk trunk{};
    if (int failed = trunk_of(inputs[0].shape, &trunk, error, error_size)) {
        return failed;
    }
    int64_t const parts = attributes.integer("parts");
    if (parts < 1) {
        return fail(error, error_size, "the node carries no tile split (parts)");
    }
    Arguments arguments;
    int failed = arguments.pointer(inputs[0].data, "qkv", error, error_size);
    failed = failed ? failed : arguments.pointer(inputs[1].data, "weight", error, error_size);
    failed = failed ? failed : arguments.pointer(inputs[2].data, "matrices", error, error_size);
    if (failed) {
        return failed;
    }
    arguments.integer(trunk.height);
    arguments.integer(trunk.width);
    for (int i = 0; i < 3; ++i) {
        arguments.integer(trunk.counts[i]);
    }
    if (int pointer_failed = arguments.pointer(output->data, "output", error, error_size)) {
        return pointer_failed;
    }
    return launch(attributes, stream, arguments, trunk.cells(), trunk.heads(), parts, error, error_size);
}

int project_shape(SHAPE_ARGUMENTS) {
    (void)attribute_items;
    (void)attribute_count;
    (void)workspace_bytes;
    int64_t channels = 0, pixels = 0;
    if (int failed = check_count(input_count, 3, error, error_size)) {
        return failed;
    }
    if (int failed = rows_of(inputs[0], &channels, &pixels, error, error_size)) {
        return failed;
    }
    if (inputs[1].type != CHAINNER_MIGX_BF16 || !same_dims(inputs[1], {channels, channels})) {
        return fail(error, error_size, "expects a BF16 [C,C] projection weight");
    }
    if (inputs[2].type != CHAINNER_MIGX_BF16 || inputs[2].rank != inputs[0].rank ||
        std::memcmp(inputs[2].dims, inputs[0].dims, sizeof(int64_t) * static_cast<size_t>(inputs[0].rank)) != 0) {
        return fail(error, error_size, "expects a BF16 residual of the input's shape");
    }
    *output = inputs[2];
    return 0;
}

int project_compute(COMPUTE_ARGUMENTS) {
    (void)input_count;
    (void)workspace;
    (void)workspace_bytes;
    Attributes attributes{attribute_items, attribute_count};
    int64_t channels = 0, pixels = 0;
    if (int failed = rows_of(inputs[0].shape, &channels, &pixels, error, error_size)) {
        return failed;
    }
    int64_t const rows = attributes.integer("rows");
    if (rows < 1) {
        return fail(error, error_size, "the node carries no tile size (rows)");
    }
    Arguments arguments;
    int failed = arguments.pointer(inputs[0].data, "x", error, error_size);
    failed = failed ? failed : arguments.pointer(inputs[1].data, "weight", error, error_size);
    failed = failed ? failed : arguments.pointer(inputs[2].data, "residual", error, error_size);
    if (failed) {
        return failed;
    }
    arguments.integer(pixels);
    if (int pointer_failed = arguments.pointer(output->data, "output", error, error_size)) {
        return pointer_failed;
    }
    return launch(attributes, stream, arguments, cdiv(pixels, rows), 1, 1, error, error_size);
}

int na_shape(SHAPE_ARGUMENTS) {
    (void)workspace_bytes;
    Attributes attributes{attribute_items, attribute_count};
    Trunk trunk{};
    if (int failed = check_count(input_count, 3, error, error_size)) {
        return failed;
    }
    if (int failed = trunk_of(inputs[0], &trunk, error, error_size)) {
        return failed;
    }
    int64_t const window = attributes.integer("window");
    if (window < 1) {
        return fail(error, error_size, "the node carries no window");
    }
    for (int i = 1; i < 3; ++i) {
        if (inputs[i].type != CHAINNER_MIGX_BF16 ||
            !same_dims(inputs[i], {trunk.heads(), (window + 15) * (window + 15), 16})) {
            return fail(error, error_size, "expects BF16 [heads,(M+15)^2,16] factor tables for window %lld",
                        static_cast<long long>(window));
        }
    }
    *output = shape_of(CHAINNER_MIGX_BF16, {1, trunk.height, trunk.width, trunk.channels});
    return 0;
}

int na_compute(COMPUTE_ARGUMENTS) {
    (void)input_count;
    (void)workspace;
    (void)workspace_bytes;
    Attributes attributes{attribute_items, attribute_count};
    Trunk trunk{};
    if (int failed = trunk_of(inputs[0].shape, &trunk, error, error_size)) {
        return failed;
    }
    int64_t const parts = attributes.integer("parts");
    if (parts < 1) {
        return fail(error, error_size, "the node carries no tile split (parts)");
    }
    Arguments arguments;
    int failed = arguments.pointer(inputs[0].data, "qkv", error, error_size);
    failed = failed ? failed : arguments.pointer(inputs[1].data, "bq", error, error_size);
    failed = failed ? failed : arguments.pointer(inputs[2].data, "bk", error, error_size);
    if (failed) {
        return failed;
    }
    arguments.integer(trunk.height);
    arguments.integer(trunk.width);
    if (int pointer_failed = arguments.pointer(output->data, "output", error, error_size)) {
        return pointer_failed;
    }
    // dual_na_kernel.grid: 16x16 query tiles, their parts, heads
    return launch(attributes, stream, arguments, trunk.cells(), parts, trunk.heads(), error, error_size);
}

int barrier_shape(SHAPE_ARGUMENTS) {
    (void)attribute_items;
    (void)attribute_count;
    (void)workspace_bytes;
    if (int failed = check_count(input_count, 1, error, error_size)) {
        return failed;
    }
    *output = inputs[0];
    return 0;
}

size_t bytes_of(chainner_migx_shape const& shape) {
    size_t element = 1;
    switch (shape.type) {
        case CHAINNER_MIGX_DOUBLE:
        case CHAINNER_MIGX_INT64:
        case CHAINNER_MIGX_UINT64:
            element = 8;
            break;
        case CHAINNER_MIGX_FLOAT:
        case CHAINNER_MIGX_INT32:
        case CHAINNER_MIGX_UINT32:
            element = 4;
            break;
        case CHAINNER_MIGX_HALF:
        case CHAINNER_MIGX_BF16:
        case CHAINNER_MIGX_INT16:
        case CHAINNER_MIGX_UINT16:
            element = 2;
            break;
        default:
            break;
    }
    return static_cast<size_t>(elements(shape)) * element;
}

int barrier_compute(COMPUTE_ARGUMENTS) {
    (void)input_count;
    (void)attribute_items;
    (void)attribute_count;
    (void)workspace;
    (void)workspace_bytes;
    hipError_t status = hipMemcpyAsync(output->data, inputs[0].data, bytes_of(inputs[0].shape),
                                       hipMemcpyDeviceToDevice, static_cast<hipStream_t>(stream));
    if (status != hipSuccess) {
        return fail(error, error_size, "the barrier's copy failed: %s", hipGetErrorString(status));
    }
    return 0;
}

chainner_migx_op const ops[] = {
    {"chainner_dual_norm", CHAINNER_MIGX_FIXED_TYPES, norm_shape, norm_compute},
    {"chainner_dual_cell", CHAINNER_MIGX_FIXED_TYPES, cell_shape, cell_compute},
    {"chainner_dual_region", CHAINNER_MIGX_FIXED_TYPES, region_shape, region_compute},
    {"chainner_dual_apply", CHAINNER_MIGX_FIXED_TYPES, apply_shape, apply_compute},
    {"chainner_dual_project", CHAINNER_MIGX_FIXED_TYPES, project_shape, project_compute},
    {"chainner_dual_na", CHAINNER_MIGX_FIXED_TYPES, na_shape, na_compute},
    {"chainner_dual_barrier", CHAINNER_MIGX_FIXED_TYPES, barrier_shape, barrier_compute},
};

}  // namespace

extern "C" CHAINNER_MIGX_EXPORT int chainner_migx_ops(uint32_t abi, chainner_migx_op const** table, int32_t* count) {
    if (abi != CHAINNER_MIGX_ABI) {
        return 1;
    }
    *table = ops;
    *count = static_cast<int32_t>(sizeof ops / sizeof ops[0]);
    return 0;
}
