/* The contract between chaiNNer-C's MIGraphX runner (_chainner_migraphx) and a custom-op library
 * (native/src/<model>_hip/ builds <model>_migx_ops.so or .dll). The library needs only this header
 * and HIP: it never links MIGraphX.
 *
 * Loading. _chainner_migraphx.load_op_library(path) opens the library and calls its exported
 * chainner_migx_ops(CHAINNER_MIGX_ABI, &ops, &count) once; the runner registers each returned op
 * as a MIGraphX operation under the op's name, which is "chainner_<model>_<op>". The table and its
 * strings stay valid while the library is loaded (it is never unloaded). Loading a library twice is
 * one load; a name registered by another library is an error.
 *
 * ONNX. A node of domain "chainner" whose op_type is a registered op's name reaches that op:
 * compile_onnx parses the model with those nodes as placeholders and puts the op in each one's place
 * once the shapes before it are known (MIGraphX's ONNX parser has no custom-op lookup). The node has
 * exactly one output and no empty optional inputs; its inputs reach the op in the node's order and
 * its attributes as chainner_migx_attribute values in the node's order. A chainner node whose op is
 * not loaded fails the compile with a message naming it. Saved programs keep each node's op name and
 * attributes: load_program needs the same libraries loaded first.
 *
 * Calls. compute_shape runs while a program is compiled (and loaded), compute each time the program
 * runs; both may run on several threads at once, for different programs and devices. Neither may
 * throw: each returns 0, or writes a message (NUL-terminated, truncated to error_size) and returns
 * another value, which fails the compile or the run with that message. */
#ifndef CHAINNER_MIGRAPHX_CUSTOM_OP_H
#define CHAINNER_MIGRAPHX_CUSTOM_OP_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(__HIP_DEVICE_COMPILE__)
/* The device pass of a HIP source on Windows sees the host's declarations; a GPU target has no
 * DLL exports. */
#define CHAINNER_MIGX_EXPORT
#elif defined(_WIN32)
#define CHAINNER_MIGX_EXPORT __declspec(dllexport)
#else
#define CHAINNER_MIGX_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* This header's version: chainner_migx_ops refuses any other. */
#define CHAINNER_MIGX_ABI 1
#define CHAINNER_MIGX_MAX_RANK 8

/* Element types (MIGraphX's shape types). FP8: CDNA3 (gfx942) computes the fnuz formats, RDNA4 and
 * CDNA4 the OCP ones (e4m3fn, e5m2). */
typedef enum chainner_migx_type {
    CHAINNER_MIGX_FLOAT = 0,
    CHAINNER_MIGX_HALF = 1,
    CHAINNER_MIGX_BF16 = 2,
    CHAINNER_MIGX_DOUBLE = 3,
    CHAINNER_MIGX_INT8 = 4,
    CHAINNER_MIGX_UINT8 = 5,
    CHAINNER_MIGX_INT16 = 6,
    CHAINNER_MIGX_UINT16 = 7,
    CHAINNER_MIGX_INT32 = 8,
    CHAINNER_MIGX_UINT32 = 9,
    CHAINNER_MIGX_INT64 = 10,
    CHAINNER_MIGX_UINT64 = 11,
    CHAINNER_MIGX_BOOL = 12,
    CHAINNER_MIGX_FP8E4M3FNUZ = 13,
    CHAINNER_MIGX_FP8E4M3FN = 14,
    CHAINNER_MIGX_FP8E5M2 = 15,
    CHAINNER_MIGX_FP8E5M2FNUZ = 16
} chainner_migx_type;

/* A tensor's element type and dimensions. Every tensor an op sees is packed row-major (standard):
 * the runner makes inputs contiguous before the op and allocates the output packed. */
typedef struct chainner_migx_shape {
    int32_t type; /* chainner_migx_type */
    int32_t rank; /* 0 to CHAINNER_MIGX_MAX_RANK */
    int64_t dims[CHAINNER_MIGX_MAX_RANK];
} chainner_migx_shape;

typedef struct chainner_migx_tensor {
    void* data; /* device memory, or host memory for a CHAINNER_MIGX_HOST op */
    chainner_migx_shape shape;
} chainner_migx_tensor;

/* ONNX attribute types an op can take. */
typedef enum chainner_migx_attribute_kind {
    CHAINNER_MIGX_ATTRIBUTE_INT = 0,    /* ints[0] */
    CHAINNER_MIGX_ATTRIBUTE_FLOAT = 1,  /* floats[0] */
    CHAINNER_MIGX_ATTRIBUTE_STRING = 2, /* bytes[0..count), binary-safe, followed by a NUL */
    CHAINNER_MIGX_ATTRIBUTE_INTS = 3,   /* ints[0..count) */
    CHAINNER_MIGX_ATTRIBUTE_FLOATS = 4  /* floats[0..count) */
} chainner_migx_attribute_kind;

/* One ONNX attribute of the node; the pointers not of its kind are NULL. Valid for the call only:
 * copy what must outlive it. They do not identify a node (copies of a program have their own). */
typedef struct chainner_migx_attribute {
    char const* name;
    int32_t kind; /* chainner_migx_attribute_kind */
    int64_t count; /* 1 for INT and FLOAT; elements for INTS and FLOATS; bytes for STRING */
    int64_t const* ints;
    float const* floats;
    char const* bytes;
} chainner_migx_attribute;

/* The op computes on the CPU from host memory (tests, and MIGraphX's ref target); on the GPU target
 * MIGraphX copies its inputs to the host and its output back around it. Without this flag the op
 * computes on the GPU: device pointers, enqueued on the stream it is given. */
#define CHAINNER_MIGX_HOST 1u
/* The op keeps the ONNX model's input types when the program is compiled with fp16 or fp8: the
 * runner leaves it out of quantization. Without this flag, after fp16 quantization float inputs
 * arrive as half, and compute_shape refuses the types it does not take. */
#define CHAINNER_MIGX_FIXED_TYPES 2u

typedef struct chainner_migx_op {
    char const* name; /* "chainner_<model>_<op>", the ONNX op_type */
    uint32_t flags;   /* CHAINNER_MIGX_HOST, CHAINNER_MIGX_FIXED_TYPES */
    /* The output's shape from the inputs' shapes and the node's attributes, and the bytes of scratch
     * memory compute needs (*workspace_bytes is 0 on entry). Called again whenever MIGraphX changes
     * the inputs' types or shapes, so it depends on its arguments only. */
    int (*compute_shape)(chainner_migx_shape const* inputs, int32_t input_count,
                         chainner_migx_attribute const* attributes, int32_t attribute_count,
                         chainner_migx_shape* output, size_t* workspace_bytes, char* error, size_t error_size);
    /* Computes the output into output->data (allocated by MIGraphX, of compute_shape's shape).
     * stream: the program's hipStream_t (NULL for a CHAINNER_MIGX_HOST op); the calling thread's
     * current HIP device is the program's. A GPU op only enqueues work on stream: no host
     * synchronization, no allocation (workspace: workspace_bytes of device memory for this call
     * only, its contents undefined, NULL when 0). It may keep per-device state (for example a loaded
     * code object) keyed by attribute contents, guarded for concurrent calls. */
    int (*compute)(void* stream, chainner_migx_tensor const* inputs, int32_t input_count,
                   chainner_migx_attribute const* attributes, int32_t attribute_count,
                   chainner_migx_tensor const* output, void* workspace, size_t workspace_bytes, char* error,
                   size_t error_size);
} chainner_migx_op;

/* The library's one export: its ops when abi is CHAINNER_MIGX_ABI (then 0), else a non-zero value. */
typedef int (*chainner_migx_ops_function)(uint32_t abi, chainner_migx_op const** ops, int32_t* count);
CHAINNER_MIGX_EXPORT int chainner_migx_ops(uint32_t abi, chainner_migx_op const** ops, int32_t* count);

#ifdef __cplusplus
}
#endif

#endif
