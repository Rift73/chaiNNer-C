"""Build a legacy DRFT TensorRT engine in a child process (drft.py runs it from
backend/src): the ONNX's plugin nodes come from chaiNNer-C's plugin library, loaded into
this process's plugin registry. The engine does not carry the library (TensorRT would
register it again at every load and refuse the second); processes that load the engine
register it once (drft.load_plugins). The builder policy is the reference trtexec build of DRFT-L (strongly
typed, no auxiliary streams, detailed profiling verbosity, a fresh timing cache) with
Build Engine's workspace (GiB; 0 is TensorRT's default), optimization level, TF32 (1/0),
sparse weights (1/0) and tiling optimization (none, fast, moderate or full); the reference values are
0, 3, 0, 1 and none.
The engine has one optimization profile, minimum, optimum and maximum input sizes
given as HxW, and its input is named for the ONNX's input alignment
(dual.aligned_input_name), which marks the engine as mixed precision when it is loaded
and which Upscale Image's tiler pads to.

    python -m nodes.impl.tensorrt.drft_engine_worker ONNX ENGINE GPU_INDEX LIBRARY
        WORKSPACE_GIB LEVEL TF32 SPARSE TILING MIN OPT MAX ALIGNMENT
"""

from __future__ import annotations

import sys


def main(
    onnx_path: str,
    engine_path: str,
    gpu_index: str,
    library: str,
    workspace_gib: str,
    level: str,
    tf32: str,
    sparse: str,
    tiling: str,
    low: str,
    opt: str,
    high: str,
    alignment: str,
) -> None:
    import tensorrt as trt
    from cuda.bindings import runtime as cudart

    from . import dual

    (error,) = cudart.cudaSetDevice(int(gpu_index))
    if error != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaSetDevice({gpu_index}) failed: {error}")
    # The plugins' creators, which the ONNX parser finds by the nodes' op types
    if trt.get_plugin_registry().load_library(library) is None:
        raise RuntimeError(f"TensorRT could not load the DRFT plugin library {library}")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_path):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("ONNX parse failed:\n" + "\n".join(errors))
    config = builder.create_builder_config()
    if tf32 != "1":
        config.clear_flag(trt.BuilderFlag.TF32)
    if sparse == "1":
        config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
    config.builder_optimization_level = int(level)
    config.tiling_optimization_level = getattr(
        trt.TilingOptimizationLevel, tiling.upper()
    )
    config.max_aux_streams = 0
    if float(workspace_gib) > 0:
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, int(float(workspace_gib) * (1 << 30))
        )
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.set_timing_cache(config.create_timing_cache(b""), ignore_mismatch=False)
    image = network.get_input(0)
    image.name = dual.aligned_input_name(int(alignment))
    sizes = [tuple(int(v) for v in size.split("x")) for size in (low, opt, high)]
    optimization = builder.create_optimization_profile()
    optimization.set_shape(image.name, *((1, 3, h, w) for h, w in sizes))
    config.add_optimization_profile(optimization)
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("TensorRT failed to build the DRFT engine")
    with open(engine_path, "wb") as file:
        file.write(engine)


if __name__ == "__main__":
    main(*sys.argv[1:])
