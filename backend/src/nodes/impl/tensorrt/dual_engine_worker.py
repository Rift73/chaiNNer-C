"""Build a DUAL TensorRT engine in a child process (run as a script by dual.py), so the
chaiNNer worker never loads the plugin libraries. The builder policy is the guide's
trtexec command (vendor/dual_tensorrt/scripts/dual_tensorrt/build.py, engine()): strongly
typed, TF32 off, optimization level 3, no auxiliary streams, an 8 GiB workspace, detailed
profiling verbosity and a fresh timing cache; the plugin libraries are serialized into
the engine.

    python dual_engine_worker.py ONNX ENGINE GPU_INDEX LIBRARY...
"""

from __future__ import annotations

import sys


def main(
    onnx_path: str, engine_path: str, gpu_index: str, libraries: list[str]
) -> None:
    import tensorrt as trt
    from cuda.bindings import runtime as cudart

    (error,) = cudart.cudaSetDevice(int(gpu_index))
    if error != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaSetDevice({gpu_index}) failed: {error}")
    logger = trt.Logger(trt.Logger.WARNING)
    registry = trt.get_plugin_registry()
    for library in libraries:
        if registry.load_library(library) is None:
            raise RuntimeError(f"TensorRT could not load the plugin library {library}")
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_path):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("ONNX parse failed:\n" + "\n".join(errors))
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.builder_optimization_level = 3
    config.max_aux_streams = 0
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.set_timing_cache(config.create_timing_cache(b""), ignore_mismatch=False)
    config.plugins_to_serialize = libraries
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("TensorRT failed to build the DUAL engine")
    with open(engine_path, "wb") as file:
        file.write(engine)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
