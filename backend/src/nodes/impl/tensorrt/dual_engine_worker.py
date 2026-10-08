"""Build a DUAL TensorRT engine in a child process (dual.py runs it from backend/src):
the guide's ONNX with its plugin nodes replaced by AOT Python plugins (dual_aot.py),
compiled by Triton for this GPU and embedded in the engine. The builder policy is the
guide's trtexec command (vendor/dual_tensorrt/scripts/dual_tensorrt/build.py, engine()):
strongly typed, TF32 off, optimization level 3, no auxiliary streams, an 8 GiB
workspace, detailed profiling verbosity and a fresh timing cache. A width-only key
(c<C>) is a dynamic-shape ONNX: its engine gets one optimization profile, minimum,
optimum and maximum input sizes given as HxW.

    python -m nodes.impl.tensorrt.dual_engine_worker ONNX ENGINE GPU_INDEX PLUGIN_KEY
        [MIN OPT MAX]
"""

from __future__ import annotations

import sys


def main(
    onnx_path: str, engine_path: str, gpu_index: str, key: str, *profile: str
) -> None:
    import onnx
    import tensorrt as trt
    from cuda.bindings import runtime as cudart

    from . import dual_aot

    dynamic = "_h" not in key
    if len(profile) != (3 if dynamic else 0):
        raise ValueError(
            "A dynamic DUAL ONNX needs MIN OPT MAX sizes; a fixed one none"
        )
    (error,) = cudart.cudaSetDevice(int(gpu_index))
    if error != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaSetDevice({gpu_index}) failed: {error}")
    lowered = dual_aot.lower(onnx.load(onnx_path), key)
    if dynamic:
        dual_aot.register_dynamic(int(key[1:]))
    else:
        dual_aot.register(key)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(lowered.model.SerializeToString()):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("ONNX parse failed:\n" + "\n".join(errors))
    dual_aot.attach(network, lowered)
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.builder_optimization_level = 3
    config.max_aux_streams = 0
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.set_timing_cache(config.create_timing_cache(b""), ignore_mismatch=False)
    if dynamic:
        sizes = [tuple(int(v) for v in size.split("x")) for size in profile]
        optimization = builder.create_optimization_profile()
        optimization.set_shape(
            network.get_input(0).name, *((1, 3, h, w) for h, w in sizes)
        )
        config.add_optimization_profile(optimization)
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("TensorRT failed to build the DUAL engine")
    with open(engine_path, "wb") as file:
        file.write(engine)


if __name__ == "__main__":
    main(*sys.argv[1:])
