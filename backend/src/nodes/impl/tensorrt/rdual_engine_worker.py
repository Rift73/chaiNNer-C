"""Build an RDUAL model's TensorRT engines for one input size in a child process, from
rdual_export.py's parts: each part's plugin nodes replaced by AOT Python plugins
(rdual_aot.py, and DUAL's through dual_aot.py), compiled by Triton for this GPU and
embedded in its engine. One builder and one timing cache serve every part, so the later
builds reuse the tactics the earlier ones timed. The builder policy is DUAL's
(dual_engine_worker.builder_config) with Build Engine's settings; the guide's are
8, 3, 0, 0 and none.

    python -m nodes.impl.tensorrt.rdual_engine_worker EXPORT ENGINES GPU_INDEX
        WORKSPACE_GIB LEVEL TF32 SPARSE TILING [TIMING_CACHE]

EXPORT is rdual_export.py's output folder; each part's engine goes to
ENGINES/<part>.engine. TIMING_CACHE, a file, is read before the builds when it exists
and written after them (engine_builder.write_timing_cache), as Build Engine keeps its
own: a later set of the same size reuses the tactics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ..pytorch.vosr2_process import end_with_parent

# The part without DUAL's plugins
FLOW = "flow"


def main(
    export: str,
    engines: str,
    gpu_index: str,
    workspace_gib: str,
    level: str,
    tf32: str,
    sparse: str,
    tiling: str,
    timing_cache_path: str = "",
) -> None:
    import onnx
    import tensorrt as trt
    from cuda.bindings import runtime as cudart

    from . import dual_aot, rdual_aot
    from .dual_engine_worker import builder_config
    from .engine_builder import write_timing_cache

    (error,) = cudart.cudaSetDevice(int(gpu_index))
    if error != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaSetDevice({gpu_index}) failed: {error}")
    source = Path(export)
    manifest = json.loads((source / "export.json").read_text(encoding="utf-8"))
    target = Path(engines)
    target.mkdir(parents=True, exist_ok=True)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    timing_cache = None
    if timing_cache_path and Path(timing_cache_path).is_file():
        probe = builder.create_builder_config()
        timing_cache = probe.create_timing_cache(Path(timing_cache_path).read_bytes())
    for part in manifest["parts"]:
        key = None if part == FLOW else manifest["plugin_key"]
        model = onnx.load(str(source / part / "model.onnx"))
        rdual_aot.register(model)
        if key is not None:
            dual_aot.register(key)
        lowered = rdual_aot.lower(model, key)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        )
        parser = trt.OnnxParser(network, logger)
        if not parser.parse(lowered.model.SerializeToString()):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError(f"ONNX parse of {part} failed:\n" + "\n".join(errors))
        dual_aot.attach(network, lowered)
        config = builder_config(
            builder, workspace_gib, level, tf32, sparse, tiling, timing_cache
        )
        timing_cache = config.get_timing_cache()
        engine = builder.build_serialized_network(network, config)
        if engine is None:
            raise RuntimeError(f"TensorRT failed to build RDUAL's {part} engine")
        (target / f"{part}.engine").write_bytes(engine)
    if timing_cache_path and timing_cache is not None:
        write_timing_cache(bytes(timing_cache.serialize()), timing_cache_path)


if __name__ == "__main__":
    end_with_parent()
    main(*sys.argv[1:])
