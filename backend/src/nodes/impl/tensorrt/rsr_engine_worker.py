"""Build an RSR model's TensorRT engines for one input size in a child process, from
rsr_export.py's parts. No plugins: the parts are plain ONNX. One builder and one timing
cache serve the three parts, so the later builds reuse the tactics the earlier ones
timed. The networks are strongly typed (the export's FP16 and FP32), with DUAL's
builder policy (dual_engine_worker.builder_config) and Build Engine's settings; the
guide's are 8, 3, 0, 0 and none. Every handoff binding (each one but the image `x`,
`output` and the validity flags) whose channels are a multiple of 8 is channel-last
(HWC8), the layout the parts compute in, so no part reorders what another wrote
(channel_last).

    python -m nodes.impl.tensorrt.rsr_engine_worker EXPORT ENGINES GPU_INDEX
        WORKSPACE_GIB LEVEL TF32 SPARSE TILING [TIMING_CACHE]

EXPORT is rsr_export.py's output folder; each part's engine goes to
ENGINES/<part>.engine. TIMING_CACHE, a file, is read before the builds when it exists
and written after them (engine_builder.write_timing_cache)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from ..pytorch.vosr2_process import end_with_parent

if TYPE_CHECKING:
    import tensorrt as trt


def channel_last(network: trt.INetworkDefinition) -> list[str]:
    """Binds `network`'s handoff tensors channel-last (HWC8) where their channels are a
    multiple of 8; returns their names. HWC8 pads the channels to a multiple of 8 in
    memory, while the session sizes each buffer by its N,C,H,W shape (Part.tensor), so a
    tensor of other channels stays linear. The rule reads only the tensor, so every
    part of a set gives a handoff the same layout."""
    import tensorrt as trt

    from .rsr_export import IMAGE_BINDINGS, VALIDITY_BINDINGS

    linear = (*IMAGE_BINDINGS, *VALIDITY_BINDINGS)
    bindings = [network.get_input(i) for i in range(network.num_inputs)]
    bindings += [network.get_output(i) for i in range(network.num_outputs)]
    names = []
    for tensor in bindings:
        if tensor.name not in linear and tensor.shape[1] % 8 == 0:
            tensor.allowed_formats = 1 << int(trt.TensorFormat.HWC8)
            names.append(tensor.name)
    return names


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
    import tensorrt as trt
    from cuda.bindings import runtime as cudart

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
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        )
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(source / part / "model.onnx")):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError(f"ONNX parse of {part} failed:\n" + "\n".join(errors))
        channel_last(network)
        config = builder_config(
            builder, workspace_gib, level, tf32, sparse, tiling, timing_cache
        )
        timing_cache = config.get_timing_cache()
        engine = builder.build_serialized_network(network, config)
        if engine is None:
            raise RuntimeError(f"TensorRT failed to build RSR's {part} engine")
        (target / f"{part}.engine").write_bytes(engine)
    if timing_cache_path and timing_cache is not None:
        write_timing_cache(bytes(timing_cache.serialize()), timing_cache_path)


if __name__ == "__main__":
    end_with_parent()
    main(*sys.argv[1:])
