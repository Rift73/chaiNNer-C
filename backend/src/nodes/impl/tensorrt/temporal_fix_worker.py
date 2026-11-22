"""Build a Temporal Fix engine set's engines for one input size in a child process, from
temporal_fix_split.py's parts, as rsr_engine_worker.py builds RSR's: one builder and one
timing cache for every part, strongly typed networks, DUAL's builder policy
(dual_engine_worker.builder_config) with Build Engine's settings, and the handoff
bindings channel-last by RSR's rule (rsr_engine_worker.channel_last). Its own module, so
that a change to Temporal Fix's build changes no other model's set key.

The parts get their ahead-of-time plugins (tfix_aot.lower) where Triton is installed;
without it they are built as exported. With them the window part is cut at its directly
launched kernels (tfix_cut.py): its pieces go to ENGINES/window_<i>.engine, the layout
and the kernels' cubins beside them; a piece's bindings are linear, but for each input a
kernel writes, which is HWC8.

    python -m nodes.impl.tensorrt.temporal_fix_worker EXPORT ENGINES GPU_INDEX
        WORKSPACE_GIB LEVEL TF32 SPARSE TILING [TIMING_CACHE]

EXPORT is temporal_fix_split.py's output folder; each engine goes to
ENGINES/<name>.engine. TIMING_CACHE, a file, is read before the builds when it exists
and written after them (engine_builder.write_timing_cache)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from ..pytorch.vosr2_process import end_with_parent

if TYPE_CHECKING:
    import tensorrt as trt

    from .dual_aot import Lowered


def lower(model: Path) -> Lowered | None:
    """The part's graph with its plugins (tfix_aot.lower, registered), when Triton can
    compile them; None without Triton."""
    from .temporal_fix import plugins

    tfix_aot = plugins()
    if tfix_aot is None:
        print(f"No Triton: {model.parent.name} is built without its plugins")
        return None
    import onnx

    lowered = tfix_aot.lower(onnx.load(str(model)))
    tfix_aot.register(lowered)
    return lowered


def parse(
    network: trt.INetworkDefinition,
    logger: trt.ILogger,
    model: Path,
    lowered: Lowered | None,
) -> None:
    """`model` into `network`, or its lowered form with the plugins attached."""
    import tensorrt as trt

    parser = trt.OnnxParser(network, logger)
    if lowered is None:
        parsed = parser.parse_from_file(str(model))
    else:
        parsed = parser.parse(lowered.model.SerializeToString())
    if not parsed:
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError(f"ONNX parse of {model} failed:\n" + "\n".join(errors))
    if lowered is not None and lowered.plugins:
        from . import dual_aot

        dual_aot.attach(network, lowered)


def pieces(
    part: str, lowered: Lowered | None, target: Path, shapes: dict[str, list[int]]
) -> list[tuple[str, Lowered | None, list[str]]]:
    """The engines of a part: (name, lowered graph, the inputs a kernel writes). The
    window part cut at its direct kernels (tfix_cut.cut) is its pieces, whose layout and
    cubins are written to `target` and its kernels' output shapes put in `shapes`; any
    other part is itself."""
    if part != "window" or lowered is None:
        return [(part, lowered, [])]
    from . import dual_aot, tfix_cut

    found = tfix_cut.cut(lowered)
    if found is None:
        return [(part, tfix_cut.whole(lowered), [])]
    shapes.update(tfix_cut.row_shapes(found.kernels))
    launches, cubins = tfix_cut.compile_kernels(found.kernels, dual_aot.device_sm())
    for name, data in cubins.items():
        (target / name).write_bytes(data)
    (target / tfix_cut.LAYOUT).write_bytes(tfix_cut.layout(found, launches))
    return [(p.name, p.lowered, p.rows) for p in found.pieces]


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
    from .rsr_engine_worker import channel_last
    from .tfix_cut import shape_inputs

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
    # the shapes the pieces built so far give (a later piece's inputs)
    shapes: dict[str, list[int]] = {}
    for part in manifest["parts"]:
        model = source / part / "model.onnx"
        for name, lowered, rows in pieces(part, lower(model), target, shapes):
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
            )
            if lowered is not None:
                shape_inputs(lowered.model, shapes)
            parse(network, logger, model, lowered)
            for i in range(network.num_outputs):
                output = network.get_output(i)
                shapes[output.name] = [int(d) for d in output.shape]
            if name == part:
                channel_last(network)
            else:
                # A cut window's piece: linear everywhere (the direct kernels read and
                # write plain planes) but for the inputs a kernel writes as HWC8 rows
                for i in range(network.num_inputs):
                    if network.get_input(i).name in rows:
                        network.get_input(i).allowed_formats = 1 << int(
                            trt.TensorFormat.HWC8
                        )
            config = builder_config(
                builder, workspace_gib, level, tf32, sparse, tiling, timing_cache
            )
            timing_cache = config.get_timing_cache()
            engine = builder.build_serialized_network(network, config)
            if engine is None:
                raise RuntimeError(f"TensorRT failed to build Temporal Fix's {name}")
            (target / f"{name}.engine").write_bytes(engine)
    if timing_cache_path and timing_cache is not None:
        write_timing_cache(bytes(timing_cache.serialize()), timing_cache_path)


if __name__ == "__main__":
    end_with_parent()
    main(*sys.argv[1:])
