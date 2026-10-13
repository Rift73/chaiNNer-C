"""TensorRT builder settings shared by the BF16 and FP8 VOSR stage compilers."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from .vosr2_trt_models import profiles

if TYPE_CHECKING:
    import tensorrt as trt

# A build worker's progress line in its build.log, which compile_worker reports.
STAGE_LINE = re.compile(r"Stage (\d+) of (\d+): (.+)")


def report_stage(index: int, total: int, task: str) -> None:
    print(f"Stage {index} of {total}: {task}", flush=True)


def timing_cache(folder: Path, device: torch.device) -> Path:
    """The file in an engine cache folder that keeps TensorRT's tactic timings for this
    GPU model and TensorRT version: later builds there reuse them."""
    import tensorrt as trt

    properties = torch.cuda.get_device_properties(device)
    key = {
        "gpu": properties.name,
        "capability": [properties.major, properties.minor],
        "tensorrt": trt.__version__,
    }
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
    return folder / f"timing-{digest[:16]}.cache"


def use_timing_cache(config: trt.IBuilderConfig, path: Path) -> trt.ITimingCache:
    """Attach the tactic timings measured by earlier builds (timing_cache). TensorRT
    replaces a damaged file's content with an empty cache; one recorded on another GPU
    or TensorRT version is refused and a new cache starts."""
    cache = config.create_timing_cache(path.read_bytes() if path.is_file() else b"")
    if not config.set_timing_cache(cache, ignore_mismatch=False):
        cache = config.create_timing_cache(b"")
        if not config.set_timing_cache(cache, ignore_mismatch=False):
            raise RuntimeError("TensorRT refused an empty timing cache.")
    return cache


def store_timing_cache(cache: trt.ITimingCache, path: Path) -> None:
    """Replace the file whole; the caller's cache lock keeps other builds out."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(bytes(cache.serialize()))
    os.replace(temporary, path)


def compile_stage(
    name: str, graph: Path, plan: Path, logger: trt.ILogger, timing: Path
) -> None:
    """Strongly typed: the graph's own types and Q/DQ pairs decide every precision."""
    import tensorrt as trt

    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    reader = trt.OnnxParser(network, logger)
    if not reader.parse_from_file(str(graph)):
        raise RuntimeError(
            "\n".join(reader.get_error(i).desc() for i in range(reader.num_errors))
        )
    config = builder.create_builder_config()
    cache = use_timing_cache(config, timing)
    workspace = (
        12288
        if name in ("encoder_pre", "decoder_post")
        else 256
        if name == "dit_kv"
        else 2048
    ) * 1024**2
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
    config.builder_optimization_level = 3
    config.max_aux_streams = 0
    if name == "dino":
        config.set_flag(trt.BuilderFlag.TF32)
    else:
        config.clear_flag(trt.BuilderFlag.TF32)
    if name != "attention":
        config.set_flag(trt.BuilderFlag.WEIGHT_STREAMING)
    profile = builder.create_optimization_profile()
    for key, shapes in profiles(name).items():
        profile.set_shape(key, shapes[0], shapes[1], shapes[2])
    config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"VOSR {name} TensorRT build failed.")
    plan.write_bytes(bytes(serialized))
    # Kept after each stage, so a stopped or failed build keeps what it measured.
    store_timing_cache(cache, timing)
