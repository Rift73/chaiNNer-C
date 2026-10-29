"""VOSR 2.0 on AMD GPUs through MIGraphX (Linux, Windows): the TensorRT route's counterpart.

DINO and the DiT's two stages run as MIGraphX programs compiled from the ONNX graphs the
BF16 TensorRT engines are built from (this code's export), with the plugin's Q/K/V packing
as the custom op chainner_vosr2_qkv (vosr2_migx_compile). The reduced set (MIGraphX FP8)
has MIGraphX quantize the products TensorRT's FP8 recipe quantizes, to the GPU's FP8
format, calibrated on synthetic images, on GPUs with FP8 matrix math; elsewhere it is the
BF16 set, or FP16 where BF16 math is slow (reduced_precision). MIGraphX 2.16 parses no OCP
FP8, so the hosted TensorRT FP8 graphs are not used. The VAE stages, whose shapes follow the image while
MIGraphX compiles fixed shapes, run as the export's own BF16 modules in PyTorch on ROCm. A
session is the TensorRT FullSession with these stages, so restoration
(vosr2_trt_restore) is the same code. MIGraphX cannot stream weights: GPU Memory Low runs
on PyTorch. Program sets are cached like the engine sets.
"""

from __future__ import annotations

import atexit
import json
import sys
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from importlib import metadata
from pathlib import Path
from typing import Any, cast
from weakref import WeakSet

import torch
from torch import nn

from api import NodeContext
from gpu import AmdDevice, amd, get_vram_usage
from vendor.vosr2.qwenimage_vae2d import AttentionBlock2D, AutoencoderKLQwenImage2D

from ..pytorch import memory_cap
from ..pytorch.vosr2 import BundlePaths, load_weights, read_configs
from ..pytorch.vosr2_engine_cache import (
    architecture,
    cache_lock,
    compile_worker,
    ensure_engine_set,
    file_hash,
    remove_unused_sets,
)
from ..pytorch.vosr2_official import FILES
from ..pytorch.vosr2_plan_file import plain
from ..pytorch.vosr2_trt_execution import StageExecution
from ..pytorch.vosr2_trt_models import (
    AttentionCore,
    AttentionProjection,
    DecoderPost,
    DecoderPre,
    EncoderPost,
    EncoderPre,
    mixed,
)
from ..pytorch.vosr2_trt_pipeline import FullSession, lease_session
from .runner import runner_built
from .vosr2_migx_compile import (
    PROGRAMS,
    load_ops,
    ops_path,
    program_inputs,
    program_name,
)

# MIGraphX's type names (_chainner_migraphx) of the programs' inputs and outputs
TYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
# A set's build: export, calibration, four programs (each minutes on a large GPU) and
# their check
COMPILE_SECONDS = 3600

_LIVE: WeakSet[MigraphxSession] = WeakSet()


class ProgramStages:
    """A set's MIGraphX programs, loaded on one device. `run` returns a buffer of
    `execution`, borrowed until the program runs again; it binds inputs of the program's
    type and layout as they are and copies others into buffers of `execution`."""

    def __init__(
        self, directory: Path, device: torch.device, execution: StageExecution
    ):
        migraphx = load_ops()
        self.device = device
        self.execution = execution
        self.programs: dict[str, Any] = {}
        try:
            for name in PROGRAMS:
                self.programs[name] = migraphx.load_program(
                    str(plain(directory, name)), device.index
                )
        except BaseException:
            self.close()
            raise

    def run(
        self,
        stage: str,
        arguments: Sequence[torch.Tensor],
        stream: torch.cuda.Stream,
    ) -> torch.Tensor:
        name = program_name(stage, arguments[0].shape[-1])
        program = self.programs[name]
        expected = program_inputs(name)
        names = list(program.input_names)
        if len(arguments) != len(expected) or set(names) != set(expected):
            raise ValueError(f"Invalid VOSR MIGraphX {name} program interface.")
        types = dict(zip(names, program.input_types, strict=True))
        pointers = {}
        for binding, tensor in zip(expected, arguments, strict=True):
            shape, dtype = expected[binding], TYPES[types[binding]]
            if tuple(tensor.shape) != shape:
                raise ValueError(
                    f"VOSR MIGraphX {name} takes {binding} {shape}, "
                    f"not {tuple(tensor.shape)}."
                )
            if (
                tensor.device == self.device
                and tensor.dtype == dtype
                and tensor.is_contiguous()
            ):
                tensor.record_stream(stream)
                value = tensor
            else:
                value = self.execution.buffer((name, binding), shape, dtype)
                value.copy_(tensor)
            pointers[binding] = value.data_ptr()
        (shape,) = program.output_shapes
        (output_type,) = program.output_types
        result = self.execution.buffer(
            (name, "result"), tuple(shape), TYPES[output_type]
        )
        program.run(pointers, [result.data_ptr()], stream.cuda_stream)
        return result

    def close(self) -> None:
        for program in self.programs.values():
            program.close()
        self.programs.clear()


def vae_stages(paths: BundlePaths) -> dict[str, nn.Module]:
    """The VAE stages of the TensorRT export (vosr2_trt_models.export_model) as PyTorch
    BF16 modules on the CPU, made from the VAE's weights alone and sharing one copy of
    them (export_model makes a copy per stage)."""
    _, config = read_configs(paths)
    with torch.inference_mode(False), torch.device("meta"):
        vae = AutoencoderKLQwenImage2D.from_config_dict(config)
    load_weights(vae, paths.vae, torch.float32)
    vae = mixed(vae, torch.bfloat16)
    stages: dict[str, nn.Module] = {
        "encoder_pre": EncoderPre(vae),
        "encoder_qkv": AttentionProjection(
            cast(AttentionBlock2D, vae.encoder.mid_block.attentions[0])
        ),
        "encoder_post": EncoderPost(vae),
        "decoder_pre": DecoderPre(vae),
        "decoder_qkv": AttentionProjection(
            cast(AttentionBlock2D, vae.decoder.mid_block.attentions[0])
        ),
        "decoder_post": DecoderPost(vae),
        "attention": AttentionCore(),
    }
    return {name: module.eval() for name, module in stages.items()}


class MigraphxSession(FullSession):
    """FullSession on an AMD GPU: DINO and the DiT as a set's MIGraphX programs, the VAE
    stages as PyTorch BF16 modules. Activation loads the programs and moves the modules to
    the GPU; offloading closes the programs (the next activation loads them again) and
    returns the modules to the CPU."""

    def __init__(self, directory: Path, paths: BundlePaths, device: torch.device):
        load_ops()
        with torch.cuda.device(device):
            self.device = torch.device("cuda", torch.cuda.current_device())
            self.stream = torch.cuda.Stream(device=self.device)
        self.directory = directory
        self.stages = {}
        self.ropes = {}
        self.budget = 0
        self.reserve = None
        self.execution = StageExecution(self.device, self.stream)
        self.geometry = None
        self.constants = {}
        self.kv_tensor = None
        self.modules = vae_stages(paths)
        self.programs: ProgramStages | None = None
        _LIVE.add(self)

    def used_memory(self) -> int:
        usage = get_vram_usage(self.device.index)
        if usage is None:
            raise RuntimeError("VOSR sessions require working GPU memory monitoring.")
        return usage.used

    def activate(self, check: Callable[[], None], reserve: int | None = None) -> None:
        """Load the programs, then lower PyTorch's share of the device by the memory
        they took (outside PyTorch's allocator), as FullSession does for its weights."""
        check()
        if reserve is not None:
            raise ValueError(
                "VOSR MIGraphX keeps every weight on the GPU: GPU Memory Low runs on "
                "PyTorch."
            )
        if self.programs is not None:
            return
        before = self.used_memory()
        self.programs = ProgramStages(self.directory, self.device, self.execution)
        try:
            taken = max(0, self.used_memory() - before)
            total = torch.cuda.get_device_properties(self.device).total_memory
            previous = torch.cuda.get_per_process_memory_fraction(self.device)
            cap = int(total * previous) - taken
            if cap <= torch.cuda.memory_allocated(self.device) + 512 * 1024**2:
                raise RuntimeError(
                    "Insufficient memory to admit VOSR MIGraphX programs."
                )
            # The enclosing vram_budget's cap, restored at its end; on its next admission
            # the programs count as another program's memory.
            memory_cap.lower(self.device, cap / total)
            for module in self.modules.values():
                module.to(self.device)
            check()
        except BaseException:
            self.offload()
            raise

    def stage(self, name: str, *arguments: torch.Tensor) -> torch.Tensor:
        if self.programs is None:
            raise RuntimeError("VOSR MIGraphX session is not activated.")
        current = torch.cuda.current_stream(self.device)
        if current != self.stream:
            self.stream.wait_stream(current)
        with torch.cuda.stream(self.stream):
            module = self.modules.get(name)
            if module is None:
                result = self.programs.run(name, arguments, self.stream)
            else:
                result = module(*(value.to(torch.bfloat16) for value in arguments))
        if current != self.stream:
            current.wait_stream(self.stream)
            result.record_stream(current)
        return result

    def offload(self) -> None:
        self.execution.clear()
        self.constants.clear()
        self.kv_tensor = None
        self.geometry = None
        self.ropes.clear()
        if self.programs is not None:
            self.programs.close()
            self.programs = None
        for module in self.modules.values():
            module.cpu()
        self.budget = 0

    def close(self) -> None:
        self.offload()
        self.modules.clear()
        _LIVE.discard(self)


@atexit.register
def close_sessions() -> None:
    for session in list(_LIVE):
        session.close()


def lease_migraphx_session(
    context: NodeContext,
    directory: Path,
    paths: BundlePaths,
    device: torch.device,
    *,
    reuse: bool,
) -> AbstractContextManager[FullSession]:
    """The node's session on a cached program set (one per node, as lease_full_session's)."""
    key = (
        "migraphx",
        tuple(
            (
                str(plain(directory, name)),
                plain(directory, name).stat().st_size,
                plain(directory, name).stat().st_mtime_ns,
            )
            for name in PROGRAMS
        ),
        paths.identity(),
        str(device),
    )
    return lease_session(
        context, key, lambda: MigraphxSession(directory, paths, device), reuse=reuse
    )


def check_usable(device: torch.device) -> AmdDevice:
    """The AMD GPU of `device`, after refusing (before any work) what cannot run VOSR on
    MIGraphX."""
    if not torch.version.hip or device.type != "cuda":
        raise ValueError(
            "VOSR MIGraphX requires an AMD GPU and the ROCm build of PyTorch. Turn off "
            "CPU mode in the PyTorch settings or select the PyTorch backend."
        )
    if sys.platform not in ("linux", "win32"):
        raise ValueError(
            "VOSR MIGraphX runs on Linux and Windows only. Select the PyTorch backend."
        )
    if not runner_built():
        raise ValueError(
            "This chaiNNer-C build has no MIGraphX runner. Select the PyTorch backend."
        )
    with torch.cuda.device(device):
        index = torch.cuda.current_device()
    gpu = next((gpu for gpu in amd.devices if gpu.index == index), None)
    if gpu is None:
        raise ValueError(
            f"VOSR MIGraphX found no AMD GPU with HIP index {index} in the system's GPU "
            "list."
        )
    return gpu


def reduced_precision(gpu: AmdDevice) -> str:
    """MIGraphX FP8's precision on `gpu`: FP8 with FP8 matrix math; else BF16 where the
    GPU has BF16 math (CDNA2 and later, RDNA3 and later), FP16 on RDNA1 and 2 (gfx10xx)
    and CDNA1 (gfx908), whose BF16 is slow."""
    if gpu.fp8_format is not None:
        return "fp8"
    if gpu.gfx.startswith("gfx10") or gpu.gfx == "gfx908":
        return "fp16"
    return "bf16"


def ensure_migraphx_engine(
    context: NodeContext, paths: BundlePaths, device: torch.device, *, reduced: bool
) -> Path:
    """The cached program set of this GPU, BF16 or `reduced` (reduced_precision),
    built and published once every program passed its check, as the TensorRT engine
    sets."""
    read_configs(paths)
    gpu = check_usable(device)
    migraphx = load_ops()
    here = Path(__file__)
    pytorch = here.parents[1] / "pytorch"
    precision = reduced_precision(gpu) if reduced else "bf16"
    root = context.storage_dir / f"vosr2-migraphx-{precision}"
    root.mkdir(parents=True, exist_ok=True)
    with cache_lock(root / "compile.lock", context):
        identity = {
            "format": 1,
            "precision": precision,
            # The model code, and what exports, rewrites, calibrates and compiles it
            "architecture": architecture(
                here.with_name("vosr2_migx_compile.py"),
                here.with_name("vosr2_migx_build.py"),
            ),
            "graph_sources": {
                name: file_hash(pytorch / name)
                for name in ("vosr2_trt_models.py", "vosr2_trt_rewrite.py")
            },
            "checkpoints": {Path(local).name: digest for _, local, _, digest in FILES},
            "torch": str(torch.__version__),
            "onnx": metadata.version("onnx"),
            "migraphx": migraphx.MIGRAPHX_VERSION,
            "gpu": gpu.name,
            "gfx": gpu.gfx,
            "ops_sha256": file_hash(ops_path()),
            "programs": {name: program_inputs(name) for name in PROGRAMS},
        }
        # Tuples as JSON has them, so a manifest reads back with the same identity
        identity = json.loads(json.dumps(identity))
        engine = ensure_engine_set(
            context,
            root,
            identity,
            f"MIGraphX {precision.upper()}",
            lambda directory: compile_worker(
                context,
                here.with_name("vosr2_migx_build.py"),
                paths.root,
                directory,
                device,
                label="MIGraphX",
                limit=COMPILE_SECONDS,
                arguments=(precision,),
            ),
            stages=tuple(PROGRAMS),
            compression=False,
        )
        remove_unused_sets(root, engine, identity)
        return engine
