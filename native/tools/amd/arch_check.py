"""The AMD kit's every-architecture check (native/AMD.md). Each architecture of the
installed spandrel and spandrel_extra_arches becomes a small network with random weights
(seed 0): its default settings, or the small settings of NETWORKS where it has no
defaults or they are large. The network's state dict goes through the architecture's
own spandrel loader, as the app loads a model, and the model runs one seeded input of
64x64 (or the size its descriptor requires) with the architecture's input channels on
the GPU in FP32, and in FP16 and BF16 where the descriptor allows them, against the same
model on the CPU in FP32.

Each architecture runs in a process of its own with a time limit, so one that hangs or
ends its process (a segmentation fault) or prints what cannot be read costs only its own
rows; past --time-cap no further architecture starts, and one the cap stops is "not run",
not a failure. An error, NaN or infinity is a FAILED row. FP32 or BF16 that fails on the
GPU runs again on the CPU: a failure there too is the architecture's, and like an
expected one (on the GPUs EXPECTED_FAILURES names) it is listed, not counted.
Differences are reported, with a marker above the precision's threshold (LARGE). The
rows are printed as each architecture finishes, the Result line last.

Usage (the test environment, from the checkout, with backend/src on the path):
  arch_check.py [--device auto|cuda|cpu] [--only REGEX] [--time-cap 900]
                [--timeout 180]
Prints the report's section (Markdown); the exit status is 1 when a counted row failed.
--device auto (the default) takes PyTorch's AMD GPU and, without one, says so and
passes; --device cpu runs the whole check on the CPU against itself (the kit's
self-test on a machine without an AMD GPU). The processes each architecture runs in are
this script's --worker ARCH, which prints that architecture's results as JSON lines."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import random
import re
import subprocess
import sys
import time
import traceback
import typing
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from spandrel import Architecture

SIZE = 64
SEED = 0
TIMED_CALLS = 3
# Larger networks are skipped: the CPU's reference would take most of the time cap.
MAX_PARAMETERS = 100_000_000
# The architectures this fork adds run first, so a time cap reached leaves out others.
FIRST = (
    "DRFT",
    "DUAL",
    "SST",
    "SSTReal",
    "WSR",
    "GaterV3",
    "SMoSR",
    "FIGSR",
    "GFISRv2",
)
# A difference above these (output values 0 to 1, as spandrel clamps them; one 8-bit
# step is 1/255 = 0.0039) is marked "large". FP32: one step, which a GPU's FP32 should
# not reach. FP16 has 3 more mantissa bits than BF16 (8 times finer): 4 and 32 steps,
# above what a network of these sizes gives in that precision on the CPU.
LARGE = {"FP32": 1 / 255, "FP16": 4 / 255, "BF16": 32 / 255}
# (architectures, GPUs (gfx regular expression), why): failures accepted on those GPUs.
# SST's blocks (in SST and SSTReal) call attention with flash or cuDNN attention as the
# only allowed kernels; no other architecture of the fork leaves out the math kernel.
EXPECTED_FAILURES = [
    (
        ("SST", "SSTReal"),
        r"gfx(103[0-9a-f]|908)",
        (
            "SST's attention takes only flash or cuDNN attention, which PyTorch has no"
            " kernel of on this GPU"
        ),
    ),
]


@dataclass(frozen=True)
class Network:
    """A network to build: `factory` (module:callable) called with `settings`."""

    factory: str
    settings: dict[str, Any] = field(default_factory=dict)
    variant: str = ""

    def build(self) -> torch.nn.Module:
        module, name = self.factory.split(":")
        return getattr(importlib.import_module(module), name)(**self.settings)

    def describe(self) -> str:
        return (
            ", ".join(f"{key}={value}" for key, value in self.settings.items())
            or "defaults"
        )


A = "spandrel.architectures"
# The architectures whose default network does not build, does not run on a 64x64
# input, is large or computes nothing (DRCT's default returns its input, NAFNet's has no
# blocks), and those with more than one kind of network, which run each.
NETWORKS: dict[str, list[Network]] = {
    "DRCT": [
        Network(
            f"{A}.DRCT.__arch.drct_arch:DRCT",
            {"upscale": 4, "upsampler": "pixelshuffle"},
        )
    ],
    "DRFT": [
        Network(f"{A}.DRFT.__arch.drft_arch:drft_nano", {"scale": 4}),
        Network(
            f"{A}.DRFT.__arch.legacy_drft_arch:DRFT",
            {
                "embed_dim": 32,
                "depths": (2, 2),
                "num_heads": (2, 2),
                "window_size": 8,
                "upscale": 2,
            },
            "legacy",
        ),
    ],
    # The default network has one input channel for a noise level only
    "DRUNet": [
        Network(f"{A}.DRUNet.__arch.network_unet:DRUNet", {"in_nc": 4, "out_nc": 3})
    ],
    "HAT": [Network(f"{A}.HAT.__arch.HAT:HAT", {"window_size": 16})],
    "IPT": [Network(f"{A}.IPT.__arch.ipt:IPT", {"n_feats": 16, "num_layers": 2})],
    "KBNet": [Network(f"{A}.KBNet.__arch.kbnet_s:KBNet_s", {"width": 16})],
    # Its input is the image and the mask
    "LaMa": [Network(f"{A}.LaMa.__arch.LaMa:LaMa", {"in_nc": 4, "out_nc": 3})],
    "MMRealSR": [
        Network(
            f"{A}.MMRealSR.__arch.mmrealsr_arch:MMRRDBNet_test",
            {"num_in_ch": 3, "num_out_ch": 3, "num_block": 2},
        )
    ],
    "NAFNet": [
        Network(
            f"{A}.NAFNet.__arch.NAFNet_arch:NAFNet",
            {"enc_blk_nums": [1, 1], "dec_blk_nums": [1, 1]},
        )
    ],
    "PLKSR": [
        Network(f"{A}.PLKSR.__arch.PLKSR:PLKSR", variant="PLKSR"),
        Network(f"{A}.PLKSR.__arch.RealPLKSR:RealPLKSR", variant="RealPLKSR"),
    ],
    "RealCUGAN": [Network(f"{A}.RealCUGAN.__arch.upcunet_v3:UpCunet2x")],
    "SAFMN": [Network(f"{A}.SAFMN.__arch.safmn:SAFMN", {"dim": 36})],
    "SAFMNBCIE": [Network(f"{A}.SAFMNBCIE.__arch.safmn_bcie:SAFMN_BCIE", {"dim": 36})],
    "SPAN": [Network(f"{A}.SPAN.__arch.span:SPAN", {"num_in_ch": 3, "num_out_ch": 3})],
    "SST": [Network(f"{A}.SST.__arch.sst_arch:sst_light", {"scale": 4})],
    "SSTReal": [Network(f"{A}.SST.__arch.sst_arch:sst_real", {"scale": 4})],
    "Swin2SR": [Network(f"{A}.Swin2SR.__arch.Swin2SR:Swin2SR", {"window_size": 8})],
    "SwinIR": [Network(f"{A}.SwinIR.__arch.SwinIR:SwinIR", {"window_size": 8})],
    "WSR": [
        Network(f"{A}.WSR.__arch.wsr_arch:WESR", variant="WESR"),
        Network(f"{A}.WSR.__arch.wsr_arch:WideContextSR", variant="WideContextSR"),
    ],
    "SRFormer": [
        Network(
            "spandrel_extra_arches.architectures.SRFormer.__arch.SRFormer:SRFormer",
            {"window_size": 8},
        )
    ],
}
# Architectures left out, and why
SKIPPED = {
    "AuraSR": "no default network (its size settings are required), and no small"
    " settings its spandrel loader is tested with (spandrel's AuraSR tests are off)",
    "MAT": "an inpainting network that takes only 512x512 inputs: on the CPU it took"
    " 13 GB and minutes",
}


def registry() -> dict[str, Architecture[Any]]:
    """The installed architectures by ID: spandrel's, then spandrel_extra_arches'
    (which the app adds to spandrel's)."""
    from spandrel import MAIN_REGISTRY

    architectures = {
        str(support.architecture.id): support.architecture for support in MAIN_REGISTRY
    }
    try:
        from spandrel_extra_arches import EXTRA_REGISTRY
    except ModuleNotFoundError as error:
        if error.name != "spandrel_extra_arches":
            raise
        return architectures
    for support in EXTRA_REGISTRY:
        architectures.setdefault(str(support.architecture.id), support.architecture)
    return architectures


def networks(architecture: Architecture[Any]) -> list[Network]:
    """The networks an architecture runs: NETWORKS', else its loader's network class
    with its defaults (the class its Architecture is generic in)."""
    import torch

    if architecture.id in NETWORKS:
        return NETWORKS[architecture.id]
    for base in getattr(type(architecture), "__orig_bases__", ()):
        for argument in typing.get_args(base):
            # Not torch.nn.Module itself, which names no network
            if isinstance(argument, type) and argument is not torch.nn.Module:
                return [Network(f"{argument.__module__}:{argument.__qualname__}")]
    raise TypeError(
        f"{type(architecture).__name__} names no single network class; add the"
        " architecture to NETWORKS"
    )


def seed() -> None:
    """Seeds PyTorch's and Python's generators, which the networks draw from."""
    import torch

    random.seed(SEED)
    torch.manual_seed(SEED)


def fresh(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    """Copies of the inputs for one call: some models change their input (RCAN)."""
    return tuple(t.clone() for t in inputs)


def first_line(error: BaseException) -> str:
    lines = str(error).strip().splitlines()
    return f"{type(error).__name__}: {lines[0] if lines else ''}"[:300]


def emit(**record: object) -> None:
    print(json.dumps(record), flush=True)


class Worker:
    """One architecture's networks, in this process."""

    def __init__(self, architecture: Architecture[Any], device: str):
        import torch

        self.architecture = architecture
        self.device = torch.device(device)

    def inputs(self, model: Any) -> tuple[torch.Tensor, ...]:
        """The seeded input (and a mask, for inpainting) the model takes."""
        import torch
        from spandrel import MaskedImageModelDescriptor

        requirements = model.size_requirements
        size = max(SIZE, requirements.minimum)
        size = -(-size // requirements.multiple_of) * requirements.multiple_of
        generator = torch.Generator().manual_seed(SEED)
        image = torch.rand(1, model.input_channels, size, size, generator=generator)
        if not isinstance(model, MaskedImageModelDescriptor):
            return (image,)
        mask = torch.zeros(1, 1, size, size)
        mask[..., size // 4 : size // 2, size // 4 : size // 2] = 1
        return image, mask

    def run(self, network: Network) -> None:
        import torch

        name = self.architecture.id
        label = {"arch": name, "variant": network.variant}
        try:
            seed()
            net = network.build()
            parameters = sum(p.numel() for p in net.parameters())
            if parameters > MAX_PARAMETERS:
                emit(
                    kind="skip",
                    **label,
                    reason=f"{parameters / 1e6:.0f} M parameters with "
                    f"{network.describe()} (the check builds up to "
                    f"{MAX_PARAMETERS / 1e6:.0f} M)",
                )
                return
            seed()
            reference_model = self.architecture.load(net.state_dict()).eval()
            del net
            # Each precision runs a copy taken before the CPU's run, which leaves the
            # same weights and buffers (FP16 to FP32 and back would round them) and
            # none of the tensors a model keeps from a call on the CPU.
            pristine = copy.deepcopy(reference_model)
        except Exception as error:
            traceback.print_exc()
            emit(
                kind="skip",
                **label,
                reason=f"no network from {network.describe()}: {first_line(error)}",
            )
            return
        precisions = ["FP32"]
        # PyTorch's FP16 on the CPU takes seconds per call, and the app runs FP16 only
        # on GPUs: the CPU's own check leaves it out.
        if reference_model.supports_half and self.device.type != "cpu":
            precisions.append("FP16")
        if reference_model.supports_bfloat16:
            precisions.append("BF16")
        inputs = self.inputs(reference_model)
        try:
            reference = reference_model(*fresh(inputs)).float()
        except Exception as error:
            traceback.print_exc()
            emit(
                kind="skip",
                **label,
                reason=f"the CPU's FP32 run failed: {first_line(error)}",
            )
            return
        del reference_model
        emit(
            kind="network",
            **label,
            settings=network.describe(),
            parameters=parameters,
            input="x".join(str(n) for n in inputs[0].shape[1:]),
            precisions=precisions,
        )
        dtypes = {"FP32": torch.float32, "FP16": torch.float16, "BF16": torch.bfloat16}
        for precision in precisions:
            dtype = dtypes[precision]
            row: dict[str, object] = {"kind": "row", **label, "precision": precision}
            row.update(self.attempt(pristine, inputs, dtype, reference, self.device))
            # A failure the CPU repeats is the architecture's, not the GPU's. FP16
            # fails on the GPU alone: the CPU's takes seconds to minutes per call.
            if (
                row["result"] != "ok"
                and self.device.type != "cpu"
                and precision != "FP16"
            ):
                cpu = torch.device("cpu")
                again = self.attempt(pristine, inputs, dtype, reference, cpu, False)
                row["cpu_too"] = again["result"] != "ok"
            emit(**row)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def attempt(
        self,
        pristine: Any,
        inputs: tuple[torch.Tensor, ...],
        dtype: torch.dtype,
        reference: torch.Tensor,
        device: torch.device,
        timed: bool = True,
    ) -> dict[str, object]:
        """A copy of the model in this precision on the device: whether its first
        call's raw output (before spandrel clamps it) is finite, its largest difference
        from the reference, and the mean time of the calls after the first; or its
        error."""
        import torch

        try:
            model = copy.deepcopy(pristine).to(device).to(dtype)
            device_inputs = tuple(t.to(device, dtype) for t in inputs)
            finite: list[bool] = []

            def check(_module: object, _arguments: object, output: object) -> None:
                if isinstance(output, torch.Tensor):
                    finite.append(bool(torch.isfinite(output).all()))

            hook = model.model.register_forward_hook(check)
            try:
                output = model(*fresh(device_inputs)).float().cpu()
            finally:
                hook.remove()
            if not all(finite) or not bool(torch.isfinite(output).all()):
                return {"result": "nonfinite"}
            if output.shape != reference.shape:
                return {
                    "result": "error",
                    "error": f"output {tuple(output.shape)}, the CPU's"
                    f" {tuple(reference.shape)}",
                }
            difference = float((output - reference).abs().max())
            if not timed:
                return {"result": "ok", "max_diff": difference}
            synchronize(device)
            start = time.perf_counter()
            for _ in range(TIMED_CALLS):
                model(*fresh(device_inputs))
            synchronize(device)
            milliseconds = (time.perf_counter() - start) / TIMED_CALLS * 1000
            return {"result": "ok", "max_diff": difference, "ms": milliseconds}
        except Exception as error:
            print(f"{self.architecture.id} {dtype} on {device}:", file=sys.stderr)
            traceback.print_exc()
            return {"result": "error", "error": first_line(error)}


def synchronize(device: torch.device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def worker(name: str, device: str) -> int:
    from packages.chaiNNer_pytorch.torch_runtime import configure_torch

    configure_torch()
    architecture = registry()[name]
    for network in networks(architecture):
        Worker(architecture, device).run(network)
    return 0


def probe(device: str) -> int:
    """The versions and the GPU, as one JSON line."""
    import spandrel
    import torch

    record: dict[str, object] = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "spandrel": spandrel.__version__,
    }
    try:
        import spandrel_extra_arches

        record["extra"] = spandrel_extra_arches.__version__
    except ModuleNotFoundError as error:
        if error.name != "spandrel_extra_arches":
            raise
    if device != "cpu" and torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        record["gpu"] = properties.name
        record["gfx"] = getattr(properties, "gcnArchName", "").split(":")[0]
    print(json.dumps(record), flush=True)
    return 0


@dataclass
class Outcome:
    """What one worker process gave: its JSON lines, and how it ended."""

    records: list[dict[str, Any]]
    ending: str | None
    """None when it exited normally, else why it did not"""
    unreadable: list[str] = field(default_factory=list)
    """The output lines that look like records but are not"""
    timed_out: bool = False


def worker_command(arguments: list[str]) -> list[str]:
    return [sys.executable, "-B", str(Path(__file__).resolve()), *arguments]


def parse(stdout: str) -> tuple[list[dict[str, Any]], list[str]]:
    """The JSON records of a worker's output, and its lines that look like one but do
    not parse as a JSON object."""
    records: list[dict[str, Any]] = []
    unreadable: list[str] = []
    for line in stdout.splitlines():
        if not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except ValueError:
            unreadable.append(line)
            continue
        if isinstance(record, dict):
            records.append(record)
        else:
            unreadable.append(line)
    return records, unreadable


def run_worker(
    command: list[str], timeout: float, log: Callable[[str], None]
) -> Outcome:
    """Runs a worker; its output is read as UTF-8 with undecodable bytes replaced."""
    timed_out = False
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            stdin=subprocess.DEVNULL,
        )
        stdout, stderr = result.stdout, result.stderr
        if result.returncode < 0:
            ending = f"ended by signal {-result.returncode}"
        elif result.returncode != 0:
            ending = f"exit status {result.returncode}"
        else:
            ending = None
    except subprocess.TimeoutExpired as expired:
        timed_out = True
        stdout = expired.stdout or ""
        stderr = expired.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        ending = f"no result after {timeout:.0f} s (stopped)"
    log(stderr)
    records, unreadable = parse(stdout)
    if ending is not None:
        lines = stderr.strip().splitlines()
        # The last line naming an error, else the last line (a warning may follow it)
        errors = [line for line in lines if re.search(r"Error|error|Fault|Abort", line)]
        if errors or lines:
            ending += f"; last error line: {(errors or lines)[-1][:200]}"
    return Outcome(records, ending, unreadable, timed_out)


@dataclass
class Row:
    arch: str
    variant: str
    settings: str
    parameters: int
    input: str
    precision: str
    result: str
    error: str = ""
    max_diff: float | None = None
    ms: float | None = None
    cpu_too: bool = False
    """It failed on the CPU too: the architecture's failure, not the GPU's"""

    @property
    def failed(self) -> bool:
        """A failure of the GPU's (one the CPU repeats is reported, not counted, and so
        is a row the check's time cap left out)"""
        counted = self.result not in ("ok", "expected failure", "not run")
        return counted and not self.cpu_too

    def markdown(self, first: bool) -> str:
        """The table row; the network's settings only on its first row."""
        name = f"{self.arch} ({self.variant})" if self.variant else self.arch
        network = ""
        if first and self.settings:
            network = f"{self.settings}; {self.parameters / 1e6:.2f} M parameters"
        failed = "FAILED on the CPU too" if self.cpu_too else "FAILED"
        if self.result == "ok":
            result = "ok"
        elif self.result == "nonfinite":
            result = f"{failed}: NaN or infinity"
        elif self.result in ("expected failure", "not run"):
            result = f"{self.result}: {self.error}"
        else:
            result = f"{failed}: {self.error}"
        result = result.replace("|", "\\|")
        difference = ""
        if self.max_diff is not None:
            difference = f"{self.max_diff:.2g}"
            if self.max_diff > LARGE[self.precision]:
                difference += " (large)"
        ms = "" if self.ms is None else f"{self.ms:.1f}"
        return (
            f"| {name} | {network} | {self.input} | {self.precision} | {result} |"
            f" {difference} | {ms} |"
        )


CAPPED = "the check's time cap ran out before it finished"


def rows_of(
    arch: str, outcome: Outcome, capped: bool = False
) -> tuple[list[Row], list[str]]:
    """An architecture's rows and skip lines from its worker. The precisions a network
    planned and gave no row for (the process ended) are FAILED rows with the reason, or
    "not run" rows when the check's time cap stopped it (`capped`); a process that ended
    before naming any network is one FAILED row (a skip line when capped). Each line or
    record that cannot be read is a FAILED row with the line."""
    rows: list[Row] = []
    skips: list[str] = []
    current: dict[str, Any] | None = None
    pending: list[str] = []
    unreadable = list(outcome.unreadable)

    def row(network: dict[str, Any], precision: str, result: str, error: str) -> Row:
        return Row(
            arch,
            network["variant"],
            network["settings"],
            network["parameters"],
            network["input"],
            precision,
            result,
            error,
        )

    def close() -> None:
        if current is None:
            return
        for precision in pending:
            if capped:
                rows.append(row(current, precision, "not run", CAPPED))
            else:
                ending = outcome.ending or "no result"
                rows.append(row(current, precision, "error", ending))

    for record in outcome.records:
        try:
            kind = record["kind"]
            if kind == "network":
                row(record, "", "", "")  # a network record names every field
                close()
                current = record
                pending = list(record["precisions"])
            elif kind == "row" and current is not None:
                pending.remove(record["precision"])
                entry = row(current, record["precision"], record["result"], "")
                entry.error = record.get("error", "")
                entry.max_diff = record.get("max_diff")
                entry.ms = record.get("ms")
                entry.cpu_too = record.get("cpu_too", False)
                rows.append(entry)
            elif kind == "skip":
                name = f"{arch} ({record['variant']})" if record["variant"] else arch
                skips.append(f"- {name}: {record['reason']}")
            else:
                unreadable.append(json.dumps(record))
        except KeyError, TypeError, ValueError:
            unreadable.append(json.dumps(record))
    close()
    rows += [
        Row(arch, "", "", 0, "", "", "error", f"unreadable worker output: {line[:200]}")
        for line in unreadable
    ]
    if not outcome.records and not unreadable and outcome.ending is not None:
        if capped:
            skips.append(f"- {arch}: not run, {CAPPED}")
        else:
            rows.append(Row(arch, "", "", 0, "", "", "error", outcome.ending))
    return rows, skips


def expected(rows: list[Row], gfx: str) -> None:
    """Marks the failures EXPECTED_FAILURES accepts on this GPU."""
    for entry in rows:
        if entry.result in ("ok", "nonfinite", "not run") or entry.cpu_too:
            continue
        for archs, gpus, why in EXPECTED_FAILURES:
            if entry.arch in archs and re.fullmatch(gpus, gfx):
                entry.error = f"{why} ({gfx}): {entry.error}"
                entry.result = "expected failure"


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="The AMD kit's every-architecture check."
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument(
        "--only", default="", help="a regular expression: only the matching IDs"
    )
    parser.add_argument(
        "--time-cap", type=float, default=900, help="seconds for the whole check"
    )
    parser.add_argument(
        "--timeout", type=float, default=180, help="seconds per architecture"
    )
    parser.add_argument("--worker", metavar="ARCH")
    parser.add_argument("--probe", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    args = arguments()
    if args.probe:
        return probe(args.device)
    if args.worker:
        return worker(args.worker, args.device)

    def log(text: str) -> None:
        if text:
            print(text, file=sys.stderr, end="" if text.endswith("\n") else "\n")

    def out(text: str = "") -> None:
        # Line by line: rows written stay in the report if the kit stops the check
        print(text, flush=True)

    out("## Every architecture")
    out()
    started = time.monotonic()
    found = run_worker(worker_command(["--probe", "--device", args.device]), 300, log)
    if found.ending is not None or not found.records:
        problem = found.ending or f"unreadable output: {found.unreadable[:1]}"
        out(f"Result: FAILED: PyTorch and spandrel did not load ({problem})")
        return 1
    versions = found.records[0]
    gpu = versions.get("gpu")
    device = args.device
    if device == "auto":
        if gpu is None or versions.get("hip") is None:
            out(
                "Result: not run: PyTorch sees no AMD GPU here"
                f" (torch {versions['torch']})."
            )
            return 0
        device = "cuda"
    where = "the CPU"
    if device == "cuda":
        where = (
            f"{gpu} ({versions.get('gfx')})" if gpu else "a GPU PyTorch did not find"
        )
    gfx = str(versions.get("gfx") or "")
    extra = versions.get("extra")
    architectures = registry()
    names = [name for name in FIRST if name in architectures]
    names += [name for name in architectures if name not in FIRST]
    names = [name for name in names if re.search(args.only, name)]

    stack = [f"spandrel {versions['spandrel']}"]
    if extra:
        stack.append(f"spandrel_extra_arches {extra}")
    stack.append(f"torch {versions['torch']}")
    if versions["hip"]:
        stack.append(f"HIP {versions['hip']}")
    out(
        f"{', '.join(stack)}. Each architecture is a network with random weights (seed"
        f" {SEED}), loaded through its spandrel loader; one seeded input of {SIZE}x{SIZE}"
        f" (or the size the model requires) runs on {where} in FP32, and in FP16 and"
        " BF16 (the model cast to it) where the model's descriptor allows them, against"
        " the same model on the CPU in FP32. A row FAILED on an error, NaN or"
        " infinity; FP32 or BF16 that fails on the GPU runs again on the CPU, and a"
        " failure there too is the architecture's, listed but not counted."
        " Max abs diff: the largest difference of an output value (0 to 1, as spandrel"
        " clamps them; one 8-bit step is 0.0039), marked large above"
        f" {LARGE['FP32']:.4f} in FP32 (one step, which FP32 should not reach),"
        f" {LARGE['FP16']:.4f} in FP16 (4 steps) and {LARGE['BF16']:.3f} in BF16 (32"
        " steps; BF16 is 8 times coarser than FP16); random networks magnify"
        " differences more than trained ones, so a large one is a lead, not a verdict."
        f" ms: the mean of {TIMED_CALLS} calls after the first (which compiles"
        " kernels). Each architecture runs in a process of its own (time limit"
        f" {args.timeout:.0f} s); none starts after {args.time_cap:.0f} s, and one"
        " that cap stops is not run, not failed. The Result line comes after the table."
    )
    out()
    out(
        "Expected: SST and SSTReal fail on GPUs without flash or cuDNN attention in"
        " PyTorch (RX 6000, gfx103x; MI100, gfx908), which is accepted and listed. The"
        " FFT-based architectures (FIGSR, GFISRv2, FFTformer, LaMa) are unverified on"
        " AMD GPUs in FP16: their FP16 rows here are the first check."
    )
    out()
    out("| Architecture | Network | Input | Precision | Result | Max abs diff | ms |")
    out("|---|---|---|---|---|---|---|")

    rows: list[Row] = []
    skips: list[str] = []
    previous = None
    for name in names:
        if name in SKIPPED:
            skips.append(f"- {name}: {SKIPPED[name]}")
            continue
        remaining = args.time_cap - (time.monotonic() - started)
        if remaining < 20:
            skips.append(
                f"- {name}: not run, the check's {args.time_cap:.0f} s ran out"
            )
            continue
        timeout = min(args.timeout, remaining)
        log(f"== {name}")
        command = worker_command(["--worker", name, "--device", device])
        outcome = run_worker(command, timeout, log)
        # Stopped by the check's time cap rather than by its own time limit
        capped = outcome.timed_out and timeout < args.timeout
        named, skipped = rows_of(name, outcome, capped)
        expected(named, gfx)
        rows += named
        skips += skipped
        for entry in named:
            out(entry.markdown((entry.arch, entry.variant) != previous))
            previous = (entry.arch, entry.variant)
            log(
                f"{name} {entry.variant} {entry.precision}: {entry.result} {entry.error}"
            )

    if skips:
        out()
        out("Skipped:")
        out()
        out("\n".join(skips))
    failed = sum(entry.failed for entry in rows)
    cpu_too = sum(entry.cpu_too for entry in rows)
    accepted = sum(entry.result == "expected failure" for entry in rows)
    not_run = sum(entry.result == "not run" for entry in rows)
    large = sum(
        entry.max_diff is not None and entry.max_diff > LARGE[entry.precision]
        for entry in rows
    )
    built = len({entry.arch for entry in rows})
    out()
    out(
        f"Result: {built} architectures on {where}, {len(rows)} rows:"
        f" {sum(entry.result == 'ok' for entry in rows)} ok, {failed} FAILED,"
        f" {cpu_too} failed on the CPU too, {accepted} expected failures,"
        f" {not_run} not run (time cap), {large} large differences;"
        f" {len(skips)} skipped. {time.monotonic() - started:.0f} s."
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
