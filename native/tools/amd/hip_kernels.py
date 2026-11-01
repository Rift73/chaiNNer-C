"""The AMD kit's check of spandrel's own HIP kernels (native/AMD.md). From 0.4.2+c12,
spandrel runs depthwise 3x3 convolutions and ESRGAN's RRDB blocks with HIP kernels of
its own on ROCm (spandrel.rocm; SPANDREL_HIP_KERNELS=0 switches them off). The ESRGAN
and the DAT2 of models.txt (DAT2 has depthwise convolutions) run one image of --size on
the GPU in FP32, and in FP16 where the model allows it, once with SPANDREL_HIP_KERNELS=0
and once with the default, each in a process of its own: the largest difference
between the two outputs (ESRGAN's RRDB kernels are made to give the same bits; the
depthwise ones may differ slightly), the time per image of each over --iterations calls
after two warm-up calls, and whether the kernels ran (spandrel.rocm's log lines; a
warning from it means they fell back to PyTorch's, which fails). A spandrel without
spandrel.rocm, a machine whose PyTorch sees no AMD GPU, and a model missing from
models.txt are each a line of the section, not a failure. Each row is printed as its two
runs finish, the Result line last.

Usage (the test environment, from the checkout, with backend/src on the path):
  hip_kernels.py --models MODELS_TXT --image INPUT_PNG [--size 256] [--iterations 5]
                 [--device auto|cuda|cpu]
Prints the report's section (Markdown); the exit status is 1 when a run failed or gave
NaN or infinity, the kernels fell back, or ESRGAN's two outputs differ. --device cpu runs both on the CPU (the
kit's self-test, which has no AMD GPU). The processes are this script's --worker."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The models.txt roles this check runs, with what the kernels promise for each
MODELS = {
    "esrgan": "RRDB blocks: the same bits",
    "dat2": "depthwise 3x3 convolutions: may differ slightly",
}
WARM_UP_CALLS = 2


def worker(args: argparse.Namespace) -> int:
    """One run: prints its JSON line, saves its output to --out."""
    import cv2
    import numpy as np
    import torch

    from packages.chaiNNer_pytorch.torch_runtime import configure_torch

    configure_torch()
    device = args.device
    if device == "auto":
        if not (torch.version.hip and torch.cuda.is_available()):
            print(json.dumps({"gpu": None, "torch": torch.__version__}), flush=True)
            return 0
        device = "cuda"
    lines: list[str] = []
    warnings: list[str] = []

    class Keep(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            line = f"{record.levelname}: {record.getMessage()}"
            lines.append(line)
            # spandrel.rocm warns when a kernel fails and its layer falls back
            if record.levelno >= logging.WARNING:
                warnings.append(line)

    # The kernels log from the model's load on
    rocm_logger = logging.getLogger("spandrel.rocm")
    rocm_logger.setLevel(logging.DEBUG)
    rocm_logger.addHandler(Keep())
    import spandrel

    available = None
    # From spandrel 0.4.2+c12 on
    if importlib.util.find_spec("spandrel.rocm") is not None:
        available = importlib.import_module("spandrel.rocm").is_available()
    # Loaded on the device, as the app's Load Model does
    loader = spandrel.ModelLoader(torch.device(device))
    model = loader.load_from_file(args.model).eval()
    if not isinstance(model, spandrel.ImageModelDescriptor):
        raise TypeError(f"{args.model} is not an image model")
    if args.precision == "FP16":
        model = model.half()
    dtype = torch.float16 if args.precision == "FP16" else torch.float32
    source = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if source is None:
        raise FileNotFoundError(f"{args.image} is not an image OpenCV reads")
    image = cv2.resize(source, (args.size, args.size), interpolation=cv2.INTER_CUBIC)
    rgb = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)) / np.float32(255)
    tensor = torch.from_numpy(rgb).unsqueeze(0).to(device, dtype)

    def synchronize() -> None:
        if device == "cuda":
            torch.cuda.synchronize()

    # A copy of the input for each call: some models change theirs
    output = model(tensor.clone())
    for _ in range(WARM_UP_CALLS - 1):
        model(tensor.clone())
    synchronize()
    start = time.perf_counter()
    for _ in range(args.iterations):
        model(tensor.clone())
    synchronize()
    milliseconds = (time.perf_counter() - start) / args.iterations * 1000
    result = output.float().cpu().numpy()
    np.save(args.out, result)
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name() if device == "cuda" else "CPU",
                "torch": torch.__version__,
                "architecture": model.architecture.name,
                "supports_half": model.supports_half,
                "available": available,
                "finite": bool(np.isfinite(result).all()),
                "ms": milliseconds,
                "logs": lines,
                "warnings": warnings,
            }
        ),
        flush=True,
    )
    return 0


@dataclass
class Run:
    record: dict[str, Any] | None
    """The worker's JSON line; None when it gave none"""
    error: str
    output: Path


def run(
    model: Path, precision: str, kernels: bool, args: argparse.Namespace, folder: Path
) -> Run:
    """One worker: kernels off (SPANDREL_HIP_KERNELS=0) or as spandrel's default."""
    output = folder / f"{model.stem}-{precision}-{'default' if kernels else 'off'}.npy"
    environment = dict(os.environ)
    environment.pop("SPANDREL_HIP_KERNELS", None)
    if not kernels:
        environment["SPANDREL_HIP_KERNELS"] = "0"
    command = [
        *(sys.executable, "-B", str(Path(__file__).resolve()), "--worker"),
        *("--model", str(model), "--precision", precision, "--out", str(output)),
        *("--image", str(args.image), "--size", str(args.size)),
        *("--iterations", str(args.iterations), "--device", args.device),
    ]
    print(
        f"== {model.name} {precision} kernels {'default' if kernels else 'off'}",
        file=sys.stderr,
        flush=True,
    )
    record, error = launch(command, environment, args.timeout)
    return Run(record, error, output)


def launch(
    command: list[str], environment: dict[str, str], timeout: float
) -> tuple[dict[str, Any] | None, str]:
    """A worker's JSON record, or why there is none. Its output is read as UTF-8 with
    undecodable bytes replaced; a last line that does not parse is the run's failure."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=environment,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return None, f"no result after {timeout:.0f} s (stopped)"
    print(result.stderr, file=sys.stderr, end="", flush=True)
    lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
    if result.returncode != 0 or not lines:
        # The last line naming an error, else the last line (a warning may follow it)
        errors = result.stderr.strip().splitlines()
        errors = [
            e for e in errors if re.search(r"Error|error|Fault|Abort", e)
        ] or errors
        ending = (
            f"ended by signal {-result.returncode}"
            if result.returncode < 0
            else f"exit status {result.returncode}"
        )
        return None, f"{ending}: {errors[-1] if errors else 'no output'}"
    try:
        record = json.loads(lines[-1])
    except ValueError:
        record = None
    if not isinstance(record, dict):
        return None, f"unreadable worker output: {lines[-1][:200]}"
    return record, ""


def kernels_used(default: dict[str, Any]) -> tuple[str, bool]:
    """Whether the default run used the kernels, and whether that fails the row: a
    warning from spandrel.rocm (a kernel that did not compile, load or run, so its layer
    fell back) does; "used" needs is_available() and no warning."""
    warnings = default.get("warnings") or []
    if warnings:
        return f"FAILED: fell back: {warnings[0]}", True
    available = default["available"]
    if available is None:
        return "no spandrel.rocm", False
    if not available:
        return "no: spandrel.rocm.is_available() is False", False
    count = len(default["logs"])
    return (f"yes ({count} log lines)" if count else "yes (no log line)"), False


def section(args: argparse.Namespace, out: Callable[[str], None]) -> int:
    """Writes the section's lines with `out` as they are known (a row as soon as its
    two runs finished), and returns the exit status."""
    import numpy as np

    # The spandrel this Python imports (which may not be the one pip installed)
    version = importlib.import_module("spandrel").__version__
    if importlib.util.find_spec("spandrel.rocm") is None:
        out(f"Result: spandrel without HIP kernels (version {version}).")
        return 0
    models: dict[str, Path] = {}
    for line in args.models.read_text(encoding="utf-8").splitlines():
        role, _, path = line.partition(" ")
        models[role] = Path(path)
    rows: list[str] = []
    notes: list[str] = []
    logs: list[str] = []
    failures = 0
    gpu = ""

    def add(row: str) -> None:
        if not rows:
            out(
                f"Each model runs one {args.size}x{args.size} image in a process of its"
                " own, with SPANDREL_HIP_KERNELS=0 (off) and with spandrel's default:"
                f" ms per image over {args.iterations} calls after {WARM_UP_CALLS}"
                " warm-up calls (the first compiles the kernels), the change (off /"
                " default; above 1x the kernels are faster), and the largest difference"
                " between the two outputs (values 0 to 1; one 8-bit step is 0.0039)."
                " ESRGAN's RRDB kernels are made to give the same bits, so any"
                " difference there fails; DAT2's depthwise ones may differ slightly. A"
                " warning from spandrel.rocm in the default run means a kernel fell"
                " back to PyTorch's, which fails. The Result line comes last."
            )
            out("")
            out(
                "| Model | Precision | Off: ms | Default: ms | Change | Max abs diff |"
                " HIP kernels used |"
            )
            out("|---|---|---|---|---|---|---|")
        rows.append(row)
        out(row)

    with tempfile.TemporaryDirectory(prefix="hip-kernels-") as temporary:
        for role, promise in MODELS.items():
            model = models.get(role)
            if model is None or not model.is_file():
                notes.append(
                    f"- {role}: skipped, no {role} model (copy it to --models as"
                    f" {role}.pth)"
                )
                continue
            precisions = ["FP32"]
            for precision in precisions:
                off = run(model, precision, False, args, Path(temporary))
                if off.record is not None and off.record["gpu"] is None:
                    out(
                        "Result: not run: PyTorch sees no AMD GPU here"
                        f" (torch {off.record['torch']}, spandrel {version})."
                    )
                    return 0
                default = run(model, precision, True, args, Path(temporary))
                name = f"| {role} | {precision} |"
                if off.record is None or default.record is None:
                    failures += 1
                    error = off.error or default.error
                    add(f"{name} | | | FAILED: {error[:300]} | |")
                    continue
                gpu = off.record["gpu"]
                kind = off.record["architecture"]
                # FP16 where the model allows it, but not on the CPU (seconds per call
                # there, and the app runs FP16 only on GPUs)
                if precision == "FP32" and off.record["supports_half"]:
                    if args.device != "cpu":
                        precisions.append("FP16")
                if not (off.record["finite"] and default.record["finite"]):
                    failures += 1
                    add(f"{name} | | | FAILED: NaN or infinity | |")
                    continue
                difference = float(
                    np.abs(np.load(off.output) - np.load(default.output)).max()
                )
                verdict = f"{difference:.2g}"
                differs = role == "esrgan" and difference != 0
                if differs:
                    verdict += " (FAILED: designed to be bit-identical)"
                used, fell_back = kernels_used(default.record)
                failures += differs or fell_back
                change = off.record["ms"] / default.record["ms"]
                add(
                    f"{name} {off.record['ms']:.1f} | {default.record['ms']:.1f} |"
                    f" {change:.2f}x | {verdict} | {used} |"
                )
                logs.append(f"- {role} ({kind}, {promise}), {precision}:")
                logged = default.record["logs"][:8] or ["none"]
                logs += [f"  - {line}" for line in logged]
                if off.record["logs"]:
                    logs.append(
                        "  - with SPANDREL_HIP_KERNELS=0 it logged:"
                        f" {off.record['logs'][0]}"
                    )
    for line in [*([""] if notes else []), *notes]:
        out(line)
    if logs:
        out("")
        out("spandrel.rocm's log lines in the default runs:")
        out("")
        for line in logs:
            out(line)
    out("")
    where = f" on {gpu}" if gpu else ""
    out(
        f"Result: spandrel {version}{where}: {len(rows)} rows, {failures} FAILED,"
        f" {len(notes)} skipped."
    )
    return 1 if failures else 0


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="The AMD kit's check of spandrel's HIP kernels."
    )
    parser.add_argument("--models", type=Path)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument(
        "--timeout", type=float, default=300, help="seconds per process"
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--precision", choices=("FP32", "FP16"), default="FP32")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if not args.worker and args.models is None:
        parser.error("--models is needed")
    return args


def main() -> int:
    args = arguments()
    if args.worker:
        return worker(args)

    def out(line: str) -> None:
        # Line by line: rows written stay in the report if the kit stops the check
        print(line, flush=True)

    out("## spandrel's HIP kernels on and off")
    out("")
    try:
        return section(args, out)
    except Exception as error:
        traceback.print_exc()
        out(f"Result: FAILED: {type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
