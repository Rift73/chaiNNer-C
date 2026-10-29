"""HAT-L and DAT2 on an AMD GPU in FP16: PyTorch on ROCm with chaiNNer's process
settings, and MIGraphX compiling chaiNNer's ONNX export (Convert To ONNX's FP32 graph,
quantised to FP16 by MIGraphX). The models carry random weights in the released models'
settings (4x, 180 channels in 6-head groups; HAT-L 12 groups of 16x16 windows, DAT2 6
groups of 8x32 windows), which time like the trained ones. One line per model and route:
the median milliseconds per image of the timed runs; the PyTorch line also says whether
every attention call took a fused kernel (AOTriton) or some fell back to the math kernel.

MIGraphX fuses attention into one rocMLIR kernel by default only on gfx94x and gfx95x;
its rows are timed with MIGraphX's default and with attention fusion requested
(MIGRAPHX_MLIR_USE_SPECIFIC_OPS=attention), and each says how many of the program's
kernels are fused attention (rocMLIR kernels holding the softmax) and how many softmaxes
run as kernels of their own. On PyTorch for AMD GPUs the export lays the attention out
for MIGraphX (nodes/impl/pytorch/attention_export.py): every window attention fuses
(HAT-L's 84, DAT2's 36), and DAT2's 18 channel attentions stay two products and a
softmax each.

    python speed_hat_dat.py [--size 256] [--runs 50] [--migraphx-driver PATH]

Without --migraphx-driver, migraphx-driver is looked up on PATH; when it is missing the
MIGraphX rows say so.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "backend" / "src"))

from spandrel import ImageModelDescriptor, ModelLoader
from spandrel.architectures.DAT import DAT
from spandrel.architectures.HAT import HAT
from torch.nn.attention import SDPBackend, sdpa_kernel

from nodes.impl.pytorch.convert_to_onnx_impl import convert_to_onnx_impl
from packages.chaiNNer_pytorch.torch_runtime import configure_torch

MODELS: dict[str, Callable[[], torch.nn.Module]] = {
    "HAT-L": lambda: HAT(
        upscale=4,
        window_size=16,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        depths=[6] * 12,
        embed_dim=180,
        num_heads=[6] * 12,
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    ),
    "DAT2": lambda: DAT(
        upscale=4,
        depth=[6] * 6,
        embed_dim=180,
        num_heads=[6] * 6,
        split_size=[8, 32],
        expansion_factor=2.0,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    ),
}
WARMUP = 5


def load(name: str) -> ImageModelDescriptor:
    torch.manual_seed(0)
    model = ModelLoader().load_from_state_dict(MODELS[name]().state_dict())
    assert isinstance(model, ImageModelDescriptor)
    model.model.requires_grad_(False)
    return model.eval()


def pytorch_row(model: ImageModelDescriptor, size: int, runs: int) -> str:
    net = model.model.to("cuda", torch.float16)
    image = torch.rand(1, 3, size, size, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        try:
            with sdpa_kernel(
                [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
            ):
                net(image)
            attention = "fused attention"
        except RuntimeError as error:
            attention = f"math attention ({str(error).splitlines()[0]})"
        times = []
        for i in range(WARMUP + runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            net(image)
            torch.cuda.synchronize()
            if i >= WARMUP:
                times.append((time.perf_counter() - start) * 1000)
    net.to("cpu")
    torch.cuda.empty_cache()
    return f"{statistics.median(times):.2f} ms, {attention}"


def migraphx_row(
    driver: str, onnx: Path, size: int, runs: int, fuse_attention: bool
) -> str:
    environment = dict(os.environ)
    if fuse_attention:
        environment["MIGRAPHX_MLIR_USE_SPECIFIC_OPS"] = "attention"
    result = subprocess.run(
        [
            driver,
            "perf",
            "--onnx",
            str(onnx),
            "--input-dim",
            "@input",
            "1",
            "3",
            str(size),
            str(size),
            "--fp16",
            "--gpu",
            "-n",
            str(runs),
        ],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    median = re.search(r"Median: ([0-9.]+)ms", result.stdout)
    if result.returncode != 0 or median is None:
        tail = (result.stderr or result.stdout).strip().splitlines()[-3:]
        return f"failed (exit {result.returncode}): {' | '.join(tail)}"
    kernels = re.findall(r"symbol_name=(\w+)", result.stdout)
    fused = sum(k.startswith("mlir_") and "reduce_max" in k for k in kernels)
    alone = sum(not k.startswith("mlir_") and "reduce_max" in k for k in kernels)
    return (
        f"{float(median.group(1)):.2f} ms, {fused} fused attention kernels, "
        f"{alone} softmax kernels"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--migraphx-driver", default=shutil.which("migraphx-driver"))
    args = parser.parse_args()
    if not (torch.version.hip and torch.cuda.is_available()):
        raise SystemExit("needs PyTorch on ROCm and an AMD GPU")
    configure_torch()
    print(
        f"GPU: {torch.cuda.get_device_properties(0).gcnArchName}, {args.size}x"
        f"{args.size}, FP16, median of {args.runs} runs",
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix="chaiNNer-speed-") as temporary:
        for name in MODELS:
            model = load(name)
            print(
                f"{name} PyTorch: {pytorch_row(model, args.size, args.runs)}",
                flush=True,
            )
            if args.migraphx_driver is None:
                print(f"{name} MIGraphX: no migraphx-driver", flush=True)
                continue
            onnx = Path(temporary) / f"{name}.onnx"
            onnx.write_bytes(convert_to_onnx_impl(model, torch.device("cpu")))
            for fuse in (False, True):
                label = "attention fusion requested" if fuse else "default"
                row = migraphx_row(
                    args.migraphx_driver, onnx, args.size, args.runs, fuse
                )
                print(f"{name} MIGraphX ({label}): {row}", flush=True)


if __name__ == "__main__":
    main()
