"""Isolated all-stage BF16 export, compilation and numerical validation."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def stage_inputs(
    name: str, width: int, generator: torch.Generator, device: torch.device
) -> list[torch.Tensor]:
    """Seeded inputs at the profile's smallest shapes; DiT at one window width."""
    import torch

    from nodes.impl.pytorch.vosr2_export import rotary_tables
    from nodes.impl.pytorch.vosr2_trt_models import profiles

    values = [
        torch.randn(shapes[0], generator=generator).bfloat16().to(device)
        for shapes in profiles(name).values()
    ]
    if name == "dino":
        values[0] = values[0].float()
    if name == "dit_cached":
        values[0] = (
            torch.randn((1, 32, 64, width), generator=generator).bfloat16().to(device)
        )
        values[2], values[3] = rotary_tables(width, device, torch.bfloat16)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("gpu", type=int)
    parser.add_argument("cache", type=Path)
    args = parser.parse_args()
    if os.name == "nt":
        ctypes.WinDLL("C:/Windows/System32/msvcp140.dll")
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from nodes.impl.pytorch.vosr2_process import end_with_parent

    end_with_parent()
    import onnx
    import tensorrt as trt
    import torch

    from api import Progress
    from nodes.impl.pytorch.vosr2 import BundlePaths, VosrBundle, VosrMemoryMode
    from nodes.impl.pytorch.vosr2_memory import vram_budget
    from nodes.impl.pytorch.vosr2_plan_file import plain
    from nodes.impl.pytorch.vosr2_trt_compile import (
        compile_stage,
        report_stage,
        timing_cache,
    )
    from nodes.impl.pytorch.vosr2_trt_models import STAGES, export_model, profiles
    from nodes.impl.pytorch.vosr2_trt_pipeline import build_session
    from nodes.impl.pytorch.vosr2_trt_plugin import load_plugin
    from nodes.impl.pytorch.vosr2_trt_rewrite import rewrite, split_dit

    load_plugin()
    torch.set_num_threads(4)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    timing = timing_cache(args.cache, device)
    progress = Progress.noop_progress()
    bundle = VosrBundle(BundlePaths(args.bundle), torch.float32, progress)
    logger = trt.Logger(trt.Logger.WARNING)
    export_names = ("dit", *(name for name in STAGES if not name.startswith("dit_")))
    total = len(export_names) + len(STAGES) + 1
    for index, name in enumerate(export_names, 1):
        report_stage(index, total, f"Exporting BF16 {name}")
        model = export_model(name, bundle).eval()
        ranges = profiles(name)
        dummy = tuple(
            torch.zeros(value[0], dtype=torch.bfloat16) for value in ranges.values()
        )
        if name == "dino":
            dummy = (dummy[0].float(),)
        if name == "dit":
            dummy = (
                torch.zeros(1, 32, 64, 2, dtype=torch.bfloat16),
                torch.zeros(1, 1024, 1024, dtype=torch.bfloat16),
                torch.zeros(32, 64, dtype=torch.bfloat16),
                torch.zeros(32, 64, dtype=torch.bfloat16),
            )
        axes = {
            key: {
                axis: f"{key}_{axis}"
                for axis in range(len(shapes[0]))
                if shapes[0][axis] != shapes[2][axis]
            }
            for key, shapes in ranges.items()
        }
        axes["result"] = (
            {2: "result_height", 3: "result_width"} if name != "dino" else {}
        )
        graph_path = args.directory / f"{name}.onnx"
        with torch.inference_mode():
            torch.onnx.export(
                model,
                dummy,
                str(graph_path),
                opset_version=20,
                dynamo=False,
                external_data=True,
                input_names=list(ranges),
                output_names=["result"],
                dynamic_axes=axes,
            )
        del model, dummy
        if name == "dit":
            projection, body = split_dit(
                onnx.load(graph_path, load_external_data=False)
            )
            onnx.save(projection, args.directory / "dit_kv.onnx")
            onnx.save(body, args.directory / "dit_cached.onnx")
            rewrite(args.directory / "dit_cached.onnx", "dit_cached")
            del projection, body
        elif name == "dino":
            rewrite(graph_path, name)

    for index, name in enumerate(STAGES, len(export_names) + 1):
        report_stage(index, total, f"Compiling BF16 {name}")
        compile_stage(
            name,
            args.directory / f"{name}.onnx",
            plain(args.directory, name),
            logger,
            timing,
        )
        torch.cuda.empty_cache()

    report_stage(
        total, total, "Validating BF16 stage boundaries and both DiT geometries"
    )
    records = []
    generator = torch.Generator(device="cpu").manual_seed(937)
    session = build_session(args.directory, device)
    try:
        with torch.inference_mode(), vram_budget(
            device, VosrMemoryMode.DEFAULT, progress
        ) as check:
            session.activate(check)
            for name in STAGES:
                model = export_model(name, bundle).eval().to(device)
                geometries = (64, 72) if name == "dit_cached" else (0,)
                for width in geometries:
                    values = stage_inputs(name, width, generator, device)
                    expected = model(*values).float()
                    fp32_relative = None
                    fp32 = None
                    if name == "dino":
                        bundle.dino.to(device)
                        fp32 = bundle.dino.forward_intermediate_layer(
                            values[0], 17
                        ).float()
                    actual = session.stage(name, *values).float().clone()
                    repeated = session.stage(name, *values).float().clone()
                    relative = float(
                        torch.linalg.vector_norm(actual - expected)
                        / torch.linalg.vector_norm(expected).clamp_min(1e-9)
                    )
                    print(
                        json.dumps(
                            {
                                "stage": name,
                                "width": width,
                                "mixed_relative_l2": relative,
                                "fp32_relative_l2": float(
                                    torch.linalg.vector_norm(actual - fp32)
                                    / torch.linalg.vector_norm(fp32).clamp_min(1e-9)
                                )
                                if fp32 is not None
                                else None,
                            }
                        ),
                        flush=True,
                    )
                    if not bool(torch.isfinite(actual).all()) or relative > 0.03:
                        raise RuntimeError(
                            f"VOSR {name} numerical validation failed: {relative}"
                        )
                    if name == "dino":
                        assert fp32 is not None
                        fp32_relative = float(
                            torch.linalg.vector_norm(actual - fp32)
                            / torch.linalg.vector_norm(fp32).clamp_min(1e-9)
                        )
                        # This quantized white-noise stress input is deliberately
                        # unlike an image. The accepted mixed engine differs by
                        # 3.6% here and <0.9% on the representative image crops.
                        # Keep export fidelity (above) separate from precision
                        # drift; image detail/temporal quality is validated too.
                        if fp32_relative > 0.05:
                            raise RuntimeError(
                                f"VOSR DINO FP32 drift failed: {fp32_relative}"
                            )
                        del fp32
                    if not torch.equal(actual, repeated):
                        raise RuntimeError(f"VOSR {name} repeatability failed.")
                    records.append(
                        {
                            "stage": name,
                            "width": width,
                            "relative_l2": relative,
                            "reference": "mixed-pytorch",
                            "fp32_relative_l2": fp32_relative,
                        }
                    )
                    check()
                    del values, expected, actual, repeated
                del model
                bundle.dino.cpu()
                torch.cuda.empty_cache()
    finally:
        session.close()
    engines = {}
    for name in STAGES:
        with plain(args.directory, name).open("rb") as file:
            engines[name] = hashlib.file_digest(file, "sha256").hexdigest()
    (args.directory / "validation.json").write_text(
        json.dumps({"engines": engines, "profiles": records}, indent=2),
        encoding="utf-8",
    )
    print("All BF16 VOSR stages validated", flush=True)


if __name__ == "__main__":
    main()
