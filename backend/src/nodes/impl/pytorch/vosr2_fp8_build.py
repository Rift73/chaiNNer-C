"""Isolated compilation of the downloaded FP8 graph set and an execution check."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("graphs", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("gpu", type=int)
    parser.add_argument("cache", type=Path)
    args = parser.parse_args()
    if os.name == "nt":
        ctypes.WinDLL("C:/Windows/System32/msvcp140.dll")
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from nodes.impl.pytorch.vosr2_process import end_with_parent

    end_with_parent()
    import tensorrt as trt
    import torch

    from api import Progress
    from nodes.impl.pytorch.vosr2 import VosrMemoryMode
    from nodes.impl.pytorch.vosr2_memory import vram_budget
    from nodes.impl.pytorch.vosr2_plan_file import plain
    from nodes.impl.pytorch.vosr2_trt_build import stage_inputs
    from nodes.impl.pytorch.vosr2_trt_compile import (
        compile_stage,
        report_stage,
        timing_cache,
    )
    from nodes.impl.pytorch.vosr2_trt_models import STAGES
    from nodes.impl.pytorch.vosr2_trt_pipeline import build_session
    from nodes.impl.pytorch.vosr2_trt_plugin import load_plugin

    load_plugin()
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    timing = timing_cache(args.cache, device)
    logger = trt.Logger(trt.Logger.WARNING)
    total = len(STAGES) + 1
    for index, name in enumerate(STAGES, 1):
        report_stage(index, total, f"Compiling FP8 set {name}")
        compile_stage(
            name,
            args.graphs / f"{name}.onnx",
            plain(args.directory, name),
            logger,
            timing,
        )
        torch.cuda.empty_cache()

    # The graphs are hash-verified and their quantization is lossy by design, so
    # there is no PyTorch reference here: check that every engine runs on this GPU.
    report_stage(total, total, "Checking every FP8 set stage and both DiT geometries")
    records = []
    generator = torch.Generator(device="cpu").manual_seed(937)
    session = build_session(args.directory, device)
    try:
        with torch.inference_mode(), vram_budget(
            device, VosrMemoryMode.DEFAULT, Progress.noop_progress()
        ) as check:
            session.activate(check)
            for name in STAGES:
                for width in (64, 72) if name == "dit_cached" else (0,):
                    values = stage_inputs(name, width, generator, device)
                    actual = session.stage(name, *values).float().clone()
                    repeated = session.stage(name, *values).float().clone()
                    if not bool(torch.isfinite(actual).all()):
                        raise RuntimeError(
                            f"VOSR FP8 {name} produced non-finite values."
                        )
                    if not torch.equal(actual, repeated):
                        raise RuntimeError(f"VOSR FP8 {name} repeatability failed.")
                    records.append(
                        {"stage": name, "width": width, "check": "finite-repeatable"}
                    )
                    check()
                    del values, actual, repeated
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
    print("All FP8 set VOSR stages checked", flush=True)


if __name__ == "__main__":
    main()
