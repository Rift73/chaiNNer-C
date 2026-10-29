"""Isolated BF16 export of VOSR's DINO and DiT graphs, their MIGraphX compilation and
numerical validation: vosr2_trt_build.py's counterpart for AMD GPUs (its VAE stages run in
PyTorch, vosr2_migx.py), and vosr2_fp8_build.py's. worker <bundle> <directory> <gpu>
<cache> <precision>: bf16 (the graphs' types), fp16 (FP16 for BF16) or fp8 (MIGraphX's FP8
quantization of TensorRT's FP8 recipe products, calibrated here, for a GPU with FP8 matrix
math); vosr2_migx_compile.compile_program.

The export is vosr2_trt_build.py's for these two graphs, which that file keeps inline: it
is part of the TensorRT engines' identity, so moving its loop here would rebuild every
user's engines."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import torch

    from nodes.impl.pytorch.vosr2 import VosrBundle

# The TensorRT engines' bounds (vosr2_trt_build.py): each stage's relative L2 from
# PyTorch BF16, and DINO's from FP32
BOUND = 0.03
DINO_FP32_BOUND = 0.05


def export_graph(name: str, bundle: VosrBundle, directory: Path) -> None:
    """Export `name` (dit or dino) as vosr2_trt_build.py does, with its rewrite: the DiT
    split into dit_kv and dit_cached (with the plugin node vosr2_migx_compile turns into
    the custom op), DINO with BF16 attention."""
    import onnx
    import torch

    from nodes.impl.pytorch.vosr2_trt_models import export_model, profiles
    from nodes.impl.pytorch.vosr2_trt_rewrite import rewrite, split_dit

    model = export_model(name, bundle).eval()
    ranges = profiles(name)
    if name == "dino":
        dummy = (torch.zeros(ranges["x"][0], dtype=torch.float32),)
    else:
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
    axes["result"] = {2: "result_height", 3: "result_width"} if name != "dino" else {}
    graph_path = directory / f"{name}.onnx"
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
        projection, body = split_dit(onnx.load(graph_path, load_external_data=False))
        onnx.save(projection, directory / "dit_kv.onnx")
        onnx.save(body, directory / "dit_cached.onnx")
        rewrite(directory / "dit_cached.onnx", "dit_cached")
    else:
        rewrite(graph_path, name)


def calibration_images() -> list[np.ndarray]:
    """Four deterministic RGB images in [0, 1], 512 x 576 (one window with its context
    strip): soft gradients and blobs, sharp shapes and thin lines, fine 1/f texture, and
    the three mixed. FP8 calibration takes each quantized input's largest value from
    them."""
    import numpy as np

    rng = np.random.default_rng(937)
    y, x = np.mgrid[0:512, 0:576] / 512

    def blobs() -> np.ndarray:
        image = np.stack((x, y, 1 - x * y), -1) * 0.6
        for _ in range(12):
            cy, cx, radius = rng.uniform(0, 1.1, 3)
            weight = np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (0.02 + radius / 20))
            image += weight[..., None] * rng.uniform(-0.5, 0.5, 3)
        return image

    def shapes() -> np.ndarray:
        image = np.ones((512, 576, 3)) * rng.uniform(0, 1, 3)
        for _ in range(40):
            top, left = rng.integers(0, 480, 2)
            height, width = rng.integers(2, 160, 2)
            image[top : top + height, left : left + width] = rng.uniform(0, 1, 3)
        for row in rng.integers(0, 512, 20):
            image[row, :] = rng.uniform(0, 1, 3)
        return image

    def texture() -> np.ndarray:
        spectrum = np.fft.fft2(rng.standard_normal((3, 512, 576)))
        frequency = np.hypot(np.fft.fftfreq(512)[:, None], np.fft.fftfreq(576)[None])
        image = np.fft.ifft2(spectrum / np.maximum(frequency, 1 / 576)).real
        image = image.transpose(1, 2, 0)
        return 0.5 + 0.18 * image / image.std()

    parts = [blobs(), shapes(), texture()]
    mask = (np.sin(x * 9) * np.cos(y * 7) > 0)[..., None]
    parts.append(np.where(mask, parts[1], parts[0]) + 0.3 * (parts[2] - 0.5))
    return [np.clip(part, 0, 1).astype(np.float32) for part in parts]


def calibration_inputs(
    bundle: VosrBundle, device: torch.device
) -> dict[str, list[dict[str, np.ndarray]]]:
    """Each program's inputs for the calibration images, made as restoration makes them:
    the window's DINO crop, its conditioning, the DiT's K/V, and the latent window with
    seeded noise, without (64 columns) and with (72) the context strip."""
    import torch
    from onnx import TensorProto, helper
    from torch.nn import functional

    from nodes.impl.pytorch.vosr2_export import rotary_tables
    from nodes.impl.pytorch.vosr2_trt_models import export_model

    bf16 = helper.tensor_dtype_to_np_dtype(TensorProto.BFLOAT16)

    def host(tensor: torch.Tensor) -> np.ndarray:
        tensor = tensor.contiguous().cpu()
        if tensor.dtype == torch.bfloat16:
            return tensor.view(torch.uint16).numpy().view(bf16)
        return tensor.numpy()

    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    tables = {width: rotary_tables(width, device, torch.bfloat16) for width in (64, 72)}
    projection = export_model("dit_kv", bundle).eval().to(device)
    bundle.vae.to(device)
    bundle.dino.to(device)
    inputs: dict[str, list[dict[str, np.ndarray]]] = {
        name: [] for name in ("dino", "dit_kv", "dit_cached_64", "dit_cached_72")
    }
    generator = torch.Generator(device="cpu").manual_seed(937)
    try:
        with torch.inference_mode():
            for picture in calibration_images():
                image = torch.from_numpy(picture).permute(2, 0, 1)[None].to(device)
                image = image * 2 - 1
                latent = bundle.vae.encode(image).latent_dist.mode()
                latent = (latent - bundle.mean.to(device)) / bundle.std.to(device)
                noise = torch.randn(latent.shape, generator=generator).to(device)
                window = torch.cat((latent, noise), dim=1).bfloat16()
                crop = functional.interpolate(
                    image[:, :, :, :512] * 0.5 + 0.5,
                    size=(448, 448),
                    mode="bicubic",
                    align_corners=False,
                ).clamp_(0, 1)
                dino = ((crop - mean) / std).float()
                conditioning = bundle.dino.forward_intermediate_layer(dino, 17)
                conditioning = conditioning.bfloat16()
                kv = projection(conditioning)
                inputs["dino"].append({"x": host(dino)})
                inputs["dit_kv"].append({"conditioning": host(conditioning)})
                for width, (cos, sin) in tables.items():
                    inputs[f"dit_cached_{width}"].append(
                        {
                            "latent": host(window[:, :, :, :width]),
                            "kv": host(kv),
                            "cos": host(cos),
                            "sin": host(sin),
                        }
                    )
    finally:
        bundle.vae.cpu()
        bundle.dino.cpu()
        del projection
        torch.cuda.empty_cache()
    return inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("gpu", type=int)
    # compile_worker's protocol: the folder of the set's builds (MIGraphX keeps no
    # timing cache there)
    parser.add_argument("cache", type=Path)
    parser.add_argument("precision", choices=("bf16", "fp16", "fp8"))
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from nodes.impl.pytorch.vosr2_process import end_with_parent

    end_with_parent()
    import torch

    from api import Progress
    from nodes.impl.migraphx.vosr2_migx import ProgramStages
    from nodes.impl.migraphx.vosr2_migx_compile import (
        PROGRAMS,
        compile_program,
        load_ops,
    )
    from nodes.impl.pytorch.vosr2 import BundlePaths, VosrBundle, VosrMemoryMode
    from nodes.impl.pytorch.vosr2_memory import vram_budget
    from nodes.impl.pytorch.vosr2_plan_file import plain
    from nodes.impl.pytorch.vosr2_trt_build import stage_inputs
    from nodes.impl.pytorch.vosr2_trt_compile import report_stage
    from nodes.impl.pytorch.vosr2_trt_execution import StageExecution
    from nodes.impl.pytorch.vosr2_trt_models import export_model

    load_ops()
    precision = args.precision
    label = precision.upper()
    torch.set_num_threads(4)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    progress = Progress.noop_progress()
    bundle = VosrBundle(BundlePaths(args.bundle), torch.float32, progress)
    exports = ("dit", "dino")
    calibrate = int(precision == "fp8")
    total = len(exports) + calibrate + len(PROGRAMS) + 1
    for index, name in enumerate(exports, 1):
        report_stage(index, total, f"Exporting BF16 {name}")
        export_graph(name, bundle, args.directory)
    calibration = {}
    if calibrate:
        report_stage(len(exports) + 1, total, "Making FP8 calibration inputs")
        calibration = calibration_inputs(bundle, device)
    for index, name in enumerate(PROGRAMS, len(exports) + calibrate + 1):
        report_stage(index, total, f"Compiling MIGraphX {label} {name}")
        compile_program(
            name,
            args.directory,
            plain(args.directory, name),
            args.gpu,
            precision,
            calibration.get(name, ()),
        )
        torch.cuda.empty_cache()

    report_stage(total, total, f"Checking the {label} programs, both DiT geometries")
    records = []
    generator = torch.Generator(device="cpu").manual_seed(937)
    stream = torch.cuda.Stream(device=device)
    programs = ProgramStages(args.directory, device, StageExecution(device, stream))
    try:
        with torch.inference_mode(), vram_budget(
            device, VosrMemoryMode.DEFAULT, progress
        ) as check:
            with torch.cuda.stream(stream):
                for name in ("dit_kv", "dit_cached", "dino"):
                    model = export_model(name, bundle).eval().to(device)
                    for width in (64, 72) if name == "dit_cached" else (0,):
                        values = stage_inputs(name, width, generator, device)
                        expected = model(*values).float()
                        actual = programs.run(name, values, stream).float().clone()
                        repeated = programs.run(name, values, stream).float().clone()
                        relative = float(
                            torch.linalg.vector_norm(actual - expected)
                            / torch.linalg.vector_norm(expected).clamp_min(1e-9)
                        )
                        fp32_relative = None
                        if name == "dino":
                            bundle.dino.to(device)
                            fp32 = bundle.dino.forward_intermediate_layer(
                                values[0], 17
                            ).float()
                            fp32_relative = float(
                                torch.linalg.vector_norm(actual - fp32)
                                / torch.linalg.vector_norm(fp32).clamp_min(1e-9)
                            )
                            bundle.dino.cpu()
                            del fp32
                        record = {
                            "stage": name,
                            "width": width,
                            "precision": precision,
                            "relative_l2": relative,
                            "reference": "mixed-pytorch",
                            "fp32_relative_l2": fp32_relative,
                        }
                        print(json.dumps(record), flush=True)
                        if not bool(torch.isfinite(actual).all()):
                            raise RuntimeError(
                                f"VOSR MIGraphX {label} {name} produced non-finite "
                                "values."
                            )
                        # FP8 is lossy by design, as TensorRT's: finite and repeatable
                        if precision != "fp8" and relative > BOUND:
                            raise RuntimeError(
                                f"VOSR MIGraphX {label} {name} numerical validation "
                                f"failed: {relative}"
                            )
                        if (
                            precision != "fp8"
                            and fp32_relative is not None
                            and fp32_relative > DINO_FP32_BOUND
                        ):
                            raise RuntimeError(
                                f"VOSR MIGraphX {label} DINO FP32 drift failed: "
                                f"{fp32_relative}"
                            )
                        if not torch.equal(actual, repeated):
                            raise RuntimeError(
                                f"VOSR MIGraphX {label} {name} repeatability failed."
                            )
                        records.append(record)
                        check()
                        del values, expected, actual, repeated
                    del model
                    torch.cuda.empty_cache()
    finally:
        stream.synchronize()
        programs.close()
    engines = {}
    for name in PROGRAMS:
        with plain(args.directory, name).open("rb") as file:
            engines[name] = hashlib.file_digest(file, "sha256").hexdigest()
    (args.directory / "validation.json").write_text(
        json.dumps({"engines": engines, "profiles": records}, indent=2),
        encoding="utf-8",
    )
    print(f"All {label} VOSR MIGraphX programs checked", flush=True)


if __name__ == "__main__":
    main()
