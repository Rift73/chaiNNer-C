"""The AMD kit's speed table (native/AMD.md): for each route and size, the time of the
first call (a MIGraphX or ONNX Runtime route compiles then, unless the chains compiled
that program already: --programs is their Build Program cache), then the mean of
--iterations timed calls after a warm-up call, and the PSNR of the output against
PyTorch FP32 with the same model on this machine (VOSR: against its PyTorch backend).
Every route runs through the node functions the chains run, with this machine's
settings for its package, one image of the size, untiled. A route that fails gives
its row the error and its traceback to stderr; the others go on.

Usage (the test environment, from the checkout, with backend/src on the path):
  bench.py --models MODELS_TXT --image INPUT_PNG --storage DIR --programs DIR
           [--iterations 50] [--only REGEX]
Prints the report's section (Markdown)."""

from __future__ import annotations

import argparse
import gc
import math
import re
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from spandrel import ImageModelDescriptor

from api import NodeContext, NodeId, SettingsParser

if TYPE_CHECKING:
    from nodes.impl.onnx.model import OnnxGeneric

Step = Callable[[np.ndarray], np.ndarray]


class Context(NodeContext):
    """One node's context: its package's settings, storage and chain cleanups."""

    def __init__(self, settings: dict[str, object], node: str, storage: Path):
        self._settings = settings
        self._node = NodeId(node)
        self._storage = storage
        self.cleanups: list[Callable[[], None]] = []

    @property
    def settings(self) -> SettingsParser:
        return SettingsParser(self._settings)

    @property
    def node_id(self) -> NodeId:
        return self._node

    @property
    def storage_dir(self) -> Path:
        return self._storage

    @property
    def aborted(self) -> bool:
        return False

    @property
    def paused(self) -> bool:
        return False

    def set_progress(self, progress: float) -> None:
        pass

    def add_cleanup(self, fn: Callable[[], None], after: str = "chain") -> None:
        self.cleanups.append(fn)

    def finish(self) -> None:
        cleanups, self.cleanups = self.cleanups, []
        for cleanup in cleanups:
            cleanup()


@dataclass(frozen=True)
class Route:
    model: str
    name: str
    sizes: tuple[int, ...]
    reference: bool = False
    fp8: bool = False
    """Runs only on a GPU with FP8"""
    iterations: int | None = None
    """None: --iterations"""


BOTH = (256, 512)
SMALL = (256,)
ROUTES = [
    Route("esrgan", "PyTorch FP32", BOTH, reference=True),
    Route("esrgan", "PyTorch FP16", BOTH),
    Route("esrgan", "ONNX Runtime MIGraphX FP32", BOTH),
    Route("esrgan", "ONNX Runtime MIGraphX FP16", BOTH),
    Route("esrgan", "MIGraphX FP32", BOTH),
    Route("esrgan", "MIGraphX FP16", BOTH),
    Route("esrgan", "NCNN Vulkan", BOTH),
    Route("dual", "PyTorch FP32", BOTH, reference=True),
    Route("dual", "PyTorch FP16", BOTH),
    Route("dual", "MIGraphX", BOTH),
    Route("drft", "PyTorch FP32", BOTH, reference=True),
    Route("drft", "PyTorch FP16", BOTH),
    Route("drft", "MIGraphX", BOTH),
    Route("hat", "PyTorch FP32", SMALL, reference=True),
    Route("hat", "PyTorch FP16", SMALL),
    Route("hat", "MIGraphX FP16", SMALL),
    Route("dat2", "PyTorch FP32", SMALL, reference=True),
    Route("dat2", "PyTorch FP16", SMALL),
    Route("dat2", "MIGraphX FP16", SMALL),
    # A diffusion restorer: seconds per call, so fewer calls
    Route("vosr", "PyTorch", SMALL, reference=True, iterations=10),
    Route("vosr", "MIGraphX", SMALL, iterations=10),
    Route("vosr", "MIGraphX FP8", SMALL, fp8=True, iterations=10),
]


class Bench:
    def __init__(
        self, models: dict[str, Path], image: Path, storage: Path, programs: Path
    ):
        self.models = models
        source = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if source is None:
            raise FileNotFoundError(f"{image} is not an image OpenCV reads")
        self.source = np.asarray(source)
        self.storage = storage
        self.programs = programs
        self.nodes = 0
        self.pytorch_models: dict[str, ImageModelDescriptor] = {}
        self.onnx_models: dict[str, OnnxGeneric] = {}

    def context(self, settings: dict[str, object]) -> Context:
        self.nodes += 1
        return Context(settings, f"bench-{self.nodes}", self.storage)

    def image(self, size: int) -> np.ndarray:
        """The input image at size x size, as chaiNNer holds images (BGR float32)."""
        resized = cv2.resize(self.source, (size, size), interpolation=cv2.INTER_CUBIC)
        return resized.astype(np.float32) / 255

    def pytorch_model(self, role: str) -> ImageModelDescriptor:
        """The role's model as Load Model gives it, loaded once."""
        from packages.chaiNNer_pytorch.pytorch.io.load_model import load_model_node

        if role not in self.pytorch_models:
            model = load_model_node(self.context({}), self.models[role])[0]
            if not isinstance(model, ImageModelDescriptor):
                raise TypeError(f"{self.models[role]} is not an upscaling model")
            self.pytorch_models[role] = model
        return self.pytorch_models[role]

    def onnx_model(self, role: str) -> OnnxGeneric:
        """The role's model as Convert To ONNX gives it (FP32, its default opset; DUAL's
        and legacy DRFT's own exports), converted once."""
        from packages.chaiNNer_pytorch.pytorch.utility.convert_to_onnx import (
            DualShape,
            OnnxDataType,
            Opset,
            convert_to_onnx_node,
        )

        if role not in self.onnx_models:
            self.onnx_models[role] = convert_to_onnx_node(
                self.context({}),
                self.pytorch_model(role),
                OnnxDataType.FP32,
                Opset.OPSET_20,
                False,
                None,
                DualShape.DYNAMIC,
                512,
                512,
            )[0]
        return self.onnx_models[role]

    def step(self, route: Route, size: int, context_out: list[Context]) -> Step:
        """The route's call at this size; its contexts go to context_out for cleanup."""
        name = route.name
        if route.model == "vosr":
            return self.vosr_step(name, context_out)
        if name.startswith("PyTorch"):
            from nodes.impl.upscale.auto_split_tiles import NO_TILING
            from packages.chaiNNer_pytorch.pytorch.processing.upscale_image import (
                PaddingType,
                upscale_image_node,
            )

            context = self.context({"use_fp16": name.endswith("FP16")})
            context_out.append(context)
            model = self.pytorch_model(route.model)
            return lambda img: upscale_image_node(
                context, img, model, False, 4, NO_TILING, 0, PaddingType.NONE, False
            )
        if name.startswith("ONNX Runtime"):
            from nodes.impl.upscale.auto_split_tiles import NO_TILING
            from packages.chaiNNer_onnx.onnx.processing.upscale_image import (
                upscale_image_node,
            )

            context = self.context(
                {
                    "execution_provider": "MIGraphXExecutionProvider",
                    "migraphx_fp16_mode": name.endswith("FP16"),
                }
            )
            context_out.append(context)
            model = self.onnx_model(route.model)
            return lambda img: upscale_image_node(
                context, img, model, NO_TILING, 0, False
            )
        if name.startswith("MIGraphX"):
            import packages.chaiNNer_migraphx as migraphx_package
            from packages.chaiNNer_migraphx.migraphx.processing import upscale_image
            from packages.chaiNNer_migraphx.migraphx.utility import build_program

            from nodes.impl.tensorrt import cache as session_cache

            if migraphx_package.package.disabled:
                raise RuntimeError(
                    "the MIGraphX package is disabled:"
                    f" {migraphx_package.package.disabled_reason}"
                )

            settings: dict[str, object] = {"program_cache": str(self.programs)}
            build_context = self.context(settings)
            program = build_program.build_program_node(
                build_context,
                self.onnx_model(route.model),
                build_program.ProgramPrecision.FP32
                if name.endswith("FP32")
                else build_program.ProgramPrecision.FP16,
                size,
                size,
                False,
            )
            context = self.context(settings)
            context.add_cleanup(lambda: session_cache.release_node(context.node_id))
            context_out.extend([build_context, context])
            return lambda img: upscale_image.upscale_image_node(
                context, img, program, False
            )
        if name == "NCNN Vulkan":
            from nodes.impl.upscale.auto_split_tiles import NO_TILING
            from packages.chaiNNer_ncnn.ncnn.processing.upscale_image import (
                upscale_image_node,
            )
            from packages.chaiNNer_onnx.onnx.utility.convert_to_ncnn import (
                convert_to_ncnn_node,
            )

            model = convert_to_ncnn_node(self.onnx_model(route.model), 0)[0]
            context = self.context({})
            context_out.append(context)
            return lambda img: upscale_image_node(
                context, img, model, NO_TILING, 0, False
            )
        raise ValueError(f"no route {name}")

    def vosr_step(self, name: str, context_out: list[Context]) -> Step:
        from nodes.utils.seed import Seed
        from packages.chaiNNer_pytorch.pytorch.processing import (
            upscale_image_vosr as vosr,
        )

        backend = {
            "PyTorch": vosr.VosrBackend.PYTORCH,
            "MIGraphX": vosr.VosrBackend.MIGRAPHX,
            "MIGraphX FP8": vosr.VosrBackend.MIGRAPHX_FP8,
        }[name]
        context = self.context({})
        context_out.append(context)
        return lambda img: vosr.upscale_image_vosr_node(
            context,
            img,
            2,
            Seed(42),
            vosr.VosrColorCorrection.WAVELET,
            vosr.VosrMemoryMode.DEFAULT,
            backend,
            vosr.TemporalMode.THREE_FRAMES,
            65,
            True,
        )


def psnr(reference: np.ndarray, output: np.ndarray) -> float:
    difference = np.clip(reference, 0, 1).astype(np.float64) - np.clip(output, 0, 1)
    mse = float(np.mean(difference**2))
    return math.inf if mse == 0 else 10 * math.log10(1 / mse)


def release() -> None:
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description="The AMD kit's speed table.")
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument("--programs", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--only",
        default="",
        help="a regular expression: only the routes whose names (esrgan-pytorch-fp32,"
        " esrgan-ncnn-vulkan, ...) it matches",
    )
    args = parser.parse_args()
    models: dict[str, Path] = {}
    for line in args.models.read_text(encoding="utf-8").splitlines():
        role, path = line.split(" ", 1)
        models[role] = Path(path)
    bench = Bench(models, args.image, args.storage, args.programs)

    print("## Speed")
    print()
    print(
        f"One image per call, untiled; {args.iterations} timed calls after a warm-up call"
        " (VOSR: 10). PSNR against PyTorch FP32 with the same model on this machine"
        " (VOSR: against its PyTorch backend, same seed)."
    )
    print()
    print("| Model | Route | Size | First call (s) | ms per call | PSNR (dB) |")
    print("|---|---|---|---|---|---|")
    import gpu

    fp8 = any(device.supports_fp8 for device in gpu.amd.devices)
    references: dict[tuple[str, int], np.ndarray] = {}
    failures = 0
    for route in ROUTES:
        name = f"{route.model}-{route.name.lower().replace(' ', '-')}"
        if args.only and not re.search(args.only, name):
            print(f"| {route.model} | {route.name} | | | | not in --only |")
            continue
        if route.model not in models:
            print(f"| {route.model} | {route.name} | | | | not in models.txt |")
            continue
        if route.fp8 and not fp8:
            print(f"| {route.model} | {route.name} | | | | this GPU has no FP8 |")
            continue
        for size in route.sizes:
            contexts: list[Context] = []
            row = f"| {route.model} | {route.name} | {size}x{size} |"
            try:
                image = bench.image(size)
                start = time.perf_counter()
                step = bench.step(route, size, contexts)
                output = step(image)
                first = time.perf_counter() - start
                step(image)
                iterations = route.iterations or args.iterations
                start = time.perf_counter()
                for _ in range(iterations):
                    step(image)
                milliseconds = (time.perf_counter() - start) / iterations * 1000
                if route.reference:
                    references[(route.model, size)] = output
                    quality = "reference"
                elif (route.model, size) in references:
                    quality = f"{psnr(references[(route.model, size)], output):.2f}"
                else:
                    quality = "no reference"
                print(
                    f"{row} {first:.1f} | {milliseconds:.1f} | {quality} |", flush=True
                )
            except Exception as error:
                failures += 1
                message = " ".join(f"{type(error).__name__}: {error}".split())[:300]
                print(f"{row} | | FAILED: {message} |", flush=True)
                print(f"{route.model} {route.name} {size}:", file=sys.stderr)
                traceback.print_exc()
            finally:
                for context in contexts:
                    context.finish()
                release()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
