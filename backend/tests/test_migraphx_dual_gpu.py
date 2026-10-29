"""DUAL on an AMD GPU through MIGraphX: Build Program's DUAL route (migraphx/dual.py)
compiles a DUAL ONNX, fixed and dynamic, with DUAL's custom ops and their kernels; the
runner upscales an image with the program as PyTorch does in FP32, within DUAL's BF16
rounding; and a new process runs a DUAL program from the Program Cache's file without
building it again.
They need an AMD GPU, PyTorch on ROCm with Triton, the runner's module, DUAL's op
library and MIGraphX: without an AMD GPU they are skipped, with one anything missing
fails them."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from spandrel import ImageModelDescriptor, ModelLoader
from spandrel.architectures.DUAL.__arch.dual_arch import dual_xs

import gpu
from nodes.impl.migraphx import build_cache, dual
from nodes.impl.migraphx.model import MIGraphXProgram
from nodes.impl.migraphx.program_info import read_program_info
from nodes.impl.migraphx.runner import runner_identity
from nodes.impl.migraphx.session import RunnerSession
from nodes.impl.onnx.load import load_onnx_model

pytestmark = pytest.mark.amd_gpu

# One program size; past both context windows (M32, M64), so every context site is
# neighborhood attention
SIZE = (64, 96)


@pytest.fixture(scope="module")
def device() -> gpu.AmdDevice:
    if not gpu.amd.is_available:
        pytest.skip("no AMD GPU")
    return gpu.amd.devices[0]


@pytest.fixture(scope="module")
def model(tmp_path_factory: pytest.TempPathFactory) -> ImageModelDescriptor:
    """A folded DUAL XS x2 whose zero-initialized parameters (the dynamic
    convolutions' experts among them) are made nonzero, so no branch hides at zero."""
    torch.manual_seed(0)
    network = dual_xs(scale=2).eval()
    with torch.no_grad():
        for parameter in network.parameters():
            if not parameter.any():
                parameter.normal_(std=0.02)
    path = tmp_path_factory.mktemp("dual") / "xs.pth"
    torch.save(network.state_dict(), path)
    descriptor = ModelLoader("cpu").load_from_file(path)
    assert isinstance(descriptor, ImageModelDescriptor)
    return descriptor


def image() -> np.ndarray:
    """A float32 BGR image of SIZE: smooth gradients plus noise."""
    h, w = SIZE
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    rng = np.random.default_rng(3)
    planes = [x / w, y / h, (x + y) / (w + h)]
    return np.clip(np.stack(planes, -1) + rng.normal(0, 0.05, (h, w, 3)), 0, 1).astype(
        np.float32
    )


def pytorch(model: ImageModelDescriptor, img: np.ndarray, dtype: torch.dtype):
    """The model's output on the GPU, in `dtype`, as a float32 BGR image."""
    network = model.model.to("cuda", dtype)
    rgb = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1]))
    with torch.inference_mode():
        out = network(rgb.permute(2, 0, 1)[None].to("cuda", dtype)).float()
    return out[0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1]


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    return float(10 * np.log10(1 / np.mean((a.astype(np.float64) - b) ** 2)))


def program_of(
    model: ImageModelDescriptor, device: gpu.AmdDevice, dynamic: bool
) -> MIGraphXProgram:
    onnx_bytes = dual.export_onnx(
        model.model.state_dict(), model.tags, model.scale, None if dynamic else SIZE
    )
    manifest = dual.export_manifest(onnx_bytes)
    assert manifest is not None
    if dynamic:
        dual.check_profile(manifest, SIZE, SIZE, SIZE)
    saved = dual.build_program(
        onnx_bytes, manifest, device.index, SIZE if dynamic else None
    )
    return MIGraphXProgram(saved, read_program_info(saved, device.index))


def upscale(program: MIGraphXProgram, device: gpu.AmdDevice) -> np.ndarray:
    session = RunnerSession(program, device.index)
    try:
        return session.upscale(image())
    finally:
        session.close()


@pytest.mark.parametrize("dynamic", [False, True])
def test_a_dual_program_upscales_as_pytorch_in_fp32(
    model: ImageModelDescriptor, device: gpu.AmdDevice, dynamic: bool
):
    program = program_of(model, device, dynamic)
    info = program.info
    assert (info.precision, info.scale, info.gfx) == ("mixed", 2, device.gfx)
    assert (info.height, info.width) == SIZE
    out = upscale(program, device)
    assert out.shape == (SIZE[0] * 2, SIZE[1] * 2, 3)
    assert np.isfinite(out).all()
    fp32 = pytorch(model, image(), torch.float32)
    # The guide's acceptance: at most twice the error of the model run in BF16 (6 dB).
    floor = psnr(pytorch(model, image(), torch.bfloat16), fp32)
    assert psnr(out, fp32) >= floor - 6, (psnr(out, fp32), floor)


# A new process: the Program Cache's file is its only copy of the program, which must
# not be built again and carries DUAL's kernels (its ops' library loads first).
FRESH_PROCESS = """
import sys

import numpy as np

source, key, directory, index, image, output = sys.argv[1:]
sys.path.insert(0, source)
from nodes.impl.migraphx import build_cache
from nodes.impl.migraphx.session import RunnerSession


def build():
    raise AssertionError("the cached program was built again")


session = RunnerSession(build_cache.cached_program(key, build, directory), int(index))
try:
    np.save(output, session.upscale(np.load(image)))
finally:
    session.close()
"""


def test_a_new_process_runs_the_cached_dual_program(
    model: ImageModelDescriptor, device: gpu.AmdDevice, tmp_path: Path
):
    onnx_bytes = dual.export_onnx(
        model.model.state_dict(), model.tags, model.scale, SIZE
    )
    key = build_cache.program_key(
        load_onnx_model(onnx_bytes),
        ("dual", None, False),
        device.name,
        device.gfx,
        runner_identity(),
        dual.build_identity(),
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    built = build_cache.cached_program(
        key, lambda: program_of(model, device, dynamic=False), str(cache)
    )
    np.save(tmp_path / "image.npy", image())
    script = tmp_path / "fresh_process.py"
    script.write_text(FRESH_PROCESS, encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            str(script),
            str(dual.SOURCE),
            key,
            str(cache),
            str(device.index),
            str(tmp_path / "image.npy"),
            str(tmp_path / "output.npy"),
        ],
        check=True,
        timeout=600,
    )
    np.testing.assert_array_equal(
        np.load(tmp_path / "output.npy"), upscale(built, device)
    )
