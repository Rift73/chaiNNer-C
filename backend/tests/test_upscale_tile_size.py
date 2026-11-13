"""Upscale Image's Tile Size output: the tiles each backend's node ran, after failures
injected where the backend sees them. Tiny stand-ins on the CPU. TensorRT's session and
the TensorRT node's retry are in test_tensorrt_session.py."""

from __future__ import annotations

import math
from collections.abc import Collection
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from spandrel import ImageModelDescriptor
from spandrel.architectures.Compact import Compact, CompactArch

# Tiling runs in the native backend, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

from api import NodeContext, NodeId
from nodes.impl.onnx.model import SizeReq
from nodes.impl.upscale.auto_split_tiles import MAX_TILE_SIZE, NO_TILING, TileSize
from nodes.impl.upscale.basic_upscale import PaddingType
from nodes.impl.upscale.tile_report import TileReport
from nodes.utils.utils import Region
from packages.chaiNNer_ncnn.ncnn.processing import upscale_image as ncnn_node
from packages.chaiNNer_onnx.onnx.processing import upscale_image as onnx_node
from packages.chaiNNer_pytorch.pytorch.processing import upscale_image as torch_node


class Context:
    aborted = False
    paused = False
    storage_dir = None
    node_id = NodeId("upscale")

    def check_aborted(self) -> None:
        pass

    def set_progress(self, _progress: float) -> None:
        pass

    def add_cleanup(self, _fn: object, after: str = "chain") -> None:
        pass


CONTEXT = cast(NodeContext, Context())


def image(height: int, width: int, channels: int = 3) -> np.ndarray:
    return np.random.default_rng(height + width).random(
        (height, width, channels), np.float32
    )


def compact() -> ImageModelDescriptor:
    torch.manual_seed(3)
    net = Compact(num_in_ch=3, num_out_ch=3, num_feat=8, num_conv=2, upscale=2)
    model = CompactArch().load(net.state_dict())
    assert isinstance(model, ImageModelDescriptor)
    return model


@pytest.fixture
def torch_model() -> ImageModelDescriptor:
    return compact()


@pytest.fixture
def run_torch(monkeypatch: pytest.MonkeyPatch, torch_model: ImageModelDescriptor):
    settings = SimpleNamespace(
        use_fp16=False,
        device=torch.device("cpu"),
        budget_limit=0,
        use_cpu_bf16=False,
        force_cache_wipe=False,
    )
    monkeypatch.setattr(torch_node, "get_settings", lambda _: settings)

    def run(
        img: np.ndarray,
        tile_size: TileSize,
        custom_scale: int | None = None,
        separate_alpha: bool = False,
    ) -> tuple[np.ndarray, str]:
        return torch_node.upscale_image_node(
            CONTEXT,
            img,
            torch_model,
            custom_scale is not None,
            custom_scale or 2,
            tile_size,
            0,
            PaddingType.NONE,
            separate_alpha,
            False,  # Use TensorRT Engine
        )

    return run


def fail_forward(model: ImageModelDescriptor, failing: set[int]) -> None:
    """The model's calls numbered in `failing` run out of memory."""
    calls = [0]

    def counted(_module: torch.nn.Module, _args: object) -> None:
        calls[0] += 1
        if calls[0] in failing:
            raise torch.OutOfMemoryError("CUDA out of memory.")

    model.model.register_forward_pre_hook(counted)


def test_pytorch_whole_image(run_torch: Any):
    result, tile = run_torch(image(48, 64), MAX_TILE_SIZE)
    assert result.shape == (96, 128, 3)
    assert tile == "Whole image"


def test_pytorch_tiles_with_their_overlap(run_torch: Any):
    # 4 x 3 tiles of 16 px, with 16 px of overlap on each shared side
    _, tile = run_torch(image(48, 64), TileSize(16))
    assert tile == "48x48 px"
    # 2 x 2 tiles of 32 x 24 px, with an overlap on one side
    _, tile = run_torch(image(48, 64), TileSize(32))
    assert tile == "48x40 px"


def test_pytorch_out_of_memory_shows_the_smaller_tile(
    run_torch: Any, torch_model: ImageModelDescriptor
):
    img = image(48, 64)
    # The start's limit for this image is 74 px: 37 px after one failure, 18 after two.
    expected, expected_tile = run_torch(img, TileSize(18))
    fail_forward(torch_model, {1, 2})
    result, tile = run_torch(img, MAX_TILE_SIZE)
    assert tile == expected_tile == "48x48 px"
    np.testing.assert_array_equal(result, expected)


def test_pytorch_failure_after_a_finished_row_shows_the_later_tiles(
    run_torch: Any, torch_model: ImageModelDescriptor
):
    # 2 x 2 tiles at first: the second row's first tile fails, and the tiles go on smaller.
    fail_forward(torch_model, {3})
    _, tile = run_torch(image(48, 64), TileSize(37))
    assert tile == "48x48 px"


def test_pytorch_custom_scale_tells_of_the_last_pass(
    run_torch: Any, torch_model: ImageModelDescriptor
):
    # The first pass fits as a whole; the second, on the first's 60 x 40 px, does not.
    def fails_on_large_tiles(_module: torch.nn.Module, args: tuple) -> None:
        if max(args[0].shape) > 50:
            raise torch.OutOfMemoryError("CUDA out of memory.")

    torch_model.model.register_forward_pre_hook(fails_on_large_tiles)
    _, tile = run_torch(image(20, 30), MAX_TILE_SIZE, custom_scale=4)
    assert tile == "46x36 px"


@pytest.mark.parametrize("separate_alpha", [False, True])
def test_pytorch_passes_of_an_image_with_alpha(run_torch: Any, separate_alpha: bool):
    img = image(48, 64, 4)
    _, tile = run_torch(img, TileSize(16), separate_alpha=separate_alpha)
    assert tile == "48x48 px"


class NcnnExtractor:
    def __init__(self, net: NcnnNet) -> None:
        self.net = net
        self.value: Any = None

    def input(self, _name: str, value: Any) -> int:
        self.value = value
        return 0

    def extract(self, _name: str) -> tuple[int, Any]:
        self.net.extractions += 1
        if self.net.extractions in self.net.failing:
            return -100, None  # the allocation failed
        return 0, self.value


class NcnnNet:
    def __init__(self, failing: set[int]) -> None:
        self.failing = failing
        self.extractions = 0

    def create_extractor(self) -> NcnnExtractor:
        return NcnnExtractor(self)


def run_ncnn(
    monkeypatch: pytest.MonkeyPatch,
    img: np.ndarray,
    tile_size: TileSize,
    failing: Collection[int] = frozenset(),
) -> tuple[np.ndarray, str]:
    model = SimpleNamespace(
        in_nc=3,
        out_nc=3,
        scale=1,
        size_multiple=1,
        model=SimpleNamespace(layers=[SimpleNamespace(outputs=["data"])]),
    )
    monkeypatch.setattr(ncnn_node, "get_settings", lambda _: SimpleNamespace())
    monkeypatch.setattr(
        ncnn_node, "get_ncnn_net", lambda *_a, **_k: NcnnNet(set(failing))
    )
    return ncnn_node.upscale_image_node(
        CONTEXT, img, cast(Any, model), tile_size, 0, False
    )


def test_ncnn_whole_image_and_tiles(monkeypatch: pytest.MonkeyPatch):
    img = image(48, 64)
    assert run_ncnn(monkeypatch, img, MAX_TILE_SIZE)[1] == "Whole image"
    assert run_ncnn(monkeypatch, img, TileSize(16))[1] == "48x48 px"


def test_ncnn_out_of_memory_shows_the_smaller_tile(monkeypatch: pytest.MonkeyPatch):
    img = image(48, 64)
    _, expected = run_ncnn(monkeypatch, img, TileSize(18))
    _, tile = run_ncnn(monkeypatch, img, MAX_TILE_SIZE, {1, 2})
    assert tile == expected == "48x48 px"


class OnnxSession:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.sizes: list[tuple[int, ...]] = []

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="data", type="tensor(float)")]

    def get_outputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="out")]

    def run(self, _names: object, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        if self.error is not None:
            raise self.error
        self.sizes.append(inputs["data"].shape)
        return [inputs["data"]]


def run_onnx(
    monkeypatch: pytest.MonkeyPatch,
    session: OnnxSession,
    img: np.ndarray,
    tile_size: TileSize,
    exact: int | None = None,
) -> tuple[np.ndarray, str]:
    settings = SimpleNamespace(
        gpu_index=0,
        execution_provider="CPUExecutionProvider",
        tensorrt_fp16_mode=False,
        tensorrt_cache_path="",
        migraphx_fp16_mode=False,
        migraphx_cache_path=None,
        migraphx_exhaustive_tune=False,
    )
    monkeypatch.setattr(onnx_node, "get_settings", lambda _: settings)
    monkeypatch.setattr(onnx_node, "get_onnx_session", lambda *_: session)
    shape = ("BCHW", 3, exact, exact)
    monkeypatch.setattr(onnx_node, "get_input_shape", lambda _: shape)
    monkeypatch.setattr(onnx_node, "get_output_shape", lambda _: shape)
    monkeypatch.setattr(onnx_node, "get_size_req", lambda *_a, **_k: SizeReq())
    return onnx_node.upscale_image_node(
        CONTEXT, img, cast(Any, SimpleNamespace()), tile_size, 0, False
    )


def test_onnx_whole_image_and_tiles(monkeypatch: pytest.MonkeyPatch):
    img = image(48, 64)
    assert run_onnx(monkeypatch, OnnxSession(), img, NO_TILING)[1] == "Whole image"
    assert run_onnx(monkeypatch, OnnxSession(), img, TileSize(16))[1] == "48x48 px"


def test_onnx_exact_size_shows_the_models_size(monkeypatch: pytest.MonkeyPatch):
    session = OnnxSession()
    _, tile = run_onnx(monkeypatch, session, image(48, 64), NO_TILING, exact=32)
    assert tile == "32x32 px"
    assert {size[2:] for size in session.sizes} == {(32, 32)}


def test_onnx_out_of_memory_keeps_its_message(monkeypatch: pytest.MonkeyPatch):
    # ONNX Runtime's sessions have no way back from it: the tile size is the user's.
    session = OnnxSession(
        RuntimeError(
            "[ONNXRuntimeError] Failed to allocate memory for requested buffer"
        )
    )
    with pytest.raises(RuntimeError, match="more extreme tiling mode"):
        run_onnx(monkeypatch, session, image(48, 64), TileSize(16))


def test_a_tile_the_size_of_its_image_is_the_whole_image():
    report = TileReport()
    img = image(16, 16)
    report.recording(img, lambda tile, _region: tile)(img[:, :], Region(0, 0, 16, 16))
    assert report.text == "Whole image"


def measured(arch: str, width: int) -> Any:
    name = torch_node.WHOLE_IMAGE_WIDTH.get(arch, "embed_dim")
    return SimpleNamespace(
        architecture=SimpleNamespace(name=arch),
        model=SimpleNamespace(embed_dim=width, hyperparameters={name: width}),
    )


def test_auto_runs_a_measured_architecture_whole_when_it_fits():
    # A 4x HAT-L's 1200x864 image is 6.7 GiB at 6.8 KiB a pixel; upstream's estimate
    # split it in two of a 19.1 GiB budget.
    image = np.zeros((864, 1200, 3), np.float32)
    budget = 10 * 1024**3
    tiler = torch_node.whole_image_tiler(measured("HAT", 180), image, budget, True)
    assert tiler.starting_tile_size(1200, 864, 3) == (1200, 1200)
    # In FP32 it does not fit: the largest square that does.
    side = math.isqrt(int(budget / (1024 * 6.8 * 2)))
    tiler = torch_node.whole_image_tiler(measured("HAT", 180), image, budget, False)
    assert tiler.starting_tile_size(1200, 864, 3) == (side, side)


def test_a_narrower_model_takes_its_memory_in_proportion():
    image = np.zeros((1080, 1920, 3), np.float32)
    budget = 1080 * 1920 * 1024 * 2  # 2 KiB a pixel
    for arch, width, whole in (
        ("DUAL", 64, True),
        ("DUAL", 128, False),
        ("DAT", 60, True),
    ):
        tiler = torch_node.whole_image_tiler(measured(arch, width), image, budget, True)
        size = tiler.starting_tile_size(1920, 1080, 3)
        assert (size == (1920, 1920)) is whole, (arch, width)


GIB = 1024**3


@pytest.mark.parametrize(
    ("arch", "width", "kib"),
    [
        ("ESRGAN", 64, 1.4),
        ("FIGSR", 48, 1.8),
        ("HEART", 180, 5.6),
        ("GaterV3", 32, 1.4),
    ],
)
@pytest.mark.parametrize("card", [8, 12, 16])
def test_esrgan_figsr_and_heart_run_what_fits_whole_and_split_the_rest(
    arch: str, width: int, kib: float, card: int
):
    # Auto's budget: 80 % of 75 % of the card
    budget = int(int(card * GIB * 0.75) * 0.8)
    model = measured(arch, width)
    # the largest 16:9 image of this budget runs whole
    h = math.isqrt(int(budget / (1024 * kib) * 9 / 16))
    w = h * 16 // 9
    image = np.zeros((h, w, 3), np.float32)
    tiler = torch_node.whole_image_tiler(model, image, budget, True)
    assert tiler.starting_tile_size(w, h, 3) == (w, w)
    big = np.zeros((4320, 7680, 3), np.float32)
    side = math.isqrt(int(budget / (1024 * kib)))
    tiler = torch_node.whole_image_tiler(model, big, budget, True)
    assert tiler.starting_tile_size(7680, 4320, 3) == (side, side)


def test_the_width_comes_from_each_architectures_own_hyperparameter():
    assert torch_node.whole_image_bytes_per_pixel(measured("ESRGAN", 32), True) == (
        1024 * 1.4 / 2
    )
    assert torch_node.whole_image_bytes_per_pixel(measured("FIGSR", 96), False) == (
        1024 * 1.8 * 2 * 2
    )
    assert torch_node.whole_image_bytes_per_pixel(measured("HAT", 90), True) == (
        1024 * 6.8 / 2
    )
