"""Each backend interleaves its model's output planes once, natively
(planar_to_interleaved), into a C-ordered image whose bytes are those of the steps it
replaces: PyTorch's device permute and flip and CPU cast, ONNX's transpose, red/blue
swap, crop and cast, NCNN's cast and the node's cvtColor. Tiny models on the CPU; each
oracle is the same module with its interleave turned off (today's path)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import cv2
import numpy as np
import pytest
import torch
from spandrel import ImageModelDescriptor
from spandrel.architectures.Compact import Compact, CompactArch

# Tiling runs in the native backend, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

from onnx import TensorProto, helper

from api import NodeContext, Progress
from nodes.impl.native_onnx_runtime import NativeSession
from nodes.impl.native_tensors import cast_numpy
from nodes.impl.ncnn import auto_split as ncnn_split
from nodes.impl.ncnn.model import NcnnModel, NcnnModelWrapper
from nodes.impl.ncnn.session import create_ncnn_net
from nodes.impl.onnx import auto_split as onnx_split
from nodes.impl.onnx.model import SizeReq
from nodes.impl.pytorch import auto_split as torch_split
from nodes.impl.upscale.auto_split_tiles import NO_TILING, TileSize
from nodes.impl.upscale.convenient_upscale import convenient_upscale
from nodes.impl.upscale.tiler import ExactTileSize, MaxTileSize, NoTiling, Tiler
from nodes.utils.utils import get_h_w_c
from packages.chaiNNer_ncnn.ncnn.processing import upscale_image as ncnn_node
from packages.chaiNNer_ncnn.settings import NcnnSettings

CPU = torch.device("cpu")


def assert_same(new: np.ndarray, old: np.ndarray) -> None:
    """The same values bit for bit, NaN payloads included; new is C-ordered."""
    assert new.shape == old.shape and new.dtype == old.dtype == np.float32
    assert new.flags.c_contiguous
    assert new.tobytes() == np.ascontiguousarray(old).tobytes()


def image(shape: tuple[int, ...], seed: int, exceptional: bool) -> np.ndarray:
    img = np.random.default_rng(seed).random(shape, dtype=np.float32)
    if exceptional:
        flat = img.reshape(-1).view(np.uint32)
        # NaN (a payload), +-inf, -0 and an out-of-range value
        words = (0x7FC12345, 0x7F800000, 0xFF800000, 0x80000000, 0x3FC00000)
        flat[[3, 17, 29, 41, 53]] = words
    return img


def tiler(tiled: bool) -> Tiler:
    return ExactTileSize((16, 16)) if tiled else NoTiling()


def compact(channels: int) -> ImageModelDescriptor:
    torch.manual_seed(channels)
    net = Compact(
        num_in_ch=channels, num_out_ch=channels, num_feat=8, num_conv=2, upscale=2
    )
    model = CompactArch().load(net.state_dict())
    assert isinstance(model, ImageModelDescriptor)
    return model


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("fp16", [False, True])
@pytest.mark.parametrize("tiled", [False, True])
@pytest.mark.parametrize("nan_input", [False, True])
def test_pytorch_planes_interleaved_as_converted(
    monkeypatch: pytest.MonkeyPatch,
    channels: int,
    fp16: bool,
    tiled: bool,
    nan_input: bool,
):
    # A result with NaN (here from NaN in the input) keeps today's conversion, whose
    # CPU cast quiets a signalling half; a finite one is interleaved.
    model = compact(channels)
    img = image((24, 32, channels), 1, exceptional=False)
    if nan_input:
        img[2, 3] = np.nan

    def run() -> np.ndarray:
        return torch_split.pytorch_auto_split(
            img, model, CPU, fp16, tiler(tiled), Progress.noop_progress()
        )

    new = run()
    with monkeypatch.context() as patch:
        patch.setattr(torch_split, "_interleaved", lambda _: None)
        old = run()
    if nan_input:
        assert np.isnan(new).any()
        assert (
            np.ascontiguousarray(new).tobytes() == np.ascontiguousarray(old).tobytes()
        )
    else:
        assert_same(new, old)


class OverlappingRowsModel:
    """The members of a model pytorch_auto_split uses; its output's rows overlap."""

    dtype = torch.float32
    device = CPU
    architecture = SimpleNamespace(name="Overlapping")
    model = torch.nn.Identity()

    def __init__(self, row_stride: int) -> None:
        self.row_stride = row_stride

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        _, c, h, w = t.shape
        strides = (c * h * w, h * w, self.row_stride, 1)
        return t.contiguous().as_strided(t.shape, strides)


# Row stride 0 (an output expanded along its height), and rows half a row apart
@pytest.mark.parametrize("row_stride", [0, 16])
def test_pytorch_output_with_overlapping_rows_takes_the_old_path(
    monkeypatch: pytest.MonkeyPatch, row_stride: int
):
    model = cast(ImageModelDescriptor, OverlappingRowsModel(row_stride))
    img = image((24, 32, 3), 1, exceptional=False)

    def run() -> np.ndarray:
        return torch_split.pytorch_auto_split(
            img, model, CPU, False, NoTiling(), Progress.noop_progress()
        )

    new = run()
    with monkeypatch.context() as patch:
        patch.setattr(torch_split, "_interleaved", lambda _: None)
        old = run()
    assert new.shape == old.shape and new.dtype == old.dtype
    assert new.tobytes() == old.tobytes()


def test_pytorch_host_out_of_memory_in_the_conversion_splits(
    monkeypatch: pytest.MonkeyPatch,
):
    # The host runs out of memory for the converted output: NumPy's MemoryError, where
    # the CPU cast before raised torch's RuntimeError, which split the tile.
    model = compact(3)
    img = image((24, 32, 3), 1, exceptional=False)
    interleave = torch_split.planar_to_interleaved_checked
    calls: list[tuple[int, ...]] = []

    def out_of_memory_once(
        planes: np.ndarray, *, swap_red_blue: bool = False
    ) -> tuple[np.ndarray, bool]:
        calls.append(planes.shape)
        if len(calls) == 1:
            raise MemoryError
        return interleave(planes, swap_red_blue=swap_red_blue)

    monkeypatch.setattr(
        torch_split, "planar_to_interleaved_checked", out_of_memory_once
    )
    result = torch_split.pytorch_auto_split(
        img, model, CPU, False, MaxTileSize(), Progress.noop_progress()
    )
    assert result.shape == (48, 64, 3)
    assert calls[0] == (3, 48, 64) and len(calls) > 2


def resize_model(channels: int, fp16: bool) -> bytes:
    """2x nearest Resize, NCHW, float16 or float32."""
    kind = TensorProto.FLOAT16 if fp16 else TensorProto.FLOAT
    graph = helper.make_graph(
        [
            helper.make_node(
                "Resize", ["input", "", "scales"], ["output"], mode="nearest"
            )
        ],
        "resize",
        [helper.make_tensor_value_info("input", kind, [1, channels, "h", "w"])],
        [helper.make_tensor_value_info("output", kind, [1, channels, "H", "W"])],
        [helper.make_tensor("scales", TensorProto.FLOAT, [4], [1, 1, 2, 2])],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=10
    ).SerializeToString()


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("fp16", [False, True])
@pytest.mark.parametrize("tiled", [False, True])
@pytest.mark.parametrize("padded", [False, True])
def test_onnx_planes_interleaved_as_converted(
    monkeypatch: pytest.MonkeyPatch,
    channels: int,
    fp16: bool,
    tiled: bool,
    padded: bool,
):
    session = NativeSession(resize_model(channels, fp16))
    # 23x31 to a multiple of 4: the padding is cropped off the planes.
    img = image((23, 31, channels), 2, exceptional=True)
    req = SizeReq(multiple_of=4) if padded else None

    def run() -> np.ndarray:
        return onnx_split.onnx_auto_split(img, session, False, tiler(tiled), req)

    try:
        new = run()
        with monkeypatch.context() as patch:
            patch.setattr(onnx_split, "_output_planes", lambda *_: None)
            old = run()
    finally:
        session.close()
    assert_same(new, old)


SETTINGS = NcnnSettings(
    gpu_index=0, winograd=False, sgemm=False, threads=1, blocktime=0, budget_limit=0
)


def ncnn_model(
    directory: Path, channels_in: int, channels_out: int
) -> NcnnModelWrapper:
    """A 1x1 convolution (an identity, the out-of-range channels zero), then a 2x
    nearest Interp."""
    weights = np.eye(channels_out, channels_in, dtype=np.float32)
    param = (
        "7767517\n3 3\nInput data 0 1 data\n"
        f"Convolution conv 1 1 data mid 0={channels_out} 1=1 6={weights.size}\n"
        "Interp up 1 1 mid out 0=1 1=2.000000e+00 2=2.000000e+00\n"
    )
    (directory / "tiny.param").write_text(param, encoding="utf-8")
    (directory / "tiny.bin").write_bytes(b"\x00\x00\x00\x00" + weights.tobytes())
    return NcnnModelWrapper(
        NcnnModel.load_from_file(
            str(directory / "tiny.param"), str(directory / "tiny.bin")
        )
    )


def old_cast(planes: np.ndarray, **_: object) -> np.ndarray:
    """ncnn_auto_split's conversion before: the planes' HWC view cast, no swap."""
    return cast_numpy(planes.transpose(1, 2, 0), np.dtype(np.float32))


@pytest.mark.parametrize(("channels_in", "channels_out"), [(1, 1), (3, 3), (4, 4)])
@pytest.mark.parametrize("tile_size", [NO_TILING, TileSize(16)])
@pytest.mark.parametrize("separate_alpha", [False, True])
def test_ncnn_node_planes_interleaved_as_converted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    channels_in: int,
    channels_out: int,
    tile_size: TileSize,
    separate_alpha: bool,
):
    model = ncnn_model(tmp_path, channels_in, channels_out)
    net = create_ncnn_net(model, SETTINGS)
    monkeypatch.setattr(ncnn_node, "get_ncnn_net", lambda *_, **__: net)
    monkeypatch.setattr(ncnn_node, "get_settings", lambda _: SETTINGS)
    progress = Progress.noop_progress()
    context = cast(
        NodeContext,
        SimpleNamespace(
            add_cleanup=lambda *_, **__: None,
            check_aborted=progress.check_aborted,
            set_progress=progress.set_progress,
            suspend=progress.suspend,
        ),
    )
    rgba = image((20, 28, 4), 3, exceptional=False)
    rgba[:, :, 3] = np.linspace(0, 1, 28, dtype=np.float32)
    for img in (rgba[:, :, :1], rgba[:, :, :3], rgba):
        new = ncnn_node.upscale_image_node(
            context, img, model, tile_size, 0, separate_alpha
        )

        def old_upscale(i: np.ndarray) -> np.ndarray:
            ic = get_h_w_c(i)[2]
            if ic == 3:
                i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
            elif ic == 4:
                i = cv2.cvtColor(i, cv2.COLOR_BGRA2RGBA)
            i = ncnn_node.upscale_impl(
                SETTINGS,
                i,
                model,
                model.model.layers[0].outputs[0],
                model.model.layers[-1].outputs[0],
                tile_size,
                collect_after=False,
            )
            if ic == 3:
                i = cv2.cvtColor(i, cv2.COLOR_RGB2BGR)
            elif ic == 4:
                i = cv2.cvtColor(i, cv2.COLOR_RGBA2BGRA)
            return i

        with monkeypatch.context() as patch:
            patch.setattr(ncnn_split, "planar_to_interleaved", old_cast)
            old = convenient_upscale(
                img, model.in_nc, model.out_nc, old_upscale, separate_alpha
            )
        assert (
            new.shape == old.shape
            and new.tobytes() == np.ascontiguousarray(old).tobytes()
        ), (get_h_w_c(img), channels_in, channels_out)


@pytest.mark.parametrize(
    ("channels_in", "channels_out"), [(1, 3), (3, 1), (3, 4), (4, 3)]
)
@pytest.mark.parametrize("tile_size", [NO_TILING, TileSize(16)])
def test_ncnn_node_returns_the_models_channels_in_bgr_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    channels_in: int,
    channels_out: int,
    tile_size: TileSize,
):
    # A model whose input and output channels differ (convenient_upscale hands it the
    # image with its input channels) returns its output channels, a colour result in
    # BGR(A) order as PyTorch's and ONNX's nodes return it: before, a 1 -> 3 model's
    # colours came back red and blue swapped, a 3 -> 1 model's grey as three
    # channels, and a 3 -> 4 model's alpha was dropped.
    model = ncnn_model(tmp_path, channels_in, channels_out)
    net = create_ncnn_net(model, SETTINGS)
    monkeypatch.setattr(ncnn_node, "get_ncnn_net", lambda *_, **__: net)
    monkeypatch.setattr(ncnn_node, "get_settings", lambda _: SETTINGS)
    progress = Progress.noop_progress()
    context = cast(
        NodeContext,
        SimpleNamespace(
            add_cleanup=lambda *_, **__: None,
            check_aborted=progress.check_aborted,
            set_progress=progress.set_progress,
            suspend=progress.suspend,
        ),
    )
    # Values NCNN's 8-bit input keeps (to a few ulps: its input scales by 1 / 255)
    rng = np.random.default_rng(23)
    img = rng.integers(0, 256, (20, 28, channels_in)).astype(np.float32) / 255
    if channels_in == 1:
        img = img[:, :, 0]
    out = ncnn_node.upscale_image_node(context, img, model, tile_size, 0, False)

    # The model sees RGB(A) and its identity convolution keeps the shared channels
    planes = img.reshape(20, 28, channels_in)
    if channels_in in (3, 4):
        planes = planes[:, :, [2, 1, 0, 3][:channels_in]]
    result = np.zeros((20, 28, channels_out), np.float32)
    shared = min(channels_in, channels_out)
    result[:, :, :shared] = planes[:, :, :shared]
    if channels_out in (3, 4):
        result = result[:, :, [2, 1, 0, 3][:channels_out]]
    expected = result.repeat(2, axis=0).repeat(2, axis=1)
    assert out.shape[:2] == (40, 56) and get_h_w_c(out)[2] == channels_out
    np.testing.assert_allclose(out.reshape(expected.shape), expected, rtol=0, atol=1e-6)
