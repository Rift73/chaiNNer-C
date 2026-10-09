from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from ncnn import ncnn

from nodes.impl.image_utils import to_uint8
from nodes.impl.native_framework_images import ncnn_input
from nodes.impl.ncnn import session
from nodes.impl.ncnn.auto_split import ncnn_auto_split
from nodes.impl.ncnn.model import NcnnModel, NcnnModelWrapper
from nodes.impl.ncnn.session import create_ncnn_net
from nodes.impl.upscale.auto_split_tiles import NO_TILING, TileSize
from nodes.impl.upscale.tiler import MaxTileSize
from packages.chaiNNer_ncnn.ncnn.processing import upscale_image
from packages.chaiNNer_ncnn.settings import NcnnSettings

BACKEND = Path(__file__).resolve().parents[1] / "src"
SETTINGS = NcnnSettings(
    gpu_index=0, winograd=False, sgemm=False, threads=1, blocktime=0, budget_limit=0
)
# A 1x1 convolution that copies its 3 channels, from blob "data" to blob "out".
PARAM = "7767517\n2 2\nInput data 0 1 data\nConvolution conv 1 1 data out 0=3 1=1 6=9\n"
BIN = b"\x00\x00\x00\x00" + np.eye(3, dtype=np.float32).tobytes()


def tiny_model(directory: Path) -> NcnnModelWrapper:
    (directory / "tiny.param").write_text(PARAM, encoding="utf-8")
    (directory / "tiny.bin").write_bytes(BIN)
    return NcnnModelWrapper(
        NcnnModel.load_from_file(
            str(directory / "tiny.param"), str(directory / "tiny.bin")
        )
    )


def test_a_model_ncnn_loads_is_accepted(tmp_path: Path):
    net = create_ncnn_net(tiny_model(tmp_path), SETTINGS)
    image = np.random.default_rng(0).random((5, 7, 3), dtype=np.float32)
    result = ncnn_auto_split(image, net, "data", "out", None, None, MaxTileSize())
    np.testing.assert_array_equal(
        result, ncnn_input(to_uint8(image)).transpose(1, 2, 0)
    )


class VulkanNet:
    """Stands in for ncnn.Net on the Vulkan branch: the tests hide every GPU, so the
    Vulkan settings go nowhere and the loads run on a CPU net."""

    def __init__(self):
        self.net = ncnn.Net()
        self.opt = SimpleNamespace()

    def set_vulkan_device(self, index: int) -> None:
        pass

    def __getattr__(self, name: str) -> Any:
        return getattr(self.net, name)


@pytest.mark.parametrize("gpu", [False, True])
@pytest.mark.parametrize("broken", ["param", "bin"])
def test_a_model_ncnn_cannot_load_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, broken: str, gpu: bool
):
    if gpu:
        monkeypatch.setattr(session, "use_gpu", True)
        monkeypatch.setattr(session, "ncnn", SimpleNamespace(Net=VulkanNet))
    model = tiny_model(tmp_path)
    if broken == "param":
        # The header counts a layer the param lacks, as a converted SPAN model's does.
        param = PARAM.replace("2 2", "3 3")
        monkeypatch.setattr(model.model, "write_param", lambda: param)
    else:
        truncated = BIN[:8]
        monkeypatch.setattr(
            model.model, "write_bin", lambda path: Path(path).write_bytes(truncated)
        )
    with pytest.raises(ValueError, match=rf"model's \.{broken} \(error code -1\)"):
        create_ncnn_net(model, SETTINGS)


def test_failures_of_the_real_binding_raise_instead_of_crashing(tmp_path: Path):
    # Run apart: reading the empty output of a failed extract crashes the process.
    (tmp_path / "tiny.bin").write_bytes(BIN)
    code = "\n".join(
        [
            "import sys",
            f"sys.path[:0] = [{str(BACKEND)!r}]",
            "import numpy as np",
            "from ncnn import ncnn",
            "from nodes.impl.ncnn.auto_split import ncnn_auto_split",
            "from nodes.impl.upscale.tiler import MaxTileSize",
            "net = ncnn.Net()",
            "net.opt.use_vulkan_compute = False",
            f"assert net.load_param_mem({PARAM!r}) == 0",
            f"assert net.load_model({str(tmp_path / 'tiny.bin')!r}) == 0",
            "image = np.zeros((8, 8, 3), np.float32)",
            "for blobs in (('data', 'missing'), ('missing', 'out')):",
            "    try:",
            "        ncnn_auto_split(image, net, *blobs, None, None, MaxTileSize())",
            "    except Exception as e:",
            "        print(f'{type(e).__name__}: {e}', flush=True)",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "RuntimeError: NCNN failed with error code -1. Its reason is in chaiNNer's log.",
        "ValueError: The NCNN model has no blob named missing.",
    ]


class Net:
    """Upscales `scale_x` times across and `scale_y` times down (nearest pixel),
    then drops `crop` pixels from each side, as a waifu2x-ncnn-vulkan model's unpadded
    convolutions do. For an input of more than `limit` pixels it returns the error
    `code` as PyPI ncnn's Extractor does: no exception, and an output that must not
    be read."""

    def __init__(
        self,
        code: int = 0,
        limit: int = 2**62,
        crop: int = 0,
        scale_x: float = 2,
        scale_y: float = 2,
    ):
        self.code = code
        self.limit = limit
        self.crop = crop
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.sizes: list[tuple[int, int]] = []

    def create_extractor(self) -> Extractor:
        return Extractor(self)


class Extractor:
    def __init__(self, net: Net):
        self.net = net
        self.image = np.zeros((3, 0, 0), np.float32)

    def input(self, name: str, mat: ncnn.Mat) -> int:
        self.image = np.array(mat)
        return 0

    def extract(self, name: str) -> tuple[int, np.ndarray | None]:
        _, h, w = self.image.shape
        self.net.sizes.append((w, h))
        if w * h > self.net.limit:
            return self.net.code, None
        c = self.net.crop
        out_h, out_w = round(h * self.net.scale_y), round(w * self.net.scale_x)
        rows = np.arange(out_h) * h // out_h
        columns = np.arange(out_w) * w // out_w
        upscaled = self.image[:, rows][:, :, columns]
        return 0, upscaled[:, c : out_h - c, c : out_w - c]


def test_out_of_memory_retries_with_smaller_tiles():
    net = Net(code=-100, limit=64 * 64)
    image = np.random.default_rng(0).random((100, 120, 3), dtype=np.float32)
    result = ncnn_auto_split(
        image, cast(ncnn.Net, net), "data", "out", None, None, MaxTileSize()
    )
    assert net.sizes[0] == (120, 100)
    assert len(net.sizes) > 1
    expected = ncnn_input(to_uint8(image)).transpose(1, 2, 0)
    np.testing.assert_allclose(
        result, expected.repeat(2, axis=0).repeat(2, axis=1), rtol=0, atol=1e-6
    )


def test_other_failures_raise_their_code_without_retrying():
    net = Net(code=-1, limit=0)
    image = np.zeros((10, 12, 3), np.float32)
    with pytest.raises(RuntimeError, match=r"NCNN failed with error code -1\."):
        ncnn_auto_split(
            image, cast(ncnn.Net, net), "data", "out", None, None, MaxTileSize()
        )
    assert net.sizes == [(12, 10)]


def upscale_with(
    monkeypatch: pytest.MonkeyPatch,
    net: Net,
    scale: int,
    image: np.ndarray,
    tile_size: TileSize = NO_TILING,
) -> np.ndarray:
    """NCNN Upscale Image's upscale with the fake `net`, for a model of `scale`."""
    monkeypatch.setattr(upscale_image, "get_ncnn_net", lambda model, settings: net)
    model = cast(NcnnModelWrapper, SimpleNamespace(scale=scale))
    return upscale_image.upscale_impl(SETTINGS, image, model, "data", "out", tile_size)


@pytest.mark.parametrize(
    ("tile_size", "message"),
    [
        (NO_TILING, "a 1564x1164 image for a 800x600 image, 36 pixels short of its 2x"),
        (TileSize(256), r"a \d+x\d+ image for a \d+x\d+ tile.*whole multiple"),
    ],
)
def test_a_model_that_crops_its_output_is_refused(
    monkeypatch: pytest.MonkeyPatch, tile_size: TileSize, message: str
):
    # 2x less 36 pixels: whole, 800x600 gives 1564x1164; every tile is cropped too.
    image = np.zeros((600, 800, 3), np.float32)
    with pytest.raises(ValueError, match=message):
        upscale_with(monkeypatch, Net(crop=18), 2, image, tile_size)


@pytest.mark.parametrize(
    ("scale_x", "scale_y", "scale", "height", "width"),
    [
        pytest.param(0.5, 0.5, 1, 60, 80, id="downscale"),
        pytest.param(0.5, 0.5, 1, 64, 64, id="downscale-square"),
        pytest.param(2, 1, 2, 60, 80, id="uneven"),
        pytest.param(1.5, 1.5, 1, 64, 64, id="non-integer"),
        pytest.param(2, 2, 4, 60, 80, id="less-than-its-scale"),
    ],
)
def test_other_sizes_are_returned_as_the_model_made_them_without_tiling(
    monkeypatch: pytest.MonkeyPatch,
    scale_x: float,
    scale_y: float,
    scale: int,
    height: int,
    width: int,
):
    image = np.random.default_rng(0).random((height, width, 3), dtype=np.float32)
    net = Net(scale_x=scale_x, scale_y=scale_y)
    result = upscale_with(monkeypatch, net, scale, image)

    pixels = ncnn_input(to_uint8(image))
    extractor = Extractor(net)
    extractor.image = pixels
    _, expected = extractor.extract("out")
    assert expected is not None
    assert result.shape[:2] == (round(height * scale_y), round(width * scale_x))
    np.testing.assert_array_equal(result, expected.transpose(1, 2, 0))


def test_an_unexpected_error_is_logged_and_kept_as_the_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    # A bare assert of the tiler used to reach the log as an empty "ERROR ''".
    def fail(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError

    monkeypatch.setattr(upscale_image, "ncnn_auto_split", fail)
    image = np.zeros((8, 8, 3), np.float32)
    with pytest.raises(RuntimeError, match="unexpected error") as raised:
        upscale_image.upscale_impl(
            SETTINGS, image, tiny_model(tmp_path), "data", "out", NO_TILING
        )
    assert isinstance(raised.value.__cause__, AssertionError)
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert [r.exc_info[0] for r in errors if r.exc_info] == [AssertionError]
