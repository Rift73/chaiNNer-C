from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from ncnn import ncnn

from nodes.impl.image_utils import to_uint8
from nodes.impl.native_framework_images import ncnn_input
from nodes.impl.ncnn.auto_split import ncnn_auto_split
from nodes.impl.ncnn.model import NcnnModel, NcnnModelWrapper
from nodes.impl.ncnn.session import create_ncnn_net
from nodes.impl.upscale.tiler import MaxTileSize
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


@pytest.mark.parametrize("broken", ["param", "bin"])
def test_a_model_ncnn_cannot_load_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, broken: str
):
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


class Extractor:
    """Upscales 2x by repeating pixels, or returns the error `code` for an input of
    more than `limit` pixels, as PyPI ncnn's Extractor does: no exception, and an
    output that must not be read."""

    def __init__(self, code: int, limit: int, sizes: list[tuple[int, int]]):
        self.code = code
        self.limit = limit
        self.sizes = sizes
        self.image = np.zeros((3, 0, 0), np.float32)

    def input(self, name: str, mat: ncnn.Mat) -> int:
        self.image = np.array(mat)
        return 0

    def extract(self, name: str) -> tuple[int, np.ndarray | None]:
        _, h, w = self.image.shape
        self.sizes.append((w, h))
        if w * h > self.limit:
            return self.code, None
        return 0, self.image.repeat(2, axis=1).repeat(2, axis=2)


class Net:
    def __init__(self, code: int, limit: int):
        self.code = code
        self.limit = limit
        self.sizes: list[tuple[int, int]] = []

    def create_extractor(self) -> Extractor:
        return Extractor(self.code, self.limit, self.sizes)


def test_out_of_memory_retries_with_smaller_tiles():
    net = Net(-100, 64 * 64)
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
    net = Net(-1, 0)
    image = np.zeros((10, 12, 3), np.float32)
    with pytest.raises(RuntimeError, match=r"NCNN failed with error code -1\."):
        ncnn_auto_split(
            image, cast(ncnn.Net, net), "data", "out", None, None, MaxTileSize()
        )
    assert net.sizes == [(12, 10)]
