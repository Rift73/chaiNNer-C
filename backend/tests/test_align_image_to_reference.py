"""Align Image to Reference without its downloaded weights: a flow block stands in where
the flow must be known, untrained weights where only the plumbing runs, and stand-ins for
XFeat's points where the matches must fall on one line."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from nodes.impl.pytorch.rife.IFNet_HDv3_v4_14_align import IFNet
from packages.chaiNNer_pytorch.pytorch.processing import align_image_to_reference
from packages.chaiNNer_pytorch.pytorch.processing.align_image_to_reference import (
    PrecisionMode,
    _align_images,
    calculate_padding,
    download_model,
)

CPU: Any = SimpleNamespace(device=torch.device("cpu"), use_fp16=False)


class ConstantFlow(torch.nn.Module):
    """A flow block whose flow is (x, y) pixels of the pair it is given, both ways."""

    def __init__(self, x: float, y: float):
        super().__init__()
        self.flow = (x, y)

    def forward(self, inputs: torch.Tensor, flow: Any = None, scale: int = 1):
        out = torch.zeros(1, 4, *inputs.shape[2:])
        out[:, 0], out[:, 1] = self.flow
        out[:, 2], out[:, 3] = self.flow
        return (out if flow is None else out * 0), out[:, :1] * 0


def flow_net(x: float, y: float) -> Any:
    net: Any = IFNet().eval()
    net.block0, net.block1 = ConstantFlow(x, y), ConstantFlow(0, 0)
    net.block2, net.block3 = ConstantFlow(0, 0), ConstantFlow(0, 0)
    return net


def ramps(height: int, width: int) -> torch.Tensor:
    """An image whose first channel is its column and second its row, over 512."""
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack((x / 512, y / 512, x * 0))[None]


@pytest.mark.parametrize(
    ("clip", "ref"),
    [
        # The width needs no padding for the network, the height does (1920x1080's case).
        ((120, 128), (120, 128)),
        # The padded width is the clip's width, the clip's height another.
        ((300, 128), (120, 128)),
        # And the other way around.
        ((128, 300), (128, 120)),
        ((120, 120), (120, 120)),
        ((240, 240), (120, 120)),
    ],
)
def test_the_flow_is_scaled_to_the_clip_on_each_axis(
    clip: tuple[int, int], ref: tuple[int, int]
):
    # The network sees both images resized to the padded reference: its flow is in those
    # pixels, and a pixel there is clip / padded pixels of the clip.
    image = ramps(*clip)
    reference = torch.zeros(1, 3, *ref)
    pads = calculate_padding(*ref, 64)
    with torch.inference_mode():
        out = flow_net(16, 32)(
            image,
            reference,
            None,
            (1, 1, 1, 1),
            pads,
            compensate=False,
            device=torch.device("cpu"),
        )
    assert out.shape[2:] == ref
    columns = torch.linspace(0, clip[1] - 1, ref[1]) + 16 * clip[1] / (ref[1] + pads[1])
    rows = torch.linspace(0, clip[0] - 1, ref[0]) + 32 * clip[0] / (ref[0] + pads[0])
    # Only where the flow stays inside the clip: the warp clamps at its border.
    inside_x, inside_y = columns < clip[1] - 2, rows < clip[0] - 2
    read_x = out[0, 0][ref[0] // 2, :] * 512
    read_y = out[0, 1][:, ref[1] // 2] * 512
    np.testing.assert_allclose(read_x[inside_x], columns[inside_x], atol=0.1)
    np.testing.assert_allclose(read_y[inside_y], rows[inside_y], atol=0.1)


@pytest.mark.parametrize("precision", list(PrecisionMode))
@pytest.mark.parametrize("lq_input", [False, True])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize(
    "size", [(1, 1), (2, 2), (3, 7), (7, 3), (16, 16), (32, 32), (45, 90)]
)
def test_small_images_align(
    precision: PrecisionMode, lq_input: bool, masked: bool, size: tuple[int, int]
):
    rng = np.random.default_rng(1)
    clip = rng.random((*size, 3), dtype=np.float32)
    ref = rng.random((*size, 3), dtype=np.float32)
    mask = np.ones(size, np.float32) if masked else None
    out = _align_images(
        clip, ref, mask, precision, lq_input, False, CPU, (IFNet().eval(), None, None)
    )
    assert out.shape == ref.shape
    assert np.isfinite(out).all()
    assert 0 <= out.min() and out.max() <= 1


def points_on_one_line(_image: torch.Tensor) -> list[dict[str, torch.Tensor]]:
    """XFeat's detection where every point lies on one line (one line of text)."""
    steps = torch.arange(200, dtype=torch.float32)
    return [
        {
            "keypoints": torch.stack((steps + 2, steps * 0 + 5), 1),
            "descriptors": torch.eye(200),
        }
    ]


def all_match(
    descriptors: torch.Tensor, _others: torch.Tensor, _similarity: float
) -> tuple[torch.Tensor, torch.Tensor]:
    everything = torch.arange(len(descriptors))
    return everything, everything


@pytest.mark.parametrize("masked", [False, True])
def test_wide_search_whose_matches_fit_no_homography_aligns_without_it(masked: bool):
    rng = np.random.default_rng(2)
    clip = rng.random((96, 128, 3), dtype=np.float32)
    ref = rng.random((96, 128, 3), dtype=np.float32)
    mask = np.ones((96, 128), np.float32) if masked else None
    points: Any = SimpleNamespace(detectAndCompute=points_on_one_line, match=all_match)
    out = _align_images(
        clip,
        ref,
        mask,
        PrecisionMode.FIFTY_PERCENT,
        False,
        True,
        CPU,
        (IFNet().eval(), points, points),
    )
    assert out.shape == ref.shape


def scattered_points(image: torch.Tensor) -> list[dict[str, torch.Tensor]]:
    """XFeat's detection where both images hold the same scattered points, wherever they
    fall in each image's size, so that the matches always fit a homography."""
    height, width = image.shape[2:]
    places = np.random.default_rng(8).random((200, 2), dtype=np.float32)
    return [
        {
            "keypoints": torch.from_numpy(places) * torch.tensor((width, height)),
            "descriptors": torch.eye(200),
        }
    ]


@pytest.mark.parametrize(
    "precision", [PrecisionMode.FIFTY_PERCENT, PrecisionMode.ONE_HUNDRED_PERCENT]
)
@pytest.mark.parametrize(
    ("clip", "ref"),
    [
        ((7, 300), (7, 300)),
        ((300, 7), (300, 7)),
        ((4, 64), (4, 64)),
        ((64, 4), (64, 4)),
        ((3, 3), (3, 3)),
        ((1, 300), (1, 300)),
        ((300, 1), (300, 1)),
        ((1, 300), (64, 300)),
        ((300, 1), (300, 64)),
        ((64, 300), (1, 300)),
    ],
)
def test_wide_search_on_tiny_images_keeps_the_references_size(
    monkeypatch: pytest.MonkeyPatch,
    clip: tuple[int, int],
    ref: tuple[int, int],
    precision: PrecisionMode,
):
    # The homography's crop has a minimum of 16 px, which does not fit a reference side
    # under 8 px with its padding, and its grid divides by a clip's side less 1.
    rng = np.random.default_rng(3)
    points: Any = SimpleNamespace(detectAndCompute=scattered_points, match=all_match)
    out = _align_images(
        rng.random((*clip, 3), dtype=np.float32),
        rng.random((*ref, 3), dtype=np.float32),
        None,
        precision,
        False,
        True,
        CPU,
        (IFNet().eval(), points, points),
    )
    assert out.shape == (*ref, 3)
    assert np.isfinite(out).all()


def test_wide_search_keeps_its_homography_at_small_sizes_it_fits(
    monkeypatch: pytest.MonkeyPatch,
):
    module = align_image_to_reference
    real = module.cv2.findHomography
    found = []

    def recording(*args: Any, **kwargs: Any) -> Any:
        result = real(*args, **kwargs)
        found.append(result[0] is not None)
        return result

    monkeypatch.setattr(module.cv2, "findHomography", recording)
    rng = np.random.default_rng(3)
    points: Any = SimpleNamespace(detectAndCompute=scattered_points, match=all_match)
    out = _align_images(
        rng.random((64, 64, 3), dtype=np.float32),
        rng.random((64, 64, 3), dtype=np.float32),
        None,
        PrecisionMode.FIFTY_PERCENT,
        False,
        True,
        CPU,
        (IFNet().eval(), points, points),
    )
    assert out.shape == (64, 64, 3)
    assert found == [True]


def answer_downloads(
    monkeypatch: pytest.MonkeyPatch, content: bytes
) -> dict[str, object]:
    """Every download gets `content`; returns the options of the last request."""
    options: dict[str, object] = {}

    def get(_url: str, **given: object) -> SimpleNamespace:
        options.update(given)
        return SimpleNamespace(content=content, raise_for_status=lambda: None)

    monkeypatch.setattr(align_image_to_reference.requests, "get", get)
    return options


def zip_of(name: str, content: bytes) -> bytes:
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr(name, content)
    return data.getvalue()


def test_a_stalled_download_ends(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    options = answer_downloads(monkeypatch, b"weights")
    download_model("https://example.invalid/m.pt", tmp_path, "m.pt")
    assert options["timeout"] == (15, 60)


def test_the_model_comes_out_of_its_zip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    answer_downloads(monkeypatch, zip_of("train_log/flownet.pkl", b"weights"))
    path = download_model(
        "https://example.invalid/z", tmp_path, "flownet.pkl", "train_log"
    )
    assert path == tmp_path / "flownet.pkl"
    assert [p.name for p in tmp_path.iterdir()] == ["flownet.pkl"]
    assert path.read_bytes() == b"weights"


@pytest.mark.parametrize(
    "content",
    [
        # Google Drive answers an exceeded download quota with a web page and status 200.
        b"<!DOCTYPE html><title>Google Drive - Quota exceeded</title>",
        zip_of("other/file.bin", b"not the model"),
    ],
    ids=["a web page", "a zip without the model"],
)
def test_a_download_that_is_not_the_zip_says_so_and_leaves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: bytes
):
    answer_downloads(monkeypatch, content)
    url = "https://example.invalid/z"
    with pytest.raises(RuntimeError, match=f"{url} is not the zip"):
        download_model(url, tmp_path, "flownet.pkl", "train_log")
    assert list(tmp_path.iterdir()) == []
