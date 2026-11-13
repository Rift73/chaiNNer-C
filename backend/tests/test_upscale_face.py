"""Upscale Face on a face its model fails on: the face is pasted back as it was, and the
log says which face failed and why, with the traceback."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from packages.chaiNNer_pytorch.pytorch.restoration import upscale_face


class Helper:
    def __init__(self, faces: list[np.ndarray]) -> None:
        self.cropped_faces = faces
        self.restored: list[np.ndarray] = []
        self.img = np.zeros((0, 0, 3), np.uint8)

    def clean_all(self) -> None:
        pass

    def read_image(self, img: np.ndarray) -> None:
        self.img = img

    def get_face_landmarks_5(self, **kwargs: object) -> None:
        pass

    def align_warp_face(self) -> None:
        pass

    def add_restored_face(self, face: np.ndarray) -> None:
        self.restored.append(face)

    def get_inverse_affine(self, _: object) -> None:
        pass

    def paste_faces_to_input_image(self) -> np.ndarray:
        return self.img


class Model:
    supports_half = False

    def __init__(self) -> None:
        self.calls = 0
        self.model = self

    def __call__(self, face: torch.Tensor, **kwargs: object) -> tuple[torch.Tensor]:
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("the second face fails")
        return (torch.zeros_like(face),)


def test_a_failed_face_is_logged_with_its_traceback(monkeypatch: pytest.MonkeyPatch):
    warnings: list[tuple[str, tuple[object, ...], dict[str, Any]]] = []
    monkeypatch.setattr(upscale_face, "on_device", lambda model, device, dtype: model)
    monkeypatch.setattr(
        upscale_face,
        "logger",
        SimpleNamespace(
            warning=lambda message, *args, **kwargs: warnings.append(
                (message, args, kwargs)
            )
        ),
    )
    faces = [np.full((4, 4, 3), value, np.uint8) for value in (10, 20, 30)]
    helper = Helper(faces)
    upscale_face.upscale(
        np.zeros((4, 4, 3), np.uint8),
        None,
        cast(Any, helper),
        cast(Any, Model()),
        0.5,
        cast(Any, SimpleNamespace(use_fp16=False)),
        torch.device("cpu"),
    )
    # The model's output of 0 maps to 128; the failed face keeps its own pixels
    assert [int(face[0, 0, 0]) for face in helper.restored] == [128, 20, 128]
    assert len(warnings) == 1
    message, args, kwargs = warnings[0]
    assert message % args == (
        "Upscale Face: face 2 of 3 failed and is pasted back unrestored"
    )
    assert kwargs == {"exc_info": True}


# Run in a fresh interpreter: who imports torchvision and facexlib first while the two
# modules load (spandrel's own architectures may import torchvision; these must not).
_WHO_IMPORTS = """
import sys

class Who:
    def find_spec(self, name, path=None, target=None):
        if name in ("torchvision", "facexlib"):
            frame = sys._getframe(1)
            while frame.f_code.co_filename.startswith("<frozen"):
                frame = frame.f_back
            importers.append((name, frame.f_code.co_filename))
        return None

importers = []
sys.meta_path.insert(0, Who())
import nodes.impl.pytorch.rife.IFNet_HDv3_v4_14_align
import packages.chaiNNer_pytorch.pytorch.restoration.upscale_face
for name, importer in importers:
    print(name, importer)
"""


def test_loading_the_nodes_does_not_import_torchvision_or_facexlib():
    src = Path(__file__).resolve().parents[1] / "src"
    result = subprocess.run(
        [sys.executable, "-c", _WHO_IMPORTS],
        cwd=src,
        capture_output=True,
        text=True,
        check=True,
        stdin=subprocess.DEVNULL,
    )
    ours = [
        line
        for line in result.stdout.splitlines()
        if line.endswith(("upscale_face.py", "IFNet_HDv3_v4_14_align.py"))
    ]
    assert ours == []
