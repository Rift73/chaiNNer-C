"""Pillow coverage-plane expansion must reproduce the original RGBA renderer."""

from __future__ import annotations

import ast
import importlib.util
import types
from enum import Enum
from pathlib import Path

import numpy as np
import pytest

from nodes.impl import native_buffers

SOURCE = Path(__file__).resolve().parents[2] / "backend/src"
REFERENCE = Path(__file__).with_name("reference_palette")


def load_text(path):
    anchors = ast.parse(
        (SOURCE / "nodes/properties/inputs/generic_inputs.py").read_text()
    )
    anchor_class = next(
        n for n in anchors.body if isinstance(n, ast.ClassDef) and n.name == "Anchor"
    )
    module = types.ModuleType("_text_reference")
    module.__dict__.update(Enum=Enum)
    exec(
        compile(ast.Module(body=[anchor_class], type_ignores=[]), str(path), "exec"),
        module.__dict__,
    )
    tree = ast.parse(path.read_text())
    tree.body = [
        n
        for n in tree.body
        if not isinstance(n, ast.ImportFrom)
        or (
            not n.level
            and n.module != "nodes.groups"
            and not (n.module or "").startswith("nodes.properties")
        )
    ]
    for n in tree.body:
        if isinstance(n, ast.FunctionDef):
            n.decorator_list = []
    exec(compile(tree, str(path), "exec"), module.__dict__)
    # Only this module sees the generated __main__ path for bundled font lookup.
    module.__dict__.update(
        sys=types.SimpleNamespace(
            modules={"__main__": types.SimpleNamespace(__file__=str(SOURCE / "run.py"))}
        )
    )
    return module


TEXT = load_text(
    SOURCE / "packages/chaiNNer_standard/image/create_images/text_as_image.py"
)
ORIGINAL = load_text(REFERENCE / "text_as_image.py")
_spec = importlib.util.spec_from_file_location(
    "nodes.impl._text_original_image_utils",
    REFERENCE.parent / "reference_buffers/image_utils.py",
)
assert _spec is not None and _spec.loader is not None
UTILS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(UTILS)
ORIGINAL.__dict__.update(normalize=UTILS.normalize, to_uint8=UTILS.to_uint8)


@pytest.mark.parametrize(
    "bold,italic", [(False, False), (True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("alignment", ["LEFT", "CENTER", "RIGHT"])
@pytest.mark.parametrize(
    "anchor",
    [
        "TOP_LEFT",
        "TOP",
        "TOP_RIGHT",
        "LEFT",
        "CENTER",
        "RIGHT",
        "BOTTOM_LEFT",
        "BOTTOM",
        "BOTTOM_RIGHT",
    ],
)
@pytest.mark.parametrize("text", ["One line", "Alpha\nBeta 42", "Δ • 漢"])
def test_text_mask_exact(bold, italic, alignment, anchor, text):
    color = TEXT.Color.bgr((0.15, 0.57, 0.91))
    expected = ORIGINAL.text_as_image_node(
        text,
        bold,
        italic,
        getattr(ORIGINAL.TextAlignment, alignment),
        color,
        137,
        61,
        getattr(ORIGINAL.Anchor, anchor),
    )
    actual = TEXT.text_as_image_node(
        text,
        bold,
        italic,
        getattr(TEXT.TextAlignment, alignment),
        color,
        137,
        61,
        getattr(TEXT.Anchor, anchor),
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("layout", ["plain", "reversed", "readonly"])
def test_text_mask_every_coverage(layout):
    mask = np.arange(256, dtype=np.uint8).reshape((16, 16))
    if layout == "reversed":
        mask = mask[::-1, ::-1]
    elif layout == "readonly":
        mask.setflags(write=False)
    ink = (37, 143, 255)
    expected = np.zeros((*mask.shape, 4), np.uint8)
    expected[:, :, :3] = np.where(mask[:, :, None] != 0, np.array(ink, np.uint8), 0)
    expected[:, :, 3] = mask
    np.testing.assert_array_equal(
        native_buffers.colorize_text(mask, ink), UTILS.normalize(expected)
    )


def test_text_contract_and_empty():
    with pytest.raises(ValueError):
        native_buffers.colorize_text(np.zeros((3, 4), np.float32), (1, 2, 3))
    with pytest.raises(ValueError):
        native_buffers.colorize_text(np.zeros((3, 4), np.uint8), (1, 2))
    np.testing.assert_array_equal(
        native_buffers.colorize_text(np.zeros((0, 3), np.uint8), (1, 2, 3)),
        np.zeros((0, 3, 4), np.float32),
    )
    for module in (TEXT, ORIGINAL):
        with pytest.raises(ZeroDivisionError):
            module.text_as_image_node(
                "",
                False,
                False,
                module.TextAlignment.CENTER,
                module.Color.bgr((0, 0, 0)),
                10,
                10,
                module.Anchor.CENTER,
            )
