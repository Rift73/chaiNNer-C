"""Actual installed Rust binary versus native payloads, with Win32 calls captured.

These tests never open/read/change the interactive clipboard. Both binary import
tables must pass the capture guard before a writer is called. OS publication is
a separate validation item; no benchmark timings or GPU work are performed.
"""

from __future__ import annotations

import ast
import ctypes
import hashlib
import json
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from clipboard_capture import ClipboardCapture

from chainner_ext import Clipboard
from chainner_ext import chainner_ext as original_binary
from nodes.impl.native_graph import graph

ROOT = Path(__file__).resolve().parents[2]
NATIVE = graph()
ORIGINAL_PATH = Path(original_binary.__file__)
assert NATIVE.__file__ is not None
NATIVE_PATH = Path(NATIVE.__file__)
REFERENCE = Path(__file__).with_name("reference_utility_clipboard")


def captured(call, original=False, failure=None):
    with ClipboardCapture(ORIGINAL_PATH if original else NATIVE_PATH, failure) as sink:
        try:
            result = call()
            outcome = ("ok", result)
        except BaseException as error:
            outcome = (type(error).__name__, error.args)
        assert not sink.callback_errors
        return (
            outcome,
            dict(sink.data),
            list(sink.calls),
            len(sink.allocations - sink.transferred),
        )


def original_image(value, pixel_format: Literal["RGB", "BGR"] = "BGR"):
    return Clipboard.create_instance().write_image(value, pixel_format)


def original_text(value):
    return Clipboard.create_instance().write_text(value)


def image_cases(channels, layout):
    # Every half-step and its adjacent float32 values, plus non-finite/range edges.
    half = (np.arange(255, dtype=np.float32) + np.float32(0.5)) / np.float32(255)
    values = np.concatenate(
        (
            half,
            np.nextafter(half, -np.inf),
            np.nextafter(half, np.inf),
            np.array([0, -0.0, -1, 1, 2, np.nan, np.inf, -np.inf], dtype=np.float32),
        )
    )
    # Even the one-channel fixture contains every threshold and both neighbors.
    array = np.resize(values, (29, 31, channels)).copy()
    if layout == "reverse":
        array = array[::-1, ::-1, ::-1]
    elif layout == "transpose":
        array = array.transpose(1, 0, 2)
    elif layout == "broadcast":
        array = np.broadcast_to(array[:1, :1], array.shape)
    elif layout == "unaligned":
        storage = bytearray(array.nbytes + 1)
        view = np.ndarray(array.shape, dtype=np.float32, buffer=storage, offset=1)
        view[...] = array
        array = view
    elif layout == "readonly":
        array.flags.writeable = False
    elif layout == "metadata":
        array = array.view(np.dtype(np.float32, metadata={"application": "clipboard"}))
    return array


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize(
    "layout",
    ["plain", "reverse", "transpose", "broadcast", "unaligned", "readonly", "metadata"],
)
@pytest.mark.parametrize("pixel_format", ["RGB", "BGR"])
def test_image_payload_against_actual_installed_binary(channels, layout, pixel_format):
    array = image_cases(channels, layout)
    before = array.tobytes()
    expected = captured(lambda: original_image(array, pixel_format), original=True)
    actual = captured(lambda: NATIVE.utility_clipboard_image(array, pixel_format))
    assert actual == expected
    assert actual[0] == ("ok", None)
    payload = NATIVE.utility_clipboard_image_bytes(array, pixel_format)
    assert payload == actual[1][17]
    assert array.tobytes() == before


@pytest.mark.parametrize(
    "shape", [(0, 0), (0, 2), (2, 0), (1, 1), (3, 5), (0, 2, 4), (2, 0, 3), (2, 3, 1)]
)
def test_gray_and_empty_image_headers(shape):
    value = np.full(shape, 0.25, dtype=np.float32)
    expected = captured(lambda: original_image(value), original=True)
    actual = captured(lambda: NATIVE.utility_clipboard_image(value, "BGR"))
    assert actual == expected
    data = actual[1][17]
    assert struct.unpack_from("<IiiHH", data) == (124, shape[1], shape[0], 1, 32)


@pytest.mark.parametrize(
    "text",
    ["", "a", "éΩ漢字", "🙂𝄞", "\0", "one\0two", "\r\n\t", "x" * 65537],
    ids=[
        "empty",
        "ascii",
        "unicode",
        "astral",
        "nul",
        "embedded-nul",
        "whitespace",
        "large",
    ],
)
def test_text_bytes_and_publication_order(text):
    expected = captured(lambda: original_text(text), original=True)
    actual = captured(lambda: NATIVE.utility_clipboard_text(text))
    assert actual == expected
    assert actual[1][13] == text.encode("utf-16-le") + b"\0\0"
    assert NATIVE.utility_clipboard_text_bytes(text) == actual[1][13]


@pytest.mark.parametrize("value", [None, 1, 3.5, b"text", [], "\ud800", "x\udfffy"])
def test_text_validation_before_open(value):
    expected = captured(lambda: original_text(value), original=True)
    actual = captured(lambda: NATIVE.utility_clipboard_text(value))
    assert actual == expected
    assert actual[2] == []


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        "image",
        np.zeros(3, np.float32),
        np.zeros((2, 3, 1, 1), np.float32),
        np.zeros((2, 3), np.float64),
        np.zeros((2, 3, 4), np.uint8),
        np.zeros((2, 3), ">f4"),
        np.zeros((2, 3, 0), np.float32),
        np.zeros((2, 3, 2), np.float32),
        np.zeros((2, 3, 5), np.float32),
    ],
)
@pytest.mark.parametrize("pixel_format", ["BGR", "bad", None, "\ud800"])
def test_image_validation_priority_and_exact_errors(value, pixel_format):
    expected = captured(lambda: original_image(value, pixel_format), original=True)
    actual = captured(lambda: NATIVE.utility_clipboard_image(value, pixel_format))
    assert conversion_rule(actual) == conversion_rule(expected)
    assert actual[2] == []


def conversion_rule(result):
    """Consult 3's rule against chaiNNer-C's chainner_ext: pyo3's argument-conversion
    TypeError, OverflowError and UnicodeEncodeError match by type only (the node path
    keeps pyo3's texts); every other outcome, a conversion ValueError included, matches
    in full."""
    outcome, *rest = result
    if outcome[0] in {"TypeError", "OverflowError", "UnicodeEncodeError"}:
        outcome = outcome[:1]
    return (outcome, *rest)


@pytest.mark.parametrize("image", [False, True])
@pytest.mark.parametrize("failure", ["open", "empty", "allocate", "lock", "set"])
def test_failures_keep_diagnostics_but_free_untransferred_memory(image, failure):
    value = np.full((2, 3, 4), 0.5, np.float32) if image else "test"
    expected = captured(
        lambda: original_image(value) if image else original_text(value),
        original=True,
        failure=failure,
    )
    actual = captured(
        lambda: (
            NATIVE.utility_clipboard_image(value, "BGR")
            if image
            else NATIVE.utility_clipboard_text(value)
        ),
        failure=failure,
    )
    # arboard leaked the image HGLOBAL after Lock/Set failures (and called DeleteObject
    # on it). The node path's RAII owns it until transfer, and chaiNNer-C's chainner_ext
    # frees it the same way (Consult 6 D-1), so the calls and the leak count match.
    assert actual == expected
    assert actual[3] == 0
    if failure == "open":
        assert actual[2].count(("OpenClipboard", None)) == 6
        assert actual[2].count(("Sleep", 5)) == 5
        assert ("CloseClipboard",) not in actual[2]
    else:
        assert actual[2][-1] == ("CloseClipboard",)


def test_node_dispatch_and_registration_unchanged():
    relative = "packages/chaiNNer_standard/utility/clipboard/copy_to_clipboard.py"
    previous = ast.parse((REFERENCE / "source/copy_to_clipboard.py").read_text("utf-8"))
    current = ast.parse((ROOT / "backend/src" / relative).read_text("utf-8"))
    old_fn = next(n for n in previous.body if isinstance(n, ast.FunctionDef))
    new_fn = next(n for n in current.body if isinstance(n, ast.FunctionDef))
    new_fn.body = old_fn.body
    assert ast.dump(new_fn) == ast.dump(old_fn)
    for value in ("text", np.full((2, 3, 4), 0.75, np.float32)):
        expected = captured(
            lambda value=value: (
                original_image(value)
                if isinstance(value, np.ndarray)
                else original_text(value)
            ),
            original=True,
        )
        actual = captured(
            lambda value=value: NATIVE.utility_copy_to_clipboard(value, np.ndarray)
        )
        assert actual == expected


def test_parallel_kernel_calls_are_repeatable_and_do_not_mutate_inputs():
    # Only pure payload construction runs concurrently; IAT capture stays serial.
    values = [np.resize(image_cases(c, "reverse"), (137, 149, c)) for c in (1, 3, 4)]
    expected = [
        captured(lambda v=v: original_image(v), original=True)[1][17] for v in values
    ]
    before = [v.tobytes() for v in values]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(
                lambda i: NATIVE.utility_clipboard_image_bytes(values[i % 3], "BGR"),
                range(48),
            )
        )
    assert all(value == expected[i % 3] for i, value in enumerate(results))
    assert [v.tobytes() for v in values] == before


def test_c_abi_rejects_invalid_sizes_strides_overlap_and_pointers():
    dll = ctypes.CDLL(str(ROOT / "backend/src/nodes/impl/chainner_native.dll"))
    fn = dll.cn_clipboard_pixels_f32
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_ssize_t,
        ctypes.c_ssize_t,
        ctypes.c_ssize_t,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    fn.restype = ctypes.c_int
    source = np.zeros((2, 3, 4), np.float32)
    output = np.full(24, 177, np.uint8)
    base = [source.ctypes.data, 2, 3, 4, 48, 16, 4, 0, output.ctypes.data, 24]
    for index, value in [
        (0, None),
        (3, 2),
        (7, 2),
        (8, None),
        (9, 23),
        (4, sys.maxsize),
        (5, -sys.maxsize),
        (6, sys.maxsize),
        (8, source.ctypes.data),
        (0, 1),
        (0, (1 << 64) - 4),
    ]:
        args: list[int | None] = list(base)
        args[index] = value
        if index == 0 and value == 1:
            args[4] = -48
        assert fn(*args) != 0
        assert np.all(output == 177)
    assert fn(None, 0, 3, 4, 0, 0, 0, 0, None, 0) == 0
    assert fn(*base) == 0


def test_reference_snapshots_have_not_changed():
    manifest = json.loads((REFERENCE / "manifest.json").read_text("utf-8"))
    for entry in manifest["files"]:
        assert (
            hashlib.sha256((REFERENCE / entry["snapshot"]).read_bytes()).hexdigest()
            == entry["sha256"]
        )
