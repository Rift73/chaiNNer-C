"""Independent installed image I/O oracles; real tiny CPU codecs, mocked viewer."""

from __future__ import annotations

import ast
import builtins
import errno
import hashlib
import json
import logging
import os
import platform
import shutil
import struct
import subprocess
import time
import uuid
import warnings
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from tempfile import mkdtemp
from threading import Barrier, Lock
from types import ModuleType, SimpleNamespace
from typing import Any, Literal, Union

import cv2
import numpy as np

# Imported for its side effect (x as x keeps it): registers Pillow's AVIF plugin.
import pillow_avif as pillow_avif
import pytest
from numpy.typing import DTypeLike
from PIL import Image, TiffImagePlugin

from nodes.impl import image_formats, image_utils, native_image_io
from nodes.impl.dds import format as dds_format
from nodes.impl.item_window import Prepared
from nodes.impl.native_graph import graph
from nodes.utils.utils import get_h_w_c, split_file_path

ROOT = Path(__file__).resolve().parents[2]
REF = Path(__file__).with_name("reference_image_io")
NODE_DIR = "packages/chaiNNer_standard/image/io"
BASELINE = Path(
    os.environ["LOCALAPPDATA"],
    "chaiNNer",
    "app-0.25.1-nightly2025-10-21",
    "resources",
    "src",
)


def extract(path, env, functions=None):
    module = ModuleType("image_io_oracle")
    module.__dict__.update(env)
    body = []
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            body.append(node)
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            if functions is None or node.name in functions:
                if isinstance(node, ast.FunctionDef):
                    node.decorator_list = []
                body.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and functions is None:
            body.append(node)
    exec(
        compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"),
        module.__dict__,
    )
    return module


OLD_UTILS = extract(
    REF / "installed/nodes/impl/image_utils.py",
    {"np": np, "cv2": cv2, "split_file_path": split_file_path},
    {"_get_iinfo", "normalize", "to_uint8", "to_uint16", "cv_save_image"},
)


def modules(current):
    env: dict[str, Any] = {
        "np": np,
        "cv2": cv2,
        "Image": Image,
        "TiffImagePlugin": TiffImagePlugin,
        "Enum": Enum,
        "Path": Path,
        "Callable": Callable,
        "Iterable": Iterable,
        "Union": Union,
        "Literal": Literal,
        "logger": logging.getLogger("image-io-test"),
        "os": os,
        "platform": platform,
        "shutil": shutil,
        "subprocess": subprocess,
        "uuid": uuid,
        "mkdtemp": mkdtemp,
        "time": time,
        "graph": graph,
        "get_h_w_c": get_h_w_c,
        "split_file_path": split_file_path,
        "sys": SimpleNamespace(
            modules={"__main__": SimpleNamespace(__file__=str(BASELINE / "server.py"))}
        ),
    }
    env.update({k: v for k, v in vars(dds_format).items() if not k.startswith("__")})
    env.update({k: v for k, v in vars(image_formats).items() if k.startswith("get_")})
    util = image_utils if current else OLD_UTILS
    env.update(to_uint8=util.to_uint8, to_uint16=util.to_uint16)
    env["cv_save_image"] = (
        native_image_io.cv_save_image if current else OLD_UTILS.cv_save_image
    )
    root = ROOT / "backend/src" if current else REF / "installed"
    dds = extract(root / "nodes/impl/dds/texconv.py", env)
    env.update(dds_to_png_texconv=dds.dds_to_png_texconv, save_as_dds=dds.save_as_dds)
    loader = extract(root / NODE_DIR / "load_image.py", env)
    saver = extract(root / NODE_DIR / "save_image.py", env)
    viewer = extract(root / NODE_DIR / "view_image_external.py", env)
    return SimpleNamespace(load=loader, save=saver, view=viewer, dds=dds)


OLD = modules(False)
NEW = modules(True)


def recorded(
    function: Callable[[], Any],
) -> tuple[Any, tuple[Any, ...], list[tuple[type, str]]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = function()
            status = ("ok",)
        except Exception as error:
            result = None
            status = (
                type(error),
                error.args,
                type(error.__cause__),
                error.__cause__.args if error.__cause__ else None,
            )
    return result, status, [(type(x.message), str(x.message)) for x in caught]


def exact(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert a.tobytes() == b.tobytes()


class LazyImage:
    def __init__(self, image=None, fail=False):
        self.image = image
        self.fail = fail
        self.calls = 0

    @property
    def value(self):
        self.calls += 1
        if self.fail:
            raise LookupError("lazy failure")
        return self.image


def save_arguments(module, lazy, directory, *, fmt="PNG", filename="image", **options):
    values: dict[str, Any] = {
        "lazy_image": lazy,
        "base_directory": directory,
        "relative_path": None,
        "filename": filename,
        "image_format": module.ImageFormat[fmt],
        "png_color_depth": module.PngColorDepth.U8,
        "webp_lossless": False,
        "quality": 80,
        "jpeg_chroma_subsampling": module.JpegSubsampling.FACTOR_420,
        "jpeg_progressive": False,
        "tiff_color_depth": module.TiffColorDepth.U8,
        "tiff_compression": module.TiffCompression.NONE,
        "dds_format": "R8G8B8A8_UNORM",
        "dds_bc7_compression": module.BC7Compression.DEFAULT,
        "dds_error_metric": module.DDSErrorMetric.PERCEPTUAL,
        "dds_dithering": False,
        "dds_mipmap_levels": 1,
        "dds_separate_alpha": False,
        "avif_chroma_subsampling": module.AvifSubsampling.FACTOR_420,
        "skip_existing_files": False,
    }
    for key, value in options.items():
        if isinstance(value, tuple):
            values[key] = getattr(getattr(module, value[0]), value[1])
        else:
            values[key] = value
    return values


def image(channels=3, dtype: DTypeLike = np.float32):
    shape = (7, 9) if channels == 1 else (7, 9, channels)
    result = np.random.default_rng(921).uniform(0, 1, shape)
    if np.dtype(dtype).kind in "iu":
        result *= np.iinfo(dtype).max
    return result.astype(dtype)


def layout(value, kind):
    if kind == "reverse":
        return value[::-1, ::-1]
    if kind == "fortran":
        return np.asfortranarray(value)
    if kind == "unaligned":
        result = np.ndarray(value.shape, value.dtype, bytearray(value.nbytes + 1), 1)
        result[:] = value
        return result
    if kind == "readonly":
        value.flags.writeable = False
    return value


@pytest.mark.parametrize(
    "dtype,alpha",
    [
        (dtype, alpha)
        for dtype in [np.uint8, np.uint16, np.float32, np.float64, np.int32]
        for alpha in [0, 1, 255, 65535, "nan", "snan"]
        if not isinstance(alpha, str) or np.dtype(dtype).kind == "f"
    ],
)
@pytest.mark.parametrize(
    "kind", ["plain", "reverse", "fortran", "unaligned", "readonly"]
)
def test_alpha_raw_dtype_strides_identity(dtype, kind, alpha):
    value = image(4, dtype)
    if isinstance(alpha, str):
        value[:, :, 3] = np.nan
        if alpha == "snan":
            value[:, :, 3].view(np.uint32 if dtype == np.float32 else np.uint64)[:] = (
                0x7F800001 if dtype == np.float32 else 0x7FF0000000000001
            )
    else:
        value[:, :, 3] = (
            0 if np.dtype(dtype).kind in "iu" and alpha > np.iinfo(dtype).max else alpha
        )
    value = layout(value, kind)
    before = value.tobytes()
    a, sa, wa = recorded(lambda: OLD.load.remove_unnecessary_alpha(value))
    b, sb, wb = recorded(lambda: NEW.load.remove_unnecessary_alpha(value))
    assert sa == sb and wa == wb
    exact(a, b)
    assert (a is value) == (b is value)
    assert np.shares_memory(a, value) == np.shares_memory(b, value)
    assert value.tobytes() == before


@pytest.mark.parametrize(
    "shape", [(0, 2, 4), (2, 0, 4), (2, 3, 4, 2), (2, 3), (2, 3, 3)]
)
def test_alpha_unusual_shapes(shape):
    value = np.ones(shape, np.float32)
    exact(
        NEW.load.remove_unnecessary_alpha(value),
        OLD.load.remove_unnecessary_alpha(value),
    )


FORMAT_OPTIONS = [
    ("PNG", {}),
    ("PNG", {"png_color_depth": ("PngColorDepth", "U16")}),
    ("JPG", {}),
    (
        "JPG",
        {
            "quality": 100,
            "jpeg_progressive": True,
            "jpeg_chroma_subsampling": ("JpegSubsampling", "FACTOR_444"),
        },
    ),
    ("GIF", {}),
    ("BMP", {}),
    ("TGA", {}),
    ("WEBP", {}),
    ("WEBP", {"webp_lossless": True}),
    ("TIFF", {}),
    (
        "TIFF",
        {
            "tiff_color_depth": ("TiffColorDepth", "U16"),
            "tiff_compression": ("TiffCompression", "LZW"),
        },
    ),
    ("TIFF", {"tiff_color_depth": ("TiffColorDepth", "F32")}),
    ("AVIF", {}),
    (
        "AVIF",
        {"avif_chroma_subsampling": ("AvifSubsampling", "FACTOR_444"), "quality": 30},
    ),
]


@pytest.mark.parametrize("fmt,options", FORMAT_OPTIONS)
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("kind", ["plain", "reverse", "readonly"])
def test_real_cpu_codec_save_load(tmp_path, fmt, options, channels, kind):
    value = layout(image(channels), kind)
    before = value.tobytes()
    paths = []
    states = []
    for number, modules_ in enumerate([OLD, NEW]):
        lazy = LazyImage(value)
        args = save_arguments(
            modules_.save,
            lazy,
            tmp_path / f"v{number}",
            fmt=fmt,
            filename="像 image",
            **options,
        )
        _, status, warnings_ = recorded(
            lambda modules_=modules_, args=args: modules_.save.save_image_node(**args)
        )
        states.append((status, warnings_))
        assert lazy.calls == 1
        paths.append(args["base_directory"] / f"像 image.{fmt.lower()}")
    # Codec messages contain output paths; failures here are checked for class
    # and corresponding codec message after mapping the separate owned roots.
    left = (
        repr(states[0]).replace(str(tmp_path / "v0"), "ROOT").replace("v0", "VARIANT")
    )
    right = (
        repr(states[1]).replace(str(tmp_path / "v1"), "ROOT").replace("v1", "VARIANT")
    )
    assert left == right
    assert value.tobytes() == before
    if states[0][0][0] != "ok":
        return
    if fmt == "PNG" and not options and channels in (3, 4):
        # User-selected fpng changes compression bytes, never decoded pixels.
        exact(
            cv2.imdecode(
                np.frombuffer(paths[0].read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED
            ),
            cv2.imdecode(
                np.frombuffer(paths[1].read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED
            ),
        )
    elif fmt == "TIFF" and channels == 4:
        # Owner-approved (upstream chaiNNer #2950): the old file plus ExtraSamples = 2
        # in a copy of the first IFD after it (backend/tests/test_save_image_tiff_extra_samples.py).
        old, new = paths[0].read_bytes(), paths[1].read_bytes()
        assert new[:4] == old[:4] and new[8 : len(old)] == old[8:]
        assert struct.unpack("<I", new[4:8]) == (len(old) + len(old) % 2,)
        assert struct.pack("<HHIHH", 338, 3, 1, 2, 0) in new[len(old) :]
    else:
        assert paths[0].read_bytes() == paths[1].read_bytes()
    old_loaded, old_status, old_warnings = recorded(
        lambda: OLD.load.load_image_node(paths[0])
    )
    new_loaded, new_status, new_warnings = recorded(
        lambda: NEW.load.load_image_node(paths[0])
    )
    assert old_status == new_status and old_warnings == new_warnings
    if old_status[0] != "ok":
        # GIF is writable but intentionally excluded from Load Image's formats.
        with Image.open(paths[0]) as first, Image.open(paths[1]) as second:
            exact(np.array(first), np.array(second))
        return
    exact(new_loaded[0], old_loaded[0])
    assert new_loaded[1:] == old_loaded[1:]
    exact(image_utils.normalize(new_loaded[0]), OLD_UTILS.normalize(old_loaded[0]))


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "P", "I;16"])
def test_pillow_file_modes_and_unicode_paths(tmp_path, mode):
    path = tmp_path / "パレット.PNG"
    im = Image.new(mode, (5, 3))
    if mode == "P":
        im.putpalette([i for x in range(256) for i in (x, 255 - x, x // 2)])
    im.save(path)
    exact(NEW.load._read_pil(path), OLD.load._read_pil(path))
    exact(NEW.load.load_image_node(path)[0], OLD.load.load_image_node(path)[0])


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("skip", [False, True])
def test_save_lazy_error_priority_and_repetition(tmp_path, existing, skip):
    for number, module in enumerate([OLD.save, NEW.save]):
        directory = tmp_path / str(number)
        if existing:
            directory.mkdir()
            (directory / "image.png").write_bytes(b"sentinel")
        lazy = LazyImage(fail=True)
        args = save_arguments(module, lazy, directory, skip_existing_files=skip)
        _, state, _ = recorded(
            lambda module=module, args=args: module.save_image_node(**args)
        )
        assert directory.exists()
        if existing and skip:
            assert state == ("ok",) and lazy.calls == 0
            assert (directory / "image.png").read_bytes() == b"sentinel"
        else:
            assert state[:2] == (LookupError, ("lazy failure",)) and lazy.calls == 1


@pytest.mark.parametrize("suffix", [".png", ".jpg", ".unknown"])
def test_corrupt_and_missing_load_errors(tmp_path, suffix):
    path = tmp_path / ("bad" + suffix)
    for contents in [None, b"invalid image"]:
        if contents is not None:
            path.write_bytes(contents)
        a = recorded(lambda: OLD.load.load_image_node(path))[1:]
        b = recorded(lambda: NEW.load.load_image_node(path))[1:]
        assert a == b


def test_decoder_order_error_identity_and_short_circuit(tmp_path):
    def run(current, success):
        module = modules(current).load
        events = []
        errors = [ValueError("first"), LookupError("last")]
        output = image()

        def first(path):
            events.append(("first", type(path)))
            raise errors[0]

        def second(path):
            events.append(("second", type(path)))
            return output if success else None

        def third(path):
            events.append(("third", type(path)))
            raise errors[1]

        module._decoders = [("first", first), ("second", second), ("third", third)]
        try:
            result = module.load_image_node(tmp_path / "name.png")
            return result[0] is output, events
        except Exception as error:
            return error is errors[1], events

    for success in [False, True]:
        assert run(False, success) == run(True, success)


@pytest.mark.parametrize("system", ["Windows", "Darwin", "Linux"])
@pytest.mark.parametrize("saved", [True, False])
def test_viewer_decisions_are_mocked(tmp_path, system, saved):
    def run(current):
        module = modules(current).view
        events = []
        module.mkdtemp = lambda **kwargs: str(tmp_path)
        module.time = SimpleNamespace(time=lambda: 123.25)
        module.platform = SimpleNamespace(system=lambda: system)
        module.os = SimpleNamespace(
            path=os.path, startfile=lambda p: events.append(("startfile", p))
        )
        module.subprocess = SimpleNamespace(call=lambda p: events.append(("call", p)))
        module.cv2 = SimpleNamespace(
            imwrite=lambda path, img: (
                events.append(("save", path, img.dtype.str, img.tobytes())) or saved
            )
        )
        # Grayscale deliberately retains OpenCV. Compatible RGB(A) fpng writes
        # and mocked shell dispatch are covered separately in test_fpng_io.py.
        module.view_image_external_node(image(1))
        return events

    assert run(False) == run(True)


@pytest.mark.parametrize("bits", range(64))
def test_dds_options_and_cleanup_mocked(tmp_path, bits):
    def run(current):
        module = modules(current).dds
        events = []
        module.mkdtemp = lambda **kwargs: str(tmp_path / "scratch")
        module.cv_save_image = lambda *args: events.append(
            ("png", str(args[0]), args[1].tobytes(), args[2])
        )
        module.__dict__["__run_texconv"] = lambda *args: events.append(("run", *args))
        module.shutil = SimpleNamespace(
            rmtree=lambda path: events.append(("cleanup", path))
        )
        module.save_as_dds(
            tmp_path / "name.dds",
            image(4, np.uint8),
            "BC7_UNORM_SRGB",
            0,
            *(bool(bits & 1 << i) for i in range(6)),
        )
        return events

    assert run(False) == run(True)


@pytest.mark.parametrize("code", [0, 1, 27])
def test_texconv_error_and_decoding(tmp_path, code):
    def run(current):
        module = modules(current).dds
        module.platform = SimpleNamespace(system=lambda: "Windows")
        module.subprocess = SimpleNamespace(
            run=lambda *args, **kwargs: SimpleNamespace(
                returncode=code, stdout=b"bad\r\n\xff", stderr=b"detail"
            )
        )
        return recorded(
            lambda: module.__dict__["__run_texconv"](["-f", "rgba"], "Failed")
        )[1:]

    assert run(False) == run(True)


@pytest.mark.parametrize("dds_name", [x[0] for x in OLD.save.SUPPORTED_DDS_FORMATS])
@pytest.mark.parametrize("compression", ["BEST_SPEED", "DEFAULT", "BEST_QUALITY"])
@pytest.mark.parametrize("uniform", [False, True])
def test_save_dds_all_exposed_format_options(tmp_path, dds_name, compression, uniform):
    def run(current):
        module = modules(current).save
        calls = []
        module.save_as_dds = lambda path, value, fmt, **kwargs: calls.append(
            (path, value.dtype.str, value.tobytes(), fmt, kwargs)
        )
        args = save_arguments(
            module,
            LazyImage(image(4)),
            tmp_path,
            fmt="DDS",
            dds_format=dds_name,
            dds_bc7_compression=("BC7Compression", compression),
            dds_error_metric=("DDSErrorMetric", "UNIFORM" if uniform else "PERCEPTUAL"),
            dds_dithering=True,
            dds_mipmap_levels=0,
            dds_separate_alpha=True,
        )
        module.save_image_node(**args)
        return calls

    assert run(False) == run(True)


@pytest.mark.parametrize(
    "fmt", ["R8G8B8A8_UNORM", "BC1_UNORM", "BC3_UNORM", "BC7_UNORM"]
)
def test_real_dds_cpu_codec(tmp_path, fmt):
    paths = []
    for variant, current in enumerate([False, True]):
        group = modules(current)
        dds = group.dds
        original_run = dds.__dict__["__run_texconv"]
        # Test-only switch enforces CPU codec execution. Command policy itself
        # is compared without this added switch in the independent mocks.
        dds.__dict__["__run_texconv"] = lambda args, error, original_run=original_run: (
            original_run(["-nogpu", *args], error)
        )
        dds.mkdtemp = lambda **kwargs: mkdtemp(dir=tmp_path, **kwargs)
        group.load.os = SimpleNamespace(remove=os.remove)
        directory = tmp_path / str(variant)
        args = save_arguments(
            group.save, LazyImage(image(4)), directory, fmt="DDS", dds_format=fmt
        )
        group.save.save_image_node(**args)
        paths.append(directory / "image.dds")
        assert paths[-1].is_file()
        original = OLD.load._read_cv
        assert original is not None
    assert paths[0].read_bytes() == paths[1].read_bytes()
    # Texconv conversion helper invocation also gets the test-only CPU switch.
    old_group, new_group = modules(False), modules(True)
    for group in [old_group, new_group]:
        dds = group.dds
        runner = dds.__dict__["__run_texconv"]
        dds.__dict__["__run_texconv"] = lambda args, error, runner=runner: runner(
            ["-nogpu", *args], error
        )
        dds.mkdtemp = lambda **kwargs: mkdtemp(dir=tmp_path, **kwargs)
    exact(old_group.load._read_dds(paths[0]), new_group.load._read_dds(paths[0]))


@pytest.mark.parametrize("failure", ["encode", "run", "cleanup"])
def test_dds_finally_failure_priority(tmp_path, failure):
    def run(current):
        module = modules(current).dds
        events = []
        module.mkdtemp = lambda **kwargs: str(tmp_path)

        def encode(*args):
            events.append("encode")
            if failure == "encode":
                raise LookupError("encode failed")

        def execute(*args):
            events.append("run")
            raise RuntimeError("run failed")

        def cleanup(path):
            events.append("cleanup")
            if failure == "cleanup":
                raise OSError("cleanup failed")

        module.cv_save_image = encode
        module.__dict__["__run_texconv"] = execute
        module.shutil = SimpleNamespace(rmtree=cleanup)
        try:
            module.save_as_dds(tmp_path / "file.dds", image(), "R8G8B8A8_UNORM")
        except Exception as error:
            return (
                type(error),
                error.args,
                type(error.__context__),
                error.__context__.args if error.__context__ else None,
                events,
            )
        raise AssertionError("expected injected failure")

    assert run(False) == run(True)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("kind", ["reverse", "unaligned", "fortran", "readonly"])
def test_pillow_channel_buffer_permutation(dtype, channels, kind):
    value = layout(image(channels, dtype), kind)

    # Fake only PIL's file object; its ndarray conversion is still real. This
    # reaches the native permutation for all codec-supported buffer depths.
    def run(current):
        module = modules(current).load
        module.Image = SimpleNamespace(
            open=lambda path: SimpleNamespace(mode="RGB", info={})
        )
        module.np = SimpleNamespace(array=lambda im: value)
        return module._read_pil(Path("fixture.png"))

    exact(run(True), run(False))


@pytest.mark.parametrize("fmt", ["PNG", "TIFF", "TGA"])
def test_nonfinite_save_quantization(tmp_path, fmt):
    value = image(3)
    value.flat[:7] = [np.nan, np.inf, -np.inf, -0.0, 1.5, -0.5, 0.5]
    expected = None
    for current in [False, True]:
        module = modules(current).save
        captured = []
        module.cv_save_image = lambda path, arr, options, captured=captured: (
            captured.append((arr.dtype.str, arr.tobytes(), options))
        )
        if fmt == "TGA":

            class Picture:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return None

                def save(self, *args, **kwargs):
                    return None

            module.Image = SimpleNamespace(
                fromarray=lambda arr, captured=captured: (
                    captured.append((arr.dtype.str, arr.tobytes())) or Picture()
                )
            )
        args = save_arguments(module, LazyImage(value), tmp_path, fmt=fmt)
        state = recorded(
            lambda module=module, args=args: module.save_image_node(**args)
        )[1:]
        result = captured, state
        if not current:
            expected = result
        else:
            assert result == expected


@pytest.mark.parametrize(
    "subsampling", ["FACTOR_444", "FACTOR_440", "FACTOR_422", "FACTOR_420"]
)
@pytest.mark.parametrize("progressive", [False, True])
@pytest.mark.parametrize("quality", [0, 1, 80, 100])
def test_all_jpeg_option_commands(tmp_path, subsampling, progressive, quality):
    def run(current):
        module = modules(current).save
        calls = []
        module.cv_save_image = lambda path, value, options: calls.append(
            (path, value.dtype.str, value.tobytes(), options)
        )
        args = save_arguments(
            module,
            LazyImage(image(3)),
            tmp_path,
            fmt="JPG",
            jpeg_chroma_subsampling=("JpegSubsampling", subsampling),
            jpeg_progressive=progressive,
            quality=quality,
        )
        module.save_image_node(**args)
        return calls

    assert run(False) == run(True)


@pytest.mark.parametrize("depth", ["U8", "U16", "F32"])
@pytest.mark.parametrize("compression", ["NONE", "LZW", "ZIP"])
def test_all_tiff_precision_options(tmp_path, depth, compression):
    def run(current):
        module = modules(current).save
        calls = []
        module.cv_save_image = lambda path, value, options: calls.append(
            (path, value.dtype.str, value.tobytes(), options)
        )
        args = save_arguments(
            module,
            LazyImage(image(4)),
            tmp_path,
            fmt="TIFF",
            tiff_color_depth=("TiffColorDepth", depth),
            tiff_compression=("TiffCompression", compression),
        )
        module.save_image_node(**args)
        return calls

    assert run(False) == run(True)


@pytest.mark.parametrize(
    "subsampling", ["FACTOR_444", "FACTOR_422", "FACTOR_420", "FACTOR_400"]
)
@pytest.mark.parametrize("failure", ["none", "save", "exit"])
def test_pillow_save_context_and_all_avif_options(tmp_path, subsampling, failure):
    def run(current):
        module = modules(current).save
        events = []

        class Picture:
            def __enter__(self):
                events.append("enter")
                return self

            def __exit__(self, kind, error, trace):
                events.append(("exit", kind, error.args if error else None))
                if failure == "exit":
                    raise OSError("exit failed")

            def save(self, path, **kwargs):
                events.append(("save", path, kwargs))
                if failure != "none":
                    raise LookupError("save failed")

        module.Image = SimpleNamespace(fromarray=lambda value: Picture())
        args = save_arguments(
            module,
            LazyImage(image(3)),
            tmp_path,
            fmt="AVIF",
            avif_chroma_subsampling=("AvifSubsampling", subsampling),
        )
        try:
            module.save_image_node(**args)
            state = None
        except Exception as error:
            state = (
                type(error),
                error.args,
                type(error.__context__),
                error.__context__.args if error.__context__ else None,
            )
        return state, events

    assert run(False) == run(True)


@pytest.mark.parametrize("fail_first", [False, True])
@pytest.mark.parametrize("second", ["valid", "none", "error"])
def test_cv_decoder_fallback_cause_and_order(fail_first, second):
    def run(current):
        module = modules(current).load
        events = []

        def fromfile(*args, **kwargs):
            events.append("read bytes")
            if fail_first:
                raise OSError("read failed")
            return np.zeros(3, np.uint8)

        def decode(*args):
            events.append("decode")

        def read(*args):
            events.append("imread")
            if second == "error":
                raise ValueError("decode failed")
            return None if second == "none" else np.zeros((2, 2), np.uint8)

        module.np = SimpleNamespace(fromfile=fromfile, uint8=np.uint8)
        module.cv2 = SimpleNamespace(imdecode=decode, imread=read, IMREAD_UNCHANGED=-1)
        result, status, messages = recorded(lambda: module._read_cv(Path("file.png")))
        return None if result is None else result.tobytes(), status, messages, events

    assert run(False) == run(True)


@pytest.mark.parametrize("relative", [None, "", ".", "sub/folder", "sub/../folder"])
def test_full_path_and_repeat_skip(tmp_path, relative):
    value = image()
    for current in [False, True]:
        module = modules(current).save
        lazy = LazyImage(value)
        args = save_arguments(
            module,
            lazy,
            tmp_path,
            relative_path=relative,
            filename="name",
            skip_existing_files=False,
        )
        path = module.get_full_path(tmp_path, relative, "name", module.ImageFormat.PNG)
        module.save_image_node(**args)
        before = path.read_bytes()
        args["skip_existing_files"] = True
        args["lazy_image"] = LazyImage(fail=True)
        module.save_image_node(**args)
        assert path.read_bytes() == before
        assert args["lazy_image"].calls == 0


def test_frozen_hashes_and_metadata():
    manifest = json.loads((REF / "manifest.json").read_text())
    for entry in manifest["files"]:
        assert (
            hashlib.sha256(
                (REF / entry["variant"] / entry["path"]).read_bytes()
            ).hexdigest()
            == entry["sha256"]
        )
    for name in ["load_image", "save_image", "view_image_external"]:

        def metadata(path):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            node = next(
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name.endswith("_node")
            )
            node.body = [ast.Pass()]
            return ast.dump(node)

        assert metadata(ROOT / "backend/src" / NODE_DIR / f"{name}.py") == metadata(
            REF / "source" / NODE_DIR / f"{name}.py"
        )


def test_concurrent_cpu_save_load(tmp_path):
    def work(i):
        values = image(3)
        args = save_arguments(
            NEW.save, LazyImage(values), tmp_path, filename=f"parallel-{i}"
        )
        NEW.save.save_image_node(**args)
        actual = NEW.load.load_image_node(tmp_path / f"parallel-{i}.png")[0]
        exact(actual, OLD_UTILS.to_uint8(values, normalized=True))

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(work, range(12)))


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("kind", ["reverse", "unaligned", "fortran", "readonly"])
def test_concurrent_larger_channel_permutation(dtype, channels, kind):
    values = np.random.default_rng(924).integers(0, 256, (257, 383, channels))
    value = layout(values.astype(dtype), kind)
    before = value.tobytes()

    def decoder(current):
        module = modules(current).load
        module.Image = SimpleNamespace(
            open=lambda path: SimpleNamespace(mode="RGB", info={})
        )
        module.np = SimpleNamespace(array=lambda im: value)
        return module._read_pil

    expected = decoder(False)(Path("fixture.png"))
    readers = [decoder(True) for _ in range(4)]
    ready = Barrier(4)

    def work(reader):
        ready.wait()
        for _ in range(3):
            actual = reader(Path("fixture.png"))
            exact(actual, expected)
            assert not np.shares_memory(actual, value)
            assert actual.flags.c_contiguous

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(work, readers))
    assert value.tobytes() == before


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
@pytest.mark.parametrize("opaque", [False, True])
@pytest.mark.parametrize("kind", ["reverse", "unaligned", "readonly"])
@pytest.mark.parametrize("shape", [(257, 383, 4), (129, 131, 4, 3)])
def test_concurrent_larger_alpha_scan(dtype, opaque, kind, shape):
    value = np.ones(shape, dtype)
    value[:, :, 3] = np.iinfo(dtype).max if np.dtype(dtype).kind == "u" else 1
    if not opaque:
        value[-1, -1, 3] = 0
    value = layout(value, kind)
    before = value.tobytes()
    expected = OLD.load.remove_unnecessary_alpha(value)
    ready = Barrier(4)

    def work(_):
        ready.wait()
        for _ in range(3):
            actual = NEW.load.remove_unnecessary_alpha(value)
            exact(actual, expected)
            assert (actual is value) == (expected is value)
            assert np.shares_memory(actual, value) == np.shares_memory(expected, value)
            assert actual.strides == expected.strides
            assert actual.flags.writeable == expected.flags.writeable

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(work, range(4)))
    assert value.tobytes() == before


# OpenCV decodes these through a temp file whose name concurrent decodes could
# share (SP3b decoder audit U2); read_cv runs their imdecode calls one at a time.
TEMP_FILE_DECODES = [".sr", ".ras", ".hdr", ".pic", ".exr", ".HDR"]


class DecodesInFlight:
    """A cv2 stand-in counting the imdecode calls that run at once."""

    IMREAD_UNCHANGED = cv2.IMREAD_UNCHANGED

    def __init__(self, decode, overlap=None):
        self.decode = decode
        self.overlap = overlap
        self.guard = Lock()
        self.running = 0
        self.peak = 0
        self.calls = 0

    def imdecode(self, buffer, flags):
        with self.guard:
            self.calls += 1
            self.running += 1
            self.peak = max(self.peak, self.running)
        try:
            if self.overlap is None:
                time.sleep(0.005)
            else:
                self.overlap.wait()
            return self.decode(buffer, flags)
        finally:
            with self.guard:
                self.running -= 1

    def imread(self, path, flags):
        raise AssertionError(f"imdecode fell back to imread for {path}")


def stub_decode(buffer, flags):
    return np.full((2, 3), buffer.size, np.uint8)


# imdecode picks its decoder by content, whatever the name: OpenCV 4.8.0's
# checkSignature rules for its temp-file decoders, and near misses that must not wait.
TEMP_FILE_SIGNATURES = [
    pytest.param(".png", b"#?RADIANCE\n", True, id="radiance-named-png"),
    pytest.param(".jpg", b"#?RGBE\n", True, id="rgbe-named-jpg"),
    pytest.param(".png", b"\x59\xa6\x6a\x95", True, id="sun-raster-named-png"),
    pytest.param(".tif", b"\x76\x2f\x31\x01", True, id="openexr-named-tif"),
    pytest.param(".ppm", b"PF\n", True, id="pfm-named-ppm"),
    pytest.param(".ppm", b"Pf ", True, id="pfm-gray-named-ppm"),
    pytest.param(".png", b"#?RGB\n", False, id="radiance-near-miss"),
    pytest.param(".ppm", b"PFx", False, id="pfm-near-miss"),
    pytest.param(".ppm", b"P6\n", False, id="pixmap-from-memory"),
]


@pytest.mark.parametrize(
    ("suffix", "header", "serialized"),
    [
        *(pytest.param(x, b"", True, id=x) for x in TEMP_FILE_DECODES),
        *(pytest.param(x, b"", False, id=x) for x in (".png", ".jpg", ".tif", ".webp")),
        *TEMP_FILE_SIGNATURES,
    ],
)
def test_only_temp_file_decodes_wait_for_each_other(
    tmp_path, suffix, header, serialized
):
    threads = 4 if serialized else 2
    paths = [tmp_path / f"stub{i}{suffix}" for i in range(threads)]
    for i, path in enumerate(paths):
        path.write_bytes(header + bytes(5 + i))
    module = modules(True).load
    # A serialized decode meets no overlap barrier: two decodes at once would show
    # in the peak. Any other decode must overlap, or the barrier times out.
    overlap = None if serialized else Barrier(threads, timeout=10)
    module.cv2 = recorder = DecodesInFlight(stub_decode, overlap)
    ready = Barrier(threads)

    def work(path):
        ready.wait()
        return [module._read_cv(path) for _ in range(2 if serialized else 1)]

    with ThreadPoolExecutor(max_workers=threads) as executor:
        results = list(executor.map(work, paths))
    for path, values in zip(paths, results, strict=True):
        for value in values:
            exact(value, np.full((2, 3), path.stat().st_size, np.uint8))
    assert recorder.peak == (1 if serialized else threads)
    assert recorder.calls == sum(map(len, results)) and recorder.running == 0


@pytest.mark.parametrize("named", ["radiance", "png"])
def test_concurrent_temp_file_decodes_equal_serial_decodes(tmp_path, named):
    # Radiance files only: OpenCV leaves an ocv*.tmp in %TEMP% per Sun-raster decode.
    # Renamed to .png, the content alone must take the lock.
    rng = np.random.default_rng(926)
    paths = []
    for i in range(6):
        path = tmp_path / f"radiance{i}{'.hdr' if i % 2 else '.pic'}"
        pixels = rng.uniform(0, 8, (29 + i, 37, 3)).astype(np.float32)
        assert cv2.imwrite(str(path), pixels)
        paths.append(path if named == "radiance" else path.rename(f"{path}.png"))
    expected = [OLD.load.load_image_node(path)[0] for path in paths]
    module = modules(True).load
    module.cv2 = recorder = DecodesInFlight(cv2.imdecode)
    ready = Barrier(4)

    def work(offset):
        ready.wait()
        order = [(offset + k) % len(paths) for k in range(len(paths))]
        return order, [module.load_image_node(paths[i])[0] for i in order]

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(work, range(4)))
    for order, values in results:
        for i, actual in zip(order, values, strict=True):
            exact(actual, expected[i])
    assert recorder.peak == 1 and recorder.calls == 4 * len(paths)


# Save Image's prepare phase (SP3-P9) per format family: fpng; OpenCV imencode with
# today's parameters (grayscale and 16-bit PNG, JPG, WEBP, TIFF, BMP); Pillow into a
# buffer with the format's own options (GIF, TGA, AVIF).
PREPARED_FORMATS = [
    pytest.param("PNG", 3, {}, id="png-u8-fpng"),
    pytest.param("PNG", 1, {}, id="png-u8-grayscale"),
    pytest.param("PNG", 4, {"png_color_depth": ("PngColorDepth", "U16")}, id="png-u16"),
    pytest.param(
        "JPG",
        3,
        {
            "quality": 90,
            "jpeg_progressive": True,
            "jpeg_chroma_subsampling": ("JpegSubsampling", "FACTOR_444"),
        },
        id="jpg",
    ),
    pytest.param("WEBP", 4, {}, id="webp"),
    pytest.param("WEBP", 3, {"webp_lossless": True}, id="webp-lossless"),
    pytest.param("TIFF", 3, {}, id="tiff-u8"),
    pytest.param(
        "TIFF",
        4,
        {
            "tiff_color_depth": ("TiffColorDepth", "U16"),
            "tiff_compression": ("TiffCompression", "LZW"),
        },
        id="tiff-u16",
    ),
    pytest.param(
        "TIFF", 3, {"tiff_color_depth": ("TiffColorDepth", "F32")}, id="tiff-f32"
    ),
    pytest.param("BMP", 3, {}, id="bmp"),
    pytest.param("GIF", 3, {}, id="gif"),
    pytest.param("TGA", 4, {}, id="tga"),
    pytest.param(
        "AVIF",
        3,
        {"quality": 30, "avif_chroma_subsampling": ("AvifSubsampling", "FACTOR_444")},
        id="avif",
    ),
]


def conversions(module, monkeypatch):
    """Record each to_uint8/to_uint16 call of Save: its input object, layout and options."""
    calls = []
    for name in ("to_uint8", "to_uint16"):

        def convert(value, *, convert=module.__dict__[name], name=name, **options):
            calls.append((name, id(value), value.dtype.str, value.strides, options))
            return convert(value, **options)

        monkeypatch.setitem(module.__dict__, name, convert)
    return calls


@pytest.mark.parametrize(("fmt", "channels", "options"), PREPARED_FORMATS)
def test_prepared_save_bytes_equal_todays_file(
    monkeypatch, tmp_path, fmt, channels, options
):
    # The prepare encodes in memory the bytes today's save writes, from the same
    # conversion input; the commit writes them where the encoder would.
    value = image(channels)
    calls = conversions(NEW.save, monkeypatch)
    today = save_arguments(
        NEW.save, LazyImage(value), tmp_path / "today", fmt=fmt, **options
    )
    NEW.save.save_image_node(**today)
    expected = (tmp_path / "today" / f"image.{fmt.lower()}").read_bytes()
    converted = list(calls)
    calls.clear()
    inputs = save_arguments(NEW.save, value, tmp_path / "unused", fmt=fmt, **options)
    prepared = NEW.save.prepare_save_image(list(inputs.values()))
    assert type(prepared) is bytes and prepared == expected
    assert calls == converted
    assert not (tmp_path / "unused").exists()  # nothing touches the file system
    lazy = LazyImage(value)
    holder = Prepared(prepared)
    committed = save_arguments(NEW.save, lazy, tmp_path / "commit", fmt=fmt, **options)
    calls.clear()
    NEW.save.commit_save_image(list(committed.values()), holder)
    assert (tmp_path / "commit" / f"image.{fmt.lower()}").read_bytes() == expected
    assert lazy.calls == 1 and holder.used and calls == []


def test_prepared_save_leaves_dds_whole(tmp_path):
    # Texconv writes the file itself, so DDS has nothing to prepare.
    inputs = save_arguments(NEW.save, image(4), tmp_path, fmt="DDS")
    assert NEW.save.prepare_save_image(list(inputs.values())) is None
    assert list(tmp_path.iterdir()) == []


class PullRecorder(LazyImage):
    def __init__(self, image, events):
        super().__init__(image)
        self.events = events

    @property
    def value(self):
        self.events.append("pull")
        return super().value


@dataclass
class ResultRecorder(Prepared):
    events: list = field(default_factory=list)

    def result(self):
        self.events.append("result")
        return super().result()


@pytest.mark.parametrize("skip", [False, True])
def test_prepared_save_commit_order(tmp_path, skip):
    # Path, exists/skip-existing and mkdir as today. A skip pulls nothing and leaves the
    # prepared value unused. A write pulls the image, then writes the prepared bytes.
    target = tmp_path / "out" / "image.png"
    target.parent.mkdir()
    target.write_bytes(b"existing")
    events = []
    holder = ResultRecorder(b"prepared", events=events)
    args = save_arguments(
        NEW.save,
        PullRecorder(image(3), events),
        target.parent,
        skip_existing_files=skip,
    )
    NEW.save.commit_save_image(list(args.values()), holder)
    assert events == ([] if skip else ["pull", "result"]) and holder.used != skip
    assert target.read_bytes() == (b"existing" if skip else b"prepared")


class FailingWrite:
    """A file whose first write stores 4 bytes, then fails as a full disk does."""

    def __init__(self, handle):
        self.handle = handle

    def write(self, data):
        self.handle.write(bytes(memoryview(data).cast("B")[:4]))
        raise OSError(errno.ENOSPC, "No space left on device")

    def __getattr__(self, name):
        return getattr(self.handle, name)

    def __enter__(self):
        return self

    def __exit__(self, *_exception):
        self.handle.close()


def failing_writes(monkeypatch, suffix):
    """Make every file opened for writing whose name ends with suffix a FailingWrite."""
    real_open = builtins.open

    def open_failing(file, mode="r", *args, **kwargs):
        handle = real_open(file, mode, *args, **kwargs)
        if "w" in mode and str(file).endswith(suffix):
            return FailingWrite(handle)
        return handle

    monkeypatch.setattr(builtins, "open", open_failing)


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("fmt", ["GIF", "TGA", "AVIF"])
def test_prepared_pillow_write_failure_leaves_todays_file(
    monkeypatch, tmp_path, fmt, existing
):
    # Image.save(path)'s file handling, mirrored for prepared Pillow bytes: a write that
    # fails leaves no file the save created, and a truncated existing one.
    value = image(3)
    inputs = save_arguments(NEW.save, value, tmp_path / "unused", fmt=fmt)
    prepared = NEW.save.prepare_save_image(list(inputs.values()))
    failing_writes(monkeypatch, f"image.{fmt.lower()}")
    states = []
    for name in ("today", "commit"):
        target = tmp_path / name / f"image.{fmt.lower()}"
        target.parent.mkdir()
        if existing:
            target.write_bytes(b"existing")
        args = list(
            save_arguments(NEW.save, LazyImage(value), target.parent, fmt=fmt).values()
        )
        with pytest.raises(OSError, match="No space left on device"):
            if name == "today":
                NEW.save.save_image_node(*args)
            else:
                NEW.save.commit_save_image(args, Prepared(prepared))
        states.append(target.read_bytes() if target.exists() else None)
    assert states[0] == states[1] == (prepared[:4] if existing else None)


@pytest.mark.skipif(os.name != "nt", reason="file names ignore case on Windows only")
def test_prepared_save_encodes_as_today_for_another_extension_spelling(tmp_path):
    # An existing image.PNG resolves to that spelling. The bytes were prepared for
    # .png, so the commit encodes as today instead and leaves them unused.
    value = image(3)
    holder = Prepared(b"prepared")
    for name in ("today", "commit"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "image.PNG").write_bytes(b"existing")
        args = list(
            save_arguments(NEW.save, LazyImage(value), tmp_path / name).values()
        )
        if name == "today":
            NEW.save.save_image_node(*args)
        else:
            NEW.save.commit_save_image(args, holder)
    assert not holder.used
    today, commit = (
        (tmp_path / name / "image.PNG").read_bytes() for name in ("today", "commit")
    )
    assert commit == today != b"existing"
