"""Installed-source CPU oracles for model buffers and removal masks."""

from __future__ import annotations

import ast
import ctypes as ct
import sys
import types
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# Imported for its side effect (x as x keeps it): ORT initializes before the
# OpenCV/SciPy Windows DLLs load.
import onnxruntime as onnxruntime
import pytest
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.special import log_softmax

from nodes.impl import native_framework_images as native
from nodes.impl.image_utils import to_uint8
from nodes.impl.native_tensors import cast_numpy
from nodes.impl.onnx.model import SizeReq
from nodes.utils.utils import get_h_w_c

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path(__file__).with_name("reference_framework_images")

_pixel_path = REFERENCE / "image_utils.py"
_pixel_tree = ast.parse(_pixel_path.read_text("utf-8"))
_pixel_tree.body = [
    node
    for node in _pixel_tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name in {"_get_iinfo", "normalize", "to_uint8"}
]
PIXELS = types.ModuleType("original_pixels")
PIXELS.__dict__.update(np=np)
exec(compile(_pixel_tree, str(_pixel_path), "exec"), PIXELS.__dict__)
normalize = PIXELS.normalize


def load(path: Path, **bindings) -> types.ModuleType:
    tree = ast.parse(path.read_text("utf-8"))
    tree.body = [
        node
        for node in tree.body
        if not isinstance(node, ast.ImportFrom)
        or (
            not node.level
            and node.module != "api"
            and not (node.module or "").startswith("nodes.")
        )
    ]
    module = types.ModuleType(path.stem)
    module.__dict__.update(bindings)
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


RESIZE = load(REFERENCE / "resize.py", get_h_w_c=get_h_w_c)
BASE = load(
    REFERENCE / "rembg/session_base.py",
    resize=RESIZE.resize,
    ResizeFilter=RESIZE.ResizeFilter,
)
SIMPLE = load(
    REFERENCE / "rembg/session_simple.py",
    BaseSession=BASE.BaseSession,
    resize=RESIZE.resize,
    ResizeFilter=RESIZE.ResizeFilter,
    get_h_w_c=get_h_w_c,
    normalize=normalize,
)
CLOTH = load(
    REFERENCE / "rembg/session_cloth.py",
    BaseSession=BASE.BaseSession,
    get_h_w_c=get_h_w_c,
    normalize=normalize,
)
BG = load(REFERENCE / "rembg/bg.py", get_h_w_c=get_h_w_c)
ONNX = load(REFERENCE / "onnx/auto_split.py", SizeReq=SizeReq)


@pytest.fixture(scope="module", autouse=True)
def ncnn_gpu_instance():
    # Loading the port's ncnn modules runs their ncnn.get_gpu_count(), which creates
    # ncnn's Vulkan instance. The wheel never destroys it, and a process that exits
    # with it alive crashes in the Vulkan driver (0xC0000005) after every test passed.
    yield
    if (ncnn := sys.modules.get("ncnn.ncnn")) is not None:
        ncnn.destroy_gpu_instance()


class NodeArg:
    """A session input or output for the OnnxSession fakes. A member a case leaves
    unset raises AttributeError, as it did on the SimpleNamespace this replaces."""

    def __init__(self, name: str, kind: str | None = None):
        self._name = name
        self._kind = kind

    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self) -> str:
        if self._kind is None:
            raise AttributeError("type")
        return self._kind

    @property
    def shape(self) -> list[int | str | None]:
        raise AttributeError("shape")


class InertSession:
    """An OnnxSession for a call whose session is never read: every member raises."""

    def get_inputs(self) -> list[NodeArg]:
        raise AssertionError("the session's inputs were read")

    def get_outputs(self) -> list[NodeArg]:
        raise AssertionError("the session's outputs were read")

    def run(self, output_names, input_feed, run_options=None) -> list[np.ndarray]:
        raise AssertionError("the session was run")


def exact(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    if actual.dtype.kind == "f":
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        finite = ~np.isnan(expected)
        kind = {2: np.uint16, 4: np.uint32, 8: np.uint64}[actual.itemsize]
        np.testing.assert_array_equal(
            actual.view(kind)[finite], expected.view(kind)[finite]
        )
    else:
        np.testing.assert_array_equal(actual, expected)


def foreign(array: np.ndarray, kind: str) -> np.ndarray:
    if kind == "reverse":
        return array[::-1, ::-1]
    if kind == "fortran":
        return np.asfortranarray(array)
    if kind == "unaligned":
        result = np.ndarray(array.shape, array.dtype, bytearray(array.nbytes + 1), 1)
        result[:] = array
        return result
    if kind == "readonly":
        array.flags.writeable = False
    if kind == "planar":  # Lens Blur's (c, h, w) base, transposed to (h, w, c)
        return np.ascontiguousarray(array.transpose(2, 0, 1)).transpose(1, 2, 0)
    return array


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("shape", [(1, 1), (3, 1, 1), (1, 4, 3), (3, 5, 4)])
@pytest.mark.parametrize("padding", [(0, 0), (1, 2), (13, 17)])
@pytest.mark.parametrize(
    "kind", ["ordinary", "reverse", "fortran", "unaligned", "readonly"]
)
def test_reflect_pad(dtype, shape, padding, kind):
    image = foreign(np.arange(np.prod(shape), dtype=dtype).reshape(shape), kind)
    before = image.copy()
    width, height = padding
    expected = np.pad(
        image,
        [(0, height), (0, width)] + ([(0, 0)] if image.ndim == 3 else []),
        "reflect",
    )
    actual = native.reflect_pad(image, width, height)
    exact(actual, expected)
    assert actual.strides == expected.strides
    exact(image, before)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("channels", [0, 1, 3, 4, 7])
@pytest.mark.parametrize(
    "kind", ["ordinary", "reverse", "fortran", "unaligned", "readonly"]
)
def test_channel_swap(dtype, channels, kind):
    shape = (5, 7, channels) if channels else (5, 7)
    image = foreign(np.arange(np.prod(shape), dtype=dtype).reshape(shape), kind)
    actual, expected = native.swap_red_blue(image), ONNX._flip_r_b_channels(image)
    exact(actual, expected)
    assert actual.strides == expected.strides


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("shape", [(1, 1), (3, 5), (16, 17)])
@pytest.mark.parametrize("kind", ["ordinary", "reverse", "unaligned", "readonly"])
def test_ncnn_pixels(channels, shape, kind):
    from ncnn import ncnn

    image = foreign(
        np.random.default_rng(871).integers(0, 256, (*shape, channels), dtype=np.uint8),
        kind,
    )
    pixel_type = {
        1: ncnn.Mat.PixelType.PIXEL_GRAY,
        3: ncnn.Mat.PixelType.PIXEL_RGB,
        4: ncnn.Mat.PixelType.PIXEL_RGBA,
    }[channels]
    original = ncnn.Mat.from_pixels(
        np.ascontiguousarray(image), pixel_type, shape[1], shape[0]
    )
    original.substract_mean_normalize([], [1 / 255.0] * channels)
    actual = native.ncnn_input(image)
    exact(actual, np.asarray(original))
    exact(np.asarray(ncnn.Mat(actual)), np.asarray(original))


@pytest.mark.parametrize("shape", [(1, 1, 3), (3, 5, 3), (17, 19, 3)])
@pytest.mark.parametrize(
    "parameters",
    [
        ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ((0.5, 0.5, 0.5), (1, 1, 1)),
        ((0, 0, 0), (1, 1, 1)),
    ],
)
@pytest.mark.parametrize(
    "kind", ["ordinary", "reverse", "fortran", "unaligned", "readonly"]
)
def test_normalize_model(shape, parameters, kind):
    image = foreign(
        np.random.default_rng(732).uniform(-2, 3, shape).astype(np.float32), kind
    )
    mean, std = parameters
    expected = np.zeros(shape)
    for channel in range(3):
        expected[:, :, channel] = (image[:, :, channel] - mean[channel]) / std[channel]
    expected = np.expand_dims(expected.transpose(2, 0, 1), 0).astype(np.float32)
    actual = native.normalize_model(image, mean, std)
    exact(actual, expected)
    assert actual.strides == expected.strides


@pytest.mark.parametrize("shape", [(1, 1, 1), (1, 5, 7), (1, 33, 43)])
@pytest.mark.parametrize(
    "values", ["random", "constant", "zero", "signed", "nan", "infinity", "extreme"]
)
@pytest.mark.parametrize(
    "kind", ["ordinary", "reverse", "fortran", "unaligned", "readonly"]
)
def test_normalize_prediction(shape, values, kind):
    image = np.random.default_rng(773).random(shape, dtype=np.float32)
    if values == "constant":
        image[:] = 0.7
    if values == "zero":
        image[:] = 0
    if values == "signed":
        image[:] = 0
        image.flat[::2] = -0.0
    if values == "nan":
        image.flat[0] = np.nan
    if values == "infinity":
        image.flat[0] = np.inf
    if values == "extreme":
        image.flat[::2] = np.finfo(np.float32).max
        image.flat[1::2] = -np.finfo(np.float32).max
    image = foreign(image, kind)
    with np.errstate(all="ignore"):
        expected = (image - np.min(image)) / (np.max(image) - np.min(image))
        actual = native.normalize_prediction(image)
    exact(actual, expected)


@pytest.mark.parametrize("shape", [(1, 1), (3, 7), (13, 17)])
@pytest.mark.parametrize("size", [-1, 0, 1, 2, 3, 4, 10, 31])
@pytest.mark.parametrize("thresholds", [(240, 10), (127, 127), (0, 255), (255, 0)])
def test_trimap(shape, size, thresholds):
    image = np.random.default_rng(715).random(shape, dtype=np.float32)
    image.flat[0] = np.nan
    foreground, background = thresholds
    structure = np.ones((size, size), np.uint8) if size > 0 else None
    expected = np.full(shape, 0.5)
    # binary_erosion returns a bool array; asarray only types it as one (no copy).
    expected[
        np.asarray(
            binary_erosion(image > foreground / 255, structure=structure), dtype=bool
        )
    ] = 1
    expected[
        np.asarray(
            binary_erosion(
                image < background / 255, structure=structure, border_value=1
            ),
            dtype=bool,
        )
    ] = 0
    exact(native.matting_trimap(image, foreground, background, size), expected)


@pytest.mark.parametrize("classes", [1, 2, 4, 7, 33, 129])
@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (13, 17)])
@pytest.mark.parametrize(
    "mode", ["random", "near_tie", "zero", "nan", "infinity", "negative_infinity"]
)
def test_cloth_labels(classes, shape, mode):
    data = (
        np.random.default_rng(17).normal(0, 10, (1, classes, *shape)).astype(np.float32)
    )
    if mode == "near_tie":
        data *= np.float32(1e-8)
    if mode == "zero":
        data.fill(0)
        data.flat[::2] = -0.0
    if mode == "nan":
        data[:, -1, :, :] = np.nan
    if mode == "infinity":
        data[:, -1, :, :] = np.inf
    if mode == "negative_infinity":
        data.fill(-np.inf)
    with np.errstate(all="ignore"):
        expected = log_softmax(data, 1).argmax(axis=1).squeeze(0).astype(np.uint8)
        actual = native.cloth_labels(data)
    exact(actual, expected)


def test_cloth_palettes_full_uint8():
    labels = np.arange(256, dtype=np.uint8).reshape(16, 16)
    actual = native.cloth_masks(labels)
    for palette, mask in zip(
        [CLOTH.pallete1, CLOTH.pallete2, CLOTH.pallete3], actual, strict=True
    ):
        reference = Image.fromarray(labels, "L")
        reference.putpalette(palette)
        expected = normalize(np.array(reference.convert("RGB").convert("L")))
        exact(mask, expected)


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (7, 1), (3, 5), (19, 31)])
@pytest.mark.parametrize("mode", ["random", "tie", "nan", "infinity", "signed"])
def test_post_process(shape, mode):
    from nodes.impl.rembg.bg import post_process

    mask = np.random.default_rng(665).random(shape, dtype=np.float32)
    if mode == "tie":
        mask.fill(0.5)
    if mode == "nan":
        mask.flat[0] = np.nan
    if mode == "infinity":
        mask.flat[0] = np.inf
    if mode == "signed":
        mask.fill(0)
        mask.flat[::2] = -0.0
    exact(post_process(mask), BG.post_process(mask))


def test_parallel_inputs_immutable():
    image = np.random.default_rng(654).random((17, 19, 3), dtype=np.float32)
    before = image.copy()
    expected = native.normalize_model(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    with ThreadPoolExecutor(8) as executor:
        for actual in executor.map(
            lambda _: native.normalize_model(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            range(24),
        ):
            exact(actual, expected)
    exact(image, before)


@pytest.mark.parametrize(
    "operation",
    [
        "pad",
        "ncnn",
        "affine",
        "mask_threshold",
        "trimap",
        "cloth_labels",
        "cloth_masks",
    ],
)
def test_native_null_guards(operation):
    f32 = np.zeros(100, np.float32)
    f64 = np.zeros(100, np.float64)
    events = (ct.c_int * 4)()
    args = {
        "pad": [None, f32.ctypes.data, 1, 1, 1, 1, 1, 4, 0],
        "ncnn": [None, f32.ctypes.data, 1, 1],
        "affine": [None, f32.ctypes.data, 1, 1, 0, 0.0, 0, events],
        "mask_threshold": [None, f32.ctypes.data, 1],
        "trimap": [None, f64.ctypes.data, 1, 1, 0.9, 0.1, 1],
        "cloth_labels": [None, f32.ctypes.data, 1, 1, 1, events],
        "cloth_masks": [None, f32.ctypes.data, 1],
    }
    assert getattr(native._api(), "cn_framework_" + operation)(*args[operation]) == 1


def recorded(operation):
    with np.errstate(all="warn"), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = operation()
    return result, [str(item.message) for item in caught]


@pytest.mark.parametrize(
    "value",
    [
        0.0,
        -0.0,
        np.nan,
        np.inf,
        -np.inf,
        np.finfo(np.float32).max,
        np.nextafter(np.float32(0), np.float32(1)),
    ],
)
def test_prediction_warning_contract(value):
    image = np.array([value, -value, 0.5], np.float32).reshape(1, 1, 3)

    def original():
        maximum, minimum = np.max(image), np.min(image)
        return (image - minimum) / (maximum - minimum)

    expected, expected_warnings = recorded(original)
    actual, actual_warnings = recorded(lambda: native.normalize_prediction(image))
    exact(actual, expected)
    assert actual_warnings == expected_warnings


@pytest.mark.parametrize("value", [0.0, 1e-40, 1e-5, 1e10, np.inf, -np.inf, np.nan])
def test_cloth_warning_contract(value):
    image = np.array([value, 0.0, -0.7, -1000.0], np.float32).reshape(1, 4, 1, 1)
    expected, expected_warnings = recorded(
        lambda: log_softmax(image, 1).argmax(axis=1).squeeze(0).astype(np.uint8)
    )
    actual, actual_warnings = recorded(lambda: native.cloth_labels(image))
    exact(actual, expected)
    assert actual_warnings == expected_warnings


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("channels", [0, 1, 3, 4])
@pytest.mark.parametrize("nhwc", [False, True])
@pytest.mark.parametrize("padding", [False, True])
def test_onnx_host_image_roundtrip(dtype, channels, nhwc, padding):
    modified = load(
        ROOT / "backend/src/nodes/impl/onnx/auto_split.py",
        SizeReq=SizeReq,
        reflect_pad=native.reflect_pad,
        swap_red_blue=native.swap_red_blue,
        cast_numpy=cast_numpy,
    )
    shape = (3, 5, channels) if channels else (3, 5)
    image = np.random.default_rng(335).random(shape, dtype=np.float32)
    req = SizeReq(minimum=7, multiple_of=4) if padding else SizeReq()

    def single(img, upscale, _tiler, **_kwargs):
        return upscale(img, None)

    class Session:
        def __init__(self):
            self.seen = None

        def get_inputs(self):
            return [
                SimpleNamespace(
                    name="input",
                    type="tensor(float16)" if dtype == np.float16 else "tensor(float)",
                )
            ]

        def get_outputs(self):
            return [SimpleNamespace(name="output")]

        def run(self, _names, feed):
            self.seen = feed["input"]
            return [self.seen.copy()]

    modified.__dict__.update(auto_split=single)
    ONNX.__dict__.update(auto_split=single)
    before, after = Session(), Session()
    expected = ONNX.onnx_auto_split(image, before, nhwc, None, req)
    actual = modified.onnx_auto_split(image, after, nhwc, None, req)
    exact(actual, expected)
    assert after.seen is not None and before.seen is not None
    exact(after.seen, before.seen)
    assert after.seen.strides == before.seen.strides
    assert actual.strides == expected.strides


@pytest.mark.parametrize("kind", ["simple", "cloth"])
@pytest.mark.parametrize("shape", [(1, 1), (3, 5), (9, 13)])
@pytest.mark.parametrize("mode", ["random", "constant", "nonfinite"])
def test_rembg_sessions(kind, shape, mode):
    from nodes.impl.rembg.session_cloth import ClothSession
    from nodes.impl.rembg.session_simple import SimpleSession

    image = np.random.default_rng(21).random((*shape, 3), dtype=np.float32)
    prediction = np.random.default_rng(23).random(
        (1, 4 if kind == "cloth" else 1, 5, 7), dtype=np.float32
    )
    if mode == "constant":
        prediction.fill(0.7)
    if mode == "nonfinite":
        prediction.flat[0] = np.nan

    class Session:
        def __init__(self):
            self.seen = None

        def get_inputs(self):
            return [NodeArg("input")]

        def get_outputs(self) -> list[NodeArg]:
            raise AssertionError("the rembg sessions never read the outputs")

        def run(self, output_names, input_feed, run_options=None):
            self.seen = input_feed["input"]
            return [prediction]

    original_type = SIMPLE.SimpleSession if kind == "simple" else CLOTH.ClothSession
    actual_type = SimpleSession if kind == "simple" else ClothSession
    before, after = Session(), Session()
    args = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), (7, 5))
    with np.errstate(all="ignore"):
        expected = original_type(before, *args).predict(image)
        actual = actual_type(after, *args).predict(image)
    assert after.seen is not None and before.seen is not None
    exact(after.seen, before.seen)
    assert after.seen.strides == before.seen.strides
    for value, reference in zip(actual, expected, strict=True):
        exact(value, reference)


@pytest.mark.parametrize("post", [False, True])
@pytest.mark.parametrize("alpha", [False, True])
@pytest.mark.parametrize("count", [1, 3])
def test_complete_background_host(monkeypatch, post, alpha, count):
    from nodes.impl.rembg import bg

    image = np.random.default_rng(51).random((7, 9, 3), dtype=np.float32)
    masks = [
        np.random.default_rng(70 + i).random((7, 9), dtype=np.float32)
        for i in range(count)
    ]

    def new_session(_session):
        return SimpleNamespace(predict=lambda _image: masks)

    def alpha_field(_image, trimap):
        return trimap

    def foreground(_image, _alpha):
        return _image.astype(np.float32)

    monkeypatch.setattr(bg, "new_session", new_session)
    monkeypatch.setattr(BG, "new_session", new_session, raising=False)
    monkeypatch.setattr(BG, "estimate_alpha_cf", alpha_field)
    monkeypatch.setattr(BG, "estimate_foreground_ml", foreground)
    monkeypatch.setattr(bg, "estimate_alpha", alpha_field)
    monkeypatch.setattr(bg, "estimate_foreground", foreground)
    # new_session is patched on both sides, so neither ever reads the session.
    session = InertSession()
    actual = bg.remove_bg(image, session, alpha, 180, 80, 2, post)
    expected = BG.remove_bg(image, session, alpha, 180, 80, 2, post)
    for a, e in zip(actual, expected, strict=True):
        exact(a, e)


@pytest.mark.parametrize("size", [0, 1, 2, 3])
def test_real_cpu_matting(size):
    from nodes.impl.rembg.bg import alpha_matting_cutout

    image = np.random.default_rng(991).random((9, 11, 3), dtype=np.float32)
    mask = np.tile(np.linspace(0, 1, 11, dtype=np.float32), (9, 1))
    reference = load(REFERENCE / "rembg/bg.py", get_h_w_c=get_h_w_c)
    exact(
        alpha_matting_cutout(image, mask, 180, 80, size),
        reference.alpha_matting_cutout(image, mask, 180, 80, size),
    )


@pytest.mark.parametrize("kind", ["simple", "cloth"])
def test_tiny_real_cpu_onnx_rembg(kind):
    import onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper

    from nodes.impl.native_onnx_runtime import NativeSession
    from nodes.impl.rembg import bg

    prediction = np.random.default_rng(42).random(
        (1, 4 if kind == "cloth" else 1, 5, 7), dtype=np.float32
    )
    width = 768 if kind == "cloth" else 7
    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 3, width, width]
    )
    output_info = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, prediction.shape
    )
    constant = numpy_helper.from_array(prediction, "mask")
    graph = helper.make_graph(
        [helper.make_node("Identity", ["mask"], ["output"])],
        "tiny-rembg",
        [input_info],
        [output_info],
        [constant],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=10
    ).SerializeToString()
    original_session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    native_session = NativeSession(model)
    image = np.random.default_rng(2).random((9, 11, 3), dtype=np.float32)
    from nodes.impl.rembg.session_factory import new_session

    factory = load(
        REFERENCE / "rembg/session_factory.py",
        get_input_shape=lambda session: (None, None, width),
        BaseSession=BASE.BaseSession,
        ClothSession=CLOTH.ClothSession,
        SimpleSession=SIMPLE.SimpleSession,
    )
    BG.__dict__.update(new_session=factory.new_session)
    try:
        expected = BG.remove_bg(image, original_session, post_process_mask=True)
        actual = bg.remove_bg(image, native_session, post_process_mask=True)
        for a, e in zip(actual, expected, strict=True):
            exact(a, e)
        assert type(new_session(native_session)).__name__ == (
            "ClothSession" if kind == "cloth" else "SimpleSession"
        )
    finally:
        native_session.close()


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize(
    "kind", ["ordinary", "reverse", "fortran", "readonly", "planar"]
)
def test_tiny_real_cpu_ncnn_roundtrip(channels, kind):
    from ncnn import ncnn

    def single(image, upscale, _tiler, **_kwargs):
        return upscale(image, None)

    source = ROOT / "backend/src/nodes/impl/ncnn/auto_split.py"
    baseline = load(
        REFERENCE / "ncnn/auto_split.py",
        get_h_w_c=get_h_w_c,
        to_uint8=PIXELS.to_uint8,
        auto_split=single,
    )
    modified = load(
        source,
        get_h_w_c=get_h_w_c,
        to_uint8=to_uint8,
        auto_split=single,
        ncnn_input=native.ncnn_input,
        cast_numpy=cast_numpy,
    )
    baseline.__dict__.update(use_gpu=False)
    modified.__dict__.update(use_gpu=False)
    network = ncnn.Net()
    network.opt.use_vulkan_compute = False
    network.opt.num_threads = 1
    network.opt.use_fp16_packed = False
    network.opt.use_fp16_storage = False
    network.opt.use_fp16_arithmetic = False
    assert (
        network.load_param_mem(
            "7767517\n2 2\nInput input 0 1 input\nNoop identity 1 1 input output\n"
        )
        == 0
    )
    image = foreign(
        np.random.default_rng(547)
        .uniform(-0.3, 1.3, (7, 9, channels))
        .astype(np.float32),
        kind,
    )
    # Upstream's Mat.from_pixels reads the raw buffer, so it reads a Fortran or
    # planar image's memory as interleaved pixels. The port reads by logical index
    # (Consult 11 D-18): its result is upstream's on the image's C copy. Upstream's
    # own read differs exactly where memory order is not logical order.
    expected = baseline.ncnn_auto_split(
        np.ascontiguousarray(image), network, "input", "output", None, None, None
    )
    raw = baseline.ncnn_auto_split(image, network, "input", "output", None, None, None)
    actual = modified.ncnn_auto_split(
        image, network, "input", "output", None, None, None
    )
    exact(actual, expected)
    assert actual.strides == expected.strides
    logical = np.array_equal(image.ravel(order="K"), image.ravel(order="C"))
    assert np.array_equal(raw, expected) == logical


@pytest.mark.parametrize(
    ("operation", "mode"),
    [
        (operation, mode)
        for operation in (
            "pad",
            "ncnn",
            "affine",
            "mask_threshold",
            "trimap",
            "cloth_labels",
            "cloth_masks",
        )
        for mode in (
            "overlap",
            "unaligned",
            "overflow",
            "events_alias",
            "events_alignment",
        )
        if not mode.startswith("events_") or operation in ("affine", "cloth_labels")
    ],
)
def test_native_foreign_buffer_guards(operation, mode):
    source = np.zeros(128, np.float32)
    destination = np.full(128, 91, np.float64)
    events = (ct.c_int * 4)()
    src, dst, n = source.ctypes.data, destination.ctypes.data, 8
    if mode == "overlap":
        dst = src + 4
    if mode == "unaligned":
        if operation in ("ncnn", "cloth_masks"):
            dst += 1
        else:
            src += 1
    if mode == "overflow":
        n = 2**64 - 1
    flags = dst if mode == "events_alias" else events
    if mode == "events_alignment":
        flags = ct.addressof(events) + 1
    args = {
        "pad": [src, dst, n, 1, 1, n, 1, 4, 0],
        "ncnn": [src, dst, n, 3],
        "affine": [src, dst, n, 1, 0, 0.0, 0, flags],
        "mask_threshold": [src, dst, n],
        "trimap": [src, dst, n, 1, 0.9, 0.1, 1],
        "cloth_labels": [src, dst, n, 4, 1, flags],
        "cloth_masks": [src, dst, n],
    }
    status = getattr(native._api(), "cn_framework_" + operation)(*args[operation])
    assert status == (2 if mode == "overflow" else 1)
    np.testing.assert_array_equal(destination, 91)


@pytest.mark.parametrize("kind", ["reverse", "fortran", "unaligned", "readonly"])
def test_cloth_foreign_and_concurrent(kind):
    image = np.random.default_rng(843).normal(0, 0.01, (1, 4, 7, 11)).astype(np.float32)
    if kind == "reverse":
        image = image[:, :, ::-1, ::-1]
    elif kind == "fortran":
        image = np.asfortranarray(image)
    elif kind == "unaligned":
        unaligned = np.ndarray(image.shape, np.float32, bytearray(image.nbytes + 1), 1)
        unaligned[:] = image
        image = unaligned
    else:
        image.flags.writeable = False
    before = image.copy()
    expected = log_softmax(image, 1).argmax(axis=1).squeeze(0).astype(np.uint8)
    with ThreadPoolExecutor(4) as executor:
        for actual in executor.map(lambda _: native.cloth_labels(image), range(8)):
            exact(actual, expected)
    exact(image, before)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("nhwc", [False, True])
def test_real_cpu_onnx_image_roundtrip(dtype, channels, nhwc):
    import onnxruntime as ort
    from onnx import TensorProto, helper

    from nodes.impl.native_onnx_runtime import NativeSession

    shape = [1, "h", "w", channels] if nhwc else [1, channels, "h", "w"]
    data_type = TensorProto.FLOAT16 if dtype == np.float16 else TensorProto.FLOAT
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "image-buffer-roundtrip",
        [helper.make_tensor_value_info("input", data_type, shape)],
        [helper.make_tensor_value_info("output", data_type, shape)],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=10
    ).SerializeToString()
    before = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    after = NativeSession(model)

    def single(image, upscale, _tiler, **_kwargs):
        return upscale(image, None)

    modified = load(
        ROOT / "backend/src/nodes/impl/onnx/auto_split.py",
        SizeReq=SizeReq,
        reflect_pad=native.reflect_pad,
        swap_red_blue=native.swap_red_blue,
        cast_numpy=cast_numpy,
        auto_split=single,
    )
    original = load(
        REFERENCE / "onnx/auto_split.py", SizeReq=SizeReq, auto_split=single
    )
    image = np.random.default_rng(158).random((3, 7, channels), dtype=np.float32)
    try:
        expected = original.onnx_auto_split(image, before, nhwc, None, SizeReq(8, 4))
        actual = modified.onnx_auto_split(image, after, nhwc, None, SizeReq(8, 4))
        exact(actual, expected)
        assert actual.strides == expected.strides
    finally:
        after.close()


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("classes", [4, 7, 9, 33])
@pytest.mark.parametrize("fortran", [False, True])
def test_cloth_rounding_ties(seed, classes, fortran):
    image = (
        np.random.default_rng(seed)
        .normal(0, 1e-7, (1, classes, 7, 11))
        .astype(np.float32)
    )
    if fortran:
        image = np.asfortranarray(image)
    expected = log_softmax(image, 1).argmax(axis=1).squeeze(0).astype(np.uint8)
    exact(native.cloth_labels(image), expected)


@pytest.mark.parametrize("mode", ["warn", "raise", "call", "log", "ignore"])
def test_prediction_error_policy(mode):
    import io

    data = np.ones((1, 3, 5), np.float32)

    def execute(function):
        messages = []
        stream = io.StringIO()
        handler = (
            stream
            if mode == "log"
            else lambda description, flags: messages.append((description, flags))
        )
        with (
            np.errstate(all=mode, call=handler),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            try:
                function()
            except FloatingPointError as error:
                messages.append(str(error))
        return messages, stream.getvalue(), [str(item.message) for item in caught]

    expected = execute(lambda: (data - data.min()) / (data.max() - data.min()))
    actual = execute(lambda: native.normalize_prediction(data))
    assert actual == expected
