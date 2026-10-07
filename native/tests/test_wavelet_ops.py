"""CPU wavelet arithmetic against frozen Torch; no GPU/model execution."""

from __future__ import annotations

import ast
import ctypes as ct
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional

from nodes.impl import native_wavelet as native
from nodes.impl.pytorch.utils import np2tensor, safe_cuda_cache_empty, tensor2np
from nodes.impl.resize import ResizeFilter, resize
from nodes.utils.utils import get_h_w_c

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "backend/src/packages/chaiNNer_pytorch/pytorch/processing/wavelet_color_fix.py"
)
REFERENCE = Path(__file__).with_name("reference_wavelet") / "wavelet_color_fix.py"


def load(path):
    tree = ast.parse(path.read_text("utf-8"))
    tree.body = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
        *[node for node in tree.body if isinstance(node, ast.FunctionDef)],
    ]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    module = types.ModuleType(path.stem)
    module.__dict__.update(
        np=np,
        torch=torch,
        F=functional,
        native_wavelet=native,
        np2tensor=np2tensor,
        tensor2np=tensor2np,
        safe_cuda_cache_empty=safe_cuda_cache_empty,
        resize=resize,
        ResizeFilter=ResizeFilter,
        get_h_w_c=get_h_w_c,
        get_settings=lambda context: types.SimpleNamespace(
            device=torch.device("cpu"), force_cache_wipe=False
        ),
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), module.__dict__)
    return module


ACTUAL, ORIGINAL = load(SOURCE), load(REFERENCE)


def image(shape=(1, 3, 13, 17), seed=731):
    return torch.from_numpy(np.random.default_rng(seed).random(shape, dtype=np.float32))


def exact(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    a, b = (
        actual.detach().cpu().float().numpy(),
        expected.detach().cpu().float().numpy(),
    )
    finite = np.isfinite(b)
    assert a[finite].tobytes() == b[finite].tobytes()


@pytest.mark.parametrize(
    "shape", [(1, 3, 1, 1), (1, 3, 1, 7), (1, 3, 5, 1), (2, 3, 7, 11)]
)
@pytest.mark.parametrize("radius", [1, 2, 7, 64, 512])
def test_blur_degenerate_and_dilated(shape, radius):
    source = image(shape)
    before = source.clone()
    result = native.blur(source, radius)
    assert result is not None
    exact(result, ORIGINAL.wavelet_blur(source, radius))
    exact(source, before)


@pytest.mark.parametrize("levels", [1, 2, 3, 5, 10])
@pytest.mark.parametrize(
    "shape", [(1, 3, 1, 1), (1, 3, 1, 7), (1, 3, 7, 1), (1, 3, 19, 23), (2, 3, 9, 11)]
)
def test_full_decomposition_and_reconstruction(shape, levels):
    content, style = image(shape), image(shape, seed=919)
    before = content.clone(), style.clone()
    actual = native.decomposition(content, levels)
    assert actual is not None
    expected = ORIGINAL.wavelet_decomposition(content, levels)
    for a, b in zip(actual, expected, strict=True):
        exact(a, b)
    result = native.reconstruction(content, style, levels)
    assert result is not None
    exact(result, ORIGINAL.wavelet_reconstruction(content, style, levels))
    exact(content, before[0])
    exact(style, before[1])


@pytest.mark.parametrize(
    "pattern", ["zero", "negative-zero", "one", "constant", "impulse", "checker", "ulp"]
)
@pytest.mark.parametrize("levels", [1, 3, 5])
def test_exact_rounding_patterns(pattern, levels):
    source = image((1, 3, 7, 11))
    if pattern == "zero":
        source.zero_()
    elif pattern == "negative-zero":
        source.fill_(-0.0)
    elif pattern == "one":
        source.fill_(1)
    elif pattern == "constant":
        source.fill_(0.37)
    elif pattern == "impulse":
        source.zero_()
        source[0, :, 3, 5] = 1
    elif pattern == "checker":
        source[:] = torch.from_numpy(
            (np.indices((7, 11)).sum(axis=0) % 2).astype(np.float32)
        )
    else:
        source[:] = torch.from_numpy(
            np.where(
                source.numpy() > 0.5,
                np.nextafter(np.float32(0.5), np.float32(1)),
                np.float32(0.5),
            )
        )
    actual = native.decomposition(source, levels)
    assert actual is not None
    for a, b in zip(
        actual, ORIGINAL.wavelet_decomposition(source, levels), strict=True
    ):
        exact(a, b)


@pytest.mark.parametrize(
    "layout", ["strided", "channels-last", "readonly", "unaligned"]
)
def test_foreign_buffers_and_framework_layout(layout):
    source = image()
    if layout == "strided":
        source = source[:, :, ::2, ::2]
    elif layout == "channels-last":
        source = source.contiguous(memory_format=torch.channels_last)
    else:
        data = source.numpy()
        if layout == "unaligned":
            copy = np.ndarray(
                data.shape, np.float32, buffer=bytearray(data.nbytes + 1), offset=1
            )
            copy[:] = data
            data = copy
        else:
            data.flags.writeable = False
        source = torch.from_numpy(data)
    before = source.clone()
    exact(
        ACTUAL.wavelet_reconstruction(source, source, 3),
        ORIGINAL.wavelet_reconstruction(source, source, 3),
    )
    exact(source, before)


@pytest.mark.parametrize(
    "value",
    [
        -0.5,
        1.5,
        np.nan,
        np.inf,
        -np.inf,
        np.float32(2**-126),
        np.nextafter(np.float32(0), np.float32(1)),
    ],
)
def test_exceptional_numbers_use_fused_c(value):
    source = image((1, 3, 7, 11))
    source[0, 1, 3, 5] = float(value)
    assert native.decomposition(source, 3) is not None
    assert native.reconstruction(source, source, 3) is not None
    exact(
        ACTUAL.wavelet_reconstruction(source, source, 3),
        ORIGINAL.wavelet_reconstruction(source, source, 3),
    )


def test_autograd_and_cpu_autocast_remain_framework_operations():
    a = image((1, 3, 7, 11)).requires_grad_()
    b = a.detach().clone().requires_grad_()
    assert native.decomposition(a, 3) is None
    ACTUAL.wavelet_reconstruction(a, a, 3).sum().backward()
    ORIGINAL.wavelet_reconstruction(b, b, 3).sum().backward()
    exact(a.grad, b.grad)
    source = a.detach()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert native.decomposition(source, 3) is None
        exact(
            ACTUAL.wavelet_reconstruction(source, source, 3),
            ORIGINAL.wavelet_reconstruction(source, source, 3),
        )


def test_unsupported_tensor_modes_do_not_transfer_devices():
    assert native.blur(torch.empty((1, 3, 7, 11), device="meta"), 1) is None
    for source in (
        image().double(),
        image().half(),
        image().squeeze(0),
        image()[:, :2],
        image()[:0],
    ):
        assert native.decomposition(source, 3) is None
    source = image((1, 3, 7, 11)).double()
    exact(
        ACTUAL.wavelet_reconstruction(source, source, 3),
        ORIGINAL.wavelet_reconstruction(source, source, 3),
    )
    assert native.reconstruction(image(), image((1, 3, 1, 1)), 3) is None
    exact(
        ACTUAL.wavelet_decomposition(source, 0)[0],
        ORIGINAL.wavelet_decomposition(source, 0)[0],
    )


@pytest.mark.parametrize(
    "shape,reference_shape",
    [((1, 1, 3), (3, 5, 3)), ((13, 17, 3), (7, 9, 3)), ((7, 11, 3), (7, 11, 3))],
)
@pytest.mark.parametrize("levels", [1, 3, 5])
def test_real_node_cpu_only(shape, reference_shape, levels):
    rng = np.random.default_rng(221)
    target = rng.random(shape, dtype=np.float32)
    source = rng.random(reference_shape, dtype=np.float32)
    cleanups = []
    context = types.SimpleNamespace(
        add_cleanup=lambda function, after: cleanups.append((function, after))
    )
    actual = ACTUAL.wavelet_color_fix_node(context, target, source, levels)
    expected = ORIGINAL.wavelet_color_fix_node(context, target, source, levels)
    assert actual.dtype == expected.dtype == np.float32
    assert actual.shape == expected.shape == shape
    assert actual.tobytes() == expected.tobytes()
    assert cleanups == [(safe_cuda_cache_empty, "chain")] * 2


def test_native_route_without_framework_convolution(monkeypatch):
    source = image((1, 3, 11, 13))
    expected = ORIGINAL.wavelet_reconstruction(source, source, 3)

    def forbidden(*args, **kwargs):
        raise AssertionError("The supported CPU path must execute C convolution")

    monkeypatch.setattr(functional, "conv2d", forbidden)
    exact(ACTUAL.wavelet_reconstruction(source, source, 3), expected)


def test_concurrent_large_images():
    sources = [image((1, 3, 91, 103), seed=i) for i in range(3)]
    expected = [
        ORIGINAL.wavelet_reconstruction(source, sources[(i + 1) % 3], 3)
        for i, source in enumerate(sources)
    ]
    with ThreadPoolExecutor(max_workers=6) as pool:
        actual = list(
            pool.map(
                lambda i: ACTUAL.wavelet_reconstruction(
                    sources[i % 3], sources[(i + 1) % 3], 3
                ),
                range(12),
            )
        )
    for i, result in enumerate(actual):
        exact(result, expected[i % 3])


def test_abi_validation_and_error_propagation(monkeypatch):
    api = native._api()  # validate the foreign function boundary
    data = np.full(16, 9, np.float32)
    pointer = data.ctypes.data_as(ct.POINTER(ct.c_float))
    maximum = ct.c_size_t(-1).value
    assert api.cn_wavelet_blur_f32(None, pointer, 1, 1, 1, 1, 1) == 1
    assert api.cn_wavelet_blur_f32(pointer, pointer, maximum, 1, 1, 1, 1) == 2
    assert api.cn_wavelet_decompose_f32(pointer, pointer, pointer, 1, 1, 1, 11, 1) == 1
    assert (
        api.cn_wavelet_decompose_f32(pointer, pointer, pointer, 1, maximum, 2, 1, 1)
        == 2
    )
    assert (
        api.cn_wavelet_reconstruct_f32(
            pointer, pointer, pointer, 1, 1, maximum // 12, 1, 1
        )
        == 2
    )
    np.testing.assert_array_equal(data, 9)

    def broken():
        raise RuntimeError("test DLL error")

    monkeypatch.setattr(native, "_api", broken)
    with pytest.raises(RuntimeError, match="test DLL error"):
        ACTUAL.wavelet_reconstruction(image(), image(), 3)


def test_schema_and_node_signature_unchanged():
    def node(path):
        return next(
            item
            for item in ast.parse(path.read_text("utf-8")).body
            if isinstance(item, ast.FunctionDef)
            and item.name == "wavelet_color_fix_node"
        )

    a, b = node(SOURCE), node(REFERENCE)
    assert ast.dump(a) == ast.dump(b)


@pytest.mark.parametrize("levels", [1, 5, 10])
def test_small_normal_values_at_accepted_boundary(levels):
    lower = float(np.finfo(np.float32).tiny) * 2.0 ** (4 * levels + 4)
    source = image((1, 3, 5, 7)) * lower + lower
    actual = native.decomposition(source, levels)
    assert actual is not None
    for a, b in zip(
        actual, ORIGINAL.wavelet_decomposition(source, levels), strict=True
    ):
        exact(a, b)


def test_torch_backend_precision_options():
    source = image((1, 3, 7, 11))
    with torch.backends.mkldnn.flags(enabled=False):
        result = native.reconstruction(source, source, 3)
        assert result is not None
        exact(result, ORIGINAL.wavelet_reconstruction(source, source, 3))
    convolution = getattr(torch.backends.mkldnn, "conv", None)
    if convolution is not None and hasattr(convolution, "fp32_precision"):
        previous = convolution.fp32_precision
        try:
            convolution.fp32_precision = "bf16"
            assert native.reconstruction(source, source, 3) is None
        finally:
            convolution.fp32_precision = previous
