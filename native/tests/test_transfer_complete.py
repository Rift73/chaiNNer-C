"""Strict transfer parameters and final node float32 parity, CPU only."""

from __future__ import annotations

import ast
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from runpy import run_path

import numpy as np
import pytest

from nodes.impl import native_transfer_complete as native
from nodes.impl.color_transfer.linear_histogram import linear_histogram_transfer
from nodes.impl.color_transfer.principal_color import principal_color_transfer

REFERENCE = Path(__file__).with_name("reference_transfer")
LINEAR = run_path(str(REFERENCE / "linear_histogram.py"))["linear_histogram_transfer"]
PRINCIPAL = run_path(str(REFERENCE / "principal_color.py"))["principal_color_transfer"]


def original_parameters(principal):
    name = "principal_color" if principal else "linear_histogram"
    source = (REFERENCE / (name + ".py")).read_text()
    if principal:
        source = source.replace(
            "    transfer = (content - mu_content).dot(transform.T) + mu_reference",
            "    return mu_content, mu_reference, transform",
        )
    else:
        source = source.replace(
            "    transfer = transfer.dot((content - mu_content).T).T",
            "    return mu_content, mu_reference, transfer",
        )
    namespace = {}
    exec(compile(source, name, "exec"), namespace)
    return namespace[name + "_transfer"]


PARAMETERS = [original_parameters(False), original_parameters(True)]


def fixtures(seed, dtype, size, kind="random"):
    rng = np.random.default_rng(seed)
    source = rng.random((1, size, 3)).astype(dtype)
    reference = rng.random((13, 17, 3)).astype(dtype)
    mask = rng.random(source.shape[:2]) > 0.2
    reference_mask = rng.random(reference.shape[:2]) > 0.2
    if kind == "rank_one":
        source[..., 1:] = source[..., :1]
        reference[..., 1:] = reference[..., :1]
    elif kind == "constant":
        source[:] = 0.25
        reference[:] = 0.75
    elif kind == "identical":
        reference = source.copy()
        reference_mask = mask.copy()
    return source, reference, mask, reference_mask


def assert_exact(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    # Upstream's layout too (Consult 11 D-16): Linear's planar result, np.dstack's
    # order K.
    assert actual.strides == expected.strides
    # Compare values and zero signs; NaN payloads are not transformed parameters.
    np.testing.assert_array_equal(
        np.ascontiguousarray(actual).view(np.uint8).reshape(-1),
        np.ascontiguousarray(expected).view(np.uint8).reshape(-1),
    )


@pytest.mark.parametrize("principal", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("size", [3, 7, 129, 8191, 8193, 17001])
@pytest.mark.parametrize("seed", [0, 349, 1821])
@pytest.mark.parametrize("kind", ["random", "rank_one", "constant", "identical"])
def test_parameters_and_final_float32(principal, dtype, size, seed, kind):
    args = fixtures(seed, dtype, size, kind)
    with np.errstate(all="ignore"):
        try:
            expected_parameters = PARAMETERS[principal](*args)
            expected = (PRINCIPAL if principal else LINEAR)(*args)
        except (ValueError, np.linalg.LinAlgError) as original_error:
            with pytest.raises(type(original_error)) as current_error:
                native.transfer(*args, principal)
            assert str(current_error.value) == str(original_error)
            return
        _, *actual_parameters = native.parameters(*args, principal)
        actual = native.transfer(*args, principal)
    for a, e in zip(actual_parameters, expected_parameters, strict=True):
        assert_exact(a, e)
    with np.errstate(all="ignore"):
        np.testing.assert_array_equal(
            actual.astype(np.float32).view(np.uint32),
            expected.astype(np.float32).view(np.uint32),
        )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("principal", [False, True])
@pytest.mark.parametrize("count", [0, 1, 2, 3])
@pytest.mark.parametrize("reference_count", [0, 1, 2, 3])
def test_transparency_count_exception_order(principal, count, reference_count):
    args = fixtures(18, np.float32, 7)
    args[2][:] = args[3][:] = False
    args[2].flat[:count] = True
    args[3].flat[:reference_count] = True
    with np.errstate(all="ignore"):
        try:
            expected = (PRINCIPAL if principal else LINEAR)(*args)
        except Exception as error:
            with pytest.raises(type(error)) as current:
                native.transfer(*args, principal)
            assert str(current.value) == str(error)
        else:
            actual = native.transfer(*args, principal)
            np.testing.assert_array_equal(
                actual.astype(np.float32), expected.astype(np.float32)
            )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize(
    "layout", ["plain", "strided", "readonly", "fortran", "unaligned"]
)
def test_mask_and_alpha_assembly(dtype, channels, layout):
    image = np.random.default_rng(93).uniform(-1, 1, (13, 17, channels)).astype(dtype)
    if channels == 4:
        image.flat[3:16:4] = [np.nan, np.inf, -0.0, -np.inf]
    if layout == "strided":
        image = image[::-1, ::2]
    elif layout == "readonly":
        image.flags.writeable = False
    elif layout == "fortran":
        image = np.asfortranarray(image)
    elif layout == "unaligned":
        buffer = np.ndarray(
            image.shape, dtype=dtype, buffer=bytearray(image.nbytes + 1), offset=1
        )
        buffer[:] = image
        image = buffer
    before = image.copy()
    expected_mask = (
        image[..., 3] > 0 if channels == 4 else np.ones(image.shape[:2], bool)
    )
    np.testing.assert_array_equal(native.transfer_mask(image), expected_mask)
    if channels == 4:
        for target_dtype in (np.float32, np.float64, np.complex128):
            result = image[..., :3].astype(target_dtype)
            expected = np.dstack((result, image[..., 3]))
            assert_exact(native.transfer_with_alpha(result, image), expected)
    np.testing.assert_array_equal(image, before)


def test_concurrency_and_no_python_matrix_algorithms(monkeypatch):
    args = fixtures(419, np.float32, 333)
    expected = [LINEAR(*args), PRINCIPAL(*args)]
    for operation in ("eig", "det", "inv"):
        monkeypatch.setattr(
            np.linalg,
            operation,
            lambda *a, **k: pytest.fail("Python matrix algorithm used"),
        )
    monkeypatch.setattr(
        np, "cov", lambda *a, **k: pytest.fail("Python covariance used")
    )
    monkeypatch.setattr(np, "mean", lambda *a, **k: pytest.fail("Python mean used"))

    def run(i):
        return (principal_color_transfer if i % 2 else linear_histogram_transfer)(*args)

    with ThreadPoolExecutor(max_workers=6) as pool:
        for i, result in enumerate(pool.map(run, range(24))):
            np.testing.assert_array_equal(
                result.astype(np.float32), expected[i % 2].astype(np.float32)
            )


@pytest.mark.parametrize(
    "bad",
    [
        np.empty((0, 3), np.float32),
        np.empty((2, 3, 2), np.float32),
        np.empty((2, 3, 3, 1), np.float32),
        np.empty((2, 3, 3), np.uint8),
    ],
)
def test_invalid_bridge_shapes_do_not_dispatch(monkeypatch, bad):
    monkeypatch.setattr(native, "_api", lambda: pytest.fail("Invalid shape reached C"))
    with pytest.raises(ValueError):
        native.parameters(bad, bad, np.ones((2, 3), bool), np.ones((2, 3), bool), False)


def load_node(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [
        node
        for node in tree.body
        if not isinstance(node, ast.ImportFrom)
        or (
            node.level == 0
            and node.module not in {"api", "nodes.groups"}
            and not (node.module or "").startswith("nodes.properties")
        )
    ]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    module = types.ModuleType(path.stem)
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


NODE = load_node(
    Path(__file__).resolve().parents[2]
    / "backend/src/packages/chaiNNer_standard/image_filter/correction/color_transfer.py"
)
ORIGINAL_NODE = load_node(REFERENCE / "color_transfer_node.py")
ORIGINAL_MEAN = run_path(
    str(
        Path(__file__).with_name("reference_analysis")
        / "nodes/impl/color_transfer/mean_std.py"
    )
)
ORIGINAL_NODE.__dict__.update(mean_std_transfer=ORIGINAL_MEAN["mean_std_transfer"])
ORIGINAL_NODE.__dict__.update(TransferColorSpace=ORIGINAL_MEAN["TransferColorSpace"])
ORIGINAL_NODE.__dict__.update(OverflowMethod=ORIGINAL_MEAN["OverflowMethod"])
ORIGINAL_NODE.__dict__.update(linear_histogram_transfer=LINEAR)
ORIGINAL_NODE.__dict__.update(principal_color_transfer=PRINCIPAL)


@pytest.mark.parametrize(
    "algorithm", ["MEAN_STD", "LINEAR_HISTOGRAM", "PRINCIPAL_COLOR"]
)
@pytest.mark.parametrize("colorspace", ["RGB", "LAB"])
@pytest.mark.parametrize("overflow", ["CLIP", "SCALE"])
@pytest.mark.parametrize("reciprocal", [False, True])
@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("layout", ["plain", "strided", "readonly"])
def test_complete_node_normalized_pixels(
    algorithm, colorspace, overflow, reciprocal, channels, layout
):
    rng = np.random.default_rng(623)
    image = rng.random((17, 23, channels)).astype(np.float32)
    reference = rng.random((19, 13, channels)).astype(np.float32)
    if channels == 4:
        image[..., 3] = np.where(image[..., 3] > 0.5, image[..., 3], 0)
        reference[..., 3] = np.where(reference[..., 3] > 0.5, reference[..., 3], 0)
    if layout == "strided":
        image, reference = image[::-1, ::2], reference[::2, ::-1]
    elif layout == "readonly":
        image.flags.writeable = reference.flags.writeable = False
    before, ref_before = image.copy(), reference.copy()
    outputs = []
    for module in [NODE, ORIGINAL_NODE]:
        outputs.append(
            module.color_transfer_node(
                image,
                reference,
                module.TransferColorAlgorithm[algorithm],
                module.TransferColorSpace[colorspace],
                module.OverflowMethod[overflow],
                reciprocal,
            )
        )
    assert outputs[0].dtype == outputs[1].dtype
    np.testing.assert_array_equal(
        outputs[0].astype(np.float32).view(np.uint32),
        outputs[1].astype(np.float32).view(np.uint32),
    )
    np.testing.assert_array_equal(image, before)
    np.testing.assert_array_equal(reference, ref_before)


@pytest.mark.parametrize("principal", [False, True])
@pytest.mark.parametrize("size,seed", [(3, 2), (35, 85)])
@pytest.mark.parametrize("complex_source", [False, True])
def test_confirmed_complex_eigensystem(principal, size, seed, complex_source):
    rank = np.random.default_rng(seed).random((1, size, 3)).astype(np.float32)
    rank[..., 1:] = rank[..., :1]
    other = np.random.default_rng(987).random((7, 11, 3)).astype(np.float32)
    source, reference = (rank, other) if complex_source else (other, rank)
    args = (
        source,
        reference,
        np.ones(source.shape[:2], bool),
        np.ones(reference.shape[:2], bool),
    )
    expected_parameters = PARAMETERS[principal](*args)
    _, *actual_parameters = native.parameters(*args, principal)
    assert expected_parameters[-1].dtype == np.complex128
    for actual, expected in zip(actual_parameters, expected_parameters, strict=True):
        assert_exact(actual, expected)
    expected = (PRINCIPAL if principal else LINEAR)(*args)
    actual = native.transfer(*args, principal)
    assert actual.dtype == np.complex128
    np.testing.assert_array_equal(actual, expected)
    with np.errstate(all="ignore"):
        np.testing.assert_array_equal(
            actual.astype(np.float32), expected.astype(np.float32)
        )


@pytest.mark.parametrize("principal", [False, True])
@pytest.mark.parametrize("special", [np.nan, np.inf, -np.inf, -0.0])
@pytest.mark.parametrize("selected", [False, True])
def test_nonfinite_pixels_and_mask_selection(principal, special, selected):
    args = fixtures(442, np.float32, 31)
    args[0][0, 0, 1] = special
    args[2][0, 0] = selected
    with np.errstate(all="ignore"):
        try:
            expected = (PRINCIPAL if principal else LINEAR)(*args)
        except np.linalg.LinAlgError as error:
            with pytest.raises(np.linalg.LinAlgError) as actual_error:
                native.transfer(*args, principal)
            assert str(actual_error.value) == str(error)
        else:
            actual = native.transfer(*args, principal)
            np.testing.assert_array_equal(
                actual.astype(np.float32).view(np.uint32),
                expected.astype(np.float32).view(np.uint32),
            )


def test_raw_parameters_reject_invalid_buffers_before_outputs():
    import ctypes as ct

    api = native._api()  # raw ABI regression
    image = np.zeros((2, 3, 3), np.float32)
    mask = np.ones((2, 3), bool)
    mean = np.full(3, 31, np.float32)
    matrix = np.full((3, 3), 43 + 7j, np.complex128)
    error = ct.c_int(19)
    arguments = [
        image.ctypes.data,
        image.ctypes.data,
        mask.ctypes.data,
        mask.ctypes.data,
        6,
        6,
        3,
        1,
        3,
        1,
        0,
        0,
        mean.ctypes.data,
        mean.ctypes.data,
        matrix.ctypes.data,
        ct.byref(error),
        ct.byref(native._backend()[1]),  # raw ABI regression
    ]
    for index, value in [
        (0, None),
        (1, None),
        (2, None),
        (3, None),
        (4, 0),
        (4, 2**64 - 1),
        (5, 2**64 - 1),
        (6, 0),
        (7, 0),
        (8, 0),
        (9, 0),
        (10, 2),
        (11, 2),
        (12, None),
        (13, None),
        (14, None),
        (15, None),
        (16, None),
    ]:
        invalid: list[object] = list(arguments)
        invalid[index] = value
        assert api.cn_transfer_parameters(*invalid) in (1, 2)
        np.testing.assert_array_equal(mean, 31)
        np.testing.assert_array_equal(matrix, 43 + 7j)
        assert error.value == 19


@pytest.mark.parametrize("kind", ["empty", "float64", "integer_mask", "uint8"])
def test_mean_std_public_helper_compatibility(kind):
    from nodes.impl.color_transfer import mean_std

    image, reference, valid, reference_valid = fixtures(311, np.float32, 19)
    if kind == "empty":
        image, valid = image[:, :0], valid[:, :0]
    elif kind == "float64":
        image, reference = image.astype(np.float64), reference.astype(np.float64)
    elif kind == "integer_mask":
        valid = valid.astype(np.int64)
        reference_valid = reference_valid.astype(np.int64)
    else:
        image, reference = (
            (image * 255).astype(np.uint8),
            (reference * 255).astype(np.uint8),
        )
    outputs = []
    for module in [ORIGINAL_MEAN, vars(mean_std)]:
        try:
            with np.errstate(all="ignore"):
                outputs.append(
                    module["mean_std_transfer"](
                        image,
                        reference,
                        module["TransferColorSpace"].RGB,
                        module["OverflowMethod"].CLIP,
                        valid,
                        reference_valid,
                        True,
                    )
                )
        except Exception as error:
            outputs.append((type(error), str(error)))
    if isinstance(outputs[0], tuple):
        assert outputs[0] == outputs[1]
    else:
        np.testing.assert_array_equal(outputs[0], outputs[1])
