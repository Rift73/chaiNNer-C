"""Exact angle-mode parity, including CPU dispatch and hemisphere boundaries."""

from __future__ import annotations

import ast
import ctypes
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import reference_normals_addition as original
import reference_normals_util as original_util
from numpy._core._multiarray_umath import __cpu_dispatch__

from nodes.impl import native_geometry
from nodes.impl.image_utils import NormalMapType
from nodes.impl.native import ptr
from nodes.impl.normals import addition, util


def exact(actual, expected):
    for a, e in zip(actual, expected, strict=True):
        assert a.dtype == e.dtype
        assert a.shape == e.shape
        np.testing.assert_array_equal(np.isnan(a), np.isnan(e))
        finite = ~np.isnan(e)
        np.testing.assert_array_equal(
            a[finite].view(np.uint32), e[finite].view(np.uint32)
        )


FACTORS = [(1, 1), (0, 0), (-0.0, 0), (0.3, 1.7), (-2, 5), (200, -200), (1e30, -1e30)]


@pytest.mark.parametrize("shape", [(0, 3, 3), (1, 1, 3), (1, 17, 4), (19, 31, 3)])
@pytest.mark.parametrize("factors", FACTORS)
@pytest.mark.parametrize("mode", ["random", "horizon", "exceptional", "strided"])
@pytest.mark.parametrize("strengthen", [False, True])
def test_angles_exact_domain(shape, factors, mode, strengthen):
    rng = np.random.default_rng(741)
    a = rng.uniform(-0.25, 1.25, shape).astype(np.float32)
    b = rng.uniform(-0.25, 1.25, shape).astype(np.float32)
    if mode == "horizon":
        values = np.array([0, 0.5, 1, np.nextafter(np.float32(1), np.float32(0))])
        a[:] = np.resize(values, shape)
        b[:] = np.resize(values[::-1], shape)
    elif mode == "exceptional":
        values = np.array([0.0, -0.0, np.nan, np.inf, -np.inf, 0.5, 1], np.float32)
        a[:] = np.resize(values, shape)
        b[:] = np.resize(values[::-1], shape)
    elif mode == "strided":
        a = a[::-1, ::-1]
        b = b[::-1, ::-1]
        a.flags.writeable = b.flags.writeable = False
    before_a, before_b = a.copy(), b.copy()
    with np.errstate(all="ignore"):
        if strengthen:
            expected = original.strengthen_normals(
                original.AdditionMethod.ANGLES, a, factors[0]
            )
            actual = addition.strengthen_normals(
                addition.AdditionMethod.ANGLES, a, factors[0]
            )
        else:
            expected = original.add_normals(
                original.AdditionMethod.ANGLES, a, b, *factors
            )
            actual = addition.add_normals(
                addition.AdditionMethod.ANGLES, a, b, *factors
            )
    exact(actual, expected)
    exact((a, b), (before_a, before_b))
    assert all(
        not np.may_share_memory(c, a) and not np.may_share_memory(c, b) for c in actual
    )


def test_angles_do_not_invoke_numpy_algorithms(monkeypatch):
    a = np.random.default_rng(519).random((19, 257, 4), dtype=np.float32)
    expected_add = original.add_normals(original.AdditionMethod.ANGLES, a, a, 1.1, 0.7)
    expected_scale = original.strengthen_normals(original.AdditionMethod.ANGLES, a, 1.3)

    def reject(*args, **kwargs):
        raise AssertionError("Retained NumPy pixel algorithm was called")

    with monkeypatch.context() as m:
        for name in ("sin", "arcsin", "sqrt", "square", "clip", "maximum", "minimum"):
            m.setattr(np, name, reject)
        actual_add = addition.add_normals(
            addition.AdditionMethod.ANGLES, a, a, 1.1, 0.7
        )
        actual_scale = addition.strengthen_normals(
            addition.AdditionMethod.ANGLES, a, 1.3
        )
    exact(actual_add, expected_add)
    exact(actual_scale, expected_scale)


def test_angles_reentrant_parallel_blocks():
    def work(seed):
        rng = np.random.default_rng(seed)
        a = rng.random((259, 513, 4), dtype=np.float32)
        b = rng.random((259, 513, 3), dtype=np.float32)
        expected = original.add_normals(
            original.AdditionMethod.ANGLES, a, b, 0.91, 1.27
        )
        for _ in range(2):
            exact(
                addition.add_normals(addition.AdditionMethod.ANGLES, a, b, 0.91, 1.27),
                expected,
            )

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(work, range(4)))


@pytest.mark.parametrize(
    "disabled",
    [
        "FMA3",
        "AVX512F",
        "AVX512F,AVX2,FMA3",
        ",".join(
            ["AVX2", "FMA3", *(t for t in __cpu_dispatch__ if t.startswith("AVX512"))]
        ),
    ],
)
def test_numpy_cpu_dispatch_in_fresh_process(disabled):
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["NPY_DISABLE_CPU_FEATURES"] = disabled
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root / "backend" / "src"), str(root / "native" / "tests")]
    )
    script = """
import numpy as np
import reference_normals_addition as reference
from nodes.impl.normals import addition
rng = np.random.default_rng(182)
for shape in [(1,1,3), (17,35,4), (131,517,3)]:
    a = rng.random(shape, dtype=np.float32)
    b = rng.random(shape, dtype=np.float32)
    expected = reference.add_normals(reference.AdditionMethod.ANGLES, a, b, .27, 1.37)
    actual = addition.add_normals(addition.AdditionMethod.ANGLES, a, b, .27, 1.37)
    for x, y in zip(actual, expected):
        np.testing.assert_array_equal(x.view(np.uint32), y.view(np.uint32))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def load_normal_node(name, reference=False):
    root = Path(__file__).resolve().parents[2]
    path = (
        root / "native/tests/reference_normal_nodes"
        if reference
        else root
        / "backend/src/packages/chaiNNer_standard/material_textures/normal_map"
    ) / (name + ".py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    definitions = [
        item for item in tree.body if isinstance(item, (ast.FunctionDef, ast.ClassDef))
    ]
    for item in definitions:
        item.decorator_list = []
    tree.body = definitions
    helper = original_util if reference else util
    namespace: dict[str, Any] = {
        "np": np,
        "Enum": Enum,
        "NormalMapType": NormalMapType,
        "native_geometry": native_geometry,
        **{
            k: getattr(helper, k)
            for k in (
                "XYZ",
                "normalize_normals",
                "gr_to_xyz",
                "xyz_to_bgr",
                "octahedral_gr_to_xyz",
                "xyz_to_octahedral_bgr",
            )
        },
    }
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


NODE_NAMES = ("balance_normals", "normalize_normals", "convert_normals")
NODES = {name: load_normal_node(name) for name in NODE_NAMES}
ORIGINAL_NODES = {name: load_normal_node(name, True) for name in NODE_NAMES}
CASES = [("balance_normals", ())]
CASES += [("normalize_normals", (b,)) for b in range(5)]
CASES += [("convert_normals", (a, b)) for a in NormalMapType for b in NormalMapType]


@pytest.mark.parametrize("name,options", CASES)
@pytest.mark.parametrize("shape", [(1, 1, 3), (3, 17, 4), (1, 8191, 3), (17, 513, 4)])
@pytest.mark.parametrize(
    "pattern", ["random", "constant", "exceptional", "strided", "fortran"]
)
def test_complete_normal_nodes(name, options, shape, pattern):
    image = np.random.default_rng(184).uniform(-0.1, 1.1, shape).astype(np.float32)
    if pattern == "constant":
        image[:] = [0, 0.5, 0.5] if shape[-1] == 3 else [0, 0.5, 0.5, 0.1]
    elif pattern == "exceptional":
        image[:] = np.resize(
            np.array([0, -0.0, 0.5, 1, np.inf, -np.inf, np.nan], np.float32), shape
        )
    elif pattern == "strided":
        image = image[::-1, ::-1]
        image.flags.writeable = False
    elif pattern == "fortran":
        image = np.asfortranarray(image)
    before = image.copy()
    actual_options = expected_options = options
    if name == "normalize_normals":
        actual_options = (NODES[name]["BChannel"](options[0]),)
        expected_options = (ORIGINAL_NODES[name]["BChannel"](options[0]),)
    with np.errstate(all="ignore"):
        expected = ORIGINAL_NODES[name][name + "_node"](image, *expected_options)
        actual = NODES[name][name + "_node"](image, *actual_options)
    exact((actual, image), (expected, before))
    assert not np.may_share_memory(actual, image)


def test_balance_normals_above_2_24_pixels():
    # np.mean divides by an np.intp count in float64; 4097**2 pixels is odd and
    # above 2**24, where float32 cannot hold the count.
    image = np.random.default_rng(185).random((4097, 4097, 3), dtype=np.float32)
    with np.errstate(all="ignore"):
        expected = ORIGINAL_NODES["balance_normals"]["balance_normals_node"](image)
        actual = NODES["balance_normals"]["balance_normals_node"](image)
    exact((actual,), (expected,))


def test_complete_normal_nodes_no_numpy_pixel_work(monkeypatch):
    image = np.random.default_rng(35).random((513, 263, 4), dtype=np.float32)
    expected = ORIGINAL_NODES["balance_normals"]["balance_normals_node"](image)

    def reject(*args, **kwargs):
        raise AssertionError("NumPy pixel algorithm was called")

    with monkeypatch.context() as m:
        for name in ("mean", "negative", "dstack", "sqrt", "square"):
            m.setattr(np, name, reject)
        actual = NODES["balance_normals"]["balance_normals_node"](image)
        NODES["normalize_normals"]["normalize_normals_node"](
            image, NODES["normalize_normals"]["BChannel"](1)
        )
        NODES["convert_normals"]["convert_normals_node"](
            image, NormalMapType.OPENGL, NormalMapType.OCTAHEDRAL
        )
    exact((actual,), (expected,))


def test_normal_map_checked_abi_before_writes():
    api = native_geometry._api()  # raw invalid ABI checks
    image = np.ones((2, 3, 4), np.float32)
    out = np.full((2, 3, 3), 73, np.float32)
    before = out.copy()
    maximum = ctypes.c_size_t(-1).value
    assert api.cn_normal_map(ptr(image), ptr(out), maximum, 4, 0, 0, 0) == 2
    assert api.cn_normal_map(ptr(image), ptr(out), 6, 2, 0, 0, 0) == 1
    assert api.cn_normal_map(ptr(image), ptr(out), 6, 4, 0, 5, 0) == 1
    assert api.cn_normal_map(ptr(image), ptr(out), 6, 4, 2, 0, 3) == 1
    assert api.cn_normal_map(ptr(image), ptr(image), 6, 4, 0, 0, 0) == 1
    assert api.cn_normal_map(None, ptr(out), 6, 4, 0, 0, 0) == 1
    unaligned = ctypes.cast(image.ctypes.data + 1, ctypes.POINTER(ctypes.c_float))
    assert api.cn_normal_map(unaligned, ptr(out), 6, 4, 0, 0, 0) == 1
    assert api.cn_normal_map(None, None, 0, 4, 0, 0, 0) == 0
    exact((out,), (before,))
