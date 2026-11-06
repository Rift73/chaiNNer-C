"""ONNX files whose DySample samples wrongly: a trace stored the sampling grid's divisor
as a constant, the size of the export's example, and every other size then samples the
wrong part of the image with no error. Load Model, Load Models and Build Engine warn."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast
from unittest import mock

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper
from spandrel.architectures.__arch_helpers.dysample import DySample
from spandrel.architectures.MoESR import MoESR
from spandrel.architectures.PLKSR import RealPLKSR
from torch import nn
from torch.nn import functional

from api import NodeContext, SettingsParser
from nodes.impl.onnx.frozen_grid import (
    FrozenGrid,
    find_frozen_grid,
    frozen_grid_size,
)
from nodes.impl.onnx.load import load_onnx_model, model_facts, warn_if_grid_frozen
from packages.chaiNNer_onnx.onnx import io_group
from packages.chaiNNer_onnx.onnx.batch_processing.load_models import load_models_node
from packages.chaiNNer_tensorrt.settings import TensorRTSettings
from packages.chaiNNer_tensorrt.tensorrt.utility import build_engine

# Runs native code, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

DIVISOR_DIMS = [1, 2, 1, 1, 1]
LOAD_MODEL = next(
    node.run for node in io_group.nodes if node.schema_id == "chainner:onnx:load_model"
)


class Warnings:
    """The NodeContext member these nodes use."""

    def __init__(self) -> None:
        self.messages: list[str] = []
        # Load Model starts its size probe for other providers (size_probe.py).
        self.settings = SettingsParser({"execution_provider": "CPUExecutionProvider"})

    def warn(self, message: str) -> None:
        self.messages.append(message)

    def context(self) -> NodeContext:
        return cast(NodeContext, self)


def dysample_model() -> torch.nn.Module:
    torch.manual_seed(0)
    model = RealPLKSR(
        dim=16, n_blocks=2, upscaling_factor=2, kernel_size=7, dysample=True
    )
    return model.eval().requires_grad_(False)


class WorkingSize(nn.Module):
    """Every input resized to one working size, DySample there, the result resized to
    the input's size: the divisor is that working size at every input size."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.up = DySample(16, 3, scale=2, groups=4, end_convolution=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        small = functional.interpolate(x, size=(24, 24), mode="bilinear")
        return functional.interpolate(
            self.up(self.conv(small)), size=size, mode="bilinear"
        )


class AbsoluteCoordinates(nn.Module):
    """A network that predicts coordinates in [0, 4), which a constant 2 maps to [-1, 1):
    no size is involved."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.register_buffer("two", torch.tensor([2.0, 2.0]).view(1, 2, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        coordinates = self.conv(x).sigmoid() * 4
        coordinates = coordinates.view(batch, 2, -1, height, width) / self.get_buffer(
            "two"
        )
        grid = (
            functional.pixel_shuffle(
                (coordinates - 1).reshape(batch, -1, height, width), 2
            )
            .view(batch, 2, 2 * height, 2 * width)
            .permute(0, 2, 3, 1)
        )
        return functional.grid_sample(
            x, grid, padding_mode="border", align_corners=False
        )


class Unshuffled(nn.Module):
    """A pixel unshuffle in front of DySample: its feature map has half the input's size."""

    def __init__(self) -> None:
        super().__init__()
        self.down = nn.PixelUnshuffle(2)
        self.conv = nn.Conv2d(12, 16, 3, padding=1)
        self.up = DySample(16, 3, scale=4, groups=4, end_convolution=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.conv(self.down(x)))


def trace(path: Path, size: int, model: torch.nn.Module | None = None) -> bytes:
    """The model's trace at `size`, dynamic in height and width, with DySample as it is:
    what Convert To ONNX wrote before its grid followed the input's size."""
    # Spandrel before 0.4.2+c14 read DySample's size as `torch.tensor([W, H])`, which a
    # trace stores as a constant: the files these tests are about.
    with mock.patch.object(
        torch, "_shape_as_tensor", lambda tensor: torch.tensor(tensor.shape)
    ):
        torch.onnx.export(
            dysample_model() if model is None else model.eval().requires_grad_(False),
            (torch.rand(1, 3, size, size),),
            str(path),
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size", 2: "height", 3: "width"},
                "output": {0: "batch_size", 2: "height", 3: "width"},
            },
            do_constant_folding=True,
            dynamo=False,
        )
    return path.read_bytes()


@pytest.fixture(scope="module")
def traced(tmp_path_factory: pytest.TempPathFactory) -> bytes:
    return trace(tmp_path_factory.mktemp("traced") / "traced.onnx", 32)


def size_following_divisor() -> list[onnx.NodeProto]:
    """The width and height from the graph's own shape."""
    return [
        helper.make_node("Shape", ["input"], ["shape_of_input"]),
        helper.make_node("Gather", ["shape_of_input", "width_height"], ["sizes"]),
        helper.make_node("Cast", ["sizes"], ["float_sizes"], to=TensorProto.FLOAT),
        helper.make_node("Reshape", ["float_sizes", "divisor_dims"], ["divisor"]),
    ]


def constant_divisor(
    data_type: int = TensorProto.FLOAT,
    values: tuple[float, ...] = (32.0, 32.0),
    dims: list[int] | None = None,
) -> list[onnx.NodeProto]:
    tensor = helper.make_tensor(
        "value", data_type, DIVISOR_DIMS if dims is None else dims, values
    )
    return [helper.make_node("Constant", [], ["divisor"], value=tensor)]


def grid_model(
    divisor: list[onnx.NodeProto],
    *,
    initializer: onnx.TensorProto | None = None,
    height: int | str = "height",
    width: int | str = "width",
    shuffled: bool = True,
    cast_grid: bool = False,
    mesh: bool = True,
    image: str = "input",
) -> onnx.ModelProto:
    """DySample's path from the division to GridSample in a trace, on the input itself,
    with a mesh of the positions along its width in the coordinates."""
    grid = "cast_grid" if cast_grid else "grid"
    nodes = [
        *divisor,
        helper.make_node("Shape", ["input"], ["mesh_shape"]),
        helper.make_node("Gather", ["mesh_shape", "width_axis"], ["mesh_width"]),
        helper.make_node("Cast", ["mesh_width"], ["mesh_limit"], to=TensorProto.FLOAT),
        helper.make_node("Range", ["zero", "mesh_limit", "one"], ["mesh"]),
        helper.make_node("Add", ["input", "mesh"], ["with_mesh"]),
        helper.make_node(
            "Div", ["with_mesh" if mesh else "input", "divisor"], ["divided"]
        ),
        helper.make_node("Sub", ["divided", "one"], ["centred"]),
        helper.make_node("Reshape", ["centred", "any_shape"], ["channels"]),
        helper.make_node("DepthToSpace", ["channels"], ["shuffled"], blocksize=2)
        if shuffled
        else helper.make_node("Identity", ["channels"], ["shuffled"]),
        helper.make_node("Reshape", ["shuffled", "any_shape"], ["viewed"]),
        helper.make_node("Transpose", ["viewed"], ["last"], perm=[0, 2, 3, 1]),
        helper.make_node("Reshape", ["last", "any_shape"], ["grid"]),
        *(
            [helper.make_node("Cast", ["grid"], [grid], to=TensorProto.FLOAT)]
            if cast_grid
            else []
        ),
        helper.make_node("Unknown", ["input"], ["unknown"], domain="custom.domain"),
        helper.make_node("GridSample", [image, grid], ["output"]),
    ]
    constants = [
        numpy_helper.from_array(np.array(0.0, np.float32), "zero"),
        numpy_helper.from_array(np.array(1.0, np.float32), "one"),
        numpy_helper.from_array(np.array(3, np.int64), "width_axis"),
        numpy_helper.from_array(np.array([0, -1, 2], np.int64), "any_shape"),
        numpy_helper.from_array(np.array([3, 2], np.int64), "width_height"),
        numpy_helper.from_array(np.array(DIVISOR_DIMS, np.int64), "divisor_dims"),
    ]
    if initializer is not None:
        constants.append(initializer)
    graph = helper.make_graph(
        nodes,
        "grid",
        [
            helper.make_tensor_value_info(
                "input", TensorProto.FLOAT, [1, 3, height, width]
            )
        ],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, None)],
        constants,
    )
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("custom.domain", 1),
        ],
    )


def clean_model() -> onnx.ModelProto:
    """A 1x model with no sampling grid."""
    graph = helper.make_graph(
        [helper.make_node("Mul", ["x", "w"], ["y"])],
        "gain",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, "h", "w"])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, "h", "w"])],
        [numpy_helper.from_array(np.full((1, 3, 1, 1), 2.0, np.float32), "w")],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8
    )


@pytest.mark.parametrize("size", [3, 32, 48])
def test_a_trace_is_found_at_its_size(tmp_path: Path, size: int):
    data = trace(tmp_path / "traced.onnx", size)
    assert frozen_grid_size(onnx.load_model_from_string(data)) == (size, size)


def test_what_is_found_really_is_wrong_at_every_other_size(traced: bytes):
    session = ort.InferenceSession(traced, providers=["CPUExecutionProvider"])
    largest_difference = {}
    for size in (32, 48):
        image = torch.rand(1, 3, size, size)
        expected = dysample_model()(image).numpy()
        found = session.run(None, {"input": image.numpy()})[0]
        largest_difference[size] = float(np.abs(expected - found).max())
    assert largest_difference[32] < 1e-4
    assert largest_difference[48] > 1e-2


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(grid_model(constant_divisor()), id="Constant"),
        pytest.param(
            grid_model(
                [],
                initializer=numpy_helper.from_array(
                    np.array([32, 32], np.float32).reshape(DIVISOR_DIMS), "divisor"
                ),
            ),
            id="initializer, as Optimize Model leaves it",
        ),
        pytest.param(grid_model(constant_divisor(TensorProto.FLOAT16)), id="FP16"),
        pytest.param(grid_model(constant_divisor(TensorProto.BFLOAT16)), id="BF16"),
        pytest.param(grid_model(constant_divisor(), cast_grid=True), id="cast grid"),
    ],
)
def test_a_constant_divisor_is_found(model: onnx.ModelProto):
    assert frozen_grid_size(model) == (32, 32)


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(
            grid_model(constant_divisor(), height=256), id="only the width follows"
        ),
        pytest.param(
            grid_model(constant_divisor(), image="unknown"),
            id="shape inference does not reach the image",
        ),
    ],
)
def test_a_grid_of_no_size_to_name_is_found_all_the_same(model: onnx.ModelProto):
    assert find_frozen_grid(model) == FrozenGrid(right_at=None)
    assert frozen_grid_size(model) is None


def warning_of(data: bytes) -> str:
    warnings = Warnings()
    warn_if_grid_frozen(warnings.context(), load_onnx_model(data), "This model")
    (message,) = warnings.messages
    return message


def test_the_warning_names_the_input_size_it_is_right_at(tmp_path: Path):
    torch.manual_seed(0)
    model = Unshuffled().eval().requires_grad_(False)
    data = trace(tmp_path / "unshuffled.onnx", 64, model)
    session = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
    largest_difference = {}
    for size in (64, 32):
        image = torch.rand(1, 3, size, size)
        found = session.run(None, {"input": image.numpy()})[0]
        largest_difference[size] = np.abs(model(image).numpy() - found).max()
    assert largest_difference[64] < 1e-4
    assert largest_difference[32] > 1e-2
    assert frozen_grid_size(onnx.load_model_from_string(data)) == (64, 64)
    message = warning_of(data)
    assert "64x64" in message
    assert "32x32" not in message


def test_a_model_shape_inference_does_not_follow_is_warned_without_a_number(
    tmp_path: Path,
):
    # MoESR pads to a multiple through a computed Pad, which shape inference leaves open.
    torch.manual_seed(0)
    model = MoESR(
        dim=16, scale=2, n_blocks=1, n_block=1, upsampler="dysample", upsample_dim=16
    )
    data = trace(tmp_path / "padded.onnx", 32, model)
    assert find_frozen_grid(onnx.load_model_from_string(data)) == FrozenGrid(None)
    message = warning_of(data)
    assert "the one it was converted at" in message
    assert "x32" not in message
    assert "converted again" in message


def test_the_size_is_width_then_height():
    model = grid_model(constant_divisor(values=(48.0, 32.0)))
    assert frozen_grid_size(model) == (48, 32)


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(
            grid_model(size_following_divisor()), id="the divisor follows the input"
        ),
        pytest.param(
            grid_model(constant_divisor(), height=256, width=256),
            id="the input has one size, the one it is right at",
        ),
        pytest.param(
            grid_model(constant_divisor(), shuffled=False),
            id="no pixel shuffle, not DySample",
        ),
        pytest.param(
            grid_model(constant_divisor(), mesh=False),
            id="no mesh of positions, no size involved",
        ),
        pytest.param(
            grid_model(constant_divisor(values=(32.0, 32.0, 32.0), dims=[3])),
            id="not a pair",
        ),
        pytest.param(grid_model(constant_divisor(values=(0.5, 0.5))), id="not a size"),
        pytest.param(clean_model(), id="no GridSample"),
    ],
)
def test_anything_else_is_not_found(model: onnx.ModelProto):
    assert frozen_grid_size(model) is None


@pytest.mark.parametrize("make", [WorkingSize, AbsoluteCoordinates])
def test_a_model_right_at_every_size_is_not_found(
    tmp_path: Path, make: Callable[[], nn.Module]
):
    torch.manual_seed(0)
    model = make().eval().requires_grad_(False)
    data = trace(tmp_path / "right.onnx", 64, model)
    session = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
    for height, width in ((64, 64), (48, 48), (40, 72)):
        image = torch.rand(1, 3, height, width)
        found = session.run(None, {"input": image.numpy()})[0]
        assert np.abs(model(image).numpy() - found).max() < 1e-3
    assert frozen_grid_size(onnx.load_model_from_string(data)) is None


def test_a_graph_that_feeds_itself_ends_the_search():
    model = grid_model(constant_divisor())
    for node in model.graph.node:
        if node.op_type == "Sub":
            node.input[0] = "last"
    assert frozen_grid_size(model) is None


def test_the_facts_keep_the_size(traced: bytes):
    assert model_facts(load_onnx_model(traced)).frozen_grid == FrozenGrid((32, 32))
    clean = clean_model().SerializeToString()
    assert model_facts(load_onnx_model(clean)).frozen_grid is None


def test_load_model_warns_and_still_loads(tmp_path: Path, traced: bytes):
    path = tmp_path / "old_export.onnx"
    path.write_bytes(traced)
    warnings = Warnings()
    model, directory, name = LOAD_MODEL(warnings.context(), path)
    assert model.bytes == traced
    assert (directory, name) == (tmp_path, "old_export")
    (message,) = warnings.messages
    assert "old_export.onnx" in message
    assert "32x32" in message
    assert "converted again" in message


def test_load_model_says_nothing_about_a_model_without_the_constant(tmp_path: Path):
    path = tmp_path / "clean.onnx"
    path.write_bytes(clean_model().SerializeToString())
    warnings = Warnings()
    LOAD_MODEL(warnings.context(), path)
    assert warnings.messages == []


def test_load_models_warns_about_each_old_file_in_its_turn(
    tmp_path: Path, traced: bytes
):
    (tmp_path / "a_old.onnx").write_bytes(traced)
    (tmp_path / "b_clean.onnx").write_bytes(clean_model().SerializeToString())
    (tmp_path / "c_old.onnx").write_bytes(traced)
    warnings = Warnings()
    generator, _ = load_models_node(warnings.context(), tmp_path, False)
    seen = []
    for item in generator.supplier():
        assert not isinstance(item, Exception)
        seen.append((item[2], len(warnings.messages)))
    assert seen == [("a_old", 1), ("b_clean", 1), ("c_old", 2)]
    assert "a_old.onnx" in warnings.messages[0]
    assert "c_old.onnx" in warnings.messages[1]


def run_build_engine(monkeypatch: pytest.MonkeyPatch, data: bytes) -> list[str]:
    """Build Engine's warnings on an ONNX, with its builder stubbed."""
    monkeypatch.setattr(
        build_engine,
        "get_settings",
        lambda _: TensorRTSettings(
            gpu_index=0, timing_cache_path=None, engine_cache_path=None
        ),
    )
    monkeypatch.setattr(build_engine, "get_cuda_compute_capability", lambda _: (12, 0))
    monkeypatch.setattr(build_engine, "get_cuda_device_name", lambda _: "RTX 5090")
    monkeypatch.setattr(
        build_engine, "cached_engine", lambda _key, build, _path: build()
    )
    monkeypatch.setattr(build_engine, "build_engine_from_onnx", lambda *_, **__: None)
    warnings = Warnings()
    build_engine.build_engine_node(
        warnings.context(),
        load_onnx_model(data),
        build_engine.Precision.FP16,
        build_engine.ShapeMode.FIXED,
        64,
        64,
        512,
        512,
        1152,
        1152,
        256,
        256,
        0.0,
        False,
        False,
        3,
        build_engine.TilingOptimization.NONE,
    )
    return warnings.messages


def test_build_engine_warns_and_still_builds(
    monkeypatch: pytest.MonkeyPatch, traced: bytes
):
    (message,) = run_build_engine(monkeypatch, traced)
    assert "engine built from it" in message
    assert "32x32" in message
    assert "converted again" in message


def test_build_engine_says_nothing_about_a_model_without_the_constant(
    monkeypatch: pytest.MonkeyPatch,
):
    data = clean_model().SerializeToString()
    assert run_build_engine(monkeypatch, data) == []
