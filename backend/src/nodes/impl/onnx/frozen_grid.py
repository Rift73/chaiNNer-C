"""DySample's sampling grid frozen at one size in an ONNX file.

DySample divides its sampling coordinates by the feature map's size through
`torch.tensor([W, H])`, which a trace stores as a constant: the size of the export's
example. Every other size then samples the wrong part of the image, with no error: a 4x
RealPLKSR with DySample, traced at 32x32, reaches 23 dB against PyTorch at 48x48 and 22 dB
at 64x64. Convert To ONNX no longer writes that (pytorch/dysample_export.py), but files
converted before, by it or by other tools, keep the constant.

A constant is wrong only where it stands for a size that changes: the coordinates are a
mesh of pixel positions (a network's absolute coordinates involve no size), and the image
they sample changes size with the input (one resized to a working size first keeps it).
Where shape inference cannot tell the image's size, as behind a padding it does not
follow, the file counts as frozen: a false alarm there is rarer than a wrong file.

The constant is the size of DySample's feature map, which a pixel unshuffle in front of
it makes smaller than the input. Shape inference maps it back to the input's size that
the grid is right at; where it cannot, no size is named."""

from __future__ import annotations

import math
from dataclasses import dataclass

import onnx
from onnx import numpy_helper
from sanic.log import logger

# What a trace's sampling grid goes through from the division to GridSample: minus 1, the
# pixel shuffle and its reshapes, and a Cast where a converter changed the precision.
_GRID_STEPS = frozenset({"Cast", "DepthToSpace", "Reshape", "Sub", "Transpose"})
# Well over the 6 of a trace; a damaged graph can feed itself.
_MOST_GRID_STEPS = 16
# Where the search for a mesh ends: a network's outputs, and shapes lent by other tensors
_NOT_MESH = frozenset({"Conv", "ConvTranspose", "Gemm", "MatMul", "Shape"})
# Input heights and widths for shape inference, multiples of what models commonly need
_INFERENCE_SIZES = ((240, 360), (480, 720))


@dataclass(frozen=True)
class FrozenGrid:
    """A DySample's sampling grid frozen at one size."""

    # The input's (width, height) the grid is right at; None where it cannot be told
    right_at: tuple[int, int] | None


def frozen_grid_size(model: onnx.ModelProto) -> tuple[int, int] | None:
    """The input's (width, height) that `model`'s frozen grid is right at; None without
    such a grid, or where the size cannot be told."""
    grid = find_frozen_grid(model)
    return None if grid is None else grid.right_at


def find_frozen_grid(model: onnx.ModelProto) -> FrozenGrid | None:
    """A DySample of `model` with its sampling grid frozen, if any. An input of one size
    has the size such a grid is right at."""
    graph = model.graph
    samplers = [node for node in graph.node if node.op_type == "GridSample"]
    if not samplers or _takes_one_size(graph):
        return None
    producers = {output: node for node in graph.node for output in node.output}
    tensors = {tensor.name: tensor for tensor in graph.initializer}
    for sampler in samplers:
        division = _grid_division(sampler.input[1], producers)
        if division is None or not _holds_mesh(division.input[0], producers):
            continue
        size = _constant_size(division.input[1], producers, tensors)
        if size is None:
            continue
        image = sampler.input[0]
        first, second = (_inferred_size(model, image, s) for s in _INFERENCE_SIZES)
        if first is None or second is None:
            return FrozenGrid(None)
        if first != second:
            return FrozenGrid(_input_size_at(model, image, size, first, second))
    return None


def _image_input(graph: onnx.GraphProto) -> onnx.ValueInfoProto | None:
    tensors = {tensor.name for tensor in graph.initializer}
    return next((i for i in graph.input if i.name not in tensors), None)


def _takes_one_size(graph: onnx.GraphProto) -> bool:
    image = _image_input(graph)
    if image is None:
        return False
    dims = image.type.tensor_type.shape.dim
    return len(dims) == 4 and all(dim.HasField("dim_value") for dim in dims[2:])


def _grid_division(
    grid: str, producers: dict[str, onnx.NodeProto]
) -> onnx.NodeProto | None:
    """The division that `grid` comes from through a pixel shuffle, which only DySample's
    grid goes through."""
    shuffled = False
    name = grid
    for _ in range(_MOST_GRID_STEPS):
        node = producers.get(name)
        if node is None:
            return None
        if node.op_type == "Div":
            return node if shuffled else None
        if node.op_type not in _GRID_STEPS:
            return None
        shuffled = shuffled or node.op_type == "DepthToSpace"
        name = node.input[0]
    return None


def _holds_mesh(coordinates: str, producers: dict[str, onnx.NodeProto]) -> bool:
    """Whether `coordinates` hold a mesh of pixel positions, which a trace computes with
    a Range of the feature map's size."""
    seen: set[str] = set()
    names = [coordinates]
    while names:
        name = names.pop()
        node = producers.get(name)
        if node is None or name in seen:
            continue
        seen.add(name)
        if node.op_type == "Range":
            return True
        if node.op_type not in _NOT_MESH:
            names.extend(node.input)
    return False


def _constant_size(
    name: str,
    producers: dict[str, onnx.NodeProto],
    tensors: dict[str, onnx.TensorProto],
) -> tuple[int, int] | None:
    """The width and height that the tensor `name` holds, if it is a constant of two."""
    tensor = tensors.get(name)
    node = producers.get(name)
    if node is not None and node.op_type == "Constant":
        tensor = next((a.t for a in node.attribute if a.name == "value"), None)
    if tensor is None or math.prod(tensor.dims) != 2:
        return None
    width, height = numpy_helper.to_array(tensor).ravel().tolist()
    if width < 1 or height < 1 or width % 1 or height % 1:
        return None
    return int(width), int(height)


def _input_size_at(
    model: onnx.ModelProto,
    image: str,
    size: tuple[int, int],
    first: tuple[int, int],
    second: tuple[int, int],
) -> tuple[int, int] | None:
    """The input's (width, height) at which `image`, which has the (height, width)
    `first` and `second` at _INFERENCE_SIZES, has the (width, height) `size`; None if
    its two sizes do not lead to one that shape inference confirms."""
    (low_height, low_width), (high_height, high_width) = _INFERENCE_SIZES
    per_height = (second[0] - first[0]) / (high_height - low_height)
    per_width = (second[1] - first[1]) / (high_width - low_width)
    if not per_height or not per_width:
        return None
    width, height = size
    input_height, input_width = round(height / per_height), round(width / per_width)
    if _inferred_size(model, image, (input_height, input_width)) != (height, width):
        return None
    return input_width, input_height


def _inferred_size(
    model: onnx.ModelProto, tensor: str, input_size: tuple[int, int]
) -> tuple[int, int] | None:
    """The (height, width) of `tensor` at an input of (height, width) `input_size`; None
    where shape inference cannot tell."""
    concrete = onnx.ModelProto()
    concrete.CopyFrom(model)
    image = _image_input(concrete.graph)
    dims = None if image is None else image.type.tensor_type.shape.dim
    if dims is None or len(dims) != 4:
        return None
    for index, value in ((0, 1), (2, input_size[0]), (3, input_size[1])):
        if not dims[index].HasField("dim_value"):
            dims[index].dim_value = value
    try:
        inferred = onnx.shape_inference.infer_shapes(
            concrete, strict_mode=False, data_prop=True
        ).graph
    except (onnx.shape_inference.InferenceError, ValueError):
        logger.warning(
            "Shape inference failed for DySample's sampling grid.", exc_info=True
        )
        return None
    for info in (*inferred.value_info, *inferred.input):
        if info.name == tensor:
            shape = info.type.tensor_type.shape.dim
            if len(shape) == 4 and all(d.HasField("dim_value") for d in shape[2:]):
                return shape[2].dim_value, shape[3].dim_value
    return None
