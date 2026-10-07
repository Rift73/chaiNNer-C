"""Parameterized converter operators with populated attributes and weights."""

import copy
import itertools
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from onnx import helper
from test_onnx_converter import check, current, model, old, snapshot


def shape_cases():
    for axis in [-4, -3, -2, -1, 0, 1, 2, 3, 4]:
        for op in ["Concat", "Softmax", "Flatten", "Split"]:
            yield op + str(axis), op, {"axis": axis}, None
    for axes in [[], [0], [1], [-1], [1, 2], [-4, 2, 3], [5]]:
        for op in [
            "Squeeze",
            "Unsqueeze",
            "ReduceSum",
            "ReduceMean",
            "ReduceProd",
            "ReduceL2",
            "ReduceLogSumExp",
        ]:
            if axes:
                yield op + str(axes), op, {"axes": axes}, None
            else:
                yield op + "empty", op, {}, None
    for perm in itertools.permutations(range(4)):
        yield "Transpose" + str(perm), "Transpose", {"perm": perm}, None
    for rank in range(1, 7):
        shape = [1, 3, 5, 7, 2, 1][:rank]
        yield "Reshape" + str(rank), "Reshape", {"shape": shape}, None
    for shape, mode in itertools.product(
        [[0, 1, 0, 2], [0, 1, 2, 0, 3, 4], [0, 0, 1, 2, 0, 0, 3, 4]],
        ["constant", "edge", "reflect"],
    ):
        yield (
            "Pad" + str(shape) + mode,
            "Pad",
            {"pads": shape, "mode": mode, "value": 0.5},
            None,
        )
    for scale, mode in itertools.product(
        [[1.0, 2.0], [1.0, 2.0, 3.0], [1.0, 1.0, 2.0, 2.0], [1.0, 2.0, 2.0, 2.0]],
        ["nearest", "linear", "cubic"],
    ):
        yield (
            "Upsample" + str(scale) + mode,
            "Upsample",
            {"scales": scale, "mode": mode},
            None,
        )
    for axis, step in itertools.product([[0], [1], [1, 2], [-1]], [None, [1], [2]]):
        attrs = {"starts": [0] * len(axis), "ends": [3] * len(axis), "axes": axis}
        if step is not None:
            attrs["steps"] = step
        yield "Slice" + str(axis) + str(step), "Slice", attrs, None
    for op, attrs in [
        ("Crop", {"starts": [1, 2], "ends": [3, 4], "axes": [2, 3]}),
        ("DepthToSpace", {"blocksize": 2, "mode": "DCR"}),
        ("DepthToSpace", {"blocksize": 2, "mode": "CRD"}),
        ("Dropout", {"ratio": 0.25}),
        ("Elu", {"alpha": 0.5}),
        ("LRN", {"size": 3, "alpha": 0.125, "beta": 0.5, "bias": 1.0}),
        ("LeakyRelu", {"alpha": 0.125}),
        ("ShuffleChannel", {"group": 3, "reverse": 1}),
        ("PixelShuffle", {"scale_factor": 2}),
        ("Reorg", {"stride": 2}),
        ("Normalize", {"eps": 1e-6}),
        ("Clip", {"min": -1.0, "max": 1.0}),
        ("ImageScaler", {"scale": 0.5, "bias": [0.0, 0.1, 0.2]}),
    ]:
        yield op + str(attrs), op, attrs, None


@pytest.mark.parametrize(
    "name,op,attrs,_",
    list(shape_cases()),
    ids=lambda v: v if isinstance(v, str) else None,
)
@pytest.mark.parametrize("fp16", [False, True])
def test_shape_and_attribute_operators(name, op, attrs, _, fp16):
    del name
    check(model([helper.make_node(op, ["x"], ["y"], name="shape", **attrs)]), fp16)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize(
    "op",
    [
        "Gemm",
        "MatMul",
        "PRelu",
        "BiasGelu",
        "SkipLayerNormalization",
        "EmbedLayerNormalization",
        "RNN",
        "GRU",
        "LSTM",
    ],
)
@pytest.mark.parametrize("fp16", [False, True])
def test_populated_weight_operators(dtype, op, fp16):
    weights = {
        "w": np.linspace(-1, 1, 9, dtype=dtype).reshape(3, 3),
        "b": np.linspace(-1, 1, 3, dtype=dtype),
    }
    attrs: dict[str, Any] = {}
    if op == "Gemm":
        inputs = ["x", "w", "b"]
        attrs = {"transB": 1}
    elif op == "MatMul":
        inputs = ["x", "w"]
    elif op in {"PRelu", "BiasGelu"}:
        inputs = ["x", "b"]
    elif op == "SkipLayerNormalization":
        inputs = ["x", "z", "b", "b", "b"]
    elif op == "EmbedLayerNormalization":
        inputs = ["x", "z", "w", "w", "b", "b", "b"]
    else:
        weights = {
            "w": np.ones((1, 3, 3), dtype),
            "r": np.ones((1, 3, 3), dtype),
            "b": np.linspace(-1, 1, 6, dtype=dtype).reshape(1, 6),
        }
        inputs = ["x", "w", "r", "b"]
        attrs = {"hidden_size": 3, "direction": "forward"}
    check(
        model(
            [helper.make_node(op, inputs, ["y"], name="weighted", **attrs)],
            weights,
            inputs=("x", "z"),
        ),
        fp16,
    )


@pytest.mark.parametrize("scale", [[1, 2], [1, 2, 3], [1, 1, 2, 2], [1, 2, 2, 2], []])
@pytest.mark.parametrize("sizes", [[], [1, 3, 10, 14]])
@pytest.mark.parametrize("mode", ["nearest", "linear", "cubic"])
@pytest.mark.parametrize("version", [10, 11])
def test_resize_tensor_inputs(scale, sizes, mode, version):
    weights: dict[str, np.ndarray] = {"scales": np.asarray(scale, np.float32)}
    inputs = ["x", "scales"]
    if version == 11:
        weights.update(roi=np.empty(0, np.float32), sizes=np.asarray(sizes, np.int64))
        inputs = ["x", "roi", "scales", "sizes"]
    check(
        model(
            [
                helper.make_node(
                    "Resize",
                    inputs,
                    ["y"],
                    name="resize",
                    mode=mode,
                    coordinate_transformation_mode="align_corners",
                )
            ],
            weights,
        )
    )


def infer(converted, image):
    from ncnn import ncnn

    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = 1
    assert net.load_param_mem(converted.write_param()) == 0
    with tempfile.TemporaryDirectory(prefix="chainner-converter-cpu-") as directory:
        path = Path(directory) / "model.bin"
        path.write_bytes(converted.serialize_weights())
        assert net.load_model(str(path)) == 0
    matrix = ncnn.Mat.from_pixels(
        image, ncnn.Mat.PixelType.PIXEL_RGB, image.shape[1], image.shape[0]
    )
    matrix.substract_mean_normalize([], [1 / 255.0] * 3)
    extractor = net.create_extractor()
    assert extractor.input("x", matrix) == 0
    code, output = extractor.extract("y")
    assert code == 0
    return np.array(output, copy=True)


@pytest.mark.parametrize("fp16", [False, True])
@pytest.mark.parametrize("activation", ["Relu", "LeakyRelu", "Sigmoid", "Tanh"])
@pytest.mark.parametrize("seed", [0, 7, 23])
def test_converted_graph_real_cpu_inference(fp16, activation, seed):
    # An even coefficient count matches this installed exporter/engine pair's
    # half-storage alignment contract; odd half arrays are covered below as
    # the original exporter's rejected-model behavior.
    weights = {
        "w": np.eye(4, 3, dtype=np.float32).reshape(4, 3, 1, 1),
        "b": np.array([0.125, -0.25, 0.5, 0.0], np.float32),
    }
    source = model(
        [
            helper.make_node(
                "Conv", ["x", "w", "b"], ["v"], name="conv", kernel_shape=[1, 1]
            ),
            helper.make_node(activation, ["v"], ["y"], name="activation"),
        ],
        weights,
    )
    a = old.Onnx2NcnnConverter(copy.deepcopy(source)).convert(fp16)
    b = current.Onnx2NcnnConverter(copy.deepcopy(source)).convert(fp16)
    assert snapshot(a) == snapshot(b)
    image = np.random.default_rng(seed).integers(0, 256, (5, 7, 3), dtype=np.uint8)
    expected = infer(a, image)
    actual = infer(b, image)
    assert actual.tobytes() == expected.tobytes()
    assert infer(b, image).tobytes() == actual.tobytes()


def test_original_odd_half_export_rejection():
    from ncnn import ncnn

    source = model(
        [
            helper.make_node(
                "Conv", ["x", "w", "b"], ["y"], name="conv", kernel_shape=[1, 1]
            )
        ],
        {
            "w": np.eye(3, dtype=np.float32).reshape(3, 3, 1, 1),
            "b": np.zeros(3, np.float32),
        },
    )
    a = old.Onnx2NcnnConverter(copy.deepcopy(source)).convert(True)
    b = current.Onnx2NcnnConverter(copy.deepcopy(source)).convert(True)
    assert a.write_param() == b.write_param()
    assert a.serialize_weights() == b.serialize_weights()
    for converted in [a, b]:
        net = ncnn.Net()
        net.opt.use_vulkan_compute = False
        assert net.load_param_mem(converted.write_param()) == 0
        with tempfile.TemporaryDirectory(
            prefix="chainner-half-rejection-"
        ) as directory:
            path = Path(directory) / "model.bin"
            path.write_bytes(converted.serialize_weights())
            assert net.load_model(str(path)) == -1
