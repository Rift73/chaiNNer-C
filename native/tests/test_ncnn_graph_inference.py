"""Tiny, deterministic CPU NCNN fixtures; no models are downloaded or inferred on GPU."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import pytest
from test_ncnn_graph import NcnnOptimizer, ReferenceOptimizer, port, reference, snapshot


def model_fixture(kind):
    if kind == "identity":
        layers = [
            "Input input 0 1 input",
            "Convolution conv 1 1 input output 0=3 1=1 6=9",
        ]
        binary = port.DTYPE_FP32 + np.eye(3, dtype=np.float32).tobytes()
        channels = 3
    elif kind == "batchnorm_relu":
        layers = [
            "Input input 0 1 input",
            "Convolution conv 1 1 input a 0=3 1=1 5=1 6=9",
            "BatchNorm bn 1 1 a b 0=3 1=0.0",
            "ReLU relu 1 1 b output 0=0.25",
        ]
        binary = (
            port.DTYPE_FP32
            + np.eye(3, dtype=np.float32).tobytes()
            + np.array([0.125, -0.25, 0.5], np.float32).tobytes()
        )
        binary += np.array(
            [[0.5, 1, 2], [0, 0.25, -0.5], [1, 1, 1], [0.125, 0.25, -0.5]], np.float32
        ).tobytes()
        channels = 3
    elif kind == "pixelshuffle":
        layers = [
            "Input input 0 1 input",
            "Convolution conv 1 1 input a 0=4 1=1 6=4",
            "PixelShuffle shuffle 1 1 a output 0=2",
        ]
        binary = port.DTYPE_FP32 + np.array([0.25, 0.5, 0.75, 1], np.float32).tobytes()
        channels = 1
    else:
        layers = [
            "Input input 0 1 input",
            "Convolution conv 1 1 input output 0=1 1=2 6=4",
        ]
        binary = port.DTYPE_FP16 + np.full(4, 0.25, np.float16).tobytes()
        channels = 1
    return (
        f"7767517\n{len(layers)} {len(layers)}\n" + "\n".join(layers) + "\n",
        binary,
        channels,
    )


def load(module, param, binary):
    model = module.NcnnModel()
    stream = io.StringIO(param)
    stream.readline()
    model.node_count, model.blob_count = map(int, stream.readline().split())
    data = io.BytesIO(binary)
    for line in stream:
        op, layer = model.parse_param_layer(line)
        layer.weight_data = model.load_layer_weights(data, op, layer)
        model.layers.append(layer)
    model.bin_length = len(binary)
    return model


def infer(model, image, packed):
    from ncnn import ncnn

    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = 1
    net.opt.use_fp16_packed = packed
    net.opt.use_fp16_storage = packed
    net.opt.use_fp16_arithmetic = False
    assert net.opt.use_vulkan_compute is False
    assert net.load_param_mem(model.write_param()) == 0
    # Match the installed session's CPU path; its memory loader is used only
    # for Vulkan. In this old binding the CPU memory-loader bias is unstable.
    with tempfile.TemporaryDirectory(prefix="chainner-ncnn-cpu-") as directory:
        path = Path(directory) / "fixture.bin"
        path.write_bytes(model.serialize_weights())
        assert net.load_model(str(path)) == 0
    channels = 1 if image.ndim == 2 else image.shape[2]
    pixel = (
        ncnn.Mat.PixelType.PIXEL_GRAY if channels == 1 else ncnn.Mat.PixelType.PIXEL_RGB
    )
    matrix = ncnn.Mat.from_pixels(image, pixel, image.shape[1], image.shape[0])
    matrix.substract_mean_normalize([], [1 / 255.0] * channels)
    extractor = net.create_extractor()
    assert extractor.input("input", matrix) == 0
    status, output = extractor.extract("output")
    assert status == 0
    return np.array(output, copy=True)


@pytest.mark.parametrize("kind", ["identity", "batchnorm_relu", "pixelshuffle", "half"])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("seed", [0, 7, 19])
def test_exact_installed_engine_cpu_output(kind, packed, seed):
    parameter, binary, channels = model_fixture(kind)
    a, b = load(reference, parameter, binary), load(port, parameter, binary)
    ReferenceOptimizer(a).optimize()
    NcnnOptimizer(b).optimize()
    assert snapshot(a) == snapshot(b)
    assert a.write_param() == b.write_param()
    assert a.serialize_weights() == b.serialize_weights()
    image = np.random.default_rng(seed).integers(
        0, 256, (6, 7, channels), dtype=np.uint8
    )
    if channels == 1:
        image = image[..., 0]
    expected = infer(a, image, packed)
    actual = infer(b, image, packed)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(infer(b, image, packed), actual)


@pytest.mark.parametrize("kind", ["identity", "batchnorm_relu", "pixelshuffle", "half"])
@pytest.mark.parametrize("amount", [0.0, 0.27, 1.0])
def test_interpolated_model_exact_cpu_output(kind, amount):
    parameter, binary, channels = model_fixture(kind)
    results = []
    for module in (reference, port):
        a, b = load(module, parameter, binary), load(module, parameter, binary)
        for layer in b.layers:
            for weight in layer.weight_data.values():
                weight.weight = (weight.weight * 0.75).astype(weight.weight.dtype)
        model = a.interpolate(b, amount)
        results.append(model)
    assert snapshot(results[0]) == snapshot(results[1])
    assert results[0].serialize_weights() == results[1].serialize_weights()
    image = np.arange(6 * 7 * channels, dtype=np.uint8).reshape(6, 7, channels)
    if channels == 1:
        image = image[..., 0]
    np.testing.assert_array_equal(
        infer(results[1], image, False), infer(results[0], image, False)
    )
