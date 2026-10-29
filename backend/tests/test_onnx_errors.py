from __future__ import annotations

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

# The node's modules load the native backend, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

from nodes.impl.onnx.auto_split import onnx_auto_split
from nodes.impl.onnx.load import load_onnx_model
from nodes.impl.onnx.model import OnnxGeneric, OnnxInfo
from nodes.impl.onnx.session import create_inference_session
from nodes.impl.rembg.session_simple import SimpleSession
from nodes.impl.upscale.auto_split_tiles import NO_TILING
from nodes.impl.upscale.tiler import NoTiling
from packages.chaiNNer_onnx.onnx.processing.upscale_image import upscale_with_session


def reshape_model() -> OnnxGeneric:
    """Reshapes its input to 1x3x16x16, so ORT fails for any other size with an error
    that names the node. The name holds byte 0x92, which is not UTF-8: it is the
    apostrophe of a French Windows message in its ANSI code page (cp1252)."""
    image = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, "h", "w"])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 16, 16])
    shape = numpy_helper.from_array(np.array([1, 3, 16, 16], np.int64), "shape")
    node = helper.make_node("Reshape", ["input", "shape"], ["output"], name="L_Qop")
    graph = helper.make_graph([node], "reshape", [image], [output], [shape])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8
    ).SerializeToString()
    assert model.count(b"L_Qop") == 1
    return OnnxGeneric(model.replace(b"L_Qop", b"L_\x92op"), OnnxInfo(13, "fp32"))


def test_an_error_message_that_is_not_utf8_reaches_the_user():
    # ORT falls back to its CPU provider for one it does not have, so this runs on the
    # CPU through ORT's binding, the path of the CUDA, DirectML and TensorRT providers.
    session = create_inference_session(reshape_model(), 0, "NoSuchExecutionProvider")
    image = np.random.default_rng(0).random((16, 16, 3), dtype=np.float32)
    np.testing.assert_array_equal(
        onnx_auto_split(image, session, False, NoTiling()), image
    )
    with pytest.raises(
        RuntimeError, match=r"(?s)Name:'L_.+op'.*cannot be reshaped"
    ) as raised:
        onnx_auto_split(np.zeros((20, 20, 3), np.float32), session, False, NoTiling())
    assert isinstance(raised.value.__cause__, UnicodeDecodeError)


def test_a_remove_background_error_that_is_not_utf8_reaches_the_user():
    session = create_inference_session(reshape_model(), 0, "NoSuchExecutionProvider")
    rembg = SimpleSession(session, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (20, 20))
    with pytest.raises(
        RuntimeError, match=r"(?s)Name:'L_.+op'.*cannot be reshaped"
    ) as raised:
        rembg.predict(np.zeros((8, 8, 3), np.float32))
    assert isinstance(raised.value.__cause__, UnicodeDecodeError)


@pytest.mark.parametrize("elem_type", [TensorProto.INT64, TensorProto.UINT8])
def test_a_model_without_a_float_input_is_refused_with_its_input_type(elem_type: int):
    # ORT's own error only says the image's type is not the one expected.
    image = helper.make_tensor_value_info("input", elem_type, [1, 3, "h", "w"])
    output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 3, "h", "w"]
    )
    node = helper.make_node("Cast", ["input"], ["output"], to=TensorProto.FLOAT)
    graph = helper.make_graph([node], "cast", [image], [output])
    model = load_onnx_model(
        helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8
        ).SerializeToString()
    )
    assert isinstance(model, OnnxGeneric)
    session = create_inference_session(model, 0, "CPUExecutionProvider")
    name = helper.tensor_dtype_to_np_dtype(elem_type).name
    with pytest.raises(ValueError) as raised:
        upscale_with_session(
            np.zeros((20, 20, 3), np.float32), model, session, NO_TILING, False
        )
    assert str(raised.value) == (
        f"This ONNX model's input is a tensor({name}), but Upscale Image gives a model"
        " its image as a tensor(float), or a tensor(float16) to an FP16 model. Use a"
        " model that takes an image of floats, such as one from Convert To ONNX."
    )
