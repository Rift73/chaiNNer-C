from __future__ import annotations

from copy import deepcopy

import numpy as np
import onnx
from google.protobuf.internal.containers import RepeatedCompositeFieldContainer
from onnx import numpy_helper as onph
from onnx.onnx_pb import TensorProto
from sanic.log import logger

from api import NodeContext
from nodes.impl.native_analysis import mean as native_mean
from nodes.impl.native_graph import graph
from nodes.impl.native_tensors import cast_numpy, interpolate_numpy
from nodes.impl.onnx.load import load_onnx_model
from nodes.impl.onnx.model import OnnxModel
from nodes.impl.onnx.utils import safely_optimize_onnx_model
from nodes.impl.upscale.auto_split_tiles import NO_TILING
from nodes.properties.inputs import OnnxModelInput, SliderInput
from nodes.properties.outputs import NumberOutput, OnnxModelOutput

from .. import utility_group
from ..processing.upscale_image import upscale_image_node


def perform_interp(
    model_a_weights: RepeatedCompositeFieldContainer,
    model_b_weights: RepeatedCompositeFieldContainer,
    amount: float,
) -> list[TensorProto]:
    return graph().onnx_perform_interp(
        model_a_weights,
        model_b_weights,
        amount,
        onph.to_array,
        onph.from_array,
        interpolate_numpy,
        cast_numpy,
    )


def check_will_upscale(context: NodeContext, model: OnnxModel):
    return graph().onnx_check_will_upscale(
        context,
        model,
        {
            "np": np,
            "native_mean": native_mean,
            "upscale_image_node": upscale_image_node,
            "NO_TILING": NO_TILING,
        },
    )


@utility_group.register(
    schema_id="chainner:onnx:interpolate_models",
    name="Interpolate Models",
    description="Interpolate two ONNX models of the same type together. \
            Note: models must share a common 'pretrained model' ancestor \
            in order to be interpolatable.",
    icon="BsTornado",
    inputs=[
        OnnxModelInput("Model A"),
        OnnxModelInput("Model B"),
        SliderInput(
            "Weights",
            step=5,
            slider_step=1,
            max=100,
            default=50,
            unit="%",
            note_expression="`Model A ${100 - value}% ― Model B ${value}%`",
            ends=("A", "B"),
        ),
    ],
    outputs=[
        OnnxModelOutput(),
        NumberOutput("Amount A", output_type="100 - Input2"),
        NumberOutput("Amount B", output_type="Input2"),
    ],
    node_context=True,
)
def interpolate_models_node(
    context: NodeContext,
    a: OnnxModel,
    b: OnnxModel,
    amount: int,
) -> tuple[OnnxModel, int, int]:
    return graph().onnx_interpolate_models(
        context,
        a,
        b,
        amount,
        {
            "onnx": onnx,
            "safely_optimize_onnx_model": safely_optimize_onnx_model,
            "logger": logger,
            "perform_interp": perform_interp,
            "deepcopy": deepcopy,
            "load_onnx_model": load_onnx_model,
            "check_will_upscale": check_will_upscale,
        },
    )
