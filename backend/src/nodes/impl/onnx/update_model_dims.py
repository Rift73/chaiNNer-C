# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Include in chaiNNer with the following changes:
# - Remove `dim_proto.dim_value != dim` for constant dimensions
# - Improved type hints
from __future__ import annotations

from typing import Sequence

from onnx import ModelProto

from ..native_graph import graph


def update_inputs_outputs_dims(
    model: ModelProto,
    input_dims: dict[str, Sequence[str | int]],
    output_dims: dict[str, Sequence[str | int]],
) -> ModelProto:
    """This function updates the dimension sizes of the model's inputs and outputs to the values
    provided in input_dims and output_dims. if the dim value provided is negative, a unique dim_param
    will be set for that dimension.

    Example. if we have the following shape for inputs and outputs:

    * shape(input_1) = ('b', 3, 'w', 'h')
    * shape(input_2) = ('b', 4)
    * shape(output)  = ('b', 'd', 5)

    The parameters can be provided as:

    ::

        input_dims = {
            "input_1": ['b', 3, 'w', 'h'],
            "input_2": ['b', 4],
        }
        output_dims = {
            "output": ['b', -1, 5]
        }

    Putting it together:

    ::

        model = onnx.load('model.onnx')
        updated_model = update_inputs_outputs_dims(model, input_dims, output_dims)
        onnx.save(updated_model, 'model.onnx')
    """
    return graph().onnx_update_dims(model, input_dims, output_dims)
