from __future__ import annotations

import numpy as np
import onnx.numpy_helper
from google.protobuf.internal.containers import (
    RepeatedCompositeFieldContainer,
    RepeatedScalarFieldContainer,
)

# `x as x` imports: the native mirror reads these names in native/include/onnx_converter_passes.hpp
from onnx.onnx_pb import AttributeProto as AttributeProto
from onnx.onnx_pb import ModelProto
from onnx.onnx_pb import NodeProto as NodeProto
from onnx.onnx_pb import TensorProto as TensorProto
from sanic.log import logger as logger

from ..native_graph import graph
from ..ncnn.model import DTYPE_FP16 as DTYPE_FP16
from ..ncnn.model import DTYPE_FP32 as DTYPE_FP32
from ..ncnn.model import (
    BinaryOpTypes,
    EltwiseOpTypes,
    GruDirectionFlags,
    InterpResizeTypes,
    NormalizeEpsModes,
    PaddingTypes,
    PadModes,
    PermuteOrderTypes,
    ReductionOpTypes,
    UnaryOpTypes,
)
from ..ncnn.model import NcnnLayer as NcnnLayer
from ..ncnn.model import NcnnModel as NcnnModel
from ..ncnn.optimizer import NcnnOptimizer as NcnnOptimizer
from .tensorproto_utils import APT as APT
from .tensorproto_utils import FLOAT32_MAX as FLOAT32_MAX
from .tensorproto_utils import get_node_attr_af as get_node_attr_af
from .tensorproto_utils import get_node_attr_ai as get_node_attr_ai
from .tensorproto_utils import get_node_attr_f as get_node_attr_f
from .tensorproto_utils import (
    get_node_attr_from_input_af as get_node_attr_from_input_af,
)
from .tensorproto_utils import (
    get_node_attr_from_input_ai as get_node_attr_from_input_ai,
)
from .tensorproto_utils import get_node_attr_from_input_f as get_node_attr_from_input_f
from .tensorproto_utils import get_node_attr_i as get_node_attr_i
from .tensorproto_utils import get_node_attr_s as get_node_attr_s
from .tensorproto_utils import get_node_attr_tensor as get_node_attr_tensor
from .tensorproto_utils import get_tensor_proto_data_size as get_tensor_proto_data_size
from .tensorproto_utils import set_node_attr_ai as set_node_attr_ai

UOT = UnaryOpTypes
BOT = BinaryOpTypes
EOT = EltwiseOpTypes
GRU = GruDirectionFlags
IRT = InterpResizeTypes
NEM = NormalizeEpsModes
PAM = PadModes
PAT = PaddingTypes
POT = PermuteOrderTypes
ROT = ReductionOpTypes
onph = onnx.numpy_helper  # read as "onph" by onnx_converter_passes.hpp


class Onnx2NcnnConverter:
    def __init__(self, onnx_model: ModelProto):
        graph().onnx_converter_init(globals(), self, onnx_model)

    @staticmethod
    def add_weight(
        layer: NcnnLayer,
        weight_name: str,
        data: float | int | np.ndarray | TensorProto,
        quantize_tag: bytes = b"",
    ) -> int:
        return graph().onnx_converter_add_weight(
            globals(), layer, weight_name, data, quantize_tag
        )

    @staticmethod
    def clear_container(
        container: RepeatedCompositeFieldContainer | RepeatedScalarFieldContainer,
    ) -> None:
        return graph().onnx_converter_clear_container(globals(), container)

    def swap_nodes(self, a: int, b: int) -> None:
        return graph().onnx_converter_swap_nodes(globals(), self, a, b)

    def fuse_rewrite_gather(self) -> None:
        return graph().onnx_converter_fuse_rewrite_gather(globals(), self)

    def fuse_weight_reshape(self, reduced_node_count: list[int]) -> None:
        return graph().onnx_converter_fuse_weight_reshape(
            globals(), self, reduced_node_count
        )

    def fuse_weight_transpose(self, reduced_node_count: list[int]) -> None:
        return graph().onnx_converter_fuse_weight_transpose(
            globals(), self, reduced_node_count
        )

    def fuse_shufflechannel(self, reduced_node_count: list[int]) -> None:
        return graph().onnx_converter_fuse_shufflechannel(
            globals(), self, reduced_node_count
        )

    def fuse_shufflechannel_split(self, reduced_node_count: list[int]) -> None:
        return graph().onnx_converter_fuse_shufflechannel_split(
            globals(), self, reduced_node_count
        )

    def fuse_hardswish(self, reduced_node_count: list[int]) -> None:
        return graph().onnx_converter_fuse_hardswish(
            globals(), self, reduced_node_count
        )

    def fuse_hardsigmoid(self, reduced_node_count: list[int]) -> None:
        return graph().onnx_converter_fuse_hardsigmoid(
            globals(), self, reduced_node_count
        )

    def fuse_swish(self, reduced_node_count: list[int]) -> None:
        return graph().onnx_converter_fuse_swish(globals(), self, reduced_node_count)

    def fuse_batchnorm1d_squeeze_unsqueeze(self, reduced_node_count: list[int]) -> None:
        return graph().onnx_converter_fuse_batchnorm1d_squeeze_unsqueeze(
            globals(), self, reduced_node_count
        )

    def fuse_unsqueeze_prelu(self, reduced_node_count: list[int]) -> None:
        return graph().onnx_converter_fuse_unsqueeze_prelu(
            globals(), self, reduced_node_count
        )

    def fuse_normalize(self, reduced_node_count: list[int]) -> None:
        return graph().onnx_converter_fuse_normalize(
            globals(), self, reduced_node_count
        )

    def fuse_groupnorm(self, reduced_node_count: list[int]) -> None:
        return graph().onnx_converter_fuse_groupnorm(
            globals(), self, reduced_node_count
        )

    def fuse_layernorm(self, reduced_node_count: list[int]) -> None:
        return graph().onnx_converter_fuse_layernorm(
            globals(), self, reduced_node_count
        )

    def fuse_flatten(self, reduced_node_count: list[int]) -> None:
        return graph().onnx_converter_fuse_flatten(globals(), self, reduced_node_count)

    def fuse_pixelshuffle(self, reduced_node_count: list[int]) -> None:
        return graph().onnx_converter_fuse_pixelshuffle(
            globals(), self, reduced_node_count
        )

    def fuse_reorg(self, reduced_node_count: list[int]) -> None:
        return graph().onnx_converter_fuse_reorg(globals(), self, reduced_node_count)

    def fuse_expand_broadcast(self, reduced_node_count: list[int]) -> None:
        return graph().onnx_converter_fuse_expand_broadcast(
            globals(), self, reduced_node_count
        )

    def fuse_lstm_gru_rnn(self, reduced_node_count: list[int]) -> None:
        return graph().onnx_converter_fuse_lstm_gru_rnn(
            globals(), self, reduced_node_count
        )

    def fuse_multiheadattention(self, reduced_node_count: list[int]) -> None:
        return graph().onnx_converter_fuse_multiheadattention(
            globals(), self, reduced_node_count
        )

    def fuse_binaryop_with_scalar(self) -> None:
        return graph().onnx_converter_fuse_binaryop_with_scalar(globals(), self)

    def convert(self, is_fp16: bool = False, include_mem_data: bool = True):
        return graph().onnx_converter_convert(
            globals(), self, is_fp16, include_mem_data
        )
