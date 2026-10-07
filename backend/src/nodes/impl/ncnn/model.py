from __future__ import annotations

import os
from json import load as jload
from pathlib import Path
from typing import BinaryIO

import numpy as np

# `x as x` imports: the native mirror reads these names in native/src/ncnn_graph.cpp
from sanic.log import logger as logger

from .. import native_ncnn_graph

param_schema_file = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "param_schema.json"
)
with open(param_schema_file, encoding="utf-8") as schemaf:
    param_schema = jload(schemaf)

DTYPE_FP32 = b"\x00\x00\x00\x00"
DTYPE_FP16 = b"\x47\x6b\x30\x01"
DTYPE_DICT = {b"\x00\x00\x00\x00": np.float32, b"\x47\x6b\x30\x01": np.float16}


class UnaryOpTypes:
    ABS = 0
    NEG = 1
    FLOOR = 2
    CEIL = 3
    SQUARE = 4
    SQRT = 5
    RSQ = 6
    EXP = 7
    LOG = 8
    SIN = 9
    COS = 10
    TAN = 11
    ASIN = 12
    ACOS = 13
    ATAN = 14
    RECIPROCAL = 15
    TANH = 16


class BinaryOpTypes:
    ADD = 0
    SUB = 1
    MUL = 2
    DIV = 3
    MAX = 4
    MIN = 5
    POW = 6
    RSUB = 7
    RDIV = 8


class CastElementTypes:
    AUTO = 0
    FLOAT32 = 1
    FLOAT16 = 2
    INT8 = 3
    BFLOAT16 = 4


class EltwiseOpTypes:
    PROD = 0
    SUM = 1
    MAX = 2


class GruDirectionFlags:
    FORWARD = 0
    REVERSE = 1
    BIDIRECTIONAL = 2


class InterpResizeTypes:
    NEAREST = 1
    BILINEAR = 2
    BICUBIC = 3


class NormalizeEpsModes:
    CAFFE = 0
    PYTORCH = 1
    TENSORFLOW = 2


class PaddingTypes:
    CONSTANT = 0
    REPLICATE = 1
    REFLECT = 2


class PadModes:
    FULL = 0
    VALID = 1
    SAMEUPPER = 2
    SAMELOWER = 3


class PermuteOrderTypes:
    WH_WHC_WHDC = 0
    HW_HWC_HWDC = 1
    WCH_WDHC = 2
    CWH_DWHC = 3
    HCW_HDWC = 4
    CHW_DHWC = 5
    WHCD = 6
    HWCD = 7
    WCHD = 8
    CWHD = 9
    HCWD = 10
    CHWD = 11
    WDCH = 12
    DWCH = 13
    WCDH = 14
    CWDH = 15
    DCWH = 16
    CDWH = 17
    HDCW = 18
    DHCW = 19
    HCDW = 20
    CHDW = 21
    DCHW = 22
    CDHW = 23


class ReductionOpTypes:
    SUM = 0
    ASUM = 1
    SUMSQ = 2
    MEAN = 3
    MAX = 4
    MIN = 5
    PROD = 6
    L1 = 7
    L2 = 8
    LOGSUM = 9
    LOGSUMEXP = 10


class GridSampleSampleTypes:
    NEAREST = 1
    BILINEAR = 2
    BICUBIC = 3


class GridSamplePadModes:
    ZEROS = 1
    BORDER = 2
    REFLECTION = 3


class LrnRegionTypes:
    ACROSS_CHANNELS = 0
    WITH_CHANNEL = 1


class NcnnWeight:
    def __init__(self, weight: np.ndarray, quantize_tag: bytes = b""):
        self.quantize_tag = quantize_tag
        self.weight = weight

    @property
    def shape(self) -> tuple:
        return self.weight.shape


class NcnnParam:
    def __init__(
        self,
        pid: str,
        name: str,
        value: float | int | list[float | int],
        default: float | int,
    ) -> None:
        self.id: str = pid
        self.name: str = name
        self.value: float | int | list[float | int] = value
        self.default: float | int = default


class NcnnParamCollection:
    def __init__(
        self,
        op: str,
        param_dict: dict[int, NcnnParam] | None = None,
    ) -> None:
        self.op: str = op
        self.param_dict: dict[int, NcnnParam] = {} if param_dict is None else param_dict
        self.weight_order: dict[str, list[int]] = (
            param_schema[self.op]["weightOrder"] if self.op else {}
        )

    def __getitem__(self, pid: int) -> NcnnParam:
        return native_ncnn_graph.param_get(self, pid)

    def __setitem__(self, pid: int, value: float | int | list[float | int]) -> None:
        native_ncnn_graph.param_set(self, pid, value)

    def __delitem__(self, key: int) -> None:
        try:
            del self.param_dict[key]
        except KeyError:
            pass

    def __contains__(self, item: int) -> bool:
        if item in self.param_dict:
            return True
        return False

    def __str__(self) -> str:
        return native_ncnn_graph.param_string(self)

    def set_op(self, op: str) -> None:
        self.op = op
        self.weight_order = param_schema[op]["weightOrder"]


class NcnnLayer:
    def __init__(
        self,
        op_type: str = "",
        name: str = "",
        num_inputs: int = 0,
        num_outputs: int = 0,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        params: NcnnParamCollection | None = None,
        weight_data: dict[str, NcnnWeight] | None = None,
    ):
        self.op_type: str = op_type
        self.name: str = name
        self.num_inputs: int = num_inputs
        self.num_outputs: int = num_outputs
        self.inputs: list[str] = [] if inputs is None else inputs
        self.outputs: list[str] = [] if outputs is None else outputs
        self.params: NcnnParamCollection = (
            NcnnParamCollection(op_type) if params is None else params
        )
        self.weight_data: dict[str, NcnnWeight] = (
            {} if weight_data is None else weight_data
        )

    def add_param(self, pid: int, value: float | int | list[float | int]) -> None:
        self.params[pid] = value

    def add_weight(
        self,
        weight_name: str,
        data: float | int | np.ndarray,
        quantize_tag: bytes = b"",
    ) -> int:
        return native_ncnn_graph.add_weight(self, weight_name, data, quantize_tag)


class NcnnModel:
    def __init__(
        self,
        node_count: int = 0,
        blob_count: int = 0,
    ) -> None:
        self.node_count: int = node_count
        self.blob_count: int = blob_count
        self.layers: list[NcnnLayer] = []
        self.bin_length = 0

    @property
    def magic(self):
        return "7767517"

    @staticmethod
    def load_from_file(param_path: str = "", bin_path: str = "") -> NcnnModel:
        if bin_path == "":
            bin_path = param_path.replace(".param", ".bin")
        elif param_path == "":
            param_path = bin_path.replace(".bin", ".param")
        with open(param_path, encoding="utf-8") as paramf:
            with open(bin_path, "rb") as binf:
                return native_ncnn_graph.read_model(paramf, binf)

    @staticmethod
    def interp_layers(
        a: NcnnLayer, b: NcnnLayer, alpha_a: float
    ) -> tuple[NcnnLayer, bytes]:
        return native_ncnn_graph.interp_layers(a, b, alpha_a)

    def add_layer(self, layer: NcnnLayer) -> None:
        self.layers.append(layer)

    def parse_param_layer(self, layer_str: str) -> tuple[str, NcnnLayer]:
        return native_ncnn_graph.parse_layer(layer_str)

    def load_layer_weights(
        self, binf: BinaryIO, op_type: str, layer: NcnnLayer
    ) -> dict[str, NcnnWeight]:
        return native_ncnn_graph.load_weights(binf, op_type, layer)

    def write_param(self, filename: Path | str = "") -> str:
        text = native_ncnn_graph.write_param(self)
        if filename:
            with open(filename, "w", encoding="utf-8") as file:
                file.write(text)
            return ""
        return text

    def serialize_weights(self) -> bytes:
        return native_ncnn_graph.serialize_weights(self)

    def write_bin(self, filename: Path | str) -> None:
        with open(filename, "wb") as f:
            f.write(self.serialize_weights())

    def interpolate(self, model_b: NcnnModel, alpha: float) -> NcnnModel:
        return native_ncnn_graph.interpolate(self, model_b, alpha)

    @property
    def bin(self) -> bytes:
        return self.serialize_weights()


class NcnnModelWrapper:
    def __init__(self, model: NcnnModel) -> None:
        self.model: NcnnModel = model
        scale, in_nc, out_nc, nf, fp = NcnnModelWrapper.get_broadcast_data(model)
        self.scale: int = scale
        self.nf: int = nf
        self.in_nc: int = in_nc
        self.out_nc: int = out_nc
        self.fp: str = fp

    @staticmethod
    def get_broadcast_data(model: NcnnModel) -> tuple[int, int, int, int, str]:
        return native_ncnn_graph.broadcast_data(model)

    @staticmethod
    def get_nf_and_in_nc(layer: NcnnLayer) -> tuple[int, int]:
        return native_ncnn_graph.get_nf(layer)
