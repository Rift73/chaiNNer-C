# This class defines an interface.
# It is important that is does not contain types that depend on ONNX.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal, Union

OnnxSubType = Literal["Generic", "RemBg"]


def _ceil_div(a: int, b: int):
    return -(a // -b)


@dataclass(init=False)
class SizeReq:
    minimum: int
    multiple_of: int

    def __init__(self, minimum: int = 1, multiple_of: int = 1):
        if minimum < 1:
            raise ValueError("minimum must be at least 1")
        if multiple_of < 1:
            raise ValueError("multiple_of must be at least 1")

        self.minimum = _ceil_div(minimum, multiple_of) * multiple_of
        self.multiple_of = multiple_of

    def get_padding(self, width: int, height: int) -> tuple[int, int]:
        """
        Given an image size, this returns the minimum amount of padding necessary to satisfy the size requirements. The returned padding is in the format `(pad_width, pad_height)` and is guaranteed to be non-negative.
        """

        def ceil_modulo(x: int, mod: int) -> int:
            if x % mod == 0:
                return x
            return (x // mod + 1) * mod

        w: int = max(self.minimum, width)
        h: int = max(self.minimum, height)

        w = ceil_modulo(w, self.multiple_of)
        h = ceil_modulo(h, self.multiple_of)

        return w - width, h - height


# The ONNX metadata key of the size requirement Convert To ONNX records from the PyTorch
# model (spandrel's minimum and multiple_of, as JSON), which ONNX Upscale Image pads
# each tile to as PyTorch pads the model's input (size_probe.py). A square requirement
# is not recorded: the export's graph pads to it (spandrel traces that padding).
SIZE_REQ_METADATA_KEY = "chainner_c.size_requirements"

# The architectures (spandrel's ids, which are also their names) that convert and build
# in FP16 with no precision choice.
FP16_ARCHS: Final = ("ESRGAN", "HAT", "DAT")
# The same, as a navi union of strings
FP16_ARCHS_NAVI: Final = " | ".join(f'"{arch}"' for arch in FP16_ARCHS)
# The ONNX metadata key of the architecture Convert To ONNX records for FP16_ARCHS, so
# that a saved ONNX loaded later builds with no precision choice too (load.py).
ARCH_METADATA_KEY = "chainner_c.architecture"


# The dtype of a DUAL or legacy DRFT ONNX from Convert To ONNX (a BF16 or FP16 body, FP32
# input and output), whose precision is its own (load.py).
MIXED = "mixed"


@dataclass
class OnnxInfo:
    opset: int
    dtype: str

    scale_width: int | None = None
    scale_height: int | None = None

    fixed_input_width: int | None = None
    fixed_input_height: int | None = None

    input_channels: int | None = None
    output_channels: int | None = None

    size_req: SizeReq = field(default_factory=SizeReq)

    # The architecture Convert To ONNX recorded (one of FP16_ARCHS); None for any other
    # ONNX. Set by load_onnx_model, as the native loader passes a fixed set of arguments.
    arch: str | None = None


class OnnxGeneric:
    def __init__(self, model_as_bytes: bytes, info: OnnxInfo):
        self.bytes: bytes = model_as_bytes
        self.sub_type: Final[Literal["Generic"]] = "Generic"
        self.info: OnnxInfo = info


class OnnxRemBg:
    def __init__(
        self,
        model_as_bytes: bytes,
        info: OnnxInfo,
    ):
        self.bytes: bytes = model_as_bytes
        self.sub_type: Final[Literal["RemBg"]] = "RemBg"
        self.info: OnnxInfo = info


OnnxModels = (OnnxGeneric, OnnxRemBg)
OnnxModel = Union[OnnxGeneric, OnnxRemBg]
