# Nothing here may depend on ONNX.
from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from threading import Lock
from typing import Final, Literal, Union
from weakref import WeakKeyDictionary

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
        """The least non-negative `(pad_width, pad_height)` that meets the requirement."""

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

# The architectures (spandrel's names, as a PyTorch model's type carries them) that convert in
# FP16, each one measured on TensorRT on trained weights, with the sizes measured (spandrel's
# tags), or none for every size: FP16's reach can differ by 300 times between sizes. Every
# other architecture or size, and any that spandrel adds, converts in FP32, which is always
# right and only slower. A pass on the CPU does not earn a line: HVI-CIDNet reads 74 dB there
# and 34 dB on TensorRT. The numbers are an FP16 engine against an FP32 engine (Fixed 256 px,
# level 3, TF32 off) on real crops and smooth images, in dB; those of the first seven are in
# native/STATUS.md.
FP16_ARCHS: Final[dict[str, tuple[str, ...]]] = {
    "ESRGAN": (),
    "HAT": (),
    "DAT": (),
    "RealESRGAN Compact": (),
    "SPAN": (),
    "PLKSR": (),
    "OmniSR": (),
    "RCAN": (),  # 63.87, 65.50
    "Swift-SRGAN": (),  # 63.29, 67.92
    "MoSR": (),  # 68.69, 69.22
    "DRUNet": (),  # 63.85, 63.47
    "DCTLSA": (),  # 61.77, 63.96
    "FBCNN": (),  # 64.63, 67.41
    "RealCUGAN": (),  # 63.62, 62.58
    "DnCNN": (),  # 54.88, 52.74
    "MMRealSR": (),  # 55.04, 64.03
    "MixDehazeNet": (),  # 58.92, 55.22
    "SwinIR": ("small", "medium", "large"),  # 59.55, 54.11; 64.18, 64.74; 62.01, 54.50
    "FDAT": ("medium", "xl"),  # 66.15, 70.85; 67.84, 68.85
    "Swin2SR": ("medium",),  # 63.31, 66.94
    "RGT": (),  # 64.61, 64.16
    "DRCT": ("large",),  # 63.15, 65.13
    # SeemoRe stays FP32: in FP16 its router's logits tie on about 1 % of natural images
    # and two experts fire; one crop reads 33.42 dB even with the tie broken.
    "RetinexFormer": (),  # 68.44 to 72.75, flat white 48.19
    # On gradients and bars its FP32 misses 50 dB too (FFT rounding: 34 dB on a horizontal one)
    "FIGSR": (),  # 53.88 to 65.93, flats 65.48 to 72.42
    # Its Dynamic FP16 engine against eager FP32 (TF32 off), 4x, 8 real crops at 256 and 384 px
    "HEART": (),  # 67.73 to 74.41; 65.76 to 71.06
    # Its Dynamic FP16 engine against eager FP32 (TF32 off), a trained 2x, 8 real crops
    "GaterV3": (),  # 68.56 on its worst crop, 70.49 on average
}
# The names, as a navi union of strings: a type knows no sizes.
FP16_ARCHS_NAVI: Final = " | ".join(f'"{arch}"' for arch in FP16_ARCHS)


def converts_in_fp16(arch: str, tags: Iterable[str]) -> bool:
    """Whether FP16_ARCHS lists a model of this architecture and these tags. The tags are read
    only at conversion: the ONNX records its precision."""
    sizes = FP16_ARCHS.get(arch)
    return sizes is not None and (not sizes or any(tag in sizes for tag in tags))


# The ONNX metadata keys of the architecture (a PyTorch model's name for it) and the
# precision ("fp16" or "fp32") Convert To ONNX records, so that a saved ONNX loaded later
# builds with no precision choice too (load.py). Released versions recorded the
# architecture alone, in FP16 files.
ARCH_METADATA_KEY = "chainner_c.architecture"
PRECISION_METADATA_KEY = "chainner_c.precision"
# The arch of an ONNX's type once it is known to have no record, the one case that
# leaves Build Engine's and Build Program's Precision to the user.
NO_RECORD_ARCH: Final = "unrecorded"


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

    # The architecture Convert To ONNX recorded, whose precision is `dtype`; None for any
    # other ONNX. Set by load_onnx_model, as the native loader passes a fixed set of
    # arguments.
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

_digest_lock = Lock()
_digests: WeakKeyDictionary[OnnxModel, bytes] = WeakKeyDictionary()


def content_digest(model: OnnxModel) -> bytes:
    """The SHA-256 of `model`'s bytes, which keys what is kept of it (its size
    requirement, sessions, engines and programs); computed once per model."""
    with _digest_lock:
        digest = _digests.get(model)
        if digest is None:
            digest = hashlib.sha256(model.bytes).digest()
            _digests[model] = digest
        return digest
