import copy
import json
from io import BytesIO
from typing import TypeVar

import onnx
import torch
from spandrel import ImageModelDescriptor, ModelDescriptor, SizeRequirements
from spandrel.architectures.CRAFT import CRAFT
from spandrel.architectures.DAT.__arch.DAT import Adaptive_Spatial_Attention
from spandrel.architectures.DRCT.__arch.drct_arch import (
    SwinTransformerBlock as DrctBlock,
)
from spandrel.architectures.FFTformer import FFTformer
from spandrel.architectures.RealCUGAN import (
    UpCunet2x,
    UpCunet2x_fast,
    UpCunet3x,
    UpCunet4x,
)
from spandrel.architectures.RGT.__arch.rgt import L_SA
from spandrel.architectures.SAFMN import SAFMN
from spandrel.architectures.SAFMNBCIE import SAFMNBCIE
from spandrel.architectures.SCUNet import SCUNet
from spandrel.architectures.Swin2SR.__arch.Swin2SR import (
    SwinTransformerBlock as Swin2srBlock,
)
from spandrel.architectures.SwinIR.__arch.SwinIR import (
    SwinTransformerBlock as SwinirBlock,
)

from ..onnx.model import SIZE_REQ_METADATA_KEY
from . import (
    attention_export,
    dysample_export,
    esrgan_export,
    fourier_export,
    fp16_export,
    hat_export,
    input_export,
    lda_export,
    omnisr_export,
    plksr_export,
    swin_export,
    traced_size_export,
)

_Descriptor = TypeVar("_Descriptor", bound=ModelDescriptor)


def is_onnx_supported(model: ModelDescriptor) -> bool:
    # The exporter stops with its own error on SAFMNBCIE's adaptive pooling, as on SAFMN's,
    # and on FFTformer's Fourier transforms
    return not isinstance(model.model, (SCUNet, SAFMN, SAFMNBCIE, FFTformer, CRAFT))


def export_copy(
    model: _Descriptor, device: torch.device, use_half: bool = False
) -> _Descriptor:
    """A copy of `model` to export: in evaluation mode on `device`, in the export's
    precision. `model` itself is not changed: Upscale Image and Save Model share it."""
    model = copy.deepcopy(model)
    model.eval().to(device)
    if use_half:
        # Spandrel's supports_half speaks for PyTorch, as for RealPLKSR's GroupNorm, whose
        # ONNX holds on TensorRT and ONNX Runtime: 62 to 74 dB against PyTorch FP32 on the
        # worst of 8 crops of six trained weights, the same as with the statistics in FP32.
        # Convert To ONNX checks what an FP16 export gives (verify_models).
        # The sampler's weights stay as loaded: a copy taken before half() rounds them.
        sampler = plksr_export.fp32_sampler(model)
        first = input_export.fp32_first_convolution(model)
        model.supports_half = True
        model.half()
        plksr_export.sample_in_fp32(model, sampler)
        input_export.take_fp32_input(model, first)
    else:
        model.float()
    return model


def _example_side(model: ModelDescriptor) -> int:
    """The side of the square example an export traces: 32, but for what a Python branch
    of the model's forward would keep for every size."""
    # RealCUGAN pads a size its convolutions do not divide inside its forward, and crops
    # the output after in a Python branch, which the trace keeps only when the example
    # takes it: an odd example (over its minimum of 32) keeps the crop for every size.
    net = model.model
    if isinstance(net, (UpCunet2x, UpCunet2x_fast, UpCunet3x, UpCunet4x)):
        return 33
    # A block with shifted windows takes the mask it stored at its training size, and
    # builds one from the input's size at any other. An example that reaches the stored
    # size keeps that mask: other sizes end in a failed Reshape, or, with as many windows,
    # at 45 dB from PyTorch with no error. Padding only grows a side, so one over it
    # never reaches it.
    side = 32
    padded = side + model.size_requirements.get_padding(side, side)[0]
    for module in net.modules():
        if isinstance(module, (SwinirBlock, Swin2srBlock, DrctBlock)):
            stored, multiple = module.input_resolution[0], net.window_size
        elif isinstance(module, (L_SA, Adaptive_Spatial_Attention)):
            stored, multiple = module.patches_resolution, max(module.split_size)
        else:
            continue
        if -(-padded // multiple) * multiple == stored:
            return stored + 1
    return side


def export_onnx(
    model: ModelDescriptor,
    input_name: str = "input",
    output_name: str = "output",
    opset_version: int = 14,
) -> bytes:
    """The ONNX export of `model`, a copy from `export_copy`, in its precision."""
    # https://github.com/onnx/onnx/issues/654
    dynamic_axes = {
        input_name: {0: "batch_size", 2: "height", 3: "width"},
        output_name: {0: "batch_size", 2: "height", 3: "width"},
    }
    size = _example_side(model)
    dummy_input = torch.rand(1, model.input_channels, size, size)
    dummy_input = dummy_input.to(model.device, input_export.input_dtype(model))

    # HAT and DAT: a copy with their attention laid out for TensorRT (exact)
    model = attention_export.for_export(model)
    # HAT's FP16 trunk a multiple of 8 channels wide
    model = hat_export.for_export(model)
    # SwinIR, Swin2SR, RGT and DRCT in FP16: a copy with their masks in FP16 and the end of
    # their last group in FP32
    model = swin_export.for_export(model)
    # ESRGAN in FP16: a copy with its dense blocks laid out for TensorRT (exact)
    model = esrgan_export.for_export(model)
    # DySample: a copy with its sampling grid following the input's size (exact)
    model = dysample_export.for_export(model)
    # PLKSR: a copy with its large convolutions split and joined (exact)
    model = plksr_export.for_export(model)
    # OmniSR in FP16: a copy that takes its vector lengths and image means in FP32
    model = omnisr_export.for_export(model)
    # RetinexFormer in FP16: a copy that holds where FP16 alone gives NaN
    model = fp16_export.for_export(model)
    # GFISRv2 and FIGSR: a copy with their Fourier transforms in valid operators
    model = fourier_export.for_export(model, opset_version)
    # The LDA upsampler: a copy with its sizes following the input's (exact)
    model = lda_export.for_export(model)
    # DITN, HVICIDNet, FBCNN: a copy with sizes following the input's (exact)
    model = traced_size_export.for_export(model)
    # FP16 below opset 17: a copy with its LayerNorms in FP32, not the exporter's float64
    model = fp16_export.for_opset(model, opset_version)

    # The trace stores the weights as constants (FakeModel below holds the descriptor,
    # not the module), and a constant must not require grad. Load Model turns grad
    # off; Interpolate Models returns a freshly loaded model with grad still on.
    model.model.requires_grad_(False)

    m = model.model

    if isinstance(model, ImageModelDescriptor):
        req = model.size_requirements
        # Spandrel pads an input below the model's minimum size in Python branches, so
        # for a minimum alone the trace freezes the example's padding into the graph:
        # for a minimum above the example, every larger input is cropped to the
        # minimum. (Spandrel exports a multiple or a square requirement dynamically.)
        pad_to_minimum = req.minimum > size and req.multiple_of == 1 and not req.square
        scale = model.scale

        class FakeModel(torch.nn.Module):
            def __init__(self, model: ImageModelDescriptor):
                super().__init__()
                self.model = model

            def forward(self, x: torch.Tensor):
                if not pad_to_minimum:
                    return self.model(x)
                # Spandrel's padding in branch-free arithmetic, which stays dynamic:
                # reflect (at most size - 1), then replicate the rest.
                h, w = x.shape[-2:]
                pad_h = (h < req.minimum) * (req.minimum - h)
                pad_w = (w < req.minimum) * (req.minimum - w)
                reflect_h = pad_h - (pad_h > h - 1) * (pad_h - (h - 1))
                reflect_w = pad_w - (pad_w > w - 1) * (pad_w - (w - 1))
                x = torch.nn.functional.pad(x, (0, reflect_w, 0, reflect_h), "reflect")
                x = torch.nn.functional.pad(
                    x, (0, pad_w - reflect_w, 0, pad_h - reflect_h), "replicate"
                )
                return self.model(x)[..., : h * scale, : w * scale]

        m = FakeModel(model)

    # HAT's and DAT's attention laid out for the GPUs' runtime (same values)
    with BytesIO() as f, attention_export.attention_layout():
        # The TorchScript exporter, which the opsets and dynamic_axes here are written
        # for. Torch's default exporter (dynamo, since torch 2.9) needs onnxscript,
        # which chaiNNer does not install.
        with traced_size_export.uformer_sides(model):
            torch.onnx.export(
                m,
                (dummy_input,),
                f,  # type: ignore[arg-type]
                opset_version=opset_version,
                verbose=False,
                input_names=[input_name],
                output_names=[output_name],
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
                dynamo=False,
            )
        f.seek(0)
        onnx_bytes = f.read()
    if isinstance(model, ImageModelDescriptor):
        recorded = model.size_requirements
        # A square requirement stays the graph's: it pads in one step, as PyTorch does,
        # where ONNX Upscale Image padding to the multiple first would be a second.
        if not recorded.square and (recorded.minimum > 1 or recorded.multiple_of > 1):
            onnx_bytes = _with_size_requirements(onnx_bytes, recorded)
    return onnx_bytes


def _with_size_requirements(onnx_bytes: bytes, req: SizeRequirements) -> bytes:
    """The ONNX with its PyTorch model's size requirement in its metadata: ONNX Upscale
    Image pads each tile to it as PyTorch pads the model's input (onnx/size_probe.py),
    which the graph does not for a minimum alone (below the example size, see
    export_onnx)."""
    model = onnx.load_from_string(onnx_bytes)
    entry = model.metadata_props.add()
    entry.key = SIZE_REQ_METADATA_KEY
    entry.value = json.dumps({"minimum": req.minimum, "multiple_of": req.multiple_of})
    return model.SerializeToString()


def convert_to_onnx_impl(
    model: ModelDescriptor,
    device: torch.device,
    use_half: bool = False,
    input_name: str = "input",
    output_name: str = "output",
    opset_version: int = 14,
) -> bytes:
    """`model` exported to ONNX on `device`; `model` is not changed."""
    return export_onnx(
        export_copy(model, device, use_half), input_name, output_name, opset_version
    )
