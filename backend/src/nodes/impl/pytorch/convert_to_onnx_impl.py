from io import BytesIO

import torch
from spandrel import ImageModelDescriptor, ModelDescriptor
from spandrel.architectures.CRAFT import CRAFT
from spandrel.architectures.SAFMN import SAFMN
from spandrel.architectures.SCUNet import SCUNet


def is_onnx_supported(model: ModelDescriptor) -> bool:
    return not isinstance(model.model, (SCUNet, SAFMN, CRAFT))


def convert_to_onnx_impl(
    model: ModelDescriptor,
    device: torch.device,
    use_half: bool = False,
    input_name: str = "input",
    output_name: str = "output",
    opset_version: int = 14,
) -> bytes:
    # https://github.com/onnx/onnx/issues/654
    dynamic_axes = {
        input_name: {0: "batch_size", 2: "height", 3: "width"},
        output_name: {0: "batch_size", 2: "height", 3: "width"},
    }
    size = 32
    dummy_input = torch.rand(1, model.input_channels, size, size)
    dummy_input = dummy_input.to(device)

    if use_half:
        if not model.supports_half:
            raise ValueError(
                f"Model of arch {model.architecture} does not support half precision."
            )
        model.half()
        dummy_input = dummy_input.half()
    else:
        model.float()
        dummy_input = dummy_input.float()

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

    with BytesIO() as f:
        # The TorchScript exporter, which the opsets and dynamic_axes here are written
        # for. Torch's default exporter (dynamo, since torch 2.9) needs onnxscript,
        # which chaiNNer does not install.
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
        return f.read()
