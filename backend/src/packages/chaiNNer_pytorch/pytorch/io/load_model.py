from __future__ import annotations

import os
from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path
from weakref import WeakKeyDictionary

import torch
from sanic.log import logger
from spandrel import MAIN_REGISTRY, ModelDescriptor, ModelLoader
from spandrel_extra_arches import EXTRA_REGISTRY

from api import NodeContext
from nodes.impl.pytorch.resource_cache import file_identity
from nodes.impl.pytorch.skip_init import skip_random_init
from nodes.properties.inputs import PthFileInput
from nodes.properties.outputs import DirectoryOutput, FileNameOutput, ModelOutput
from nodes.utils.utils import split_file_path

from ...settings import PyTorchSettings, get_settings
from .. import io_group

# Without duplicates in the worker. vsview's Reload Script imports this module again
# for a script's AI Upscale, while spandrel keeps the architectures added before.
MAIN_REGISTRY.add(*EXTRA_REGISTRY, ignore_duplicates=True)


def parse_ckpt_state_dict(checkpoint: dict):
    state_dict = {}
    for i, j in checkpoint.items():
        if "netG." in i:
            key = i.replace("netG.", "")
            state_dict[key] = j
        elif "module." in i:
            key = i.replace("module.", "")
            state_dict[key] = j
    return state_dict


# The file and its identity at load of each model loaded in half precision, which
# Convert To ONNX reads again for an export in full precision.
_half_sources: WeakKeyDictionary[ModelDescriptor, tuple[Path, Hashable]] = (
    WeakKeyDictionary()
)


@dataclass(frozen=True)
class ReadModel:
    """A model file read and built, load_model's first step: in any thread, and on the
    CPU without GPU memory (Load Models reads its next model meanwhile)."""

    path: Path
    model: ModelDescriptor
    identity: Hashable


def read_model(path: Path, device: torch.device) -> ReadModel:
    try:
        logger.debug(f"Reading state dict from path: {path}")

        identity = file_identity(path)
        with skip_random_init():
            return ReadModel(path, ModelLoader(device).load_from_file(path), identity)
    except Exception as e:
        raise _unsupported(path) from e


def finish_model(read: ReadModel, exec_options: PyTorchSettings) -> ModelDescriptor:
    """A read model on the settings' device and precision, ready for inference."""
    try:
        model_descriptor = read.model
        for _, v in model_descriptor.model.named_parameters():
            v.requires_grad = False
        model_descriptor.model.eval()
        model_descriptor = model_descriptor.to(exec_options.device)
        should_use_fp16 = exec_options.use_fp16 and model_descriptor.supports_half
        if should_use_fp16:
            model_descriptor.model.half()
            _half_sources[model_descriptor] = (read.path, read.identity)
        else:
            model_descriptor.model.float()
    except Exception as e:
        raise _unsupported(read.path) from e
    return model_descriptor


def _unsupported(path: Path) -> ValueError:
    return ValueError(
        f"Model {os.path.basename(path)} is unsupported by chaiNNer. Please try"
        " another."
    )


def load_model(path: Path, exec_options: PyTorchSettings) -> ModelDescriptor:
    """A model file on the settings' device and precision, ready for inference (also
    used by VapourSynth's AI Upscale, which has no node context)."""
    return finish_model(read_model(path, exec_options.device), exec_options)


def check_model_file(path: Path) -> None:
    assert os.path.exists(path), f"Model file at location {path} does not exist"

    assert os.path.isfile(path), f"Path {path} is not a file"


def full_precision(model: ModelDescriptor) -> ModelDescriptor | str | None:
    """`model`, which Load Model loaded in half precision, read again from its file on
    the CPU in full precision; or why it cannot be (it has FP16 weights not from Load
    Model, or the file changed or no longer holds the weights it was loaded with);
    None for a model in full precision."""
    source = _half_sources.get(model)
    if source is None:
        return (
            "it does not come from Load Model" if model.dtype == torch.float16 else None
        )
    path, identity = source
    if file_identity(path) != identity:
        return f"{path.name} changed after Load Model loaded it"
    with skip_random_init():
        reread = ModelLoader(torch.device("cpu")).load_from_file(path)
    reread.model.float()
    if not _rounds_to(reread.model.state_dict(), model.model.state_dict()):
        return f"{path.name} no longer holds the weights Load Model loaded"
    return reread


def _rounds_to(full: dict[str, torch.Tensor], loaded: dict[str, torch.Tensor]) -> bool:
    """Whether `full`'s floating-point tensors rounded as model.half() rounds them are
    bitwise the loaded ones (still in half precision, or converted back since by
    Upscale Image), and its other tensors are the loaded ones."""
    if full.keys() != loaded.keys():
        return False
    for key, value in full.items():
        actual = loaded[key].detach().cpu()
        expected = value
        if value.is_floating_point() and actual.is_floating_point():
            expected = value.half().to(actual.dtype)
        if expected.dtype != actual.dtype or expected.shape != actual.shape:
            return False
        # Bitwise, so that NaNs compare equal
        if not torch.equal(_bytes(expected), _bytes(actual)):
            return False
    return True


def _bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().reshape(-1).view(torch.uint8)


@io_group.register(
    schema_id="chainner:pytorch:load_model",
    name="Load Model",
    description=[
        ("Load PyTorch model file into an auto-detected supported model architecture."),
        (
            "- For Super-Resolution, we support most variations of the RRDB"
            " architecture (ESRGAN, Real-ESRGAN, RealSR, BSRGAN, SPSR), Real-ESRGAN's"
            " SRVGG architecture, Swift-SRGAN, SwinIR, Swin2SR, HAT, Omni-SR, SRFormer, and DAT."
        ),
        (
            "- For Face-Restoration, we support GFPGAN (1.2, 1.3, 1.4), RestoreFormer,"
            " and CodeFormer."
        ),
        "- For Inpainting, we support LaMa and MAT.",
        (
            "Links to the official models can be found in [chaiNNer's"
            " README](https://github.com/chaiNNer-org/chaiNNer#pytorch), and"
            " community-trained models on [OpenModelDB](https://openmodeldb.info/)."
        ),
    ],
    icon="PyTorch",
    inputs=[PthFileInput(primary_input=True)],
    outputs=[
        ModelOutput(kind="tagged").suggest(),
        DirectoryOutput("Directory", of_input=0).with_id(2),
        FileNameOutput("Name", of_input=0).with_id(1),
    ],
    node_context=True,
    see_also=[
        "chainner:pytorch:load_models",
    ],
    side_effects=True,
)
def load_model_node(
    context: NodeContext, path: Path
) -> tuple[ModelDescriptor, Path, str]:
    check_model_file(path)

    model_descriptor = load_model(path, get_settings(context))

    dirname, basename, _ = split_file_path(path)
    return model_descriptor, dirname, basename
