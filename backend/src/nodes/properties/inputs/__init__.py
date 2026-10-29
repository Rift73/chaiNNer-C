"""The input types of node schemas.

The inputs of a framework's models (MIGraphX, NCNN, ONNX, PyTorch, TensorRT, VapourSynth) import
the framework, so their modules load on first use: a node module that uses none of
them loads without the framework, while the worker imports the frameworks in the
background.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from .colour_inputs import *
from .file_inputs import *
from .generic_inputs import *
from .image_dropdown_inputs import *
from .numeric_inputs import *
from .numpy_inputs import *

if TYPE_CHECKING:
    from .migraphx_inputs import *
    from .ncnn_inputs import *
    from .onnx_inputs import *
    from .pytorch_inputs import *
    from .tensorrt_inputs import *
    from .vapoursynth_inputs import *

# Each name the framework modules define, by its module.
FRAMEWORK_INPUTS: dict[str, str] = {
    "MIGraphXProgramInput": "migraphx_inputs",
    "NcnnModelInput": "ncnn_inputs",
    "OnnxModelInput": "onnx_inputs",
    "OnnxGenericModelInput": "onnx_inputs",
    "OnnxRemBgModelInput": "onnx_inputs",
    "OnnxFpDropdown": "onnx_inputs",
    "ModelInput": "pytorch_inputs",
    "SrModelInput": "pytorch_inputs",
    "FaceModelInput": "pytorch_inputs",
    "InpaintModelInput": "pytorch_inputs",
    "TorchScriptInput": "pytorch_inputs",
    "TensorRTEngineInput": "tensorrt_inputs",
    "TensorRTPrecisionDropdown": "tensorrt_inputs",
    "TensorRTShapeModeDropdown": "tensorrt_inputs",
    "FILTER_OR_CLIP": "vapoursynth_inputs",
    "VideoClipInput": "vapoursynth_inputs",
    "VSFilterInput": "vapoursynth_inputs",
}


def __getattr__(name: str) -> object:
    """Imports a framework's inputs on first use. A framework that fails to import fails
    the node module that uses it with the framework's error."""
    module = FRAMEWORK_INPUTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{module}", __name__), name)
