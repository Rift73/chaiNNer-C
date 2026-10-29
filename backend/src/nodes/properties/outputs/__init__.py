"""The output types of node schemas.

The outputs of a framework's models (MIGraphX, NCNN, ONNX, PyTorch, TensorRT, VapourSynth) import
the framework, so their modules load on first use, as nodes.properties.inputs' do.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from .file_outputs import *
from .generic_outputs import *
from .numpy_outputs import *

if TYPE_CHECKING:
    from .migraphx_outputs import *
    from .ncnn_outputs import *
    from .onnx_outputs import *
    from .pytorch_outputs import *
    from .tensorrt_outputs import *
    from .vapoursynth_outputs import *

# Each name the framework modules define, by its module.
FRAMEWORK_OUTPUTS: dict[str, str] = {
    "MIGraphXProgramOutput": "migraphx_outputs",
    "NcnnModelOutput": "ncnn_outputs",
    "OnnxModelOutput": "onnx_outputs",
    "ModelOutput": "pytorch_outputs",
    "TorchScriptOutput": "pytorch_outputs",
    "TensorRTEngineOutput": "tensorrt_outputs",
    "VideoClipOutput": "vapoursynth_outputs",
}


def __getattr__(name: str) -> object:
    """Imports a framework's outputs on first use. A framework that fails to import fails
    the node module that uses it with the framework's error."""
    module = FRAMEWORK_OUTPUTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{module}", __name__), name)
