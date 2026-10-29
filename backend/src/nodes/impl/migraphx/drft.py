"""Legacy DRFT's MIGraphX path, the TensorRT one's counterpart (tensorrt/drft.py): Build
Program takes the DRFT ONNX that Convert To ONNX makes for both (tensorrt/drft_export.py),
rewrites its TensorRT plugin nodes into chaiNNer-C's DRFT MIGraphX ops (migraphx_onnx;
domain "chainner", native/src/drft_hip), and compiles it for one input size in a child
process that loads the ops' library (drft_migx_ops.so, on Windows drft_migx_ops.dll).
Programs do not carry the library: every process that loads one registers it first
(load_ops)."""

from __future__ import annotations

import hashlib
import sys
from functools import cache, lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..tensorrt.drft import check_profile, export_manifest
from ..tensorrt.drft_export import PLUGINS
from ..tensorrt.dual import aligned_input_name
from . import program, runner

if TYPE_CHECKING:
    from api import NodeContext

# The ops' library (native/src/drft_hip, built with the ROCm SDK).
OP_LIBRARY = Path(__file__).with_name(
    "drft_migx_ops.dll" if sys.platform == "win32" else "drft_migx_ops.so"
)
# Every file a DRFT program build reads besides MIGraphX and the ONNX.
BUILD_SOURCES = (OP_LIBRARY, Path(__file__))
DOMAIN = "chainner"
# The TensorRT plugins that become one MIGraphX op each, with their attributes.
_OPS = {
    "DrftWindowNeuralBias_TRT": "chainner_drft_window_neural_bias",
    "DrftOCABRelativeBiasSafe_TRT": "chainner_drft_ocab_relative_bias",
    "DrftOCABAttentionSafe_TRT": "chainner_drft_ocab_attention",
    "DrftFinalAffineCrop_TRT": "chainner_drft_final_affine_crop",
}
# The q/k/v pack's three outputs become one window pack each.
_QKV_PACK = "DrftOCABQKVPackTokenMajor_TRT"
_WINDOW_PACK = "chainner_drft_ocab_window_pack"
# A TensorRT partition boundary, the identity.
_BARRIER = "DrftIdentityBarrier_TRT"
# Attributes TensorRT's ONNX parser reads, not the plugins.
_PLUGIN_ATTRIBUTES = {"plugin_version", "plugin_namespace"}


def op_library() -> Path:
    """The ops' library a program build loads, or an error that says how to get it."""
    if not OP_LIBRARY.is_file():
        raise RuntimeError(
            f"The DRFT MIGraphX op library {OP_LIBRARY} is missing: this build was"
            " made without it. Use the PyTorch backend, or build it with the ROCm"
            " 7.14 SDK (native/build_linux.sh -D CHAINNER_C_ROCM_ROOT=<SDK>, on Windows"
            " native/Build.ps1 -RocmRoot <SDK>)."
        )
    return OP_LIBRARY


@lru_cache(maxsize=1)
def load_ops() -> None:
    """Register the library's ops in this process's MIGraphX, once, before any DRFT
    program is loaded. A build without the library registers nothing; DRFT programs
    then fail to load, other programs are unaffected."""
    if OP_LIBRARY.is_file():
        runner.load_migraphx().load_op_library(str(OP_LIBRARY))


@cache
def build_identity() -> str:
    """The SHA-256 of BUILD_SOURCES (a missing one skipped: a build without the
    library fails, so no program is kept under its identity)."""
    digest = hashlib.sha256()
    for source in BUILD_SOURCES:
        if source.is_file():
            digest.update(source.read_bytes())
    return digest.hexdigest()


def migraphx_onnx(onnx_bytes: bytes, manifest: dict[str, Any]) -> bytes:
    """A DRFT ONNX (export_manifest) with its plugin nodes as the MIGraphX ops: the
    neural bias, relative bias, OCAB attention and output tail op for op (with the
    plugins' attributes), OCAB's q/k/v pack as three window packs (q at the query
    window, k and v at the key window), and the identity barriers as Identity. Its
    input is named for the ONNX's alignment (dual.aligned_input_name), which marks the
    program as mixed precision when it is loaded and which Upscale Image's tiler pads
    to."""
    import onnx
    from onnx import helper

    if manifest["plugins"] != PLUGINS:
        raise ValueError(
            f"This DRFT ONNX uses the plugin set {manifest['plugins']!r}; MIGraphX takes"
            f" {PLUGINS!r} only. Convert the model to ONNX again."
        )
    model = onnx.load_from_string(onnx_bytes)
    graph = model.graph
    source, name = graph.input[0].name, aligned_input_name(manifest["alignment"])
    graph.input[0].name = name
    nodes: list[onnx.NodeProto] = []
    for node in graph.node:
        inputs = [name if value == source else value for value in node.input]
        attributes = {
            attribute.name: helper.get_attribute_value(attribute)
            for attribute in node.attribute
            if attribute.name not in _PLUGIN_ATTRIBUTES
        }
        if node.op_type == _BARRIER:
            nodes.append(helper.make_node("Identity", inputs, node.output, node.name))
        elif node.op_type == _QKV_PACK:
            query, key = (
                attributes["query_window_size"],
                attributes["key_window_size"],
            )
            for component, (output, window) in enumerate(
                zip(node.output, (query, key, key), strict=True)
            ):
                nodes.append(
                    helper.make_node(
                        _WINDOW_PACK,
                        inputs,
                        [output],
                        f"{node.name}_{component}",
                        domain=DOMAIN,
                        heads=attributes["heads"],
                        query_window_size=query,
                        window_size=window,
                        component=component,
                    )
                )
        elif node.op_type in _OPS:
            nodes.append(
                helper.make_node(
                    _OPS[node.op_type],
                    inputs,
                    node.output,
                    node.name,
                    domain=DOMAIN,
                    **attributes,
                )
            )
        elif node.op_type.endswith("_TRT"):
            raise ValueError(f"MIGraphX has no op for the DRFT plugin {node.op_type}.")
        else:
            kept = onnx.NodeProto()
            kept.CopyFrom(node)
            kept.input[:] = inputs
            nodes.append(kept)
    graph.ClearField("node")
    graph.node.extend(nodes)
    model.opset_import.append(helper.make_opsetid(DOMAIN, 1))
    return model.SerializeToString()


Size = tuple[int, int]


def build_program(
    onnx_bytes: bytes,
    manifest: dict[str, Any],
    gpu_index: int,
    size: Size,
    *,
    exhaustive_tune: bool = False,
    context: NodeContext | None = None,
) -> bytes:
    """A MIGraphX program of a DRFT ONNX for one (height, width) input size (see
    check_profile), compiled in a child process that loads the ops' library. The ONNX
    keeps its own precision (an FP16 body with BF16 attention, FP32 input and output),
    so MIGraphX converts nothing to FP16 or FP8."""
    library = op_library()
    return program.compile_in_child(
        migraphx_onnx(onnx_bytes, manifest),
        gpu_index,
        size,
        fp16=False,
        fp8=False,
        exhaustive_tune=exhaustive_tune,
        op_libraries=(library,),
        label="DRFT program build",
        context=context,
    )


__all__ = [
    "BUILD_SOURCES",
    "DOMAIN",
    "OP_LIBRARY",
    "build_identity",
    "build_program",
    "check_profile",
    "export_manifest",
    "load_ops",
    "migraphx_onnx",
    "op_library",
]
