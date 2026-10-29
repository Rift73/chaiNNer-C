"""Lower a DUAL ONNX for MIGraphX in a child process (dual.build_program runs it from
backend/src): the exporter's plugin nodes become DUAL's custom-op nodes, each carrying
its Triton kernel compiled for the AMD GPU named by its gfx (dual_hip.py); a dynamic
ONNX is made one of the program's size (dual_hip.fix_size); and the input is named for
the ONNX's alignment (tensorrt/dual.aligned_input_name), as a TensorRT engine's.
Compiling for an explicit target needs no GPU.

    python -m nodes.impl.migraphx.dual_lower_worker ONNX LOWERED GFX [HEIGHT WIDTH]

HEIGHT and WIDTH: a dynamic ONNX's program size (a fixed one has its own).
"""

from __future__ import annotations

import sys

from ..pytorch.vosr2_process import end_with_parent


def main(onnx_path: str, lowered_path: str, gfx: str, *size: str) -> None:
    import onnx

    from ..tensorrt import dual
    from . import dual_hip

    model = onnx.load(onnx_path)
    manifest = dual.export_manifest(model)
    if manifest is None:
        raise ValueError(f"{onnx_path} is not a DUAL ONNX from Convert To ONNX")
    spec = manifest["specialization"]
    dynamic = spec.get("dynamic", False)
    if len(size) != (2 if dynamic else 0):
        raise ValueError("A dynamic DUAL ONNX needs HEIGHT WIDTH; a fixed one none")
    lowered = dual_hip.lower(model, spec["plugin_key"], dual_hip.Target.amd(gfx))
    if dynamic:
        height, width = (int(value) for value in size)
        dual_hip.fix_size(lowered, height, width, spec["unshuffle"])
    alignment = spec["alignment"] if dynamic else dual.ALIGNMENT
    dual_hip.name_input(lowered, dual.aligned_input_name(alignment))
    with open(lowered_path, "wb") as file:
        file.write(lowered.SerializeToString())


if __name__ == "__main__":
    end_with_parent()
    main(*sys.argv[1:])
