"""RSR's TensorRT ONNX parts for one input size, exported in a child process
(dual.run_child) from a folded model's weights:

- prefix:  x -> current6, c3, c4 (RSR.encode: the trunk up to the Halo stage and the
  two context maps its later gates read)
- revcore: current6, c3, c4 + state_future, valid_future -> state (the reverse sweep's
  step: RSR.step_encoded with a future neighbour only)
- fwdcore: x, current6, c3, c4 + state_past, valid_past, state_future, valid_future ->
  output, state (the forward sweep's step with the image)

The parts compute in FP16 (the model's weights in half precision) with the image `x` and
`output` in FP32 and every Resize (the context gates' 4x bilinear upsampling) in FP32:
in FP16 those maps lose the precision the gates' sigmoid needs. The validity flags
(BOOL [1]) are the model's own per-sample masks: an absent neighbour is a zero state
with its flag false, whose tokens the temporal read masks. Every shape is static.

    python -m nodes.impl.tensorrt.rsr_export WEIGHTS OUTPUT HEIGHT WIDTH [DEVICE]

DEVICE (a torch device, default cpu) is where the parts are traced; the graphs do not
depend on it. WEIGHTS is a safetensors file of the folded model's state dict whose
metadata holds the model's hyperparameters (rdual.HYPERPARAMETERS_KEY, JSON); each part
goes to OUTPUT/<part> (model.onnx and weights.bin), the manifest to
OUTPUT/export.json."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import onnx
import torch
from onnx import TensorProto, helper
from spandrel.architectures.RSR.__arch.rsr_arch import RSR
from torch import nn

from .rdual import HYPERPARAMETERS_KEY
from .rdual_export import save

PARTS = ("prefix", "revcore", "fwdcore")
# The bindings in FP32 and the validity flags; every other one is a handoff between the
# parts in FP16, which the build binds channel-last (rsr_engine_worker.py).
IMAGE_BINDINGS = ("x", "output")
VALIDITY_BINDINGS = ("valid_past", "valid_future")
OPSET = 20


class Prefix(nn.Module):
    def __init__(self, model: RSR) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.model.encode(x.half())


class RevCore(nn.Module):
    def __init__(self, model: RSR) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        current6: torch.Tensor,
        c3: torch.Tensor,
        c4: torch.Tensor,
        state_future: torch.Tensor,
        valid_future: torch.Tensor,
    ) -> torch.Tensor:
        # The frame is read only by the image, which this step does not make.
        _, state = self.model.step_encoded(
            current6,
            (current6, c3, c4),
            future=state_future,
            future_valid=valid_future,
            output=False,
        )
        return state


class FwdCore(nn.Module):
    def __init__(self, model: RSR) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        x: torch.Tensor,
        current6: torch.Tensor,
        c3: torch.Tensor,
        c4: torch.Tensor,
        state_past: torch.Tensor,
        valid_past: torch.Tensor,
        state_future: torch.Tensor,
        valid_future: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image, state = self.model.step_encoded(
            x.half(),
            (current6, c3, c4),
            state_past,
            state_future,
            valid_past,
            valid_future,
        )
        assert image is not None
        return image.float(), state


def load(weights: Path) -> RSR:
    """The folded model of a WEIGHTS file, on the CPU in FP32."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    with safe_open(str(weights), "pt") as file:
        metadata = file.metadata()
    hyperparameters = json.loads(metadata[HYPERPARAMETERS_KEY])
    model = RSR(**hyperparameters).float().eval()
    model.prepare_for_export()
    model.load_state_dict(load_file(str(weights)), strict=True)
    # The attention's deployment scale is a buffer the state dict does not hold: set
    # it from the logit scale just loaded.
    model.attention.prepare_for_export()
    model.requires_grad_(False)
    return model


def fp32_resize(model: onnx.ModelProto) -> int:
    """Every Resize of `model` computed in FP32: its data input cast to FP32 and its
    output cast back to FP16 (the graph's own Resize keeps its name and output name,
    so the rest of the graph is unchanged). Returns the number of Resizes."""
    graph = model.graph
    nodes: list[onnx.NodeProto] = []
    count = 0
    for node in graph.node:
        if node.op_type != "Resize":
            nodes.append(node)
            continue
        count += 1
        source, result = node.input[0], node.output[0]
        wide_in, wide_out = f"{node.name}_fp32_in", f"{node.name}_fp32_out"
        nodes.append(
            helper.make_node(
                "Cast",
                [source],
                [wide_in],
                name=f"{node.name}_to_fp32",
                to=TensorProto.FLOAT,
            )
        )
        node.input[0] = wide_in
        node.output[0] = wide_out
        nodes.append(node)
        nodes.append(
            helper.make_node(
                "Cast",
                [wide_out],
                [result],
                name=f"{node.name}_to_fp16",
                to=TensorProto.FLOAT16,
            )
        )
    del graph.node[:]
    graph.node.extend(nodes)
    # The FP16 types recorded for the Resizes' outputs no longer hold.
    del graph.value_info[:]
    return count


def gates(model: RSR) -> tuple[int, int]:
    """The context gates the prefix runs (before the attention stage) and those each
    core runs (from it on): each upsamples its context map with one Resize."""
    before = sum(position < model.xsa_position for position in model.gate_at)
    return before, len(model.gate_at) - before


def part(
    module: nn.Module,
    inputs: dict[str, torch.Tensor],
    outputs: tuple[str, ...],
    resizes: int,
) -> onnx.ModelProto:
    """`module` traced on `inputs` (name -> example, in argument order) at their static
    shapes, with its outputs named `outputs` and its `resizes` Resizes (one per context
    gate it runs; a part may have none) in FP32."""
    with tempfile.TemporaryDirectory(prefix="chaiNNer-rsr-part-") as temporary:
        path = Path(temporary) / "part.onnx"
        torch.onnx.export(
            module,
            tuple(inputs.values()),
            str(path),
            opset_version=OPSET,
            input_names=list(inputs),
            output_names=list(outputs),
            dynamo=False,
            do_constant_folding=True,
            external_data=False,
        )
        model = onnx.load(str(path))
    # Another count means the exporter's graph is not the one the pass expects (a gate's
    # upsampling under another operator would stay in FP16).
    found = fp32_resize(model)
    if found != resizes:
        raise RuntimeError(
            f"RSR's export found {found} Resizes to run in FP32, not one per context"
            f" gate ({resizes})"
        )
    onnx.checker.check_model(model)
    return model


def parts(model: RSR, height: int, width: int, device: str) -> dict[str, Any]:
    """The three parts of the folded FP32 `model` for frames of height x width, traced
    in FP16 on `device`."""
    half = model.to(device).half()
    generator = torch.Generator().manual_seed(0)

    def example(*shape: int, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        return torch.rand(shape, generator=generator).to(device=device, dtype=dtype)

    x = example(1, 3, height, width, dtype=torch.float32)
    with torch.inference_mode():
        current6, c3, c4 = half.encode(x.half())
    state = example(*current6.shape)
    valid = torch.ones(1, dtype=torch.bool, device=device)
    encoded = {"current6": current6, "c3": c3, "c4": c4}
    before, after = gates(model)
    with torch.no_grad():
        return {
            "prefix": part(Prefix(half), {"x": x}, ("current6", "c3", "c4"), before),
            "revcore": part(
                RevCore(half),
                {**encoded, "state_future": state, "valid_future": valid},
                ("state",),
                after,
            ),
            "fwdcore": part(
                FwdCore(half),
                {
                    "x": x,
                    **encoded,
                    "state_past": state,
                    "valid_past": valid,
                    "state_future": state,
                    "valid_future": valid,
                },
                ("output", "state"),
                after,
            ),
        }


def main(
    weights: str, output: str, height: str, width: str, device: str = "cpu"
) -> None:
    import spandrel

    model = load(Path(weights))
    exported = parts(model, int(height), int(width), device)
    root = Path(output)
    manifest: dict[str, Any] = {
        "height": int(height),
        "width": int(width),
        "scale": model.scale,
        # The bundle's architecture (rdual_bundle.architecture)
        "architecture": "RSR",
        # The library that traced the parts, for the log of a failed build
        "spandrel": spandrel.__version__,
        "parts": {},
    }
    for name in PARTS:
        graph = exported[name]
        save(graph, root / name)
        manifest["parts"][name] = {
            "nodes": len(graph.graph.node),
            "inputs": [
                (i.name, helper.printable_type(i.type)) for i in graph.graph.input
            ],
            "outputs": [
                (o.name, helper.printable_type(o.type)) for o in graph.graph.output
            ],
        }
    (root / "export.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:])
