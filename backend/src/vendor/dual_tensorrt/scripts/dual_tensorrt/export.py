"""CPU-only checkpoint -> TensorRT-specific ONNX. Run in an isolated worker."""

import argparse
import builtins
import json
from pathlib import Path
import sys
import types

from .spec import FACTORIES, SCHEMA, SOURCE, SOURCE_SHA256, from_model, sha256


def load_arch():
    """Execute canonical source unchanged, with its optional dependencies absent.

    A private builtins mapping suppresses ONLY optional direct Triton attention
    and trainer registration. It does not monkeypatch the caller's import system,
    import other architectures, or alter DUAL's model implementation.
    """
    if sha256(SOURCE) != SOURCE_SHA256:
        raise RuntimeError(
            "Canonical dual_arch.py changed: review the lowering before exporting"
        )
    name = "_dual_tensorrt_canonical_source"
    if name in sys.modules:
        return sys.modules[name]
    module = types.ModuleType(name)
    module.__file__ = str(SOURCE)
    original_import = builtins.__import__

    def source_import(name, *args, **kwargs):
        if name in ("triton", "triton.language", "traiNNer.utils.registry"):
            raise ModuleNotFoundError(
                "Optional dependency disabled in export worker", name=name
            )
        return original_import(name, *args, **kwargs)

    module.__dict__["__builtins__"] = {**vars(builtins), "__import__": source_import}
    sys.modules[name] = module
    try:
        exec(
            compile(SOURCE.read_text(encoding="utf-8"), str(SOURCE), "exec"),
            module.__dict__,
        )
    except BaseException:
        del sys.modules[name]
        raise
    return module


def make_model(factory, scale, *, meta=False):
    import torch

    arch = load_arch()
    with torch.device("meta" if meta else "cpu"):
        model = getattr(arch, factory)(
            scale=scale,
            use_checkpoint=False,
            dual_core_checkpoint=False,
            g_attention_impl="sdpa",
            compile_g_attention_impl="sdpa",
        )
    return arch, model.eval().requires_grad_(False)


def read_state(path, key):
    import torch

    if path.suffix.lower() == ".safetensors":
        if key:
            raise ValueError(
                "safetensors contains a direct state dict: omit --state-key"
            )
        from safetensors.torch import load_file

        state = load_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if key:
            state = state[key]
    if (
        not isinstance(state, dict)
        or not state
        or not all(
            isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()
        )
    ):
        raise ValueError(
            "Expected a tensor state dict; select --state-key explicitly for wrapped checkpoints"
        )
    if any(
        v.is_floating_point() and not torch.isfinite(v).all().item()
        for v in state.values()
    ):
        raise ValueError("Checkpoint contains nonfinite weights")
    return state


def export_checkpoint(
    factory, scale, height, width, checkpoint, output, *, state_key=None, folded=False
):
    import onnx
    import torch
    from .graph import Graph

    # No random/synthetic checkpoint fallback, prefix stripping, or strict=False.
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite export directory: {output}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha = sha256(checkpoint)
    torch.set_num_threads(1)
    arch, model = make_model(factory, scale)
    spec = from_model(model, factory, scale, height, width)
    if folded:
        model.prepare_for_export()
    model.load_state_dict(read_state(checkpoint, state_key), strict=True)
    model.prepare_for_export()
    if sha256(checkpoint) != checkpoint_sha:
        raise RuntimeError("Checkpoint changed during export")
    graph = Graph(arch, spec)
    with torch.inference_mode():
        onnx_model = graph.build(model)
    onnx.checker.check_model(onnx_model)
    output.mkdir(parents=True, exist_ok=False)
    # External data avoids protobuf's 2GiB limit and keeps XL viable.
    onnx.save_model(
        onnx_model,
        str(output / "model.onnx"),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.bin",
        size_threshold=1024,
    )
    onnx.checker.check_model(str(output / "model.onnx"))
    manifest = {
        "schema": SCHEMA,
        "status": "exported_unvalidated",
        "source_sha256": SOURCE_SHA256,
        "checkpoint_sha256": checkpoint_sha,
        "state_key": state_key,
        "checkpoint_folded": folded,
        "specialization": spec.to_dict(),
        "precision": "FP32_IO_BF16_FP32_BODY",
        "dynamic_sites": graph.dynamic_sites,
        "plugin_names": [
            f"Dual{part}_{spec.plugin_key}_TRT" for part in ("Norm", "Core", "Project")
        ]
        + ["DualAttentionBarrier_TRT"],
        "files": {
            p.name: sha256(p)
            for p in (output / "model.onnx", output / "weights.bin")
            if p.exists()
        },
        "validation": "ONNX structural checker only; custom plugin semantics and runtime NOT validated",
    }
    (output / "export.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory", choices=FACTORIES, required=True)
    parser.add_argument("--scale", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--state-key")
    parser.add_argument("--folded", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_checkpoint(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
