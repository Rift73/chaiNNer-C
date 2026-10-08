"""Explicit offline Triton cubin generation; no GPU query or kernel launch.

Generate headers on a supported Triton host (normally WSL/Linux); the same
headers can be compiled into a Windows DLL or Linux SO for the SAME CUDA SM.
This is compilation, not numerical validation.
"""

import argparse
import json
from pathlib import Path
import re

from .spec import ROOT, SOURCE_SHA256, sha256


def generate(export_dir, output, sm):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from . import kernels

    export_dir, output = Path(export_dir).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((export_dir / "export.json").read_text(encoding="utf-8"))
    if manifest["source_sha256"] != SOURCE_SHA256 or type(sm) is not int or sm < 80:
        raise ValueError("Unsupported source identity or pre-BF16 CUDA target")
    spec = manifest["specialization"]
    c = spec["channels"]
    h, w = spec["trunk"]
    counts = [
        ((h + sy + 31) // 32) * ((w + sx + 31) // 32)
        for sy, sx in ((0, 0), (0, 16), (16, 0), (16, 16))
    ]
    common = {"H": h, "W": w, "C": c}
    regions = {**common, **dict(zip(("N0", "N1", "N2"), counts[:3], strict=True))}
    descriptors = [
        (
            kernels.cell_stats,
            {"X": "*bf16", "WEIGHT": "*bf16", "STATS": "*fp32"},
            {**common, "BT": 256},
        ),
        (
            kernels.region_stats,
            {"STATS": "*fp32", "TAU": "*fp32", "A": "*bf16"},
            regions,
        ),
        (
            kernels.apply_dw,
            {"X": "*bf16", "WEIGHT": "*bf16", "A": "*bf16", "OUT": "*bf16"},
            {**regions, "BT": 256},
        ),
        (
            kernels.project_add,
            {"X": "*bf16", "WEIGHT": "*bf16", "RESIDUAL": "*bf16", "OUT": "*bf16"},
            {"PIXELS": h * w, "C": c, "BC": 1 << (c - 1).bit_length(), "BM": 64},
        ),
    ]
    compiled = []
    for fn, pointers, constants in descriptors:
        signature = {**pointers, **{name: "constexpr" for name in constants}}
        kernel = triton.compile(
            ASTSource(fn=fn, signature=signature, constexprs=constants),
            target=GPUTarget("cuda", sm, 32),
            options={"num_warps": 4, "num_stages": 1, "enable_fp_fusion": False},
        )
        metadata = kernel.metadata
        if metadata.global_scratch_size or metadata.profile_scratch_size:
            raise RuntimeError(
                "Nonzero Triton scratch memory is not supported by this plugin ABI"
            )
        # The driver wrapper passes tensor pointers plus two null scratch pointers.
        # Fail on a changed ABI rather than silently launching with wrong args.
        entry = re.search(r"\.entry\s+\w+\s*\((.*?)\)", kernel.asm["ptx"], re.S)
        params = (
            [] if entry is None else re.findall(r"\.param\s+(\.\w+)", entry.group(1))
        )
        if params != [".u64"] * (len(pointers) + 2):
            raise RuntimeError(
                f"Unrecognized Triton launch ABI for {fn.__name__}: {params}"
            )
        compiled.append(kernel)
    output.mkdir(parents=True, exist_ok=False)
    core = "constexpr int APPLY_BT=256;\nstruct Blob { int warps,shared; const char* name; const unsigned char* data; };\n"
    entries = []
    for index, kernel in enumerate(compiled[:3]):
        meta = kernel.metadata
        core += (
            f"alignas(16) static const unsigned char b{index}[]={{"
            + ",".join(map(str, kernel.asm["cubin"]))
            + "};\n"
        )
        entries.append(f'{{{meta.num_warps},{meta.shared},"{meta.name}",b{index}}}')
    core += "static const Blob blobs[]={" + ",".join(entries) + "};\n"
    kernel = compiled[3]
    meta = kernel.metadata
    project = (
        "constexpr int PROJECT_BM=64;\nalignas(16) static const unsigned char project_blob[]={"
        + ",".join(map(str, kernel.asm["cubin"]))
        + "};\n"
    )
    project += f'const Blob project_kernel={{{meta.num_warps},{meta.shared},"{meta.name}",project_blob}};\n'
    (output / "Core.h").write_text(core, encoding="utf-8")
    (output / "Project.h").write_text(project, encoding="utf-8")
    receipt = {
        "schema": 1,
        "status": "compiled_unvalidated",
        "sm": sm,
        "plugin_key": spec["plugin_key"],
        "triton": triton.__version__,
        "source_sha256": SOURCE_SHA256,
        "kernels_sha256": sha256(ROOT / "scripts/dual_tensorrt/kernels.py"),
        "files": {name: sha256(output / name) for name in ("Core.h", "Project.h")},
    }
    (output / "aot.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--sm",
        type=int,
        required=True,
        help="Explicit CUDA SM, e.g. 120; never auto-detected",
    )
    print(json.dumps(generate(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
