"""Owned CPU HTTP/SSE resize checks and coarse whole-batch throughput.

No user chain is executed. Inputs are copied from the explicitly supplied folder;
outputs/profiles/processes belong to the report. Durations are rounded to whole
seconds, never precise kernel benchmarks. PNG comparisons use decoded pixels
because the selected lossless encoder deliberately changes compressed bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import verify_framework_runtime as runtime
from verify_runtime import PROJECT, edge, make_node, write_json

SCHEMAS = {"chainner:image:load_images", "chainner:image:resize", "chainner:image:save"}


def prepare(root: Path, source: Path, package: Path) -> None:
    root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(
        package / "resources/src",
        root / "before-src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    candidates = sorted(
        p
        for p in source.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )[:32]
    if not candidates:
        raise ValueError("No supported input images in the selected folder")
    inputs = root / "fixtures"
    (inputs / "filters").mkdir(parents=True)
    (inputs / "batch").mkdir()
    manifest = []
    for index, path in enumerate(candidates):
        data = path.read_bytes()
        manifest.append({"path": str(path), "sha256": hashlib.sha256(data).hexdigest()})
        # Twelve varied files per filter; all 32 inputs repeated for a useful batch.
        if index < 12:
            (inputs / "filters" / f"image{index:03}{path.suffix}").write_bytes(data)
        for repeat in range(8):
            (
                inputs
                / "batch"
                / f"image{repeat * len(candidates) + index:03}{path.suffix}"
            ).write_bytes(data)
    write_json(root / "inputs.json", manifest)
    write_json(
        root / "baseline-provenance.json", runtime.source_hashes(root / "before-src")
    )


def graphs(schemas: dict, assets: Path, output: Path, *, batch_only: bool = False):
    # Include Auto and Nearest as well as every filtered resize, in both directions.
    filters = schemas["chainner:image:resize"]["inputs"][5]["options"]
    for option in () if batch_only else filters:
        value = option["value"]
        for scale in (50, 150):
            name = f"filter-{value}-scale-{scale}"
            yield graph(schemas, assets / "filters", output / name, name, value, scale)
    yield graph(schemas, assets / "batch", output / "batch", "batch", 5, 50)


def graph(
    schemas: dict, assets: Path, output: Path, name: str, filter_id: int, scale: int
):
    load = make_node(
        schemas, "load", "chainner:image:load_images", {0: str(assets), 2: True}
    )
    resize = make_node(
        schemas,
        "resize",
        "chainner:image:resize",
        {0: edge("load", 0), 1: 0, 2: scale, 5: filter_id},
    )
    save = make_node(
        schemas,
        "save",
        "chainner:image:save",
        {
            0: edge("resize", 0),
            1: str(output),
            2: edge("load", 2),
            3: edge("load", 3),
            4: "png",
            15: "u8",
        },
    )
    return {
        "name": name,
        "nodes": [load, resize, save],
        "expected_error": False,
        "files": sorted(p.stem + ".png" for p in assets.iterdir()),
    }


def provenance(backend: Path) -> dict[str, str]:
    result = runtime.source_hashes(backend)
    for path in sorted(backend.rglob("*")):
        if path.is_file() and path.suffix in {".dll", ".pyd"}:
            result[path.relative_to(backend).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return result


def run(
    root: Path, label: str, backend: Path, package: Path, *, batch_only: bool = False
) -> None:
    samples = []
    original_request = runtime.request
    before = provenance(backend)
    write_json(root / f"{label}-provenance.json", before)

    def coarse_request(
        port: int, path: str, data: dict | None = None, timeout: float = 15
    ):
        if path != "/run":
            return original_request(port, path, data, timeout)
        assert data is not None
        start = time.monotonic()
        result = original_request(port, path, data, max(timeout, 300))
        seconds = math.ceil(time.monotonic() - start)
        samples.append(
            {
                "case": Path(data["data"][-1]["inputs"][1]["value"]).name,
                "whole_seconds_rounded_up": seconds,
            }
        )
        return result

    runtime.request = coarse_request
    try:
        result = runtime.run_backend(
            label,
            backend,
            package / "python/python/python.exe",
            root,
            180,
            required_schema_ids=SCHEMAS,
            graphs=lambda s, a, o: graphs(s, a, o, batch_only=batch_only),
            synthetic_model_inference=False,
        )
    finally:
        runtime.request = original_request
        write_json(root / f"{label}-coarse.json", samples)
    assert provenance(backend) == before, "Backend changed during validation"
    write_json(root / f"{label}-result.json", result)


def compare(root: Path, before: str, after: str) -> None:
    import cv2
    import numpy as np

    a, b = root / before / "output", root / after / "output"
    inventories = []
    for label, directory in ((before, a), (after, b)):
        result = json.loads((root / f"{label}-result.json").read_text())
        assert result["success"] and result["fixtures"], f"Incomplete run: {label}"
        expected = {
            Path(case["name"]) / name: digest
            for case in result["fixtures"]
            for name, digest in case["contract"]["files"].items()
        }
        assert expected, f"Empty run: {label}"
        actual = {p.relative_to(directory) for p in directory.rglob("*") if p.is_file()}
        assert actual == expected.keys(), f"Incomplete files: {label}"
        for relative, digest in expected.items():
            assert (
                hashlib.sha256((directory / relative).read_bytes()).hexdigest()
                == digest
            ), str(relative)
        inventories.append(set(expected))
    assert inventories[0] == inventories[1], "Different workloads"
    paths = sorted(p.relative_to(a) for p in a.rglob("*.png"))
    assert len(paths) == len(inventories[0])
    assert paths == sorted(p.relative_to(b) for p in b.rglob("*.png"))
    byte_differences = 0
    for relative in paths:
        left, right = (a / relative).read_bytes(), (b / relative).read_bytes()
        byte_differences += left != right
        x = cv2.imdecode(np.frombuffer(left, np.uint8), cv2.IMREAD_UNCHANGED)
        y = cv2.imdecode(np.frombuffer(right, np.uint8), cv2.IMREAD_UNCHANGED)
        assert x is not None and y is not None
        assert x.dtype == y.dtype, str(relative)
        np.testing.assert_array_equal(x, y, err_msg=str(relative))
    write_json(
        root / "comparison.json",
        {
            "decoded_pixel_exact": len(paths),
            "different_compressed_bytes": byte_differences,
            "success": True,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "compare"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--package", type=Path, default=PROJECT / "out/chaiNNer-C")
    parser.add_argument("--label", default="before")
    parser.add_argument("--backend", type=Path)
    parser.add_argument("--batch-only", action="store_true")
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.package = args.package.resolve()
    if args.backend is not None:
        args.backend = args.backend.resolve()
    if args.mode == "prepare":
        if args.source is None:
            parser.error("prepare requires --source")
        prepare(args.root, args.source, args.package)
    elif args.mode == "run":
        run(
            args.root,
            args.label,
            args.backend or args.root / "before-src",
            args.package,
            batch_only=args.batch_only,
        )
    else:
        compare(args.root, "before", args.label)


if __name__ == "__main__":
    main()
