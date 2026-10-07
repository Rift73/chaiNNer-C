"""Tracked CPU-only benchmark workloads: no AI models, no GPU.

Image cases process 32 distinct 512x384 JPEGs derived from the user's dataset.
resize-real-256 replays the user's saved 50% Hermite resize chain on 256
original files. Video cases process a 600-frame 1280x720 test clip.

Adding a case:
  1. Add its name to CASES.
  2. Add a graph() branch that builds its nodes and returns
     (nodes, items per request, throughput unit).
  3. Extend ALLOWED with new node schemas only after they have been reviewed
     as CPU-only and deterministic.
  4. Keep exactly one load_images or load_video generator per graph; the
     completion oracle (bench_oracle.validate_record) checks that generator.
  5. Its shared throughput floor is the stand-in's decision: the harness never
     writes the shared floors, and until one is recorded its throughput verdict
     is "unknown". --record-noise records its CPU floors, per K setting, on
     trees with identical sources, using at least 6 pairs and an even number of
     repeats, for example (3 launches at each K setting used):
       bench.py --record-noise --cases NEW --repeats 6 --a TREE --b TREE
     Until then its CPU verdict is "unknown". Confirm the floors with a later A/A
     run without --record-noise (see bench.py), judged by the FIRST run's
     summary verdicts (all neutral), not by its confirmation.md.
"""

from __future__ import annotations

import json
from pathlib import Path

from verify_runtime import edge, make_node

DATA = Path(__file__).resolve().parent / "bench-data"
IMAGE_COUNT = 32
VIDEO_FRAMES = 600
CASES = (
    "resize-real-256",
    "gaussian",
    "lens-spectral",
    "morphology",
    "normal-map",
    "caption",
    "split-channels",
    "palette-median",
    "palette-dither",
    "parallel-branches",
    "video-ffv1",
    "video-h264",
    "video-resize-h264",
)
ALLOWED = frozenset(
    "chainner:image:" + name
    for name in (
        "load_images",
        "save",
        "resize",
        "gaussian_blur",
        "lens_blur",
        "dilate",
        "erode",
        "normal_generator",
        "caption",
        "split_channels",
        "palette_from_image",
        "palette_dither",
        "blend",
        "load_video",
        "save_video",
    )
)
OPERATIONS = {
    "gaussian": ("gaussian_blur", {1: 2.5, 2: 1.25}),
    "lens-spectral": ("lens_blur", {1: 30, 2: 3, 3: 3}),
    "normal-map": ("normal_generator", {5: "multi-gauss"}),
    "caption": ("caption", {1: "CPU preparation parity", 2: 24, 3: "bottom"}),
    "split-channels": ("split_channels", {}),
    "palette-median": ("palette_from_image", {1: "median", 2: 16}),
}


def graph(
    schemas: dict, case: str, assets: Path, output: Path
) -> tuple[list[dict], int, str]:
    """Return (nodes, items per request, throughput unit) for one case."""

    def node(identifier: str, operation: str, values: dict) -> dict:
        return make_node(schemas, identifier, "chainner:image:" + operation, values)

    if case == "resize-real-256":
        request = json.loads(
            (DATA / "resize-real-256.json").read_text(encoding="utf-8")
        )
        nodes = request["data"]
        save = next(n for n in nodes if n["schemaId"] == "chainner:image:save")
        save["inputs"][1]["value"] = str(output)
        return _allowed(nodes), 256, "images/s"
    if case.startswith("video-"):
        nodes = [node("load", "load_video", {0: str(assets / "moving.mkv"), 1: 0})]
        source = edge("load")
        if case == "video-resize-h264":
            nodes.append(
                node("resize", "resize", {0: source, 1: 0, 2: 50, 5: 5, 6: False})
            )
            source = edge("resize")
        params: dict[int, object] = {
            0: source,
            1: str(output),
            2: "video",
            14: edge("load", 4),
            15: None,
        }
        if case == "video-ffv1":
            params.update({16: 1, 4: "mkv", 3: "ffv1", 13: "-threads 8 -pix_fmt bgr0"})
        elif case in {"video-h264", "video-resize-h264"}:
            params.update({16: 0, 17: "mp4_h264", 18: 75})
        else:
            raise ValueError(case)
        nodes.append(node("save", "save_video", params))
        return _allowed(nodes), VIDEO_FRAMES, "frames/s"
    nodes = [node("load", "load_images", {0: str(assets / "images"), 2: 0, 4: 0, 6: 1})]
    source = edge("load")
    if case in OPERATIONS:
        operation, options = OPERATIONS[case]
        nodes.append(node("operation", operation, {0: source, **options}))
        source = edge("operation")
    elif case == "morphology":
        nodes += [
            node("dilate", "dilate", {0: source, 1: 2, 2: 3, 3: 2}),
            node("erode", "erode", {0: edge("dilate"), 1: 1, 2: 2, 3: 2}),
        ]
        source = edge("erode")
    elif case == "palette-dither":
        nodes += [
            node("palette", "palette_from_image", {0: source, 1: "median", 2: 16}),
            node(
                "dither",
                "palette_dither",
                {0: source, 1: edge("palette"), 2: "Diffusion", 3: "FS"},
            ),
        ]
        source = edge("dither")
    elif case == "parallel-branches":
        nodes += [
            node("blur1", "gaussian_blur", {0: source, 1: 3, 2: 3}),
            node("blur2", "gaussian_blur", {0: source, 1: 6, 2: 6}),
            node("blend", "blend", {0: edge("blur1"), 1: edge("blur2"), 2: 1}),
        ]
        source = edge("blend")
    else:
        raise ValueError(case)
    save = {0: source, 1: str(output), 2: None, 3: edge("load", 3), 4: "png", 15: "u8"}
    nodes.append(node("save", "save", save))
    return _allowed(nodes), IMAGE_COUNT, "images/s"


def _allowed(nodes: list[dict]) -> list[dict]:
    foreign = {n["schemaId"] for n in nodes} - ALLOWED
    if foreign:
        raise ValueError(f"Unreviewed benchmark nodes: {sorted(foreign)}")
    return nodes
