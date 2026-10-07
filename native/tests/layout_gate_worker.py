"""One backend tree's side of the layout gate (test_layout_gate.py, D-7 condition 4).

Run as ``python -B layout_gate_worker.py ROOT OUT [--layouts L,...] [--schemas S,...]
[--trace FILE ...] [--mixed] [--chains [--producers S,...]]``. ROOT is a backend
source tree: chaiNNer-C's backend/src, or the oracle's src (make_oracle.py). Both
trees use the same package names, so each runs in its own process.

Every chaiNNer_standard node with an image input is run on cases built from that
node's own definition:
- each image input gets a 3- or a 4-channel image, or the count it allows when it
  doesn't take that one;
- every other input gets its default or the value in OVERRIDES;
- besides the default case, one case per other option of each dropdown, with the
  rest left at their defaults.
A case exists for a channel count only when at least one image input takes it.

Images are uniform float32 noise, seeded per case and input, so both trees build
the same values. Each image reaches the node as it would in the UI: as the result
of ImageOutput.enforce on the image in one of two layouts.
- "permuted": a planar (c, h, w) array's transpose(1, 2, 0), Lens's layout.
- "c": the same values, C-contiguous.
With --mixed, a case with more than one image (a collector's iterated images
included) also runs on two mixed layouts (MIXED), where NumPy's fallback decides
its result's layout (Consult 11 D-16).

OUT receives results.json (and the files Save nodes write). For each run it holds
either the error, or:
- the outputs, with each array's dtype, shape, strides and the SHA-256 of its
  C-ordered bytes;
- the outputs' broadcast data, as the executor would send it;
- the SHA-256 of each file a Save node wrote.
With --trace, a run also records whether it entered any of the given source files.
With --chains, it also runs the UI chain of Lens followed by each of CHAIN_TARGETS,
with the Lens output as the target's first image; with --producers, also Lens
followed by each first-seed case of those nodes, the Lens output as the case's
first image, followed by each first-seed case of CHAIN_TARGETS for the producer's
channel count, the producer's output as the target's first image (D-16).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import threading
import traceback
import zlib
from pathlib import Path
from typing import Any

import numpy as np

HEIGHT, WIDTH = 72, 120
CHANNELS = (3, 4)
LAYOUTS = ("permuted", "c")
# "first": only the first image permuted, the rest C; "rest": the first C, the
# rest permuted. A collector's images count in the order they are iterated.
MIXED = ("first", "rest")
PACKAGE = "packages.chaiNNer_standard"
LENS = "chainner:image:lens_blur"
CHAIN_TARGETS = ("chainner:image:color_transfer", "chainner:image:stretch_contrast")
# Nodes the gate can't run here, with the reason; test_layout_gate.py lists them too.
EXCLUDED = {
    "chainner:image:save_video": "needs an FFmpeg environment from a node context; "
    "its frames are to_uint8's result, written as C-order bytes "
    "(video_io.cpp frame_payload: a memoryview only of a C-contiguous frame, else "
    "tobytes)",
}
# Non-image inputs by label; "@save" is the run's own output directory.
OVERRIDES: dict[str, dict[str, object]] = {
    "chainner:image:add": {"Add": 12},
    "chainner:image:brightness_and_contrast": {"Brightness": 15, "Contrast": 20},
    "chainner:image:caption": {"Caption": "chaiNNer layout"},
    # Without confusion, noise around the default key (gray 0.5) gives the trimap
    # about 13 % background, 36 % foreground and the rest unknown, so trimap
    # matting runs its solvers.
    "chainner:image:chroma_key": {
        "Threshold BG": 20,
        "Threshold FG": 35,
        "Confusion BG": 0,
        "Confusion FG": 0,
    },
    "chainner:image:clamp": {"Minimum": 0.2, "Maximum": 0.7},
    "chainner:image:color_levels": {"Gamma": 1.4, "In Black": 0.1, "Out White": 0.9},
    "chainner:image:crop": {
        "Amount": 5,
        "Left": 3,
        "Top": 4,
        "Right": 5,
        "Bottom": 6,
        "Width": 50,
        "Height": 40,
    },
    "chainner:image:divide": {"Divide": 1.7},
    "chainner:image:gamma": {"Gamma": 1.6},
    "chainner:image:hue_and_saturation": {
        "Hue": 30,
        "Saturation": 20,
        "Lightness": 10,
    },
    "chainner:image:image_convolve": {"Kernel String": "0 -1 0\n-1 5 -1\n0 -1 0"},
    "chainner:image:merge_spritesheet": {"Number of columns (width)": 2},
    "chainner:image:multiply": {"Multiply": 1.3},
    "chainner:image:opacity": {"Opacity": 60},
    "chainner:image:pad": {
        "Amount": 5,
        "Left": 3,
        "Top": 4,
        "Right": 5,
        "Bottom": 6,
        "Width": 130,
        "Height": 80,
    },
    "chainner:image:pick_color": {"X": 7, "Y": 5},
    "chainner:image:resize": {"Percentage": 55.0, "Width": 50, "Height": 30},
    "chainner:image:resize_to_side": {"Size Target": 60},
    "chainner:image:rotate": {"Rotation Angle": 30},
    "chainner:image:save": {"Directory": "@save", "Image Name": "gate"},
    "chainner:image:shift": {"X": 7, "Y": -5},
    "chainner:image:split_spritesheet": {
        "Number of rows (height)": 2,
        "Number of columns (width)": 2,
    },
}
# Cases beyond the dropdown ones, as named label overrides.
EXTRA_CASES: dict[str, dict[str, dict[str, object]]] = {
    # A threshold above 0 takes native_analysis.binary instead of weighted.
    "chainner:image:sharpen": {"Threshold=10": {"Threshold": 10}},
    # UNROUNDED's case.
    "chainner:image:palette_from_image": {
        "All colors, unrounded": {"Palette Extraction Method": "all"}
    },
}
# Image inputs whose size differs from HEIGHT x WIDTH, by label.
SIZES: dict[str, dict[str, tuple[int, int]]] = {
    "chainner:image:average_color_fix": {"Reference Image": (36, 60)},
    "chainner:image:lut": {"Palette": (1, 16)},
    "chainner:image:palette_dither": {"Palette": (1, 8)},
    "chainner:image:quantize_to_referece": {"Reference Image": (36, 60)},
}
# Optional image inputs that get an image, by label: Merge Channels' second image
# gives its mixed layouts something to disagree with.
FILLED = {
    "chainner:image:high_pass": {"Blurred Image"},
    "chainner:image:merge_channels": {"Channel(s) B"},
}
# Image inputs with a fixed channel count, by label: High Pass blurs only the
# colour channels of a 4-channel image, so its custom blurred image has 3.
FIXED_CHANNELS = {"chainner:image:high_pass": {"Blurred Image": 3}}
# Nodes whose images are rounded to this many steps per unit, -0 kept. So the
# "All colors" palette sees at most 6**4 values (+0 and -0 apart), within its 4096
# colours, and signed-zero ties: np.unique keeps whichever of +0 and -0 its unstable
# sort puts first, which the port mirrors (Consult 11 D-17.2).
LEVELS = {"chainner:image:palette_from_image": 4}
# The one case of a LEVELS node whose images stay unrounded: noise has more distinct
# colours than "All colors" allows (MAX_COLORS, 4096), so its ValueError, type and
# message with the count, is what both trees give.
UNROUNDED = {"chainner:image:palette_from_image": "All colors, unrounded"}
# Each case runs on this many seeds: whether the two layouts resolve a signed-zero
# tie differently depends on the values, so one seed may show none.
SEEDS = 3
# The number of images a collector's iterated image input receives.
ITERATIONS = 2


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--layouts", default=",".join(LAYOUTS))
    parser.add_argument("--schemas", default="")
    parser.add_argument("--trace", nargs="*", default=[])
    parser.add_argument("--mixed", action="store_true")
    parser.add_argument("--chains", action="store_true")
    parser.add_argument("--producers", default="")
    return parser.parse_args()


class Trace:
    """A profile function that notes whether any frame of the given files ran."""

    def __init__(self, files: list[str]) -> None:
        self.files = {os.path.normcase(str(Path(f).resolve())) for f in files}
        self.hit = False

    def __call__(self, frame: Any, event: str, _arg: object) -> None:
        if (
            event == "call"
            and not self.hit
            and os.path.normcase(frame.f_code.co_filename) in self.files
        ):
            self.hit = True


class Tree:
    """One backend tree's registry, process module and image enforcement."""

    def __init__(self, root: Path) -> None:
        sys.path.insert(0, str(root))
        api = importlib.import_module("api")
        self.process = importlib.import_module("process")
        importlib.import_module(PACKAGE)
        errors = api.registry.load_nodes(str(root / "server.py"))
        if errors:
            raise RuntimeError(f"Node load errors: {[e.module for e in errors]}")
        self.nodes = {schema: node for schema, (node, _) in api.registry.nodes.items()}
        inputs = importlib.import_module("nodes.properties.inputs")
        outputs = importlib.import_module("nodes.properties.outputs")
        self.image_input = inputs.ImageInput
        self.dropdown = inputs.DropDownInput
        self.enforce = outputs.ImageOutput().enforce


def allowed_channels(image_input: Any) -> list[int] | None:
    channels = image_input.channels
    return [channels] if isinstance(channels, int) else channels


def image_indices(tree: Tree, node: Any) -> list[int]:
    """The image inputs a case fills: the required ones and those in FILLED."""
    filled = FILLED.get(node.schema_id, set())
    return [
        index
        for index, i in enumerate(node.inputs)
        if isinstance(i, tree.image_input) and (not i.optional or i.label in filled)
    ]


def case_channels(tree: Tree, node: Any, count: int) -> dict[int, int] | None:
    """Each filled image input's channel count, or None if none takes count."""
    fixed = FIXED_CHANNELS.get(node.schema_id, {})
    chosen = {}
    for index in image_indices(tree, node):
        allowed = allowed_channels(node.inputs[index])
        chosen[index] = fixed.get(
            node.inputs[index].label,
            count if allowed is None or count in allowed else max(allowed or [count]),
        )
    return chosen if count in chosen.values() else None


def variants(tree: Tree, node: Any) -> list[tuple[str, dict[int, object]]]:
    """The default case, each other option of each dropdown on its own, EXTRA_CASES."""
    found: list[tuple[str, dict[int, object]]] = [("default", {})]
    for index, i in enumerate(node.inputs):
        if isinstance(i, tree.dropdown):
            found.extend(
                (f"{i.label}={option['value']}", {index: option["value"]})
                for option in i.options
                if option["value"] != i.default
            )
    for name, values in EXTRA_CASES.get(node.schema_id, {}).items():
        found.append(
            (
                name,
                {
                    index: values[i.label]
                    for index, i in enumerate(node.inputs)
                    if i.label in values
                },
            )
        )
    return found


def cases(tree: Tree, schemas: list[str]) -> dict[str, dict[str, Any]]:
    found = {}
    for schema in schemas:
        node = tree.nodes[schema]
        for count in CHANNELS:
            chosen = case_channels(tree, node, count)
            if chosen is None:
                continue
            for name, values in variants(tree, node):
                unrounded = UNROUNDED.get(schema) == name
                for seed in range(SEEDS):
                    found[f"{schema}|{count}ch|{name}|s{seed}"] = {
                        "schema": schema,
                        "count": count,
                        "channels": chosen,
                        "values": values,
                        "levels": None if unrounded else LEVELS.get(schema),
                    }
    return found


def noise(
    case_id: str, index: int, shape: tuple[int, int, int], levels: int | None
) -> np.ndarray:
    """A planar (c, h, w) float32 image seeded by the case and input.

    A tenth of the values are +0, a tenth -0 and a twentieth 1: the extremes'
    signed-zero ties are what a reduction's visiting order shows (g-report,
    "Held for Consult 9"), so they must be present for the layout to matter.
    """
    rng = np.random.default_rng(zlib.crc32(f"{case_id}#{index}".encode()))
    image = rng.random(shape, dtype=np.float32)
    pick = rng.random(shape)
    image[pick < 0.1] = 0.0
    image[(pick >= 0.1) & (pick < 0.2)] = -0.0
    image[pick >= 0.95] = 1.0
    if levels is None:
        return image
    return np.round(image * levels) / np.float32(levels)


def staged(tree: Tree, planar: np.ndarray, layout: str, position: int) -> np.ndarray:
    """The enforced image a node receives as its image number position from a node
    that returned this layout (a MIXED layout's for that image)."""
    if layout in MIXED:
        layout = "permuted" if (position == 0) == (layout == "first") else "c"
    image = planar.transpose(1, 2, 0)
    if layout == "c":
        image = np.ascontiguousarray(image)
    return tree.enforce(image)


def mixed(tree: Tree, case: dict[str, Any]) -> bool:
    """Whether the case has more than one image: MIXED layouts can differ."""
    node = tree.nodes[case["schema"]]
    return len(case["channels"]) > 1 or (node.kind == "collector" and ITERATIONS > 1)


def inputs_for(
    tree: Tree, case_id: str, case: dict[str, Any], layout: str, save: Path
) -> tuple[list[object], list[np.ndarray]]:
    """The node's input list and, for a collector, the images it iterates."""
    node = tree.nodes[case["schema"]]
    overrides = OVERRIDES.get(case["schema"], {})
    sizes = SIZES.get(case["schema"], {})
    levels = case["levels"]
    iterated = (
        set(node.single_iterable_input.inputs) if node.kind == "collector" else set()
    )
    values: list[object] = []
    tiles: list[np.ndarray] = []
    position = 0
    for index, i in enumerate(node.inputs):
        if index in case["channels"]:
            height, width = sizes.get(i.label, (HEIGHT, WIDTH))
            shape = (case["channels"][index], height, width)
            if i.id in iterated:
                tiles = [
                    staged(
                        tree,
                        noise(case_id, index * 100 + n, shape, levels),
                        layout,
                        position + n,
                    )
                    for n in range(ITERATIONS)
                ]
                values.append(None)
            else:
                values.append(
                    staged(tree, noise(case_id, index, shape, levels), layout, position)
                )
            position += 1
        elif index in case["values"]:
            values.append(case["values"][index])
        elif i.label in overrides:
            value = overrides[i.label]
            values.append(str(save) if value == "@save" else value)
        elif isinstance(i, tree.image_input):
            values.append(None)
        else:
            values.append(i.default)
    return values, tiles


def encode(value: object) -> object:
    """A JSON form that keeps every bit: arrays as the SHA-256 of their C-ordered
    bytes plus dtype, shape and strides, floats as hex."""
    if isinstance(value, np.ndarray):
        return {
            "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "strides": list(value.strides),
        }
    if isinstance(value, np.generic):
        return {"numpy": str(value.dtype), "bytes": value.tobytes().hex()}
    if isinstance(value, float):
        return {"float": value.hex()}
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    fields = getattr(value, "__dict__", None)
    return {
        "type": type(value).__name__,
        "fields": encode(fields if fields is not None else repr(value)),
    }


def execute(
    tree: Tree, schema: str, values: list[object], tiles: list[np.ndarray]
) -> tuple[list[object], list[object]]:
    """The node's enforced outputs and their broadcast data, as the executor makes them."""
    node = tree.nodes[schema]
    result = tree.process.run_node(node, None, values, "layout-gate")
    if node.kind == "collector":
        index = next(
            n
            for n, i in enumerate(node.inputs)
            if i.id in node.single_iterable_input.inputs
        )
        for tile in tiles:
            result.collector.on_iterate(node.inputs[index].enforce_(tile))
        outputs = tree.process.enforce_output(
            result.collector.on_complete(), node
        ).output
    elif node.kind == "generator":
        iterated = node.single_iterable_output.outputs
        items = []
        for item in result.generator.supplier():
            if isinstance(item, Exception):
                raise item
            pending = [item] if len(iterated) == 1 else list(item)
            items.append(
                [o.enforce(pending.pop(0)) for o in node.outputs if o.id in iterated]
            )
        return [items, result.partial_output], []
    else:
        outputs = result.output
    broadcasts: list[object] = [
        None if value is None else o.get_broadcast_data(value)
        for o, value in zip(node.outputs, outputs, strict=True)
    ]
    return outputs, broadcasts


def saved_files(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def run(
    tree: Tree,
    schema: str,
    values: list[object],
    tiles: list[np.ndarray],
    *,
    save: Path,
    trace: list[str],
    outputs: list[object] | None = None,
) -> dict[str, object]:
    """One run's record; with outputs, its output values are appended there too. A
    node's own error (NodeExecutionError, as run_node raises it) is a result to
    compare; any other exception ends the worker."""
    tracer = Trace(trace) if trace else None
    if tracer is not None:
        threading.setprofile_all_threads(tracer)
    try:
        result, broadcasts = execute(tree, schema, values, tiles)
        if outputs is not None:
            outputs.extend(result)
        record: dict[str, object] = {
            "outputs": encode(result),
            "broadcasts": encode(broadcasts),
        }
    except tree.process.NodeExecutionError as error:
        cause = error.__cause__ or error
        record = {
            "error": f"{type(cause).__name__}: {cause}",
            "traceback": "".join(traceback.format_exception(error)),
        }
    finally:
        if tracer is not None:
            threading.setprofile_all_threads(None)
    if save.is_dir():
        record["files"] = saved_files(save)
    if tracer is not None:
        record["traced"] = tracer.hit
    return record


def chained(
    tree: Tree,
    case_id: str,
    case: dict[str, Any],
    image: np.ndarray,
    unused: Path,
    outputs: list[object] | None = None,
) -> dict[str, object]:
    """The case's run with image as its first image (a collector's first iterated
    image) and its other images on the C layout."""
    values, tiles = inputs_for(tree, case_id, case, "c", unused)
    if tiles:
        tiles[0] = image
    else:
        values[min(case["channels"])] = image
    return run(
        tree, case["schema"], values, tiles, save=unused, trace=[], outputs=outputs
    )


def chains(
    tree: Tree, found: dict[str, dict[str, Any]], out: Path, producers: list[str]
) -> dict[str, object]:
    """Lens on a C-layout image, then each target with Lens's output as its first
    image; and Lens, then each producer case, then each target case (the module
    docstring)."""
    unused = out / "unused"
    records: dict[str, object] = {}
    first_seed = {
        case_id: case for case_id, case in found.items() if case_id.endswith("|s0")
    }
    for count in CHANNELS:
        lens_id = f"{LENS}|{count}ch|default|s0"
        values, tiles = inputs_for(
            tree, lens_id, cases(tree, [LENS])[lens_id], "c", unused
        )
        (lens_output,), _ = execute(tree, LENS, values, tiles)
        assert isinstance(lens_output, np.ndarray)
        records[f"chain|{lens_id}"] = {"outputs": encode(lens_output)}
        for case_id, case in found.items():
            if case["schema"] in CHAIN_TARGETS and case["count"] == count:
                records[f"chain|{case_id}"] = chained(
                    tree, case_id, case, lens_output, unused
                )
        for case_id, case in first_seed.items():
            if case["schema"] not in producers or case["count"] != count:
                continue
            produced: list[object] = []
            records[f"chain|{case_id}"] = chained(
                tree, case_id, case, lens_output, unused, produced
            )
            image = produced[0] if produced else None
            if not isinstance(image, np.ndarray):
                continue
            channels = image.shape[2] if image.ndim == 3 else 1
            for target_id, target in first_seed.items():
                if target["schema"] in CHAIN_TARGETS and target["count"] == channels:
                    records[f"chain|{case_id}|{target_id}"] = chained(
                        tree, target_id, target, image, unused
                    )
    return records


def main() -> int:
    args = parse_arguments()
    # Torch before the registry: chaiNNer_standard imports Pillow before its
    # Torch-typed inputs, and Pillow's bundled msvcp140.dll breaks a later Torch
    # import in the same process (the owner's import-order rule).
    importlib.import_module("torch")
    root = args.root.resolve()
    # Caption loads its font from beside __main__'s file, the app's run.py.
    sys.modules["__main__"].__file__ = str(root / "run.py")
    tree = Tree(root)
    schemas = [s for s in args.schemas.split(",") if s] or sorted(
        schema
        for schema, node in tree.nodes.items()
        if any(isinstance(i, tree.image_input) for i in node.inputs)
        and schema not in EXCLUDED
    )
    found = cases(tree, schemas)
    runs = {}
    for case_id, case in found.items():
        layouts = args.layouts.split(",")
        if args.mixed and mixed(tree, case):
            layouts += MIXED
        for layout in layouts:
            key = f"{case_id}@{layout}"
            save = args.out / "save" / f"{zlib.crc32(key.encode()):08x}"
            values, tiles = inputs_for(tree, case_id, case, layout, save)
            runs[key] = run(
                tree, case["schema"], values, tiles, save=save, trace=args.trace
            )
    results = {
        "nodes": {
            schema: sum(case["schema"] == schema for case in found.values())
            for schema in schemas
        },
        "cases": {
            case_id: {
                "schema": case["schema"],
                "count": case["count"],
                "levels": case["levels"],
            }
            for case_id, case in found.items()
        },
        "runs": runs,
        "chains": (
            chains(tree, found, args.out, [s for s in args.producers.split(",") if s])
            if args.chains
            else {}
        ),
    }
    (args.out / "results.json").write_text(
        json.dumps(results, indent=1), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
