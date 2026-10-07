"""D-7 condition 4: every image-input node on Lens Blur's permuted layout.

Consult 9 ruled (A): ImageOutput.enforce keeps np.clip's order-K layout, so Lens
Blur's planar (c, h, w) transpose(1, 2, 0) reaches the next node as it does
upstream. layout_gate_worker.py runs every chaiNNer_standard node with an image
input on 3- and 4-channel images in that permuted layout and C-contiguous. It runs
on the port (backend/src, this interpreter) and on the oracle (make_oracle.py's
tree, refused if its chainner_ext isn't backend/src's). The oracle runs on the
provisioned runtime this venv was made from, the interpreter
verify_runtime.oracle_python picks from a package manifest.

The other packages' image nodes are not run. PyTorch, ONNX and NCNN pick a GPU
engine or need a model, and the owner's rule allows no GPU probe; Stable
Diffusion calls a network API. save_video is excluded too (EXCLUDED holds the reason).

- Layout-sensitive consumers are the nodes whose run enters native_palette,
  native_transfer, native_analysis or native_numpy_reduce (CONSUMERS; the trace must
  find exactly these). They equal the oracle bit for bit on both layouts.
- Their inputs hold signed-zero ties, so on some case the oracle's own two
  layouts differ. Without that difference, the comparison above could not tell a
  mirror that ignores the layout from one that follows it.
- The nodes in ORACLE_VALUES equal the oracle bit for bit on both layouts too,
  strides aside (Consult 11 D-17).
- The producers in ORACLE_EXACT equal the oracle with their strides, on both
  layouts and, where a case has more than one image, on the mixed ones (Consult 11
  D-16: a mirror returns the strides upstream's np.clip result has). The oracle
  gives each of them a permuted result somewhere, so the strides are compared
  where they matter.
- Every other node's permuted result equals its C-layout result:
  - its outputs;
  - its broadcast data;
  - the files a Save node writes;
  - its error.
- The UI chains of Lens followed by Color Transfer, and of Lens followed by Stretch
  Contrast, equal the oracle's, strides included. Lens's output reaches them with
  the oracle's strides. So do the chains of Lens, each ORACLE_EXACT producer case
  and each Stretch Contrast and Color Transfer case after it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from bounded import run_python
from layout_gate_worker import (
    EXCLUDED,
    HEIGHT,
    LENS,
    MIXED,
    UNROUNDED,
    WIDTH,
    noise,
)
from package_files import hash_file
from verify_runtime import CHAINNER_EXT, ORACLE, oracle_tree

PROJECT = Path(__file__).resolve().parents[2]
BACKEND = PROJECT / "backend/src"
WORKER = Path(__file__).with_name("layout_gate_worker.py")
CONSUMER_FILES = (
    "native_palette.py",
    "native_transfer.py",
    "native_analysis.py",
    "native_numpy_reduce.py",
)
CONSUMERS = frozenset(
    {
        "chainner:image:average_color_fix",
        "chainner:image:color_transfer",
        "chainner:image:high_pass",
        "chainner:image:image_metrics",
        "chainner:image:lut",
        "chainner:image:palette_from_image",
        "chainner:image:sharpen",
        "chainner:image:sharpen_hbf",
        "chainner:image:stretch_contrast",
    }
)
# Nodes outside CONSUMERS whose values the oracle checks on both layouts, strides
# aside (Consult 11 D-17): Alpha Matting and Chroma Key's foreground starts from
# PyMatting 1.1.16's mean colours, and Blend's clip is np.clip's, which keeps -0.
ORACLE_VALUES = frozenset(
    {
        "chainner:image:alpha_matting",
        "chainner:image:blend",
        "chainner:image:chroma_key",
    }
)
# The producers whose upstream NumPy result np.clip keeps permuted (D-7's report,
# concern 1): compared with the oracle strides included (Consult 11 D-16).
ORACLE_EXACT = frozenset(
    {
        "chainner:image:add",
        "chainner:image:brightness_and_contrast",
        "chainner:image:chroma_key",
        "chainner:image:clamp",
        "chainner:image:color_levels",
        "chainner:image:color_transfer",
        "chainner:image:crop_content",
        "chainner:image:divide",
        "chainner:image:high_pass",
        "chainner:image:invert",
        "chainner:image:log2lin",
        "chainner:image:merge_channels",
        "chainner:image:merge_spritesheet",
        "chainner:image:merge_transparency",
        "chainner:image:multiply",
        "chainner:image:premultiplied_alpha",
        "chainner:image:shift",
        "chainner:image:stretch_contrast",
    }
)
# The producers with more than one image input, run on the mixed layouts too.
MULTIPLE = frozenset(
    {
        "chainner:image:color_transfer",
        "chainner:image:high_pass",
        "chainner:image:merge_channels",
        "chainner:image:merge_spritesheet",
        "chainner:image:merge_transparency",
    }
)
# Nodes whose image inputs take one channel only. A (h, w, 1) transpose squeezes to
# a C-contiguous (h, w), so no permuted layout can reach them.
GRAY_ONLY = frozenset(
    {
        "chainner:image:combine_rgba",
        "chainner:image:distance_transform",
        "chainner:image:get_bbox",
        "chainner:image:image_statistics",
        "chainner:image:threshold_adaptive",
    }
)


def worker(out: Path, root: Path, options: list[str], python: Path | None) -> dict:
    """The worker's results for one tree. Every GPU engine is hidden, and
    PyMatting's numba cache stays in out instead of site-packages."""
    out.mkdir()
    env = {
        "CUDA_VISIBLE_DEVICES": "-1",
        "VK_LOADER_DRIVERS_DISABLE": "*",
        "PYTHONDONTWRITEBYTECODE": "1",
        "NUMBA_CACHE_DIR": str(out / "numba"),
    }
    done = run_python(
        [str(WORKER), str(root), str(out), *options],
        PROJECT,
        1800,
        python=python,
        env=env,
    )
    assert done.returncode == 0, done.output[-4000:]
    return json.loads((out / "results.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def gate(tmp_path_factory) -> tuple[dict, dict]:
    """The port's results on every case and the oracle's on CONSUMERS' and
    ORACLE_VALUES'."""
    built = {name: hash_file(BACKEND / name) for name in CHAINNER_EXT}
    source, _ = oracle_tree(ORACLE, built, "backend/src")
    runtime = Path(sys.base_prefix, "python.exe")
    assert runtime.is_file(), runtime
    root = tmp_path_factory.mktemp("layout_gate")
    impl = BACKEND / "nodes/impl"
    producers = ["--mixed", "--chains", "--producers", ",".join(sorted(ORACLE_EXACT))]
    port = worker(
        root / "port",
        BACKEND,
        [*producers, "--trace", *(str(impl / f) for f in CONSUMER_FILES)],
        None,
    )
    oracle = worker(
        root / "oracle",
        source,
        [
            *producers,
            "--schemas",
            ",".join(sorted(CONSUMERS | ORACLE_VALUES | ORACLE_EXACT)),
        ],
        runtime,
    )
    return port, oracle


def bits(record: object, strides: bool = False) -> object:
    """What a bit-for-bit comparison sees: the record without diagnostics, and
    without strides unless asked."""
    dropped = ("traceback", "traced") if strides else ("strides", "traceback", "traced")
    if isinstance(record, dict):
        return {
            key: bits(value, strides)
            for key, value in record.items()
            if key not in dropped
        }
    if isinstance(record, list):
        return [bits(value, strides) for value in record]
    return record


def schema_cases(results: dict, wanted: frozenset[str], inside: bool) -> list[str]:
    return [
        case_id
        for case_id, case in results["cases"].items()
        if (case["schema"] in wanted) == inside
    ]


def test_every_image_input_node_has_cases_unless_it_takes_only_gray(gate):
    port, _ = gate
    assert not EXCLUDED.keys() & port["nodes"].keys()
    assert {schema for schema, count in port["nodes"].items() if not count} == GRAY_ONLY
    failed = {
        key: run["error"]
        for key, run in port["runs"].items()
        if "|default|" in key and "error" in run
    }
    assert not failed


def test_the_consumers_are_the_nodes_that_enter_the_reduction_bridges(gate):
    port, _ = gate
    traced = {key.split("|")[0] for key, run in port["runs"].items() if run["traced"]}
    assert traced == CONSUMERS


def oracle_differences(
    port: dict, oracle: dict, wanted: frozenset[str], strides: bool = False
) -> list[str]:
    """The runs of wanted's cases, on every layout each ran on, that differ from
    the oracle's."""
    keys = [
        key
        for case_id in schema_cases(port, wanted, True)
        for key in port["runs"]
        if key.startswith(f"{case_id}@")
    ]
    assert {key.split("|")[0] for key in keys} == wanted
    assert set(keys) == {key for key in oracle["runs"] if key.split("|")[0] in wanted}
    return [
        key
        for key in keys
        if bits(port["runs"][key], strides) != bits(oracle["runs"][key], strides)
    ]


def test_consumers_equal_the_oracle_bit_for_bit_on_both_layouts(gate):
    port, oracle = gate
    assert oracle_differences(port, oracle, CONSUMERS) == []


def test_oracle_values_equal_the_oracle_bit_for_bit_on_both_layouts(gate):
    port, oracle = gate
    assert oracle_differences(port, oracle, ORACLE_VALUES) == []


def test_producers_equal_the_oracle_with_their_strides(gate):
    port, oracle = gate
    assert oracle_differences(port, oracle, ORACLE_EXACT, strides=True) == []


def test_multiple_image_producers_ran_on_mixed_layouts(gate):
    port, _ = gate
    suffixes = tuple(f"@{layout}" for layout in MIXED)
    mixed = {key.split("|")[0] for key in port["runs"] if key.endswith(suffixes)}
    assert MULTIPLE <= mixed


def c_ordered(output: dict) -> bool:
    """Whether an encoded array has C-contiguous strides."""
    stride, strides = np.dtype(output["dtype"]).itemsize, []
    for size in reversed(output["shape"]):
        strides.insert(0, stride)
        stride *= size
    return output["strides"] == strides


def test_the_oracle_permutes_every_exact_producer(gate):
    """Each ORACLE_EXACT producer gives a result that isn't C-ordered on some run,
    so the strides comparison sees a layout for every one of them."""
    _, oracle = gate
    permuted = {
        key.split("|")[0]
        for key, run in oracle["runs"].items()
        if key.split("|")[0] in ORACLE_EXACT
        and any(
            isinstance(output, dict) and "strides" in output and not c_ordered(output)
            for output in run.get("outputs", [])
        )
    }
    assert permuted == ORACLE_EXACT


def test_the_oracle_shows_a_layout_on_the_consumers_inputs(gate):
    _, oracle = gate
    shown = [
        case_id
        for case_id in schema_cases(oracle, CONSUMERS, True)
        if bits(oracle["runs"][f"{case_id}@permuted"])
        != bits(oracle["runs"][f"{case_id}@c"])
    ]
    assert shown


def test_all_colors_sees_signed_zero_ties_and_its_colour_limit(gate):
    """Consult 11 D-17.2: each rounded "All colors" image holds -0 and pixel rows
    that are equal but for the sign of a zero, so the comparison with the oracle
    sees which row np.unique keeps. The unrounded case exceeds MAX_COLORS: both
    trees raise the same ValueError (the oracle comparison covers its message)."""
    port, _ = gate
    schema = "chainner:image:palette_from_image"
    rounded = unrounded = 0
    for case_id, case in port["cases"].items():
        if case["schema"] != schema:
            continue
        name = case_id.split("|")[2]
        runs = [port["runs"][f"{case_id}@{layout}"] for layout in ("permuted", "c")]
        if name == UNROUNDED[schema]:
            assert case["levels"] is None
            assert all(
                run["error"].startswith("ValueError: Image has ") for run in runs
            )
            unrounded += 1
        elif name == "Palette Extraction Method=all":
            # The node's image is its input 0.
            shape = (case["count"], HEIGHT, WIDTH)
            image = noise(case_id, 0, shape, case["levels"])
            assert (np.signbit(image) & (image == 0)).any()
            rows = image.transpose(1, 2, 0).reshape(-1, case["count"])
            values = len(np.unique(rows, axis=0))
            assert len(np.unique(rows.view(np.uint32), axis=0)) > values
            assert all("outputs" in run for run in runs)
            rounded += 1
    assert rounded and unrounded


def test_every_other_node_equals_its_c_layout_result(gate):
    port, _ = gate
    others = schema_cases(port, CONSUMERS, False)
    assert others
    assert [
        case_id
        for case_id in others
        if bits(port["runs"][f"{case_id}@permuted"])
        != bits(port["runs"][f"{case_id}@c"])
    ] == []


def test_lens_chains_equal_the_oracle_chain(gate):
    port, oracle = gate
    assert port["chains"].keys() == oracle["chains"].keys()
    assert len(port["chains"]) > 2
    assert [
        key
        for key, run in port["chains"].items()
        if bits(run, strides=True) != bits(oracle["chains"][key], strides=True)
    ] == []
    # Lens, a producer case, a target case: "chain|" and two case ids.
    producers = {key.split("|")[1] for key in port["chains"] if key.count("|") == 8}
    assert producers == ORACLE_EXACT
    for count in (3, 4):
        key = f"chain|{LENS}|{count}ch|default|s0"
        strides = port["chains"][key]["outputs"]["strides"]
        assert strides == oracle["chains"][key]["outputs"]["strides"]
        assert strides == [4 * WIDTH, 4, 4 * HEIGHT * WIDTH]
