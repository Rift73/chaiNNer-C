"""The generated VapourSynth family nodes: schemas from the catalog, overlays, Use as
filter, and the statements their runs write (the VapourSynth tests check those against
reference.vpy's lines in vspipe)."""

from __future__ import annotations

import ast
import json
import logging
import random
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import pytest

# chaiNNer's inputs load the native backend, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

from api import BaseInput, Group, NodeData, NodeGroup, NodeId, Package
from nodes.impl.vapoursynth import catalog
from nodes.impl.vapoursynth.catalog import CLASS, Catalog, input_ids, load_catalog
from nodes.impl.vapoursynth.runner import clip_info
from nodes.impl.vapoursynth.schema_gen import (
    FAMILIES,
    FILTER_ID,
    MAIN_ID,
    METHOD_ID,
    MODE_ID,
    Dropped,
    register_families,
    schema_id,
)
from nodes.impl.vapoursynth.script import VideoClip, VSFilter, compile_script
from process import enforce_inputs

DATA = Path(__file__).parent / "data" / "vapoursynth"
SMALL = DATA / "catalog_small.json"
SHIPPED = sorted(catalog.SHIPPED_OVERLAYS.glob("*.toml"))
PACKAGE_DIR = Path(__file__).parents[1] / "src" / "packages" / "chaiNNer_vapoursynth"
NODE_CAP = 25


def _param(name: str, annotation: str, default: str | None, kind: str = "keyword"):
    return {"name": name, "annotation": annotation, "default": default, "kind": kind}


def _kernel(cls: str, linear: bool = True) -> dict[str, Any]:
    """vskernels' scale of one kernel class (the shape of the real catalog)."""
    params = [
        _param("clip", "VideoNode", None),
        _param("width", "int | None", "None"),
        _param("height", "int | None", "None"),
    ]
    if linear:
        params.append(_param("linear", "bool | None", "None"))
    return {
        "id": f"vskernels.{cls}.scale",
        "source": "jetpack",
        "module": "vskernels",
        "import": f"from vskernels import {cls}",
        "call": f"{cls}(...).scale",
        "kind": "bound",
        "init_params": ([_param("taps", "float", "3")] if cls == "Lanczos" else [])
        + [_param("kwargs", "Any", None, "var_keyword")],
        "params": [*params, _param("kwargs", "Any", None, "var_keyword")],
        "returns": "VideoNode",
        "doc": "Scale a clip.",
        "arg_docs": {"width": "Target width."},
    }


# More functions for the small catalog: vfm, five kernels (one group) and an enum
# whose members filter a clip.
EXTRA_FUNCTIONS = [
    {
        "id": "vsdeinterlace.vfm",
        "source": "jetpack",
        "module": "vsdeinterlace",
        "import": "from vsdeinterlace import vfm",
        "call": "vfm",
        "kind": "function",
        "params": [
            _param("clip", "VideoNode", None),
            _param("tff", "vstools.enums.generic.FieldBasedLike | bool | None", "None"),
            _param("clip2", "VideoNode | None", "None"),
            _param("postprocess", "VideoNode | VSFunctionNoArgs | None", "None"),
        ],
        "returns": "VideoNode",
        "doc": "Field matching.",
        "arg_docs": {},
    },
    *(
        _kernel(cls, linear=cls != "Point")
        for cls in ("Catrom", "Hermite", "Lanczos", "Mitchell", "Point")
    ),
    {
        "id": "vsdenoise.Prefilter",
        "source": "jetpack",
        "module": "vsdenoise",
        "import": "from vsdenoise import Prefilter",
        "call": "Prefilter.<member>",
        "kind": "members",
        "members": ["MINBLUR", "DFTTEST"],
        "params": [
            _param("clip", "VideoNode | MISSING", "vstools.MissingT"),
            _param("planes", "Planes", "None"),
            _param("kwargs", "Any", None, "var_keyword"),
        ],
        "returns": "VideoNode | PrefilterPartial",
        "doc": "Prefilters.",
        "arg_docs": {},
    },
]
EXTRA_ENUMS = {
    "vstools.enums.generic.FieldBasedLike": {
        "import": "from vstools import FieldBased",
        "members": [
            {"name": "PROGRESSIVE", "expr": "FieldBased.PROGRESSIVE"},
            {"name": "BFF", "expr": "FieldBased.BFF"},
            {"name": "TFF", "expr": "FieldBased.TFF"},
        ],
    }
}


def small_catalog(
    tmp_path: Path, shuffle: int | None = None, extra: Sequence[dict[str, Any]] = ()
) -> Path:
    data = json.loads(SMALL.read_text(encoding="utf-8"))
    data["functions"] += [*EXTRA_FUNCTIONS, *extra]
    data["enums"].update(EXTRA_ENUMS)
    if shuffle is not None:
        random.Random(shuffle).shuffle(data["functions"])
    path = tmp_path / f"catalog-{shuffle}-{len(extra)}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def new_group() -> NodeGroup:
    package = Package(
        where=__file__, id="t", name="T", description="", icon="", color=""
    )
    return package.add_category("VS", "", "", "").add_node_group("Families")


def nodes_of(cat: Catalog, families: Iterable[str] = FAMILIES) -> dict[str, NodeData]:
    group = new_group()
    register_families(cat.methods, group, cat.enums, list(families))
    return {node.name: node for node in group.nodes}


@pytest.fixture
def small(tmp_path: Path) -> dict[str, NodeData]:
    return nodes_of(load_catalog(small_catalog(tmp_path), SHIPPED))


def walk(items: Iterable[Any]) -> Iterable[Any]:
    for item in items:
        yield item
        if isinstance(item, Group):
            yield from walk(item.items)


def ids_in(items: Iterable[Any]) -> set[int]:
    return {item for item in walk(items) if isinstance(item, int)}


def method_group(node: NodeData, method_id: str) -> Group:
    (found,) = [
        g
        for g in walk(node.group_layout)
        if isinstance(g, Group)
        and g.info.options.get("condition")
        == {"kind": "enum", "enum": METHOD_ID, "values": [method_id]}
    ]
    return found


def default_of(item: BaseInput) -> object:
    """The value chaiNNer's frontend gives a new node's input."""
    return item.to_dict().get("def")


def run(
    node: NodeData,
    node_id: str,
    method: str,
    values: Mapping[str, object] | None = None,
    clip: VideoClip | None = None,
    as_filter: bool = False,
    raw: Mapping[int, object] | None = None,
) -> VideoClip | VSFilter:
    """Runs a family node as the executor does: every input's value (a new node's
    default unless given; `values` by parameter key, or `raw` by input id), enforced,
    then the run."""
    present = {i.id for i in node.inputs}
    if raw is None:
        values = values or {}
        ids = input_ids(node.name, [(method, key) for key in values])
        given = {ids[method, key]: value for key, value in values.items()}
        assert set(given) <= present, "a parameter key names no input"
        fixed: dict[int, object] = {MAIN_ID: clip, METHOD_ID: method}
        fixed.update({MODE_ID: "simple", FILTER_ID: int(as_filter)})
        given.update({k: v for k, v in fixed.items() if k in present})
        raw = {
            i.id: given[i.id] if i.id in given else default_of(i) for i in node.inputs
        }
    inputs = [raw.get(i.id) for i in node.inputs]
    enforced = enforce_inputs(inputs, node, NodeId(node_id), [])
    context = SimpleNamespace(node_id=node_id)
    return node.run(context, *enforced)


def source(name: str, code: str) -> VideoClip:
    """A clip of the test: `name = <code>`."""
    return VideoClip(name, f"{name} = {code}", (), frozenset(), name)


SRC = source("n_src_0", "core.std.BlankClip(format=vs.YUV420P8, length=4)")
REF = source("n_ref_0", "core.std.BlankClip(format=vs.YUV420P8, length=4)")


# Pure tests (no VapourSynth).


def test_family_count_under_cap(tmp_path: Path) -> None:
    """The families plus the package's hand-written nodes stay under the cap."""
    nodes = nodes_of(load_catalog(small_catalog(tmp_path), SHIPPED))
    assert set(nodes) <= set(FAMILIES)
    assert len(FAMILIES) == 17
    hand_written = sum(
        len(re.findall(r"\.register\(", path.read_text(encoding="utf-8")))
        for path in PACKAGE_DIR.rglob("*.py")
        if path.name != "families.py"
    )
    # AI Upscale (Task 7) registers in another package folder; count it too.
    assert len(FAMILIES) + hand_written + 1 <= NODE_CAP


def test_bm3d_inputs_conditional_on_method(small: dict[str, NodeData]) -> None:
    denoise = small["Denoise"]
    assert denoise.schema_id == schema_id("Denoise") == "chainner:vapoursynth:denoise"
    bm3d = method_group(denoise, "vsdenoise.bm3d")
    ids = input_ids("Denoise", [("vsdenoise.bm3d", p) for p in ("sigma", "tr", "ref")])
    inside = ids_in(bm3d.items)
    assert set(ids.values()) <= inside
    # Every input of the node is in one method's group, except the fixed ones.
    groups = [
        method_group(denoise, m)
        for m in (
            o["value"] for o in denoise.inputs[1].to_dict()["options"]
        )  # the Method dropdown
    ]
    seen: list[int] = [i for g in groups for i in ids_in(g.items)]
    assert len(seen) == len(set(seen))
    fixed = {MAIN_ID, METHOD_ID, MODE_ID, FILTER_ID}
    assert set(seen) | fixed == {i.id for i in denoise.inputs}
    # The main clip is required unless Use as filter is on.
    layout = denoise.group_layout[0]
    assert isinstance(layout, Group)
    assert layout.info.options["condition"] == {
        "kind": "enum",
        "enum": FILTER_ID,
        "values": [0],
    }


def test_simple_set_from_overlay(small: dict[str, NodeData]) -> None:
    bm3d = method_group(small["Denoise"], "vsdenoise.bm3d")
    by_id: dict[int, BaseInput] = {i.id: i for i in small["Denoise"].inputs}
    simple = {by_id[i].label for i in bm3d.items if isinstance(i, int)}
    (advanced,) = [g for g in bm3d.items if isinstance(g, Group)]
    assert advanced.info.options["condition"] == {
        "kind": "enum",
        "enum": MODE_ID,
        "values": ["advanced"],
    }
    # The overlay's simple set, with its labels; the rest under Advanced.
    assert simple == {
        "Strength (sigma)",
        "Temporal radius",
        "Profile",
        "Reference clip",
        "Planes",
    }
    assert {by_id[i].label for i in ids_in(advanced.items)} == {
        "Refine",
        "More arguments",
    }


def test_generated_statement_matches_reference(small: dict[str, NodeData]) -> None:
    """Denoise · BM3D with the walkthrough's values (sigma 0.8, radius 4, profile
    HIGH, a reference clip, planes Y)."""
    out = run(
        small["Denoise"],
        "bm3d",
        "vsdenoise.bm3d",
        {
            "sigma": "0.8",
            "tr": "4",
            "profile": "vsdenoise.bm3d.Profile.HIGH",
            "ref": REF,
            "planes": "[0]",
        },
        clip=SRC,
    )
    assert isinstance(out, VideoClip)
    # Module-qualified: no two imports of a script can shadow each other.
    assert out.stmt == (
        "n_bm3d_0 = vsdenoise.bm3d(n_src_0, sigma=0.8, tr=4,"
        " profile=vsdenoise.bm3d.Profile.HIGH, ref=n_ref_0, planes=[0])"
    )
    assert out.imports == frozenset(["import vsdenoise"])
    assert set(out.deps) == {SRC, REF}


def nnedi3(nodes: dict[str, NodeData]) -> VideoClip | VSFilter:
    return run(
        nodes["Deinterlace"],
        "nnedi3",
        "vsaa.NNEDI3.deinterlace",
        {"pscrn": "0", "tff": "True", "double_rate": False},
        as_filter=True,
    )


def test_use_as_filter_emits_bound_method(small: dict[str, NodeData]) -> None:
    out = nnedi3(small)
    assert isinstance(out, VSFilter)
    assert out.expr == ("vsaa.NNEDI3(pscrn=0, tff=True, double_rate=False).deinterlace")
    assert out.imports == frozenset(["import vsaa"])
    # Switch off: the statement on the main clip.
    out = run(
        small["Deinterlace"],
        "nn",
        "vsaa.NNEDI3.deinterlace",
        {"pscrn": "0", "tff": "True", "double_rate": False},
        clip=SRC,
    )
    assert isinstance(out, VideoClip)
    assert out.stmt == (
        "n_nn_0 = vsaa.NNEDI3(pscrn=0, tff=True,"
        " double_rate=False).deinterlace(n_src_0)"
    )
    # Arguments of the method itself: a partial; a function: itself or a partial.
    out = run(small["Denoise"], "b", "vsdenoise.bm3d", {"sigma": "0.8"}, as_filter=True)
    assert isinstance(out, VSFilter)
    assert out.expr == "functools.partial(vsdenoise.bm3d, sigma=0.8)"
    assert out.imports == frozenset(["import functools", "import vsdenoise"])
    out = run(small["Denoise"], "b", "vsdenoise.bm3d", as_filter=True)
    assert isinstance(out, VSFilter) and out.expr == "vsdenoise.bm3d"
    # Positional arguments after the clip: a lambda.
    out = run(
        small["Convert Format"], "d", "vstools.depth", {"bitdepth": 16}, as_filter=True
    )
    assert isinstance(out, VSFilter)
    assert out.expr == "lambda clip: vstools.depth(clip, 16)"


def test_vfm_accepts_filter(small: dict[str, NodeData]) -> None:
    post = nnedi3(small)
    out = run(
        small["Deinterlace"],
        "vfm",
        "vsdeinterlace.vfm",
        {"postprocess": post},
        clip=SRC,
    )
    assert isinstance(out, VideoClip)
    assert out.stmt == (
        "n_vfm_0 = vsdeinterlace.vfm(n_src_0, postprocess=vsaa.NNEDI3(pscrn=0,"
        " tff=True, double_rate=False).deinterlace)"
    )
    script = compile_script([(out, "out")], cache_mb=64)
    lines = script.splitlines()
    assert "import vsaa" in lines
    assert "import vsdeinterlace" in lines
    assert not any(line.startswith("from ") for line in lines)
    assert script.count("n_vfm_0 = vsdeinterlace.vfm(") == 1
    # The filter writes no statement of its own.
    assert "  # node:nnedi3" not in script


def test_group_collapses_classes(small: dict[str, NodeData]) -> None:
    """Five kernels' scale are one method with a class dropdown; a parameter only
    some classes have shows for them only."""
    resize = small["Resize"]
    options = [o["value"] for o in resize.inputs[1].to_dict()["options"]]
    assert "vskernels.*.scale" in options
    assert not any(o.startswith("vskernels.Mitchell") for o in options)
    group = method_group(resize, "vskernels.*.scale")
    ids = input_ids(
        "Resize", [("vskernels.*.scale", k) for k in (CLASS, "linear", "init.taps")]
    )
    by_id: dict[int, BaseInput] = {i.id: i for i in resize.inputs}
    dropdown = by_id[ids["vskernels.*.scale", CLASS]].to_dict()
    assert [o["value"] for o in dropdown["options"]] == [
        "Catrom",
        "Hermite",
        "Lanczos",
        "Mitchell",
        "Point",
    ]
    assert dropdown["def"] == "Catrom"
    assert dropdown["label"] == "Kernel"
    # Linear light: every class but Point; taps: Lanczos only.
    shown_for: dict[int, tuple[str, ...]] = {}
    for g in walk(group.items):
        condition = g.info.options.get("condition", {}) if isinstance(g, Group) else {}
        if condition.get("enum") == ids["vskernels.*.scale", CLASS]:
            shown_for.update(dict.fromkeys(ids_in(g.items), tuple(condition["values"])))
    assert shown_for[ids["vskernels.*.scale", "linear"]] == (
        "Catrom",
        "Hermite",
        "Lanczos",
        "Mitchell",
    )
    assert shown_for[ids["vskernels.*.scale", "init.taps"]] == ("Lanczos",)

    out = run(
        resize,
        "k",
        "vskernels.*.scale",
        {CLASS: "Mitchell", "width": "1280", "height": "1000", "linear": "True"},
        clip=SRC,
    )
    assert isinstance(out, VideoClip)
    assert out.stmt == (
        "n_k_0 = vskernels.Mitchell().scale(n_src_0, width=1280, height=1000,"
        " linear=True)"
    )
    assert out.imports == frozenset(["import vskernels"])
    # The overlay's defaults: 1280x960 unless set.
    out = run(
        resize, "k", "vskernels.*.scale", {CLASS: "Lanczos", "init.taps": 4.0}, clip=SRC
    )
    assert isinstance(out, VideoClip)
    assert out.stmt == (
        "n_k_0 = vskernels.Lanczos(taps=4.0).scale(n_src_0, width=1280, height=960)"
    )


def test_prefilter_members_as_filter(small: dict[str, NodeData]) -> None:
    """R9: Denoise has the Prefilter method; as a filter it is the member itself."""
    denoise = small["Denoise"]
    out = run(denoise, "p", "vsdenoise.Prefilter", {CLASS: "MINBLUR"}, as_filter=True)
    assert isinstance(out, VSFilter)
    assert out.expr == "vsdenoise.Prefilter.MINBLUR"
    assert out.imports == frozenset(["import vsdenoise"])
    out = run(denoise, "p", "vsdenoise.Prefilter", {"planes": "[0]"}, clip=SRC)
    assert isinstance(out, VideoClip)
    # The overlay preselects DFTTEST.
    assert out.stmt == "n_p_0 = vsdenoise.Prefilter.DFTTEST(n_src_0, planes=[0])"


def test_more_clips_and_positional_gaps(small: dict[str, NodeData]) -> None:
    other = source("n_b_0", "core.std.BlankClip(length=1)")
    planes = small["Planes"]
    out = run(planes, "m", "std.Merge", {"clipb": other, "weight": "0.35, 0"}, clip=SRC)
    assert isinstance(out, VideoClip)
    assert out.stmt == "n_m_0 = core.std.Merge(n_src_0, clipb=n_b_0, weight=[0.35, 0])"
    # depth: positional-only parameters stay positional.
    out = run(small["Convert Format"], "d", "vstools.depth", {"bitdepth": 16}, clip=SRC)
    assert isinstance(out, VideoClip)
    assert out.stmt == "n_d_0 = vstools.depth(n_src_0, 16)"


def test_overlay_defaults(
    tmp_path: Path, small: dict[str, NodeData], caplog: pytest.LogCaptureFixture
) -> None:
    """An overlay's defaults are the values inputs start with, and are written (the
    function's own default is not): depth's Bits starts at 16."""
    convert = small["Convert Format"]
    (bits,) = [
        i
        for i in convert.inputs
        if i.id
        == input_ids("Convert Format", [("vstools.depth", "bitdepth")])[
            "vstools.depth", "bitdepth"
        ]
    ]
    assert default_of(bits) == 16
    out = run(convert, "d", "vstools.depth", clip=SRC)
    assert isinstance(out, VideoClip)
    assert out.stmt == "n_d_0 = vstools.depth(n_src_0, 16)"
    # A default the input cannot take is left out, with a warning.
    bad = tmp_path / "bad.toml"
    bad.write_text(
        '[method."vstools.depth"]\ndefaults = { bitdepth = "sixteen", nope = 1 }\n',
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING):
        nodes = nodes_of(load_catalog(small_catalog(tmp_path), [*SHIPPED, bad]))
    (bits,) = [i for i in nodes["Convert Format"].inputs if i.id == bits.id]
    assert default_of(bits) == 0
    messages = "\n".join(r.getMessage() for r in caplog.records)
    assert "vstools.depth" in messages and "bitdepth" in messages
    assert "nope" in messages


def test_missing_required_value_names_the_input(small: dict[str, NodeData]) -> None:
    planes = small["Planes"]
    with pytest.raises(ValueError, match="Second clip needs a value"):
        run(planes, "m", "std.Merge", clip=SRC)
    with pytest.raises(ValueError, match="Connect a clip"):
        run(planes, "m", "std.Merge", {"clipb": REF})
    with pytest.raises(ValueError, match=r"Weight per plane: .*not a number"):
        run(planes, "m", "std.Merge", {"clipb": REF, "weight": "a, b"}, clip=SRC)


def test_missing_file_says_why(tmp_path: Path) -> None:
    """A required file input whose file is gone (a saved chain's video was moved)
    says so, not that it has no value."""
    video_source = {
        "id": "bs.VideoSource",
        "source": "plugin",
        "module": "bs",
        "import": "",
        "call": "core.bs.VideoSource",
        "kind": "plugin",
        "params": [
            _param("source", "str | bytes", None),
            _param("track", "int | None", "None"),
        ],
        "returns": "VideoNode",
        "doc": "BestSource.",
        "arg_docs": {},
    }
    cat = load_catalog(small_catalog(tmp_path, extra=[video_source]), SHIPPED)
    load = nodes_of(cat)["Load Clip"]
    video = tmp_path / "clip.mkv"
    video.write_bytes(b"")
    out = run(load, "l", "bs.VideoSource", {"source": str(video)})
    assert isinstance(out, VideoClip)
    assert out.stmt == f"n_l_0 = core.bs.VideoSource(source={str(video)!r})"
    moved = tmp_path / "moved.mkv"
    with pytest.raises(ValueError, match=r"does not exist") as error:
        run(load, "l", "bs.VideoSource", {"source": str(moved)})
    assert "needs a value" not in str(error.value)
    with pytest.raises(ValueError, match="needs a value"):
        run(load, "l", "bs.VideoSource")


def test_unknown_input_values_are_dropped_with_warning(
    small: dict[str, NodeData], caplog: pytest.LogCaptureFixture
) -> None:
    """A chain saved before a catalog change: the frontend sends no value for an id
    the schema lost and None for one it gained (or a saved value a dropdown lost).
    The run takes defaults, logs one warning naming them, and never fails."""
    denoise = small["Denoise"]
    method = "vsdenoise.bm3d"
    ids = input_ids("Denoise", [(method, k) for k in ("sigma", "refine", "profile")])
    saved: dict[int, object] = {i.id: default_of(i) for i in denoise.inputs}
    saved.update(
        {
            MAIN_ID: SRC,
            METHOD_ID: method,
            ids[method, "sigma"]: "0.8",
            # A member a newer vs-jetpack renamed.
            ids[method, "profile"]: "vsdenoise.bm3d.Profile.GONE",
            # A parameter the old catalog did not have: no saved value.
            ids[method, "refine"]: None,
            # A parameter the new catalog dropped.
            123456789: "4",
        }
    )
    raw: dict[int, object] = {i.id: saved.get(i.id) for i in denoise.inputs}
    # Enforcing never fails, even for inputs of methods that are not picked.
    enforced = enforce_inputs(
        [raw[i.id] for i in denoise.inputs], denoise, NodeId("x"), []
    )
    assert any(isinstance(v, Dropped) for v in enforced)
    with caplog.at_level(logging.WARNING):
        out = run(denoise, "bm3d", method, raw=raw)
    assert isinstance(out, VideoClip)
    assert out.stmt == "n_bm3d_0 = vsdenoise.bm3d(n_src_0, sigma=0.8)"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "the values of Refine, Profile are missing" in warnings[0].getMessage()
    # A method the catalog lost is a clear error on the node.
    raw[METHOD_ID] = "vsdenoise.gone"
    with pytest.raises(ValueError, match=r"'vsdenoise\.gone' is not among"):
        run(denoise, "bm3d", method, raw=raw)


def schema_ids(nodes: Mapping[str, NodeData]) -> dict[str, dict[str, int]]:
    return {
        node.schema_id: {i.label + f"#{i.id}": i.id for i in node.inputs}
        for node in nodes.values()
    }


def test_schema_ids_stable_across_rescan(tmp_path: Path) -> None:
    """A rescan that lists the functions in another order, or one more function, keeps
    every schema id and every input id."""
    first = nodes_of(load_catalog(small_catalog(tmp_path), SHIPPED))
    again = nodes_of(load_catalog(small_catalog(tmp_path, shuffle=1), SHIPPED))
    assert schema_ids(first) == schema_ids(again)
    new = {
        **EXTRA_FUNCTIONS[0],
        "id": "vsdeinterlace.vfm2",
        "call": "vfm2",
        "import": "from vsdeinterlace import vfm2",
    }
    more = nodes_of(
        load_catalog(small_catalog(tmp_path, shuffle=2, extra=[new]), SHIPPED)
    )
    for name, node in first.items():
        before = {i.id for i in node.inputs}
        after = {i.id for i in more[name].inputs}
        assert before <= after, name
    for node in first.values():
        assert node.schema_id == schema_id(node.name)


def test_keywords_list_methods(small: dict[str, NodeData]) -> None:
    denoise = small["Denoise"]
    assert {"BM3D", "vsdenoise.bm3d", "Prefilter", "MINBLUR"} <= set(denoise.keywords)
    assert "Mitchell" in small["Resize"].keywords
    # Typing a method's label or function name offers the node with it picked.
    found = {
        (s.query, value) for s in denoise.suggestions for value in s.inputs.values()
    }
    assert ("bm3d", "vsdenoise.bm3d") in found
    # An overlay's plain name too, when it is another word.
    deinterlace = {s.query: dict(s.inputs) for s in small["Deinterlace"].suggestions}
    nnedi3 = {METHOD_ID: "vsaa.NNEDI3.deinterlace"}
    assert deinterlace["NNEDI3 deinterlace"] == deinterlace["deinterlace"] == nnedi3
    # The /nodes handler sends them.
    server = (Path(__file__).parents[1] / "src" / "server.py").read_text(
        encoding="utf-8"
    )
    assert '"keywords": node.keywords' in server


def test_any_function_rows(small: dict[str, NodeData]) -> None:
    node = small["Any Function"]
    rows = {i.label: i.id for i in node.inputs}
    out = run(
        node,
        "a",
        "vstools.padder",
        raw={
            MAIN_ID: SRC,
            METHOD_ID: "vstools.padder",
            rows["Argument 1"]: "left = 8",
            # A bare tuple is one argument.
            rows["Argument 2"]: "pad = 1, 2",
            rows["Clip or filter 1"]: REF,
            rows["Argument name 1"]: "ref",
        },
    )
    assert isinstance(out, VideoClip)
    assert out.stmt == (
        "n_a_0 = vstools.padder(n_src_0, left=8, pad=(1, 2), ref=n_ref_0)"
    )
    assert out.imports == frozenset(["import vstools"])
    for bad, message in (
        ("left = __import__('os')", "not a plain Python value"),
        ("left 8", "name = value"),
        ("for = 1", "not an argument name"),
    ):
        with pytest.raises(ValueError, match=message):
            run(
                node,
                "a",
                "vstools.padder",
                raw={
                    MAIN_ID: SRC,
                    METHOD_ID: "vstools.padder",
                    rows["Argument 1"]: bad,
                },
            )


# Text a shared chain file may hold in any input: quotes of each kind, backslashes,
# line breaks (\r and Unicode separators too), f-string braces, a null byte, code
# after a closing parenthesis, very long and deeply nested text.
HOSTILE = [
    "'",
    '"',
    "'''",
    '"""',
    "\\",
    "a\\",
    "\\'",
    "\n",
    "\r",
    "\r\n",
    "\u2028",
    "\x85",
    "\x0b",
    "\x00",
    "a\x00b",
    "{x}",
    "{__import__('os').getcwd()}",
    "f'{1}'",
    "1)\nimport os\n(",
    "0); __import__('os').system('x') #",
    "x'+__import__('os').name+'y",
    '"; import os; "',
    "__import__('os')",
    "lambda: 0",
    "1 if 1 else 2",
    "1 # c",
    "1\n2",
    "1\r2",
    "1\u20282",
    "1;2",
    "1, 2",
    "a" * 100_000,
    "[" * 1000 + "]" * 1000,
    "\ud800",
    "\ufeff1",
    "1e309",
    "nan",
    "0x10",
    "\u0661\u0662\u0663",
    "[0, None]",
    "{'__class__': 1}",
    "b'x'",
    "...",
    "**{}",
    "x = __import__('os')",
    "sigma = 1)\nimport os\n#",
    "sigma = __import__('os')",
    "sigma = 1\r\nimport os",
    "\U0001d526\U0001d52a = 1",
]
# Plain values, one of which each input takes: its script is the hostile one's baseline.
PLAIN = ["{'k': 1}", "1", "0.5", "24000/1001", "0-1", "[0]", "x", "'x'", "sigma = 1"]


def code_shape(code: str) -> tuple[Counter[str], int, int]:
    """The names and attributes in `code` (a literal has none), its calls and lambdas,
    and its statements."""
    tree = ast.parse(code)
    names: Counter[str] = Counter()
    calls = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names[node.id] += 1
        elif isinstance(node, ast.Attribute):
            names[f".{node.attr}"] += 1
        elif isinstance(node, (ast.Call, ast.Lambda)):
            calls += 1
        elif isinstance(node, (ast.JoinedStr, ast.NamedExpr, ast.Starred, ast.IfExp)):
            names[type(node).__name__] += 1
    return names, calls, len(tree.body)


def script_of(out: VideoClip | VSFilter) -> str:
    clip = (
        out
        if isinstance(out, VideoClip)
        else VideoClip(
            "n_use_0",
            f"n_use_0 = ({out.expr})(n_src_0)",
            (SRC, out),
            frozenset(),
            "use",
        )
    )
    return compile_script([(clip, "Clip 1")], cache_mb=1)


def test_hostile_values_stay_literals(small: dict[str, NodeData]) -> None:
    # A chain file is untrusted: whatever text it puts in an input, the node refuses it
    # (ValueError) or its script has the names, calls and statements a plain value of
    # that input gives (a keyword in a filter adds functools.partial), so the text stays
    # a literal.
    checked = 0
    for name, node in small.items():
        by_id = {i.id: i for i in node.inputs}
        methods = [o["value"] for o in by_id[METHOD_ID].to_dict()["options"]]
        for method in methods:
            for as_filter in (0, 1) if FILTER_ID in by_id else (0,):
                base = {i.id: default_of(i) for i in node.inputs}
                base.update({MAIN_ID: SRC, METHOD_ID: method, MODE_ID: "advanced"})
                if FILTER_ID in by_id:
                    base[FILTER_ID] = as_filter
                if name == "Any Function":
                    base.update({30: SRC, 31: REF, 32: "ref", 33: "other"})
                for input_id in by_id.keys() - {MAIN_ID}:
                    plain: str | None = None
                    for value in PLAIN:
                        try:
                            out = run(node, "h", method, raw={**base, input_id: value})
                        except ValueError:
                            continue
                        plain = script_of(out)
                        break
                    for value in HOSTILE:
                        try:
                            out = run(node, "h", method, raw={**base, input_id: value})
                        except ValueError:
                            continue
                        where = (
                            name,
                            method,
                            as_filter,
                            by_id[input_id].label,
                            value[:40],
                        )
                        assert plain is not None, where
                        names, calls, statements = code_shape(script_of(out))
                        p_names, p_calls, p_statements = code_shape(plain)
                        extra = set(names - p_names) - {".partial", "functools"}
                        partial = int(bool(names - p_names))
                        assert not extra, where
                        assert calls <= p_calls + partial, where
                        assert statements <= p_statements + partial, where
                        checked += 1
    assert checked > 1000


def test_no_methods_no_nodes() -> None:
    group = new_group()
    register_families([], group, {})
    assert group.nodes == []


def empty_groups(nodes: Mapping[str, NodeData]) -> dict[str, int]:
    """Per node, its groups without an item. chaiNNer's frontend rejects such a node
    (src/common/group-inputs.ts), and the uncaught error leaves a blank window."""
    found = {
        name: sum(isinstance(g, Group) and not g.items for g in walk(node.group_layout))
        for name, node in nodes.items()
    }
    return {name: count for name, count in found.items() if count}


# std.Reverse takes the clip only: its method has no inputs of its own.
REVERSE = {
    "id": "std.Reverse",
    "source": "plugin",
    "module": "std",
    "import": "",
    "call": "core.std.Reverse",
    "kind": "plugin",
    "params": [_param("clip", "VideoNode", None)],
    "returns": "VideoNode",
    "doc": "VapourSynth Core Functions",
    "arg_docs": {},
}


SHIFT_MULTI = {
    "id": "vstools.shift_clip_multi",
    "source": "jetpack",
    "module": "vstools",
    "import": "from vstools import shift_clip_multi",
    "call": "shift_clip_multi",
    "kind": "function",
    "params": [
        _param("clip", "VideoNode", None),
        _param("offsets", "StrictRange", "(-1, 1)"),
    ],
    "returns": "list[VideoNode]",
    "doc": "Shift a clip forwards or backwards multiple times.",
    "arg_docs": {},
}
INTERLEAVE = {
    **REVERSE,
    "id": "std.Interleave",
    "call": "core.std.Interleave",
    "params": [_param("clips", "VideoNode | Sequence[VideoNode]", None)],
}
HELPER = {**REVERSE, "id": "vstools.FunctionUtil.return_clip"}


BARE_USER = {
    "id": "mine.sharpen",
    "source": "user",
    "module": "mine",
    "import": "from mine import sharpen",
    "call": "sharpen",
    "kind": "function",
    "params": [_param("clip", "", None), _param("amount", "", "1.0")],
    "returns": "",
    "doc": "",
    "arg_docs": {},
}


def test_an_unannotated_user_function_takes_the_clip(tmp_path: Path) -> None:
    # def sharpen(clip, amount=1.0): its first parameter without a default is the clip.
    nodes = nodes_of(load_catalog(small_catalog(tmp_path, extra=[BARE_USER]), SHIPPED))
    node = nodes["User Functions"]
    assert any(i.id == MAIN_ID for i in node.inputs)
    out = run(node, "n_u", "mine.sharpen", clip=SRC)
    assert isinstance(out, VideoClip)
    assert out.stmt.endswith("mine.sharpen(n_src_0)")


def test_internal_helpers_are_in_no_node(tmp_path: Path) -> None:
    cat = load_catalog(small_catalog(tmp_path, extra=[HELPER]), SHIPPED)
    assert "vstools.FunctionUtil.return_clip" not in {m.id for m in cat.methods}


def test_a_list_of_clips_feeds_an_input_that_takes_several(tmp_path: Path) -> None:
    extra = [REVERSE, SHIFT_MULTI, INTERLEAVE]
    nodes = nodes_of(load_catalog(small_catalog(tmp_path, extra=extra), SHIPPED))
    cut = nodes["Cut and Splice"]
    shifted = run(cut, "n_m", "vstools.shift_clip_multi", clip=SRC)
    assert isinstance(shifted, VideoClip) and shifted.many
    alone = run(cut, "n_i", "std.Interleave", clip=shifted)
    assert isinstance(alone, VideoClip) and not alone.many
    assert alone.stmt.endswith("core.std.Interleave(n_n_m_0)")
    # With a second clip: the list's clips, then that one.
    second = input_ids(cut.name, [("std.Interleave", "clips#2")])
    raw: dict[int, object] = {int(i.id): default_of(i) for i in cut.inputs}
    raw.update({MAIN_ID: shifted, METHOD_ID: "std.Interleave", MODE_ID: "simple"})
    raw[second["std.Interleave", "clips#2"]] = REF
    both = run(cut, "n_j", "std.Interleave", raw=raw)
    assert isinstance(both, VideoClip)
    assert both.stmt.endswith("core.std.Interleave([*n_n_m_0, n_ref_0])")
    with pytest.raises(ValueError, match="gives several"):
        run(cut, "n_r", "std.Reverse", clip=shifted)
    with pytest.raises(ValueError, match="gives several"):
        compile_script([(shifted, "out")], cache_mb=64)
    # A filter gives one clip.
    with pytest.raises(ValueError, match="several clips, so it is no filter"):
        run(cut, "n_f", "vstools.shift_clip_multi", as_filter=True)


# A class made from the clip: MVTools(clip).degrain(super=None).
DEGRAIN = {
    "id": "vsdenoise.MVTools.degrain",
    "source": "jetpack",
    "module": "vsdenoise",
    "import": "from vsdenoise import MVTools",
    "call": "MVTools(...).degrain",
    "kind": "bound",
    "init_params": [
        _param("clip", "VideoNode", None),
        _param("pel", "int | None", "None"),
    ],
    "params": [
        _param("super", "VideoNode | None", "None"),
        _param("tr", "int | None", "None"),
    ],
    "returns": "VideoNode",
    "doc": "Motion-compensated degrain.",
    "arg_docs": {},
}


def test_the_clip_of_a_class_made_from_it_is_the_main_clip(tmp_path: Path) -> None:
    nodes = nodes_of(load_catalog(small_catalog(tmp_path, extra=[DEGRAIN]), SHIPPED))
    denoise = nodes["Denoise"]
    out = run(denoise, "mv", "vsdenoise.MVTools.degrain", clip=SRC)
    assert isinstance(out, VideoClip)
    assert out.stmt == "n_mv_0 = vsdenoise.MVTools(n_src_0).degrain()"
    assert out.deps == (SRC,)
    out = run(denoise, "mv", "vsdenoise.MVTools.degrain", as_filter=True)
    assert isinstance(out, VSFilter)
    assert out.expr == "lambda clip: vsdenoise.MVTools(clip).degrain()"
    with pytest.raises(ValueError, match="Connect a clip"):
        run(denoise, "mv", "vsdenoise.MVTools.degrain")


STACK_CLIPS = {
    "id": "vstools.stack_clips",
    "source": "jetpack",
    "module": "vstools",
    "import": "from vstools import stack_clips",
    "call": "stack_clips",
    "kind": "function",
    "params": [_param("clips", "Iterable[VideoNodeIterable]", None)],
    "returns": "VideoNode",
    "doc": "Stacks clips.",
    "arg_docs": {},
}


def test_clips_of_vstools_iterable_take_the_clip_inputs(tmp_path: Path) -> None:
    nodes = nodes_of(
        load_catalog(small_catalog(tmp_path, extra=[STACK_CLIPS]), SHIPPED)
    )
    (node,) = [n for n in nodes.values() if "vstools.stack_clips" in n.keywords]
    out = run(node, "st", "vstools.stack_clips", clip=SRC)
    assert isinstance(out, VideoClip)
    assert out.stmt == "n_st_0 = vstools.stack_clips([n_src_0])"
    second = input_ids(node.name, [("vstools.stack_clips", "clips#2")])
    raw: dict[int, object] = {int(i.id): default_of(i) for i in node.inputs}
    raw.update({MAIN_ID: SRC, METHOD_ID: "vstools.stack_clips", MODE_ID: "simple"})
    raw[second["vstools.stack_clips", "clips#2"]] = REF
    out = run(node, "st", "vstools.stack_clips", raw=raw)
    assert isinstance(out, VideoClip)
    assert out.stmt == "n_st_0 = vstools.stack_clips([n_src_0, n_ref_0])"
    # As a filter, the clip it is given comes first.
    raw[FILTER_ID] = 1
    out = run(node, "st", "vstools.stack_clips", raw=raw)
    assert isinstance(out, VSFilter)
    assert out.expr == "lambda clip: vstools.stack_clips([clip, n_ref_0])"
    assert out.deps == (REF,)


# vstools.limiter takes a clip, or a function it decorates.
LIMITER = {
    "id": "vstools.limiter",
    "source": "jetpack",
    "module": "vstools",
    "import": "from vstools import limiter",
    "call": "limiter",
    "kind": "function",
    "params": [
        _param(
            "clip_or_func",
            "vs.VideoNode | Callable[P, vs.VideoNode] | None",
            "None",
            "positional",
        ),
        _param("tv_range", "bool", "False"),
    ],
    "returns": "VideoNode",
    "doc": "Limits the pixel values.",
    "arg_docs": {},
}


def test_a_clip_or_function_parameter_is_the_main_clip(tmp_path: Path) -> None:
    nodes = nodes_of(load_catalog(small_catalog(tmp_path, extra=[LIMITER]), SHIPPED))
    (node,) = [n for n in nodes.values() if "vstools.limiter" in n.keywords]
    out = run(node, "li", "vstools.limiter", {"tv_range": True}, clip=SRC)
    assert isinstance(out, VideoClip)
    assert out.stmt == "n_li_0 = vstools.limiter(n_src_0, tv_range=True)"
    out = run(node, "li", "vstools.limiter", as_filter=True)
    assert isinstance(out, VSFilter) and out.expr == "vstools.limiter"


def test_a_method_without_inputs_adds_no_empty_group(tmp_path: Path) -> None:
    nodes = nodes_of(load_catalog(small_catalog(tmp_path, extra=[REVERSE]), SHIPPED))
    (methods,) = [i for i in nodes["Cut and Splice"].inputs if i.id == METHOD_ID]
    assert "std.Reverse" in [o["value"] for o in methods.to_dict()["options"]]
    assert empty_groups(nodes) == {}


# VapourSynth: the real catalog, and reference.vpy's lines in vspipe.


# Two user files whose functions have the name of vs_colorfix's.
USER_AVERAGES = {
    "one": (
        "import vapoursynth as vs\n\n\n"
        "def average(clip: vs.VideoNode, amount: int = 10) -> vs.VideoNode:\n"
        '    """Adds amount."""\n'
        '    return clip.std.Expr(f"x {amount} +")\n'
    ),
    "two": (
        "import vapoursynth as vs\n\n\n"
        "def average(clip: vs.VideoNode) -> vs.VideoNode:\n"
        '    """Inverts."""\n'
        "    return clip.std.Invert()\n"
    ),
}


@pytest.fixture(scope="module")
def real(tmp_path_factory: pytest.TempPathFactory) -> Catalog:
    """The real catalog of the venv (vs-jetpack, its plugins, vs_colorfix listed as a
    user module), with the shipped overlays."""
    pytest.importorskip("vapoursynth")
    storage = tmp_path_factory.mktemp("vapoursynth")
    (storage / "user_modules.toml").write_text(
        'modules = ["vs_colorfix"]\n', encoding="utf-8"
    )
    (storage / "user").mkdir()
    for name, body in USER_AVERAGES.items():
        (storage / "user" / f"{name}.py").write_text(body, encoding="utf-8")
    path = catalog.ensure_catalog(storage)
    assert path is not None
    return load_catalog(path, SHIPPED)


@pytest.fixture(scope="module")
def real_nodes(real: Catalog) -> dict[str, NodeData]:
    return nodes_of(real)


# The shipped overlays' defaults: (family, method, parameter key, starting value).
SHIPPED_DEFAULTS = [
    # bitdepth defaults to None in vs-jetpack 2.2.6: an optional text field.
    ("Convert Format", "vstools.depth", "bitdepth", "16"),
    ("Convert Format", "resize.Point", "format", "vs.RGBS"),
    ("Clip Properties", "std.SetFrameProps", "_Matrix", "6"),
    ("Clip Properties", "std.SetFrameProps", "_Transfer", "6"),
    ("Clip Properties", "std.SetFrameProps", "_Primaries", "6"),
    ("Clip Properties", "std.AssumeFPS", "fpsnum", "25"),
    ("Clip Properties", "std.AssumeFPS", "fpsden", "1"),
    ("Clip Properties", "vstools.change_fps", "fps", "24000/1001"),
    ("Resize", "vskernels.*.scale", "width", "1280"),
    ("Resize", "vskernels.*.scale", "height", "960"),
    ("Resize", "std.Crop", "top", "20"),
    ("Resize", "std.Crop", "bottom", "20"),
    ("Cut and Splice", "std.Trim", "first", "30"),
    ("Cut and Splice", "std.DeleteFrames", "frames", "0"),
    ("Cut and Splice", "std.BlankClip", "length", "1"),
]


def test_shipped_overlay_defaults(
    real: Catalog, caplog: pytest.LogCaptureFixture
) -> None:
    """Every shipped overlay default fits its input on the real catalog (no overlay
    warning), and the inputs start with them."""
    with caplog.at_level(logging.WARNING):
        nodes = nodes_of(real)
    assert not [r.getMessage() for r in caplog.records if "overlay" in r.getMessage()]
    for family, method, key, value in SHIPPED_DEFAULTS:
        input_id = input_ids(family, [(method, key)])[method, key]
        (found,) = [i for i in nodes[family].inputs if i.id == input_id]
        assert default_of(found) == value, (method, key)


def test_real_catalog_has_no_empty_group(real_nodes: dict[str, NodeData]) -> None:
    assert empty_groups(real_nodes) == {}


def test_real_catalog_families(real: Catalog, real_nodes: dict[str, NodeData]) -> None:
    """Every family has a node; groups keep the nodes small (R13)."""
    assert set(real_nodes) == set(FAMILIES)
    methods = {
        name: len(n.inputs[1 if name != "Load Clip" else 0].to_dict()["options"])
        for name, n in real_nodes.items()
    }
    assert methods["Resize"] < 60
    for node in real_nodes.values():
        assert len(node.inputs) <= 600, node.name
    # The walkthrough's methods are where it puts them.
    where = {
        "Load Clip": ["bs.VideoSource", "vssource.*.source"],
        "Clip Properties": ["std.SetFrameProps", "std.AssumeFPS", "vstools.change_fps"],
        "Convert Format": ["resize.Point", "vstools.depth"],
        "Resize": ["vskernels.*.scale", "std.Crop"],
        "Cut and Splice": [
            "std.Trim",
            "vstools.replace_ranges",
            "std.DeleteFrames",
            "std.BlankClip",
            "std.Splice",
        ],
        "Planes": ["vstools.join", "std.Merge"],
        "Deinterlace": [
            "vsdeinterlace.vfm",
            "vsdeinterlace.vdecimate",
            "vsaa.NNEDI3.deinterlace",
        ],
        "Denoise": [
            "vsdenoise.mc_degrain",
            "vsdenoise.bm3d",
            "vsdenoise.nl_means",
            "vsdenoise.DFTTest.denoise",
            "vsdenoise.Prefilter",
        ],
        "Sharpen and Blur": ["cas.CAS"],
        "Colour": ["vs_colorfix.average"],
        "Masks": ["vsmasktools.*.edgemask"],
    }
    for family, wanted in where.items():
        node = real_nodes[family]
        options = [
            o["value"]
            for o in node.inputs[1 if family != "Load Clip" else 0].to_dict()["options"]
        ]
        assert set(wanted) <= set(options), family
    assert "vstools.split" not in str(real_nodes["Planes"].inputs[1].to_dict())
    # Any Function's dropdown groups by module: each module's methods together, its
    # group starting at the first.
    dropdown = real_nodes["Any Function"].inputs[1].to_dict()
    modules = [o["value"].split(".", 1)[0] for o in dropdown["options"]]
    starts = [g["startAt"] for g in dropdown["groups"]]
    assert [g["label"] for g in dropdown["groups"]] == list(dict.fromkeys(modules))
    assert starts == [
        o["value"]
        for i, o in enumerate(dropdown["options"])
        if i == 0 or modules[i] != modules[i - 1]
    ]


class Chain:
    """Family nodes run into script records, each beside reference.vpy's line for the
    same operation (reference variables r1, r2...), compared in one script."""

    def __init__(self, nodes: dict[str, NodeData]) -> None:
        self.nodes = nodes
        self.count = 0
        self.lines: list[str] = []
        self.records: dict[str, VideoClip] = {}
        self.compare: list[tuple[str, str, str]] = []

    def __call__(
        self, family: str, method: str, clip: VideoClip | None = None, **values: object
    ) -> VideoClip:
        self.count += 1
        out = run(self.nodes[family], f"g{self.count}", method, values, clip=clip)
        assert isinstance(out, VideoClip)
        return out

    def filter(self, family: str, method: str, **values: object) -> VSFilter:
        self.count += 1
        out = run(self.nodes[family], f"f{self.count}", method, values, as_filter=True)
        assert isinstance(out, VSFilter)
        return out

    def ref(self, expr: str) -> str:
        """A reference line `r<n> = <expr>`; its variable."""
        name = f"r{len(self.lines) + 1}"
        self.lines.append(f"{name} = {expr}")
        return name

    def same(self, clip: VideoClip, ref: str, what: str) -> None:
        self.records[clip.var] = clip
        self.compare.append((clip.var, ref, what))


SAME = '''
def _same(a, b, what):
    """Equal clips: format, size, length, rate, and every frame's planes and props."""
    shape = lambda c: (c.format.name, c.width, c.height, c.num_frames, c.fps)
    assert shape(a) == shape(b), (what, shape(a), shape(b))
    for n in range(a.num_frames):
        fa, fb = a.get_frame(n), b.get_frame(n)
        assert dict(fa.props) == dict(fb.props), (what, n, dict(fa.props), dict(fb.props))
        for p in range(fa.format.num_planes):
            assert bytes(fa[p]) == bytes(fb[p]), (what, n, p)
'''

REF_IMPORTS = [
    "from fractions import Fraction",
    "import vs_colorfix",
    "from vsaa import NNEDI3",
    "from vsdeinterlace import vdecimate, vfm",
    "from vsdenoise import DFTTest, MVToolsPreset, Prefilter, bm3d, mc_degrain, nl_means",
    "from vskernels import Hermite, Mitchell",
    "from vstools import change_fps, depth, join, replace_ranges, split",
]

# An NTSC-like interlaced test clip with X, Y and N patterns, so that every filter
# has work (the PAL-like one is a .y4m file of noise, for BestSource).
NTSC = (
    "core.std.BlankClip(format=vs.YUV420P8, width=96, height=72, length=30,"
    " fpsnum=30000, fpsden=1001).akarin.Expr(['X Y + N 7 * + 256 %',"
    " 'X 3 * Y + N + 256 %', 'X Y 2 * + N 5 * + 256 %']).std.SetFrameProps("
    "_FieldBased=2)"
)


def split_planes(clip: VideoClip, node_id: str) -> list[VideoClip]:
    """The Split Planes node's records: one statement, three clips."""
    names = [f"n_{node_id}_{i}" for i in range(3)]
    stmt = f"{', '.join(names)} = vstools.split({clip.var})"
    imports = frozenset(["import vstools"])
    return [VideoClip(name, stmt, (clip,), imports, node_id) for name in names]


def write_y4m(path: Path, width: int, height: int, frames: int) -> None:
    rng = random.Random(4)
    with path.open("wb") as out:
        out.write(f"YUV4MPEG2 W{width} H{height} F25:1 Ip A1:1 C420jpeg\n".encode())
        for _ in range(frames):
            out.write(b"FRAME\n")
            out.write(rng.randbytes(width * height * 3 // 2))


def test_reference_lines_match_in_vspipe(
    real_nodes: dict[str, NodeData], tmp_path: Path
) -> None:
    """Each reference.vpy operation the walkthrough maps to a family node, written by
    the node with the walkthrough's values, gives the frames and frame properties of
    reference.vpy's own line, on small test clips in vspipe (sizes scaled down). The
    AI Upscale lines are left out (another node), and split is written as the Split
    Planes node writes it."""
    video = tmp_path / "pal 25p.y4m"
    write_y4m(video, 96, 72, 30)
    path = repr(str(video))
    c = Chain(real_nodes)

    # Sources (example.vpy 18-31).
    src = c("Load Clip", "bs.VideoSource", source=str(video))
    r_src = c.ref(f"core.bs.VideoSource({path})")
    c.same(src, r_src, "BestSource")
    src = c(
        "Clip Properties",
        "std.SetFrameProps",
        src,
        _Matrix="6",
        _Transfer="6",
        _Primaries="6",
        _SARNum="16",
        _SARDen="15",
    )
    r_src = c.ref(
        f"core.std.SetFrameProps({r_src}, _Matrix=6, _Transfer=6, _Primaries=6,"
        " _SARNum=16, _SARDen=15)"
    )
    c.same(src, r_src, "SetFrameProps")
    src = c("Clip Properties", "std.AssumeFPS", src, fpsnum="25", fpsden="1")
    r_src = c.ref(f"core.std.AssumeFPS(clip={r_src}, fpsnum=25, fpsden=1)")
    c.same(src, r_src, "AssumeFPS")
    src = c("Cut and Splice", "std.Trim", src, first="3")
    r_src = c.ref(f"{r_src}[3:]")
    c.same(src, r_src, "Trim")
    src = c("Clip Properties", "vstools.change_fps", src, fps="24000/1001")
    r_src = c.ref(f"change_fps({r_src}, Fraction(24000, 1001))")
    c.same(src, r_src, "change_fps")
    cf = c(
        "Convert Format",
        "resize.Point",
        src,
        format="vs.YUV444P16",
        matrix="6",
        transfer="6",
        primaries="6",
    )
    r_cf = c.ref(
        f"core.resize.Point({r_src}, format=vs.YUV444P16, matrix=6, transfer=6,"
        " primaries=6)"
    )
    c.same(cf, r_cf, "resize.Point to YUV444P16")
    cf = c(
        "Resize",
        "vskernels.*.scale",
        cf,
        **{CLASS: "Mitchell", "width": "128", "height": "100", "linear": "True"},
    )
    r_cf = c.ref(f"Mitchell().scale({r_cf}, 128, 100, linear=True)")
    c.same(cf, r_cf, "Mitchell scale")
    cf = c("Resize", "std.Crop", cf, top="4", bottom="4")
    r_cf = c.ref(f"core.std.Crop({r_cf}, top=4, bottom=4, left=0, right=0)")
    c.same(cf, r_cf, "Crop")
    cf = c("Sharpen and Blur", "cas.CAS", cf)
    r_cf = c.ref(f"core.cas.CAS({r_cf})")
    c.same(cf, r_cf, "CAS")
    rgbs = c("Convert Format", "resize.Point", src, format="vs.RGBS")
    r_rgbs = c.ref(f"core.resize.Point({r_src}, format=vs.RGBS)")
    c.same(rgbs, r_rgbs, "resize.Point to RGBS")
    warped = c(
        "Resize",
        "vskernels.*.scale",
        c("Load Clip", "bs.VideoSource", source=str(video)),
        **{CLASS: "Hermite", "width": "128", "height": "96", "linear": "True"},
    )
    r_warped = c.ref(
        f"Hermite().scale(core.bs.VideoSource({path}), 128, 96, linear=True)"
    )
    c.same(warped, r_warped, "Hermite scale")

    # IVTC (example.vpy 33-42).
    ntsc = VideoClip("ntsc", f"ntsc = {NTSC}", (), frozenset(), "ntsc")
    src2 = c(
        "Clip Properties",
        "std.SetFrameProps",
        ntsc,
        _Matrix="6",
        _Transfer="6",
        _Primaries="6",
        _SARNum="8",
        _SARDen="9",
    )
    r_src2 = c.ref(
        f"core.std.SetFrameProps({NTSC}, _Matrix=6, _Transfer=6, _Primaries=6,"
        " _SARNum=8, _SARDen=9)"
    )
    c.same(src2, r_src2, "SetFrameProps (NTSC)")
    post = c.filter(
        "Deinterlace",
        "vsaa.NNEDI3.deinterlace",
        pscrn="0",
        tff="True",
        double_rate=False,
    )
    iv = c("Deinterlace", "vsdeinterlace.vfm", src2, postprocess=post)
    r_iv = c.ref(
        f"vfm({r_src2}, postprocess=NNEDI3(pscrn=0, tff=True,"
        " double_rate=False).deinterlace)"
    )
    c.same(iv, r_iv, "vfm with the NNEDI3 filter")
    iv = c("Deinterlace", "vsdeinterlace.vdecimate", iv)
    r_iv = c.ref(f"vdecimate({r_iv})")
    c.same(iv, r_iv, "vdecimate")
    pp = c("Deinterlace", "vsdeinterlace.vfm", src2)
    pp = c("Deinterlace", "vsdeinterlace.vdecimate", pp)
    pp = c(
        "Deinterlace",
        "vsaa.NNEDI3.deinterlace",
        pp,
        pscrn="0",
        tff="True",
        double_rate=False,
    )
    r_pp = c.ref(
        "NNEDI3(pscrn=0, tff=True, double_rate=False).deinterlace("
        f"vdecimate(vfm({r_src2})))"
    )
    c.same(pp, r_pp, "NNEDI3 deinterlace")
    iv = c(
        "Cut and Splice",
        "vstools.replace_ranges",
        iv,
        clip_b=pp,
        ranges="3-5, 14-15, 17, 19-20",
    )
    r_iv = c.ref(f"replace_ranges({r_iv}, {r_pp}, [(3, 5), (14, 15), (17), (19, 20)])")
    c.same(iv, r_iv, "replace_ranges")

    # Planes (example.vpy 44-58, 75-83, 95).
    y, _, _ = split_planes(src, "split")
    y1, u1, v1 = split_planes(iv, "split1")
    r_y = c.ref(f"split({r_src})[0]")
    r_y1, r_u1, r_v1 = (c.ref(f"split({r_iv})[{i}]") for i in range(3))
    y0 = c("Convert Format", "vstools.depth", y1, bitdepth=16)
    r_y0 = c.ref(f"depth({r_y1}, 16)")
    c.same(y0, r_y0, "depth 16")
    y = c("Convert Format", "resize.Point", y, format="vs.RGBS")
    y = c("Convert Format", "vstools.depth", y, bitdepth=32)
    r_yf = c.ref(f"depth(core.resize.Point({r_y}, format=vs.RGBS), 32)")
    c.same(y, r_yf, "resize.Point to RGBS, depth 32")
    luma = c(
        "Convert Format",
        "resize.Point",
        y,
        format="vs.GRAY8",
        matrix="6",
        transfer="6",
        primaries="6",
    )
    luma = c(
        "Resize",
        "vskernels.*.scale",
        luma,
        **{CLASS: "Hermite", "width": "128", "height": "100", "linear": "True"},
    )
    luma = c("Resize", "std.Crop", luma, top="4", bottom="4")
    luma = c("Convert Format", "vstools.depth", luma, bitdepth=16)
    r_luma = c.ref(
        f"depth(core.std.Crop(Hermite().scale(core.resize.Point({r_yf},"
        " format=vs.GRAY8, matrix=6, transfer=6, primaries=6), 128, 100,"
        " linear=True), top=4, bottom=4), 16)"
    )
    c.same(luma, r_luma, "GRAY8, Hermite, Crop, depth 16")
    chroma = c(
        "Convert Format",
        "resize.Point",
        rgbs,
        format="vs.YUV444P16",
        matrix="6",
        transfer="6",
        primaries="6",
    )
    chroma = c(
        "Resize",
        "vskernels.*.scale",
        chroma,
        **{CLASS: "Hermite", "width": "128", "height": "100", "linear": "True"},
    )
    chroma = c("Resize", "std.Crop", chroma, top="4", bottom="4")
    r_chroma = c.ref(
        f"core.std.Crop(Hermite().scale(core.resize.Point({r_rgbs},"
        " format=vs.YUV444P16, matrix=6, transfer=6, primaries=6), 128, 100,"
        " linear=True), top=4, bottom=4)"
    )
    c.same(chroma, r_chroma, "YUV444P16, Hermite, Crop")
    clip_a = c("Planes", "vstools.join", luma, plane1_or_family=chroma)
    r_a = c.ref(f"join({r_luma}, {r_chroma})")
    c.same(clip_a, r_a, "join luma and chroma")
    clip_a = c(
        "Colour",
        "vs_colorfix.average",
        clip_a,
        ref=cf,
        radius=10,
        planes="[0, 1, 2]",
        fast=False,
    )
    r_a = c.ref(
        f"vs_colorfix.average({r_a}, {r_cf}, radius=10, planes=[0, 1, 2], fast=False)"
    )
    c.same(clip_a, r_a, "vs_colorfix.average")
    u16 = c("Convert Format", "vstools.depth", u1, bitdepth=16)
    v16 = c("Convert Format", "vstools.depth", v1, bitdepth=16)
    joined = c("Planes", "vstools.join", y0, plane1_or_family=u16, plane2=v16)
    r_joined = c.ref(f"join({r_y0}, depth({r_u1}, 16), depth({r_v1}, 16))")
    c.same(joined, r_joined, "join Y, U and V")

    # Merge and scene replacement (example.vpy 105-125).
    sharp = c("Sharpen and Blur", "cas.CAS", clip_a)
    merged = c("Planes", "std.Merge", clip_a, clipb=sharp, weight="0.35, 0")
    r_merged = c.ref(f"core.std.Merge({r_a}, core.cas.CAS({r_a}), [0.35, 0])")
    c.same(merged, r_merged, "Merge")
    minus1 = c("Cut and Splice", "std.DeleteFrames", clip_a, frames="0")
    r_minus1 = c.ref(f"core.std.DeleteFrames({r_a}, 0)")
    c.same(minus1, r_minus1, "DeleteFrames")
    plus1 = c("Cut and Splice", "std.BlankClip", clip_a, length="1")
    r_plus1 = c.ref(f"core.std.BlankClip({r_a}, length=1)")
    c.same(plus1, r_plus1, "BlankClip")
    plus1 = c("Cut and Splice", "std.Splice", plus1, **{"clips#2": clip_a})
    r_plus1 = c.ref(f"{r_plus1} + {r_a}")
    c.same(plus1, r_plus1, "Splice")
    decen = c(
        "Cut and Splice",
        "vstools.replace_ranges",
        merged,
        clip_b=minus1,
        ranges="1, 3, 5, 10",
    )
    decen = c(
        "Cut and Splice",
        "vstools.replace_ranges",
        decen,
        clip_b=plus1,
        ranges="1-2, 4, 9",
    )
    decen = c(
        "Clip Properties",
        "std.SetFrameProps",
        decen,
        _Matrix="6",
        _Transfer="6",
        _Primaries="6",
        _SARNum="1",
        _SARDen="1",
    )
    r_decen = c.ref(
        f"core.std.SetFrameProps(replace_ranges(replace_ranges({r_merged}, {r_minus1},"
        f" [(1), (3), (5), (10)]), {r_plus1}, [(1, 2), (4), (9)]), _Matrix=6,"
        " _Transfer=6, _Primaries=6, _SARNum=1, _SARDen=1)"
    )
    c.same(decen, r_decen, "replace_ranges twice, SAR 1:1")

    # Denoise (example.vpy 127-129, with reference.vpy's CPU backends).
    dft = c.filter(
        "Denoise",
        "vsdenoise.DFTTest.denoise",
        backend="vsdenoise.DFTTest.Backend.CPU",
        sloc="{0.0: 4, 0.2: 9, 1.0: 15}",
    )
    mvref = c(
        "Denoise",
        "vsdenoise.mc_degrain",
        decen,
        prefilter=dft,
        preset="vsdenoise.MVToolsPreset.HQ_SAD",
        thsad="100",
    )
    r_ref = c.ref(
        f"mc_degrain({r_decen}, prefilter=DFTTest(sloc={{0.0: 4, 0.2: 9, 1.0: 15}},"
        " backend=DFTTest.Backend.CPU).denoise, preset=MVToolsPreset.HQ_SAD,"
        " thsad=100)"
    )
    c.same(mvref, r_ref, "mc_degrain with the DFTTest filter")
    denoise = c(
        "Denoise",
        "vsdenoise.bm3d",
        decen,
        sigma="0.8",
        tr="4",
        profile="vsdenoise.bm3d.Profile.HIGH",
        ref=mvref,
        planes="[0]",
        backend="vsdenoise.bm3d.Backend.OLD",
    )
    r_dn = c.ref(
        f"bm3d({r_decen}, sigma=0.8, tr=4, profile=bm3d.Profile.HIGH, ref={r_ref},"
        " planes=0, backend=bm3d.Backend.OLD)"
    )
    c.same(denoise, r_dn, "bm3d")
    denoise = c(
        "Denoise",
        "vsdenoise.nl_means",
        denoise,
        h="0.2",
        tr="4",
        ref=mvref,
        planes="[1, 2]",
        backend="vsdenoise.nl_means.Backend.ISPC",
    )
    r_dn = c.ref(
        f"nl_means({r_dn}, 0.2, tr=4, ref={r_ref}, planes=[1, 2],"
        " backend=nl_means.Backend.ISPC)"
    )
    c.same(denoise, r_dn, "nl_means")
    # R9: the Prefilter method with Use as filter on, wired into mc_degrain.
    pre = c.filter("Denoise", "vsdenoise.Prefilter", **{CLASS: "MINBLUR"})
    mv2 = c("Denoise", "vsdenoise.mc_degrain", decen, prefilter=pre)
    r_mv2 = c.ref(f"mc_degrain({r_decen}, prefilter=Prefilter.MINBLUR)")
    c.same(mv2, r_mv2, "mc_degrain with the Prefilter filter")

    # One script: the chain's statements (each compared clip an output), then the
    # reference lines and the comparisons.
    script = compile_script(
        [(c.records[var], var) for var, _, _ in c.compare], cache_mb=1024
    )
    checks = (f"_same({var}, {ref}, {what!r})" for var, ref, what in c.compare)
    path = tmp_path / "families.vpy"
    path.write_text(
        "\n".join([script, *REF_IMPORTS, *c.lines, SAME, *checks, ""]),
        encoding="utf-8",
    )
    assert len(c.compare) == 32
    assert clip_info(path, 0).num_frames > 0


def test_user_functions_do_not_collide(
    real_nodes: dict[str, NodeData], tmp_path: Path
) -> None:
    """one.average, two.average and vs_colorfix.average in one chain: each node calls
    its own function (each output equals that function's own result)."""
    user = real_nodes["User Functions"]
    labels = {o["option"] for o in user.inputs[1].to_dict()["options"]}
    assert {"Average (one)", "Average (two)"} <= labels
    src = source(
        "n_src_0",
        "core.std.BlankClip(format=vs.YUV444P8, width=64, height=48, length=3)"
        ".akarin.Expr(['X Y + N 7 * + 256 %', 'X 3 * Y + 256 %', 'Y 2 * N + 256 %'])",
    )
    a = run(user, "a", "one.average", {"amount": 20}, clip=src)
    assert isinstance(a, VideoClip)
    b = run(user, "b", "two.average", clip=a)
    assert isinstance(b, VideoClip)
    c = run(real_nodes["Colour"], "c", "vs_colorfix.average", {"ref": src}, clip=b)
    assert isinstance(c, VideoClip)
    assert a.stmt == "n_a_0 = one.average(n_src_0, amount=20)"
    assert b.stmt == "n_b_0 = two.average(n_a_0)"
    assert c.stmt.startswith("n_c_0 = vs_colorfix.average(n_b_0, ref=n_src_0")
    script = compile_script([(c, "c"), (a, "a"), (b, "b")], cache_mb=256)
    lines = script.splitlines()
    assert {"import one", "import two", "import vs_colorfix"} <= set(lines)
    assert not any(line.startswith("from ") for line in lines)
    checks = [
        SAME,
        "_same(n_a_0, core.std.Expr(n_src_0, 'x 20 +'), 'one.average')",
        "_same(n_b_0, core.std.Invert(n_a_0), 'two.average')",
        "_same(n_c_0, vs_colorfix.average(n_b_0, n_src_0), 'vs_colorfix.average')",
        # The three are different clips.
        "assert bytes(n_a_0.get_frame(0)[0]) != bytes(n_b_0.get_frame(0)[0])",
        "assert bytes(n_b_0.get_frame(0)[0]) != bytes(n_c_0.get_frame(0)[0])",
    ]
    path = tmp_path / "averages.vpy"
    path.write_text("\n".join([script, *checks, ""]), encoding="utf-8")
    assert clip_info(path, 0).num_frames == 3
