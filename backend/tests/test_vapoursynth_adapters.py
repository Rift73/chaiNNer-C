"""Type adapters: a parameter of a vs-jetpack or plugin signature -> a chaiNNer input,
and the input's value -> the Python source of the argument (no VapourSynth)."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

# chaiNNer's inputs load the native backend, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

import navi
from nodes.impl.vapoursynth.adapters import DEFAULT, Adapter, adapter_for
from nodes.impl.vapoursynth.catalog_types import Param
from nodes.impl.vapoursynth.script import VideoClip, VSFilter
from nodes.properties.inputs import (
    DropDownInput,
    FileInput,
    NumberInput,
    TextInput,
    VideoClipInput,
    VSFilterInput,
)


def _enum(import_line: str, *exprs: str) -> dict[str, Any]:
    return {
        "import": import_line,
        "members": [{"name": e.rpartition(".")[2], "expr": e} for e in exprs],
    }


# The catalog's enums, in the scanner's shape: key (the dotted path annotations use)
# -> {"import", "members": [{"name", "expr"}]}.
ENUMS: dict[str, dict[str, Any]] = {
    "vsdenoise.prefilters.Prefilter": _enum(
        "from vsdenoise import Prefilter",
        "Prefilter.MINBLUR",
        "Prefilter.DFTTEST",
        "Prefilter.BM3D",
    ),
    "vsdenoise.blockmatch.BM3D.Profile": _enum(
        "from vsdenoise import bm3d",
        "bm3d.Profile.FAST",
        "bm3d.Profile.LOW_COMPLEXITY",
        "bm3d.Profile.NORMAL",
        "bm3d.Profile.HIGH",
    ),
    "vsdenoise.mvtools.presets.MVToolsPreset": _enum(
        "from vsdenoise import MVToolsPreset",
        "MVToolsPreset.HQ_COHERENCE",
        "MVToolsPreset.HQ_SAD",
    ),
    "vsdenoise.mvtools.enums.MVDirection": _enum(
        "from vsdenoise import MVDirection",
        "MVDirection.BACKWARD",
        "MVDirection.FORWARD",
    ),
    "vstools.enums.generic.FieldBasedLike": _enum(
        "from vstools import FieldBased",
        "FieldBased.PROGRESSIVE",
        "FieldBased.BFF",
        "FieldBased.TFF",
    ),
    "vapoursynth.FieldBased": _enum(
        "from vapoursynth import FieldBased",
        "FieldBased.FIELD_PROGRESSIVE",
        "FieldBased.FIELD_TOP",
    ),
    "vstools.functions.utils.DitherType": _enum(
        "from vstools import DitherType", "DitherType.NONE", "DitherType.RANDOM"
    ),
    "vskernels.abstract.base.KernelLike": {
        "import": "",
        "members": [{"name": k, "expr": repr(k)} for k in ("Catrom", "Mitchell")],
    },
}

CLIP = VideoClip(
    var="n_a_0",
    stmt="n_a_0 = core.std.BlankClip()",
    deps=(),
    imports=frozenset(),
    node_id="a",
)
FILTER = VSFilter(
    expr="vsaa.NNEDI3(pscrn=0, tff=True, double_rate=False).deinterlace",
    deps=(),
    imports=frozenset(["import vsaa"]),
    node_id="b",
)


def param(
    annotation: str,
    default: str | None = None,
    name: str = "x",
    kind: str = "keyword",
) -> Param:
    return Param(name=name, annotation=annotation, default=default, kind=kind)


def adapt(
    annotation: str,
    default: str | None = None,
    name: str = "x",
    kind: str = "keyword",
    plugin: bool = False,
) -> Adapter:
    return adapter_for(param(annotation, default, name, kind), ENUMS, plugin)


def widget(
    annotation: str,
    default: str | None = None,
    name: str = "x",
    kind: str = "keyword",
    input_id: int = 1234,
    plugin: bool = False,
) -> Any:
    """The input; its class depends on the kind, which each test checks."""
    p = param(annotation, default, name, kind)
    return adapter_for(p, ENUMS, plugin).make_input("Label", input_id)


def options(dropdown: DropDownInput) -> dict[str, str | int]:
    """Option label -> value."""
    return {o["option"]: o["value"] for o in dropdown.options}


# The annotations of vs-jetpack 2.2.6 (as source and as the catalog writes them,
# enums by their full keys), and their kinds.
@pytest.mark.parametrize(
    ("annotation", "default", "name", "kind"),
    [
        ("vs.VideoNode", None, "clip", "clip"),
        ("vs.VideoNode | None", "None", "ref", "optional_clip"),
        ("vs.VideoNode | Sequence[vs.VideoNode]", None, "clips", "clip"),
        ("Sequence[vs.VideoNode]", None, "clips", "clip"),
        (
            "vs.VideoNode | VSFunctionNoArgs | None",
            "None",
            "postprocess",
            "clip_or_filter",
        ),
        (
            "vs.VideoNode | PrefilterLike | VSFunctionNoArgs | None",
            "None",
            "prefilter",
            "clip_or_filter",
        ),
        ("VSFunctionNoArgs", None, "sharpener", "clip_or_filter"),
        ("int", "1", "refine", "int"),
        ("int | None", "None", "width", "int"),
        ("float", "12", "scthresh", "float"),
        ("float | None", "None", "blur", "float"),
        ("bool", "False", "export_globals", "bool"),
        ("bool | None", "None", "linear", "bool"),
        ("bool | tuple[Slope, Center]", "False", "sigmoid", "bool"),
        ("BM3D.Profile", "<Profile.FAST: 'fast'>", "profile", "enum"),
        ("PrefilterLike", None, "prefilter", "enum"),
        ("MVToolsPreset", None, "preset", "enum"),
        ("FieldBasedLike | bool | None", "None", "tff", "enum"),
        ("str | DitherType", "'random'", "dither_type", "enum"),
        ('Literal["h264", "ffv1"]', None, "fmt", "enum"),
        ("KernelLike", None, "kernel", "enum"),
        ("ScalerLike", None, "scaler", "enum"),
        ("ScalerLike | Literal[False]", None, "supersampler", "enum"),
        ("Planes", "None", "planes", "planes"),
        ("FrameRangeN | FrameRangesN", None, "ranges", "ranges"),
        (
            "FrameRangeN | FrameRangesN | _RangesCallBackLike | None",
            None,
            "ranges",
            "ranges",
        ),
        ("Fraction", None, "fps", "fraction"),
        ("VideoFormatLike", None, "format", "format"),
        ("VideoFormatLike | HoldsVideoFormat | int | None", "None", "bitdepth", "int"),
        ("MatrixLike", None, "matrix", "matrix"),
        ("TransferLike | None", "None", "transfer", "transfer"),
        ("PrimariesLike | None", "None", "primaries", "primaries"),
        ("RangeLike | None", "None", "range_in", "range"),
        ("int | Sequence[int]", "1", "radius", "sequence"),
        ("float | Sequence[float]", "0.5", "sigma", "sequence"),
        ("int | Sequence[int | None] | None", "None", "tr", "sequence"),
        ("int | tuple[int, int]", "16", "blksize", "sequence"),
        ("tuple[float, float] | Literal[False]", "False", "strength", "sequence"),
        ("SPathLike", None, "file", "path"),
        ("str | PathLike[str]", None, "path", "path"),
        ("SPathLike | None", "None", "out_file", "text"),
        ("str", "'random'", "dither_type", "text"),
        ("SupportsString | None", "None", "name", "text"),
        ("dict[str, Any] | None", "None", "basic_args", "extra"),
        ("Mapping[str, Any]", "{'a': 1}", "preset", "extra"),
        ("Any", None, "kwargs", "extra"),
        ("vsdenoise.blockmatch.BM3D.Profile", "bm3d.Profile.FAST", "profile", "enum"),
        (
            "vstools.enums.generic.FieldBasedLike | bool | None",
            "None",
            "tff",
            "enum",
        ),
        ("vstools.enums.color.MatrixLike | None", "None", "matrix", "matrix"),
        ("vskernels.abstract.base.KernelLike", "vskernels.Catrom", "kernel", "enum"),
    ],
)
def test_adapter_for_kinds(
    annotation: str, default: str | None, name: str, kind: str
) -> None:
    assert adapt(annotation, default, name).kind == kind


# Plugin parameters, as the catalog writes their signature types (scan.plugin_params:
# "T[]" is "T | Sequence[T]", ":opt" adds " | None" with the default None).
@pytest.mark.parametrize(
    ("annotation", "default", "name", "kind"),
    [
        ("VideoNode", None, "clipa", "clip"),
        ("VideoNode | None", "None", "src", "optional_clip"),
        ("VideoNode | Sequence[VideoNode]", None, "clips", "clip"),
        ("Callable[..., Any]", None, "selector", "clip_or_filter"),
        ("Callable[..., Any] | None", "None", "function", "clip_or_filter"),
        ("int", None, "first", "int"),
        ("int | None", "None", "last", "int"),
        ("float | None", "None", "sharpness", "float"),
        ("float | Sequence[float] | None", "None", "weight", "sequence"),
        ("int | Sequence[int]", None, "frames", "sequence"),
        ("int | Sequence[int] | None", "None", "planes", "planes"),
        ("int | None", "None", "format", "format"),
        ("int | None", "None", "matrix_in", "matrix"),
        ("int | None", "None", "transfer", "transfer"),
        ("int | None", "None", "primaries", "primaries"),
        ("int | None", "None", "range_in", "range"),
        ("str | bytes", None, "source", "path"),
        ("str | bytes | None", "None", "matrix_s", "text"),
        ("str | bytes | Sequence[str | bytes]", None, "expr", "text"),
        ("AudioNode", None, "clip", "extra"),
        ("Any", None, "kwargs", "extra"),
    ],
)
def test_plugin_adapter_kinds(
    annotation: str, default: str | None, name: str, kind: str
) -> None:
    assert adapt(annotation, default, name, plugin=True).kind == kind
    # The same names in a Python function follow its annotation instead.
    if name in ("format", "matrix_in"):
        assert adapt(annotation, default, name).kind == "int"


def test_plugin_range_is_a_colour_range_beside_range_in() -> None:
    """resize's `range` (beside `range_in`) is the colour range; vszip.Deband's
    `range` is f3kdb's banding radius, a number (default 15)."""
    p = param("int | None", "None", "range")
    resize = adapter_for(p, ENUMS, True, frozenset({"clip", "range", "range_in"}))
    assert resize.kind == "range"
    assert resize.to_source("1") == "1"
    deband = adapter_for(p, ENUMS, True, frozenset({"clip", "range", "thr"}))
    assert deband.kind == "int"
    assert deband.to_source("24") == "24"
    made = deband.make_input("Range", 5)
    assert isinstance(made, NumberInput)
    assert made.optional


def test_make_input_takes_the_callers_id() -> None:
    for annotation in ("vs.VideoNode", "int", "Planes", "Fraction", "Any", "SPathLike"):
        assert widget(annotation, input_id=987654321).id == 987654321


def test_clip() -> None:
    a = adapt("vs.VideoNode", name="clip")
    inp = a.make_input("Clip", 7)
    assert isinstance(inp, VideoClipInput)
    assert inp.input_type == "VideoClip"
    assert not inp.optional
    assert inp.label == "Clip"
    assert a.to_source(CLIP) == "n_a_0"
    # A plugin's clip array takes one clip as it is.
    clips = adapt("VideoNode | Sequence[VideoNode]", plugin=True)
    assert clips.to_source(CLIP) == "n_a_0"
    # A jetpack parameter that only takes a sequence of clips gets a list.
    assert adapt("Sequence[vs.VideoNode]").to_source(CLIP) == "[n_a_0]"
    with pytest.raises(ValueError):
        a.to_source(FILTER)
    # vstools' VideoNodeIterable (a clip, or clips in lists): norm_expr's and
    # stack_clips' clips, which also take a list of clips (shift_clip_multi's).
    many = VideoClip("n_m_0", "n_m_0 = f(n_a_0)", (CLIP,), frozenset(), "m", True)
    for annotation, one in (
        ("VideoNodeIterable", "n_a_0"),
        ("Iterable[VideoNodeIterable]", "[n_a_0]"),
    ):
        clips = adapt(annotation, name="clips")
        assert clips.kind == "clip"
        assert clips.to_source(CLIP) == one
        assert clips.to_source(many) == "n_m_0"


def test_video_node_iterable_is_a_clip() -> None:
    """vstools' VideoNodeIterable, a clip or (nested) lists of clips: the clips of
    vsexprtools.norm_expr and combine, and inside vstools.stack_clips' Iterable."""
    many = VideoClip("n_m_0", "n_m_0 = x", (), frozenset(), "m", many=True)
    a = adapt("VideoNodeIterable", None, "clips")
    assert a.kind == "clip"
    assert isinstance(widget("VideoNodeIterable", None, "clips"), VideoClipInput)
    assert a.to_source(CLIP) == "n_a_0"
    # It takes the list of a node that gives several clips as it is.
    assert a.to_source(many) == "n_m_0"
    a = adapt("Iterable[VideoNodeIterable]", None, "clips")
    assert a.kind == "clip"
    assert a.to_source(CLIP) == "[n_a_0]"
    assert a.to_source(many) == "n_m_0"
    assert adapt("VideoNodeIterable | None", "None", "ref").kind == "optional_clip"


def test_optional_clip() -> None:
    a = adapt("vs.VideoNode | None", "None", "ref")
    inp = widget("vs.VideoNode | None", "None", "ref")
    assert isinstance(inp, VideoClipInput)
    assert inp.optional
    assert a.to_source(None) is None
    assert a.to_source(CLIP) == "n_a_0"


def test_clip_or_filter() -> None:
    for annotation in (
        "vs.VideoNode | VSFunctionNoArgs | None",
        "vs.VideoNode | PrefilterLike | VSFunctionNoArgs | None",
    ):
        a = adapt(annotation, "None", "postprocess")
        inp = widget(annotation, "None", "postprocess")
        assert isinstance(inp, VSFilterInput)
        assert inp.input_type == navi.union("VideoClip", "VSFilter")
        assert inp.optional
        assert a.to_source(FILTER) == FILTER.expr
        assert a.to_source(CLIP) == "n_a_0"
        assert a.to_source(None) is None
    # Required callables are required sockets.
    assert not widget("VSFunctionNoArgs", None, "sharpener").optional
    assert not widget("Callable[..., Any]", None, "selector", plugin=True).optional
    assert widget("Callable[..., Any] | None", "None", "function", plugin=True).optional


def test_int() -> None:
    a = adapt("int", "1", "refine")
    inp = widget("int", "1", "refine")
    assert isinstance(inp, NumberInput)
    assert inp.precision == 0
    assert inp.default == 1
    assert inp.min is None
    assert inp.max is None
    assert a.to_source(4) == "4"
    # The function's own default is left out of the call.
    assert a.to_source(1) is None

    # Without a default the number is required and always written.
    required = adapt("int", None, "first")
    assert widget("int", None, "first").default == 0
    assert required.to_source(0) == "0"

    # A parameter whose default is None: a number field that may stay empty, empty =
    # the default.
    optional = adapt("int | None", "None", "width")
    inp = widget("int | None", "None", "width")
    assert isinstance(inp, NumberInput)
    assert inp.optional
    assert inp.default is None
    assert optional.to_source(None) is None
    assert optional.to_source(1280) == "1280"
    assert optional.to_source("") is None
    assert optional.to_source(" 1280 ") == "1280"
    assert optional.to_source("-4") == "-4"
    with pytest.raises(ValueError, match="whole number"):
        optional.to_source("12.5")
    with pytest.raises(ValueError, match="whole number"):
        optional.to_source("ten")
    assert isinstance(widget("int | None", "None", "last", plugin=True), NumberInput)


def test_float() -> None:
    a = adapt("float", "12", "scthresh")
    inp = widget("float", "12", "scthresh")
    assert isinstance(inp, NumberInput)
    assert inp.precision == "unlimited"
    assert inp.default == 12.0
    assert a.to_source(0.8) == "0.8"
    assert a.to_source(12.0) is None
    assert a.to_source(12) is None

    optional = adapt("float | None", "None", "blur")
    assert isinstance(widget("float | None", "None", "blur"), NumberInput)
    assert optional.to_source(0.2) == "0.2"
    assert optional.to_source("0.2") == "0.2"
    assert optional.to_source("1e-3") == "0.001"
    assert optional.to_source(None) is None
    for bad in ("fast", "nan", "inf"):
        with pytest.raises(ValueError, match="number"):
            optional.to_source(bad)


def test_bool() -> None:
    a = adapt("bool", "False", "export_globals")
    inp = widget("bool", "False", "export_globals")
    assert isinstance(inp, DropDownInput)
    assert inp.preferred_style == "checkbox"
    assert inp.default == 0
    assert a.to_source(True) == "True"
    assert a.to_source(False) is None
    assert adapt("bool", "True").to_source(False) == "False"

    # None means "decide for me": a third choice keeps it.
    a = adapt("bool | None", "None", "linear")
    inp = widget("bool | None", "None", "linear")
    assert isinstance(inp, DropDownInput)
    assert options(inp) == {"Default": DEFAULT, "Yes": "True", "No": "False"}
    assert inp.default == DEFAULT
    assert a.to_source(DEFAULT) is None
    assert a.to_source("True") == "True"
    assert a.to_source("False") == "False"


@pytest.mark.parametrize(
    ("annotation", "default", "kind", "other"),
    [
        # One kind of value: the input takes them all.
        ("bool", "False", "bool", False),
        ("float | Sequence[float]", "0.5", "sequence", False),
        ("vstools.enums.generic.FieldBasedLike | bool | None", "None", "enum", False),
        ("vskernels.abstract.base.KernelLike", "vskernels.Catrom", "enum", False),
        ("vs.VideoNode | VSFunctionNoArgs | None", "None", "clip_or_filter", False),
        ("Any", None, "extra", False),
        # bool wins over numbers and sequences, a number over Literal[False], an enum
        # over per-plane lists: a Python value input takes the rest.
        ("bool | float", "False", "bool", True),
        ("bool | tuple[int, int] | float", "True", "bool", True),
        ("bool | Sequence[bool]", "False", "bool", True),
        ("int | Literal[False]", "5", "int", True),
        (
            "int | vsdenoise.prefilters.Prefilter | Sequence[int]",
            "None",
            "enum",
            True,
        ),
        ("str | Sequence[str]", None, "text", True),
    ],
)
def test_union_with_values_its_input_cannot_give(
    annotation: str, default: str | None, kind: str, other: bool
) -> None:
    a = adapt(annotation, default, "x")
    assert (a.kind, a.other) == (kind, other)


def test_enum_beside_bool_offers_true_and_false() -> None:
    """ClampScaler.scale's limit (Repair.Mode | bool = True) could not be False."""
    names = options(widget("vsdenoise.prefilters.Prefilter | bool", "True", "limit"))
    assert names["True"] == "True"
    assert names["False"] == "False"
    a = adapt("vsdenoise.prefilters.Prefilter | bool", "True", "limit")
    assert a.to_source("False") == "False"
    assert a.to_source("True") is None
    assert a.to_source("vsdenoise.Prefilter.BM3D") == "vsdenoise.Prefilter.BM3D"


def test_enum_from_the_catalog() -> None:
    # A nested enum, found by its key; the default's repr picks its option. Members
    # are written module-qualified (the import is the module), so that no two
    # imports of one script can shadow each other.
    a = adapt("BM3D.Profile", "<Profile.FAST: 'fast'>", "profile")
    assert a.imports == frozenset(["import vsdenoise"])
    inp = widget("BM3D.Profile", "<Profile.FAST: 'fast'>", "profile")
    assert isinstance(inp, DropDownInput)
    assert options(inp) == {
        "Fast": "vsdenoise.bm3d.Profile.FAST",
        "Low complexity": "vsdenoise.bm3d.Profile.LOW_COMPLEXITY",
        "Normal": "vsdenoise.bm3d.Profile.NORMAL",
        "High": "vsdenoise.bm3d.Profile.HIGH",
    }
    assert inp.default == "vsdenoise.bm3d.Profile.FAST"
    assert a.to_source("vsdenoise.bm3d.Profile.HIGH") == "vsdenoise.bm3d.Profile.HIGH"
    assert a.to_source("vsdenoise.bm3d.Profile.FAST") is None
    with pytest.raises(ValueError):
        a.to_source("vsdenoise.bm3d.Profile.MISSING")
    # The unqualified form is no choice.
    with pytest.raises(ValueError):
        a.to_source("bm3d.Profile.HIGH")

    # The catalog's full keys, as its annotations are written; its defaults are
    # unqualified expressions.
    full = adapt("vsdenoise.blockmatch.BM3D.Profile", "bm3d.Profile.FAST", "profile")
    assert full.to_source("vsdenoise.bm3d.Profile.HIGH") == (
        "vsdenoise.bm3d.Profile.HIGH"
    )
    assert full.to_source("vsdenoise.bm3d.Profile.FAST") is None
    tff = adapt("vstools.enums.generic.FieldBasedLike | bool | None", "None", "tff")
    assert tff.to_source("vstools.FieldBased.TFF") == "vstools.FieldBased.TFF"
    assert tff.imports == frozenset(["import vstools"])
    # VapourSynth's own enums go through the script's `vs`, without an import.
    vs_tff = adapt("vapoursynth.FieldBased", "FieldBased.FIELD_TOP", "field_based")
    assert vs_tff.to_source("vs.FieldBased.FIELD_PROGRESSIVE") == (
        "vs.FieldBased.FIELD_PROGRESSIVE"
    )
    assert vs_tff.to_source("vs.FieldBased.FIELD_TOP") is None
    assert vs_tff.imports == frozenset()

    # "XLike" aliases find the enum X; presets are enums too.
    prefilter = adapt("PrefilterLike", None, "prefilter")
    assert prefilter.to_source("vsdenoise.Prefilter.DFTTEST") == (
        "vsdenoise.Prefilter.DFTTEST"
    )
    preset = adapt("MVToolsPreset", None, "preset")
    assert preset.to_source("vsdenoise.MVToolsPreset.HQ_SAD") == (
        "vsdenoise.MVToolsPreset.HQ_SAD"
    )
    # Required: the first option is the default and is always written.
    assert widget("MVToolsPreset", None, "preset").default == (
        "vsdenoise.MVToolsPreset.HQ_COHERENCE"
    )
    assert preset.to_source("vsdenoise.MVToolsPreset.HQ_COHERENCE") == (
        "vsdenoise.MVToolsPreset.HQ_COHERENCE"
    )

    # A default no option names gets a "Default" option that leaves it out.
    a = adapt("FieldBasedLike | bool | None", "None", "tff")
    inp = widget("FieldBasedLike | bool | None", "None", "tff")
    assert inp.options[0] == {"option": "As the video says", "value": DEFAULT}
    assert inp.default == DEFAULT
    assert a.to_source(DEFAULT) is None
    assert a.to_source("vstools.FieldBased.TFF") == "vstools.FieldBased.TFF"

    # A text default names its option case-insensitively.
    assert widget("str | DitherType", "'random'", "dither_type").default == (
        "vstools.DitherType.RANDOM"
    )


def test_literal_is_enum() -> None:
    a = adapt('Literal["h264", "ffv1"]', "'ffv1'", "fmt")
    inp = widget('Literal["h264", "ffv1"]', "'ffv1'", "fmt")
    assert options(inp) == {"h264": "'h264'", "ffv1": "'ffv1'"}
    assert inp.default == "'ffv1'"
    assert a.to_source("'h264'") == "'h264'"
    assert a.to_source("'ffv1'") is None
    # Enum members inside Literal, qualified like their enum's members.
    a = adapt("Literal[MVDirection.FORWARD, MVDirection.BACKWARD]")
    assert a.to_source("vsdenoise.MVDirection.BACKWARD") == (
        "vsdenoise.MVDirection.BACKWARD"
    )
    assert a.imports == frozenset(["import vsdenoise"])


def test_make_input_starts_with_a_given_value() -> None:
    """An overlay's default: the value the input starts with, which (unlike the
    function's default) is written."""
    p = param("int", None, "bitdepth")
    a = adapter_for(p, ENUMS)
    assert a.make_input("Bits", 5).to_dict()["def"] == 0
    assert a.make_input("Bits", 5, value=16).to_dict()["def"] == 16
    assert a.to_source(16) == "16"
    p = param("int | None", "None", "top")
    top = adapter_for(p, ENUMS)
    number = top.make_input("Top", 6, value=top.start_value(20))
    assert isinstance(number, NumberInput)
    assert number.optional
    assert number.to_dict()["def"] == 20
    # A start value written as a text is the number.
    assert top.start_value("20") == 20
    p = param("vstools.enums.color.MatrixLike | None", "None", "matrix")
    dropdown = adapter_for(p, ENUMS).make_input("Matrix", 7, value="6")
    assert dropdown.to_dict()["def"] == "6"
    p = param("bool | None", "None", "linear")
    assert adapter_for(p, ENUMS).make_input("Linear", 8, value="True").to_dict()[
        "def"
    ] == ("True")


def test_kernel_like_is_enum() -> None:
    catrom = "<class 'vskernels.kernels.zimg.bicubic.Catrom'>"
    a = adapt("ComplexScalerLike", catrom, "scaler")
    inp = widget("ComplexScalerLike", catrom, "scaler")
    names = options(inp)
    for kernel in ("Point", "Bilinear", "Catrom", "Mitchell", "Hermite", "Lanczos"):
        # Kernel parameters take the kernel's name as a string.
        assert names[kernel] == repr(kernel)
    assert inp.default == "'Catrom'"
    assert a.to_source("'Mitchell'") == "'Mitchell'"
    assert a.to_source("'Catrom'") is None
    assert ast.literal_eval(a.to_source("'Hermite'") or "") == "Hermite"
    # Literal values join the kernel names.
    names = options(widget("ScalerLike | Literal[False]", None, "supersampler"))
    assert names["False"] == "False"
    assert names["Lanczos"] == "'Lanczos'"
    # The catalog's kernel alias wins; string choices need no import.
    a = adapt("vskernels.abstract.base.KernelLike", "vskernels.Catrom", "kernel")
    assert a.choices == (("Catrom", "'Catrom'"), ("Mitchell", "'Mitchell'"))
    assert a.default == "'Catrom'"
    assert a.imports == frozenset()


def test_planes() -> None:
    a = adapt("Planes", "None", "planes")
    inp = widget("Planes", "None", "planes")
    assert isinstance(inp, DropDownInput)
    assert options(inp) == {
        "Default": DEFAULT,
        "Y, U and V": "[0, 1, 2]",
        "Y": "[0]",
        "U and V": "[1, 2]",
        "U": "[1]",
        "V": "[2]",
        "Y and U": "[0, 1]",
        "Y and V": "[0, 2]",
    }
    assert inp.default == DEFAULT
    assert a.to_source(DEFAULT) is None
    assert a.to_source("[0]") == "[0]"
    assert a.to_source("[1, 2]") == "[1, 2]"
    assert a.to_source("[0, 1, 2]") == "[0, 1, 2]"

    # A plane default picks its option and is left out of the call.
    for default in ("0", "[0]", "(0,)"):
        a = adapt("Planes", default, "planes")
        assert widget("Planes", default, "planes").default == "[0]"
        assert a.to_source("[0]") is None
        assert a.to_source("[0, 1, 2]") == "[0, 1, 2]"
    # Plugins: an int array named planes.
    a = adapt("int | Sequence[int] | None", "None", "planes", plugin=True)
    assert a.to_source("[1, 2]") == "[1, 2]"
    # A required one lists a source plane per output plane (std.ShufflePlanes): any
    # order and repeats, typed; a value saved from the old dropdown still runs.
    a = adapt("int | Sequence[int]", None, "planes", plugin=True)
    assert a.kind == "sequence"
    assert a.to_source("2, 1, 0") == "[2, 1, 0]"
    assert a.to_source("0, 0, 0") == "[0, 0, 0]"
    assert a.to_source("0") == "0"
    assert a.to_source("[0, 1, 2]") == "[0, 1, 2]"


def test_ranges_text() -> None:
    a = adapt("FrameRangeN | FrameRangesN", None, "ranges")
    inp = widget("FrameRangeN | FrameRangesN", None, "ranges")
    assert isinstance(inp, TextInput)
    src = a.to_source("682-687, 713-717, 721")
    assert src == "[(682, 687), (713, 717), 721]"
    assert ast.literal_eval(src) == [(682, 687), (713, 717), 721]
    assert a.to_source(" 0-0,5 ,  7 - 9, ") == "[(0, 0), 5, (7, 9)]"
    assert a.to_source("721") == "[721]"
    assert a.to_source("") is None
    assert a.to_source(None) is None
    with pytest.raises(ValueError, match="ends before it starts"):
        a.to_source("10-5")
    for bad in ("1-2-3", "a-b", "1.5", "1, , 2", "-3"):
        with pytest.raises(ValueError, match="frame number or a range"):
            a.to_source(bad)


def test_fraction() -> None:
    a = adapt("Fraction", None, "fps")
    inp = widget("Fraction", None, "fps")
    assert isinstance(inp, TextInput)
    assert a.imports == frozenset(["import fractions"])
    assert a.to_source("24000/1001") == "fractions.Fraction(24000, 1001)"
    assert a.to_source(" 30000 / 1001 ") == "fractions.Fraction(30000, 1001)"
    assert a.to_source("25") == "fractions.Fraction(25, 1)"
    assert a.to_source("50/2") == "fractions.Fraction(25, 1)"
    assert a.to_source("23.976") == "fractions.Fraction(2997, 125)"
    assert a.to_source("") is None
    for bad in ("fast", "1/0", "24000/1001/2"):
        with pytest.raises(ValueError, match="fraction"):
            a.to_source(bad)
    # A Fraction default shows as text and is left out of the call.
    a = adapt("Fraction | None", "Fraction(30000, 1001)", "fps")
    assert widget("Fraction | None", "Fraction(30000, 1001)", "fps").default == (
        "30000/1001"
    )
    assert a.to_source("30000/1001") is None
    # Other kinds need no import.
    assert adapt("int").imports == frozenset()


def test_format() -> None:
    a = adapt("VideoFormatLike", None, "format")
    inp = widget("VideoFormatLike", None, "format")
    assert isinstance(inp, DropDownInput)
    names = options(inp)
    assert names["YUV444P16"] == "vs.YUV444P16"
    assert names["RGBS"] == "vs.RGBS"
    assert names["GRAY8"] == "vs.GRAY8"
    assert len(names) == 53
    assert [g.label for g in inp.groups] == ["Gray", "YUV", "RGB"]
    assert a.to_source("vs.YUV444P16") == "vs.YUV444P16"
    # Plugins: an optional int named format.
    a = adapt("int | None", "None", "format", plugin=True)
    inp = widget("int | None", "None", "format", plugin=True)
    assert inp.default == DEFAULT
    assert a.to_source(DEFAULT) is None
    assert a.to_source("vs.RGBS") == "vs.RGBS"
    # A preset default picks its option.
    default = "<PresetVideoFormat.RGBS: 555745280>"
    assert widget("VideoFormatLike", default, "format").default == "vs.RGBS"
    assert adapt("VideoFormatLike", default, "format").to_source("vs.RGBS") is None


def test_colour_family() -> None:
    # std.ShufflePlanes' colorfamily: VapourSynth's names, not a number to know.
    a = adapt("int", None, "colorfamily", plugin=True)
    inp = widget("int", None, "colorfamily", plugin=True)
    assert isinstance(inp, DropDownInput)
    assert options(inp) == {"Gray": "vs.GRAY", "RGB": "vs.RGB", "YUV": "vs.YUV"}
    assert a.to_source("vs.RGB") == "vs.RGB"
    # A Python function's colorfamily follows its annotation.
    assert adapt("int", None, "colorfamily").kind == "int"


def test_colour_properties() -> None:
    a = adapt("MatrixLike", None, "matrix")
    names = options(widget("MatrixLike", None, "matrix"))
    assert names["BT.709 (1)"] == "1"
    assert names["BT.601 / SMPTE 170M (6)"] == "6"
    assert a.to_source("6") == "6"
    names = options(widget("TransferLike | None", "None", "transfer"))
    assert names["Default"] == DEFAULT
    assert names["BT.601 (6)"] == "6"
    assert names["sRGB / IEC 61966-2-1 (13)"] == "13"
    names = options(widget("PrimariesLike | None", "None", "primaries"))
    assert names["BT.2020 (9)"] == "9"
    names = options(widget("int | None", "None", "range_in", plugin=True))
    assert names == {"Default": DEFAULT, "Limited (0)": "0", "Full (1)": "1"}
    # An enum default ("<Matrix.BT709: 1>") or a number picks its option.
    assert widget("MatrixLike", "<Matrix.BT709: 1>", "matrix").default == "1"
    assert widget("RangeLike", "1", "range").default == "1"
    assert adapt("RangeLike", "1", "range").to_source("1") is None
    assert adapt("RangeLike", "1", "range").to_source("0") == "0"


def test_sequence() -> None:
    a = adapt("float | Sequence[float]", "0.5", "sigma")
    inp = widget("float | Sequence[float]", "0.5", "sigma")
    assert isinstance(inp, TextInput)
    assert inp.default == "0.5"
    assert a.to_source("0.35, 0") == "[0.35, 0]"
    assert a.to_source("0.8") == "0.8"
    assert a.to_source("0.5") is None
    assert a.to_source("") is None

    a = adapt("int | tuple[int, int]", "16", "blksize")
    assert a.to_source("16, 8") == "(16, 8)"
    assert a.to_source("8") == "8"
    with pytest.raises(ValueError, match="whole number"):
        a.to_source("1.5")

    # Only a tuple: one value is still a tuple; the default shows as text.
    a = adapt("tuple[int, int]", "(16, 16)", "y")
    assert widget("tuple[int, int]", "(16, 16)", "y").default == "16, 16"
    assert a.to_source("16, 16") is None
    assert a.to_source("8") == "(8,)"
    assert a.to_source("8, 4") == "(8, 4)"

    # None inside the sequence.
    a = adapt("int | Sequence[int | None] | None", "None", "tr")
    assert a.to_source("4, None") == "[4, None]"
    with pytest.raises(ValueError, match="number"):
        adapt("float | Sequence[float]").to_source("1, None")

    # Plugin arrays take one value as it is.
    a = adapt("float | Sequence[float] | None", "None", "weight", plugin=True)
    assert a.to_source("0.35, 0") == "[0.35, 0]"
    assert a.to_source("0.5") == "0.5"
    with pytest.raises(ValueError, match="number"):
        a.to_source("0.35,, 0")


def test_path() -> None:
    a = adapt("SPathLike", None, "file")
    inp = widget("SPathLike", None, "file")
    assert isinstance(inp, FileInput)
    assert ".mkv" in inp.filetypes
    assert ".vob" in inp.filetypes
    assert not inp.optional
    assert a.to_source(Path("C:/clips/a.mkv")) == repr(str(Path("C:/clips/a.mkv")))
    assert widget("SPathLike | None", "None", "model").optional
    assert widget("SPathLike | None", "None", "model").filetypes == ["*"]
    assert adapt("SPathLike | None", "None", "model").to_source(None) is None
    assert isinstance(widget("str | bytes", None, "source", plugin=True), FileInput)
    # vstools' FilePathType (HardsubASS' subtitle file): a file of any type, also
    # when it is named like a video source; a folder stays a text.
    subtitles = adapt("FilePathType", None, "filename")
    assert subtitles.kind == "path"
    assert widget("FilePathType", None, "filename").filetypes == ["*"]
    path = "C:\\subs\\new.ass"
    assert ast.literal_eval(subtitles.to_source(path) or "") == path
    assert adapt("FilePathType", None, "folder_path").kind == "text"


@pytest.mark.parametrize(
    "path",
    [
        r"C:\My Videos\clip 1.mkv",
        r"D:\my clips\it's.mkv",
        "E:\\ünïcødé\\日本語\\clip.mkv",
        r"\\server\share\a'b\"c.mkv",
        "F:\\new\\table\\x.mkv",
        "C:\\trailing\\",
    ],
)
def test_path_literal_roundtrip(path: str) -> None:
    a = adapt("SPathLike", None, "file")
    src = a.to_source(path)
    assert src is not None
    assert ast.literal_eval(src) == path
    src = a.to_source(Path(path))
    assert src is not None
    assert ast.literal_eval(src) == str(Path(path))
    # The same holds for paths typed as text (output files).
    src = adapt("SPathLike | None", "None", "out_file").to_source(path)
    assert src is not None
    assert ast.literal_eval(src) == path


def test_text() -> None:
    a = adapt("str", "'random'", "dither_type")
    inp = widget("str", "'random'", "dither_type")
    assert isinstance(inp, TextInput)
    assert inp.default == "random"
    assert a.to_source("error_diffusion") == "'error_diffusion'"
    assert a.to_source("random") is None
    assert ast.literal_eval(a.to_source('it\'s "x"\n') or "") == 'it\'s "x"\n'
    expr = adapt("str | bytes", None, "expr", plugin=True)
    assert expr.to_source("x y +") == "'x y +'"
    matrix_s = adapt("str | bytes | None", "None", "matrix_s", plugin=True)
    assert matrix_s.to_source(None) is None


def test_expressions_per_plane() -> None:
    # A plugin's text array (std.Expr's expr) takes one expression for every plane,
    # or a list with one per plane ('' copies the plane).
    exprs = adapt("str | bytes | Sequence[str | bytes]", None, "expr", plugin=True)
    assert exprs.kind == "text"
    assert exprs.to_source("x 2 *") == "'x 2 *'"
    assert exprs.to_source("['x 2 *', '']") == "['x 2 *', '']"
    assert exprs.to_source(' ["x 2 *",\n "x"] ') == "['x 2 *', 'x']"
    for bad in ("[1, 2]", "['x' + 'y']", "[x 2 *]"):
        with pytest.raises(ValueError, match="list of texts"):
            exprs.to_source(bad)
    inp = widget("str | bytes | Sequence[str | bytes]", None, "expr", plugin=True)
    assert isinstance(inp, TextInput)
    assert inp.placeholder == "e.g. x 2 *"
    # One text stays one text, brackets and all.
    one = adapt("str | bytes", None, "expr", plugin=True)
    assert one.to_source("['x']") == repr("['x']")
    # vsexprtools' ExprLike (norm_expr): a plain text, no quotes to type.
    norm = adapt("ExprLike", None, "expr")
    assert norm.kind == "text"
    assert norm.to_source("x 2 /") == "'x 2 /'"
    assert norm.to_source("['x 2 /', 'x']") == "['x 2 /', 'x']"
    # A chain saved with the quoted text the Python-value box took still runs.
    assert norm.to_source("'x 2 /'") == "'x 2 /'"


def test_norm_expr_takes_the_literals_its_python_value_box_took() -> None:
    # ExprLike also takes a tuple (an expression per plane), None and numbers, and
    # lists or tuples inside: the values the Python-value box saved still run.
    norm = adapt("ExprLike", None, "expr")
    for saved in (
        "('x 2 /', 'x')",
        "('x 2 /',)",
        "['x 2 /', None]",
        "[['x', '2', '/'], 'x']",
        "(['x', 2, '/'], None, ('x', 0.5, '*'))",
    ):
        assert norm.to_source(saved) == saved
    assert norm.to_source(' ( "x 2 /",\n None ) ') == "('x 2 /', None)"
    for bad in ("('x' + 'y',)", "[True]", "['x', 1e400]", "[b'x']", "[{'x': 1}]"):
        with pytest.raises(ValueError, match=r"tuple of texts like \('x 2 \*', ''\)"):
            norm.to_source(bad)
    # Its hint shows the tuple, an expression per plane (a list is one expression).
    inp = widget("ExprLike", None, "expr")
    assert isinstance(inp, TextInput)
    assert inp.placeholder == "e.g. x 2 *"
    # std.Expr's expr keeps its flat list of texts, and a text in parentheses is
    # one expression.
    exprs = adapt("str | bytes | Sequence[str | bytes]", None, "expr", plugin=True)
    for bad in ("['x 2 *', None]", "[['x'], 'x']"):
        with pytest.raises(ValueError, match="list of texts"):
            exprs.to_source(bad)
    assert exprs.to_source("('x', 'y')") == repr("('x', 'y')")


def test_extra() -> None:
    a = adapt("dict[str, Any] | None", "None", "basic_args")
    inp = widget("dict[str, Any] | None", "None", "basic_args")
    assert isinstance(inp, TextInput)
    assert inp.optional
    assert inp.label == "Label"
    assert a.to_source(None) is None
    assert a.to_source("") is None
    assert a.to_source("{'bs': 8}") == "{'bs': 8}"
    # Written as Python writes the value back: one form for one value.
    assert a.to_source(' {"bs": 8, "x": [1, 2.5, None, True]} ') == (
        "{'bs': 8, 'x': [1, 2.5, None, True]}"
    )
    # A bare tuple would be two arguments in the call.
    assert a.to_source("1, 2") == "(1, 2)"
    assert a.to_source("1,") == "(1,)"
    # A comment would comment out the rest of the statement: written without it.
    assert a.to_source("{'bs': 8}  # mine") == "{'bs': 8}"
    assert a.to_source("{'bs':\n 8}") == "{'bs': 8}"
    assert a.to_source("{'bs':\r8,\t'x': 1}") == "{'bs': 8, 'x': 1}"
    # A literal default is shown and left out of the call.
    a = adapt("Mapping[str, Any]", "{'a': 1}", "preset")
    assert widget("Mapping[str, Any]", "{'a': 1}", "preset").default == "{'a': 1}"
    assert a.to_source("{'a': 1}") is None
    assert a.to_source('{"a": 1}') is None
    assert a.to_source("{'a': 2}") == "{'a': 2}"


def test_unknown_annotation_is_extra() -> None:
    for annotation in (
        "SomethingNew",
        "SomethingNew[int, str]",
        "tuple[TopShift | list[TopShift], LeftShift | list[LeftShift]]",
        "P.kwargs",
        "int |",
    ):
        a = adapt(annotation, None, "x")
        assert a.kind == "extra"
        assert a.to_source("[1, 2]") == "[1, 2]"
        label = widget(annotation, None, "x").label
        assert label == "Label"


def test_extra_rejects_code() -> None:
    a = adapt("Any", None, "x")
    for code in (
        "__import__('os')",
        "__import__('os').system('calc')",
        "open('x')",
        "1 + x",
        "lambda: 1",
        "[x for x in ()]",
        "{'a': f()}",
    ):
        with pytest.raises(ValueError, match="not a plain Python value"):
            a.to_source(code)


def test_var_keyword_is_extra() -> None:
    a = adapt("Any", None, "kwargs", kind="var_keyword")
    assert a.kind == "extra"
    inp = widget("Any", None, "kwargs", kind="var_keyword")
    assert inp.optional
    # Its label is the caller's (the annotation says nothing).
    assert inp.label == "Label"
    assert a.to_source(None) is None
    assert a.to_source("{'pscrn': 0, 'nsize': 1}") == "{'pscrn': 0, 'nsize': 1}"
    for bad in ("[1]", "{1: 2}", "{'not a name': 2}", "3"):
        with pytest.raises(ValueError, match="names"):
            a.to_source(bad)
    # Whatever the annotation of **kwargs says.
    assert adapt("int", None, "kwargs", kind="var_keyword").kind == "extra"


def test_unannotated_parameters_follow_their_default() -> None:
    assert adapt("", "10", "radius").kind == "int"
    assert adapt("", "0.5", "strength").kind == "float"
    assert adapt("", "False", "fast").kind == "bool"
    assert adapt("", "'x'", "mode").kind == "text"
    assert adapt("", "[0, 1, 2]", "planes").kind == "sequence"
    assert adapt("", "[0, 1, 2]", "planes").to_source("1, 2") == "[1, 2]"
    assert adapt("", "None", "ref").kind == "extra"
    assert adapt("", None, "clip").kind == "extra"


@pytest.mark.parametrize(
    ("annotation", "default", "name", "plugin", "placeholder", "example"),
    [
        ("int | Sequence[int] | None", "None", "radius", False, "e.g. 0, 2", "0, 2"),
        ("float | Sequence[float]", "0.5", "sigma", False, "e.g. 0.35, 0", "0.35, 0"),
        ("int | Sequence[int]", None, "map", True, "e.g. 0, 2", "0, 2"),
        ("FrameRangesN", "None", "ranges", False, "e.g. 10-20, 35", "10-20, 35"),
        ("Fraction", None, "fps", False, "e.g. 24000/1001", "24000/1001"),
        ("ExprLike", None, "expr", False, "e.g. x 2 *", "x 2 *"),
        ("str | bytes | Sequence[str | bytes]", None, "expr", True, "e.g. x 2 *", None),
        (
            "str | bytes | Sequence[str | bytes]",
            None,
            "props",
            True,
            "text, or ['a', 'b']",
            "['a', 'b']",
        ),
        ("SomethingNew", None, "x", False, "Python value, e.g. [1, 2]", "[1, 2]"),
    ],
)
def test_placeholders_fit_and_parse(
    annotation: str,
    default: str | None,
    name: str,
    plugin: bool,
    placeholder: str,
    example: str | None,
) -> None:
    # The node shows about 31 characters; the example after "e.g." is a value the
    # box takes.
    p = param(annotation, default, name)
    a = adapter_for(p, ENUMS, plugin)
    inp = a.make_input("Label", 1)
    assert isinstance(inp, TextInput)
    assert inp.placeholder == placeholder
    assert len(placeholder) <= 28
    if example is not None:
        assert a.to_source(example) is not None


def test_more_arguments_placeholder() -> None:
    a = adapt("Any", None, "kwargs", kind="var_keyword")
    inp = widget("Any", None, "kwargs", kind="var_keyword")
    assert inp.placeholder == "e.g. {'name': 1}"
    assert a.to_source("{'name': 1}") == "{'name': 1}"


def test_missing_value_says_what_to_type() -> None:
    assert adapt("FrameRangesN", None, "ranges").missing("Frames") == (
        "Frames needs a value: type one in its box (e.g. 10-20, 35)."
    )
    assert adapt("str", None, "name").missing("Name") == (
        "Name needs a value: type one in its box."
    )
    # A socket or a dropdown is not typed into.
    assert adapt("VideoNode", None, "ref").missing("Ref") == "Ref needs a value."
    assert adapt("MatrixLike", None, "matrix").missing("Matrix") == (
        "Matrix needs a value."
    )


def test_field_order_choices_read_as_words() -> None:
    inp = widget("vstools.enums.generic.FieldBasedLike | bool | None", "None", "tff")
    # Top field first first; no True and False (they repeat TFF and BFF); no
    # Progressive for a `tff`, which every such function refuses.
    assert list(options(inp).items()) == [
        ("As the video says", DEFAULT),
        ("Top field first (TFF)", "vstools.FieldBased.TFF"),
        ("Bottom field first (BFF)", "vstools.FieldBased.BFF"),
    ]
    # A saved True is still a value the function takes (top field first).
    assert adapt(
        "vstools.enums.generic.FieldBasedLike | bool", "True", "tff"
    ).default == ("vstools.FieldBased.TFF")
    # Required: starts at its first choice, which is always written.
    required = adapt("vstools.enums.generic.FieldBasedLike | bool", None, "tff")
    assert required.to_source("vstools.FieldBased.TFF") == "vstools.FieldBased.TFF"
    # A field_based (a kernel's) keeps Progressive: scaling progressive video.
    assert list(
        options(
            widget("vstools.enums.generic.FieldBasedLike | None", "None", "field_based")
        )
    ) == [
        "As the video says",
        "Top field first (TFF)",
        "Bottom field first (BFF)",
        "Progressive",
    ]
    # Another dropdown's Default and booleans keep their names.
    names = options(widget("vsdenoise.prefilters.Prefilter | bool | None", "None"))
    assert names["Default"] == DEFAULT
    assert names["True"] == "True"


def test_enum_names_of_several_words_are_in_sentence_case() -> None:
    a = adapt("MVToolsPreset", None, "preset")
    assert [label for label, _ in a.choices if label != "Default"] == [
        "HQ coherence",
        "HQ SAD",
    ]
    # One-word names too; acronyms and codes stay (BM3D, MINBLUR), and so do Literal
    # texts and kernel names.
    assert "Normal" in options(widget("BM3D.Profile", None, "profile"))
    assert set(options(widget("vsdenoise.prefilters.Prefilter", None, "p"))) == {
        "MINBLUR",
        "DFTTEST",
        "BM3D",
    }
    assert set(options(widget("DitherType", None, "dither"))) == {"None", "Random"}
    assert "Catrom" in options(widget("KernelLike", None, "kernel"))
    assert set(options(widget("Literal['C4F16_DN', 'x_y']", None, "model"))) >= {
        "C4F16_DN",
        "x_y",
    }


def test_a_number_beside_a_clip_or_a_function_is_typed() -> None:
    # Dehalo's blur: a sigma, a list per plane or per iteration; the socket took none.
    blur = adapt(
        "IterArr[float] | VSFunctionPlanesArgs"
        " | tuple[float | list[float] | VSFunctionPlanesArgs, ...]",
        "<vsdenoise.prefilters.PrefilterPartial object>",
        "blur",
    )
    assert blur.kind == "extra"
    assert blur.to_source("1.4") == "1.4"
    assert blur.to_source("(1.4, [1.4, 1.65])") == "(1.4, [1.4, 1.65])"
    # finalize_clip's bits: a number of bits, 10 by default.
    bits = adapt(
        "vstools.vs_proxy.enums.PresetVideoFormat | VideoFormat | VideoNode | VideoFrame"
        " | int | None",
        "10",
        "bits",
    )
    assert (bits.kind, bits.default) == ("int", 10)
    # A fade's colour: a value, or one per plane.
    colour = adapt("float | Sequence[float] | VideoNode", "0.0", "color")
    assert colour.kind == "sequence"
    assert colour.to_source("0.5, 0") == "[0.5, 0]"
    # DPIR's strength (10 by default).
    strength = adapt(
        "SupportsFloat | VideoNode | None | Sequence[SupportsFloat | VideoNode | None]",
        "10",
        "strength",
    )
    assert (strength.kind, strength.default) == ("float", 10.0)
    # A clip or a function alone keeps its socket.
    assert adapt("VideoNode | VSFunctionNoArgs | None", "None", "x").kind == (
        "clip_or_filter"
    )
    assert adapt("_DebanderFunc[Any]", "vsdeband.f3k_deband", "debander").kind == (
        "clip_or_filter"
    )


def test_a_format_or_a_kernel_beside_a_clip_is_a_dropdown() -> None:
    fmt = adapt(
        "SupportsInt | vstools.vs_proxy.enums.PresetVideoFormat | VideoFormat"
        " | VideoNode | VideoFrame",
        None,
        "format",
    )
    assert fmt.kind == "format"
    assert fmt.to_source("vs.YUV444PS") == "vs.YUV444PS"
    reference = adapt(
        "vskernels.abstract.base.ScalerLike | VideoNode", None, "reference"
    )
    assert reference.kind == "enum"
    assert reference.to_source("'Catrom'") == "'Catrom'"


CONV = {
    "vstools.enums.funcs.ConvMode": _enum(
        "from vstools import ConvMode",
        *(
            f"ConvMode.{n}"
            for n in ("SQUARE", "VERTICAL", "HORIZONTAL", "HV", "TEMPORAL")
        ),
    ),
    "vsexprtools.exprop.ExprOp": _enum(
        "from vsexprtools import ExprOp", "ExprOp.MAX", "ExprOp.MIN", "ExprOp.ADD"
    ),
}


def test_type_aliases_of_enums_are_dropdowns() -> None:
    mode = adapter_for(
        param("OneDimConvMode | TempConvMode", "ConvMode.HV", "mode"), CONV
    )
    assert [s for _, s in mode.choices] == [
        "vstools.ConvMode.VERTICAL",
        "vstools.ConvMode.HORIZONTAL",
        "vstools.ConvMode.HV",
        "vstools.ConvMode.TEMPORAL",
    ]
    assert mode.default == "vstools.ConvMode.HV"
    assert mode.imports == frozenset(["import vstools"])
    assert mode.to_source("vstools.ConvMode.TEMPORAL") == "vstools.ConvMode.TEMPORAL"
    operator = adapter_for(
        param("vsexprtools.exprop.ExprOpBase", "ExprOp.MAX", "operator"), CONV
    )
    assert operator.kind == "enum"
    assert operator.default == "vsexprtools.ExprOp.MAX"
    assert operator.to_source("vsexprtools.ExprOp.ADD") == "vsexprtools.ExprOp.ADD"


def test_a_switch_or_a_function_is_the_switch() -> None:
    progress = adapt(
        "bool | Callable[[float], None] | None", "False", "show_pretty_progress"
    )
    assert (progress.kind, progress.default) == ("bool", False)


def test_a_format_or_an_int_is_bits_only_when_named_so() -> None:
    zones = adapt(
        "int | vstools.vs_proxy.enums.PresetVideoFormat | VideoFormat | VideoNode"
        " | VideoFrame",
        "PresetVideoFormat.GRAYS",
        "format",
    )
    assert (zones.kind, zones.default) == ("format", "vs.GRAYS")
    # A required clip beside a number keeps its socket (based_diff_mask's ref).
    assert adapt("VideoNode | int", None, "ref_or_height").kind == "clip"
