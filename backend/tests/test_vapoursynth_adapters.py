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
    return adapter_for(p, ENUMS, plugin).make_input("Label", p, input_id)


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
        ("int | None", "None", "range", "range"),
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


def test_make_input_takes_the_callers_id() -> None:
    for annotation in ("vs.VideoNode", "int", "Planes", "Fraction", "Any", "SPathLike"):
        assert widget(annotation, input_id=987654321).id == 987654321


def test_clip() -> None:
    a = adapt("vs.VideoNode", name="clip")
    inp = a.make_input("Clip", param("vs.VideoNode"), 7)
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

    # A parameter whose default is None: an optional text, empty = the default.
    optional = adapt("int | None", "None", "width")
    inp = widget("int | None", "None", "width")
    assert isinstance(inp, TextInput)
    assert inp.optional
    assert optional.to_source(None) is None
    assert optional.to_source("") is None
    assert optional.to_source(" 1280 ") == "1280"
    assert optional.to_source("-4") == "-4"
    with pytest.raises(ValueError, match="whole number"):
        optional.to_source("12.5")
    with pytest.raises(ValueError, match="whole number"):
        optional.to_source("ten")
    assert isinstance(widget("int | None", "None", "last", plugin=True), TextInput)


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
    assert isinstance(widget("float | None", "None", "blur"), TextInput)
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


def test_enum_from_the_catalog() -> None:
    # A nested enum, found by its key; the default's repr picks its option. Members
    # are written module-qualified (the import is the module), so that no two
    # imports of one script can shadow each other.
    a = adapt("BM3D.Profile", "<Profile.FAST: 'fast'>", "profile")
    assert a.imports == frozenset(["import vsdenoise"])
    inp = widget("BM3D.Profile", "<Profile.FAST: 'fast'>", "profile")
    assert isinstance(inp, DropDownInput)
    assert options(inp) == {
        "FAST": "vsdenoise.bm3d.Profile.FAST",
        "LOW COMPLEXITY": "vsdenoise.bm3d.Profile.LOW_COMPLEXITY",
        "NORMAL": "vsdenoise.bm3d.Profile.NORMAL",
        "HIGH": "vsdenoise.bm3d.Profile.HIGH",
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
    assert inp.options[0] == {"option": "Default", "value": DEFAULT}
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
    assert a.make_input("Bits", p, 5).to_dict()["def"] == 0
    assert a.make_input("Bits", p, 5, value=16).to_dict()["def"] == 16
    assert a.to_source(16) == "16"
    p = param("int | None", "None", "top")
    text = adapter_for(p, ENUMS).make_input("Top", p, 6, value="20")
    assert isinstance(text, TextInput)
    assert text.to_dict()["def"] == "20"
    p = param("vstools.enums.color.MatrixLike | None", "None", "matrix")
    dropdown = adapter_for(p, ENUMS).make_input("Matrix", p, 7, value="6")
    assert dropdown.to_dict()["def"] == "6"
    p = param("bool | None", "None", "linear")
    assert adapter_for(p, ENUMS).make_input("Linear", p, 8, value="True").to_dict()[
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


@pytest.mark.parametrize(
    "path",
    [
        r"C:\My Videos\clip 1.mkv",
        r"D:\owner's clips\it's.mkv",
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


def test_extra() -> None:
    a = adapt("dict[str, Any] | None", "None", "basic_args")
    inp = widget("dict[str, Any] | None", "None", "basic_args")
    assert isinstance(inp, TextInput)
    assert inp.optional
    assert inp.label == "Label (dict[str, Any] | None)"
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
        assert label == f"Label ({annotation})"


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
