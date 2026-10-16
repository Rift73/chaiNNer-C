"""Type adapters: each maps one kind of Python parameter (a vs-jetpack annotation or a
plugin signature type) to a chaiNNer input, and the input's value to the Python source
of the argument in the generated script."""

from __future__ import annotations

import ast
import math
import os
import re
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Any, Mapping

from api import BaseInput

from ...properties.inputs import (
    BoolInput,
    DropDownGroup,
    DropDownInput,
    FileInput,
    NumberInput,
    TextInput,
    VideoClipInput,
    VSFilterInput,
)
from .catalog_types import Param, qualified
from .script import VideoClip, VSFilter, one_clip

# The catalog's enums map: annotation key -> {"import", "members": [{"name", "expr"}]}.
Enums = Mapping[str, Mapping[str, Any]]

# The dropdown choice that keeps the function's own default. It is not valid Python,
# so no choice's source can equal it.
DEFAULT = "<default>"

# H.273 numbers, as VapourSynth and vs-jetpack take them, with plain names.
MATRICES = (
    (0, "RGB"),
    (1, "BT.709"),
    (2, "Unspecified"),
    (4, "FCC"),
    (5, "BT.470BG"),
    (6, "BT.601 / SMPTE 170M"),
    (7, "SMPTE 240M"),
    (8, "YCgCo"),
    (9, "BT.2020 non-constant"),
    (10, "BT.2020 constant"),
    (12, "Chromaticity-derived non-constant"),
    (13, "Chromaticity-derived constant"),
    (14, "ICtCp"),
)
TRANSFERS = (
    (1, "BT.709"),
    (2, "Unspecified"),
    (4, "BT.470M / gamma 2.2"),
    (5, "BT.470BG / gamma 2.8"),
    (6, "BT.601"),
    (7, "SMPTE 240M"),
    (8, "Linear"),
    (9, "Log 100:1"),
    (10, "Log 316:1"),
    (11, "xvYCC / IEC 61966-2-4"),
    (13, "sRGB / IEC 61966-2-1"),
    (14, "BT.2020 10-bit"),
    (15, "BT.2020 12-bit"),
    (16, "PQ / SMPTE 2084"),
    (17, "SMPTE 428"),
    (18, "HLG / ARIB B67"),
)
PRIMARIES = (
    (1, "BT.709"),
    (2, "Unspecified"),
    (4, "BT.470M"),
    (5, "BT.470BG"),
    (6, "SMPTE 170M"),
    (7, "SMPTE 240M"),
    (8, "Film"),
    (9, "BT.2020"),
    (10, "XYZ / SMPTE 428"),
    (11, "DCI-P3 / SMPTE 431-2"),
    (12, "Display P3 / SMPTE 432-1"),
    (22, "EBU 3213-E"),
)
# Since R74 the frame property and the resize plugin agree: 0 limited, 1 full.
RANGES = ((0, "Limited"), (1, "Full"))
_COLOUR = {
    "matrix": MATRICES,
    "transfer": TRANSFERS,
    "primaries": PRIMARIES,
    "range": RANGES,
}

# VapourSynth's preset formats (vs.PresetVideoFormat without NONE), in its order.
FORMATS = (
    "GRAY8 GRAY9 GRAY10 GRAY12 GRAY14 GRAY16 GRAY32 GRAYH GRAYS "
    "YUV410P8 YUV411P8 YUV440P8 YUV420P8 YUV422P8 YUV444P8 YUV420P9 YUV422P9 YUV444P9 "
    "YUV420P10 YUV422P10 YUV444P10 YUV420P12 YUV422P12 YUV444P12 YUV420P14 YUV422P14 "
    "YUV444P14 YUV410P16 YUV411P16 YUV440P16 YUV420P16 YUV422P16 YUV444P16 YUV410PH "
    "YUV410PS YUV411PH YUV411PS YUV440PH YUV440PS YUV420PH YUV420PS YUV422PH YUV422PS "
    "YUV444PH YUV444PS RGB24 RGB27 RGB30 RGB36 RGB42 RGB48 RGBH RGBS"
).split()
_FORMAT_CHOICES = tuple((f, f"vs.{f}") for f in FORMATS)
_FORMAT_GROUPS = (("Gray", "vs.GRAY8"), ("YUV", "vs.YUV410P8"), ("RGB", "vs.RGB24"))

# vskernels kernels; kernel and scaler parameters take the class name as a string.
KERNELS = (
    "Point",
    "Bilinear",
    "Bicubic",
    "Catrom",
    "Mitchell",
    "Hermite",
    "BSpline",
    "BicubicSharp",
    "Robidoux",
    "RobidouxSoft",
    "RobidouxSharp",
    "Lanczos",
    "Spline16",
    "Spline36",
    "Spline64",
    "Gaussian",
    "Box",
    "Sinc",
    "Hann",
    "Hamming",
    "Welch",
    "Cosine",
    "Bohman",
    "Blackman",
    "BlackmanMinLobe",
    "FFmpegBicubic",
    "AdobeBicubic",
    "AdobeBicubicSharper",
    "AdobeBicubicSmoother",
)

# Subsets of planes, most used first: (label, source).
PLANES = (
    ("Y, U and V", "[0, 1, 2]"),
    ("Y", "[0]"),
    ("U and V", "[1, 2]"),
    ("U", "[1]"),
    ("V", "[2]"),
    ("Y and U", "[0, 1]"),
    ("Y and V", "[0, 2]"),
)

_BOOL = (("Default", DEFAULT), ("Yes", "True"), ("No", "False"))

# chaiNNer's video types plus the DVD, Blu-ray and index files VapourSynth opens.
VIDEO_FILE_TYPES = tuple(
    (
        ".mp4 .h264 .hevc .webm .avi .gif .mov .mkv .flv .m4v .avs .ts .m2ts .mts .vob"
        " .mpg .mpeg .m2v .wmv .ogv .y4m .ivf .dgi .d2v"
    ).split()
)

_PLACEHOLDERS = {
    "int": "Default",
    "float": "Default",
    "ranges": "e.g. 682-687, 713-717, 721",
    "fraction": "e.g. 24000/1001",
    "sequence": "e.g. 0.35, 0",
    "extra": "Python value, e.g. 1, 'text', [1, 2] or {'name': 1}",
}

# The last part of annotation names, by what they mean.
# vstools' VideoNodeIterable is a clip or (nested) iterables of clips.
_CLIP_LISTS = {"VideoNodeIterable"}
_CLIPS = {
    "VideoNode",
    "ConstantFormatVideoNode",
    "VariableFormatVideoNode",
    *_CLIP_LISTS,
}
_CALLABLES = ("VSFunction", "Callable", "GenericVSFunction")
_RANGES = {
    "FrameRange",
    "FrameRangeN",
    "FrameRanges",
    "FrameRangesN",
    "SoftRange",
    "SoftRangeN",
    "SoftRanges",
    "SoftRangesN",
}
_PLANES = {"Planes", "PlanesT"}
_PATHS = {"SPathLike", "SPath", "Path", "PathLike", "PurePath", "StrPath"}
_FORMATS = {"VideoFormatLike", "PresetVideoFormat", "VideoFormat"}
_COLOUR_NAMES = {
    "matrix": {"MatrixLike", "MatrixT", "Matrix", "MatrixCoefficients"},
    "transfer": {"TransferLike", "TransferT", "Transfer", "TransferCharacteristics"},
    "primaries": {"PrimariesLike", "PrimariesT", "Primaries", "ColorPrimaries"},
    "range": {"RangeLike", "ColorRangeLike", "ColorRangeT", "Range", "ColorRange"},
}
_KERNEL_ALIASES = {
    "KernelLike",
    "ScalerLike",
    "DescalerLike",
    "ResamplerLike",
    "ComplexKernelLike",
    "ComplexScalerLike",
    "ComplexDescalerLike",
    "ZimgComplexKernelLike",
    "CustomComplexKernelLike",
}
_SEQUENCES = {
    "Sequence",
    "MutableSequence",
    "Collection",
    "Iterable",
    "list",
    "List",
    "tuple",
    "Tuple",
}
_BUILTINS = {
    "int",
    "float",
    "bool",
    "str",
    "bytes",
    "dict",
    "Mapping",
    "Any",
    "SupportsString",
    *_SEQUENCES,
}

# The catalog's annotations of plugin signature types (scan.plugin_params): an array
# "T[]" is "T | Sequence[T]", an optional one adds " | None" with the default None.
_PLUGIN_TYPES = {
    "int": "int",
    "float": "float",
    "str | bytes": "data",
    "VideoNode": "vnode",
    "AudioNode": "anode",
    "VideoFrame": "vframe",
    "AudioFrame": "aframe",
    "Callable[..., Any]": "func",
}
# Plugin ints whose name says what they hold.
_PLUGIN_INTS = {
    "format": "format",
    "matrix": "matrix",
    "matrix_in": "matrix",
    "transfer": "transfer",
    "transfer_in": "transfer",
    "primaries": "primaries",
    "primaries_in": "primaries",
    "range": "range",
    "range_in": "range",
}
# Path parameters named for the video (or other source) they open; a plugin string
# with one of these names is a file to open too.
_VIDEO_NAMES = {
    "file",
    "files",
    "filename",
    "file_name",
    "filepath",
    "path",
    "source",
    "src",
    "src_file",
    "source_file",
    "source_path",
    "iso_path",
    "video",
    "video_path",
}
# Path parameters for a file the function writes, or a folder.
_OUTPUT_PATH = re.compile(r"^(out|dest|log)|folder|dir|cache|index", re.IGNORECASE)

_RANGE = re.compile(r"\s*(\d+)\s*(?:-\s*(\d+)\s*)?")
_WHOLE = re.compile(r"[+-]?\d+")
_ENUM_REPR = re.compile(r"<(?:class ')?([\w.]+)(?:'>|: (.*)>)")
_LITERAL_ERRORS = (ValueError, TypeError, SyntaxError, MemoryError, RecursionError)


@dataclass(frozen=True)
class Adapter:
    """How one parameter is shown as a chaiNNer input and written as an argument."""

    kind: str
    # The parameter's name, for error messages.
    name: str = ""
    # The function has a default, so the input may be left empty (or on "Default")
    # and the argument left out.
    optional: bool = False
    # The function's default as a value of the input (a number, a text, a choice's
    # source, DEFAULT), or None when it has none or none the input can show. A value
    # equal to it is left out of the call.
    default: object = None
    # Dropdown kinds: (label, source) per choice; the value of a choice is its source.
    choices: tuple[tuple[str, str], ...] = ()
    # sequence: written as a "list" or "tuple"; clip: "list" wraps the clip in a list.
    container: str = ""
    # clip: the parameter also takes a list of clips (Shift Clip Multi's output).
    clip_list: bool = False
    # sequence: one value is written as it is (the parameter also takes a single one).
    scalar: bool = False
    # sequence: whole numbers only; None is allowed as an element.
    integers: bool = False
    none_ok: bool = False
    # path: the file types the file dialog offers.
    file_types: tuple[str, ...] = ()
    # extra: the text is the **kwargs dict.
    var_keyword: bool = False
    # Import statements the written sources need.
    imports: frozenset[str] = frozenset()

    def make_input(
        self, label: str, param: Param, input_id: int, value: object = None
    ) -> BaseInput:
        """The input; `value` (from start_value) is the value it starts with instead of
        the function's default."""
        return self._widget(label, param, value).with_id(input_id)

    def start_value(self, value: object) -> object:
        """A value an overlay sets as the input's starting value (a TOML number, text
        or bool), as the input holds it: dropdowns hold their choice's source ("6",
        "vs.RGBS", "True"), number fields numbers, text fields text. Raises ValueError
        when the input cannot take it."""
        kind = self.kind
        if kind in ("clip", "optional_clip", "clip_or_filter", "path"):
            raise ValueError("a clip, filter or file input has no default")
        if self.choices:
            shown: object = str(value)
        elif kind == "bool":
            if not isinstance(value, bool):
                raise ValueError(f"{value!r} is not true or false")
            shown = value
        elif self._number_field():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{value!r} is not a number")
            shown = value
        elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
            shown = str(value).strip()
            if not shown:
                raise ValueError("an empty text is no default")
        else:
            raise ValueError(f"{value!r} is not a text or a number")
        self._source(shown)
        return shown

    def _number_field(self) -> bool:
        # A number without a default (None) is an optional text instead: chaiNNer's
        # number field cannot be empty.
        return self.kind in ("int", "float") and not (
            self.optional and self.default is None
        )

    def to_source(self, value: object) -> str | None:
        """The argument's Python source, or None to leave the argument out (the function
        then uses its default). Raises ValueError, in plain English, for bad input."""
        source = self._source(value)
        if source is None or source == self._source(self.default):
            return None
        return source

    def _widget(self, label: str, param: Param, value: object) -> BaseInput:
        kind = self.kind
        shown = self.default if value is None else value
        if kind in ("clip", "optional_clip"):
            clip = VideoClipInput(label)
            return clip.make_optional() if kind == "optional_clip" else clip
        if kind == "clip_or_filter":
            socket = VSFilterInput(label)
            return socket.make_optional() if self.optional else socket
        if kind == "path":
            file = FileInput(
                label,
                "video" if self.file_types == VIDEO_FILE_TYPES else "bin",
                list(self.file_types),
            )
            return file.make_optional() if self.optional else file
        if kind == "bool" and not self.choices:
            return BoolInput(label, default=bool(shown))
        if self.choices:
            return DropDownInput(
                input_type="string",
                label=label,
                options=[{"option": o, "value": s} for o, s in self.choices],
                default_value=shown if isinstance(shown, str) else None,
                groups=[DropDownGroup(g, s) for g, s in _FORMAT_GROUPS]
                if kind == "format"
                else None,
            )
        if self._number_field():
            return NumberInput(
                label,
                precision=0 if kind == "int" else "unlimited",
                step=1 if kind == "int" else 0.1,
                default=shown if isinstance(shown, (int, float)) else 0,
                min=None,
                max=None,
            )
        placeholder = _PLACEHOLDERS.get(kind)
        if kind == "extra":
            if self.var_keyword:
                placeholder = "Python dict, e.g. {'name': 1}"
            elif param.annotation.strip():
                label = f"{label} ({param.annotation.strip()})"
        text = TextInput(
            label,
            placeholder=placeholder,
            default=shown if isinstance(shown, str) else None,
        )
        return text.make_optional() if self.optional else text

    def _source(self, value: object) -> str | None:
        kind = self.kind
        if value is None or value == "":
            return None
        if kind in ("clip", "optional_clip", "clip_or_filter"):
            if isinstance(value, VideoClip):
                if value.many and self.clip_list:
                    return value.var
                if self.container == "list":
                    return f"[{value.var}]"
                return one_clip(value, self.name)
            if isinstance(value, VSFilter) and kind == "clip_or_filter":
                return value.expr
            raise ValueError(f"{self.name}: this input takes a clip.")
        if kind == "bool" and not isinstance(value, str):
            return "True" if value else "False"
        if self.choices:
            if value == DEFAULT:
                return None
            if not any(value == source for _, source in self.choices):
                raise ValueError(f"{self.name}: {value!r} is not one of the choices.")
            return str(value)
        if kind == "path" and isinstance(value, (str, os.PathLike)):
            return repr(os.fspath(value))
        if kind == "text":
            return repr(str(value))
        if kind in ("int", "float") and isinstance(value, (int, float)):
            return _number(value, kind == "int", self.name)
        if not isinstance(value, str):
            raise ValueError(f"{self.name}: {value!r} is not a value this input takes.")
        text = value.strip()
        if not text:
            return None
        if kind in ("int", "float"):
            return _number(text, kind == "int", self.name)
        if kind == "ranges":
            return _ranges(text, self.name)
        if kind == "fraction":
            return _fraction(text, self.name)
        if kind == "sequence":
            return self._sequence(text)
        return self._extra(text)

    def _sequence(self, text: str) -> str:
        items: list[str] = []
        for item in text.split(","):
            token = item.strip()
            if token == "None" and self.none_ok:
                items.append("None")
            elif not token:
                raise ValueError(
                    f"{self.name}: an empty value between commas is not a number."
                )
            else:
                # A whole number stays one ("0.35, 0" -> [0.35, 0]).
                whole = self.integers or _WHOLE.fullmatch(token) is not None
                items.append(_number(token, whole, self.name))
        if len(items) == 1 and self.scalar:
            return items[0]
        if self.container == "tuple":
            return f"({items[0]},)" if len(items) == 1 else f"({', '.join(items)})"
        return f"[{', '.join(items)}]"

    def _extra(self, text: str) -> str:
        try:
            value = ast.literal_eval(text)
        except _LITERAL_ERRORS:
            raise ValueError(
                f"{self.name}: {text!r} is not a plain Python value (a number, 'text' in"
                " quotes, True, False, None, or a list, tuple or dict of those);"
                " expressions and function calls are not allowed."
            ) from None
        if self.var_keyword and not (
            isinstance(value, dict)
            and all(isinstance(k, str) and k.isidentifier() for k in value)
        ):
            raise ValueError(
                f"{self.name}: extra arguments are a dict of parameter names to values,"
                " like {'name': 1}."
            )
        # Python's own writing of the value: a bare tuple ("1, 2") gets its
        # parentheses, and a comment or a line break (also a lone \r), which would
        # break the one-line statement the text goes into, goes.
        return ast.unparse(ast.parse(text, mode="eval"))


def adapter_for(param: Param, enums: Enums, plugin: bool = False) -> Adapter:
    """The adapter of a catalog parameter; `plugin` for a plugin function's. `enums` is
    the catalog's enums map: key -> {"import", "members": [{"name", "expr"}]}."""
    annotation = param.annotation.strip()
    if param.kind == "var_keyword":
        return Adapter("extra", name=param.name, optional=True, var_keyword=True)
    signature = _plugin_signature(param) if plugin else None
    if signature is not None:
        return _plugin_adapter(param, *signature)
    if not annotation:
        return _unannotated(param, enums)
    try:
        node = ast.parse(annotation, mode="eval").body
    except SyntaxError:
        return _extra(param)
    members: list[ast.expr] = []
    _flatten(node, members)
    members = [m for m in members if not _is_none(m)]
    return _python_adapter(param, members, enums)


def _python_adapter(param: Param, members: list[ast.expr], enums: Enums) -> Adapter:
    """The adapter of a vs-jetpack (or user) parameter: the first rule that matches a
    member of its union annotation wins."""
    name = param.name
    optional = param.default is not None
    lasts = {_last(m) for m in members}
    callable_ = any(last.startswith(_CALLABLES) for last in lasts)
    sequences = [(m, _elements(m)) for m in members if _last(m) in _SEQUENCES]

    if lasts & _CLIPS or any(e and e <= _CLIPS for _, e in sequences):
        if callable_:
            return Adapter("clip_or_filter", name=name, optional=optional)
        return Adapter(
            "optional_clip" if optional else "clip",
            name=name,
            optional=optional,
            container="" if lasts & _CLIPS else "list",
            clip_list=bool(lasts & _CLIP_LISTS)
            or any(e and e <= _CLIPS for _, e in sequences),
        )
    if lasts & _RANGES:
        return Adapter("ranges", name=name, optional=optional)
    if lasts & _PLANES:
        return _planes(param, optional)
    if "Fraction" in lasts:
        return Adapter(
            "fraction",
            name=name,
            optional=optional,
            default=_fraction_text(param.default),
            imports=frozenset(["import fractions"]),
        )
    if lasts & _PATHS:
        return _path(param, optional)
    for kind, names in _COLOUR_NAMES.items():
        if lasts & names:
            return _choice(param, kind, _colour_choices(kind), optional)
    # depth(clip, 16): a format-or-bits parameter is a number of bits.
    if lasts & _FORMATS and "int" not in lasts:
        return _choice(param, "format", _FORMAT_CHOICES, optional)

    choices, literals, imports = _enum_choices(members, enums)
    numbers = lasts & {"int", "float"}
    number_sequences = [
        (m, e) for m, e in sequences if e and e <= {"int", "float", "None"}
    ]
    # Literal[False] beside a number or a sequence is only its "off" value.
    if choices or (literals and not numbers and not number_sequences):
        enum = _choice(param, "enum", _unique(choices + literals), optional)
        return replace(enum, imports=imports)
    if callable_:
        return Adapter("clip_or_filter", name=name, optional=optional)
    if "bool" in lasts:
        default = _literal(param.default)
        if isinstance(default, bool) or not optional:
            return Adapter("bool", name=name, optional=optional, default=default)
        # None: the function decides, a third choice.
        return Adapter("bool", name=name, optional=True, default=DEFAULT, choices=_BOOL)
    if number_sequences:
        elements = numbers.union(*(e for _, e in number_sequences))
        tuples = all(_last(m) in ("tuple", "Tuple") for m, _ in number_sequences)
        return _sequence(
            param,
            optional,
            container="tuple" if tuples else "list",
            scalar=bool(numbers),
            integers="float" not in elements,
            none_ok="None" in elements,
        )
    if numbers:
        kind = "float" if "float" in numbers else "int"
        return _number_adapter(param, kind, optional)
    if lasts & {"str", "SupportsString"}:
        return _text(param, optional)
    return _extra(param)


def _plugin_signature(param: Param) -> tuple[str, bool] | None:
    """A plugin parameter's signature type and whether it is an array, from the
    catalog's annotation ("float | Sequence[float] | None" -> ("float", True))."""
    text = param.annotation.strip().removesuffix(" | None")
    for annotation, base in _PLUGIN_TYPES.items():
        if text == annotation:
            return base, False
        if text == f"{annotation} | Sequence[{annotation}]":
            return base, True
    return None


def _plugin_adapter(param: Param, base: str, array: bool) -> Adapter:
    """A parameter of a plugin function, from its signature type."""
    name = param.name
    optional = param.default is not None
    if base == "vnode":
        return Adapter(
            "optional_clip" if optional else "clip",
            name=name,
            optional=optional,
            clip_list=array,
        )
    if base == "func":
        return Adapter("clip_or_filter", name=name, optional=optional)
    if base == "int" and array:
        if name == "planes":
            return _planes(param, optional)
        return _sequence(param, optional, "list", scalar=True, integers=True)
    if base == "int" and name in _PLUGIN_INTS:
        kind = _PLUGIN_INTS[name]
        choices = _FORMAT_CHOICES if kind == "format" else _colour_choices(kind)
        return _choice(param, kind, choices, optional)
    if base in ("int", "float"):
        if array:
            return _sequence(param, optional, "list", scalar=True, integers=False)
        return _number_adapter(param, base, optional)
    if base == "data":
        if name in _VIDEO_NAMES and not array:
            return _path(param, optional)
        return _text(param, optional)
    return _extra(param)


def _unannotated(param: Param, enums: Enums) -> Adapter:
    """A parameter without an annotation (user functions) takes its default's type."""
    default = _literal(param.default)
    if isinstance(default, (bool, int, float, str)):
        annotation = type(default).__name__
    elif (
        isinstance(default, (list, tuple))
        and default
        and all(_is_number(x) for x in default)
    ):
        number = "float" if any(isinstance(x, float) for x in default) else "int"
        annotation = f"{type(default).__name__}[{number}]"
    else:
        return _extra(param)
    return adapter_for(replace(param, annotation=annotation), enums)


def _number_adapter(param: Param, kind: str, optional: bool) -> Adapter:
    default = _literal(param.default)
    if isinstance(default, bool) or not isinstance(default, (int, float)):
        default = None
    elif kind == "float":
        default = float(default)
    elif not isinstance(default, int):
        default = None
    return Adapter(kind, name=param.name, optional=optional, default=default)


def _sequence(
    param: Param,
    optional: bool,
    container: str,
    scalar: bool,
    integers: bool,
    none_ok: bool = False,
) -> Adapter:
    adapter = Adapter(
        "sequence",
        name=param.name,
        optional=optional,
        container=container,
        scalar=scalar,
        integers=integers,
        none_ok=none_ok,
    )
    default = _literal(param.default)
    if _is_number(default):
        text = repr(default)
    elif isinstance(default, (list, tuple)) and default:
        text = ", ".join(repr(x) for x in default)
    else:
        return adapter
    try:
        adapter.to_source(text)
    except ValueError:
        # Not a default this input can show (a nested tuple, say).
        return adapter
    return replace(adapter, default=text)


def _text(param: Param, optional: bool) -> Adapter:
    default = _literal(param.default)
    return Adapter(
        "text",
        name=param.name,
        optional=optional,
        default=default if isinstance(default, str) and default else None,
    )


def _extra(param: Param) -> Adapter:
    """Python literals as text; a literal default other than None is shown."""
    shown = param.default if _literal(param.default) is not None else None
    return Adapter(
        "extra", name=param.name, optional=param.default is not None, default=shown
    )


def _path(param: Param, optional: bool) -> Adapter:
    if _OUTPUT_PATH.search(param.name):
        # A file the function writes, or a folder: the file input only takes
        # existing files, so a text.
        return _text(param, optional)
    return Adapter(
        "path",
        name=param.name,
        optional=optional,
        file_types=VIDEO_FILE_TYPES if param.name in _VIDEO_NAMES else ("*",),
    )


def _planes(param: Param, optional: bool) -> Adapter:
    default = _literal(param.default)
    if isinstance(default, int) and not isinstance(default, bool):
        default = [default]
    source = None
    if isinstance(default, (list, tuple)) and all(
        isinstance(x, int) and not isinstance(x, bool) for x in default
    ):
        source = f"[{', '.join(str(p) for p in sorted(set(default)))}]"
    if not any(source == s for _, s in PLANES):
        source = None
    return _choice(param, "planes", PLANES, optional, source)


def _choice(
    param: Param,
    kind: str,
    choices: tuple[tuple[str, str], ...],
    optional: bool,
    default: str | None = None,
) -> Adapter:
    """A dropdown. A default no choice names gets a "Default" choice that keeps it."""
    if default is None:
        default = _match_default(param.default, choices)
    if default is None and optional:
        choices = (("Default", DEFAULT), *choices)
        default = DEFAULT
    return Adapter(
        kind, name=param.name, optional=optional, default=default, choices=choices
    )


def _colour_choices(kind: str) -> tuple[tuple[str, str], ...]:
    return tuple((f"{label} ({n})", str(n)) for n, label in _COLOUR[kind])


def _enum_choices(
    members: list[ast.expr], enums: Enums
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], frozenset[str]]:
    """The choices of the members that are enums (from the catalog, or kernel names),
    those of Literal[...] members, and the imports their sources need."""
    choices: list[tuple[str, str]] = []
    literals: list[tuple[str, str]] = []
    imports: set[str] = set()
    for member in members:
        if isinstance(member, ast.Subscript) and _last(member.value) == "Literal":
            items = member.slice
            for item in items.elts if isinstance(items, ast.Tuple) else [items]:
                if isinstance(item, ast.Constant):
                    literals.append((str(item.value), repr(item.value)))
                    continue
                # An enum member (Literal[MVDirection.FORWARD]): qualified as its
                # enum's members are.
                expr = ast.unparse(item)
                entry = next(
                    (
                        e
                        for e in enums.values()
                        if any(m["expr"] == expr for m in e["members"])
                    ),
                    None,
                )
                if entry is not None:
                    import_line, expr = qualified(entry["import"], expr)
                    if import_line:
                        imports.add(import_line)
                literals.append((_plain(expr.rpartition(".")[2]), expr))
            continue
        # Built-in names are never enums; skipping them skips the catalog scan.
        if _last(member) in _BUILTINS:
            continue
        entry = _enum_entry(_dotted(member), enums)
        if entry is not None:
            # Module-qualified (vsdenoise.bm3d.Profile.HIGH; VapourSynth's own
            # through `vs`): vstools has enums of VapourSynth's names (FieldBased).
            for m in entry["members"]:
                import_line, expr = qualified(entry["import"], m["expr"])
                choices.append((_plain(m["name"]), expr))
                if import_line:
                    imports.add(import_line)
        elif _last(member) in _KERNEL_ALIASES:
            choices.extend((k, repr(k)) for k in KERNELS)
    return choices, literals, frozenset(imports)


def _enum_entry(name: str, enums: Enums) -> Mapping[str, Any] | None:
    """The catalog enum an annotation names: its key itself, an "XLike" alias of X, or
    the one key that ends with the name (or that the name ends with)."""
    if not name:
        return None
    candidates = [name, name[: -len("Like")]] if name.endswith("Like") else [name]
    for candidate in candidates:
        if candidate in enums:
            return enums[candidate]
    for candidate in candidates:
        hits = [
            key
            for key in enums
            if key.endswith(f".{candidate}") or candidate.endswith(f".{key}")
        ]
        if len(hits) == 1:
            return enums[hits[0]]
    return None


def _match_default(
    default: str | None, choices: tuple[tuple[str, str], ...]
) -> str | None:
    """The choice the function's default (a repr) names, if any: its source (or its
    source module-qualified: the catalog writes "bm3d.Profile.FAST"), an enum member
    ("<Profile.FAST: 'fast'>"), a class ("<class '...Catrom'>"), a name or a value,
    case-insensitively."""
    # None is never a member (an enum may have one named NONE).
    if default is None or default == "None":
        return None
    for _, source in choices:
        if source == default or source.endswith(f".{default}"):
            return source
    match = _ENUM_REPR.fullmatch(default)
    keys = {_key(match[1]), _key(match[2] or match[1])} if match else {_key(default)}
    for _, source in choices:
        if _key(source) in keys:
            return source
    return None


def _key(text: str) -> str:
    """A comparable form of a source, name or value: a literal's value, else the last
    part of a dotted name; case-insensitive."""
    try:
        value = ast.literal_eval(text)
    except _LITERAL_ERRORS:
        return text.rpartition(".")[2].casefold()
    return str(value).casefold()


def _number(value: str | float, integer: bool, name: str) -> str:
    try:
        number = int(value) if integer else float(value)
    except ValueError:
        what = "a whole number" if integer else "a number"
        raise ValueError(f"{name}: {value!r} is not {what}.") from None
    if not math.isfinite(number):
        raise ValueError(f"{name}: {value!r} is not a finite number.")
    if integer and number != value and not isinstance(value, str):
        raise ValueError(f"{name}: {value!r} is not a whole number.")
    return repr(number)


def _ranges(text: str, name: str) -> str:
    """ "682-687, 713-717, 721" -> "[(682, 687), (713, 717), 721]" (inclusive)."""
    items: list[int | tuple[int, int]] = []
    for token in text.rstrip(",").split(","):
        match = _RANGE.fullmatch(token)
        if not match:
            raise ValueError(
                f"{name}: {token.strip()!r} is not a frame number or a range like 10-20."
            )
        start = int(match[1])
        if match[2] is None:
            items.append(start)
            continue
        end = int(match[2])
        if end < start:
            raise ValueError(
                f"{name}: the range {token.strip()} ends before it starts."
            )
        items.append((start, end))
    return repr(items)


def _fraction(text: str, name: str) -> str:
    try:
        value = Fraction("".join(text.split()))
    except (ValueError, ZeroDivisionError):
        raise ValueError(
            f"{name}: {text!r} is not a fraction like 24000/1001 or a number like 25."
        ) from None
    return f"fractions.Fraction({value.numerator}, {value.denominator})"


def _fraction_text(default: str | None) -> str | None:
    """A Fraction default as the input's text: "Fraction(24000, 1001)" -> "24000/1001"."""
    match = re.fullmatch(r"Fraction\((\d+), (\d+)\)", default or "")
    if match:
        return f"{match[1]}/{match[2]}"
    value = _literal(default)
    return (
        str(value) if isinstance(value, int) and not isinstance(value, bool) else None
    )


def _literal(text: str | None) -> object:
    """The value of a default's repr, or None when it is not a Python literal."""
    if text is None:
        return None
    try:
        return ast.literal_eval(text)
    except _LITERAL_ERRORS:
        return None


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _flatten(node: ast.expr, members: list[ast.expr]) -> None:
    """The members of a union: `A | B`, Union[A, B], Optional[A], "quoted" names."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        _flatten(node.left, members)
        _flatten(node.right, members)
    elif isinstance(node, ast.Subscript) and _last(node.value) in ("Union", "Optional"):
        items = node.slice
        for item in items.elts if isinstance(items, ast.Tuple) else [items]:
            _flatten(item, members)
        if _last(node.value) == "Optional":
            members.append(ast.Constant(None))
    elif isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            _flatten(ast.parse(node.value, mode="eval").body, members)
        except SyntaxError:
            members.append(node)
    else:
        members.append(node)


def _elements(node: ast.expr) -> set[str]:
    """The last name parts of a sequence's element types ("None" for None)."""
    if not isinstance(node, ast.Subscript):
        return set()
    items = node.slice
    found: list[ast.expr] = []
    for item in items.elts if isinstance(items, ast.Tuple) else [items]:
        if not (isinstance(item, ast.Constant) and item.value is Ellipsis):
            _flatten(item, found)
    return {"None" if _is_none(m) else _last(m) for m in found}


def _dotted(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else ""
    if isinstance(node, ast.Subscript):
        return _dotted(node.value)
    return ""


def _last(node: ast.expr) -> str:
    return _dotted(node).rpartition(".")[2]


def _is_none(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _plain(name: str) -> str:
    return name.replace("_", " ")


def _unique(choices: list[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    """The choices without repeated sources (the first one stays)."""
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for label, source in choices:
        if source not in seen:
            seen.add(source)
            unique.append((label, source))
    return tuple(unique)
