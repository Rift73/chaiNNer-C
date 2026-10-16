"""Lists the VapourSynth functions chaiNNer can call, as the catalog JSON its VapourSynth
nodes are generated from: vs-jetpack's public functions and class-based operations
(`Mitchell().scale`, `NNEDI3(...).deinterlace`), every loaded plugin's functions, and
the user's functions (the *.py files of --user-dir and the pip packages listed in
--modules-file).

A child process of chaiNNer's worker, which never imports VapourSynth:

    python -I -B scan.py --out <json> --user-dir <dir> --modules-file <toml>

Standalone: the standard library and VapourSynth only (the parsing helpers import
neither, so tests use them in-process). A module, function or plugin that fails is
recorded under "errors" and skipped; the scan goes on.

Each function has the import line of its call's first name ("from vsaa import
NNEDI3"; "" for plugins, which the script's `core` reaches). Function kinds:
"function", "bound" (`Cls(<init>).method`), "plugin", and "members" (an enum whose
members are callable on a clip, `Prefilter.DFTTEST(clip)`; "members" lists their
names). Parameter kinds:
"positional" must be passed by position (positional-only), "keyword" may be passed
by name, "var_keyword" is **kwargs; *args is left out (no value a node could pass).
Annotations are the resolved types as text: classes by their qualified names
("VideoNode | None", "Planes"), enums and type aliases with choices by their "enums"
keys, their dotted paths ("vsdenoise.blockmatch.BM3D.Profile"); the source text when
they do not resolve. "enums" maps each such key to {"import", "members": [{"name",
"expr"}]}: every member's expression starts with the name the import line imports
(`from vsdenoise import bm3d`, "bm3d.Profile.HIGH"); string and literal choices need
none. Defaults are source text: an enum member or preset as its expression, a class,
function or exported object as its dotted path (vskernels.Catrom), else the repr."""

from __future__ import annotations

import argparse
import enum
import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
import inspect
import json
import re
import sys
import types
import typing
import warnings
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import annotationlib
import tomllib

SCHEMA = 1
JETPACK = "vsjetpack"
# Longest error text kept per failure.
ERROR_CHARS = 1000

_KINDS = {
    inspect.Parameter.POSITIONAL_ONLY: "positional",
    inspect.Parameter.POSITIONAL_OR_KEYWORD: "keyword",
    inspect.Parameter.KEYWORD_ONLY: "keyword",
    inspect.Parameter.VAR_KEYWORD: "var_keyword",
}

# Plugin signature types (VapourSynth API 4) as annotations.
_PLUGIN_TYPES = {
    "int": "int",
    "float": "float",
    "data": "str | bytes",
    "vnode": "VideoNode",
    "anode": "AudioNode",
    "vframe": "VideoFrame",
    "aframe": "AudioFrame",
    "func": "Callable[..., Any]",
}

# Google-style docstring sections; the description ends at the first one.
_SECTION = re.compile(
    r"^(Args|Arguments|Parameters|Params|Keyword Args|Keyword Arguments"
    r"|Other Parameters|Raises|Returns?|Yields?|Examples?)\s*:\s*$"
)
_ARG_SECTIONS = {
    "Args",
    "Arguments",
    "Parameters",
    "Params",
    "Keyword Args",
    "Keyword Arguments",
    "Other Parameters",
}
_ARG_ENTRY = re.compile(r"^\*{0,2}(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$")
_BULLET = ("-", "*", "•")
# Names chaiNNer's scripts bind besides module names (script.py's prologue and outputs,
# and the clips n_<node>_<output>): a user file must not take them.
SCRIPT_NAMES = frozenset({"sys", "vs", "core", "set_output"})
_NODE_VAR = re.compile(r"n_\w+_\d+")
# chaiNNer's backend/src (script.py's BACKEND_SRC), which the scripts put on sys.path
# before the user folder: a user file must not be named like one of its modules.
_BACKEND_SRC = Path(__file__).resolve().parents[3]
# Most wrappers unwrapped to reach a class's method (inject_self(inject_kwargs_params)).
_WRAPPER_DEPTH = 4
# annotationlib's names for the resolved parts of a partly resolved annotation.
_PLACEHOLDER = "__annotationlib_name_"
_ADDRESS = re.compile(r" at 0x[0-9A-Fa-f]+")

# Enums every catalog lists, whether or not an annotation names them: the format,
# matrix, transfer, primaries, range, field order and chroma location choices of the
# plugins' integer arguments.
_SEEDED_ENUMS = (
    ("vapoursynth", "PresetVideoFormat"),
    ("vstools", "Matrix"),
    ("vstools", "Transfer"),
    ("vstools", "Primaries"),
    ("vstools", "Range"),
    ("vstools", "FieldBased"),
    ("vstools", "ChromaLocation"),
)


def plugin_params(signature: str) -> list[dict[str, Any]]:
    """The parameters of a plugin function from its signature string, e.g.
    "clipa:vnode;clipb:vnode;weight:float[]:opt;". Plugin arguments go by name;
    arrays also take one value; "any" takes further named arguments."""
    params: list[dict[str, Any]] = []
    for entry in filter(None, signature.split(";")):
        if entry == "any":
            params.append(_param("kwargs", "Any", None, "var_keyword"))
            continue
        name, type_name, *flags = entry.split(":")
        annotation = _plugin_type(type_name)
        if type_name.endswith("[]"):
            annotation = f"{annotation} | Sequence[{annotation}]"
        if "opt" in flags:
            params.append(_param(name, f"{annotation} | None", "None", "keyword"))
        else:
            params.append(_param(name, annotation, None, "keyword"))
    return params


def plugin_returns(signature: str) -> str:
    """The return annotation of a plugin function from its return signature."""
    entries = [entry for entry in signature.split(";") if entry]
    if not entries:
        return "None"
    if len(entries) > 1 or entries[0] == "any":
        return "dict[str, Any]"
    type_name = entries[0].split(":")[1]
    annotation = _plugin_type(type_name)
    return f"list[{annotation}]" if type_name.endswith("[]") else annotation


def _plugin_type(type_name: str) -> str:
    return _PLUGIN_TYPES.get(type_name.removesuffix("[]"), "Any")


def _param(
    name: str, annotation: str, default: str | None, kind: str
) -> dict[str, Any]:
    return {"name": name, "annotation": annotation, "default": default, "kind": kind}


def split_doc(doc: str) -> tuple[str, dict[str, str]]:
    """A cleaned docstring's description (the text before its first section) and the
    help of each argument of its Args sections."""
    lines = doc.splitlines()
    description_end = len(lines)
    arg_docs: dict[str, str] = {}
    index = 0
    while index < len(lines):
        section = _SECTION.match(lines[index])
        index += 1
        if section is None:
            continue
        description_end = min(description_end, index - 1)
        if section[1] not in _ARG_SECTIONS:
            continue
        block: list[str] = []
        while index < len(lines) and (
            not lines[index].strip() or lines[index][0] == " "
        ):
            block.append(lines[index])
            index += 1
        arg_docs.update(_arg_entries(block))
    return "\n".join(lines[:description_end]).strip(), arg_docs


def _arg_entries(block: list[str]) -> dict[str, str]:
    indents = [len(line) - len(line.lstrip()) for line in block if line.strip()]
    if not indents:
        return {}
    entry_indent = indents[0]
    entries: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in block:
        text = line.strip()
        indent = len(line) - len(line.lstrip())
        match = _ARG_ENTRY.match(text) if indent == entry_indent and text else None
        if match:
            current = entries.setdefault(match[1], [])
            current.append(match[2])
        elif current is not None:
            current.append(text)
    return {name: _join_doc_lines(parts) for name, parts in entries.items()}


def _join_doc_lines(parts: list[str]) -> str:
    """Continuation lines joined with spaces; bullets and paragraphs keep their
    line breaks."""
    text = ""
    blank = False
    for part in parts:
        if not part:
            blank = True
            continue
        if not text:
            text = part
        elif blank or part.startswith(_BULLET):
            text += "\n" + part
        else:
            text += " " + part
        blank = False
    return text


def user_module_names(modules_file: Path) -> list[str]:
    """The pip names listed in user_modules.toml: `modules = ["vs_colorfix", ...]`.
    No file means none. Raises ValueError for a file that is not such a list."""
    if not modules_file.is_file():
        return []
    try:
        # utf-8-sig: Notepad and Windows PowerShell may start it with a BOM.
        data = tomllib.loads(modules_file.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"{modules_file.name} cannot be read: {error}") from error
    names = data.get("modules", [])
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise ValueError(
            f'{modules_file.name} must hold a list of pip names: modules = ["name"]'
        )
    return list(dict.fromkeys(name.strip() for name in names if name.strip()))


def _normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def import_names(pip_name: str, distributions: Mapping[str, list[str]]) -> list[str]:
    """The top-level modules a pip package installs (its name with "_" for "-" when
    the metadata does not tell)."""
    wanted = _normalized(pip_name)
    tops = sorted(
        top
        for top, names in distributions.items()
        if top.isidentifier()
        and not top.startswith("_")
        and any(_normalized(name) == wanted for name in names)
    )
    return tops or [pip_name.replace("-", "_").replace(".", "_")]


def _unwrapped(value: object) -> object:
    """The function behind a wrapper: vs-jetpack's wrapper classes keep it on `_func`
    (BM3D also sets __wrapped__), functools.wraps on __wrapped__."""
    target = getattr(value, "_func", None)
    if not callable(target):
        target = value
    return inspect.unwrap(target) if inspect.isfunction(target) else target


def _owned(obj: object, top: str) -> bool:
    module = getattr(obj, "__module__", None) or ""
    return module == top or module.startswith(top + ".")


def _public_members(module: types.ModuleType) -> Iterable[tuple[str, object]]:
    names = getattr(module, "__all__", None)
    if not isinstance(names, (list, tuple)):
        names = [name for name in dir(module) if not name.startswith("_")]
    for name in sorted({name for name in names if isinstance(name, str)}):
        if name.startswith("_"):
            continue
        try:
            yield name, getattr(module, name)
        except Exception:  # a lazy or deprecated name that fails on access
            continue


def _walk(value: object, attributes: Iterable[str]) -> object:
    for attribute in attributes:
        value = getattr(value, attribute)
    return value


# Where a script gets an object: the module to import from and the attribute chain
# whose first name it imports, ("vsdenoise", ("bm3d", "Profile")) for
# `from vsdenoise import bm3d` then `bm3d.Profile`.
Binding = typing.Tuple[str, typing.Tuple[str, ...]]


def import_line(binding: Binding) -> str:
    return f"from {binding[0]} import {binding[1][0]}"


def _key(obj: Any) -> str:
    """The dotted path of an enum, alias or preset class: its "enums" key, and its
    name in annotations."""
    name = getattr(obj, "__qualname__", None) or obj.__name__
    return f"{obj.__module__}.{name}"


class Paths:
    """How a script imports classes, functions and wrapper objects: by their public
    name in the package that exports them (`from vskernels import Mitchell`; a nested
    class through its exported owner, `from vsdenoise import bm3d` for
    bm3d.Profile), else from the defining module. Each binding is checked by walking
    it."""

    def __init__(self, modules: Mapping[str, types.ModuleType]) -> None:
        # By id: the modules keep these objects alive.
        self._direct: dict[int, list[Binding]] = {}
        # Classes reached through an exported instance (BM3D through bm3d).
        self._instances: dict[int, list[Binding]] = {}
        # By id, with the object kept alive so that its id is never reused.
        self._cache: dict[int, tuple[object, Binding | None]] = {}
        for top in sorted(modules):
            for name, value in _public_members(modules[top]):
                binding = (top, (name,))
                self._direct.setdefault(id(value), []).append(binding)
                if not isinstance(value, (type, types.ModuleType, types.FunctionType)):
                    self._instances.setdefault(id(type(value)), []).append(binding)

    def binding(self, obj: object) -> Binding | None:
        key = id(obj)
        if key not in self._cache:
            self._cache[key] = (obj, self._find(obj))
        return self._cache[key][1]

    def dotted(self, obj: object) -> str | None:
        """The object as `module.name`, valid after `import module`."""
        binding = self.binding(obj)
        return None if binding is None else ".".join((binding[0], *binding[1]))

    def _find(self, obj: object) -> Binding | None:
        # Its own package first, then the shortest module, then its own name (vstools
        # exports Range as ColorRange too).
        top = (getattr(obj, "__module__", None) or "").split(".")[0]
        own_name = getattr(obj, "__name__", None)
        candidates = sorted(
            self._direct.get(id(obj), []),
            key=lambda binding: (
                binding[0] != top,
                len(binding[0]),
                binding[1][0] != own_name,
                binding,
            ),
        )
        for candidate in candidates:
            if self._reaches(candidate, obj):
                return candidate
        qualname = getattr(obj, "__qualname__", None)
        module = getattr(obj, "__module__", None)
        if not isinstance(qualname, str) or not isinstance(module, str):
            return None
        if "<" in qualname:
            return None
        *outer_names, inner = qualname.split(".")
        if outer_names:
            try:
                outer = _walk(sys.modules[module], outer_names)
            except (KeyError, AttributeError):
                outer = None
            if outer is not None:
                for outer_module, chain in [
                    *self._direct.get(id(outer), []),
                    *self._instances.get(id(outer), []),
                ]:
                    candidate = (outer_module, (*chain, inner))
                    if self._reaches(candidate, obj):
                        return candidate
        candidate = (module, tuple(qualname.split(".")))
        return candidate if self._reaches(candidate, obj) else None

    @staticmethod
    def _reaches(binding: Binding, obj: object) -> bool:
        try:
            return _walk(importlib.import_module(binding[0]), binding[1]) is obj
        except Exception:
            return False


class Scanner:
    """Builds the catalog's functions and enums from imported modules and plugins."""

    def __init__(self, paths: Paths, errors: list[str]) -> None:
        self.paths = paths
        self.functions: list[dict[str, Any]] = []
        # Each enum, alias or preset class's entry by its key; None when it has no
        # choices or no importable name.
        self.enums: dict[str, dict[str, Any] | None] = {}
        self.errors = errors
        self._ids: set[str] = set()
        # By id, with the object kept alive so that its id is never reused.
        self._docs: dict[int, tuple[object, tuple[str, dict[str, str]]]] = {}

    def error(self, where: str, error: BaseException) -> None:
        self.errors.append(failure(where, error))

    # Enums and annotations.

    def seed(self, modules: Mapping[str, types.ModuleType]) -> None:
        for top, name in _SEEDED_ENUMS:
            module = modules.get(top)
            value = getattr(module, name, None) if module is not None else None
            if isinstance(value, type) and issubclass(value, enum.Enum):
                self._enum(value)

    def _enum(self, cls: type[enum.Enum]) -> dict[str, Any] | None:
        """The "enums" entry of an enum class: the import of its name and its members
        as expressions on that name (bm3d.Profile.HIGH)."""
        key = _key(cls)
        if key not in self.enums:
            binding = self.paths.binding(cls)
            if binding is None:
                self.errors.append(f"enum {key}: no importable name")
                self.enums[key] = None
            else:
                base = ".".join(binding[1])
                self.enums[key] = {
                    "import": import_line(binding),
                    "members": [
                        {"name": member.name, "expr": f"{base}.{member.name}"}
                        for member in cls
                    ],
                }
        return self.enums[key]

    def _member(self, member: enum.Enum) -> str:
        """An enum member as its "enums" expression, else its repr."""
        binding = self.paths.binding(type(member))
        if binding is None:
            return repr(member)
        self._enum(type(member))
        return f"{'.'.join(binding[1])}.{member.name}"

    def _alias(self, alias: typing.TypeAliasType) -> dict[str, Any] | None:
        """The "enums" entry of a type alias with choices: the members of its enum
        (vs-jetpack's when it unites VapourSynth's and vs-jetpack's, MatrixLike ->
        Matrix), the class names of a `str | type[X]` alias (KernelLike -> "Mitchell",
        which needs no import) and Literal values."""
        key = _key(alias)
        if key in self.enums:
            return self.enums[key]
        self.enums[key] = None  # also ends a recursive alias
        try:
            value = alias.__value__
        except Exception:  # a name only defined for type checkers
            return None
        parts = _union_parts(value)
        choices: list[tuple[bool, dict[str, Any]]] = []
        literals: list[dict[str, str]] = []
        for part in parts:
            origin = typing.get_origin(part)
            entry = None
            if isinstance(part, typing.TypeAliasType):
                entry = self._alias(part)
            elif isinstance(part, type) and issubclass(part, enum.Enum):
                entry = self._enum(part)
            elif origin is type and str in parts:
                literals += self._class_names(typing.get_args(part)[0])
            elif origin is typing.Literal:
                literals += [
                    {"name": str(arg), "expr": repr(arg)}
                    for arg in typing.get_args(part)
                    if not isinstance(arg, enum.Enum)
                ]
            if entry is not None:
                choices.append((part.__module__ == "vapoursynth", entry))
        chosen = min(choices, key=lambda choice: choice[0])[1] if choices else None
        members = [*(chosen["members"] if chosen else []), *literals]
        unique = list({member["expr"]: member for member in members}.values())
        if unique:
            self.enums[key] = {
                "import": chosen["import"] if chosen else "",
                "members": unique,
            }
        return self.enums[key]

    def _class_names(self, base: object) -> list[dict[str, str]]:
        """The concrete public subclasses of a class as string choices."""
        if not isinstance(base, type) or base.__module__ == "builtins":
            return []
        found: dict[int, type] = {}
        stack = [base]
        while stack:
            cls = stack.pop()
            if id(cls) not in found:
                found[id(cls)] = cls
                stack.extend(cls.__subclasses__())
        names = {
            cls.__name__
            for cls in found.values()
            if not cls.__name__.startswith("_") and not inspect.isabstract(cls)
        }
        return [{"name": name, "expr": repr(name)} for name in sorted(names)]

    def _preset(self, value: object) -> str | None:
        """A default that is a preset of its own class (MVToolsPreset.HQ_SAD, a new
        object on each access): the preset's expression; the class's presets become
        its "enums" entry."""
        cls = type(value)
        if cls.__module__ == "builtins" or issubclass(cls, enum.Enum):
            return None
        binding = self.paths.binding(cls)
        if binding is None:
            return None
        base = ".".join(binding[1])
        members: list[dict[str, str]] = []
        match = None
        for attribute in dir(cls):
            if attribute.startswith("_"):
                continue
            try:
                preset = getattr(cls, attribute)
                if not isinstance(preset, cls):
                    continue
                expr = f"{base}.{attribute}"
                members.append({"name": attribute, "expr": expr})
                if match is None and (preset is value or preset == value):
                    match = expr
            except Exception:
                continue
        if match is not None:
            self.enums.setdefault(
                _key(cls), {"import": import_line(binding), "members": members}
            )
        return match

    def _literal(self, value: object) -> str:
        return self._member(value) if isinstance(value, enum.Enum) else repr(value)

    def format(self, obj: Any) -> str:
        """A resolved annotation as text: classes by their qualified names
        (VideoNode), enums and aliases with choices by their "enums" keys."""
        if obj is None or obj is type(None):
            return "None"
        if obj is Ellipsis:
            return "..."
        if isinstance(obj, str):
            return obj
        if isinstance(obj, annotationlib.ForwardRef):
            # A partly resolved union names its resolved parts by placeholders.
            text = obj.__forward_arg__
            extra = getattr(obj, "__extra_names__", None) or {}
            if text in extra:
                return self.format(extra[text])
            for placeholder, value in extra.items():
                text = text.replace(placeholder, self.format(value))
            return text
        if isinstance(obj, list):
            return "[" + ", ".join(self.format(arg) for arg in obj) + "]"
        if isinstance(obj, typing.TypeAliasType):
            return _key(obj) if self._alias(obj) else obj.__name__
        origin = typing.get_origin(obj)
        args = typing.get_args(obj)
        if origin is typing.Union:
            return " | ".join(self.format(arg) for arg in args)
        if origin is typing.Literal:
            return f"Literal[{', '.join(self._literal(arg) for arg in args)}]"
        if origin is typing.Annotated:
            return self.format(args[0])
        if origin is not None:
            if not args:
                return self.format(origin)
            return f"{self.format(origin)}[{', '.join(self.format(a) for a in args)}]"
        if isinstance(obj, type):
            if issubclass(obj, enum.Enum) and self._enum(obj):
                return _key(obj)
            if obj.__module__ == "builtins" or "<" in obj.__qualname__:
                return obj.__name__
            return obj.__qualname__
        if isinstance(obj, (typing.TypeVar, typing.ParamSpec, typing.TypeVarTuple)):
            return obj.__name__
        text = repr(obj)
        return text.removeprefix("typing.")

    def annotation(
        self, value: object, globals_: Mapping[str, Any], locals_: Mapping[str, Any]
    ) -> str:
        """An annotation (source text from the signature) resolved in the function's
        own module, as text; the source text when it does not resolve."""
        if value is inspect.Parameter.empty:
            return ""
        if not isinstance(value, str):
            return self.format(value)
        try:
            resolved = annotationlib.ForwardRef(value).evaluate(
                globals=dict(globals_),
                locals=dict(locals_),
                format=annotationlib.Format.FORWARDREF,
            )
        except Exception:
            return value
        text = self.format(resolved)
        return value if _PLACEHOLDER in text else text

    def default(self, value: object) -> str | None:
        """A default as source text: an enum member or a preset as its "enums"
        expression, a class, function or exported object as its dotted path
        (vskernels.Catrom), else its repr."""
        if value is inspect.Parameter.empty:
            return None
        if isinstance(value, enum.Enum):
            return self._member(value)
        is_class = isinstance(value, type)
        if (
            is_class
            or inspect.isfunction(value)
            or type(value).__module__ != "builtins"
        ):
            # A class, a function or an exported object (the gauss_blur wrapper).
            dotted = self.paths.dotted(value)
            if dotted is not None:
                return dotted
            preset = None if is_class else self._preset(value)
            if preset is not None:
                return preset
        try:
            # Without memory addresses ("<function <lambda>>"), so that equal scans
            # write equal catalogs.
            return _ADDRESS.sub("", repr(value))
        except Exception:
            return f"<{type(value).__name__}>"

    # Signatures.

    def params(
        self,
        signature: inspect.Signature,
        globals_: Mapping[str, Any],
        locals_: Mapping[str, Any],
        skip_first: bool = False,
    ) -> list[dict[str, Any]]:
        params: list[dict[str, Any]] = []
        values = list(signature.parameters.values())
        for param in values[1:] if skip_first else values:
            kind = _KINDS.get(param.kind)
            if kind is None:
                continue
            annotation = self.annotation(param.annotation, globals_, locals_)
            if annotation == "Never":
                continue
            default = None if kind == "var_keyword" else self.default(param.default)
            params.append(_param(param.name, annotation, default, kind))
        return params

    def doc(self, obj: object) -> tuple[str, dict[str, str]]:
        """The description and argument help of a docstring (inherited methods share
        one)."""
        key = id(obj)
        if key not in self._docs:
            self._docs[key] = (obj, split_doc(inspect.getdoc(obj) or ""))
        return self._docs[key][1]

    def add(self, entry: dict[str, Any]) -> None:
        if entry["id"] in self._ids:
            self.errors.append(f"{entry['id']}: listed twice; the first one is kept")
            return
        self._ids.add(entry["id"])
        self.functions.append(entry)

    # Python modules.

    def scan_module(self, top: str, module: types.ModuleType, source: str) -> None:
        for name, value in _public_members(module):
            try:
                if isinstance(value, type):
                    self.scan_class(top, name, value, source)
                elif callable(value) and not isinstance(value, types.ModuleType):
                    self.scan_function(top, name, value, source)
            except Exception as error:
                self.error(f"{top}.{name}", error)

    def scan_function(self, top: str, name: str, value: object, source: str) -> None:
        target = _unwrapped(value)
        if not inspect.isfunction(target) or not _owned(target, top):
            return
        signature = inspect.signature(
            target, annotation_format=annotationlib.Format.STRING
        )
        globals_ = target.__globals__
        returns = self.annotation(signature.return_annotation, globals_, {})
        # Library helpers that make no clip are no nodes; a user function without a
        # return annotation is taken at its word.
        if "VideoNode" not in returns and (source != "user" or returns):
            return
        doc, arg_docs = self.doc(target)
        self.add(
            {
                "id": f"{top}.{name}",
                "source": source,
                "module": top,
                "import": f"from {top} import {name}",
                "call": name,
                "kind": "function",
                "params": self.params(signature, globals_, {}),
                "returns": returns,
                "doc": doc,
                "arg_docs": arg_docs,
            }
        )

    def scan_enum(self, top: str, name: str, cls: type[enum.Enum], source: str) -> None:
        """An enum whose members filter a clip when called (vsdenoise's
        `Prefilter.DFTTEST(clip)`): one entry of kind "members", its member names
        under "members", called as `<name>.<member>(<clip>, <params>)`."""
        owner = next(
            (k for k in cls.__mro__ if "__call__" in vars(k) and k is not object),
            None,
        )
        function = vars(owner)["__call__"] if owner is not None else None
        if not inspect.isfunction(function) or not _owned(function, top):
            return
        signature = inspect.signature(
            function, annotation_format=annotationlib.Format.STRING
        )
        globals_ = function.__globals__
        returns = self.annotation(signature.return_annotation, globals_, vars(owner))
        params = self.params(signature, globals_, vars(owner), skip_first=True)
        # Members that take a clip first (not *clips; not vstools' decorators, which
        # take a clip or a function).
        members = [member.name for member in cls]
        first = params[0]["annotation"] if params else ""
        if (
            "VideoNode" not in returns
            or not members
            or "VideoNode" not in first
            or "Callable" in first
        ):
            return
        doc, arg_docs = self.doc(function)
        self.add(
            {
                "id": f"{top}.{name}",
                "source": source,
                "module": top,
                "import": f"from {top} import {name}",
                "call": f"{name}.<member>",
                "kind": "members",
                "members": members,
                "params": params,
                "returns": returns,
                "doc": self.doc(cls)[0] or doc,
                "arg_docs": arg_docs,
            }
        )

    def scan_class(self, top: str, name: str, cls: type, source: str) -> None:
        """Class-based operations: each public method that returns a clip, called as
        `Cls(<init>).method(<clip>, <params>)` (sources such as
        `BestSource(<init>).source(<path>)` take no clip)."""
        if issubclass(cls, enum.Enum) and _owned(cls, top):
            self.scan_enum(top, name, cls, source)
            return
        if (
            not _owned(cls, top)
            or issubclass(cls, (enum.Enum, BaseException))
            or inspect.isabstract(cls)
            or typing.is_typeddict(cls)
            or getattr(cls, "_is_protocol", False)
        ):
            return
        init: list[dict[str, Any]] | None = None
        init_docs: dict[str, str] = {}
        for attribute in sorted(dir(cls)):
            if attribute.startswith("_"):
                continue
            try:
                function = inspect.getattr_static(cls, attribute)
            except AttributeError:
                continue
            # jetpytools' inject_self keeps the method on `_function`, and so does an
            # inject_kwargs_params inside it (the edge detectors' edgemask).
            for _ in range(_WRAPPER_DEPTH):
                if inspect.isfunction(function):
                    break
                function = getattr(function, "_function", None)
            if not inspect.isfunction(function):
                continue
            try:
                if init is None:
                    init, init_docs = self._init(cls)
                self._scan_method(
                    (top, name, source), cls, attribute, function, init, init_docs
                )
            except Exception as error:
                self.error(f"{top}.{name}.{attribute}", error)
                if init is None:
                    return

    def _init(self, cls: type) -> tuple[list[dict[str, Any]], dict[str, str]]:
        """The constructor's parameters and their help (class and __init__ Args)."""
        signature = inspect.signature(
            cls, annotation_format=annotationlib.Format.STRING
        )
        init = getattr(cls, "__init__", None)
        globals_ = getattr(init, "__globals__", None) or vars(
            sys.modules[cls.__module__]
        )
        docs = {**self.doc(cls)[1], **self.doc(init)[1]} if init is not None else {}
        return self.params(signature, globals_, vars(cls)), docs

    def _scan_method(
        self,
        where: tuple[str, str, str],
        cls: type,
        attribute: str,
        function: Callable[..., Any],
        init: list[dict[str, Any]],
        init_docs: dict[str, str],
    ) -> None:
        top, name, source = where
        owner = next((k for k in cls.__mro__ if attribute in vars(k)), cls)
        signature = inspect.signature(
            function, annotation_format=annotationlib.Format.STRING
        )
        globals_ = function.__globals__
        returns = self.annotation(signature.return_annotation, globals_, vars(owner))
        if "VideoNode" not in returns:
            return
        params = self.params(signature, globals_, vars(owner), skip_first=True)
        doc, arg_docs = self.doc(function)
        self.add(
            {
                "id": f"{top}.{name}.{attribute}",
                "source": source,
                "module": top,
                "import": f"from {top} import {name}",
                "call": f"{name}(...).{attribute}",
                "kind": "bound",
                "init_params": init,
                "params": params,
                "returns": returns,
                "doc": doc,
                "arg_docs": {**arg_docs, **init_docs},
            }
        )

    # Plugins.

    def scan_plugins(self, core: Any) -> None:
        for plugin in sorted(core.plugins(), key=lambda plugin: plugin.namespace):
            try:
                for function in plugin.functions():
                    returns = plugin_returns(function.return_signature)
                    if "VideoNode" not in returns:
                        continue
                    self.add(
                        {
                            "id": f"{plugin.namespace}.{function.name}",
                            "source": "plugin",
                            "module": plugin.namespace,
                            "import": "",
                            "call": f"core.{plugin.namespace}.{function.name}",
                            "kind": "plugin",
                            "params": plugin_params(function.signature),
                            "returns": returns,
                            "doc": plugin.name,
                            "arg_docs": {},
                        }
                    )
            except Exception as error:
                self.error(f"plugin {plugin.namespace}", error)


def _union_parts(value: object) -> list[object]:
    if typing.get_origin(value) is typing.Union:
        return list(typing.get_args(value))
    return [value]


def failure(where: str, error: BaseException) -> str:
    return f"{where}: {type(error).__name__}: {error}"[:ERROR_CHARS]


def _import_all(
    tops: Iterable[str], label: str, errors: list[str]
) -> dict[str, types.ModuleType]:
    modules: dict[str, types.ModuleType] = {}
    for top in tops:
        try:
            modules[top] = importlib.import_module(top)
        except (Exception, SystemExit) as error:
            errors.append(failure(f"{label} {top}", error))
    return modules


def _import_user_files(
    user_dir: Path, errors: list[str]
) -> dict[str, types.ModuleType]:
    """The user folder's *.py files as modules named by their file names. A file
    whose name an installed module has is skipped: the generated script would import
    that module instead."""
    if not user_dir.is_dir():
        return {}
    files: list[Path] = []
    for path in sorted(user_dir.glob("*.py")):
        if not path.stem.isidentifier():
            errors.append(
                f"user file {path.name}: the name is not a Python module name"
            )
            continue
        if path.stem in SCRIPT_NAMES or _NODE_VAR.fullmatch(path.stem):
            errors.append(
                f"user file {path.name}: chaiNNer's scripts use the name {path.stem};"
                " rename the file"
            )
            continue
        try:
            taken = (
                path.stem in sys.modules
                or importlib.util.find_spec(path.stem)
                or importlib.machinery.PathFinder.find_spec(
                    path.stem, [str(_BACKEND_SRC)]
                )
            )
        except (ImportError, ValueError):
            taken = True
        if taken:
            errors.append(
                f"user file {path.name}: an installed module or one of chaiNNer's has"
                f" the name {path.stem}; rename the file"
            )
            continue
        files.append(path)
    # Imported as the generated scripts import them, so user files may import each
    # other; no installed module has these names.
    sys.path.append(str(user_dir))
    modules: dict[str, types.ModuleType] = {}
    for path in files:
        try:
            modules[path.stem] = importlib.import_module(path.stem)
        except (Exception, SystemExit) as error:
            errors.append(failure(f"user file {path.name}", error))
    return modules


def scan(user_dir: Path, modules_file: Path) -> dict[str, Any]:
    """The catalog: everything is imported first, so that subclass lists (the
    kernels of KernelLike) include the user's classes."""
    vapoursynth = importlib.import_module("vapoursynth")
    errors: list[str] = []
    distributions = importlib.metadata.packages_distributions()
    jetpack = _import_all(
        sorted(
            top
            for top, names in distributions.items()
            if _normalized(JETPACK) in map(_normalized, names)
        ),
        "vs-jetpack module",
        errors,
    )
    try:
        names = user_module_names(modules_file)
    except ValueError as error:
        errors.append(str(error))
        names = []
    user_tops = [top for name in names for top in import_names(name, distributions)]
    for top in [top for top in user_tops if top in SCRIPT_NAMES]:
        errors.append(f"user module {top}: chaiNNer's scripts use the name {top}")
        user_tops.remove(top)
    user_modules = _import_all(dict.fromkeys(user_tops), "user module", errors)
    user_files = _import_user_files(user_dir, errors)

    modules = {"vapoursynth": vapoursynth, **jetpack, **user_modules, **user_files}
    scanner = Scanner(Paths(modules), errors)
    scanner.seed(modules)
    for top, module in jetpack.items():
        scanner.scan_module(top, module, "jetpack")
    scanner.scan_plugins(vapoursynth.core)
    for top, module in {**user_modules, **user_files}.items():
        scanner.scan_module(top, module, "user")
    return {
        "schema": SCHEMA,
        # chaiNNer writes the key of the scanned state.
        "key": {},
        "functions": scanner.functions,
        "enums": {
            key: entry
            for key, entry in sorted(scanner.enums.items())
            if entry and entry["members"]
        },
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Lists VapourSynth functions as chaiNNer's catalog JSON."
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--user-dir", required=True, type=Path)
    parser.add_argument("--modules-file", required=True, type=Path)
    args = parser.parse_args(argv)
    warnings.simplefilter("ignore")
    catalog = scan(args.user_dir, args.modules_file)
    args.out.write_text(
        json.dumps(catalog, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
