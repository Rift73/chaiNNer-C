"""The VapourSynth family nodes, generated from the catalog: one node per family
(FAMILIES) whose Method dropdown lists that family's functions, and whose other inputs
are each method's parameters, shown only while that method is picked (typed by
adapters.py; Simple shows the method's main ones, Advanced all). A node writes one
statement of the chain's script (script.VideoClip); with Use as filter on it writes
none and gives the callable instead (script.VSFilter), for a function parameter of
the node it is wired into: `vsdeinterlace.vfm(clip,
postprocess=vsaa.NNEDI3(...).deinterlace)`. Every name a statement uses is
module-qualified (`import vsdenoise`, `vsdenoise.bm3d(...)`), so that no two imports
of a script can shadow each other; plugins go through the script's `core`.

Input ids: 0 main clip, 1 Method, 2 Simple/Advanced, 3 Use as filter; every other id is
catalog.input_ids' hash of family/method/parameter, so saved chains keep their values
when the catalog changes. A value an input cannot take (a choice a newer catalog
dropped, no value at all) never fails the run by itself: the parameter takes its
default and one warning names it."""

from __future__ import annotations

import ast
import functools
import keyword
import operator
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence, TypeVar, cast

from sanic.log import logger

from api import BaseInput, Group, NestedGroup, NodeContext, NodeGroup, SpecialSuggestion

from ...condition import Condition
from ...groups import if_enum_group, if_group, optional_list_group, required
from ...properties.inputs import (
    BoolInput,
    DropDownGroup,
    DropDownInput,
    TextInput,
    VideoClipInput,
    VSFilterInput,
)
from ...properties.outputs import VideoClipOutput
from .adapters import Adapter, Enums, adapter_for
from .catalog import CLASS, FAMILIES, input_ids
from .catalog_types import Method, Param
from .script import VideoClip, VSFilter, clip_var

LOAD_CLIP = "Load Clip"
ANY_FUNCTION = "Any Function"
# Families without the Use as filter switch.
NO_FILTER = frozenset({LOAD_CLIP, "User Functions", ANY_FUNCTION})
# Hand-written nodes cover these (Split Planes, with its three outputs).
EXCLUDED = frozenset({"vstools.split"})

MAIN_ID = 0
METHOD_ID = 1
MODE_ID = 2
FILTER_ID = 3
# Any Function: the `name = value` rows, and the clip-or-filter sockets with the name
# of the argument each one is.
ANY_ROW_IDS = tuple(range(10, 22))
ANY_SOCKET_IDS = (30, 31)
ANY_SOCKET_NAME_IDS = (32, 33)
# A main parameter that takes a list of clips (std.Splice's clips) gets "Clip 2" to
# "Clip 8" beside the main clip.
MORE_CLIPS = 7

_CLIP_KINDS = ("clip", "optional_clip")
# The parameter name a filter's lambda gives the clip.
_FILTER_CLIP = "clip"
# The label of a **kwargs parameter.
_MORE = "More arguments"

# Icon and description of each family node.
FAMILY_TEXT: dict[str, tuple[str, str]] = {
    "Load Clip": ("MdVideoCameraBack", "Opens a video file as a VapourSynth clip."),
    "Clip Properties": (
        "ImInfo",
        (
            "Sets what a clip says about itself: colour information, aspect ratio, frame"
            " rate."
        ),
    ),
    "Convert Format": (
        "MdChangeCircle",
        "Converts a clip to another format, bit depth or colour space.",
    ),
    "Resize": (
        "MdOutlinePhotoSizeSelectLarge",
        "Scales, descales, crops and shifts clips.",
    ),
    "Cut and Splice": ("MdContentCut", "Cuts, joins and replaces frames."),
    "Planes": ("BsLayersHalf", "Joins, merges and stacks the planes of clips."),
    "Deinterlace": ("MdVideoLabel", "Deinterlaces, field-matches and decimates."),
    "Denoise": ("CgEditNoise", "Removes noise."),
    "Sharpen and Blur": ("MdBlurOn", "Sharpens, blurs and cleans up clips."),
    "Anti-alias": ("MdAutoFixHigh", "Smooths jagged edges."),
    "Dehalo": ("MdLens", "Removes halos and ringing."),
    "Deband and Grain": ("MdGradient", "Removes banding and adds grain."),
    "Masks": ("MdOutlineOpacity", "Makes masks: edges, details, ranges, credits."),
    "Colour": ("MdColorLens", "Adjusts and matches colours."),
    "Expression": ("MdCalculate", "Computes pixels with expressions."),
    "User Functions": (
        "MdInput",
        (
            "Runs your own VapourSynth functions: the .py files of the VapourSynth user"
            " folder and the modules listed in user_modules.toml."
        ),
    ),
    "Any Function": (
        "MdAllOut",
        (
            "Runs any other VapourSynth or vs-jetpack function. Arguments are typed as"
            " `name = value` with a Python value (a number, 'text', True, a list); a clip"
            " or a filter goes in a socket, with the name of the argument it is."
        ),
    ),
}
_METHOD_HELP = (
    "Pick what the node does in Method; its settings appear below it (Simple shows the"
    " main ones, Advanced all of them)."
)
_FILTER_HELP = (
    "Gives this method itself instead of its result, for an argument of another node"
    " that takes a function (Field match's post-process, Motion-compensated degrain's"
    " prefilter). The Clip input is not used then."
)

I = TypeVar("I", bound=BaseInput)
Item = BaseInput | NestedGroup


class Dropped:
    """A value an input could not take (saved with an older catalog): the run uses
    the parameter's default."""

    def __init__(self, value: object, reason: str) -> None:
        self.value = value
        self.reason = reason


def _tolerant_enforce(self: BaseInput, value: object) -> object:
    if value is None:
        return None
    try:
        return self.enforce(value)
    except (AssertionError, ValueError, TypeError) as error:
        return Dropped(value, str(error))


# Each input class's tolerant subclass.
_TOLERANT: dict[type, type] = {}


def _tolerant(item: I) -> I:
    """The input, made to pass None and values it cannot take to the node (as None and
    Dropped) instead of failing the run: an input of a method that is not picked has
    no value, and a saved value may be older than the catalog. Its class becomes a
    subclass that only replaces enforce_ (so to_dict and isinstance are unchanged)."""
    base = type(item)
    tolerant = _TOLERANT.get(base)
    if tolerant is None:
        tolerant = type(base.__name__, (base,), {"enforce_": _tolerant_enforce})
        _TOLERANT[base] = tolerant
    item.__class__ = cast("type[I]", tolerant)
    return item


def slug(family: str) -> str:
    return family.lower().replace(" ", "_").replace("-", "_")


def schema_id(family: str) -> str:
    return f"chainner:vapoursynth:{slug(family)}"


def _plain(name: str) -> str:
    """A parameter name as a label: "sample_grid_model" -> "Sample grid model"."""
    text = name.strip("_").replace("_", " ") or name
    return text[:1].upper() + text[1:]


@dataclass(frozen=True)
class _Slot:
    """The input of a parameter (of one or more members of a group)."""

    input_id: int
    adapter: Adapter
    label: str
    # The input may have no value (else chaiNNer always gives it one).
    optional: bool


@dataclass(frozen=True)
class _Member:
    """How to call a method (or one member of a group)."""

    method: Method
    # The parameter the main clip input fills, and its adapter.
    main: Param | None
    main_adapter: Adapter | None
    # The input of every other parameter, by (constructor argument, name).
    slots: Mapping[tuple[bool, str], _Slot]


@dataclass
class _Plan:
    """A method of a family node: its members and inputs."""

    method: Method
    members: dict[str, _Member] = field(default_factory=dict)
    # Simple and Advanced inputs, each with the group members that have it (empty:
    # all of them).
    simple: list[tuple[frozenset[str], Item]] = field(default_factory=list)
    advanced: list[tuple[frozenset[str], Item]] = field(default_factory=list)
    class_id: int | None = None
    more_clips: tuple[int, ...] = ()

    def main_required(self) -> bool:
        return all(
            m.main is not None and m.main.default is None for m in self.members.values()
        )


# A method's parameters: (constructor argument, parameter, adapter).
_Params = list[tuple[bool, Param, Adapter]]


def _member_params(
    member: Method, enums: Enums, has_main: bool
) -> tuple[Param | None, Adapter | None, _Params]:
    """The main parameter (the method's first clip parameter) with its adapter, and the
    others: (constructor argument, parameter, adapter)."""
    plugin = member.kind == "plugin"
    params = [
        (init, param, adapter_for(param, enums, plugin))
        for init, group in ((True, member.init_params), (False, member.params))
        for param in group
    ]
    main = next(
        (
            (param, adapter)
            for init, param, adapter in params
            if has_main and not init and adapter.kind in _CLIP_KINDS
        ),
        None,
    )
    if main is None:
        return None, None, params
    return main[0], main[1], [p for p in params if p[1] is not main[0]]


def _start_value(member: Method, param: Param, adapter: Adapter) -> object:
    """The overlay's default of a parameter as its input's starting value, or None
    (the function's default); one the input cannot take is logged and left out."""
    if param.name not in member.defaults:
        return None
    try:
        return adapter.start_value(member.defaults[param.name])
    except ValueError as error:
        logger.warning(
            "The overlay default of %s for %s is left out: %s",
            member.id,
            param.name,
            error,
        )
        return None


def _takes_clip_list(main: Param, others: _Params) -> bool:
    """The main parameter takes a list of clips and no other parameter is a clip:
    std.Splice(clips), std.StackHorizontal(clips); not vstools.join, which has a
    parameter per plane."""
    return "Sequence[VideoNode]" in main.annotation.replace("vs.", "") and not any(
        adapter.kind in (*_CLIP_KINDS, "clip_or_filter") for _, _, adapter in others
    )


def _variant_keys(
    variants: Mapping[tuple[bool, str], Mapping[Param, set[str]]], group: bool
) -> dict[tuple[bool, Param], str]:
    """The input id key of each parameter: its name. In a group a constructor argument
    is "init.<name>", and when members differ in a parameter's annotation or default,
    the variant most of them have keeps the name and the others add
    "|<annotation>=<default>"."""
    keys: dict[tuple[bool, Param], str] = {}
    for (init, name), by_param in variants.items():
        base = f"init.{name}" if init and group else name
        ranked = sorted(by_param, key=lambda p: (-len(by_param[p]), repr(p)))
        for index, param in enumerate(ranked):
            keys[init, param] = (
                base if index == 0 else f"{base}|{param.annotation}={param.default}"
            )
    return keys


@dataclass
class _Parsed:
    """A method's members with their main and other parameters, each parameter
    variant with the members that have it, and the input id keys."""

    method: Method
    members: list[tuple[Method, Param | None, Adapter | None, _Params]]
    variants: dict[tuple[bool, str], dict[Param, set[str]]]
    keys: dict[tuple[bool, Param], str]
    more: int


class _Family:
    """The inputs and plans of one family node."""

    def __init__(self, family: str, methods: Sequence[Method], enums: Enums) -> None:
        self.family = family
        self.enums = enums
        self.has_main = family != LOAD_CLIP
        self.switch = family not in NO_FILTER
        # Common first (methods with an overlay), each part by label; Any Function's
        # by module (its dropdown groups).
        self.methods = sorted(
            methods,
            key=lambda m: (
                m.id.split(".", 1)[0] if family == ANY_FUNCTION else not m.common,
                m.label.lower(),
            ),
        )
        self.plans: dict[str, _Plan] = {}

    def _parse(self, method: Method) -> _Parsed:
        members: list[tuple[Method, Param | None, Adapter | None, _Params]] = []
        variants: dict[tuple[bool, str], dict[Param, set[str]]] = {}
        for member in method.members or (method,):
            main, main_adapter, params = _member_params(
                member, self.enums, self.has_main
            )
            members.append((member, main, main_adapter, params))
            for init, param, _ in params:
                by_param = variants.setdefault((init, param.name), {})
                by_param.setdefault(param, set()).add(member.choice)
        _, main, _, params = members[0]
        more = (
            MORE_CLIPS
            if not method.members
            and main is not None
            and _takes_clip_list(main, params)
            else 0
        )
        keys = _variant_keys(variants, bool(method.members))
        return _Parsed(method, members, variants, keys, more)

    def build(self) -> None:
        parsed = [self._parse(method) for method in self.methods]
        wanted: list[tuple[str, str]] = []
        for p in parsed:
            if p.method.members:
                wanted.append((p.method.id, CLASS))
            if p.more:
                main = p.members[0][1]
                assert main is not None
                wanted.extend(
                    (p.method.id, f"{main.name}#{n}") for n in range(2, p.more + 2)
                )
            wanted.extend((p.method.id, key) for key in p.keys.values())
        ids = input_ids(self.family, wanted)
        for p in parsed:
            self.plans[p.method.id] = self._plan(p, ids)

    def _plan(self, parsed: _Parsed, ids: Mapping[tuple[str, str], int]) -> _Plan:
        method = parsed.method
        plan = _Plan(method)
        everyone = frozenset(member.choice for member, *_ in parsed.members)
        slots: dict[str, _Slot] = {}
        for member, main, main_adapter, params in parsed.members:
            by_name: dict[tuple[bool, str], _Slot] = {}
            for init, param, adapter in params:
                key = parsed.keys[init, param]
                slot = slots.get(key)
                if slot is None:
                    label = member.labels.get(param.name) or (
                        _MORE if param.kind == "var_keyword" else _plain(param.name)
                    )
                    start = _start_value(member, param, adapter)
                    made = adapter.make_input(label, param, ids[method.id, key], start)
                    slot = slots[key] = _Slot(made.id, adapter, label, made.optional)
                    doc = member.arg_docs.get(param.name)
                    if doc:
                        made.with_docs(doc)
                    choices = frozenset(parsed.variants[init, param.name][param])
                    simple = param.name in method.simple or (
                        param.default is None and param.kind != "var_keyword"
                    )
                    (plan.simple if simple else plan.advanced).append(
                        (
                            frozenset() if choices == everyone else choices,
                            _tolerant(made),
                        )
                    )
                by_name[init, param.name] = slot
            plan.members[member.choice] = _Member(member, main, main_adapter, by_name)
        unknown = {k for m, *_ in parsed.members for k in m.defaults} - {
            name for _, name in parsed.variants
        }
        if unknown:
            logger.warning(
                "The overlay of %s sets defaults of %s, which it does not take.",
                method.id,
                ", ".join(sorted(unknown)),
            )
        if method.members:
            plan.class_id = ids[method.id, CLASS]
            choices = [m.choice for m in method.members]
            dropdown = DropDownInput(
                input_type="string",
                label=method.labels.get(CLASS, "Class"),
                options=[{"option": c, "value": c} for c in choices],
                default_value=method.default_choice or choices[0],
            )
            plan.simple.insert(
                0, (frozenset(), _tolerant(dropdown.with_id(plan.class_id)))
            )
        if parsed.more:
            main = parsed.members[0][1]
            assert main is not None
            plan.more_clips = tuple(
                ids[method.id, f"{main.name}#{n}"] for n in range(2, parsed.more + 2)
            )
            sockets = [
                _tolerant(VideoClipInput(f"Clip {n}").make_optional().with_id(input_id))
                for n, input_id in enumerate(plan.more_clips, start=2)
            ]
            plan.simple.insert(0, (frozenset(), optional_list_group(*sockets)))
        return plan

    # The node's inputs.

    def _method_input(self) -> BaseInput:
        if self.family == ANY_FUNCTION:
            groups = self._module_groups()
        else:
            common = [m for m in self.methods if m.common]
            rest = [m for m in self.methods if not m.common]
            groups = [
                DropDownGroup(label, part[0].id)
                for label, part in (("Common", common), ("All", rest))
                if part
            ]
        return _tolerant(
            DropDownInput(
                input_type="string",
                label="Method",
                options=[{"option": m.label, "value": m.id} for m in self.methods],
                default_value=self.methods[0].id,
                groups=groups,
            ).with_id(METHOD_ID)
        )

    def _module_groups(self) -> list[DropDownGroup]:
        groups: dict[str, DropDownGroup] = {}
        for method in self.methods:
            module = method.id.split(".", 1)[0]
            groups.setdefault(module, DropDownGroup(module, method.id))
        return list(groups.values())

    def _main_input(self) -> Item:
        clip = _tolerant(VideoClipInput("Clip").make_optional().with_id(MAIN_ID))
        if self.family == ANY_FUNCTION:
            return clip
        optional = [m.id for m in self.methods if not self.plans[m.id].main_required()]
        item = required(~Condition.enum(METHOD_ID, optional) if optional else None)(
            clip
        )
        return if_group(Condition.bool(FILTER_ID, False))(item) if self.switch else item

    def inputs(self) -> list[Item]:
        items: list[Item] = []
        if self.has_main:
            items.append(self._main_input())
        items.append(self._method_input())
        if self.family == ANY_FUNCTION:
            return [*items, *_any_inputs()]
        items.append(
            _tolerant(
                DropDownInput(
                    input_type="string",
                    label="Inputs",
                    options=[
                        {"option": "Simple", "value": "simple"},
                        {"option": "Advanced", "value": "advanced"},
                    ],
                    default_value="simple",
                    preferred_style="tabs",
                ).with_id(MODE_ID)
            )
        )
        if self.switch:
            items.append(
                _tolerant(
                    BoolInput("Use as filter", default=False)
                    .with_id(FILTER_ID)
                    .with_docs(_FILTER_HELP)
                )
            )
        for method in self.methods:
            plan = self.plans[method.id]
            body = _by_choices(plan.simple, plan.class_id)
            if plan.advanced:
                body.append(
                    if_group(Condition.enum(MODE_ID, "advanced"))(
                        *_by_choices(plan.advanced, plan.class_id)
                    )
                )
            # A method without inputs of its own (std.Reverse) gets no group: the
            # frontend rejects an empty one and then shows a blank window.
            if body:
                items.append(if_enum_group(METHOD_ID, method.id)(*body))
        return items


def _by_choices(
    entries: Sequence[tuple[frozenset[str], Item]], class_id: int | None
) -> list[Item]:
    """The inputs; those only some group members have go in a group shown for them,
    one per set of members, where its first input is."""
    buckets: dict[frozenset[str], list[Item]] = {}
    for choices, item in entries:
        buckets.setdefault(choices, []).append(item)
    items: list[Item] = []
    for choices, bucket in buckets.items():
        if not choices or class_id is None:
            items.extend(bucket)
        else:
            items.append(if_enum_group(class_id, sorted(choices))(*bucket))
    return items


def _any_inputs() -> list[Item]:
    rows = [
        _tolerant(
            TextInput(f"Argument {n}", placeholder="name = value, e.g. sigma = 0.8")
            .make_optional()
            .with_id(input_id)
        )
        for n, input_id in enumerate(ANY_ROW_IDS, start=1)
    ]
    items: list[Item] = [optional_list_group(*rows)]
    for n, (socket_id, name_id) in enumerate(
        zip(ANY_SOCKET_IDS, ANY_SOCKET_NAME_IDS), start=1
    ):
        items.append(
            _tolerant(
                VSFilterInput(f"Clip or filter {n}").make_optional().with_id(socket_id)
            )
        )
        items.append(
            _tolerant(
                TextInput(f"Argument name {n}", placeholder="e.g. ref")
                .make_optional()
                .with_id(name_id)
            )
        )
    return items


def _flat(items: Iterable[Item]) -> list[BaseInput]:
    """The inputs in the order the node lists them (and the run gets their values)."""
    found: list[BaseInput] = []
    for item in items:
        if isinstance(item, Group):
            found.extend(_flat(item.items))
        else:
            found.append(item)
    return found


# Running a node.


@dataclass
class _Call:
    """The arguments of one call, as source text."""

    positional: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    deps: list[VideoClip | VSFilter] = field(default_factory=list)
    imports: set[str] = field(default_factory=set)
    # Positional-only parameters left out, until a later one is given.
    gaps: list[Param] = field(default_factory=list)

    def text(self) -> str:
        return ", ".join([*self.positional, *self.keywords])

    def add(self, param: Param, source: str, positional: bool) -> None:
        if param.kind == "var_keyword":
            self.keywords.append(f"**{source}")
        elif positional or param.kind == "positional":
            self.positional.extend(_gap_source(gap) for gap in self.gaps)
            self.gaps.clear()
            self.positional.append(source)
        else:
            self.keywords.append(f"{param.name}={source}")

    def skip(self, param: Param) -> None:
        if param.kind == "positional":
            self.gaps.append(param)


def _gap_source(param: Param) -> str:
    """A positional-only argument left out before a given one: its default."""
    try:
        if param.default is not None:
            ast.parse(param.default, mode="eval")
            return param.default
    except SyntaxError:
        pass
    raise ValueError(
        f"{_plain(param.name)} needs a value: an argument after it is set, and its"
        " default cannot be written in a script."
    )


class _Run:
    """The values of one run by input id, and the parameters that took their
    default because their value was missing or not valid."""

    def __init__(self, label: str, values: Mapping[int, object]) -> None:
        self.label = label
        self.values = values
        self.defaulted: list[str] = []

    def value(self, input_id: int, label: str, optional: bool) -> object:
        value = self.values.get(input_id)
        if isinstance(value, Dropped):
            self.defaulted.append(label)
            return None
        if value is None and not optional:
            self.defaulted.append(label)
        return value

    def warn(self) -> None:
        if self.defaulted:
            logger.warning(
                "%s: the values of %s are missing or no longer valid (did the"
                " VapourSynth functions change?); their defaults are used.",
                self.label,
                ", ".join(dict.fromkeys(self.defaulted)),
            )


def _args(
    run: _Run, member: _Member, init: bool, call: _Call, main: str | None
) -> None:
    """Adds `member`'s constructor arguments (init) or its method's arguments; `main`
    is the main clip's source."""
    for index, param in enumerate(
        member.method.init_params if init else member.method.params
    ):
        if param is member.main:
            if main is not None:
                call.add(param, main, index == 0 and not call.keywords)
            elif param.default is None:
                raise ValueError("Connect a clip to the Clip input.")
            else:
                call.skip(param)
            continue
        slot = member.slots[init, param.name]
        value = run.value(slot.input_id, slot.label, slot.optional)
        try:
            source = slot.adapter.to_source(value)
        except ValueError as error:
            raise ValueError(f"{slot.label}: {error}") from error
        if source is None:
            if param.default is None and param.kind != "var_keyword":
                raise ValueError(f"{slot.label} needs a value.")
            call.skip(param)
            continue
        if isinstance(value, (VideoClip, VSFilter)):
            call.deps.append(value)
        call.imports.update(slot.adapter.imports)
        call.add(param, source, False)


def _main_source(
    plan: _Plan, member: _Member, values: Mapping[int, object]
) -> tuple[str | None, list[VideoClip]]:
    """The main clip's source (with the clips of "Clip 2"... in a list) and clips."""
    clip = values.get(MAIN_ID)
    if member.main_adapter is None or not isinstance(clip, VideoClip):
        return None, []
    clips = [clip]
    clips.extend(
        c for c in (values.get(i) for i in plan.more_clips) if isinstance(c, VideoClip)
    )
    if len(clips) > 1:
        # A list of clips (Shift Clip Multi's) adds all of its clips.
        listed = (f"*{c.var}" if c.many else c.var for c in clips)
        return f"[{', '.join(listed)}]", clips
    return member.main_adapter.to_source(clip), clips


def _statement(
    plan: _Plan, run: _Run, node_id: str, as_filter: bool
) -> VideoClip | VSFilter:
    choice = ""
    if plan.class_id is not None:
        value = run.value(plan.class_id, plan.method.labels.get(CLASS, "Class"), False)
        choice = (
            value
            if isinstance(value, str) and value in plan.members
            else plan.method.default_choice or next(iter(plan.members))
        )
    member = plan.members[choice]
    method = member.method
    init = _Call()
    _args(run, member, True, init, None)
    call = _Call()
    if as_filter:
        if member.main is None:
            raise ValueError(f"{method.label} takes no clip, so it is no filter.")
        main, clips = _FILTER_CLIP, []
    else:
        main, clips = _main_source(plan, member, run.values)
    _args(run, member, False, call, main)
    run.warn()

    imports = {*init.imports, *call.imports, *_imports(method)}
    deps = (*clips, *init.deps, *call.deps)
    callee = method.call
    if method.kind == "bound":
        cls, _, name = callee.partition("(...)")
        callee = f"{cls}({init.text()}){name}"
    if not as_filter:
        var = clip_var(node_id)
        stmt = f"{var} = {callee}({call.text()})"
        return VideoClip(var, stmt, deps, frozenset(imports), node_id, method.many)
    # The callable: the method itself when nothing else is set, a partial with the
    # keyword arguments, a lambda when more positional ones are set.
    if call.positional == [_FILTER_CLIP] and not call.keywords:
        expr = callee
    elif call.positional == [_FILTER_CLIP]:
        expr = f"functools.partial({callee}, {', '.join(call.keywords)})"
        imports.add("import functools")
    else:
        expr = f"lambda {_FILTER_CLIP}: {callee}({call.text()})"
    return VSFilter(expr, deps, frozenset(imports), node_id)


def _any_statement(
    method: Method, values: Mapping[int, object], node_id: str
) -> VideoClip:
    """Any Function: the main clip first, then the rows and sockets by name."""
    call = _Call()
    clip = values.get(MAIN_ID)
    if isinstance(clip, VideoClip):
        call.positional.append(clip.var)
        call.deps.append(clip)
    names: set[str] = set()

    def named(name: str, where: str) -> str:
        if not name.isidentifier() or keyword.iskeyword(name):
            raise ValueError(f"{where}: {name!r} is not an argument name.")
        if name in names:
            raise ValueError(f"{where}: the argument {name} is given twice.")
        names.add(name)
        return name

    for n, input_id in enumerate(ANY_ROW_IDS, start=1):
        text = values.get(input_id)
        if not isinstance(text, str) or not text.strip():
            continue
        name, sep, value = text.partition("=")
        if not sep:
            raise ValueError(
                f"Argument {n}: write it as name = value, e.g. sigma = 0.8."
            )
        name = named(name.strip(), f"Argument {n}")
        source = Adapter("extra", name=name).to_source(value)
        if source is None:
            raise ValueError(f"Argument {n}: {name} needs a value after the =.")
        call.keywords.append(f"{name}={source}")
    for n, (socket_id, name_id) in enumerate(
        zip(ANY_SOCKET_IDS, ANY_SOCKET_NAME_IDS), start=1
    ):
        value = values.get(socket_id)
        if not isinstance(value, (VideoClip, VSFilter)):
            continue
        text = values.get(name_id)
        name = named(
            text.strip() if isinstance(text, str) else "", f"Argument name {n}"
        )
        source = value.var if isinstance(value, VideoClip) else value.expr
        call.keywords.append(f"{name}={source}")
        call.deps.append(value)
    var = clip_var(node_id)
    stmt = f"{var} = {method.call.replace('(...)', '()')}({call.text()})"
    return VideoClip(
        var, stmt, tuple(call.deps), _imports(method), node_id, method.many
    )


def _imports(method: Method) -> frozenset[str]:
    """The method's import, after its folder on sys.path when it has one (the
    script compiler writes sys.path lines before the imports)."""
    lines = [method.import_line] if method.import_line else []
    if method.search_path:
        lines.append(f"sys.path.append({method.search_path!r})")
    return frozenset(lines)


def _runner(
    family: str, plans: Mapping[str, _Plan], input_order: Sequence[int]
) -> Callable[..., VideoClip | VSFilter]:
    def run(context: NodeContext, *values: object) -> VideoClip | VSFilter:
        by_id = dict(zip(input_order, values))
        method_id = by_id.get(METHOD_ID)
        plan = plans.get(method_id) if isinstance(method_id, str) else None
        if plan is None:
            shown = method_id.value if isinstance(method_id, Dropped) else method_id
            raise ValueError(
                f"The method {shown!r} is not among the VapourSynth functions any more"
                " (vs-jetpack or a plugin changed); pick another one in Method."
            )
        if family == ANY_FUNCTION:
            return _any_statement(plan.method, by_id, context.node_id)
        run_ = _Run(f"{family} ({plan.method.label})", by_id)
        return _statement(plan, run_, context.node_id, by_id.get(FILTER_ID) is True)

    run.__name__ = f"{slug(family)}_node"
    return run


def _any_methods(methods: Sequence[Method]) -> list[Method]:
    """Any Function lists a group's members one by one (it has no class dropdown)."""
    flat: list[Method] = []
    for method in methods:
        if method.members:
            flat.extend(
                Method(
                    id=member.id,
                    family=method.family,
                    label=f"{method.label}: {member.choice}",
                    call=member.call,
                    kind=member.kind,
                    init_params=member.init_params,
                    params=member.params,
                    doc=member.doc,
                    arg_docs=member.arg_docs,
                    simple=member.simple,
                    import_line=member.import_line,
                    search_path=member.search_path,
                )
                for member in method.members
            )
        else:
            flat.append(method)
    return flat


def _keywords(methods: Sequence[Method]) -> list[str]:
    """What node search finds the node by: method labels and ids, group members."""
    words: list[str] = []
    for method in methods:
        words += [method.label, method.id, *(m.choice for m in method.members)]
    return list(dict.fromkeys(words))


def _suggestions(family: str, methods: Sequence[Method]) -> list[SpecialSuggestion]:
    """Typing a method's function name ("bm3d", "scale"), or the plain name an overlay
    gives it, in the node search offers the node with that method picked (the
    search matches a whole query, so generated labels like "VFM (vivtc)" are left
    out)."""
    suggestions: list[SpecialSuggestion] = []
    for method in methods:
        queries = [method.id.rpartition(".")[2]]
        if method.common and method.label.lower() != queries[0].lower():
            queries.append(method.label)
        suggestions.extend(
            SpecialSuggestion(
                query, name=f"{family}: {method.label}", inputs={METHOD_ID: method.id}
            )
            for query in queries
        )
    return suggestions


def register_families(
    methods: Sequence[Method],
    group: NodeGroup,
    enums: Enums | None = None,
    families: Sequence[str] = FAMILIES,
) -> None:
    """Registers in `group` the node of each of `families` that has methods. `enums`
    is the catalog's enums map (Catalog.enums)."""
    by_family: dict[str, list[Method]] = {}
    for method in methods:
        if method.id not in EXCLUDED:
            by_family.setdefault(method.family, []).append(method)
    for family in families:
        found = by_family.get(family, [])
        if family == ANY_FUNCTION:
            found = _any_methods(found)
        if found:
            _register(family, found, group, enums or {})


def _register(
    family: str, methods: Sequence[Method], group: NodeGroup, enums: Enums
) -> None:
    builder = _Family(family, methods, enums)
    if family == ANY_FUNCTION:
        builder.plans = {m.id: _Plan(m) for m in builder.methods}
    else:
        builder.build()
    inputs = builder.inputs()
    flat = _flat(inputs)
    output = VideoClipOutput(
        label="Clip or filter" if builder.switch else "Clip",
        output_type="if Input3 { VSFilter } else { VideoClip }"
        if builder.switch
        else "VideoClip",
    )
    run = _runner(family, builder.plans, [i.id for i in flat])
    # For chaiNNer's schema check (CHECK_LEVEL): the inputs' and the output's types.
    types = [i.associated_type for i in flat]
    run.__annotations__ = {
        "context": NodeContext,
        "values": functools.reduce(operator.or_, types)
        if None not in types
        else object,
        "return": output.associated_type,
    }
    icon, text = FAMILY_TEXT[family]
    group.register(
        schema_id(family),
        name=family,
        description=[text] if family == ANY_FUNCTION else [text, _METHOD_HELP],
        icon=icon,
        inputs=inputs,
        outputs=[output],
        node_context=True,
        keywords=_keywords(methods),
        suggestions=_suggestions(family, builder.methods),
        generated=True,
    )(run)
