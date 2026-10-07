"""Independent installed text oracles, including public helper protocols."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import random
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from nodes.impl import rust_regex
from nodes.impl.native_graph import graph
from nodes.impl.rust_regex import RustRegex
from nodes.utils import replacement

ROOT = Path(__file__).resolve().parents[2]
REF = Path(__file__).with_name("reference_utility_text")
NODE_DIR = "packages/chaiNNer_standard/utility/text"
NAMES = [
    "regex_find",
    "regex_replace",
    "text_padding",
    "text_pattern",
    "text_replace",
    "text_slice",
]


def load_helper(path):
    spec = importlib.util.spec_from_file_location("frozen_" + path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OLD_REPLACEMENT = load_helper(REF / "installed/nodes/utils/replacement.py")
OLD_REGEX = load_helper(REF / "installed/nodes/impl/rust_regex.py")


def load_node(path, current):
    parsed = ast.parse(path.read_text(encoding="utf-8"))
    body = []
    for node in parsed.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            node.decorator_list = []
            body.append(node)
        elif isinstance(node, ast.ImportFrom) and node.module == "__future__":
            body.append(node)
    namespace = {
        "Enum": Enum,
        "RustRegex": RustRegex,
        "ReplacementString": (
            replacement.ReplacementString
            if current
            else OLD_REPLACEMENT.ReplacementString
        ),
        "get_range_text": (
            rust_regex.get_range_text if current else OLD_REGEX.get_range_text
        ),
        "match_to_replacements_dict": (
            rust_regex.match_to_replacements_dict
            if current
            else OLD_REGEX.match_to_replacements_dict
        ),
        "graph": graph,
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


OLD = {
    name: load_node(REF / "installed" / NODE_DIR / f"{name}.py", False)
    for name in NAMES
}
NEW = {
    name: load_node(ROOT / "backend/src" / NODE_DIR / f"{name}.py", True)
    for name in NAMES
}


def outcome(function):
    try:
        value = function()
        return "value", type(value), value
    except Exception as error:
        return (
            "error",
            type(error),
            error.args,
            type(error.__cause__) if error.__cause__ else None,
            error.__cause__.args if error.__cause__ else None,
        )


def compare_regex_result(actual, expected, unordered_names):
    if not unordered_names:
        assert actual == expected
        return
    # Rust returns each groupindex from a newly randomized HashMap. Check the
    # full failure contract and every name. The fake mapping tests exact order.
    assert actual[:2] == expected[:2] == ("error", ValueError)
    assert actual[3:] == expected[3:]
    assert len(actual[2]) == len(expected[2]) == 1
    actual_prefix, actual_keys = actual[2][0].split("Available replacements: ")
    expected_prefix, expected_keys = expected[2][0].split("Available replacements: ")
    assert actual_prefix == expected_prefix
    assert actual_keys.endswith(".") and expected_keys.endswith(".")
    a = actual_keys[:-1].split(", ")
    b = expected_keys[:-1].split(", ")
    assert a[:3] == b[:3] == ["0", "1", "2"]
    assert len(a[3:]) == len(b[3:]) == 2
    assert set(a[3:]) == set(b[3:]) == {"a", "b"}


def token_state(instance):
    return (
        [
            token if isinstance(token, str) else (token.name,)
            for token in instance.tokens
        ],
        instance.names,
    )


def construct(cls, pattern):
    value = cls.__new__(cls)
    status = outcome(lambda: cls.__init__(value, pattern))
    return status, token_state(value), value


def compare_pattern(pattern, values):
    old_status, old_state, old = construct(OLD_REPLACEMENT.ReplacementString, pattern)
    status, state, new = construct(replacement.ReplacementString, pattern)
    assert status == old_status
    assert state == old_state
    if status[0] == "value":
        assert outcome(lambda: new.replace(values)) == outcome(
            lambda: old.replace(values)
        )


PATTERNS = [
    "",
    "plain🙂é\ud800\x00",
    "{1}",
    "{2}{1}{2}",
    "{missing}",
    "{}",
    "x{1}y{}tail",
    "{valid}{bad-name}",
    "{{",
    "}}",
    "{{x}}",
    "{{{1}}",
    "{a{b}}",
    "{unclosed",
    "a}b",
    "{{}}",
    "{_}",
    "{é}",
    "{²}",
    "{𐐀}",
    "{零}",
    "{e\u0301}",
    "{\ud800}",
    "{a\n}",
    "{\x00}",
    "{1}tail{2}",
]


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize(
    "values",
    [
        {},
        {"1": "", "2": "é🙂", "b": "nested"},
        {"missing": "value", "é": "Ω", "_": "under", "²": "2", "𐐀": "x"},
        {"2": "second", "1": "first", "valid": "good", "bad-name": "unused"},
    ],
)
def test_parser_and_interpolation(pattern, values):
    compare_pattern(pattern, values)


@pytest.mark.parametrize("seed", range(100))
def test_seeded_codepoint_parser(seed):
    rng = random.Random(seed)
    alphabet = "abc_19é零²\u0301🙂\ud800{}-\n\x00"
    for _ in range(12):
        pattern = "".join(rng.choice(alphabet) for _ in range(rng.randrange(80)))
        compare_pattern(pattern, {"a": "a🙂", "1": "one", "é": "unicode"})


@pytest.mark.parametrize(
    "codepoint",
    [*list(range(256)), 769, 8551, 8205, 55296, 65536, 66560, 128578, 1114111],
)
def test_parser_unicode_word_classification(codepoint):
    name = chr(codepoint)
    compare_pattern("{" + name + "}", {name: "ok"})


@pytest.mark.parametrize(
    "pattern", [None, 17, 1.5, b"a", bytearray(b"a"), memoryview(b"a"), [], {}]
)
def test_parser_nonstring_partial_state(pattern):
    compare_pattern(pattern, {})


@pytest.mark.parametrize("stride", [1, 2, -1])
def test_parser_buffer_acquisition_errors(stride):
    compare_pattern(memoryview(b"abcdef")[::stride], {})
    compare_pattern(np.arange(8, dtype=np.uint8)[::stride], {})


def test_parser_released_memoryview_and_buffer_release():
    view = memoryview(b"abc")
    view.release()
    compare_pattern(view, {})
    storage = bytearray(b"abc")
    compare_pattern(storage, {})
    storage.extend(b"def")  # A leaked Py_buffer would prevent resizing.
    assert storage == bytearray(b"abcdef")


def test_parser_input_type_name_truncation():
    long_named_type = type("X" * 250, (), {})
    compare_pattern(long_named_type(), {})


def test_public_token_mutation_and_mapping_order():
    for values in [{"z": "z", "a": "a"}, {3: "x"}, {"x": 1}, {"x": None}]:
        compare_pattern("before{x}", values)
    old = OLD_REPLACEMENT.ReplacementString("")
    new = replacement.ReplacementString("")
    old.tokens = ["start", OLD_REPLACEMENT.ReplacementInterpolation("x"), "end"]
    new.tokens = ["start", replacement.ReplacementInterpolation("x"), "end"]
    assert outcome(lambda: new.replace({"x": "ok"})) == outcome(
        lambda: old.replace({"x": "ok"})
    )
    old.tokens.append(object())
    new.tokens.append(object())  # pyright: ignore[reportArgumentType] -- deliberate: an invalid token pins the error path against upstream
    assert outcome(lambda: new.replace({"x": "ok"})) == outcome(
        lambda: old.replace({"x": "ok"})
    )


TEXTS = ["", "abcdef", "é🙂𐐀e\u0301", "\ud800\x00z"]


@pytest.mark.parametrize("text", TEXTS)
@pytest.mark.parametrize("width", [-1, 0, 1, 4, 5, 6, 11, 2**100, 1.2])
@pytest.mark.parametrize("padding", [" ", "0", "🙂", "\ud800", "", "ab"])
@pytest.mark.parametrize("mode", ["START", "END", "CENTER", "bad"])
def test_padding(text, width, padding, mode):
    def call(module):
        alignment = getattr(module.PaddingAlignment, mode, None)
        return module.text_padding_node(text, width, padding, alignment)

    assert outcome(lambda: call(NEW["text_padding"])) == outcome(
        lambda: call(OLD["text_padding"])
    )


@pytest.mark.parametrize("text", TEXTS)
@pytest.mark.parametrize(
    "old,new", [("", "X"), ("a", "aa"), ("🙂", "é"), ("none", ""), ("\ud800", "\x00")]
)
@pytest.mark.parametrize("mode", ["REPLACE_ALL", "REPLACE_FIRST", "bad"])
def test_text_replace(text, old, new, mode):
    def call(module):
        return module.text_replace_node(
            text, old, new, getattr(module.ReplacementMode, mode, None)
        )

    assert outcome(lambda: call(NEW["text_replace"])) == outcome(
        lambda: call(OLD["text_replace"])
    )


@pytest.mark.parametrize("text", TEXTS)
@pytest.mark.parametrize(
    "operation", ["START", "START_AND_LENGTH", "MAX_LENGTH", "bad"]
)
@pytest.mark.parametrize("alignment", ["START", "END", "bad"])
@pytest.mark.parametrize(
    "start,length,maximum",
    [
        (-50, 1, 1),
        (-3, 2, 2),
        (0, 0, 0),
        (1, 3, 3),
        (50, 100, 100),
        (-(2**100), 2**100, 2**100),
        (2, -3, -2),
    ],
)
def test_text_slice(text, operation, alignment, start, length, maximum):
    def call(module):
        return module.text_slice_node(
            text,
            getattr(module.SliceOperation, operation, None),
            start,
            length,
            maximum,
            getattr(module.SliceAlignment, alignment, None),
        )

    assert outcome(lambda: call(NEW["text_slice"])) == outcome(
        lambda: call(OLD["text_slice"])
    )


@pytest.mark.parametrize("pattern", [*PATTERNS, "{9}", "{10}"])
@pytest.mark.parametrize(
    "args",
    [
        (),
        (None,),
        ("one", None, "three"),
        tuple(str(i) for i in range(12)),
        ("",),
        (None, "two"),
        (1,),
    ],
)
def test_text_pattern_node(pattern, args):
    assert outcome(
        lambda: NEW["text_pattern"].text_pattern_node(pattern, *args)
    ) == outcome(lambda: OLD["text_pattern"].text_pattern_node(pattern, *args))


REGEX_CASES = [
    ("a12b34", r"\d+"),
    ("é🙂𐐀", r"\w+"),
    ("é🙂", ""),
    ("aaab", "(a+)(b)?"),
    ("a", "(a)(b)?"),
    ("word", "(?P<name>word)"),
    ("ab", "(?P<a>a)(?P<b>b)"),
    ("no match", "XYZ"),
    ("", "^$"),
    ("abc", "["),
    ("aaa", "(?=a)"),
    ("\ud800", "."),
    ("abc", "\ud800"),
]


@pytest.mark.parametrize("text,regex", REGEX_CASES)
@pytest.mark.parametrize("mode", ["FULL_MATCH", "PATTERN", "bad"])
@pytest.mark.parametrize("pattern", ["{0}", "{{{0}}", "{}", "{1}|{2}", "{name}"])
def test_regex_find(text, regex, mode, pattern):
    def call(module):
        return module.regex_find_node(
            text, regex, getattr(module.OutputMode, mode, None), pattern
        )

    compare_regex_result(
        outcome(lambda: call(NEW["regex_find"])),
        outcome(lambda: call(OLD["regex_find"])),
        "?P<a>" in regex and pattern == "{name}" and mode == "PATTERN",
    )


@pytest.mark.parametrize("text,regex", REGEX_CASES)
@pytest.mark.parametrize("mode", ["REPLACE_ALL", "REPLACE_FIRST", "bad"])
@pytest.mark.parametrize("pattern", ["", "{0}", "{{{0}}", "{}", "{1}|{2}", "{name}"])
def test_regex_replace(text, regex, mode, pattern):
    def call(module):
        return module.regex_replace_node(
            text, regex, pattern, getattr(module.ReplacementMode, mode, None)
        )

    compare_regex_result(
        outcome(lambda: call(NEW["regex_replace"])),
        outcome(lambda: call(OLD["regex_replace"])),
        "?P<a>" in regex and pattern == "{name}",
    )


def test_word_class_is_unicode():
    # chainner_ext's regex crate (and its C port) has Unicode classes: "é" is a word
    # character.
    match = RustRegex(r"\w").search("é")
    assert match is not None
    assert (match.start, match.end) == (0, 1)


# Expected values recorded from chainner_ext 0.3.10's RustRegex: character offsets,
# and no empty match directly after the previous match.
@pytest.mark.parametrize(
    "pattern,text,spans",
    [
        ("a*", "baaa", [(0, 0), (1, 4)]),
        ("x*", "axb", [(0, 0), (1, 2), (3, 3)]),
        (r"\b", " a b", [(1, 1), (2, 2), (3, 3), (4, 4)]),
        ("", "é🙂𐐀", [(0, 0), (1, 1), (2, 2), (3, 3)]),
        ("a", "é🙂𐐀a", [(3, 4)]),
        (r"\d+", "a12b34", [(1, 3), (4, 6)]),
        ("XYZ", "no match", []),
    ],
)
def test_rust_offsets_and_iteration(pattern, text, spans):
    regex = RustRegex(pattern)
    assert [(m.start, m.end) for m in regex.findall(text)] == spans
    first = regex.search(text)
    assert (first and (first.start, first.end)) == (spans[0] if spans else None)


def test_rust_captures_and_errors():
    regex = RustRegex("(?P<n>a)(b)?")
    assert (regex.groups, regex.groupindex) == (2, {"n": 1})
    match = regex.search("xxéab")
    assert match is not None
    groups = [match.get(i) for i in range(4)]
    assert [g and (g.start, g.end) for g in groups] == [(3, 5), (3, 4), (4, 5), None]
    assert [
        [g and (g.start, g.end) for g in map(m.get, range(3))]
        for m in RustRegex("(a)|(b)").findall("ab")
    ] == [[(0, 1), (0, 1), None], [(1, 2), None, (1, 2)]]
    with pytest.raises(OverflowError, match="can't convert negative int to unsigned"):
        match.get(-1)
    with pytest.raises(ValueError, match=r"^Invalid regex: "):
        RustRegex("(")
    with pytest.raises(UnicodeEncodeError):
        RustRegex("\ud800")


@pytest.mark.parametrize(
    "text,regex",
    [
        ("aaab", "(a+)(b)?"),
        ("ab", "(?P<a>a)(?P<b>b)"),
        ("xyé🙂", "(?P<n>x)|(?P<m>y)|(é)"),
        ("é🙂𐐀", r"\w+"),
    ],
)
def test_engine_capture_maps_match_frozen_helper(text, regex):
    # A RustRegex's groups and names come from the engine through chainner_ext's
    # capsule; the frozen helper reads groups and groupindex.
    compiled = RustRegex(regex)
    matches = compiled.findall(text)
    assert matches
    for match in matches:
        actual = rust_regex.match_to_replacements_dict(compiled, match, text)
        expected = OLD_REGEX.match_to_replacements_dict(compiled, match, text)
        assert list(actual.items()) == list(expected.items())


def test_custom_constructor_of_rust_regexes_replaces_as_builtin():
    mode_type = NEW["regex_replace"].ReplacementMode

    def replace(constructor, mode):
        return graph().utility_regex_replace(
            "aé🙂b12é",
            r"(\d)|(é)",
            "<{1}{2}>",
            mode,
            mode_type,
            constructor,
            replacement.ReplacementString,
        )

    def construct(pattern):
        return RustRegex(pattern)

    expected = {
        mode_type.REPLACE_ALL: "a<é>🙂b<1><2><é>",
        mode_type.REPLACE_FIRST: "a<é>🙂b12é",
    }
    for mode, result in expected.items():
        assert replace(RustRegex, mode) == result
        assert replace(construct, mode) == result


def test_fake_capture_maps_protocol_order():
    def run(function):
        events = []

        class Group:
            @property
            def start(self):
                events.append("start")
                return 1

            @property
            def end(self):
                events.append("end")
                return 3

        class Regex:
            @property
            def groups(self):
                events.append("groups")
                return 2

            @property
            def groupindex(self):
                events.append("groupindex")
                return {"last": 2, "first": 1}

        class Match:
            def get(self, index):
                events.append(("get", index))
                return None if index == 2 else Group()

        result = function(Regex(), Match(), "aé🙂z")
        return result, list(result), events

    assert run(rust_regex.match_to_replacements_dict) == run(
        OLD_REGEX.match_to_replacements_dict
    )


@pytest.mark.parametrize("groups", [-2, 1.5, 0])
def test_fake_capture_errors_and_partial_order(groups):
    regex = SimpleNamespace(groups=groups, groupindex={"x": 0})
    match = SimpleNamespace(get=lambda index: None)
    assert outcome(
        lambda: rust_regex.match_to_replacements_dict(regex, match, "")  # pyright: ignore[reportArgumentType] -- deliberate: SimpleNamespace fakes with invalid group data pin the error path against upstream
    ) == outcome(lambda: OLD_REGEX.match_to_replacements_dict(regex, match, ""))


@pytest.mark.parametrize("pairs", [[()], [("a",)], [("a", 0, "extra")], [3]])
def test_capture_mapping_unpack_failure(pairs):
    regex = SimpleNamespace(groups=0, groupindex=SimpleNamespace(items=lambda: pairs))
    match = SimpleNamespace(get=lambda index: None)
    assert outcome(
        lambda: rust_regex.match_to_replacements_dict(regex, match, "")  # pyright: ignore[reportArgumentType] -- deliberate: SimpleNamespace fakes with invalid group data pin the error path against upstream
    ) == outcome(lambda: OLD_REGEX.match_to_replacements_dict(regex, match, ""))


def test_parser_owned_partial_state_after_constructor_failure():
    def run(module):
        events = []
        original = module.ReplacementInterpolation

        def fail(name):
            events.append(name)
            if name == "fail":
                raise LookupError("constructor stopped")
            return original(name)

        module.ReplacementInterpolation = fail
        try:
            status, state, _ = construct(module.ReplacementString, "a{ok}b{fail}c")
            return status, state, events
        finally:
            module.ReplacementInterpolation = original

    assert run(replacement) == run(OLD_REPLACEMENT)


def test_replacement_property_mapping_access_order():
    def run(module):
        events = []

        class Token:
            @property
            def name(self):
                events.append("name")
                return "missing"

        class Mapping(dict):
            def __contains__(self, key):
                events.append(("contains", key))
                return False

            def keys(self):  # pyright: ignore[reportIncompatibleMethodOverride] -- deliberate contract violation: keys() returns a list to pin iteration order
                events.append("keys")
                return ["second", "first"]

        value = module.ReplacementString("")
        value.tokens = ["prefix", Token()]
        result = outcome(lambda: value.replace(Mapping()))
        return result, events

    assert run(replacement) == run(OLD_REPLACEMENT)


@pytest.mark.parametrize("first", [False, True])
def test_regex_application_evaluation_order(first):
    def run(current):
        module = load_node(
            (ROOT / "backend/src" if current else REF / "installed")
            / NODE_DIR
            / "regex_replace.py",
            current,
        )
        events = []

        class FakeRegex:
            groups = 0

            def __init__(self, pattern):
                self.groupindex = {}
                events.append(("regex", pattern))

            def findall(self, text):
                events.append(("findall", text))
                return [SimpleNamespace(start=1, end=2, get=lambda index: None)]

        class FakeReplacement:
            def __init__(self, pattern):
                events.append(("replacement", pattern))

            def replace(self, mapping):
                events.append(("replace", mapping.copy()))
                return "changed"

        function = module.regex_replace_node
        function.__globals__["RustRegex"] = FakeRegex
        function.__globals__["ReplacementString"] = FakeReplacement
        mode = (
            module.ReplacementMode.REPLACE_FIRST
            if first
            else module.ReplacementMode.REPLACE_ALL
        )
        return function("abc", "rx", "rp", mode), events

    assert run(True) == run(False)


def test_errors_keep_handled_exception_context():
    def run(module):
        try:
            raise LookupError("outer handled context")
        except LookupError:
            try:
                module.ReplacementString("{}")
            except ValueError as error:
                assert error.__context__ is not None
                return (
                    type(error.__context__),
                    error.__context__.args,
                    error.__suppress_context__,
                )
        raise AssertionError("expected invalid pattern")

    assert run(replacement) == run(OLD_REPLACEMENT)


@pytest.mark.parametrize("count", [1, 2, 128, 4096])
def test_many_escape_and_interpolation_fragments(count):
    compare_pattern("prefix" + "{{é🙂" * count + "suffix", {})
    compare_pattern("{1}:{2}" * count, {"1": "é🙂", "2": "tail"})
    text = "aé🙂" * count
    for name in ["REPLACE_ALL", "REPLACE_FIRST"]:
        old = OLD["regex_replace"]
        new = NEW["regex_replace"]
        assert new.regex_replace_node(
            text, "a", "({0})", getattr(new.ReplacementMode, name)
        ) == old.regex_replace_node(
            text, "a", "({0})", getattr(old.ReplacementMode, name)
        )


@pytest.mark.parametrize("custom_result", ["plain", "subclass", "object"])
def test_fragment_assembly_preserves_custom_add_protocol(custom_result):
    def run(module):
        events = []

        class Accumulator:
            def __init__(self, value):
                self.value = value

            def __iadd__(self, other):
                events.append(("iadd", str(other)))
                self.value += str(other)
                return self

        class Custom(str):
            def __radd__(self, other):
                events.append(("radd", other, str(self)))
                value = other + str(self)
                if custom_result == "object":
                    return Accumulator(value)
                if custom_result == "subclass":
                    return Custom(value)
                return value

        value = module.ReplacementString("ab{1}cd{2}ef")
        result = value.replace({"1": Custom("X"), "2": "Y"})
        return (
            type(result).__name__,
            result.value if isinstance(result, Accumulator) else str(result),
            events,
        )

    assert run(replacement) == run(OLD_REPLACEMENT)


def test_fragment_empty_and_single_string_identity():
    token = "unique-fragment-" + str(object())
    for module in [OLD_REPLACEMENT, replacement]:
        value = module.ReplacementString("")
        for tokens in [[token], ["", token], [token, ""], ["", token, ""]]:
            value.tokens = tokens
            assert value.replace({}) is token
        parsed = module.ReplacementString(token)
        assert parsed.tokens[0] is token


@pytest.mark.parametrize("fail", [False, True])
def test_public_token_isinstance_class_proxy(fail):
    def run(module):
        events = []

        class Token:
            @property
            def __class__(self):  # pyright: ignore[reportIncompatibleMethodOverride] -- deliberate contract violation: a getter-only __class__ for isinstance's class protocol
                events.append("class")
                if fail:
                    raise LookupError("class failed")
                return str

            def __radd__(self, other):
                events.append(("radd", other))
                return other + "proxy"

        value = module.ReplacementString("")
        value.tokens = ["prefix", Token(), "suffix"]
        return outcome(lambda: value.replace({})), events

    assert run(replacement) == run(OLD_REPLACEMENT)


def test_string_subclass_methods_and_literal_slices():
    def run(current):
        events = []

        class Text(str):
            def __getitem__(self, key):
                events.append(("getitem", key.start, key.stop, key.step))
                return super().__getitem__(key)

            def center(self, *args):
                events.append(("center", args))
                return "overridden"

            def replace(self, *args, **kwargs):
                events.append(("replace", args, kwargs))
                return "overridden"

        node = NEW if current else OLD
        parser = replacement if current else OLD_REPLACEMENT
        a = parser.ReplacementString(Text("before{1}after{{tail")).replace({"1": "x"})
        b = node["text_padding"].text_padding_node(
            Text("x"), 3, "-", node["text_padding"].PaddingAlignment.CENTER
        )
        c = node["text_replace"].text_replace_node(
            Text("x"), "x", "y", node["text_replace"].ReplacementMode.REPLACE_FIRST
        )
        return a, b, c, events

    assert run(True) == run(False)


def test_no_match_and_noop_identity():
    text = "unique-text-" + str(object())
    for nodes in [OLD, NEW]:
        assert (
            nodes["regex_replace"].regex_replace_node(
                text, "nomatch", "", nodes["regex_replace"].ReplacementMode.REPLACE_ALL
            )
            is text
        )
        assert (
            nodes["text_replace"].text_replace_node(
                text, "nomatch", "", nodes["text_replace"].ReplacementMode.REPLACE_ALL
            )
            is text
        )
        assert (
            nodes["text_padding"].text_padding_node(
                text, 0, " ", nodes["text_padding"].PaddingAlignment.START
            )
            is text
        )


def test_thread_local_parser_and_node_state():
    def work(index):
        for _ in range(15):
            text = f"é{index}🙂"
            pattern = replacement.ReplacementString("{{{x}:{x}")
            assert pattern.replace({"x": text}) == "{" + text + ":" + text
            module = NEW["regex_replace"]
            assert (
                module.regex_replace_node(
                    text, r"\d+", "({0})", module.ReplacementMode.REPLACE_ALL
                )
                == f"é({index})🙂"
            )
        return index

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(work, range(32))) == list(range(32))


def test_frozen_snapshot_hashes():
    manifest = json.loads((REF / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["files"]) == 16
    for entry in manifest["files"]:
        assert (
            hashlib.sha256(
                (REF / entry["variant"] / entry["path"]).read_bytes()
            ).hexdigest()
            == entry["sha256"]
        )


@pytest.mark.parametrize("name", NAMES)
def test_node_metadata_and_signatures_unchanged(name):
    def signature(path):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
        function.body = [ast.Pass()]
        return ast.dump(function)

    assert signature(ROOT / "backend/src" / NODE_DIR / f"{name}.py") == signature(
        REF / "installed" / NODE_DIR / f"{name}.py"
    )
