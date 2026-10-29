"""Native utility text entry points (native/src/run/nodes/utility_text.cpp) against their Python bodies.

A str holding a lone surrogate and an int beyond int64 cross as Python objects and decline to the Python body, as
do Parse Number's errors and non-ASCII text; each test derives which cases decline and asserts both paths occur."""

import itertools
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import api
import process
from native_differential import assert_native_matches, node_data
from nodes.impl.native_graph import graph
from packages.chaiNNer_standard.utility.text.regex_replace import ReplacementMode

# Empty, ASCII, every UTF-8 length, U+0000, repeats for overlapping matches, lone surrogates; numbers (the input
# stringifies them).
TEXTS = ["", "a", "abc", "h\u00e9llo", "\u4e16\u754c", "\U0001f600x\U0001f600", "a\x00b", "aaa", "abab", "\ud800",
         "x\udc80y", 3.0, 2.5]
INT64 = range(-(2**63), 2**63)


def surrogate(value: object) -> bool:
    return isinstance(value, str) and any(0xD800 <= ord(c) <= 0xDFFF for c in value)


def big(value: object) -> bool:
    return isinstance(value, int) and value not in INT64


def check(schema_id: str, inputs: list[object], declines: Callable[[list[object]], bool], counts: dict[bool, int]):
    """Asserts the native outcome on inputs, a decline where declines(enforced inputs) says so."""
    enforced = process.enforce_inputs(list(inputs), node_data(schema_id), api.NodeId("text-test"), [])
    declined = declines(enforced)
    counts[declined] += 1
    assert_native_matches(schema_id, inputs, declines=declined)


def any_surrogate(enforced: list[object]) -> bool:
    return any(surrogate(value) for value in enforced)


@pytest.fixture
def counts():
    """Native and declined case counts; every test that uses it runs both paths."""
    seen = {False: 0, True: 0}
    yield seen
    assert seen[False] > 0 and seen[True] > 0, seen


@pytest.mark.parametrize("text", [None, "", "# Title\n\n*note*", "\ud800", 4.0])
@pytest.mark.parametrize("markdown", [False, True])
def test_note_matches(text, markdown):
    assert assert_native_matches("chainner:utility:note", [text, markdown]) == ("outputs", [])


def test_text_append_matches(counts):
    for separator in ["", "-", " | ", "\U0001f600", "abcdef", "\ud800"]:
        for parts in [["a", "b"], ["", ""], ["\u00e9", "\U0001f600", None, "c", None, None, None, None, None, "j"],
                      ["x", "y", "z", "\udfff"], ["x", "y", *(["w"] * 8)], [3.0, 2.5]]:
            inputs: list[object] = [separator, *parts, *([None] * (10 - len(parts)))]
            check("chainner:utility:text_append", inputs, any_surrogate, counts)


def test_text_length_matches(counts):
    for text in TEXTS:
        check("chainner:utility:text_length", [text], any_surrogate, counts)


@pytest.mark.parametrize("alignment", ["start", "end", "center"])
def test_text_padding_matches(alignment, counts):
    for text, width, fill in itertools.product(TEXTS, range(13), ["0", " ", "\u00e9", "\U0001f600", "ab", "\ud800"]):
        check("chainner:utility:text_padding", [text, width, fill, alignment], any_surrogate, counts)
    for width in [2**26 + 1, 2**64]:
        check("chainner:utility:text_padding", ["abc", width, "0", alignment], lambda _: True, counts)


@pytest.mark.parametrize("mode", [0, 1])
def test_text_replace_matches(mode, counts):
    olds = ["", "a", "b", "ab", "\u00e9", "\U0001f600", "aa", "zz", "\ud800"]
    for text, old, new in itertools.product(TEXTS, olds, ["", "-", "XYZ", "\U0001f600"]):
        check("chainner:utility:text_replace", [text, old, new, mode], any_surrogate, counts)


STARTS = [-100, -3, -1, 0, 1, 2, 5, 100, -(2**63), 2**63 - 1, 2**70]
LENGTHS = [0, 1, 2, 3, 100, 2**63 - 1, 2**70]


def test_text_slice_matches(counts):
    # The operation reads only its own inputs: a big int elsewhere does not decline. Enforce rounds through a
    # float (round_half_up), so 2**63 - 1 arrives as the big int 2**63.
    def reads(*indices: int) -> Callable[[list[object]], bool]:
        return lambda enforced: surrogate(enforced[0]) or any(big(enforced[i]) for i in indices)

    for text in TEXTS:
        for start in STARTS:
            check("chainner:utility:text_slice", [text, 0, start, 2**70, 2**70, "end"], reads(2), counts)
            for length in LENGTHS:
                check("chainner:utility:text_slice", [text, 1, start, length, 2**70, "start"], reads(2, 3), counts)
        for maximum in LENGTHS:
            for alignment in ["start", "end"]:
                check("chainner:utility:text_slice", [text, 2, 2**70, 2**70, maximum, alignment], reads(4), counts)


def test_text_slice_unknown_operation_is_none():
    assert graph().run_native("chainner:utility:text_slice", ["abc", 7, 0, 0, 0, "start"]) == [None]
    assert graph().run_native("chainner:utility:text_slice", ["abc", 2, 0, 0, 1, "middle"]) == [None]


PATTERNS = ["", "abc", "{1}", "{1}{2}", "{{1}}", "{{", "{", "}", "{1", "a{1}b{2}c{3}", "{9}", "{0}", "{}", "{a-b}",
            "{\u00e9}", "{\u00e9-}", "{name}", "{{{1}}}", "x{1}{1}y", "\U0001f600{1}\U0001f600", "{1}{10}", "{_1}",
            "{ 1}", "{1}{}", "{\u00e9}{}", "\ud800{1}"]
ARGUMENTS = [["one", "two", "three", None, None, None, None, None, "nine"], [None] * 9, ["", "\u00e9", *([None] * 7)],
             ["\udfff", *([None] * 8)], [None, "\ud800", *([None] * 7)]]


def test_text_pattern_matches(counts):
    # A non-ASCII name that no invalid ASCII character fails is decided by CPython's Unicode tables: the scan
    # declines there ("{\u00e9}" comes before any other error in these patterns).
    for pattern, arguments in itertools.product(PATTERNS, ARGUMENTS):
        check("chainner:utility:text_pattern", [pattern, *arguments],
              lambda enforced: any_surrogate(enforced) or "{\u00e9}" in str(enforced[0]), counts)


@pytest.mark.parametrize(
    "pattern, message",
    [
        ("{}", 'Invalid replacement pattern. {} is not a valid replacement. Either specify a name or id number, or '
               'escape a single "{" as "{{". Full pattern: {}'),
        ("a{b c}", "Invalid replacement pattern. {b c} is not a valid replacement. Names and ids only allow letters "
                   "and digits. Full pattern: a{b c}"),
        ("{1}{4}", "Unknown replacement. There is no replacement with the name or id 4. Available replacements: 1, 3."),
    ],
)
def test_text_pattern_errors(pattern, message):
    inputs: list[object] = [pattern, "x", None, "z", *([None] * 6)]
    assert assert_native_matches("chainner:utility:text_pattern", inputs) == ("error", ("ValueError", message))


def directory_declines(enforced: list[object]) -> bool:
    """A lone surrogate, which UTF-8 cannot hold; on Linux, whose path is bytes, any name that is not ASCII."""
    text = str(enforced[0])
    return surrogate(text) if sys.platform == "win32" else not text.isascii()


def test_directory_to_text_matches(counts):
    for directory in ["C:/a/b", "C:\\a\\.\\b\\", "relative\\x", "\\\\server\\share\\x", "C:", ".", "\u00e9\u4e16/x",
                      "D:/\U0001f600/y", "C:/\udc80bad", "C:/\ud83d", Path("D:/x/y")]:
        check("chainner:utility:directory_to_text", [directory], directory_declines, counts)


PARSE_TEXTS = ["0", "-0", "+5", "  42  ", "\t-17\n", "\x0b\x0c\r7", "1_000", "1__0", "_1", "1_", "0x1f", "0X_1F",
               "0x", "0x_", "0x__1", "0b101", "0B1_0", "0o17", "0o", "ff", "FF", "zz", "Zz", "9223372036854775807",
               "-9223372036854775808", "9223372036854775808", "-9223372036854775809", "7fffffffffffffff",
               "8000000000000000", "", " ", "-", "abc", "\u0661\u0662\u0663", "\u00a012", "0" * 641 + "1",
               "0" * 639 + "1", "1 2", "- 1", "\x1c1", "12\x7f", "0x10", "1\x002", "007", "+-1", "١"]


def parse_declines(enforced: list[object]) -> bool:
    """Parse Number runs natively exactly when int(text, base) succeeds within int64 on ASCII text of at most 640
    characters."""
    text, base = enforced
    assert isinstance(text, str) and isinstance(base, int)
    try:
        result = int(text, base)
    except ValueError:
        return True
    return big(result) or len(text) > 640 or any(ord(c) >= 0x7F for c in text)


def test_parse_number_matches(counts):
    for text, base in itertools.product(PARSE_TEXTS, [2, 8, 10, 16, 36, 10.4, 1, 40]):
        check("chainner:utility:parse_number", [text, base], parse_declines, counts)


REGEX_TEXTS = ["", "My name is Jane.", "My two cats caught 32 mice in 14 days", "aaa", "abab", "h\u00e9llo w\u00f6rld",
               "\U0001f600 x \U0001f600", "a\x00b", "user@host other@x", "line1\nline2"]
REGEXES = [r"\d+", r"(\w+) is (\w+)", r"(?P<name>\w+)@(?P<dom>\w+)", "", "a*", "(a)|(b)", r"\b", "x", "\u00e9+",
           "\U0001f600", "^", "$", r"(?m)^\w", "(?i)ABAB", r"\pL+", ".", "(?P<first>a)?b", r"\u{1F600}", "(",
           "a{2,1}", "[", "(?P<n>a)(?P<n>b)", r"\p{Foo}", r"(?-u:\xFF)", "a{1000000}"]
OUTPUT_PATTERNS = ["{0}", "Found {0}", "{1}={2}", "{name}@{dom}", "{3}", "{}", "{{0}}", "{first}", "", "{2}{1}{0}"]


@pytest.mark.parametrize("mode", [0, 1])
def test_regex_find_matches(mode):
    for text, regex, output in itertools.product(REGEX_TEXTS, REGEXES, OUTPUT_PATTERNS if mode else ["{0}"]):
        assert_native_matches("chainner:utility:regex_find", [text, regex, mode, output])


@pytest.mark.parametrize("mode", [0, 1])
def test_regex_replace_matches(mode):
    for text, regex, replacement in itertools.product(REGEX_TEXTS, REGEXES, OUTPUT_PATTERNS):
        assert_native_matches("chainner:utility:regex_replace", [text, regex, replacement, mode])


@pytest.mark.parametrize(
    "inputs, declines",
    [
        (["\ud800 32", r"\d+", 0, "{0}"], True),
        (["32", "\\d\ud800", 0, "{0}"], True),
        (["32", r"\d+", 1, "{0}\udfff"], True),
        (["32", r"\d+", 0, "{0}\udfff"], False),  # Full Match never reads the output pattern
        (["32", r"\d+", 1, "{\u00e9}"], True),  # a non-ASCII name: CPython's Unicode tables decide
        (["ab", "x", 1, "{\u00e9}"], False),  # no match: the output pattern is never parsed
    ],
)
def test_regex_find_declines(inputs, declines):
    assert_native_matches("chainner:utility:regex_find", inputs, declines=declines)


@pytest.mark.parametrize(
    "inputs, declines",
    [
        (["\ud800 32", r"\d+", "<{0}>", 0], True),
        (["32", "\\d\ud800", "<{0}>", 0], True),
        (["32", r"\d+", "<{0}>\udfff", 0], True),
        (["32", r"\d+", "{\u00e9}", 1], True),
        (["32", "(", "{\u00e9}", 1], False),  # the regex fails to compile first
    ],
)
def test_regex_replace_declines(inputs, declines):
    assert_native_matches("chainner:utility:regex_replace", inputs, declines=declines)


def test_regex_errors():
    assert assert_native_matches("chainner:utility:regex_find", ["abc", r"\d", 0, "{0}"]) == (
        "error",
        ("RuntimeError", "No match found. Unable to find the pattern '\\d' in the text."),
    )
    outcome = assert_native_matches("chainner:utility:regex_replace", ["abc", "(", "x", 0])
    assert outcome == ("error", ("ValueError", "Invalid regex: regex parse error:\n    (\n    ^\nerror: unclosed group"))


def test_regex_cache_keeps_results_across_many_patterns():
    # More distinct patterns than the cache holds, twice over: evicted and cached patterns give the same results.
    for _ in range(2):
        for n in range(200):
            assert_native_matches("chainner:utility:regex_replace", [f"a{n}b{n}", f"{n}", "<{0}>", 0])


def test_regex_nodes_run_concurrently():
    # run_native releases the GIL: eight threads search, compile and evict the shared cache at once.
    def work(thread: int) -> list[object]:
        return [
            graph().run_native("chainner:utility:regex_replace", [f"x{n}y{thread}", f"{n}|{thread}", "<{0}>", 0])[0]
            for n in range(300)
        ]

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(work, range(8)))
    for thread, outputs in enumerate(results):
        for n, output in enumerate(outputs):
            expected = node_data("chainner:utility:regex_replace").run(f"x{n}y{thread}", f"{n}|{thread}", "<{0}>",
                                                                       ReplacementMode.REPLACE_ALL)
            assert output == expected
