"""Bounded preparation reuse and first-match execution, with real chainner_ext matching."""

from concurrent.futures import ThreadPoolExecutor
from enum import Enum

import pytest

from nodes.impl import text_cache
from nodes.impl.native_graph import graph
from nodes.impl.rust_regex import RustRegex
from nodes.utils.replacement import ReplacementString


class Mode(Enum):
    REPLACE_FIRST = 0
    REPLACE_ALL = 1


@pytest.fixture(autouse=True)
def empty_caches():
    text_cache._regex.cache_clear()
    text_cache._replacement.cache_clear()
    yield
    text_cache._regex.cache_clear()
    text_cache._replacement.cache_clear()


def replace(text, pattern, replacement, mode=Mode.REPLACE_FIRST):
    return graph().utility_regex_replace(
        text, pattern, replacement, mode, Mode, RustRegex, ReplacementString
    )


@pytest.mark.parametrize(
    "text,pattern,replacement",
    [
        ("one two three", r"\w+", "new"),
        ("aaaa", "a", "b"),
        ("abc", "", "-"),
        ("αβγ🙂", r".", "{0}"),
        ("abc42 abc99", r"(?P<word>\w+?)(\d+)", "{word}/{2}"),
        ("no match", r"[0-9]+", "{0}"),
    ],
)
def test_first_capture_matches_old_findall_semantics(text, pattern, replacement):
    regex = RustRegex(pattern)
    matches = regex.findall(text)
    if matches:
        from nodes.impl.rust_regex import match_to_replacements_dict

        m = matches[0]
        # Match offsets are Unicode character indices in the installed binding.
        expected = (
            text[: m.start]
            + ReplacementString(replacement).replace(
                match_to_replacements_dict(regex, m, text)
            )
            + text[m.end :]
        )
    else:
        expected = text
    assert replace(text, pattern, replacement) == expected


def test_first_mode_calls_search_without_materializing_all_matches(monkeypatch):
    counts = {"construct": 0, "search": 0, "findall": 0}

    class Engine:
        def __init__(self, pattern):
            counts["construct"] += 1
            self.inner = RustRegex(pattern)

        def search(self, text):
            counts["search"] += 1
            return self.inner.search(text)

        def findall(self, text):
            counts["findall"] += 1
            return self.inner.findall(text)

        @property
        def groups(self):
            return self.inner.groups

        @property
        def groupindex(self):
            return self.inner.groupindex

    monkeypatch.setattr(text_cache, "RustRegex", Engine)
    for _ in range(2):
        out = graph().utility_regex_replace(
            "a" * 10000, "a", "b", Mode.REPLACE_FIRST, Mode, Engine, ReplacementString
        )
        assert out == "b" + "a" * 9999
    assert counts == {"construct": 1, "search": 2, "findall": 0}


def test_normal_preparation_reused_and_public_parsers_independent():
    assert replace("aaa", "a", "x") == "xaa"
    assert replace("aaaa", "a", "x") == "xaaa"
    assert text_cache._regex.cache_info().hits == 1
    assert text_cache._replacement.cache_info().hits == 1
    public = ReplacementString("x")
    public.tokens[:] = ["corrupt"]
    assert replace("aa", "a", "x") == "xa"


def test_cache_bounds_and_long_pattern_bypass():
    for i in range(150):
        text_cache.compile_regex(str(i), RustRegex)
        text_cache.compile_replacement(str(i), ReplacementString)
    assert text_cache._regex.cache_info().currsize == 128
    assert text_cache._replacement.cache_info().currsize == 128
    before = (text_cache._regex.cache_info(), text_cache._replacement.cache_info())
    for _ in range(2):
        text_cache.compile_regex("a" * 4097, RustRegex)
        text_cache.compile_replacement("a" * 4097, ReplacementString)
    assert (
        text_cache._regex.cache_info(),
        text_cache._replacement.cache_info(),
    ) == before


def test_custom_constructor_and_string_protocols_not_cached():
    events = []

    class Constructor:
        def __init__(self, pattern):
            events.append(pattern)

    for _ in range(2):
        text_cache.compile_regex("x", Constructor)
        text_cache.compile_replacement("x", Constructor)
    assert events == ["x"] * 4

    class Text(str):
        pass

    for _ in range(2):
        text_cache.compile_regex(Text("a"), RustRegex)
        text_cache.compile_replacement(Text("a"), ReplacementString)
    assert text_cache._regex.cache_info().currsize == 0
    assert text_cache._replacement.cache_info().currsize == 0


@pytest.mark.parametrize("pattern", ["(", "[", "*"])
def test_invalid_patterns_not_cached_and_errors_unchanged(pattern):
    results = []
    for call in (
        lambda: RustRegex(pattern),
        lambda: text_cache.compile_regex(pattern, RustRegex),
        lambda: text_cache.compile_regex(pattern, RustRegex),
    ):
        with pytest.raises(ValueError) as caught:
            call()
        results.append(caught.value.args)
    assert results[0] == results[1] == results[2]
    assert text_cache._regex.cache_info().currsize == 0


def test_concurrent_reuse_keeps_per_match_state_isolated():
    def run(i):
        return replace(f"a{i} a{i + 1}", r"a(\d+)", "{1}")

    with ThreadPoolExecutor(8) as pool:
        values = list(pool.map(run, range(64)))
    assert values == [f"{i} a{i + 1}" for i in range(64)]
