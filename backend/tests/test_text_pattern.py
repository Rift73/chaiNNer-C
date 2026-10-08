"""Text Pattern's description states the grammar the node parses (upstream chaiNNer
#2106): {1} to {9} insert the inputs, {{ writes a literal {, and Python's format
specs, conversions and }} escape are not part of it."""

from __future__ import annotations

import pytest

from packages.chaiNNer_standard.utility.text.text_pattern import text_pattern_node


def run(pattern: str, *args: str) -> str:
    return text_pattern_node(pattern, *args, *[None] * (9 - len(args)))


def test_described_grammar():
    assert run("{1} and {2}", "a", "b") == "a and b"
    assert run("{9}", *"123456789") == "9"
    assert run("{{1}") == "{1}"
    # Only "{" is escaped; "}}" stays as written.
    assert run("{{1}}") == "{1}}"
    assert run("a}}b") == "a}}b"


@pytest.mark.parametrize("pattern", ["{1:x}", "{1!r}", "{:x}", "{0}", "{}"])
def test_python_format_syntax_is_rejected(pattern: str):
    with pytest.raises(ValueError):
        run(pattern, "255")
