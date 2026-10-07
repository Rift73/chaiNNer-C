from __future__ import annotations

from typing import Protocol

from chainner_ext import RegexMatch, RustRegex
from nodes.impl.native_graph import graph


class Range(Protocol):
    @property
    def start(self) -> int: ...
    @property
    def end(self) -> int: ...


def get_range_text(text: str, range: Range) -> str:
    return graph().utility_range_text(text, range)


def match_to_replacements_dict(
    regex: RustRegex, match: RegexMatch, text: str
) -> dict[str, str]:
    return graph().utility_capture_map(regex, match, text)
