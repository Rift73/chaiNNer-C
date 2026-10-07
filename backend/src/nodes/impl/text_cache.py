"""Bounded preparation caches for the built-in, immutable text-node inputs.

Custom constructors and string subclasses retain their observable protocols.
Cached replacement parsers are private to this module and never mutated.
"""

from functools import lru_cache
from typing import Callable

from nodes.impl.rust_regex import RustRegex
from nodes.utils.replacement import ReplacementString

_MAX_PATTERN_LENGTH = 4096


@lru_cache(maxsize=128)
def _regex(pattern: str) -> RustRegex:
    return RustRegex(pattern)


@lru_cache(maxsize=128)
def _replacement(pattern: str) -> ReplacementString:
    return ReplacementString(pattern)


def compile_regex(pattern: str, constructor: Callable[[str], object]) -> object:
    if (
        constructor is RustRegex
        and type(pattern) is str
        and len(pattern) <= _MAX_PATTERN_LENGTH
    ):
        return _regex(pattern)
    return constructor(pattern)


def compile_replacement(pattern: str, constructor: Callable[[str], object]) -> object:
    if (
        constructor is ReplacementString
        and type(pattern) is str
        and len(pattern) <= _MAX_PATTERN_LENGTH
    ):
        return _replacement(pattern)
    return constructor(pattern)
