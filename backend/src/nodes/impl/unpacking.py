"""CPython's own unpacking, for the native graph module's ports of `a, b = value`.

The native code calls these (graphpy::unpack), so CPython raises its own unpacking
errors, whose text changes between versions: 3.14 appends ", got N" to some.
"""

from __future__ import annotations

from collections.abc import Iterable


def unpack2(value: Iterable[object]) -> tuple[object, object]:
    a, b = value
    return a, b


def unpack3(value: Iterable[object]) -> tuple[object, object, object]:
    a, b, c = value
    return a, b, c
