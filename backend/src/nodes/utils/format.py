from __future__ import annotations

from typing import Callable, Iterable, Literal, TypeVar

T = TypeVar("T")
Conj = Literal["and", "or"]


def join_english(
    items: Iterable[T],
    to_str: Callable[[T], str] = str,
    conj: Conj = "and",
) -> str:
    s = list(map(to_str, items))

    l = len(s)
    assert l > 0

    if l == 1:
        return s[0]
    if l == 2:
        return f"{s[0]} {conj} {s[1]}"
    return ", ".join(s[:-1]) + f", {conj} " + s[-1]


def _format_with_channels(
    noun: str, article: str, channels: list[int], conj: Conj, plural: bool
) -> str:
    assert len(channels) > 0

    named = {1: "grayscale", 3: "RGB", 4: "RGBA"}
    if all(x in named for x in channels):
        names = join_english(channels, lambda c: named[c], conj=conj)
        return f"{names} {noun}s" if plural else f"a {names} {noun}"

    numbers = join_english(channels, conj=conj)
    if plural:
        return f"{noun}s with {numbers} channel(s)"
    return f"{article} {noun} with {numbers} channel(s)"


def format_image_with_channels(
    channels: list[int],
    conj: Conj = "and",
    plural: bool = False,
) -> str:
    return _format_with_channels("image", "an", channels, conj, plural)


def format_color_with_channels(
    channels: list[int],
    conj: Conj = "and",
    plural: bool = False,
) -> str:
    return _format_with_channels("color", "a", channels, conj, plural)


_CHANNEL_NUMBER_NAME = {1: "GRAY", 3: "RGB", 4: "RGBA"}


def format_channel_numbers(input_channels: int, output_channels: int) -> str:
    i = _CHANNEL_NUMBER_NAME.get(input_channels, str(input_channels))
    o = _CHANNEL_NUMBER_NAME.get(output_channels, str(output_channels))
    return f"{i}🠚{o}"
