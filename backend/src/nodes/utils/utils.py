# From https://github.com/victorca25/iNNfer/blob/main/utils/utils.py
from __future__ import annotations

import ctypes
import math
import os
import re
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Tuple

import numpy as np
from sanic.log import logger

Size = Tuple[int, int]
"""
The width and height (in that order) of an image.
"""

NUMBERS = re.compile(r"(\d+)")

ALPHABET = [*"ABCDEFGHIJKLMNOPQRSTUVWXYZ"]


def round_half_up(number: float | int) -> int:
    """
    Python's `round` method implements round-half-to-even rounding which is very unintuitive.
    This function implements round-half-up rounding.

    Round half up is consistent with JavaScript's `Math.round`.

    https://en.wikipedia.org/wiki/Rounding#Rounding_to_the_nearest_integer
    """
    return math.floor(number + 0.5)


def get_h_w_c(image: np.ndarray) -> tuple[int, int, int]:
    """Returns the height, width, and number of channels."""
    h, w = image.shape[:2]
    c = 1 if image.ndim == 2 else image.shape[2]
    return h, w, c


def alphanumeric_sort(value: str) -> list[str | int]:
    """Key function to sort strings containing numbers by proper
    numerical order."""

    lcase_value = value.upper()
    parts = NUMBERS.split(lcase_value)
    parts[1::2] = map(int, parts[1::2])
    return parts  # type: ignore


__SPLIT_SNAKE_CASE = re.compile(r"(\d+|_+)")
__SPLIT_PASCAL_CASE = re.compile(r"(\d+)|(?<=[a-z])(?=[A-Z])")


def split_snake_case(s: str) -> list[str]:
    """Splits a snake case identifier into its parts. E.g. `SNAKE_CASE` -> [`snake`, `case`]"""
    return [
        x.lower() for x in __SPLIT_SNAKE_CASE.split(s) if x and not x.startswith("_")
    ]


def split_pascal_case(s: str) -> list[str]:
    """Splits a snake case identifier into its parts. E.g. `SNAKE_CASE` -> [`snake`, `case`]"""
    return [
        x.lower() for x in __SPLIT_PASCAL_CASE.split(s) if x and not x.startswith("_")
    ]


def join_pascal_case(words: list[str]) -> str:
    return "".join([x.capitalize() for x in words])


__ABBREVIATIONS = {"rgb", "rgba"}


def smart_capitalize(word: str) -> str:
    if word in __ABBREVIATIONS:
        return word.upper()
    return word.capitalize()


def join_space_case(words: list[str]) -> str:
    return " ".join([smart_capitalize(x) for x in words])


def split_file_path(path: Path | str) -> tuple[Path, str, str]:
    """
    Returns the base directory, file name, and extension of the given file path.
    """
    base, ext = os.path.splitext(path)
    dirname, basename = os.path.split(base)
    return Path(dirname), basename, ext


class _FileBasicInfo(ctypes.Structure):
    _fields_ = [
        ("CreationTime", ctypes.c_int64),
        ("LastAccessTime", ctypes.c_int64),
        ("LastWriteTime", ctypes.c_int64),
        ("ChangeTime", ctypes.c_int64),
        ("FileAttributes", ctypes.c_uint32),
    ]


@cache
def _kernel32() -> ctypes.WinDLL:
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel32.GetFileInformationByHandleEx.restype = wintypes.BOOL
    kernel32.GetFileInformationByHandleEx.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    return kernel32


def _change_time(path: Path) -> int | None:
    """The time of the file's last data or metadata change (FILE_BASIC_INFO's
    ChangeTime) on Windows, where os.stat's st_ctime_ns is its creation time; None
    elsewhere, where st_ctime_ns is this time."""
    if sys.platform != "win32":
        return None
    kernel32 = _kernel32()
    # Reading attributes only: allowed whatever sharing the file's other users chose.
    handle = kernel32.CreateFileW(
        str(path),
        0x80,  # FILE_READ_ATTRIBUTES
        0x7,  # FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE
        None,
        3,  # OPEN_EXISTING
        0x02000000,  # FILE_FLAG_BACKUP_SEMANTICS, so directories open too
        None,
    )
    if handle in (None, ctypes.c_void_p(-1).value):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        info = _FileBasicInfo()
        basic_info = 0  # the FILE_INFO_BY_HANDLE_CLASS FileBasicInfo
        if not kernel32.GetFileInformationByHandleEx(
            handle, basic_info, ctypes.byref(info), ctypes.sizeof(info)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel32.CloseHandle(handle)
    return info.ChangeTime


def file_identity(path: Path) -> tuple:
    """A file's version, cheap to check: its absolute path, device, index, size,
    last-write time, creation (Windows) or change time and, on Windows, the change
    time, so a same-size rewrite in place whose last-write time was put back is a
    new version. A missing file is (absolute path, None), so it still reaches the
    loader, which reports it."""
    absolute = str(path.absolute())
    try:
        stat = path.stat()
        changed = _change_time(path)
    except FileNotFoundError:
        return (absolute, None)
    return (
        absolute,
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
        changed,
    )


def walk_error_handler(exception_instance: Exception):
    logger.warning(
        f"Exception occurred during walk: {exception_instance} Continuing..."
    )


def list_all_files_sorted(
    directory: Path, ext_filter: list[str] | None = None
) -> list[Path]:
    just_files: list[Path] = []
    for root, dirs, files in os.walk(
        directory, topdown=True, onerror=walk_error_handler
    ):
        dirs.sort(key=alphanumeric_sort)
        for name in sorted(files, key=alphanumeric_sort):
            filepath = os.path.join(root, name)
            _base, ext = os.path.splitext(filepath)
            if ext_filter is None or ext.lower() in ext_filter:
                just_files.append(Path(filepath))
    return just_files


@dataclass(frozen=True)
class Padding:
    top: int
    right: int
    bottom: int
    left: int

    @staticmethod
    def all(value: int) -> Padding:
        return Padding(value, value, value, value)

    @staticmethod
    def to(value: Padding | int) -> Padding:
        if isinstance(value, int):
            return Padding.all(value)
        return value

    @property
    def horizontal(self) -> int:
        return self.left + self.right

    @property
    def vertical(self) -> int:
        return self.top + self.bottom

    @property
    def empty(self) -> bool:
        return self.top == 0 and self.right == 0 and self.bottom == 0 and self.left == 0

    def scale(self, factor: int) -> Padding:
        return Padding(
            self.top * factor,
            self.right * factor,
            self.bottom * factor,
            self.left * factor,
        )

    def min(self, other: Padding | int) -> Padding:
        other = Padding.to(other)
        return Padding(
            min(self.top, other.top),
            min(self.right, other.right),
            min(self.bottom, other.bottom),
            min(self.left, other.left),
        )

    def remove_from(self, image: np.ndarray) -> np.ndarray:
        h, w, _ = get_h_w_c(image)

        return image[
            self.top : (h - self.bottom),
            self.left : (w - self.right),
            ...,
        ]


@dataclass(frozen=True)
class Region:
    x: int
    y: int
    width: int
    height: int

    @property
    def size(self) -> Size:
        return self.width, self.height

    def scale(self, factor: int) -> Region:
        return Region(
            self.x * factor,
            self.y * factor,
            self.width * factor,
            self.height * factor,
        )

    def intersect(self, other: Region) -> Region:
        x = max(self.x, other.x)
        y = max(self.y, other.y)
        width = min(self.x + self.width, other.x + other.width) - x
        height = min(self.y + self.height, other.y + other.height) - y
        return Region(x, y, width, height)

    def add_padding(self, pad: Padding) -> Region:
        return Region(
            x=self.x - pad.left,
            y=self.y - pad.top,
            width=self.width + pad.horizontal,
            height=self.height + pad.vertical,
        )

    def remove_padding(self, pad: Padding) -> Region:
        return self.add_padding(pad.scale(-1))

    def child_padding(self, child: Region) -> Padding:
        """
        Returns the padding `p` such that `child.add_padding(p) == self`.
        """
        left = child.x - self.x
        top = child.y - self.y
        right = self.width - child.width - left
        bottom = self.height - child.height - top
        return Padding(top, right, bottom, left)

    def read_from(self, image: np.ndarray) -> np.ndarray:
        h, w, _ = get_h_w_c(image)
        if (w, h) == self.size:
            return image

        return image[
            self.y : (self.y + self.height),
            self.x : (self.x + self.width),
            ...,
        ]

    def write_into(self, lhs: np.ndarray, rhs: np.ndarray):
        h, w, c = get_h_w_c(rhs)
        assert (w, h) == self.size
        assert c == get_h_w_c(lhs)[2]

        if c == 1:
            if lhs.ndim == 2 and rhs.ndim == 3:
                rhs = rhs[:, :, 0]
            if lhs.ndim == 3 and rhs.ndim == 2:
                rhs = np.expand_dims(rhs, axis=2)

        lhs[
            self.y : (self.y + self.height),
            self.x : (self.x + self.width),
            ...,
        ] = rhs
