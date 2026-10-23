"""Get Native Resolution's analysis: Vernier 0.2's core over the native library (native/src/getnative_ops.cpp)."""

from __future__ import annotations

import ctypes as ct
import json
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .native import lib

# The light domains as Vernier names them; the index is the native linear-plane model (0: none).
LIGHT_DOMAINS = (
    "encoded",
    "linear-bt709",
    "linear-srgb",
    "linear-gamma22",
    "linear-gamma24",
)
# The native renders (GetnativeText).
_REPORT, _PARAMETERS, _SUMMARY, _CURVES = range(4)


class _Options(ct.Structure):
    _fields_ = [
        (name, ct.c_int)
        for name in (
            "min_height",
            "max_height",
            "min_width",
            "max_width",
            "lines",
            "refinement_lines",
            "threads",
            "refine",
            "kernels",
        )
    ]


class _Frame(ct.Structure):
    _fields_ = [
        ("source", ct.c_char_p),
        ("sample", ct.c_char_p),
        ("provenance", ct.c_char_p),
        ("light_domain", ct.c_char_p),
        ("color_assumptions", ct.c_char_p),
        ("requested_seconds", ct.c_double),
        ("weights", ct.c_double * 3),
        ("keyframe_only", ct.c_int),
    ]


_error = [ct.c_char_p, ct.c_size_t]
_lib = lib()
_lib.cn_getnative_create.argtypes = [
    ct.POINTER(_Options),
    ct.POINTER(ct.c_void_p),
    *_error,
]
_lib.cn_getnative_free.argtypes = [ct.c_void_p]
_lib.cn_getnative_free.restype = None
_lib.cn_getnative_cancel.argtypes = [ct.c_void_p]
_lib.cn_getnative_cancel.restype = None
_lib.cn_getnative_add_frame.argtypes = [
    ct.c_void_p,
    ct.POINTER(ct.c_double),
    ct.c_int,
    ct.c_int,
    ct.POINTER(_Frame),
    *_error,
]
_lib.cn_getnative_render.argtypes = [
    ct.c_void_p,
    ct.c_int,
    ct.c_int,
    ct.c_int,
    ct.c_int,
    ct.c_int,
    ct.c_int,
    ct.c_char_p,
    ct.c_char_p,
    ct.POINTER(ct.c_size_t),
    *_error,
]
_lib.cn_getnative_copy.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_size_t, *_error]
_lib.cn_getnative_curve.argtypes = [
    ct.c_void_p,
    ct.c_int,
    ct.c_int,
    ct.POINTER(ct.c_int),
    ct.POINTER(ct.c_double),
    ct.c_size_t,
    ct.POINTER(ct.c_size_t),
    *_error,
]
_lib.cn_getnative_linear_plane.argtypes = [
    ct.POINTER(ct.c_float),
    ct.c_size_t,
    ct.c_int,
    ct.POINTER(ct.c_double),
    ct.POINTER(ct.c_double),
]
for _name in (
    "cn_getnative_create",
    "cn_getnative_add_frame",
    "cn_getnative_render",
    "cn_getnative_copy",
    "cn_getnative_curve",
    "cn_getnative_linear_plane",
):
    getattr(_lib, _name).restype = ct.c_int


def _utf8(text: str) -> bytes:
    """Text for the native session: UTF-8, with a lone surrogate (a file name's bytes that
    are not UTF-8, as Python decodes them) in UTF-8's form for its code point, which the
    session's JSON writes as a \\u escape."""
    return text.encode("utf-8", "surrogatepass")


def _call(function: Callable[..., int], *args: object) -> None:
    error = ct.create_string_buffer(4096)
    status = function(*args, error, len(error))
    if status:
        message = error.value.decode("utf-8", "replace")
        if status == 1:
            raise ValueError(message)
        if status == 3:
            raise MemoryError(message)
        raise RuntimeError(message)


@dataclass(frozen=True)
class SearchOptions:
    """Vernier's Options. kernels: 0 all 13 presets, 1 Bilinear, 2 Bicubic (Catmull-Rom with
    the free B/C refinement), 3 Lanczos 3. threads 0: half the logical processors, at most 16."""

    min_height: int
    max_height: int
    min_width: int = 0
    max_width: int = 0
    lines: int = 24
    refinement_lines: int = 128
    threads: int = 0
    refine: bool = True
    kernels: int = 0


@dataclass(frozen=True)
class FrameFacts:
    """What Vernier's Frame records about an analyzed plane."""

    source: str
    sample: str
    provenance: str
    light_domain: str = "encoded"
    color_assumptions: str = ""
    requested_seconds: float = 0.0
    weights: tuple[float, float, float] = (0.2126, 0.7152, 0.0722)
    keyframe_only: bool = False


class Session:
    """One analysis: planes added one by one (add_frame), then rendered as Vernier's JSON.
    cancel() may be called from another thread while add_frame runs; the analysis then stops
    with a RuntimeError at the core's next check."""

    def __init__(self, options: SearchOptions):
        handle = ct.c_void_p()
        values = _Options(
            options.min_height,
            options.max_height,
            options.min_width,
            options.max_width,
            options.lines,
            options.refinement_lines,
            options.threads,
            int(options.refine),
            options.kernels,
        )
        _call(_lib.cn_getnative_create, ct.byref(values), ct.byref(handle))
        self._handle: int | None = handle.value

    def __del__(self):
        self.close()

    def close(self) -> None:
        if self._handle is not None:
            _lib.cn_getnative_free(self._handle)
            self._handle = None

    def cancel(self) -> None:
        if self._handle is not None:
            _lib.cn_getnative_cancel(self._handle)

    def add_frame(self, plane: np.ndarray, facts: FrameFacts) -> None:
        """Analyzes `plane` (H x W) as `facts` describe it; its analysis id is the count before."""
        samples = np.ascontiguousarray(plane, np.float64)
        height, width = samples.shape
        frame = _Frame(
            _utf8(facts.source),
            _utf8(facts.sample),
            _utf8(facts.provenance),
            _utf8(facts.light_domain),
            _utf8(facts.color_assumptions),
            facts.requested_seconds,
            (ct.c_double * 3)(*facts.weights),
            int(facts.keyframe_only),
        )
        _call(
            _lib.cn_getnative_add_frame,
            self._handle,
            samples.ctypes.data_as(ct.POINTER(ct.c_double)),
            width,
            height,
            ct.byref(frame),
        )

    def _render(
        self,
        what: int,
        frame: int = -1,
        candidate: int = -1,
        planned: int = 0,
        completed: int = 0,
        domains: int = 1,
        sampling: str = "null",
        error: str = "",
    ) -> str:
        length = ct.c_size_t()
        _call(
            _lib.cn_getnative_render,
            self._handle,
            what,
            frame,
            candidate,
            planned,
            completed,
            domains,
            _utf8(sampling),
            _utf8(error),
            ct.byref(length),
        )
        text = ct.create_string_buffer(length.value + 1)
        _call(_lib.cn_getnative_copy, self._handle, text, len(text))
        return text.value.decode("utf-8")

    def report(self, frame: int) -> str:
        """Analysis `frame`'s full report (vernier.analysis.v2, scans included)."""
        return self._render(_REPORT, frame)

    def parameters(self, frame: int, candidate: int = -1) -> str:
        """vernier.parameters.v3 of analysis `frame`'s candidate (index; -1 the recommended one)."""
        return self._render(_PARAMETERS, frame, candidate)

    def summary(
        self,
        planned: int,
        completed: int,
        domains: int,
        sampling: object,
        error: str = "",
        candidate: int = -1,
    ) -> str:
        """Vernier's batch summary with a short record per analysis, the kernel decision and
        the selection (candidate: an index into the selected analysis's ranking, -1 automatic)."""
        return self._render(
            _SUMMARY,
            -1,
            candidate,
            planned,
            completed,
            domains,
            json.dumps(sampling),
            error,
        )

    def curves_csv(self, frame: int) -> str:
        return self._render(_CURVES, frame)

    def curve(self, frame: int, horizontal: bool) -> tuple[np.ndarray, np.ndarray]:
        """Analysis `frame`'s best score per native size of its width (horizontal) or height scan."""
        count = ct.c_size_t()
        _call(
            _lib.cn_getnative_curve,
            self._handle,
            frame,
            horizontal,
            None,
            None,
            0,
            ct.byref(count),
        )
        sizes = np.empty(count.value, np.int32)
        scores = np.empty(count.value, np.float64)
        _call(
            _lib.cn_getnative_curve,
            self._handle,
            frame,
            horizontal,
            sizes.ctypes.data_as(ct.POINTER(ct.c_int)),
            scores.ctypes.data_as(ct.POINTER(ct.c_double)),
            count.value,
            ct.byref(count),
        )
        return sizes, scores


def linear_plane(
    image: np.ndarray, domain: str, weights: tuple[float, float, float]
) -> np.ndarray:
    """The linear-light analysis plane of a B, G, R float image: the weighted R, G, B sum of the
    channels through `domain`'s inverse transfer, signed and unclipped (Vernier's linearFrame)."""
    source = np.ascontiguousarray(image, np.float32)
    height, width = source.shape[:2]
    plane = np.empty((height, width), np.float64)
    status = _lib.cn_getnative_linear_plane(
        source.ctypes.data_as(ct.POINTER(ct.c_float)),
        height * width,
        LIGHT_DOMAINS.index(domain),
        (ct.c_double * 3)(*weights),
        plane.ctypes.data_as(ct.POINTER(ct.c_double)),
    )
    if status:
        raise ValueError(f"Invalid linear-light plane request ({domain})")
    return plane
