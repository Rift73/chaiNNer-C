"""Which SIMD path NumPy dispatches on this CPU, for the C kernels that mirror it.

Every fact is read from NumPy's own dispatch at run time, never from CPU features
alone. The mapping comes from NumPy v2.5.3's sources:

- meson_cpu/x86/meson.build: the x86-64 targets are X86_V2 (SSE to SSE4.2, the
  baseline), X86_V3 (adds AVX, AVX2, FMA3, F16C) and X86_V4 (adds AVX-512
  F/CD/VL/BW/DQ). MSVC builds, which include the Windows wheels, disable X86_V4
  ("Considered broken by Highway on MSVC"), so `__cpu_dispatch__` is ['X86_V3'] there.
- meson_cpu/main_config.h.in, NPY_CPU_DISPATCH_CALL: a loop runs the first of its own
  source's targets whose features the CPU has, else the baseline. Each source lists
  its own targets in numpy/_core/meson.build (loops_arithm_fp: X86_V3, X86_V2;
  loops_trigonometric: X86_V4, X86_V3), so a mirror asks for its loop: loop_target(),
  which numpy.lib.introspect.opt_func_info reports exactly.
- numpy/_core/src/common/simd/{sse,avx2,avx512}: NPY_SIMD_WIDTH is 16, 32 and 64
  bytes, and Highway's vectors in the same targets match. NPY_SIMD_FMA3 is native
  under X86_V3 and X86_V4 only.
"""

from __future__ import annotations

from functools import lru_cache

from numpy.lib.introspect import opt_func_info

# Vector bytes per x86-64 target (simd/*/: NPY_SIMD_WIDTH), lowest target first.
_VECTOR_BYTES = {"X86_V2": 16, "X86_V3": 32, "X86_V4": 64}


def _known(target: str) -> str:
    if target not in _VECTOR_BYTES:
        raise RuntimeError(f"NumPy dispatches {target}, which no C mirror models")
    return target


def _highest(targets: list[str]) -> str:
    return max((_known(t) for t in targets), key=list(_VECTOR_BYTES).index)


@lru_cache(maxsize=None)
def loop_target(function: str, signature: str) -> str:
    """The target NumPy runs the ufunc loop on, e.g. ("sin", "ff") or
    ("maximum", "fff"): opt_func_info's "current", with "baseline(X86_V2)" read as
    the baseline's highest target. KeyError for a loop NumPy does not dispatch."""
    current = opt_func_info(func_name=f"^{function}$")[function][signature]["current"]
    if current.startswith("baseline(") and current.endswith(")"):
        return _highest(current[len("baseline(") : -1].split())
    return _known(current)


def float32_lanes(target: str) -> int:
    """float32 lanes of one vector under the target: 4, 8 or 16."""
    return _VECTOR_BYTES[_known(target)] // 4


def fma3(target: str) -> bool:
    """Whether the target has native FMA3 (NPY_SIMD_FMA3): X86_V3 and X86_V4."""
    return _known(target) != "X86_V2"
