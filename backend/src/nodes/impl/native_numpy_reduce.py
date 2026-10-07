"""NumPy 2.5.3's float32 full-reduction iteration, for the native mirrors.

A full reduction (``np.sum``/``np.mean``/``np.min``/``np.max`` with
``axis=None``) hands its inner loop the lengths that
``npyiter_find_buffering_setup`` (NumPy 2.5.3
``_core/src/multiarray/nditer_constr.c``) picks; each call continues from the
running result. A buffered operand is cut into blocks of at most the buffer
size that never cross an index of the axes beyond the chosen outer dimension
("Never buffer beyond the first outer dimension", :1950, example :1958-1964):
each such run of ``period`` elements restarts its blocks. Verified bit-exact
against np.sum on float32 (1, n) for n = 8191..65539, and on (300, 301) C, F,
transposed and reversed (whole), unaligned (8192), [::2, ::2] and [:, ::2]
(8154), [1:-1, 1:-1] (8073, also unaligned) and its F-order copy (8046); rows
wider than 4096 of a (3, n) view (one row per call); (200, 200, c) views
[::2, ::2, :] (8100) and [:, :, :3] (8190); a channel plane [:, :, 0] (whole);
and with restarts, (5, 4000, 4)[:, :, :3][:, 10:3990] (8190 within 11940) and
(3, 9000, 4)[::-1, :, :2] (8192 within 18000).
"""

from __future__ import annotations

import numpy as np

# NPY_BUFSIZE, the default maximum buffer length.
_BUFSIZE = 8192


def numpy_sum_block(array: np.ndarray) -> tuple[int, int]:
    """``(block, period)`` for ``cn_numpy_sum_f32`` and ``cn_numpy_extreme_f32``:
    the inner-loop calls cut every run of ``period`` elements (0 = the whole
    array) into ``block``-element calls (0 = the whole run), the last shorter.

    ``array`` is the operand upstream reduces, in its own layout.
    """
    block, period, _ = _reduce_setup(array)
    return block, period


def numpy_reduce_contiguous(array: np.ndarray) -> bool:
    """Whether the reduce inner loop reads a contiguous float32 stream (a buffer
    or a 4-byte stride), which selects loops_minmax's SIMD path over its strided
    scalar one. ``array`` is the operand upstream reduces, in its own layout.
    """
    return _reduce_setup(array)[2]


def _reduce_setup(array: np.ndarray) -> tuple[int, int, bool]:
    # Unaligned or non-float32 input is a CAST operand: always buffered.
    cast = not array.flags.aligned or array.dtype != np.dtype(np.float32)
    # The iterator drops length-1 axes, stably orders the rest (innermost
    # first) by |stride| and coalesces axes that continue one signed stride:
    # reductions iterate with NPY_ITER_DONT_NEGATE_STRIDES (umath/reduction.c).
    axes: list[list[int]] = []
    for stride, length in sorted(
        (
            (stride, length)
            for length, stride in zip(
                reversed(array.shape), reversed(array.strides), strict=True
            )
            if length != 1
        ),
        key=lambda axis: abs(axis[0]),
    ):
        if axes and axes[-1][0] * axes[-1][1] == stride:
            axes[-1][1] *= length
        else:
            axes.append([stride, length])
    if not axes:
        return 0, 0, True
    # The core-size search for the reduce output (all strides 0, never
    # buffered) and this one input operand. The reduction iterator grows its
    # inner loop (GROWINNER), so only buffering limits it to _BUFSIZE.
    cost = 2 if cast else 1
    size = axes[0][1]
    best_cost, best_size, best_core, best_dim = cost, size, 1, 0
    for dim in range(1, len(axes)):
        if size >= _BUFSIZE and cost > 1:
            break
        if dim == 1 and not cast:
            cost += 1  # a coalesced axis boundary: the input must be buffered
        core = size
        size *= axes[dim][1]
        limit = _BUFSIZE if size > _BUFSIZE and cost > 1 else size
        if cost * best_size <= best_cost * limit:
            best_cost, best_size, best_core, best_dim = cost, size, core, dim
    # A call never crosses an index of the axes beyond best_dim: the elements of
    # axes[0..best_dim] form one run, whose blocks restart at every outer index.
    period = best_size
    buffered = cast or best_dim > 0
    block = period
    if buffered and block > _BUFSIZE:
        block = best_core * (_BUFSIZE // best_core)
    return (
        0 if block >= period else block,
        0 if period >= array.size else period,
        buffered or axes[0][0] == array.itemsize,
    )
