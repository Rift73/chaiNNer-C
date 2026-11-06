"""pytest plugin of run_tests.py: holds to a shard's thread cap the libraries that read no
variable for it. The CHAINNER_C_SHARD_THREADS environment variable gives the cap.

ONNX Runtime sizes each session's pool to the machine's physical cores, whatever the
process's affinity, and its threads spin between runs: on a shard's two logical CPUs, a
session of a tiny model starts 15 threads and runs 12 times slower than with two that
do not spin. Tests that convert models make a session each. A session whose caller chose
its threads keeps them. OpenCV's pool follows the machine's CPUs too.

They are capped when a test is about to run, not when this plugin loads, as a shard's
order of imports stays as it is. The native image session (onnx_runtime.cpp) makes its
own pool, which this does not reach.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import pytest

_capped: set[str] = set()


def cap_onnx_runtime(threads: int) -> None:
    import onnxruntime as ort

    class CappedSession(ort.InferenceSession):
        def __init__(
            self,
            path_or_bytes: Any,
            sess_options: Any = None,
            *args: Any,
            **kwargs: Any,
        ) -> None:
            options = sess_options or ort.SessionOptions()
            if options.intra_op_num_threads == 0:
                options.intra_op_num_threads = threads
                options.inter_op_num_threads = 1
                options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            super().__init__(path_or_bytes, options, *args, **kwargs)

    ort.InferenceSession = CappedSession


def cap_opencv(threads: int) -> None:
    import cv2

    cv2.setNumThreads(threads)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup() -> None:
    threads = int(os.environ["CHAINNER_C_SHARD_THREADS"])
    for module, cap in (("onnxruntime", cap_onnx_runtime), ("cv2", cap_opencv)):
        if module in sys.modules and module not in _capped:
            _capped.add(module)
            cap(threads)
