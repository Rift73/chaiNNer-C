"""The one TensorRT logger of this process. TensorRT keeps the first logger that a
builder, runtime or refitter registers while any of them exists and ignores later ones
(it warns so), so every TensorRT object in the worker uses this one. Warnings and errors
go to the backend's log, and a build collects those logged while it runs (capture)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from threading import Lock
from typing import Any

import tensorrt as trt
from sanic.log import logger


class TensorRTLogger(trt.ILogger):
    def __init__(self) -> None:
        trt.ILogger.__init__(self)
        self._lock = Lock()
        self._captures: list[list[tuple[int, str]]] = []

    def log(self, severity: Any, msg: str) -> None:
        level = int(severity)
        warning = int(trt.ILogger.Severity.WARNING)
        if level > warning:
            return
        if level == warning:
            logger.warning("[TRT] %s", msg)
        else:
            logger.error("[TRT] %s", msg)
        with self._lock:
            for messages in self._captures:
                messages.append((level, msg))

    @contextmanager
    def capture(self) -> Iterator[list[tuple[int, str]]]:
        """The (severity, message) pairs logged while the block runs, from any thread."""
        messages: list[tuple[int, str]] = []
        with self._lock:
            self._captures.append(messages)
        try:
            yield messages
        finally:
            with self._lock:
                self._captures.remove(messages)


TRT_LOGGER = TensorRTLogger()
