"""The process's GPU setup turn, with no dependency, so that the PyTorch package takes it
without TensorRT's: TensorRT synchronizes the GPU's streams when it deserializes an engine
or makes or destroys an execution context, which invalidates a CUDA graph capture on any
thread; these and the captures take turns, across engines and routes."""

from __future__ import annotations

from threading import Lock

gpu_setup_lock = Lock()

__all__ = ["gpu_setup_lock"]
