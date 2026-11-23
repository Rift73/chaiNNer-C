"""The process's two GPU turns, with no dependency, so that the PyTorch package takes them
without TensorRT's.

A CUDA graph capture on one thread is ended by another thread's device-wide
synchronization (measured: VOSR's warm-up ending with torch.cuda.synchronize ended the
node's first capture, about one start in thirteen). Ordinary kernels, copies and
empty_cache from another thread were measured to be tolerated by a thread_local capture
(the model preview's warm-up held inside VOSR's first capture), and so was the
deserialization of VOSR's engines on its loader threads. TensorRT's other setup (making
and destroying execution contexts) is kept out of captures as well.

- gpu_setup_lock, the GPU setup turn: TensorRT's setup takes it, on every route, and so
  does every capture, so that the two take turns.
- capture_lock, the capture turn: every capture takes it first, then gpu_setup_lock, never
  the other way round. Work that may run beside a capture and synchronizes the device
  takes it alone, so that it waits only for captures and TensorRT's setup never waits for
  it: VOSR's warm-up (vosr2_warm_up) and a VOSR session's offload between images
  (vosr2_trt_pipeline.FullSession._release). The worker's check for a broken CUDA context
  (server.cuda_context_broken) takes it without waiting, and is skipped for that response
  while a capture holds it.

The captures that take both turns: inference.TensorRTSession._capture,
temporal_session.capture_graph, temporal_fix_eager.GraphedParts, vosr2_trt_execution's
stages, vosr2_replay.Replays, and the native runner's in this process (Upscale Image on
the runner, CHAINNER_C_TRT_RUNNER=1): it captures on its own timing, so
runner.RunnerSession.upscale holds both turns for the whole native call, giving them back
between tiles. The runner's engine setup takes the GPU setup turn. The VapourSynth plugin
of AI Upscale runs only in VapourSynth's child processes, with a CUDA context of their
own: nothing of this process can land inside its captures."""

from __future__ import annotations

from threading import Lock

gpu_setup_lock = Lock()
# Taken before gpu_setup_lock, never after it
capture_lock = Lock()

__all__ = ["capture_lock", "gpu_setup_lock"]
