"""Cached BF16 model stages, one CUDA stream and bounded global VAE attention."""

from __future__ import annotations

import atexit
from collections.abc import Callable, Hashable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from weakref import WeakSet

import torch

from api import NodeContext, NodeId
from gpu import get_vram_usage
from nodes.impl.tensorrt.cache import SharedCache

from .vosr2_export import rotary_tables
from .vosr2_memory import available_memory
from .vosr2_plan_file import compressed, plain
from .vosr2_trt_execution import BoundedQueue, Stage, StageExecution
from .vosr2_trt_models import STAGES
from .vosr2_trt_plugin import load_plugin

LANE_BYTES = 512 * 1024**2
# A lane's pair lane: its windows' second predictions, which read its keys
PAIRED = 8

_LIVE: WeakSet[FullSession] = WeakSet()


class FullSession:
    """Resident model stages; live scratch follows actual input dimensions. `plan`
    places a stage's plan in `directory`: compressed in a cached set (the default), plain
    in a build's (build_session)."""

    def __init__(
        self,
        directory: Path,
        device: torch.device,
        plan: Callable[[Path, str], Path] = compressed,
    ):
        from nodes.impl.tensorrt.trt_logger import TRT_LOGGER

        load_plugin()
        with torch.cuda.device(device):
            self.device = torch.device("cuda", torch.cuda.current_device())
            self.stream = torch.cuda.Stream(device=self.device)
        # The worker's one TensorRT logger (trt_logger.py).
        self.logger = TRT_LOGGER
        self.stages: dict[str, Stage] = {}
        self.ropes: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.budget = 0
        # The activation reserve the weights were admitted with (None: all resident)
        self.reserve: int | None = None
        self.execution = StageExecution(self.device, self.stream)
        self.geometry: tuple[int, int, int] | None = None
        self.constants: dict[object, torch.Tensor] = {}
        self.kv_tensor: torch.Tensor | None = None
        # What lanes() allows for the current geometry, and its images before this one
        self.lane_count: int | None = None
        self.images = 0

        def load(name: str) -> Stage:
            with torch.cuda.device(self.device):
                return Stage(plan(directory, name), self.logger, self.device)

        try:
            with ThreadPoolExecutor(max_workers=len(STAGES)) as pool:
                loads = [pool.submit(load, name) for name in STAGES]
                for name, stage in zip(STAGES, loads, strict=True):
                    self.stages[name] = stage.result()
        except BaseException:
            self.close()
            raise
        _LIVE.add(self)

    def prepare(self, reserve: int | None = None) -> None:
        """Before admission, drop warm weights admitted with another activation reserve
        (another memory mode: a Low budget would count them as another program's), or
        when the device is nearly full."""
        if self.budget and reserve != self.reserve:
            self.offload()
            return
        usage = get_vram_usage(self.device.index)
        if usage is None:
            raise RuntimeError("VOSR sessions require working GPU memory monitoring.")
        if usage.used >= usage.total - 1_200_000_000:
            self.offload()

    def activate(self, check: Callable[[], None], reserve: int | None = None) -> None:
        """Admit the stages with every weight on the device, or, given a `reserve` (Low
        memory), with as much of each stage's weights as leaves `reserve` bytes of the
        budget for activations; TensorRT streams the rest from host memory as it runs."""
        import tensorrt as trt

        from nodes.impl.tensorrt.memory import gpu_setup_lock

        check()
        if self.budget and reserve == self.reserve:
            return
        if self.budget:
            # Admitted for another memory mode
            self.offload()
        engines = [stage.engine for stage in self.stages.values()]
        if any(engine is None for engine in engines):
            raise RuntimeError("VOSR BF16 session is closed.")
        streamable = sum(
            engine.streamable_weights_size for engine in engines if engine is not None
        )
        total = torch.cuda.get_device_properties(self.device).total_memory
        previous = torch.cuda.get_per_process_memory_fraction(self.device)
        budget = streamable
        if reserve is not None:
            free = int(total * previous) - torch.cuda.memory_allocated(self.device)
            budget = max(0, min(streamable, free - reserve))
        cap = int(total * previous) - budget
        if cap <= torch.cuda.memory_allocated(self.device) + 512 * 1024**2:
            raise RuntimeError("Insufficient memory to admit VOSR BF16 model stages.")
        # The enclosing vram_budget restores the fraction. Warm weights are
        # already measured as external allocations on its next admission.
        torch.cuda.set_per_process_memory_fraction(cap / total, self.device)
        try:
            admitted = []
            for stage in self.stages.values():
                engine = stage.engine
                if engine is None:
                    raise RuntimeError("Closed VOSR stage.")
                size = engine.streamable_weights_size
                if size:
                    # Each stage keeps the same share of its weights
                    resident = (
                        size if budget == streamable else size * budget // streamable
                    )
                    engine.weight_streaming_budget_v2 = resident
                    if engine.weight_streaming_budget_v2 != resident:
                        raise RuntimeError(
                            f"TensorRT rejected VOSR {stage.name} weight residency."
                        )
                admitted.append((stage, engine))
            check()

            def make_context(stage: Stage, engine: trt.ICudaEngine) -> None:
                with torch.cuda.device(self.device):
                    stage.context = engine.create_execution_context(
                        trt.ExecutionContextAllocationStrategy.USER_MANAGED
                    )

            # Each context copies its stage's weights to the device: they go together.
            with gpu_setup_lock, ThreadPoolExecutor(len(admitted)) as pool:
                list(pool.map(lambda pair: make_context(*pair), admitted))
            for stage in self.stages.values():
                if stage.context is None:
                    raise RuntimeError(f"VOSR could not allocate {stage.name} context.")
            self.budget = budget
            self.reserve = reserve
        except BaseException:
            self.offload()
            raise

    def begin_image(self, height: int, width: int, seed: int) -> None:
        if not (512 <= height <= 2176 and 512 <= width <= 3840):
            raise ValueError(
                "BF16 TensorRT supports padded outputs up to 2176 high by 3840 wide."
            )
        geometry = (height, width, seed)
        if self.geometry is None or self.geometry[:2] != geometry[:2]:
            self.execution.clear()
            self.constants.clear()
            self.ropes.clear()
            self.kv_tensor = None
            self.lane_count = None
            self.images = 0
        else:
            self.images += 1
        if self.geometry is not None and self.geometry[2] != seed:
            # Seed changes affect noise, not engine shapes, addresses or graphs.
            self.constants.pop("noise", None)
        self.geometry = geometry

    def begin_window(self) -> None:
        # Conditioning is constant only across the local/context pair in this window.
        self.kv_tensor = None

    def constant(self, key: object, value: torch.Tensor) -> torch.Tensor:
        found = self.constants.get(key)
        if found is None:
            found = value.to(self.device)
            self.constants[key] = found
        return found

    def stage(self, name: str, *arguments: torch.Tensor) -> torch.Tensor:
        return self.execution.run(self.stages[name], arguments)

    def condition(self, image: torch.Tensor) -> torch.Tensor:
        return self.stage("dino", image)

    def predict(self, latent: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        cos, sin = self.rope(latent.shape[-1])
        if self.kv_tensor is None:
            self.kv_tensor = self.stage("dit_kv", condition)
        return self.stage("dit_cached", latent, self.kv_tensor, cos, sin)

    def concurrent(self) -> bool:
        """Whether streams may run side by side: every weight is resident (Default
        memory)."""
        return self.reserve is None

    def lanes(self) -> int:
        """How many windows run side by side: two from a geometry's eighth image when
        streams may and the free memory allows, else one. A second lane costs a geometry
        about 0.5 s once (its contexts and graphs) and saves about 0.06 s an image beside
        the pair lanes; a third saved nothing (RTX 5090, FP8, 640x360 at 4x)."""
        if not self.concurrent() or self.images < 7:
            return 1
        if self.lane_count is None:
            # A lane's keys, scratch, conditioning and inputs: about 470 MiB, and one
            # lane's worth stays free.
            self.lane_count = int(available_memory(self.device) // LANE_BYTES - 1)
        return max(1, min(2, self.lane_count))

    def windows(
        self, work: list[tuple[torch.Tensor, list[torch.Tensor]]]
    ) -> list[tuple[torch.Tensor, list[torch.Tensor]]]:
        """Each window's DINO conditioning and DiT predictions, given its normalized DINO
        input and latents, for up to lanes() windows. They run side by side, each on a
        lane of its own (stream, scratch, graphs), and a window's second prediction
        beside its first, on its lane's pair lane, which reads the lane's keys: one
        prediction leaves much of the GPU idle (RTX 5090, a window's two: FP8 26.8 ->
        20.6 ms, BF16 39.8 -> 30.6 ms). The same kernels, so the same bits. Streamed
        weights (Low) take everything in turn. Outputs are borrowed until the next
        call."""
        if not self.concurrent():
            ((normalized, latents),) = work
            self.begin_window()
            condition = self.condition(normalized)
            return [
                (condition, [self.predict(latent, condition) for latent in latents])
            ]
        lanes = {}
        for lane, (_, latents) in enumerate(work):
            lanes[lane] = self.execution.lane_stream(lane)
            if len(latents) == 2:
                lanes[PAIRED + lane] = self.execution.lane_stream(PAIRED + lane)
        # The rotary tables are made here, on the caller's stream, before the lanes read
        # them.
        for _, latents in work:
            for latent in latents:
                self.rope(latent.shape[-1])
        current = torch.cuda.current_stream(self.device)
        for stream in lanes.values():
            if stream != current:
                stream.wait_stream(current)
        try:
            results = self._run_lanes(work, lanes)
        finally:
            # Also when a launch failed: the caller's stream follows every lane.
            for stream in lanes.values():
                if stream != current:
                    current.wait_stream(stream)
        for condition, predictions in results:
            for value in (condition, *predictions):
                value.record_stream(current)
        return results

    def _run_lanes(
        self,
        work: list[tuple[torch.Tensor, list[torch.Tensor]]],
        lanes: dict[int, torch.cuda.Stream],
    ) -> list[tuple[torch.Tensor, list[torch.Tensor]]]:
        results = []
        for lane, (normalized, latents) in enumerate(work):
            stream = lanes[lane]
            with torch.cuda.stream(stream):
                condition = self.execution.launch(
                    self.stages["dino"], (normalized,), stream, lane
                )
                kv = self.execution.launch(
                    self.stages["dit_kv"], (condition,), stream, lane
                )
                keys = torch.cuda.Event()
                keys.record(stream)
                predictions = [self._predict_on(latents[0], kv, stream, lane)]
            if len(latents) == 2:
                pair = lanes[PAIRED + lane]
                pair.wait_event(keys)
                with torch.cuda.stream(pair):
                    predictions.append(
                        self._predict_on(latents[1], kv, pair, PAIRED + lane)
                    )
            results.append((condition, predictions))
        return results

    def _predict_on(
        self,
        latent: torch.Tensor,
        kv: torch.Tensor,
        stream: torch.cuda.Stream,
        lane: int,
    ) -> torch.Tensor:
        cos, sin = self.rope(latent.shape[-1])
        return self.execution.launch(
            self.stages["dit_cached"], (latent, kv, cos, sin), stream, lane
        )

    def rope(self, width: int) -> tuple[torch.Tensor, torch.Tensor]:
        if width not in self.ropes:
            self.ropes[width] = rotary_tables(width, self.device, torch.bfloat16)
        return self.ropes[width]

    def vae(
        self, name: str, image: torch.Tensor, check: Callable[[], None]
    ) -> torch.Tensor:
        check()
        self.execution.attention_epoch += 1
        feature = self.stage(name + "_pre", image)
        qkv = self.stage(name + "_qkv", feature)
        q, k, v = (part.unsqueeze(1).contiguous() for part in qkv.unbind(1))
        k = k.transpose(-1, -2).contiguous()
        # A key count that is a multiple of 8 has an engine that reads K and V in place.
        attention = "attention_grouped" if k.shape[-1] % 8 == 0 else "attention"
        if attention == "attention_grouped":
            k, v = k.unflatten(-1, (-1, 8)), v.unflatten(2, (-1, 8))
        result = self.execution.buffer(
            (name, "attention_result"), tuple(q.shape), q.dtype
        )
        queue = BoundedQueue()
        for start in range(0, q.shape[2], 1024):
            check()
            end = min(start + 1024, q.shape[2])
            result[:, :, start:end] = self.stage(attention, q[:, :, start:end], k, v)
            queue.advance(check)
        attention = result.squeeze(1).transpose(1, 2).reshape_as(feature).contiguous()
        del qkv, q, k, v, result
        check()
        return self.stage(name + "_post", feature, attention).float()

    def offload(self) -> None:
        self.execution.clear()
        self.constants.clear()
        self.kv_tensor = None
        self.geometry = None
        for stage in self.stages.values():
            stage.offload()
        self.ropes.clear()
        self.lane_count = None
        self.images = 0
        torch.cuda.synchronize(self.device)
        self.budget = 0

    def close(self) -> None:
        self.offload()
        for stage in self.stages.values():
            stage.close()
        self.stages.clear()
        _LIVE.discard(self)


def build_session(directory: Path, device: torch.device) -> FullSession:
    """A build's session over the plans it just compiled, to check them: plain, since the
    cache compresses a set only once its build passed."""
    return FullSession(directory, device, plain)


@atexit.register
def close_sessions() -> None:
    for session in list(_LIVE):
        session.close()


_SESSIONS: SharedCache[FullSession] = SharedCache(FullSession.close)


def release_full_session(node_id: NodeId) -> None:
    _SESSIONS.release_node(node_id)


@contextmanager
def lease_session(
    context: NodeContext,
    key: Hashable,
    create: Callable[[], FullSession],
    *,
    reuse: bool,
) -> Iterator[FullSession]:
    """The node's session: with `reuse`, kept across runs under `key` (one per node,
    whichever backend made it); else made for this use and closed after it."""
    if not reuse:
        release_full_session(context.node_id)
        session = create()
        try:
            yield session
        finally:
            session.close()
        return
    with _SESSIONS.use(key, context.node_id, create) as session:
        yield session


def lease_full_session(
    context: NodeContext, directory: Path, device: torch.device, *, reuse: bool
) -> AbstractContextManager[FullSession]:
    key = (
        tuple(
            (
                str(compressed(directory, name)),
                compressed(directory, name).stat().st_size,
                compressed(directory, name).stat().st_mtime_ns,
            )
            for name in STAGES
        ),
        str(device),
    )
    return lease_session(
        context, key, lambda: FullSession(directory, device), reuse=reuse
    )
