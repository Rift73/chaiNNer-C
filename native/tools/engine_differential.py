"""The native run engine's whole-chain differential (native/local/instant-exec/DESIGN.md section 2, amendment A6).

A chain runs in this process through process.Executor (the oracle) and through the native engine
(native_executor.NativeExecutor) on the same inputs, each executor as server.py's /run drives it (an error becomes
the execution-error event, after the run's own events). The two runs must agree:
- the request's outcome: no error, or the same error type and message;
- the output files, by relative path and bytes; a lossless video (FFV1 in Matroska) by its decoded frames (muxers
  write run-dependent bytes), a lossy one (MP4, WebM, MOV, AVI) by its frame count and size only: the pinned FFmpeg's
  libx264 output varies from run to run on either executor (measured), so the frames such a writer receives are
  pinned by a chain that saves them as PNGs instead;
- the SSE events, as verify_runtime.sse_event_mismatches judges them (per node: broadcasts a sub-multiset of the
  oracle's, the final merged state equal, a generator started once, every other start, finish and error event
  exact; times, ETAs and traces left out). The oracle runs with every broadcast sent (its LatestBroadcasts would
  drop a node's pending preview under load, timing-dependently), so the engine's coalesced broadcasts are judged
  against all of them;
and the engine's events must keep the order invariants of amendment A6 (order_violations): chain-start first, the
execution error last, each node's start before its finish, a node fed only through lazy inputs running between the
start and finish of one of its reader's executions (at most once in each), each item's side-effect nodes finishing before the item's node-progress, and a writer's
file complete before its finish (checked when the event is queued).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import itertools
import json
import subprocess
import time
import traceback
from collections import defaultdict
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bench_oracle import FFMPEG
from verify_runtime import sse_event_mismatches, sse_record

import api
import native_executor
import process
from api import ExecutionOptions, NodeId
from chain.cache import OutputCache
from chain.chain import Chain, CollectorNode, FunctionNode, GeneratorNode
from events import Event, EventConsumer
from native_executor import NativeExecutor
from nodes.impl import numpy_pool
from nodes.impl.execution_scheduler import LatestBroadcasts
from process import ExecutionId, Executor, NodeExecutionError, NodeOutput

KINDS = ("python", "native")

# The owner's stand-in's ruling P7 (final review 1, H1): the native engine's compact (C-contiguous) images are the
# reference; a reduction depends on the pixel values and shape only. Downstream of a NumPy view (a crop, a split
# channel) the Python executor's reductions may differ in the last digit or a zero's sign. Per layout chain of
# native/tests/test_engine_differential.py, the mismatches the oracle's run is recorded to give; any other fails.
# Lens Blur into Image Statistics agrees: the statistics take one channel, and the review's difference was a
# multi-channel call no chain makes.
LAYOUT_DIFFERENCES: dict[str, list[str]] = {
    "crop-statistics": [
        "files differ: ['stats_0_255_118.6884.png', 'stats_0_255_118.6885.png']",
        (
            'name: broadcast {"data": {"data": {"0": null}, "nodeId": "name", "sequenceTypes": {}, "types": {"0":'
            ' {"type": "string-literal", "value": "stats_0_255_118.6885"}}}, "event": "node-broadcast"} sent 1 ti'
            "mes by converted, 0 by baseline"
        ),
        (
            'stats: broadcast {"data": {"data": {"0": null, "1": null, "2": null, "3": null}, "nodeId": "stats", '
            '"sequenceTypes": {}, "types": {"0": {"type": "numeric-literal", "value": 0.0}, "1": {"type": "numeri'
            'c-literal", "value": 255.0}, "2": {"type": "numeric-literal", "value": 118.6885}, "3": {"type": "num'
            'eric-literal", "value": 124.1034}}}, "event": "node-broadcast"} sent 1 times by converted, 0 by base'
            "line"
        ),
        (
            'name: final state differs ({"data": {"0": null}, "nodeId": "name", "sequenceTypes": {}, "types": {"0'
            '": {"type": "string-literal", "value": "stats_0_255_118.6884"}}} baseline, {"data": {"0": null}, "no'
            'deId": "name", "sequenceTypes": {}, "types": {"0": {"type": "string-literal", "value": "stats_0_255_'
            '118.6885"}}} converted)'
        ),
        (
            'stats: final state differs ({"data": {"0": null, "1": null, "2": null, "3": null}, "nodeId": "stats"'
            ', "sequenceTypes": {}, "types": {"0": {"type": "numeric-literal", "value": 0.0}, "1": {"type": "nume'
            'ric-literal", "value": 255.0}, "2": {"type": "numeric-literal", "value": 118.6884}, "3": {"type": "n'
            'umeric-literal", "value": 124.1034}}} baseline, {"data": {"0": null, "1": null, "2": null, "3": null'
            '}, "nodeId": "stats", "sequenceTypes": {}, "types": {"0": {"type": "numeric-literal", "value": 0.0},'
            ' "1": {"type": "numeric-literal", "value": 255.0}, "2": {"type": "numeric-literal", "value": 118.688'
            '5}, "3": {"type": "numeric-literal", "value": 124.1034}}} converted)'
        ),
    ],
    "split-statistics": [
        (
            'stats: broadcast {"data": {"data": {"0": null, "1": null, "2": null, "3": null}, "nodeId": "stats", '
            '"sequenceTypes": {}, "types": {"0": {"type": "numeric-literal", "value": -0.0}, "1": {"type": "numer'
            'ic-literal", "value": 254.998}, "2": {"type": "numeric-literal", "value": 127.3388}, "3": {"type": "'
            'numeric-literal", "value": 127.1562}}}, "event": "node-broadcast"} sent 1 times by converted, 0 by b'
            "aseline"
        ),
    ],
    "lens-blur-statistics": [],
}


@dataclass
class Result:
    """One run of a chain: its events as queued, its error, its output files' digests and the writers whose
    node-finish was queued before their file existed."""

    events: list[dict[str, Any]]
    error: BaseException | None
    files: dict[str, str]
    seconds: float
    early_finishes: list[str] = field(default_factory=list)


class _Queue(EventConsumer):
    def __init__(self, on_put: Callable[[dict[str, Any]], None]) -> None:
        self.events: list[dict[str, Any]] = []
        self.on_put = on_put

    def put(self, event: Event) -> None:
        # As /sse sends it: JSON (output ids become strings).
        sent = json.loads(json.dumps(event))
        self.on_put(sent)
        self.events.append(sent)


class _EveryBroadcast(LatestBroadcasts):
    """LatestBroadcasts that never replaces a pending broadcast: each submit is sent."""

    def __init__(self) -> None:
        super().__init__()
        self.__sequence = itertools.count()

    def submit(self, key: str, send: Callable[[], Awaitable[None]]) -> None:
        super().submit(f"{key}#{next(self.__sequence)}", send)


@contextlib.contextmanager
def every_broadcast():
    """process.Executor sending every broadcast (the oracle: its LatestBroadcasts would drop some by timing), and the
    engine without its loop preview rate limit."""
    original, interval = process.LatestBroadcasts, native_executor.PREVIEW_INTERVAL
    process.LatestBroadcasts = _EveryBroadcast
    native_executor.PREVIEW_INTERVAL = 0
    try:
        yield
    finally:
        process.LatestBroadcasts = original
        native_executor.PREVIEW_INTERVAL = interval


def _error_event(error: BaseException) -> dict[str, Any]:
    """server.py's execution-error event of a failed /run."""
    if isinstance(error, NodeExecutionError) and error.__cause__ is not None:
        trace = "".join(
            traceback.format_exception(
                type(error.__cause__), error.__cause__, error.__cause__.__traceback__
            )
        )
    else:
        trace = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )
    data: dict[str, Any] = {
        "message": "Error running nodes!",
        "source": None,
        "exception": str(error),
        "exceptionTrace": trace,
    }
    if isinstance(error, NodeExecutionError):
        data["source"] = {
            "nodeId": error.node_id,
            "schemaId": error.node_data.schema_id,
            "inputs": error.inputs,
        }
    return {"event": "execution-error", "data": data}


LOSSLESS_VIDEO = frozenset({".mkv"})
LOSSY_VIDEO = frozenset({".mp4", ".webm", ".avi", ".mov"})


def _file_digest(path: Path) -> str:
    """A file's SHA-256; with the pinned FFmpeg, a lossless video's of its decoded frames' MD5s (framemd5), a lossy
    one's frame count and size."""
    suffix = path.suffix.lower()
    if suffix in LOSSLESS_VIDEO | LOSSY_VIDEO and FFMPEG.exists():
        frames = subprocess.run(
            [
                str(FFMPEG),
                "-nostdin",
                "-v",
                "error",
                "-i",
                str(path),
                "-map",
                "0:v",
                "-f",
                "framemd5",
                "-",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
            stdin=subprocess.DEVNULL,
        ).stdout
        if suffix in LOSSLESS_VIDEO:
            return "frames:" + hashlib.sha256(frames.encode()).hexdigest()
        sizes = [line for line in frames.splitlines() if line.startswith("#dimensions")]
        count = sum(
            1 for line in frames.splitlines() if line and not line.startswith("#")
        )
        return f"lossy video: {count} frames, {sizes}"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(folder: Path | None) -> dict[str, str]:
    """Every file under the folder by relative path, as _file_digest gives it."""
    if folder is None or not folder.exists():
        return {}
    return {
        path.relative_to(folder).as_posix(): _file_digest(path)
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


# Third-party modules a package may lack in the CPU runtime: its nodes stay out, as the worker logs them (e.g. the
# TensorRT package, registered once another test imported it).
OPTIONAL_MODULES = frozenset({"tensorrt"})


def fatal_load_errors(errors: list[api.LoadErrorInfo]) -> list[api.LoadErrorInfo]:
    """Load errors but a missing optional third-party module's (OPTIONAL_MODULES): an in-project import that fails
    (a renamed module under packages/ or nodes/) stays an error."""
    return [
        e
        for e in errors
        if not (
            isinstance(e.error, ModuleNotFoundError)
            and (e.error.name or "").split(".")[0] in OPTIONAL_MODULES
        )
    ]


def load_registry_errors(server: Path) -> list[api.LoadErrorInfo]:
    """The worker's node registration (registry.load_nodes) from server.py; its fatal load errors."""
    return fatal_load_errors(api.registry.load_nodes(str(server)))


def execute(
    kind: str,
    chain: Chain,
    *,
    storage: Path,
    output: Path | None = None,
    writers: dict[NodeId, Callable[[], int]] | None = None,
    options: dict[str, Any] | None = None,
    send_broadcast_data: bool = True,
    pool_size: int = 8,
    during: Callable[[object], None] | None = None,
    parent_cache: dict[NodeId, NodeOutput] | None = None,
    preview_limit: bool = False,
) -> Result:
    """Runs the chain on one executor ("python" or "native") as /run does. writers: per writer node, a count of its
    files, checked against its node-finish events as they are queued. during: called with the executor once it
    runs (e.g. to pause or stop it). parent_cache: individually run outputs, as /run passes the server's cache.
    The oracle sends every broadcast, and the engine runs without its loop preview rate limit unless preview_limit."""
    early: list[str] = []
    finishes: defaultdict[str, int] = defaultdict(int)

    def check(event: dict[str, Any]) -> None:
        data = event.get("data")
        node_id = data.get("nodeId") if isinstance(data, dict) else None
        if (
            event["event"] == "node-finish"
            and writers
            and isinstance(node_id, str)
            and NodeId(node_id) in writers
        ):
            finishes[node_id] += 1
            if writers[NodeId(node_id)]() < finishes[node_id]:
                early.append(
                    f"{node_id}: node-finish {finishes[node_id]} before its file"
                )

    queue = _Queue(check)

    async def main() -> tuple[BaseException | None, float]:
        loop = asyncio.get_running_loop()
        pool = ThreadPoolExecutor(pool_size, initializer=numpy_pool.install)
        arguments: dict[str, Any] = {
            "id": ExecutionId(f"differential {kind}"),
            "chain": chain,
            "send_broadcast_data": send_broadcast_data,
            "options": ExecutionOptions.parse(options or {}),
            "loop": loop,
            "queue": queue,
            "storage_dir": storage,
            "pool": pool,
            "pool_size": pool_size,
            "parent_cache": None
            if parent_cache is None
            else OutputCache(static_data=parent_cache.copy()),
        }
        executor = (
            Executor(**arguments) if kind == "python" else NativeExecutor(**arguments)
        )
        start = time.perf_counter()
        error: BaseException | None = None
        try:
            run = asyncio.ensure_future(executor.run())
            if during is not None:
                await asyncio.sleep(0)
                await loop.run_in_executor(None, during, executor)
            await run
        except Exception as e:
            error = e
        seconds = time.perf_counter() - start
        # The pool's last callbacks (late queue puts) run before the comparison.
        await asyncio.sleep(0)
        pool.shutdown()
        return error, seconds

    with (
        every_broadcast()
        if kind == "python" or not preview_limit
        else contextlib.nullcontext()
    ):
        error, seconds = asyncio.run(main())
    if error is not None:
        queue.events.append(json.loads(json.dumps(_error_event(error))))
    return Result(queue.events, error, digest(output), seconds, early)


def _outcome(error: BaseException | None) -> tuple[str, str] | None:
    if error is None:
        return None
    cause = (
        error.__cause__
        if isinstance(error, NodeExecutionError) and error.__cause__
        else error
    )
    return type(error).__name__, f"{type(cause).__name__}: {error}"


def _occurrences(events: list[dict[str, Any]]) -> dict[str, list[list[int | None]]]:
    """Per node, its executions as [start index, finish index or None] in event order."""
    runs: dict[str, list[list[int | None]]] = defaultdict(list)
    for index, event in enumerate(events):
        data = event.get("data")
        if not isinstance(data, dict):
            continue
        node_id = data.get("nodeId")
        if not isinstance(node_id, str):
            continue
        if event["event"] == "node-start":
            runs[node_id].append([index, None])
        elif event["event"] == "node-finish":
            if runs[node_id] and runs[node_id][-1][1] is None:
                runs[node_id][-1][1] = index
            else:
                runs[node_id].append([None, index])
    return runs


def order_violations(events: list[dict[str, Any]], chain: Chain) -> list[str]:
    """The A6 order invariants the events break (none: an empty list)."""
    violations: list[str] = []
    if not events or events[0]["event"] != "chain-start":
        violations.append("chain-start is not the first event")
    errors = [i for i, e in enumerate(events) if e["event"] == "execution-error"]
    if errors and (len(errors) > 1 or errors[0] != len(events) - 1):
        violations.append("execution-error is not the last event")
    runs = _occurrences(events)
    for node_id, executions in runs.items():
        for start, finish in executions:
            if start is None:
                violations.append(
                    f"{node_id}: node-finish without its node-start (event {finish})"
                )
            elif finish is not None and finish < start:
                violations.append(f"{node_id}: node-finish before node-start")
    failed = bool(errors) or any(
        "Errors occurred during iteration" in str(e["data"].get("exception", ""))
        for e in events
        if e["event"] == "execution-error"
    )
    if failed:
        return violations
    # A regular node fed only through lazy inputs of one reader runs inside one of that reader's executions, at most
    # once in each: a reader that does not read it (Save Image skipping an existing file, a Conditional's untaken
    # branch, Logic Operation's B) runs without it. Not a side-effect node (e.g. Load Image), which runs as a sink of
    # its own, in sink order, nor a generator or collector, which its loop runs.
    for upstream_id, upstream in chain.nodes.items():
        edges = chain.edges_from(upstream_id)
        readers = {edge.target.id for edge in edges}
        if (
            not edges
            or len(readers) != 1
            or upstream.data.side_effects
            or not isinstance(upstream, FunctionNode)
        ):
            continue
        (reader_id,) = readers
        reader = chain.nodes[reader_id]
        lazy_ids = {i.id for i in reader.data.inputs if i.lazy}
        if not all(edge.target.input_id in lazy_ids for edge in edges):
            continue
        outer = runs.get(reader_id, [])
        enclosing: dict[int, int] = {}
        for k, (start, finish) in enumerate(runs.get(upstream_id, [])):
            j = next(
                (
                    j
                    for j, (outer_start, outer_finish) in enumerate(outer)
                    if start is not None
                    and outer_start is not None
                    and outer_start <= start
                    and (
                        outer_finish is None
                        or (finish is not None and finish <= outer_finish)
                    )
                ),
                None,
            )
            if j is None:
                violations.append(
                    f"{upstream_id}: execution {k} is outside every execution of {reader_id} (lazy input)"
                )
            elif j in enclosing:
                violations.append(
                    f"{upstream_id}: executions {enclosing[j]} and {k} inside {reader_id}'s execution {j}"
                )
            else:
                enclosing[j] = k
    # Each item's side-effect nodes finish before the item's node-progress.
    for generator_id, node in chain.nodes.items():
        if not isinstance(node, GeneratorNode):
            continue
        iterated = _iterated_side_effects(chain, node)
        progress = [
            index
            for index, e in enumerate(events)
            if e["event"] == "node-progress"
            and e["data"]["nodeId"] == generator_id
            and e["data"]["progress"] != 1
        ]
        for item, position in enumerate(progress[1:]):
            for sink in iterated:
                sink_runs = runs.get(sink, [])
                finish = sink_runs[item][1] if item < len(sink_runs) else None
                if finish is None or finish > position:
                    violations.append(
                        f"{sink}: item {item} finishes after {generator_id}'s progress for it"
                    )
    return violations


def _iterated_side_effects(chain: Chain, generator: GeneratorNode) -> list[str]:
    found: list[str] = []
    seen = {generator.id}
    pending = [
        edge.target.id
        for edge in chain.edges_from(generator.id)
        if edge.source.output_id in generator.data.single_iterable_output.outputs
    ]
    while pending:
        node_id = pending.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        node = chain.nodes[node_id]
        if isinstance(node, FunctionNode) and node.has_side_effects():
            found.append(node_id)
        if not isinstance(node, CollectorNode):
            pending.extend(edge.target.id for edge in chain.edges_from(node_id))
    return found


def mismatches(
    name: str, chain: Chain, oracle: Result, engine: Result, directory: Path
) -> list[str]:
    """How the engine's run differs from the oracle's, and the engine's order violations."""
    found: list[str] = []
    if _outcome(oracle.error) != _outcome(engine.error):
        found.append(
            f"outcome: oracle {_outcome(oracle.error)}, engine {_outcome(engine.error)}"
        )
    if oracle.files != engine.files:
        changed = sorted(
            k
            for k in oracle.files.keys() | engine.files.keys()
            if oracle.files.get(k) != engine.files.get(k)
        )
        found.append(f"files differ: {changed[:8]}")
    generators = sorted(
        n.id for n in chain.nodes.values() if isinstance(n, GeneratorNode)
    )

    def record(result: Result) -> dict[str, Any]:
        return {
            "name": name,
            "generator_node_ids": generators,
            **sse_record(result.events, directory),
        }

    found.extend(sse_event_mismatches(record(oracle), record(engine)))
    found.extend(f"order: {v}" for v in order_violations(engine.events, chain))
    found.extend(f"write: {v}" for v in engine.early_finishes)
    return found
