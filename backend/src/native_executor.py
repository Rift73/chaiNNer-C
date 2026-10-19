"""The native run engine's Python side (native/local/instant-exec/DESIGN.md section 2).

CHAINNER_C_EXECUTOR selects the executor of /run and /run/individual: "native" (this module, the default) or "python"
(process.Executor, the diagnostic oracle: DESIGN.md amendment A5); a diagnostic switch, never a user setting. The engine (native/src/run/executor.cpp,
_chainner_graph.RunEngine) owns the run: its driver thread follows process.Executor's control flow (generator
groups one after another, items in order, each item's sinks then collectors, then the non-iterated sinks) and runs
every node that may not run beside others (side effects, a context, a GPU, a lazy input) there, in today's order;
its worker threads run pure nodes (pure_cpu_node) as soon as their inputs are ready, across branches, sinks and the
items of the item window. This module builds the engine's plan from an optimized chain and serves the engine's
calls into Python: node bodies through process.run_node (enforcement and errors as today), broadcasts, generator and
collector steps, contexts, cleanups, the item window's decision and the events the engine posts to the queue.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import sys
import time
import weakref
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast

import numpy as np
from sanic.log import logger

import navi
from api import (
    BroadcastData,
    Collector,
    ExecutionOptions,
    Generator,
    IterationView,
    IterOutputId,
    NodeContext,
    NodeData,
    NodeId,
    OutputId,
    SettingsParser,
    registry,
)
from chain.cache import OutputCache, get_cache_strategies
from chain.chain import Chain, CollectorNode, FunctionNode, GeneratorNode, Node
from chain.input import EdgeInput, Input, InputMap
from events import EventConsumer, PreviewUrl
from nodes.impl.execution_cleanup import run_cleanup_functions, run_cleanup_groups
from nodes.impl.execution_scheduler import pure_cpu_node
from nodes.impl.item_window import (
    OUTPUT_UNRESOLVED,
    PATH_CHECK_FAILED,
    SOURCE_UNDER_OUTPUT,
    SOURCES_UNKNOWN,
    Prepared,
    affinity_cpus,
    escaping_targets,
    memory_budget,
    node_phases,
    output_nbytes,
    read_ahead_blocker,
    window_size,
)
from nodes.impl.native_graph import graph
from nodes.impl.result_cache import cache_budget, keyed_settings
from process import (
    CollectorOutput,
    ExecutionId,
    GeneratorOutput,
    NodeExecutionError,
    NodeOutput,
    RegularOutput,
    collect_input_information,
    compute_broadcast,
    compute_sequence_broadcast,
    enforce_output,
    on_failing_node,
    prepare_node,
    run_collector_iterate,
    run_node,
)
from progress_controller import Aborted
from util import combine_sets

if TYPE_CHECKING:
    from events import Event
    from nodes.properties.inputs.numpy_inputs import ImageInput
    from nodes.properties.outputs.numpy_outputs import ImageOutput
    from process import _SplitIterator

EXECUTOR_VARIABLE = "CHAINNER_C_EXECUTOR"
EXECUTORS = ("python", "native")
DEFAULT_EXECUTOR = "native"
STATIC_HITS = -1  # a static cache entry (StaticCaching) in the plan
# A /run's LargeImage previews of a loop's items: per node, the first at once, then at most one start per this many
# seconds, the newest kept meanwhile; a loop's and the run's end send what waits (hunt-misc 6, Fable R4). 0: no limit
# (engine_differential.every_broadcast, as the oracle sends every broadcast).
PREVIEW_INTERVAL = 0.25
# A stopped run's own end (its events and cleanups) is awaited this long; a node still running then (a long Python
# body or native kernel without an abort check) finishes in the background, well within the host's 3 s for /kill.
STOP_GRACE_SECONDS = 0.5
# The runs ended for their requests while a node still ran, until their engines end and close (a task's only strong
# reference: the loop keeps weak ones), with the names of the nodes they still ran.
_ENDING_RUNS: dict[asyncio.Task[None], str] = {}


def ending_runs() -> list[asyncio.Task[None]]:
    """The stopped runs whose node still runs: a later run starts once they ended, a later Stop waits for them (a
    node that never returns then exceeds the host's /kill timeout, which restarts the worker), and the server's CUDA
    checks and collection wait for them (the node may still queue CUDA work)."""
    return [task for task in _ENDING_RUNS if not task.done()]


def _log_stopped_run_error(done: asyncio.Future[object]) -> None:
    """Reads a stopped run's result, which its request no longer does: the node's real error is only logged, never
    dropped, and asyncio does not report it as never retrieved."""
    error = None if done.cancelled() else done.exception()
    if error is not None and not isinstance(error, Aborted):
        logger.error("A stopped run's node failed: %s", error, exc_info=error)


# The memo cache's allow-list (DESIGN.md amendment A3): native nodes without side effects or a context, except the
# random generators below; these Python nodes, audited as deterministic and side-effect free (no file, clock, random,
# network or GPU reads); and Load Image, keyed by its file's identity. test_engine_memo.py lists the sets.
MEMO_PYTHON_NODES: frozenset[str] = frozenset(
    {
        "chainner:image:add_normals",
        "chainner:image:average_color_fix",
        "chainner:image:balance_normals",
        "chainner:image:bilateral_blur",
        "chainner:image:convert_normal_map",
        "chainner:image:generate_hash",
        "chainner:image:metal_to_specular",
        "chainner:image:normal_generator",
        "chainner:image:normalize_normal_map",
        "chainner:image:planes_to_rgb",
        "chainner:image:rgb_to_planes",
        "chainner:image:specular_to_metal",
        "chainner:image:strengthen_normals",
        "chainner:utility:color_from_channels",
        "chainner:utility:derive_seed",
        "chainner:utility:separate_color",
    }
)
MEMO_EXCLUDED_NATIVE: frozenset[str] = frozenset(
    {"chainner:image:add_noise", "chainner:image:create_noise"}
)
# The generators whose per-run (not iterated) outputs follow from their inputs alone, so their run is keyed by them
# (the key Load Images' items extend with their files' identities): Load Images' and Load Models' Directory is their
# input, Clip Frames' FPS and Colour Info are its clip's, and Range, Split Spritesheet and Load DDS Images have none.
# Load Video is not one: its FPS, Audio Stream, Colour Info and Colour are read from its file.
MEMO_GENERATORS: frozenset[str] = frozenset(
    {
        "chainner:image:load_dds_images",
        "chainner:image:load_images",
        "chainner:image:split_spritesheet",
        "chainner:ncnn:load_models",
        "chainner:onnx:load_models",
        "chainner:pytorch:load_models",
        "chainner:utility:range",
        "chainner:vapoursynth:clip_frames",
    }
)
LOAD_IMAGE = "chainner:image:load"


def memo_eligible(data: NodeData, native: bool) -> bool:
    """Whether the memo cache may keep the node's outputs (MEMO_PYTHON_NODES)."""
    if data.schema_id == LOAD_IMAGE:
        return True
    if data.kind != "regularNode" or data.side_effects or data.node_context:
        return False
    return (
        data.schema_id not in MEMO_EXCLUDED_NATIVE
        if native
        else data.schema_id in MEMO_PYTHON_NODES
    )


def memo_constant(value: object) -> str | None:
    """A constant input's part of a memo key: its JSON, as the UI sent it; None when it has none."""
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return None


def memo_bytes(values: list[object]) -> int:
    """The memory a memo entry of these values keeps alive: each NumPy array's root buffer once (a view keeps its base
    alive, e.g. a slice of a loaded image) and bytes objects."""
    roots: dict[int, int] = {}
    total = 0
    for value in values:
        if isinstance(value, np.ndarray):
            root = value
            while isinstance(root.base, np.ndarray):
                root = root.base
            roots[id(root)] = root.nbytes
        elif isinstance(value, bytes):
            total += len(value)
    return total + sum(roots.values())


def file_identity(path: object) -> str | None:
    """A file's identity for memo keys: its path, size, modification time, volume and file id; None when it cannot
    be read."""
    if not isinstance(path, (str, os.PathLike)):
        return None
    try:
        stat = os.stat(path)
    except (OSError, ValueError):
        return None
    return f"{os.fspath(path)}|{stat.st_size}|{stat.st_mtime_ns}|{stat.st_dev}|{stat.st_ino}"


# Load Image outputs of individual runs, with their files' identities as they were loaded: a later run that finds one in
# the server's cache keys it in the memo by that identity, not by the file's identity now (it may have changed since).
_LOADED: dict[int, tuple[weakref.ref[RegularOutput], str]] = {}


def remember_identity(output: RegularOutput, identity: str) -> None:
    key = id(output)
    _LOADED[key] = (weakref.ref(output, lambda _: _LOADED.pop(key, None)), identity)


def loaded_identity(output: object) -> str | None:
    """The file identity an individual run of Load Image recorded with this output, if any."""
    entry = _LOADED.get(id(output))
    return entry[1] if entry is not None and entry[0]() is output else None


def executor_kind() -> str:
    """The executor CHAINNER_C_EXECUTOR selects; an unknown value is logged and the default is used."""
    value = os.environ.get(EXECUTOR_VARIABLE, "") or DEFAULT_EXECUTOR
    if value not in EXECUTORS:
        logger.warning(
            "%s=%r is not one of %s; using %s",
            EXECUTOR_VARIABLE,
            value,
            EXECUTORS,
            DEFAULT_EXECUTOR,
        )
        return DEFAULT_EXECUTOR
    return value


class _Engine(Protocol):
    """The parts of _chainner_graph.RunEngine that this module calls."""

    @property
    def commit_slot(self) -> int: ...
    @property
    def aborted(self) -> bool: ...
    @property
    def paused(self) -> bool: ...
    def start(self, mode: str, target: int) -> None: ...
    def kill(self) -> None: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def flush_broadcasts(self) -> None: ...
    def close(self) -> None: ...
    def lookahead(self, node: int, input_index: int) -> int: ...
    def peek(
        self, node: int, input_index: int, offset: int, origin: int
    ) -> object | None: ...


def _iterated_nodes(
    chain: Chain, node: GeneratorNode
) -> tuple[list[CollectorNode], list[FunctionNode], set[Node]]:
    """process.Executor.__get_iterated_nodes: the collectors and side-effect nodes a generator iterates and every
    node it reaches (itself included), in the same set iteration orders."""
    collectors: set[CollectorNode] = set()
    output_nodes: set[FunctionNode] = set()
    seen: set[Node] = {node}

    def visit(n: Node):
        if n in seen:
            return
        seen.add(n)
        if isinstance(n, CollectorNode):
            collectors.add(n)
        elif isinstance(n, GeneratorNode):
            raise ValueError("Nested sequences are not supported")
        else:
            if n.has_side_effects():
                output_nodes.add(n)
            for edge in chain.edges_from(n.id):
                visit(chain.nodes[edge.target.id])

    iterable_output = node.data.single_iterable_output
    for edge in chain.edges_from(node.id):
        if edge.source.output_id in iterable_output.outputs:
            visit(chain.nodes[edge.target.id])
    return list(collectors), list(output_nodes), seen


def _groups(chain: Chain) -> list[list[GeneratorNode]]:
    """process.Executor.__process_nodes: the generator groups (generators sharing sinks), in its order."""
    gens_by_outs: dict[NodeId, set[NodeId]] = {}
    for node_id in chain.topological_order():
        node = chain.nodes[node_id]
        if isinstance(node, GeneratorNode):
            collectors, outputs, _ = _iterated_nodes(chain, node)
            for sink in [*collectors, *outputs]:
                gens_by_outs.setdefault(sink.id, set()).add(node.id)
    groups: list[list[GeneratorNode]] = []
    for group in combine_sets(list(gens_by_outs.values())):
        groups.append(
            [
                n
                for n in (chain.nodes[node_id] for node_id in group)
                if isinstance(n, GeneratorNode)
            ]
        )
    return groups


def _relative_path_input(i: object) -> bool:
    """process._relative_path_input: whether the input is a RelativePathInput (a writer's subdirectory or file name)."""
    module = sys.modules.get("nodes.properties.inputs.file_inputs")
    return module is not None and isinstance(i, module.RelativePathInput)


class _EngineIterationView(IterationView):
    """A node's iteration view for one loop run: its index is the slot the engine commits (or the later slot a serial
    node is computed ahead at, see at()); a look-ahead waits in the engine (GIL released) for a later item."""

    def __init__(
        self,
        engine: _Engine,
        index: int,
        node: Node,
        sources: tuple[str | None, ...],
        base: _EngineIterationView | None = None,
        slot: int = -1,
    ) -> None:
        self.__engine = engine
        self.__index = index
        self.__node = node
        self.__sources = sources
        self.__state: object | None = None
        self.__base = base  # the view whose state this one shares
        # The item the node is computed ahead at; -1: the committing one.
        self.__slot = slot

    def at(self, slot: int) -> _EngineIterationView:
        """This view as the node computed ahead at a later item sees it (process._IterationView.at): that item's
        index, this view's state, look-aheads from that item."""
        return _EngineIterationView(
            self.__engine, self.__index, self.__node, self.__sources, self, slot
        )

    @property
    def index(self) -> int:
        return self.__engine.commit_slot if self.__slot < 0 else self.__slot

    @property
    def source(self) -> str:
        return next(source for source in self.__sources if source is not None)

    def iterated_by(self, input_id: int) -> str | None:
        return self.__sources[self.__input_index(input_id)]

    @property
    def state(self) -> object | None:
        return self.__state if self.__base is None else self.__base.state

    @state.setter
    def state(self, value: object | None) -> None:
        if self.__base is None:
            self.__state = value
        else:
            self.__base.state = value

    def lookahead(self, input_id: int) -> int:
        return self.__engine.lookahead(self.__index, self.__input_index(input_id))

    def peek(self, input_id: int, offset: int) -> object | None:
        if offset < 1:
            raise ValueError(f"A look-ahead offset must be at least 1, not {offset}.")
        index = self.__input_index(input_id)
        value = self.__engine.peek(self.__index, index, offset, self.__slot)
        if value is None:
            return None
        try:
            return self.__node.data.inputs[index].enforce_(value)
        except Exception:
            # The item fails at this input when it comes, with this error (run_node).
            return None

    def __input_index(self, input_id: int) -> int:
        for index, i in enumerate(self.__node.data.inputs):
            if i.id == input_id:
                return index
        raise ValueError(f"{self.__node.data.name} has no input {input_id}.")


class _EngineContext(NodeContext):
    """A node's context in an engine run (process._ExecutorNodeContext's contract): Stop and Pause are the engine's;
    cleanups are kept here and run by the engine after the node and at the end of the run."""

    def __init__(
        self,
        engine: Callable[[], _Engine],
        settings: SettingsParser,
        storage_dir: Path,
        node_id: NodeId,
    ):
        super().__init__()
        self.__engine = engine
        self.__settings = settings
        self.__storage_dir = storage_dir
        self.__node_id = node_id
        self.chain_cleanup_fns: set[Callable[[], None]] = set()
        self.node_cleanup_fns: set[Callable[[], None]] = set()
        self.iteration_view: _EngineIterationView | None = None

    @property
    def iteration(self) -> IterationView | None:
        return self.iteration_view

    @property
    def aborted(self) -> bool:
        return self.__engine().aborted

    @property
    def paused(self) -> bool:
        time.sleep(0)
        return self.__engine().paused

    def set_progress(self, progress: float) -> None:
        self.check_aborted()

    @property
    def settings(self) -> SettingsParser:
        return self.__settings

    @property
    def node_id(self) -> NodeId:
        return self.__node_id

    @property
    def storage_dir(self) -> Path:
        return self.__storage_dir

    def add_cleanup(
        self, fn: Callable[[], None], after: Literal["node", "chain"] = "chain"
    ) -> None:
        if after == "chain":
            self.chain_cleanup_fns.add(fn)
        elif after == "node":
            self.node_cleanup_fns.add(fn)
        else:
            raise ValueError(f"Unknown cleanup type: {after}")


class _Hooks:
    """The engine's calls into Python, made on engine threads with the GIL held."""

    def __init__(
        self,
        executor_id: ExecutionId,
        nodes: list[Node],
        contexts: list[_EngineContext],
        queue: EventConsumer,
        loop: asyncio.AbstractEventLoop,
        inputs: InputMap,
        chain: Chain,
    ) -> None:
        self.executor_id = executor_id
        self.nodes = nodes
        self.contexts = contexts
        self.queue = queue
        self.loop = loop
        self.inputs = inputs
        self.chain = chain
        self.aborted_type = Aborted
        self.preview_url = PreviewUrl
        self.done: asyncio.Future[object] = loop.create_future()
        # The run ended for its request while its engine still ran a node: no event of it reaches the queue any more.
        self.ended = False
        # The nodes started and not finished, as the events report them (the ones a stopped run still runs).
        self.running: dict[str, None] = {}
        self.views: dict[int, _EngineIterationView] = {}
        self.engine: _Engine | None = None
        # The running loop's generators and iterated nodes, and its nodes' lineages (process._LoopRun).
        self.loop_iterated: set[NodeId] = set()
        self.lineages: dict[NodeId, NodeId | None] = {}

    # Threads and events

    def post(self, events: list[dict[str, object]]) -> None:
        self.loop.call_soon_threadsafe(self.__put, events)

    def __put(self, events: list[dict[str, object]]) -> None:
        if self.ended:
            return
        for event in events:
            data = event.get("data")
            node_id = data.get("nodeId") if isinstance(data, dict) else None
            if event["event"] == "node-start" and isinstance(node_id, str):
                self.running[node_id] = None
            elif event["event"] == "node-finish" and isinstance(node_id, str):
                self.running.pop(node_id, None)
            self.queue.put(cast("Event", event))

    def finish(self, error: BaseException | None, result: object) -> None:
        self.loop.call_soon_threadsafe(self.__finish, error, result)

    def __finish(self, error: BaseException | None, result: object) -> None:
        if self.done.done():
            return
        if error is not None:
            self.done.set_exception(error)
        else:
            self.done.set_result(result)

    def warn(self, message: str) -> None:
        logger.warning(message)

    def info(self, message: str) -> None:
        logger.info(message)

    # Nodes

    def run_node(
        self, index: int, inputs: list[object], prepared: object, ahead: int
    ) -> NodeOutput | CollectorOutput:
        """process.run_node; prepared: the value the node's prepare phase computed ahead (None: none); ahead: the
        later item a serial node is computed at for a look-ahead, with its context there (process.Executor's
        __ahead_context: its own, with its iteration view at that item), -1: none."""
        node = self.nodes[index]
        phase = None if prepared is None else Prepared(prepared)
        context = self.contexts[index]
        if ahead >= 0:
            view = context.iteration_view
            context = copy.copy(context)
            context.iteration_view = None if view is None else view.at(ahead)
        return run_node(node.data, context, inputs, node.id, phase)

    def enforce_input(self, index: int, input_index: int, value: object) -> object:
        return self.nodes[index].data.inputs[input_index].enforce_(value)

    def enforce_output(self, index: int, output_index: int, value: object) -> object:
        return self.nodes[index].data.outputs[output_index].enforce_(value)

    def native_failure(
        self, index: int, cause: Exception, inputs: list[object]
    ) -> NodeExecutionError:
        """A native entry point's error (cause: the engine's python_exception of
        its NodeError) as run_node reports a body's error."""
        node = self.nodes[index]
        error = NodeExecutionError(
            node.id, node.data, str(cause), collect_input_information(node.data, inputs)
        )
        error.__cause__ = cause
        return error

    def regular_outputs(self, output: object) -> list[object]:
        assert isinstance(output, RegularOutput)
        return output.output

    def generator_outputs(self, output: object) -> list[object]:
        assert isinstance(output, GeneratorOutput)
        return output.partial_output

    def collector(self, output: object) -> Collector:
        assert isinstance(output, CollectorOutput)
        return output.collector

    def node_cleanups(self, index: int, preserve_error: bool) -> None:
        context = self.contexts[index]
        if context.node_cleanup_fns:
            run_cleanup_functions(
                context.node_cleanup_fns, preserve_error=preserve_error
            )

    def chain_cleanups(self, preserve_error: bool) -> None:
        run_cleanup_groups(
            (
                callbacks
                for context in self.contexts
                for callbacks in (context.node_cleanup_fns, context.chain_cleanup_fns)
            ),
            preserve_error=preserve_error,
        )

    def output_bytes(self, values: list[object]) -> int:
        return output_nbytes(values)

    # Broadcasts

    def broadcast(
        self,
        index: int,
        outputs: list[object],
        generator: object,
        native: dict[int, tuple[BroadcastData, navi.ExpressionJson]],
    ) -> dict[str, object]:
        """process.Executor's broadcast payload; native: the engine's own image broadcasts by output position (their
        values are None in outputs)."""
        node = self.nodes[index]
        computed, computed_types = compute_broadcast(outputs, node.data.outputs)
        data: dict[OutputId, BroadcastData | None] = {}
        types: dict[OutputId, navi.ExpressionJson | None] = {}
        for position, output in enumerate(node.data.outputs):
            if position in native:
                data[output.id], types[output.id] = native[position]
            elif output.id in computed:
                data[output.id], types[output.id] = (
                    computed[output.id],
                    computed_types[output.id],
                )
        sequence_types: dict[IterOutputId, navi.ExpressionJson] = {}
        if isinstance(generator, GeneratorOutput):
            sequence_types, item_types = compute_sequence_broadcast(
                [generator.generator], node.data.iterable_outputs
            )
            for output_id, kind in item_types.items():
                types[output_id] = kind
        return {
            "event": "node-broadcast",
            "data": {
                "nodeId": node.id,
                "data": data,
                "types": types,
                "sequenceTypes": sequence_types,
            },
        }

    # Generators

    def items(self, output: object) -> tuple[Iterator[object], bool, int, bool]:
        """A generator node's items: (iterator, split, expected length, fail_fast). split: the iterator names its
        items (describe) apart from loading them (materialize), as Load Images' native iterator does."""
        assert isinstance(output, GeneratorOutput)
        iterator = output.generator.supplier().__iter__()
        return (
            iterator,
            hasattr(iterator, "describe"),
            output.generator.expected_length,
            output.generator.fail_fast,
        )

    def file_identity(self, path: object) -> str | None:
        return file_identity(path)

    def item_identity(self, token: object) -> str | None:
        """A split iterator's item token's file identity (Load Images' (path, index)); None for any other token."""
        if not (isinstance(token, tuple) and len(token) == 1):
            return None
        described = token[0]
        if not (
            isinstance(described, tuple)
            and len(described) == 2
            and isinstance(described[1], int)
        ):
            return None
        identity = file_identity(described[0])
        return None if identity is None else f"{identity}|{described[1]}"

    def memo_bytes(self, values: list[object]) -> int:
        return memo_bytes(values)

    def loaded(self, output: object, identity: str) -> None:
        """An individual run of Load Image: its output (the server caches it) and its file's identity as it loaded."""
        assert isinstance(output, RegularOutput)
        remember_identity(output, identity)

    def regular_output(self, outputs: list[object]) -> RegularOutput:
        return RegularOutput(outputs)

    def native_generator(
        self,
        index: int,
        expected_length: int,
        fail_fast: bool,
        source_paths: tuple[Path, ...] | None,
        partial: list[object],
    ) -> GeneratorOutput:
        """A native generator's GeneratorOutput, for its broadcasts and the item window's decision: the engine reads
        its items from the native item source, never through this generator's supplier."""

        def supplier() -> Iterator[object]:
            raise RuntimeError("A native generator's items are read by the engine")

        node = self.nodes[index]
        generator = Generator(
            supplier, expected_length, fail_fast=fail_fast, source_paths=source_paths
        )
        return GeneratorOutput(
            info=node.data.single_iterable_output,
            generator=generator,
            partial_output=partial,
        )

    def fill(self, index: int, output: object, values: object) -> list[object]:
        """process.Executor.__generator_fill_partial_output: an item's outputs, its iterated values enforced."""
        assert isinstance(output, GeneratorOutput)
        node = self.nodes[index]
        iterable_output = node.data.single_iterable_output
        values_list: list[object] = []
        if len(iterable_output.outputs) == 1:
            values_list.append(values)
        else:
            assert isinstance(values, (tuple, list))
            values_list.extend(values)
        assert len(values_list) == len(iterable_output.outputs)
        filled = output.partial_output.copy()
        for position, o in enumerate(node.data.outputs):
            if o.id in iterable_output.outputs:
                filled[position] = o.enforce_(values_list.pop(0))
        return filled

    def describe(self, iterator: Iterator[object], split: bool) -> tuple[object] | None:
        """The next item's token (process.Executor.__describe / __describe_split); None at the end; a yielded error
        is raised."""
        if split:
            try:
                return (cast("_SplitIterator", iterator).describe(),)
            except StopIteration:
                return None
        try:
            values = next(iterator)
        except StopIteration:
            return None
        if isinstance(values, Exception):
            raise values
        return (values,)

    def materialize(
        self,
        index: int,
        iterator: Iterator[object],
        output: object,
        token: tuple[object],
        split: bool,
    ) -> list[object]:
        """The item's outputs (process.Executor.__materialize / __materialize_split)."""
        (values,) = token
        if split:
            values = cast("_SplitIterator", iterator).materialize(values)
            if isinstance(values, Exception):
                raise values
        return self.fill(index, output, values)

    # Collectors

    def collector_iterate(
        self, index: int, inputs: list[object], collector: Collector, prepared: object
    ) -> None:
        node = self.nodes[index]
        assert isinstance(node, CollectorNode)
        run_collector_iterate(
            node, inputs, collector, None if prepared is None else Prepared(prepared)
        )

    def prepare(
        self, index: int, inputs: list[object], collector: Collector | None
    ) -> object | None:
        """A sink's prepare phase ahead of its commit (process.prepare_node)."""
        return prepare_node(self.nodes[index].data, inputs, collector)

    def collector_complete(self, index: int, collector: Collector) -> list[object]:
        return enforce_output(collector.on_complete(), self.nodes[index].data).output

    # Iteration views and the item window

    def begin_loop(self, generators: list[int], iterated: list[int]) -> None:
        self.loop_iterated = {self.nodes[index].id for index in iterated}
        self.lineages = {}

    def iteration(self, index: int) -> None:
        """Attaches the node's view for this loop run, made at its first commit (process.Executor.__iteration_view),
        to its context."""
        view = self.views.get(index)
        if view is None:
            assert self.engine is not None
            node = self.nodes[index]
            lineages = [self.__input_lineage(i) for i in self.inputs.get(node.id)]
            assert any(lineage is not None for lineage in lineages), (
                f"Iterated node {node.id} has no lineage"
            )
            sources = tuple(
                None if lineage is None else self.chain.nodes[lineage].data.schema_id
                for lineage in lineages
            )
            view = self.views[index] = _EngineIterationView(
                self.engine, index, node, sources
            )
        self.contexts[index].iteration_view = view

    def __lineage(self, node_id: NodeId) -> NodeId | None:
        """process.Executor.__lineage: the generator iterating the node, through its first iterated input."""
        if node_id not in self.lineages:
            lineage = None
            for i in self.inputs.get(node_id):
                lineage = self.__input_lineage(i)
                if lineage is not None:
                    break
            self.lineages[node_id] = lineage
        return self.lineages[node_id]

    def __input_lineage(self, i: Input) -> NodeId | None:
        """process.Executor.__input_lineage."""
        if not isinstance(i, EdgeInput):
            return None
        source = self.chain.nodes[i.id]
        if isinstance(source, GeneratorNode):
            if (
                source.data.outputs[i.index].id
                in source.data.single_iterable_output.outputs
            ):
                return i.id
        elif isinstance(source, FunctionNode) and i.id in self.loop_iterated:
            return self.__lineage(i.id)
        return None

    def end_loop(self) -> None:
        for index in self.views:
            self.contexts[index].iteration_view = None
        self.views.clear()
        self.loop_iterated = set()
        self.lineages = {}

    def window(
        self,
        generators: list[int],
        generator_outputs: list[object],
        writers: list[int],
        iterated: list[int],
        cached: Callable[[int, int], tuple[bool, object]],
        collector_inputs: dict[int, list[object]],
    ) -> tuple[int, str, str | None, int, list[dict[str, object]]]:
        """process.Executor.__open_window's decision: (K, reason, blocker, memory budget, warning events for writers
        whose target folder escapes their directory, which the engine sends in its order); logs it."""
        k, reason = window_size(affinity_cpus())
        blocker = None
        warnings: list[dict[str, object]] = []
        if k > 1:
            sources = []
            for node_index, output in zip(generators, generator_outputs):
                assert isinstance(output, GeneratorOutput)
                sources.append(
                    (self.nodes[node_index].id, output.generator.source_paths)
                )
            directories, targets = self.__writer_directories(
                writers,
                dict(zip(generators, generator_outputs)),
                set(iterated),
                cached,
                collector_inputs,
            )
            blocker = read_ahead_blocker(sources, directories)
            if blocker is not None:
                k, reason = 1, blocker
            for writer, folder in escaping_targets(targets):
                name = self.chain.nodes[writer].data.name
                message = f"{name} writes to {folder}, outside its chosen folder."
                logger.warning(message)
                warnings.append(
                    {"event": "execution-warning", "data": {"message": message}}
                )
        logger.info("item window K=%d (%s)", k, reason)
        return k, reason, blocker, memory_budget() if k > 1 else 0, warnings

    def __writer_directories(
        self,
        writers: list[int],
        generator_outputs: dict[int, object],
        iterated: set[int],
        cached: Callable[[int, int], tuple[bool, object]],
        collector_inputs: dict[int, list[object]],
    ) -> tuple[list[tuple[NodeId, Path | None]], list[tuple[NodeId, Path, Path]]]:
        """process.Executor.__writer_directories over the engine's cells."""
        positions = {node.id: index for index, node in enumerate(self.nodes)}
        directories: list[tuple[NodeId, Path | None]] = []
        targets: list[tuple[NodeId, Path, Path]] = []
        for writer in sorted(writers, key=lambda w: self.nodes[w].id):
            node = self.nodes[writer]
            assigned_inputs = self.inputs.get(node.id)
            base: Path | None = None
            parts: list[str] = []
            complete = False
            for index, i in enumerate(node.data.inputs):
                relative = _relative_path_input(i)
                if i.kind != "directory" and not relative:
                    continue
                node_input = assigned_inputs[index]
                value: object
                if not isinstance(node_input, EdgeInput):
                    value = node_input.value
                elif positions[node_input.id] in generator_outputs:
                    source = self.chain.nodes[node_input.id]
                    output = generator_outputs[positions[node_input.id]]
                    assert isinstance(output, GeneratorOutput)
                    is_iterated = (
                        source.data.outputs[node_input.index].id
                        in source.data.single_iterable_output.outputs
                    )
                    value = (
                        None if is_iterated else output.partial_output[node_input.index]
                    )
                elif positions[node_input.id] in iterated:
                    value = None
                elif isinstance(node, CollectorNode):
                    value = collector_inputs[writer][index]
                else:
                    found, value = cached(positions[node_input.id], node_input.index)
                    if not found:
                        value = None
                if not relative:
                    self.__add_target(
                        node.id, base, parts, complete, directories, targets
                    )
                    base = Path(value) if isinstance(value, (str, Path)) else None
                    directories.append((node.id, base))
                    parts, complete = [], True
                elif complete and isinstance(value, str):
                    parts.append(value)
                elif not (value is None and not isinstance(node_input, EdgeInput)):
                    complete = False
            self.__add_target(node.id, base, parts, complete, directories, targets)
        return directories, targets

    @staticmethod
    def __add_target(
        writer: NodeId,
        base: Path | None,
        parts: list[str],
        complete: bool,
        directories: list[tuple[NodeId, Path | None]],
        targets: list[tuple[NodeId, Path, Path]],
    ) -> None:
        if base is None or not parts:
            return
        joined = base.joinpath(*parts)
        folder = joined.parent if complete else joined
        directories.append((writer, folder))
        targets.append((writer, base, folder))

    def warn_look_ahead_off(self, index: int, blocker: str) -> dict[str, object]:
        """process.Executor.__warn_look_ahead_off's warning event (the engine sends it once per node and loop run)."""
        kind, _, subject = blocker.partition(": ")
        name = self.nodes[index].data.name
        if kind == PATH_CHECK_FAILED:
            reason = f"The output folder could not be checked ({subject})"
        else:
            other = self.chain.nodes[NodeId(subject)].data.name
            reason = {
                SOURCES_UNKNOWN: f"The files {other} reads are unknown",
                OUTPUT_UNRESOLVED: f"{other}'s output folder is computed after {name}",
                SOURCE_UNDER_OUTPUT: f"{other}'s output folder may overlap the source folder",
            }[kind]
        message = f"{reason}, so {name} runs without its look-ahead: it uses only the current and earlier frames."
        logger.warning(message)
        return {"event": "execution-warning", "data": {"message": message}}


def _image_input(node_input: object) -> tuple[list[int] | None, bool] | None:
    """(channels, allow_colors) of an input whose enforce the engine performs for images and colours: an ImageInput
    itself (read from sys.modules, as _relative_path_input: importing the inputs package loads torch)."""
    module = sys.modules.get("nodes.properties.inputs.numpy_inputs")
    if module is None or type(node_input) is not module.ImageInput:
        return None
    image = cast("ImageInput", node_input)
    return image.channels, image.allow_colors


def _image_output(output: object) -> tuple[int, bool, bool] | None:
    """(channels or 0, assume_normalized, large) of an output whose enforce and broadcast the engine performs for
    images: an ImageOutput or LargeImageOutput itself (large: LargeImageOutput, whose broadcast has previews)."""
    module = sys.modules.get("nodes.properties.outputs.numpy_outputs")
    if module is None or type(output) not in (
        module.ImageOutput,
        module.LargeImageOutput,
    ):
        return None
    image = cast("ImageOutput", output)
    return (
        image.channels or 0,
        image.assume_normalized,
        type(output) is module.LargeImageOutput,
    )


def build_plan(
    chain: Chain,
    options: ExecutionOptions,
    storage_dir: Path,
    send_broadcast_data: bool,
    parent_cache: OutputCache[NodeOutput] | None,
    engine: Callable[[], _Engine],
) -> tuple[dict[str, object], list[Node], list[_EngineContext], InputMap]:
    """The engine's plan of an optimized chain (nodes, groups, sinks) with every node's context."""
    nodes = list(chain.nodes.values())
    positions = {node.id: index for index, node in enumerate(nodes)}
    inputs = InputMap.from_chain(chain)
    strategies = get_cache_strategies(chain)
    native_ids = set(graph().native_schema_ids())
    contexts: list[_EngineContext] = []
    node_plans: list[dict[str, object]] = []
    for node in nodes:
        data: NodeData = node.data
        package_id = registry.get_package(data.schema_id).id
        contexts.append(
            _EngineContext(
                engine, options.get_package_settings(package_id), storage_dir, node.id
            )
        )
        ignored = (
            set(data.single_iterable_input.inputs)
            if isinstance(node, CollectorNode)
            else set()
        )
        native = data.schema_id in native_ids and not isinstance(node, CollectorNode)
        plan_inputs: list[
            tuple[
                int,
                int,
                object,
                bool,
                bool,
                tuple[list[int] | None, bool] | None,
                object,
                str | None,
            ]
        ] = []
        for node_input, assigned in zip(data.inputs, inputs.get(node.id)):
            source, output_index, value = -1, 0, None
            enforced: object = None
            if isinstance(assigned, EdgeInput):
                source, output_index = positions[assigned.id], assigned.index
            else:
                value = assigned.value
                if native:
                    try:
                        enforced = node_input.enforce_(value)
                    except Exception:
                        # The Python body raises this enforce error, as run_node reports it.
                        native = False
            plan_inputs.append(
                (
                    source,
                    output_index,
                    value,
                    node_input.lazy,
                    node_input.id in ignored,
                    _image_input(node_input),
                    enforced,
                    None if source >= 0 else memo_constant(value),
                )
            )
        strategy = strategies[node.id]
        precached = None if parent_cache is None else parent_cache.peek(node.id)
        node_plans.append(
            {
                "id": node.id,
                "schema": data.schema_id,
                "name": data.name,
                "kind": {"regularNode": 0, "generator": 1, "collector": 2}[data.kind],
                "side_effects": data.side_effects,
                "pure": pure_cpu_node(data),
                "node_context": data.node_context,
                "inputs": plan_inputs,
                "outputs": len(data.outputs),
                "iterable_outputs": [
                    i
                    for i, o in enumerate(data.outputs)
                    if o.id in data.single_iterable_output.outputs
                ]
                if data.kind == "generator"
                else [],
                "hits": STATIC_HITS if strategy.static else strategy.hits_to_live,
                "native": native,
                "phased": node_phases(data.schema_id) is not None,
                "memo": memo_eligible(data, data.schema_id in native_ids),
                "run_keyed": data.schema_id in MEMO_GENERATORS,
                "identity_input": 0 if data.schema_id == LOAD_IMAGE else -1,
                "settings_key": json.dumps(
                    keyed_settings(
                        package_id, options.get_package_settings_json(package_id)
                    ),
                    sort_keys=True,
                    default=str,
                ),
                "image_outputs": [_image_output(o) for o in data.outputs],
                "output_ids": [o.id for o in data.outputs],
                "settings": options.get_package_settings_json(package_id),
                "precached": None
                if precached is None
                else (
                    precached.output
                    if isinstance(precached, RegularOutput)
                    else precached.partial_output
                ),
                "precached_identity": None
                if precached is None
                else loaded_identity(precached),
            }
        )
    groups: list[dict[str, object]] = []
    for generators in _groups(chain):
        entries: list[dict[str, object]] = []
        output_nodes: set[FunctionNode] = set()
        iterated: set[int] = set()
        for generator in generators:
            collectors, outputs, seen = _iterated_nodes(chain, generator)
            output_nodes.update(outputs)
            iterated.update(positions[n.id] for n in seen)
            entries.append(
                {
                    "node": positions[generator.id],
                    "collectors": [positions[c.id] for c in collectors],
                }
            )
        groups.append(
            {
                "generators": entries,
                "outputs": [positions[n.id] for n in output_nodes],
                "iterated": sorted(iterated),
            }
        )
    tail = [
        positions[node.id]
        for node, iterator in chain.get_parent_iterator_map().items()
        if iterator is None and node.has_side_effects()
    ]
    # process.Executor.__send_chain_start's list, from the same set operations (the same order).
    chain_start = set(chain.nodes.keys())
    if parent_cache is not None:
        chain_start.difference_update(parent_cache.keys())
    plan: dict[str, object] = {
        "nodes": node_plans,
        "groups": groups,
        "tail": tail,
        "chain_start": list(chain_start),
        "send_broadcast_data": send_broadcast_data,
        "preview_interval": PREVIEW_INTERVAL,
        "storage_dir": storage_dir,
    }
    return plan, nodes, contexts, inputs


class NativeExecutor:
    """process.Executor's interface over the native engine (server.py's /run, /run/individual, /pause, /resume,
    /kill and /status)."""

    def __init__(
        self,
        id: ExecutionId,
        chain: Chain,
        send_broadcast_data: bool,
        options: ExecutionOptions,
        loop: asyncio.AbstractEventLoop,
        queue: EventConsumer,
        storage_dir: Path,
        parent_cache: OutputCache[NodeOutput] | None = None,
        pool: object = None,
        pool_size: int | None = None,
    ) -> None:
        self.id = id
        self.chain = chain
        self.loop = loop
        # The contexts read the engine from this list, not through this executor: nothing this executor holds may
        # refer back to it. server.py never collects after a successful run or preview, so a cycle would keep the
        # executor, and the output an individual run returned (its done future's), alive after the server drops them.
        # The engine's own cycle (it holds the hooks, which hold it) ends at its close.
        engine: list[_Engine] = []
        plan, nodes, contexts, inputs = build_plan(
            chain,
            options,
            storage_dir,
            send_broadcast_data,
            parent_cache,
            lambda: engine[0],
        )
        self.__nodes = nodes
        self.__hooks = _Hooks(id, nodes, contexts, queue, loop, inputs, chain)
        # The Result Cache settings, before the engine reads its plan (whether the memo is on): a smaller size trims
        # the memo, off empties it; CHAINNER_C_MEMO_BUDGET_MIB wins over both.
        graph().memo_set_budget(cache_budget(options))
        self.__run_engine: _Engine = graph().RunEngine(plan, self.__hooks)
        engine.append(self.__run_engine)
        self.__hooks.engine = self.__run_engine
        self.__closed = False
        # Set by kill(), on any thread.
        self.__stopped: asyncio.Future[None] = loop.create_future()

    @property
    def progress(self) -> NativeExecutor:
        """process.Executor's progress controller, whose aborted and paused server.py reads: this executor (a
        property, not an attribute, which would be a cycle)."""
        return self

    @property
    def aborted(self) -> bool:
        return self.__run_engine.aborted

    @property
    def paused(self) -> bool:
        return self.__run_engine.paused

    async def __drive(self, mode: str, target: int) -> object:
        """The run's result. Stopped (kill(), or the request dropped: the app's Stop drops its /run before /kill), it
        ends with its engine's end, or after STOP_GRACE_SECONDS as Aborted while a node still runs: that node's
        thread finishes in the background, where its cleanups and the run's then run and the engine closes."""
        try:
            await self.__after_stopped_runs()
        except BaseException:
            # Never started: its engine releases the plan's and hooks' references now.
            self.__closed = True
            self.__run_engine.close()
            raise
        self.__run_engine.start(mode, target)
        done = self.__hooks.done
        try:
            if not done.done():
                await asyncio.wait(
                    (done, self.__stopped), return_when=asyncio.FIRST_COMPLETED
                )
            if not done.done():
                await asyncio.wait((done,), timeout=STOP_GRACE_SECONDS)
        except asyncio.CancelledError:
            self.__run_engine.kill()
            if not done.done():
                try:
                    await asyncio.wait((done,), timeout=STOP_GRACE_SECONDS)
                except asyncio.CancelledError:
                    pass  # cancelled again: the request ends as cancelled at once
            if done.done():
                _log_stopped_run_error(done)
            else:
                self.__end_in_background()
            raise
        if not done.done():
            self.__end_in_background()
            raise Aborted()
        return done.result()

    async def __after_stopped_runs(self) -> None:
        """A run that ended while its node still ran ends wholly (that node, its cleanups and the run's, its engine's
        close) before this run starts, nor do a stopped run's cleanups land in this run's nodes. Not so for a run
        stopped within its grace (a superseded preview's old run): it is not ending yet, so this one may start beside
        its last node. Stopped meanwhile, this run ends as Aborted without starting."""
        warned = False
        while not self.__stopped.done():
            ending = ending_runs()
            if not ending:
                return
            if not warned:
                warned = True
                names = ", ".join(n for n in (_ENDING_RUNS[t] for t in ending) if n)
                message = f"Waiting for the stopped node {names or '(unknown)'} to finish before this run starts."
                logger.warning(message)
                self.__hooks.queue.put(
                    {"event": "execution-warning", "data": {"message": message}}
                )
            await asyncio.wait(
                (*ending, self.__stopped), return_when=asyncio.FIRST_COMPLETED
            )
        raise Aborted()

    def __end_in_background(self) -> None:
        self.__hooks.ended = True
        self.__closed = True  # the engine closes once it ends
        task = self.loop.create_task(self.__close_when_ended())
        _ENDING_RUNS[task] = ", ".join(
            self.chain.nodes[NodeId(node_id)].data.name
            for node_id in self.__hooks.running
            if NodeId(node_id) in self.chain.nodes
        )
        task.add_done_callback(_ENDING_RUNS.pop)

    async def __close_when_ended(self) -> None:
        done = self.__hooks.done
        await asyncio.wait((done,))
        _log_stopped_run_error(done)
        await self.loop.run_in_executor(None, self.__run_engine.close)

    async def run(self) -> None:
        logger.debug("Running native executor %s", self.id)
        try:
            await self.__drive("run", -1)
        except Exception as error:
            moved = on_failing_node(error, self.chain.nodes)
            if moved is error:
                raise
            raise moved from moved.__cause__
        finally:
            await self.__close()

    async def process_regular_node(self, node: FunctionNode) -> RegularOutput:
        output = await self.__drive("regular", self.__position(node.id))
        assert isinstance(output, RegularOutput)
        return output

    async def process_generator_node(self, node: GeneratorNode) -> GeneratorOutput:
        output = await self.__drive("generator", self.__position(node.id))
        assert isinstance(output, GeneratorOutput)
        return output

    def __position(self, node_id: NodeId) -> int:
        return next(
            index for index, node in enumerate(self.__nodes) if node.id == node_id
        )

    async def flush_broadcasts(self) -> None:
        await self.loop.run_in_executor(None, self.__run_engine.flush_broadcasts)
        await self.__close()

    async def __close(self) -> None:
        if not self.__closed:
            self.__closed = True
            await self.loop.run_in_executor(None, self.__run_engine.close)

    def resume(self) -> None:
        logger.debug("Resuming native executor %s", self.id)
        self.__run_engine.resume()

    def pause(self) -> None:
        logger.debug("Pausing native executor %s", self.id)
        self.__run_engine.pause()

    def kill(self) -> None:
        logger.debug("Killing native executor %s", self.id)
        self.__run_engine.kill()
        self.loop.call_soon_threadsafe(self.__stop)

    def __stop(self) -> None:
        if not self.__stopped.done():
            self.__stopped.set_result(None)
