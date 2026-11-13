from __future__ import annotations

import asyncio
import copy
import functools
import os
import sys
import threading
import time
import typing
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import NoneType
from typing import (
    Any,
    Callable,
    Collection,
    Iterable,
    List,
    Literal,
    Mapping,
    NewType,
    Sequence,
    Union,
)

import numpy as np
from sanic.log import logger

import navi
from api import (
    BaseInput,
    BaseOutput,
    BroadcastData,
    Collector,
    ExecutionOptions,
    Generator,
    InputId,
    IterationView,
    IteratorOutputInfo,
    IterOutputId,
    Lazy,
    NodeAttributedError,
    NodeContext,
    NodeData,
    NodeId,
    OutputId,
    SettingsParser,
    registry,
)
from chain.cache import CacheStrategy, OutputCache, StaticCaching, get_cache_strategies
from chain.chain import Chain, CollectorNode, FunctionNode, GeneratorNode, Node
from chain.input import EdgeInput, Input, InputMap
from events import (
    EventConsumer,
    ExecutionWarningEvent,
    InputsDict,
    NodeBroadcastData,
    NodeProgressUpdateEvent,
)
from nodes.impl import native_profile
from nodes.impl.execution_cleanup import (
    await_owned_node,
    run_cleanup_functions,
    run_cleanup_groups,
)
from nodes.impl.execution_scheduler import (
    LOCKED_DEVICE_NODES,
    LatestBroadcasts,
    NodeFlights,
    advance_iterator,
    may_go_unread,
    owned_gather,
    pure_cpu_dependency,
    pure_cpu_node,
)
from nodes.impl.item_window import (
    OUTPUT_UNRESOLVED,
    PATH_CHECK_FAILED,
    SOURCE_UNDER_OUTPUT,
    SOURCES_UNKNOWN,
    Candidate,
    ItemWindow,
    Phases,
    Prepared,
    SlotGenerator,
    Stored,
    affinity_cpus,
    engine_jobs,
    escaping_targets,
    memory_budget,
    node_phases,
    output_nbytes,
    read_ahead_blocker,
    serial_producer,
    window_size,
)
from nodes.utils.utils import split_file_path
from progress_controller import Aborted, ProgressController, ProgressToken
from util import combine_sets, timed_supplier, waiting

Output = List[object]


def collect_input_information(
    node: NodeData, inputs: list[object | Lazy[object]], enforced: bool = True
) -> InputsDict:
    try:
        input_dict: InputsDict = {}
        for raw, node_input in zip(inputs, node.inputs):
            value = raw
            if isinstance(value, Lazy) and value.has_value:
                value = value.value
            if isinstance(value, Lazy):
                input_dict[node_input.id] = {"type": "pending"}
                continue
            if not enforced:
                try:
                    value = node_input.enforce_(value)
                except Exception:
                    logger.error(
                        f"Error enforcing input {node_input.label} (id {node_input.id})",
                        exc_info=True,
                    )
            try:
                input_dict[node_input.id] = node_input.get_error_value(value)
            except Exception:
                logger.error(
                    f"Error getting error value for input {node_input.label} (id {node_input.id})",
                    exc_info=True,
                )
        return input_dict
    except Exception:
        logger.error("Error collecting input information.", exc_info=True)
        return {}


def enforce_inputs(
    inputs: list[object], node: NodeData, node_id: NodeId, ignored_inputs: list[InputId]
) -> list[object]:

    def enforce(i: BaseInput, value: object) -> object:
        if i.id in ignored_inputs:
            return None
        if i.lazy:
            if isinstance(value, Lazy):
                return Lazy(lambda: i.enforce_(value.value))
            return Lazy.ready(i.enforce_(value))
        if isinstance(value, Lazy):
            value = value.value
        return i.enforce_(value)

    try:
        enforced_inputs: list[object] = []
        for index, value in enumerate(inputs):
            enforced_inputs.append(enforce(node.inputs[index], value))
        return enforced_inputs
    except Exception as e:
        input_dict = collect_input_information(node, inputs, enforced=False)
        raise NodeExecutionError(node_id, node, str(e), input_dict) from e


def enforce_output(raw_output: object, node: NodeData) -> RegularOutput:
    l = len(node.outputs)
    output: Output
    if l == 0:
        assert raw_output is None, f"Expected all {node.name} nodes to return None."
        output = []
    elif l == 1:
        output = [raw_output]
    else:
        assert isinstance(raw_output, (tuple, list))
        output = list(raw_output)
        assert len(output) == l, (
            f"Expected all {node.name} nodes to have {l} output(s) but found {len(output)}."
        )
    for i, o in enumerate(node.outputs):
        output[i] = o.enforce_(output[i])
    return RegularOutput(output)


def enforce_generator_output(raw_output: object, node: NodeData) -> GeneratorOutput:
    l = len(node.outputs)
    generator_output = node.single_iterable_output
    partial: list[object] = [None] * l
    if l == len(generator_output.outputs):
        assert isinstance(raw_output, Generator), (
            "Expected the output to be a generator"
        )
        return GeneratorOutput(
            info=generator_output, generator=raw_output, partial_output=partial
        )
    assert l > len(generator_output.outputs)
    assert isinstance(raw_output, (tuple, list))
    iterator, *rest = raw_output
    assert isinstance(iterator, Generator), (
        "Expected the first tuple element to be a generator"
    )
    assert len(rest) == l - len(generator_output.outputs)
    for i, o in enumerate(node.outputs):
        if o.id not in generator_output.outputs:
            partial[i] = o.enforce_(rest.pop(0))
    return GeneratorOutput(
        info=generator_output, generator=iterator, partial_output=partial
    )


def run_node(
    node: NodeData,
    context: NodeContext,
    inputs: list[object],
    node_id: NodeId,
    prepared: Prepared | None = None,
) -> NodeOutput | CollectorOutput:
    if node.kind == "collector":
        ignored_inputs = node.single_iterable_input.inputs
    else:
        ignored_inputs = []
    enforced_inputs = enforce_inputs(inputs, node, node_id, ignored_inputs)
    try:
        if prepared is not None:
            raw_output = registered_phases(node).commit(enforced_inputs, prepared)
        elif node.node_context:
            raw_output = node.run(context, *enforced_inputs)
        else:
            raw_output = node.run(*enforced_inputs)
        if node.kind == "collector":
            assert isinstance(raw_output, Collector)
            return CollectorOutput(raw_output)
        if node.kind == "generator":
            return enforce_generator_output(raw_output, node)
        assert node.kind == "regularNode"
        return enforce_output(raw_output, node)
    except Aborted:
        raise
    except NodeExecutionError:
        raise
    except Exception as e:
        input_dict = collect_input_information(node, enforced_inputs)
        raise NodeExecutionError(node_id, node, str(e), input_dict) from e


def with_node_cleanups(job: Callable[[], _T], context: _ExecutorNodeContext) -> _T:
    """The job (a node's timed run), then the cleanups the node registered in `context`
    for after itself, outside its time as on the commit path."""
    try:
        return job()
    finally:
        run_cleanup_functions(
            context.node_cleanup_fns, preserve_error=sys.exc_info()[0] is not None
        )


def registered_phases(node: NodeData) -> Phases:
    phases = node_phases(node.schema_id)
    assert phases is not None, f"{node.schema_id} has no prepare and commit phases"
    return phases


def phase_inputs(node: NodeData) -> list[int]:
    """
    The indices of the inputs a prepare phase receives: a collector's iterated inputs, as on_iterate does, else all of them.
    """
    if node.kind == "collector":
        return [
            index
            for index, i in enumerate(node.inputs)
            if i.id in node.single_iterable_input.inputs
        ]
    return list(range(len(node.inputs)))


def prepare_node(
    node: NodeData, inputs: list[object], collector: Collector | None = None
) -> object | None:
    """
    A side-effect node's prepare phase: its inputs enforced as the node would receive them, a lazy input as the value it produces; a collector's prepare also receives its collector.
    """
    enforced = [
        node.inputs[index].enforce_(value)
        for index, value in zip(phase_inputs(node), inputs)
    ]
    prepare = registered_phases(node).prepare
    if node.kind == "collector":
        assert collector is not None, f"{node.schema_id}'s prepare has no collector"
        return prepare(collector, enforced)
    return prepare(enforced)


def run_collector_iterate(
    node: CollectorNode,
    inputs: list[object],
    collector: Collector,
    prepared: Prepared | None = None,
) -> None:
    iterable_input = node.data.single_iterable_input

    def get_partial_inputs(values: list[object]) -> list[object]:
        partial_inputs: list[object] = []
        index = 0
        for i in node.data.inputs:
            if i.id in iterable_input.inputs:
                partial_inputs.append(values[index])
                index += 1
            else:
                partial_inputs.append(None)
        return partial_inputs

    def iterated_information(values: list[object]) -> InputsDict:
        """The iterated inputs' error values, as given. The others are not this
        call's: their None placeholders would report None for a value they have,
        and enforcing them would log a traceback each (20 for one wrong frame into
        Save Video)."""
        information = collect_input_information(node.data, get_partial_inputs(values))
        return {
            key: value
            for key, value in information.items()
            if key in iterable_input.inputs
        }

    enforced_inputs: list[object] = []
    try:
        for i in node.data.inputs:
            if i.id in iterable_input.inputs:
                enforced_inputs.append(i.enforce_(inputs[len(enforced_inputs)]))
    except Exception as e:
        input_dict = iterated_information(inputs)
        raise NodeExecutionError(node.id, node.data, str(e), input_dict) from e
    input_value = (
        enforced_inputs[0] if len(enforced_inputs) == 1 else tuple(enforced_inputs)
    )
    try:
        if prepared is not None:
            raw_output = registered_phases(node.data).commit(
                collector, enforced_inputs, prepared
            )
        else:
            raw_output = collector.on_iterate(input_value)
        assert raw_output is None
    except Exception as e:
        input_dict = iterated_information(enforced_inputs)
        raise NodeExecutionError(node.id, node.data, str(e), input_dict) from e


def run_collector_complete(
    node: CollectorNode, collector: Collector, assigned: Sequence[Input]
) -> RegularOutput:
    """The collector's output (on_complete's, enforced); its error names the collector
    (loop_node_error), as on_iterate's does."""
    try:
        return enforce_output(collector.on_complete(), node.data)
    except Exception as error:
        failure = loop_node_error(node, assigned, error)
        if failure is error:
            raise
        raise failure from error


class _Timer:
    def __init__(self) -> None:
        self.duration: float = 0

    @contextmanager
    def run(self):
        start = time.monotonic()
        try:
            yield None
        finally:
            self.add_since(start)

    def add_since(self, start: float):
        self.duration += time.monotonic() - start


class _IterationTimer:
    def __init__(self, progress: ProgressController) -> None:
        self.times: list[float] = []
        self.progress = progress
        self._start_time = time.monotonic()
        self._start_paused = progress.time_paused
        self._last_time = self._start_time
        self._last_paused = self._start_paused

    @property
    def iterations(self) -> int:
        return len(self.times)

    def get_time_since_start(self) -> float:
        now = time.monotonic()
        paused = self.progress.time_paused
        current_paused = max(0, paused - self._start_paused)
        return now - self._start_time - current_paused

    def add(self):
        now = time.monotonic()
        paused = self.progress.time_paused
        current_paused = max(0, paused - self._last_paused)
        self.times.append(now - self._last_time - current_paused)
        self._last_time = now
        self._last_paused = paused


def compute_broadcast(output: Output, node_outputs: Iterable[BaseOutput]):
    data: dict[OutputId, BroadcastData | None] = {}
    types: dict[OutputId, navi.ExpressionJson | None] = {}
    for index, node_output in enumerate(node_outputs):
        try:
            value = output[index]
            if value is not None:
                data[node_output.id] = node_output.get_broadcast_data(value)
                types[node_output.id] = node_output.get_broadcast_type(value)
        except Exception as e:
            logger.error(f"Error broadcasting output: {e}", exc_info=True)
    return (data, types)


def compute_sequence_broadcast(
    generators: Iterable[Generator], node_iter_outputs: Iterable[IteratorOutputInfo]
):
    sequence_types: dict[IterOutputId, navi.ExpressionJson] = {}
    item_types: dict[OutputId, navi.ExpressionJson] = {}
    for g, iter_output in zip(generators, node_iter_outputs):
        try:
            sequence_types[iter_output.id] = iter_output.get_broadcast_sequence_type(g)
            for output_id, type in iter_output.get_broadcast_item_types(g).items():
                item_types[output_id] = type
        except Exception as e:
            logger.error(f"Error broadcasting output: {e}", exc_info=True)
    return (sequence_types, item_types)


class NodeExecutionError(Exception):
    def __init__(
        self, node_id: NodeId, node_data: NodeData, cause: str, inputs: InputsDict
    ):
        super().__init__(cause)
        self.node_id: NodeId = node_id
        self.node_data: NodeData = node_data
        self.inputs: InputsDict = inputs


def on_failing_node(error: Exception, nodes: Mapping[NodeId, Node]) -> Exception:
    """The error as the run reports it: a NodeAttributedError goes to the node it names
    (e.g. the node whose statement failed in a script that another node ran). It may
    arrive wrapped by the node that raised it, or by the generator whose iteration
    raised it (loop_node_error)."""
    wrapped = isinstance(error, NodeExecutionError)
    cause = error.__cause__ if wrapped else error
    if not isinstance(cause, NodeAttributedError) or cause.node_id is None:
        return error
    node = nodes.get(NodeId(cause.node_id))
    if node is None or (wrapped and node.id == error.node_id):
        return error
    moved = NodeExecutionError(node.id, node.data, str(cause), {})
    moved.__cause__ = cause
    return moved


def loop_node_error(
    node: Node, assigned: Sequence[Input], error: Exception, token: object = None
) -> Exception:
    """An error a loop's node raises outside its node's run, a generator's item (Load
    Images' undecodable file, Load Video's Start after End) or a collector's completion
    (Merge Spritesheet's sprite count), as the run reports it: the node's
    NodeExecutionError, so the error names the node, with the error as its cause and
    its message, after the file a split iterator's item token names (Load Images'
    (path, index)): among thousands of files, the failing one is otherwise unknown. Its
    inputs are its constants (an edge's value is not kept). A Stop, the end of the
    items, or an error a node already reports stays as it is."""
    if isinstance(error, (Aborted, StopIteration, NodeExecutionError)):
        return error
    message = str(error)
    if (
        isinstance(token, tuple)
        and len(token) == 2
        and isinstance(token[0], (str, Path))
        and isinstance(token[1], int)
    ):
        message = f"{token[0]}: {message}"
    values = [None if isinstance(i, EdgeInput) else i.value for i in assigned]
    edges = {
        i.id for i, a in zip(node.data.inputs, assigned) if isinstance(a, EdgeInput)
    }
    inputs = collect_input_information(node.data, values)
    failure = NodeExecutionError(
        node.id,
        node.data,
        message,
        {key: value for key, value in inputs.items() if key not in edges},
    )
    failure.__cause__ = error
    return failure


@dataclass(frozen=True)
class RegularOutput:
    output: Output


@dataclass(frozen=True)
class GeneratorOutput:
    info: IteratorOutputInfo
    generator: Generator
    partial_output: Output


@dataclass(frozen=True)
class CollectorOutput:
    collector: Collector


NodeOutput = Union[RegularOutput, GeneratorOutput]
ExecutionId = NewType("ExecutionId", str)


def _output_value(output: NodeOutput, index: int) -> object:
    if isinstance(output, GeneratorOutput):
        return output.partial_output[index]
    return output.output[index]


def _relative_path_input(i: object) -> bool:
    """
    Whether the input is a RelativePathInput (a writer's subdirectory or file name). Read from sys.modules: importing the inputs package loads torch, and with its module not loaded no such input exists.
    """
    module = sys.modules.get("nodes.properties.inputs.file_inputs")
    return module is not None and isinstance(i, module.RelativePathInput)


def run_warning(
    loop: asyncio.AbstractEventLoop, queue: EventConsumer
) -> Callable[[str], None]:
    """NodeContext.warn in a run. It holds no executor, whose contexts keep it: server.py
    collects nothing after a run, so a cycle would keep the run's outputs alive."""

    def warn(message: str) -> None:
        logger.warning(message)
        event: ExecutionWarningEvent = {
            "event": "execution-warning",
            "data": {"message": message},
        }
        loop.call_soon_threadsafe(queue.put, event)

    return warn


NODE_PROGRESS_SECONDS = 0.1
"""The least time between two of a node's own progress events: its bar repaints with each,
and every repaint on screen interrupts the GPU's work."""


def run_progress(
    loop: asyncio.AbstractEventLoop, queue: EventConsumer, node_id: NodeId
) -> Callable[[float], None]:
    """NodeContext.set_progress's event in a run, without items (total 0). It holds no
    executor, as run_warning. The first update of a pass (a value not above the last) only
    starts its clock and its last (1) is left to the node's finish, so a pass shorter than
    NODE_PROGRESS_SECONDS, a video's frame or an untiled image, sends nothing."""
    sent = 0.0
    last = 1.0

    def send(progress: float) -> None:
        nonlocal sent, last
        now = time.monotonic()
        if progress <= last:
            sent = now
        elif progress < 1 and now - sent >= NODE_PROGRESS_SECONDS:
            sent = now
            event: NodeProgressUpdateEvent = {
                "event": "node-progress",
                "data": {
                    "nodeId": node_id,
                    "progress": progress,
                    "index": 0,
                    "total": 0,
                    "eta": 0,
                },
            }
            loop.call_soon_threadsafe(queue.put, event)
        last = progress

    return send


class _ExecutorNodeContext(NodeContext):
    def __init__(
        self,
        progress: ProgressToken,
        settings: SettingsParser,
        storage_dir: Path,
        node_id: NodeId,
        warn: Callable[[str], None],
        send_progress: Callable[[float], None],
    ) -> None:
        super().__init__()
        self.progress = progress
        self.__settings = settings
        self._storage_dir = storage_dir
        self._node_id = node_id
        self.__warn = warn
        self.__send_progress = send_progress
        self.chain_cleanup_fns: set[Callable[[], None]] = set()
        self.node_cleanup_fns: set[Callable[[], None]] = set()
        # Set while a generator loop runs this node on its commit path.
        self.iteration_view: _IterationView | None = None
        # The chosen item of the loop the node belongs to (NodeContext.chosen_item).
        self.chosen: int | None = None

    @property
    def iteration(self) -> IterationView | None:
        return self.iteration_view

    @property
    def chosen_item(self) -> int | None:
        return self.chosen

    @property
    def aborted(self) -> bool:
        return self.progress.aborted

    @property
    def paused(self) -> bool:
        time.sleep(0)
        return self.progress.paused

    def set_progress(self, progress: float) -> None:
        self.check_aborted()
        self.__send_progress(progress)

    @property
    def settings(self) -> SettingsParser:
        """
        Returns the settings of the current node execution.
        """
        return self.__settings

    @property
    def node_id(self) -> NodeId:
        return self._node_id

    @property
    def storage_dir(self) -> Path:
        return self._storage_dir

    def warn(self, message: str) -> None:
        self.__warn(message)

    def add_cleanup(
        self, fn: Callable[[], None], after: Literal["node", "chain"] = "chain"
    ) -> None:
        if after == "chain":
            self.chain_cleanup_fns.add(fn)
        elif after == "node":
            self.node_cleanup_fns.add(fn)
        else:
            raise ValueError(f"Unknown cleanup type: {after}")


_T = typing.TypeVar("_T")

LAZY_POLL_SECONDS = 0.05
"""How often a thread waiting on a lazy input checks for Stop and pool starvation."""
LAZY_STARVED_SECONDS = 1.0
"""How long every pool thread may wait on a lazy input before the wait fails."""


class _LazyWaits:
    """
    The threads of one executor pool that wait on a lazy input, for every executor
    sharing it (a /run and its previews). A node reading a lazy input holds its pool
    thread until the input's nodes ran, on other pool threads; once every thread waits,
    nothing runs them.
    """

    def __init__(self, threads: int) -> None:
        self.threads = threads
        self.__lock = threading.Lock()
        self.__waiting = 0
        self.__full_since: float | None = None

    @contextmanager
    def waiting(self):
        with self.__lock:
            self.__waiting += 1
            if self.__waiting >= self.threads and self.__full_since is None:
                self.__full_since = time.monotonic()
        try:
            yield
        finally:
            with self.__lock:
                self.__waiting -= 1
                if self.__waiting < self.threads:
                    self.__full_since = None

    def starved(self) -> bool:
        """Whether every pool thread has waited on a lazy input for LAZY_STARVED_SECONDS."""
        with self.__lock:
            return (
                self.__full_since is not None
                and time.monotonic() - self.__full_since >= LAZY_STARVED_SECONDS
            )


_lazy_waits: weakref.WeakKeyDictionary[ThreadPoolExecutor, _LazyWaits] = (
    weakref.WeakKeyDictionary()
)


class _LoopRun:
    """
    One run of a generator group's loop, as its nodes' iteration views see it: the slot the commit path runs, the range's first slot (context items before it) and index base, and the group's item window decision, made at the range's second slot or by a look-ahead before it.
    """

    def __init__(
        self,
        generators: set[NodeId],
        iterated: set[NodeId],
        open_window: Callable[
            [int], typing.Awaitable[tuple[ItemWindow | None, str | None]]
        ],
        first: int = 0,
        base: int = 0,
        unread: frozenset[NodeId] = frozenset(),
    ) -> None:
        self.slot = 0
        self.generators = generators
        self.iterated = iterated
        self.open_window = open_window  # Executor.__open_window, given the stops
        # Slots before `first` are context items; the slot `first` is item `base`.
        self.first = first
        self.base = base
        # The temporal nodes whose output no node reads on a context item.
        self.unread = unread
        self.decided = False
        self.deciding: asyncio.Task[None] | None = None
        # pure_in_loop per temporal node on a pure schema, made at its first run on the commit path.
        self.pure: dict[NodeId, bool] = {}
        # The read-ahead blocker that set K = 1, and the look-ahead nodes warned of it.
        self.blocker: str | None = None
        self.warned: set[NodeId] = set()
        self.lineages: dict[NodeId, NodeId | None] = {}
        self.views: dict[NodeId, _IterationView] = {}


class _IterationView(IterationView):
    """
    A node's iteration view for one loop run: its index counts the run's slots from the range's first one, numbered by the index base; a look-ahead runs on the event loop while the node's worker thread waits for it.
    """

    def __init__(
        self,
        run: _LoopRun,
        context: _ExecutorNodeContext,
        node: FunctionNode,
        sources: tuple[str | None, ...],
        lookahead: Callable[[int], typing.Coroutine[typing.Any, typing.Any, int]],
        peek: Callable[
            [int, int, int], typing.Coroutine[typing.Any, typing.Any, object]
        ],
        loop: asyncio.AbstractEventLoop,
        base: _IterationView | None = None,
        slot: int | None = None,
    ) -> None:
        self.__run = run
        self.__context = context
        self.__node = node
        self.__sources = sources
        self.__lookahead = lookahead
        self.__peek = peek
        self.__loop = loop
        self.__state: object | None = None
        self.__base = base  # the view whose state slot this one shares
        self.__slot = slot  # the item the node is computed ahead at

    def at(self, slot: int, context: _ExecutorNodeContext) -> _IterationView:
        """
        This view as the node computed ahead at a later item sees it: that item's index, this view's state slot, look-aheads from that item.
        """
        return _IterationView(
            self.__run,
            context,
            self.__node,
            self.__sources,
            self.__lookahead,
            self.__peek,
            self.__loop,
            self,
            slot,
        )

    @property
    def index(self) -> int:
        return self.__run.base + self.__at() - self.__run.first

    @property
    def context(self) -> bool:
        return self.__at() < self.__run.first

    @property
    def unread(self) -> bool:
        return self.context and self.__node.id in self.__run.unread

    @property
    def source(self) -> str:
        return next(source for source in self.__sources if source is not None)

    def iterated_by(self, input_id: int) -> str | None:
        return self.__sources[self.__input_index(input_id)]

    @property
    def state(self) -> object | None:
        if self.__base is not None:
            return self.__base.state
        return self.__state

    @state.setter
    def state(self, value: object | None) -> None:
        if self.__base is not None:
            self.__base.state = value
        else:
            self.__state = value

    def lookahead(self, input_id: int) -> int:
        return self.__wait(self.__lookahead(self.__input_index(input_id)))

    def peek(self, input_id: int, offset: int) -> object | None:
        if offset < 1:
            raise ValueError(f"A look-ahead offset must be at least 1, not {offset}.")
        index = self.__input_index(input_id)
        self.__context.check_aborted()
        value = self.__wait(self.__peek(self.__at(), index, offset))
        self.__context.check_aborted()
        if value is None:
            return None
        try:
            return self.__node.data.inputs[index].enforce_(value)
        except Exception:
            # The item fails at this input when it comes, with this error (run_node).
            return None

    def close(self) -> None:
        """
        The loop run ended: drop the state and detach the view from the node's context.
        """
        self.__state = None
        self.__context.iteration_view = None

    def __at(self) -> int:
        """The slot of the item the node runs for: the committing one, or the later one it is computed ahead at."""
        return self.__run.slot if self.__slot is None else self.__slot

    def __input_index(self, input_id: int) -> int:
        for index, i in enumerate(self.__node.data.inputs):
            if i.id == input_id:
                return index
        raise ValueError(f"{self.__node.data.name} has no input {input_id}.")

    def __wait(self, coroutine: typing.Coroutine[typing.Any, typing.Any, _T]) -> _T:
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:  # a worker thread
            running = None
        if running is self.__loop:
            coroutine.close()
            raise RuntimeError(
                "A look-ahead waits for the event loop: call it from the node's own call."
            )
        future = asyncio.run_coroutine_threadsafe(coroutine, self.__loop)
        # Later items' values come from other work: not this node's time.
        with waiting():
            return future.result()


@dataclass(frozen=True)
class _ItemCache:
    """
    The cache test of one later item: its generator values and stored results, and the shared static cache for nodes that are not iterated. The live cache's iterated entries belong to the committing item.
    """

    available: set[NodeId]
    iterated: set[NodeId]
    shared: OutputCache[NodeOutput]

    def has(self, node_id: NodeId) -> bool:
        return node_id in self.available or (
            node_id not in self.iterated and self.shared.has(node_id)
        )


@dataclass(frozen=True)
class _ItemView:
    chain: Chain
    inputs: InputMap
    node_cache: _ItemCache


# Into Directory resolves the on-disk spelling of a folder that earlier items' saves may create, so running it ahead could change its output.
_NOT_SPECULATED: frozenset[str] = frozenset({"chainner:utility:into_directory"})


def folder_sources(
    chain: Chain,
    inputs: InputMap,
    writers: Iterable[Node],
    iterated: set[NodeId],
    computed: Callable[[NodeId], bool],
) -> list[NodeId]:
    """
    The nodes that compute a loop's writers' folders (the upstream of a regular writer's directory and relative path inputs, as Save Image's gather would run it at its turn) that its item window decision runs first, so that the read-ahead check knows those folders (__writer_directories): not iterated, not computed yet, and pure CPU but not Into Directory, as is every node they need that is not computed yet. In the writers' id order; a collector's folders are known from its construction.
    """
    sources: list[NodeId] = []
    for writer in sorted(writers, key=lambda n: n.id):
        if not isinstance(writer, FunctionNode):
            continue
        assigned = inputs.get(writer.id)
        for index, i in enumerate(writer.data.inputs):
            source = assigned[index]
            if (
                (i.kind == "directory" or _relative_path_input(i))
                and isinstance(source, EdgeInput)
                and source.id not in iterated
                and source.id not in sources
                and _computable_folder(chain, inputs, source.id, computed)
            ):
                sources.append(source.id)
    return sources


def _computable_folder(
    chain: Chain, inputs: InputMap, node_id: NodeId, computed: Callable[[NodeId], bool]
) -> bool:
    """
    Whether folder_sources runs the node: it is not computed, and it and every node it needs that is not computed are pure CPU (pure_cpu_node) and not Into Directory.
    """
    if computed(node_id):
        return False
    pending, seen = [node_id], set()
    while pending:
        current = pending.pop()
        if current in seen or computed(current):
            continue
        seen.add(current)
        data = chain.nodes[current].data
        if not pure_cpu_node(data) or data.schema_id in _NOT_SPECULATED:
            return False
        pending.extend(i.id for i in inputs.get(current) if isinstance(i, EdgeInput))
    return True


def writer_directories(
    chain: Chain,
    inputs: InputMap,
    writers: Iterable[Node],
    generator_outputs: Mapping[NodeId, GeneratorOutput],
    iterated: set[NodeId],
    resolved: Callable[[Node, int, EdgeInput], object],
) -> tuple[list[tuple[NodeId, Path | None]], list[tuple[NodeId, Path, Path]]]:
    """
    Returns the base directory of every directory input of the given writers, each followed by the folder its file goes to when the relative path inputs after it (subdirectory, file name) are known: the directory joined with them, `..` included, the file name's own folder last. A known part is a constant (an unset one adds nothing) or a resolved value; the join stops at the first unknown part. Also returns (writer, directory, target folder) for each such folder.

    The blocker checks the directory and the target folder alike, so a target outside the directory can only set K = 1 in more cases. Nothing is computed: an input fed by an iterated output or node, a value `resolved` does not have (None: missing from the cache), or a value that is not a path is unresolved (None). `resolved(writer, index, input)` reads the other inputs: a collector's from its construction, a regular writer's from the cache.
    """
    directories: list[tuple[NodeId, Path | None]] = []
    targets: list[tuple[NodeId, Path, Path]] = []

    def add_target(writer: NodeId, base: Path | None, parts: list[str], complete: bool):
        # The file name's folder when every part is known, else the folder of the known ones.
        if base is None or not parts:
            return
        joined = base.joinpath(*parts)
        folder = joined.parent if complete else joined
        directories.append((writer, folder))
        targets.append((writer, base, folder))

    for node in sorted(writers, key=lambda n: n.id):
        assigned_inputs = inputs.get(node.id)
        base: Path | None = None
        parts: list[str] = []
        complete = False  # every relative path input after the directory is known
        for index, i in enumerate(node.data.inputs):
            relative = _relative_path_input(i)
            if i.kind != "directory" and not relative:
                continue
            node_input = assigned_inputs[index]
            value: object
            if not isinstance(node_input, EdgeInput):
                value = node_input.value
            elif node_input.id in generator_outputs:
                source = chain.nodes[node_input.id]
                value = (
                    None
                    if source.data.outputs[node_input.index].id
                    in source.data.single_iterable_output.outputs
                    else _output_value(
                        generator_outputs[node_input.id], node_input.index
                    )
                )
            elif node_input.id in iterated:
                value = None
            else:
                value = resolved(node, index, node_input)
            if not relative:
                add_target(node.id, base, parts, complete)
                base = Path(value) if isinstance(value, (str, Path)) else None
                directories.append((node.id, base))
                parts, complete = [], True
            elif complete and isinstance(value, str):
                parts.append(value)
            elif not (value is None and not isinstance(node_input, EdgeInput)):
                complete = False  # unknown: the join stops before it
        add_target(node.id, base, parts, complete)
    return directories, targets


def escaping_target_warning(
    chain: Chain, writer: NodeId, folder: Path
) -> ExecutionWarningEvent:
    """The run warning for a writer whose target folder escapes its directory (escaping_targets); logs it."""
    message = f"{chain.nodes[writer].data.name} writes to {folder}, outside its chosen folder."
    logger.warning(message)
    return {"event": "execution-warning", "data": {"message": message}}


def look_ahead_off_warning(
    chain: Chain, node_id: NodeId, blocker: str
) -> ExecutionWarningEvent:
    """
    The run warning for a node that looks ahead in a loop whose read-ahead blocker set K = 1, so it runs without its look-ahead; logs it.
    """
    kind, _, subject = blocker.partition(": ")
    name = chain.nodes[node_id].data.name
    if kind == PATH_CHECK_FAILED:
        reason = f"The output folder could not be checked ({subject})"
    else:
        other = chain.nodes[NodeId(subject)].data.name
        reason = {
            SOURCES_UNKNOWN: f"The files {other} reads are unknown",
            OUTPUT_UNRESOLVED: f"{other}'s output folder is computed after {name}",
            SOURCE_UNDER_OUTPUT: f"{other}'s output folder may overlap the source folder",
        }[kind]
    message = f"{reason}, so {name} runs without its look-ahead: it uses only the current and earlier frames."
    logger.warning(message)
    return {"event": "execution-warning", "data": {"message": message}}


_UNKNOWN = object()
LOAD_IMAGES = "chainner:image:load_images"
IMAGES_MODULE = "packages.chaiNNer_standard.image.batch_processing.load_images"


def _reached(
    start: Iterable[NodeId], step: Callable[[NodeId], Iterable[NodeId]]
) -> set[NodeId]:
    reached = set(start)
    pending = list(reached)
    while pending:
        for node_id in step(pending.pop()):
            if node_id not in reached:
                reached.add(node_id)
                pending.append(node_id)
    return reached


def _upstream(chain: Chain, start: Iterable[NodeId]) -> set[NodeId]:
    return _reached(start, lambda n: [e.source.id for e in chain.edges_to(n)])


def _scalar_node(data: NodeData) -> bool:
    """A node the collision check may run ahead of a file: pure, and no output of it is an image."""
    return (
        pure_cpu_node(data)
        and not data.node_context
        and all(o.associated_type is not np.ndarray for o in data.outputs)
    )


@dataclass(frozen=True)
class _LoopItems:
    """A loop's items as the collision check compares them: its generator, the nodes that run once per item, the
    items' source paths in the loop's order, and `named`, the generator's outputs that follow from an item's
    (position, path) alone."""

    node: Node
    per_item: set[NodeId]
    paths: Sequence[Path]
    named: Callable[[int, Path], dict[OutputId, object]]


@dataclass
class _FileScalars:
    """The strings, paths and numbers of one item's run that follow from its path, before it runs. The generator's own
    outputs among them are `named`; a value nothing here knows is _UNKNOWN."""

    chain: Chain
    loop: _LoopItems
    computed: dict[NodeId, NodeOutput]
    named: dict[OutputId, object]
    outputs: dict[NodeId, list[object]] = field(default_factory=dict)

    def input(self, node_id: NodeId, i: BaseInput) -> object:
        edge = self.chain.edge_to(node_id, i.id)
        if edge is None:
            value = self.chain.inputs.get(node_id, i.id)
        elif edge.source.id == self.loop.node.id:
            value = self.named.get(edge.source.output_id, _UNKNOWN)
        else:
            source = self.chain.nodes[edge.source.id].data.outputs
            index = next(
                n for n, o in enumerate(source) if o.id == edge.source.output_id
            )
            if edge.source.id in self.loop.per_item:
                results = self.outputs.get(edge.source.id)
                value = _UNKNOWN if results is None else results[index]
            else:
                output = self.computed.get(edge.source.id)
                value = _UNKNOWN if output is None else _output_value(output, index)
        return (
            value if isinstance(value, (str, Path, int, float, NoneType)) else _UNKNOWN
        )

    def run(self, node_id: NodeId) -> None:
        """Runs a scalar node on this item's values, if it knows them all. Its error leaves its output unknown: this
        item's run reports it, where the node runs."""
        data = self.chain.nodes[node_id].data
        inputs = [self.input(node_id, i) for i in data.inputs]
        if any(value is _UNKNOWN for value in inputs):
            return
        try:
            enforced = enforce_inputs(inputs, data, node_id, [])
            self.outputs[node_id] = enforce_output(data.run(*enforced), data).output
        except Exception as error:
            logger.warning(
                "%s, on an item's values, before the run: %s",
                data.name,
                error,
                exc_info=error,
            )


def _colliding_files(
    chain: Chain, loops: list[_LoopItems], computed: dict[NodeId, NodeOutput]
) -> tuple[NodeId, Path, Path] | None:
    """A writer of the loops' items and two items that would give it one path, where the later one would replace the
    earlier one's output; None if there are none. It compares the final paths, for each item: the generator's outputs
    that follow from the item's path, the shared nodes' values and the scalar nodes over those, run here. Two writers
    also have to agree on their dropdowns, which hold the format. A path that reads anything else (an image's content,
    FPS, a zipped generator's output) is not compared: that may tell the items apart. Nor is one without a file name
    input: a folder alone does not say which files go there. Two writers on one item, and an item that writes over its
    own source, are that chain's own choice."""
    order = list(reversed(chain.topological_order()))
    seen: dict[tuple[object, ...], tuple[NodeId, Path]] = {}
    for loop in loops:
        writers: list[tuple[NodeId, list[BaseInput], list[object]]] = []
        for node_id in order:
            writer = chain.nodes[node_id]
            data = writer.data
            if (
                node_id in loop.per_item
                and (
                    isinstance(writer, CollectorNode)
                    or (isinstance(writer, FunctionNode) and writer.has_side_effects())
                )
                and any(_relative_path_input(i) for i in data.inputs)
            ):
                path_inputs = [
                    i
                    for i in data.inputs
                    if i.kind == "directory" or _relative_path_input(i)
                ]
                choices = [
                    chain.inputs.get(node_id, i.id)
                    for i in data.inputs
                    if i.kind == "dropdown"
                ]
                writers.append((node_id, path_inputs, choices))
        if not writers:
            continue
        feeding = _upstream(
            chain,
            [
                edge.source.id
                for node_id, path_inputs, _ in writers
                for edge in chain.edges_to(node_id)
                if edge.target.input_id in {i.id for i in path_inputs}
            ],
        )
        steps = [
            node_id
            for node_id in order
            if node_id in feeding
            and node_id in loop.per_item
            and _scalar_node(chain.nodes[node_id].data)
        ]
        for position, path in enumerate(loop.paths):
            scalars = _FileScalars(chain, loop, computed, loop.named(position, path))
            for node_id in steps:
                scalars.run(node_id)
            for node_id, path_inputs, choices in writers:
                parts = [scalars.input(node_id, i) for i in path_inputs]
                if any(part is _UNKNOWN for part in parts):
                    continue
                try:
                    enforced = [i.enforce_(v) for i, v in zip(path_inputs, parts)]
                    final = os.path.join(*(p for p in enforced if p is not None))
                except Exception as error:
                    logger.warning(
                        "%s's path, on an item's values, before the run: %s",
                        chain.nodes[node_id].data.name,
                        error,
                        exc_info=error,
                    )
                    continue
                key = (
                    chain.nodes[node_id].schema_id,
                    *choices,
                    os.path.normcase(os.path.normpath(final)),
                )
                first_loop, first = seen.setdefault(key, (loop.node.id, path))
                if first != path or first_loop != loop.node.id:
                    return node_id, first, path
    return None


def _collision_message(
    chain: Chain, collision: tuple[NodeId, Path, Path], advice: str
) -> str:
    writer, first, second = collision
    return (
        f"{chain.nodes[writer].data.name} would write {first} and {second} to one file:"
        f" its folder and file name do not tell them apart. {advice}"
    )


def check_loop_outputs(
    chain: Chain,
    generators: Iterable[tuple[Node, GeneratorOutput]],
    per_item: set[NodeId],
) -> None:
    """Refuses a Load Images loop, before its first item, two of whose items would give a writer one path
    (_colliding_files): the later would replace the earlier's output, or a source not read yet. Only constants and the
    items' own values are compared, as both run engines know them then."""
    loops: list[_LoopItems] = []
    for node, output in generators:
        paths = output.generator.source_paths
        if node.schema_id != LOAD_IMAGES or not paths:
            continue
        module = sys.modules[IMAGES_MODULE]
        directory = Path(
            typing.cast(Path, output.partial_output[module.DIRECTORY_INDEX])
        )
        base = output.generator.index_base

        def named(
            position: int,
            path: Path,
            module: Any = module,
            directory: Path = directory,
            base: int = base,
        ) -> dict[OutputId, object]:
            return {
                module.DIRECTORY: directory,
                module.SUBDIRECTORY: os.path.relpath(path.parent, directory),
                module.NAME: split_file_path(path)[1],
                module.INDEX: base + position,
            }

        loops.append(_LoopItems(node, per_item, paths, named))
    collision = _colliding_files(chain, loops, {}) if loops else None
    if collision is not None:
        # Named after the writer, whichever node the run engine was at (on_failing_node).
        raise NodeAttributedError(
            _collision_message(
                chain,
                collision,
                "Connect Load Images' Name to its file name and its Subdirectory Path"
                " to its subdirectory, or narrow the glob expression.",
            ),
            collision[0],
        )


def _refuse(
    chain: Chain,
    node_id: NodeId,
    generator: NodeId,
    ended: NodeId,
    per_item: dict[NodeId, set[NodeId]],
    members: list[NodeId],
) -> typing.NoReturn:
    """refuse_reads_across_loops' error: on the node, its first input an item of the generator reaches it through."""
    node = chain.nodes[node_id]
    iterated = chain.nodes[generator].data.single_iterable_output.outputs
    edge = next(
        edge
        for edge in sorted(chain.edges_to(node_id), key=lambda e: e.target.input_id)
        if edge.source.id in per_item[generator]
        or (edge.source.id == generator and edge.source.output_id in iterated)
    )
    label = next(i.label for i in node.data.inputs if i.id == edge.target.input_id)
    later = next(
        (
            other
            for other in members
            if node_id in per_item[other] and ended in _upstream(chain, [other])
        ),
        None,
    )
    names = {key: chain.nodes[key].data.name for key in (generator, ended)}
    advice = (
        f"Take {label} from the outputs of {chain.nodes[later].data.name} instead, or connect a value that does not change per item."
        if later is not None
        else f"Connect a value to {label} that does not change per item."
    )
    raise loop_node_error(
        node,
        InputMap.from_chain(chain).get(node_id),
        ValueError(
            f"{label} of {node.data.name} comes from the items of {names[generator]}, one value per item, but {node.data.name} also depends on {names[ended]}, which has its result only after the loop of {names[generator]} has ended. {advice}"
        ),
    )


def refuse_reads_across_loops(chain: Chain) -> None:
    """
    Refuses, before anything runs, a node that runs once per item of a generator's loop (an iterated output reaches it) and also depends on a collector of that loop, which has its result only once the loop has ended: no order of the run gives it both (it would fail as "A collector was not run before another node needed it"). The error names the node, the input the item's value reaches it through, the collector and, when a later loop gives the node its items, that loop's generator.
    """
    order = chain.topological_order()
    position = {node_id: index for index, node_id in enumerate(order)}
    per_item: dict[NodeId, set[NodeId]] = {}
    collectors: dict[NodeId, set[NodeId]] = {}
    for generator_id in order:
        node = chain.nodes[generator_id]
        if not isinstance(node, GeneratorNode):
            continue
        iterated = node.data.single_iterable_output.outputs
        reached: set[NodeId] = set()
        ends: set[NodeId] = set()
        pending = [
            edge.target.id
            for edge in chain.edges_from(node.id)
            if edge.source.output_id in iterated
        ]
        while pending:
            node_id = pending.pop()
            target = chain.nodes[node_id]
            if isinstance(target, CollectorNode):
                ends.add(node_id)
            elif isinstance(target, FunctionNode) and node_id not in reached:
                reached.add(node_id)
                pending.extend(edge.target.id for edge in chain.edges_from(node_id))
        per_item[node.id], collectors[node.id] = reached, ends
    # A loop: the generators whose items reach a common node, which the run zips.
    for loop in combine_sets(
        [{g, *reached, *collectors[g]} for g, reached in per_item.items()]
    ):
        members = [g for g in order if g in loop and g in per_item]
        loop_ends = set().union(*(collectors[g] for g in members))
        # Sources first: the node where an item first meets the loop's collector.
        for node_id in reversed(order):
            needed = _upstream(chain, [e.source.id for e in chain.edges_to(node_id)])
            for generator in members:
                if node_id not in per_item[generator] | collectors[generator]:
                    continue
                # A collector the generator itself reads is the input of a later loop.
                ended = sorted(
                    (loop_ends & needed) - _upstream(chain, [generator]),
                    key=position.__getitem__,
                )
                if ended:
                    _refuse(chain, node_id, generator, ended[0], per_item, members)


def shown_alone(chain: Chain, node: Node, looped: set[NodeId]) -> bool:
    """Whether a generator that feeds no sink (outside `looped`) runs for what it shows,
    its items unread: one with a seek bar whose input is connected (Clip Frames behind
    its clip), whose Preview and frame count nothing else renders. Load Video alone
    renders them by itself; folder loaders and the rest stay skipped, as before."""
    return (
        isinstance(node, GeneratorNode)
        and node.id not in looped
        and any(i.kind == "video-seek" for i in node.data.inputs)
        and len(chain.edges_to(node.id)) > 0
    )


def generator_loops(
    chain: Chain, gens_by_outs: dict[NodeId, set[NodeId]]
) -> list[list[GeneratorNode]]:
    """
    The chain's loops from each sink's generators: the generators that share sinks, in the chain's order. combine_sets' order follows the number of passes it makes, so a loop whose sinks need another loop's collector moves after that loop and every other order stays. Of loops that need each other the first runs, and the reader of the other's collector reports it.
    """
    groups = combine_sets(list(gens_by_outs.values()))
    loop_of = {
        sink: next(index for index, group in enumerate(groups) if generators <= group)
        for sink, generators in gens_by_outs.items()
    }
    needs: list[set[int]] = [set() for _ in groups]
    for sink, loop in loop_of.items():
        pending, seen = [sink], {sink}
        while pending:
            for edge in chain.edges_to(pending.pop()):
                source = edge.source.id
                if source in seen:
                    continue
                seen.add(source)
                if not isinstance(chain.nodes[source], CollectorNode):
                    pending.append(source)
                elif source in loop_of and loop_of[source] != loop:
                    needs[loop].add(loop_of[source])
    loops: list[list[GeneratorNode]] = []
    waiting = list(range(len(groups)))
    while waiting:
        ready = next((i for i in waiting if needs[i].isdisjoint(waiting)), waiting[0])
        waiting.remove(ready)
        # In the chain's order: a set's order changes with the hash seed.
        loops.append(
            [
                node
                for node in chain.nodes.values()
                if node.id in groups[ready] and isinstance(node, GeneratorNode)
            ]
        )
    return loops


def temporal_nodes(
    chain: Chain, iterated: Collection[NodeId], sinks: Iterable[Node]
) -> list[FunctionNode]:
    """
    A loop's nodes without side effects that declare a temporal context (NodeData.temporal_context) and feed one of its sinks, so the loop runs them, in the chain's sink-first order.
    """
    ancestors: set[NodeId] = set()
    pending = [node.id for node in sinks]
    while pending:
        for edge in chain.edges_to(pending.pop()):
            if edge.source.id not in ancestors:
                ancestors.add(edge.source.id)
                pending.append(edge.source.id)
    temporal: list[FunctionNode] = []
    for node_id in chain.topological_order():
        node = chain.nodes[node_id]
        if (
            node_id in ancestors
            and node_id in iterated
            and isinstance(node, FunctionNode)
            and (not node.has_side_effects())
            and node.data.temporal_context is not None
        ):
            temporal.append(node)
    return temporal


def eager_ancestors(
    chain: Chain, inputs: InputMap, sinks: Iterable[Node]
) -> set[NodeId]:
    """
    The ancestors the sinks reach without a lazy input their node may leave unread (may_go_unread: a Conditional's other branch, Save Image's image with Skip existing files).
    """
    eager: set[NodeId] = set()
    pending = [node.id for node in sinks]
    while pending:
        target = pending.pop()
        data = chain.nodes[target].data
        unread = {
            i.id
            for index, i in enumerate(data.inputs)
            if i.lazy and may_go_unread(data, inputs.get(target), index)
        }
        for edge in chain.edges_to(target):
            if edge.target.input_id not in unread and edge.source.id not in eager:
                eager.add(edge.source.id)
                pending.append(edge.source.id)
    return eager


def reads_context(generator: Generator) -> bool:
    """
    Whether the generator may read context items: it reads part of its source (Generator.context) and has items. Only then are a loop's known gates resolved before its items.
    """
    return generator.context is not None and generator.expected_length > 0


def known_gates(
    chain: Chain,
    inputs: InputMap,
    iterated: Collection[NodeId],
    sinks: Iterable[Node],
    temporal: Iterable[FunctionNode],
) -> list[tuple[FunctionNode, int]]:
    """
    The gated temporal nodes (NodeData.temporal_if) whose gate value is known before the loop, with the gate's input position: a constant or an input fed from outside the loop, of a node the sinks read eagerly (eager_ancestors), so the gate's source would run anyway.
    """
    eager = eager_ancestors(chain, inputs, sinks)
    gates: list[tuple[FunctionNode, int]] = []
    for node in temporal:
        gate = node.data.temporal_if
        if gate is None or node.id not in eager:
            continue
        position = gate_position(node.data)
        assigned = inputs.get(node.id)[position]
        if not (isinstance(assigned, EdgeInput) and assigned.id in iterated):
            gates.append((node, position))
    return gates


def gate_position(data: NodeData) -> int:
    """The position of the gated node's gate input (NodeData.temporal_if)."""
    assert data.temporal_if is not None
    return next(
        i
        for i, node_input in enumerate(data.inputs)
        if node_input.id == data.temporal_if[0]
    )


def temporal_applies(
    node: Node, gates: Mapping[NodeId, object], inputs: InputMap
) -> bool:
    """
    Whether the temporal node's context applies: it has no gate, its gate's value is not known before the loop (not in `gates`, the known gates' values), or the gate's predicate is true for it. The predicate's error is the node's (loop_node_error).
    """
    gate = node.data.temporal_if
    if gate is None or node.id not in gates:
        return True
    try:
        return gate[1](gates[node.id])
    except Exception as error:
        failure = loop_node_error(node, inputs.get(node.id), error)
        if failure is error:
            raise
        raise failure from error


def temporal_pure(data: NodeData) -> bool:
    """A temporal node on a pure schema (a locked-device one): in a loop, pure only as pure_in_loop decides."""
    return pure_cpu_node(data) and data.temporal_context is not None


def pure_in_loop(
    node: Node, inputs: InputMap, iterated: Collection[NodeId], value: object
) -> bool:
    """
    Whether a temporal node on a pure schema (temporal_pure) runs as a pure one for the rest of the loop run, decided at its first run there from its gate input's value then (`value`): it needs its iteration view wherever it runs, so it is pure only while its context does not apply: its gate known (a constant or an input fed from outside the loop) and judged false. The predicate's error is the node's (temporal_applies).
    """
    gate = node.data.temporal_if
    if gate is None:
        return False
    assigned = inputs.get(node.id)[gate_position(node.data)]
    if not isinstance(assigned, EdgeInput):
        value = assigned.value
    elif assigned.id in iterated:
        return False
    return not temporal_applies(node, {node.id: value}, inputs)


def context_nodes(
    chain: Chain, iterated: Collection[NodeId], temporal: Sequence[Node]
) -> tuple[frozenset[NodeId], frozenset[NodeId]]:
    """
    The loop's nodes that run for a context item: its temporal nodes whose context applies and the nodes they need; and those temporal nodes none of them reads (IterationView.unread).
    """
    needed: set[NodeId] = set()
    pending = [node.id for node in temporal]
    while pending:
        node_id = pending.pop()
        if (
            node_id not in needed
            and node_id in iterated
            and isinstance(chain.nodes[node_id], FunctionNode)
        ):
            needed.add(node_id)
            pending.extend(edge.source.id for edge in chain.edges_to(node_id))
    unread = frozenset(
        node.id
        for node in temporal
        if all(edge.target.id not in needed for edge in chain.edges_from(node.id))
    )
    return frozenset(needed), unread


def context_wanted(
    chain: Chain, iterated: Collection[NodeId], temporal: Iterable[Node]
) -> tuple[int, int]:
    """
    The context items (before, after) the loop's temporal nodes whose context applies want: their radii added along each path, sources first, taking maxima where paths merge.
    """
    radii = {
        n.id: typing.cast(tuple[int, int], n.data.temporal_context) for n in temporal
    }
    needs: dict[NodeId, tuple[int, int]] = {}
    for node_id in reversed(chain.topological_order()):
        if node_id not in iterated:
            continue
        inputs = [needs.get(edge.source.id, (0, 0)) for edge in chain.edges_to(node_id)]
        radius = radii.get(node_id, (0, 0))
        needs[node_id] = (
            max((need[0] for need in inputs), default=0) + radius[0],
            max((need[1] for need in inputs), default=0) + radius[1],
        )
    return (
        max((need[0] for need in needs.values()), default=0),
        max((need[1] for need in needs.values()), default=0),
    )


class _ItemPlanner:
    """
    Which iterated nodes of a generator group run ahead for a later item, and their jobs; then the prepare phases of its sinks that registered them.
    """

    def __init__(
        self,
        chain: Chain,
        inputs: InputMap,
        node_cache: OutputCache[NodeOutput],
        generators: set[NodeId],
        iterated: set[NodeId],
        sinks: list[Node],
        collectors: Mapping[NodeId, Collector],
        pure_context: Callable[[Node], _ExecutorNodeContext],
        ahead_context: Callable[[FunctionNode, int], _ExecutorNodeContext],
        fail_fast: bool,
        decided: Mapping[NodeId, bool],
    ) -> None:
        self.__chain = chain
        self.__inputs = inputs
        self.__cache = node_cache
        self.__generators = generators
        self.__iterated = iterated
        # The collector sinks' collectors, which their prepares receive.
        self.__collectors = collectors
        self.__pure_context = pure_context
        self.__ahead_context = ahead_context
        # Deferred errors drop an item at whatever node fails first, maybe before a serial node's turn, which a run ahead could not undo: serial nodes run ahead only when every error ends the loop.
        self.__fail_fast = fail_fast
        # The loop run's pure_in_loop decisions (_LoopRun.pure), as they are made.
        self.__decided = decided
        ancestors: set[NodeId] = set()
        pending = [node.id for node in sinks]
        while pending:
            for edge in chain.edges_to(pending.pop()):
                if edge.source.id not in ancestors:
                    ancestors.add(edge.source.id)
                    pending.append(edge.source.id)
        eager = eager_ancestors(chain, inputs, sinks)

        def depends(sources: Iterable[Input]) -> frozenset[NodeId]:
            return frozenset(
                i.id
                for i in sources
                if isinstance(i, EdgeInput)
                and i.id in iterated
                and (i.id not in generators)
            )

        # A node that may not run beside others (a GPU or context node) is computed ahead only for a look-ahead, one item at a time in order (serial); never with a lazy input, which a look-ahead could reach before the node ran at the committing item. So is a locked-device node that the sinks reach only through lazy inputs their node may leave unread (may_go_unread), and the pure nodes that need it wait for it.
        self.__candidates = [
            Candidate(
                node_id,
                depends(inputs.get(node_id)),
                serial=(
                    self.__pure(node_id) is False
                    and (not any(i.lazy for i in chain.nodes[node_id].data.inputs))
                )
                or (
                    node_id not in eager
                    and chain.nodes[node_id].data.schema_id in LOCKED_DEVICE_NODES
                ),
            )
            for node_id in reversed(chain.topological_order())
            if node_id in ancestors
            and node_id in iterated
            and isinstance(chain.nodes[node_id], FunctionNode)
            and (not chain.nodes[node_id].has_side_effects())
        ]
        self.__prepared = {
            node.id for node in sinks if node_phases(node.data.schema_id) is not None
        }
        self.__candidates += [
            Candidate(node_id, depends(self.__phase_sources(node_id)), prepare=True)
            for node_id in reversed(chain.topological_order())
            if node_id in self.__prepared
        ]
        self.__serial = {c.node_id for c in self.__candidates if c.serial}
        self.__ahead: dict[NodeId, bool] = {}

    @property
    def candidates(self) -> Sequence[Candidate]:
        return self.__candidates

    def __pure(self, node_id: NodeId) -> bool | None:
        """
        Whether the node runs as a pure one in this loop run: pure_cpu_node, and for a temporal node on a pure schema the loop run's pure_in_loop, None until its first run on the commit path decided it (the window may be made before that, by a look-ahead during the first item). An undecided node is neither a serial candidate nor eligible, and no answer that depends on it is remembered.
        """
        data = self.__chain.nodes[node_id].data
        if not temporal_pure(data):
            return pure_cpu_node(data)
        return self.__decided.get(node_id)

    def ahead(self, node_id: NodeId) -> bool:
        """
        Whether the window computes this node ahead in every later slot whose generators all yielded and whose candidates it depends on did not fail, so a look-ahead can wait for it: a candidate (not a prepare), pure CPU and not Into Directory, or serial when every error ends the loop, whose inputs are constants, generator outputs, cached static values or such nodes. Asked once the commit path ran the node, so its static inputs are cached.
        """
        if node_id not in self.__ahead:
            if self.__pure(node_id) is None:
                return False
            candidate = next(
                (
                    c
                    for c in self.__candidates
                    if c.node_id == node_id and (not c.prepare)
                ),
                None,
            )
            data = self.__chain.nodes[node_id].data
            self.__ahead[node_id] = (
                candidate is not None
                and (
                    (candidate.serial and self.__fail_fast)
                    or (
                        data.schema_id not in _NOT_SPECULATED
                        and self.__pure(node_id) is True
                    )
                )
                and all(
                    not isinstance(i, EdgeInput)
                    or i.id in self.__generators
                    or (i.id not in self.__iterated and self.__cache.has(i.id))
                    or self.ahead(i.id)
                    for i in self.__inputs.get(node_id)
                )
            )
            if not all(
                not isinstance(i, EdgeInput)
                or i.id in self.__generators
                or i.id not in self.__iterated
                or i.id in self.__ahead
                for i in self.__inputs.get(node_id)
            ):
                # It depends on an undecided node: asked again later.
                return self.__ahead.pop(node_id)
        return self.__ahead[node_id]

    def threads(self, node_id: NodeId) -> int:
        """
        The pool threads a look-ahead on this node needs so it never starves: the waiting node's, one per serial candidate among the node and the iterated nodes it depends on (each may wait in its own look-ahead), and one that works.
        """
        seen: set[NodeId] = set()
        pending = [node_id]
        while pending:
            current = pending.pop()
            if current not in seen and current in self.__iterated:
                seen.add(current)
                pending.extend(
                    i.id for i in self.__inputs.get(current) if isinstance(i, EdgeInput)
                )
        return 2 + len(seen & self.__serial)

    def __phase_sources(self, node_id: NodeId) -> list[Input]:
        sources = self.__inputs.get(node_id)
        return [
            sources[index] for index in phase_inputs(self.__chain.nodes[node_id].data)
        ]

    def eligible(
        self,
        node_id: NodeId,
        values: Mapping[NodeId, object],
        store: Mapping[NodeId, Stored],
    ) -> bool:
        if node_id in self.__prepared:
            return all(
                not isinstance(i, EdgeInput)
                or i.id in values
                or (i.id in store and store[i.id].error is None)
                or (i.id not in self.__iterated and self.__cache.has(i.id))
                for i in self.__phase_sources(node_id)
            )
        if (
            self.__chain.nodes[node_id].data.schema_id in _NOT_SPECULATED
            or self.__pure(node_id) is None
        ):
            return False
        available = {*values, *(n for n, s in store.items() if s.error is None)}
        cache = _ItemCache(available, self.__iterated, self.__cache)
        if node_id in self.__serial:
            # Its inputs only: it runs alone, in order, as the loop runs it.
            return all(
                not isinstance(i, EdgeInput) or cache.has(i.id)
                for i in self.__inputs.get(node_id)
            )
        view = _ItemView(self.__chain, self.__inputs, cache)
        return pure_cpu_dependency(view, node_id) and all(
            not isinstance(i, EdgeInput)
            or i.id in self.__iterated
            or self.__cache.has(i.id)
            for i in self.__inputs.get(node_id)
        )

    def job(
        self,
        node_id: NodeId,
        values: Mapping[NodeId, object],
        store: Mapping[NodeId, Stored],
        slot: int,
    ) -> Callable[[], tuple[object, float]]:
        node = self.__chain.nodes[node_id]
        inputs: list[object] = []
        for i in (
            self.__phase_sources(node_id)
            if node_id in self.__prepared
            else self.__inputs.get(node_id)
        ):
            if not isinstance(i, EdgeInput):
                inputs.append(i.value)
            elif i.id in self.__generators:
                inputs.append(
                    _output_value(typing.cast(NodeOutput, values[i.id]), i.index)
                )
            elif i.id in store:
                inputs.append(
                    _output_value(typing.cast(NodeOutput, store[i.id].output), i.index)
                )
            else:
                assert i.id not in self.__iterated, (
                    f"Iterated input {i.id} of {node_id} is not stored"
                )
                cached = self.__cache.peek(i.id)
                assert cached is not None, (
                    f"Static input {i.id} of {node_id} is not shared"
                )
                inputs.append(_output_value(cached, i.index))
        if node_id in self.__prepared:
            return timed_supplier(
                functools.partial(
                    prepare_node, node.data, inputs, self.__collectors.get(node_id)
                )
            )
        context = (
            self.__ahead_context(typing.cast(FunctionNode, node), slot)
            if node_id in self.__serial
            else self.__pure_context(node)
        )
        # Its own cleanups after it, as the commit path runs a node's.
        return functools.partial(
            with_node_cleanups,
            timed_supplier(
                functools.partial(run_node, node.data, context, inputs, node_id)
            ),
            context,
        )

    def nbytes(self, value: object) -> int:
        return output_nbytes(
            value.output if isinstance(value, RegularOutput) else [value]
        )


class _SplitIterator(typing.Protocol):
    """
    Load Images' native iterator: describe() names its next item and raises StopIteration at the end; materialize(token) loads that item and returns an error as its value, as __next__ yields it.
    """

    def describe(self) -> object: ...

    def materialize(self, token: object) -> object: ...


class Executor:
    """
    Class for executing chaiNNer's processing logic
    """

    def __init__(
        self,
        id: ExecutionId,
        chain: Chain,
        send_broadcast_data: bool,
        options: ExecutionOptions,
        loop: asyncio.AbstractEventLoop,
        queue: EventConsumer,
        pool: ThreadPoolExecutor,
        storage_dir: Path,
        parent_cache: OutputCache[NodeOutput] | None = None,
        pool_size: int | None = None,
    ):
        self.id: ExecutionId = id
        self.chain = chain
        self.inputs: InputMap = InputMap.from_chain(chain)
        self.send_broadcast_data: bool = send_broadcast_data
        self.options: ExecutionOptions = options
        self.node_cache: OutputCache[NodeOutput] = OutputCache(parent=parent_cache)
        self.__broadcasts = LatestBroadcasts()
        self.__flights = NodeFlights()
        self.__context_cache: dict[str, _ExecutorNodeContext] = {}
        self.progress = ProgressController()
        self.loop: asyncio.AbstractEventLoop = loop
        self.queue: EventConsumer = queue
        self.pool: ThreadPoolExecutor = pool
        self.__pool_size = pool_size
        # Shared by every executor on the pool; made on the loop thread, as executors are.
        self.__lazy_waits: _LazyWaits | None = (
            None
            if pool_size is None
            else _lazy_waits.setdefault(pool, _LazyWaits(pool_size))
        )
        # The lazy reads a starved pool ended whose nodes still run (__wait_lazy).
        self.__starved_reads: set[Future[typing.Any]] = set()
        self.cache_strategy: dict[NodeId, CacheStrategy] = get_cache_strategies(chain)
        self._storage_dir = storage_dir
        self.__window: ItemWindow | None = None
        self.__planner: _ItemPlanner | None = None  # the window's
        self.__run: _LoopRun | None = None
        self.__collector_directories: dict[NodeId, dict[int, object]] = {}
        self.__item_bytes: dict[NodeId, int] | None = None
        self.__budget: int | None = None
        # The nodes of loops that run one chosen item, and that item (chosen_item).
        self.__chosen_items: dict[NodeId, int] = {}
        # The generators with a loop to run (__process_nodes)
        self.__looped: set[NodeId] = set()

    async def process(
        self, node_id: NodeId, perform_cache: bool = True
    ) -> NodeOutput | CollectorOutput:
        if perform_cache:
            cached = self.node_cache.peek(node_id)
            if cached is not None:
                return cached
        node = self.chain.nodes[node_id]
        try:
            if perform_cache:
                return await self.__flights.run(
                    node_id, functools.partial(self.__process, node, perform_cache)
                )
            return await self.__process(node, perform_cache)
        except Aborted:
            raise
        except NodeExecutionError:
            raise
        except Exception as e:
            raise NodeExecutionError(node.id, node.data, str(e), {}) from e

    async def process_regular_node(self, node: FunctionNode) -> RegularOutput:
        """
        Processes the given regular node.

        This will run all necessary node events.
        """
        result = await self.process(node.id)
        assert isinstance(result, RegularOutput)
        if node.has_side_effects():
            self.node_cache.get(node.id)
        return result

    async def process_generator_node(
        self, node: GeneratorNode, perform_cache: bool = True
    ) -> GeneratorOutput:
        """
        Processes the given iterator node.

        This will **not** iterate the returned generator. Only `node-start` and
        `node-broadcast` events will be sent.
        """
        result = await self.process(node.id, perform_cache)
        assert isinstance(result, GeneratorOutput)
        return result

    async def process_collector_node(self, node: CollectorNode) -> CollectorOutput:
        """
        Processes the given collector node.

        This will **not** iterate the returned collector. Only a `node-start` event
        will be sent.
        """
        result = await self.process(node.id)
        assert isinstance(result, CollectorOutput)
        return result

    async def __get_node_output(self, node_id: NodeId, output_index: int) -> object:
        """
        Returns the output value of the given node.

        Note: `output_index` is NOT an output ID.
        """
        output = await self.process(node_id)
        self.node_cache.get(node_id)
        if isinstance(output, CollectorOutput):
            raise ValueError("A collector was not run before another node needed it.")
        if isinstance(output, GeneratorOutput):
            value = _output_value(output, output_index)
            assert value is not None, "A generator output was not assigned correctly"
            return value
        assert isinstance(output, RegularOutput)
        return _output_value(output, output_index)

    def __wait_lazy(
        self, unfinished: list[Future[typing.Any]], completion: Future[_T]
    ) -> _T:
        """
        Waits on a lazy input's value, on the pool thread of the node reading it. Stop ends
        the wait (Aborted), and so does a pool whose every thread waits on a lazy input
        (an error): nothing would run the nodes the inputs need. The input's nodes still
        run then: the read goes to `unfinished`, and the reading node waits for them before
        it ends (__finish_lazy_reads). Until the reads a starved pool ended have finished,
        every other wait that times out ends with that error at once: the pool is full
        again while they drain, and a verdict per node would cost LAZY_STARVED_SECONDS
        each.
        """
        waits = self.__lazy_waits
        with nullcontext() if waits is None else waits.waiting():
            while True:
                try:
                    return completion.result(timeout=LAZY_POLL_SECONDS)
                except TimeoutError:
                    if self.progress.aborted:
                        unfinished.append(completion)
                        raise Aborted() from None
                    if waits is not None and (self.__starved_reads or waits.starved()):
                        self.__starved_reads.add(completion)
                        unfinished.append(completion)
                        raise RuntimeError(
                            "This chain's lazy inputs need more worker threads than"
                            f" the pool has: all {waits.threads} wait on a lazy input"
                            " (a Conditional, Logic Operation or Save Image input fed"
                            " through a long series of such nodes)."
                        ) from None

    async def __finish_lazy_reads(self, unfinished: list[Future[typing.Any]]) -> None:
        """
        Waits for the nodes of the lazy inputs whose read ended before their value came
        (__wait_lazy): the reading node ends after them, as after its own pool job
        (await_owned_node), so its cleanups, the run's end and the server's release of the
        run's memory follow them, and a next run cannot start beside them. Their error,
        other than Stop's, is logged: no read reports it.
        """
        if not unfinished:
            return
        if not all(completion.done() for completion in unfinished):
            logger.info(
                "Waiting for the nodes of a lazy input whose read was stopped..."
            )
        # Each read's outcome (Lazy's): the value or the error.
        outcomes = await await_owned_node(
            asyncio.gather(*(asyncio.wrap_future(read) for read in unfinished))
        )
        self.__starved_reads.difference_update(unfinished)
        unfinished.clear()
        for outcome in outcomes:
            error = outcome.error
            if error is not None and not isinstance(error, Aborted):
                logger.error(
                    "A lazy input's node failed after its read ended: %s",
                    error,
                    exc_info=error,
                )

    def __drop_unread(self, node: Node, inputs: list[object]) -> None:
        """
        The node ran without reading some of its lazy inputs (a Conditional's untaken
        branch, Save Image's existing file): each such edge gives up its read of its
        source's counted output, which is then freed after its last other read instead
        of at the run's end. A source not cached at this point keeps the edge's read
        (the native engine also counts it ahead of the source's run).
        """
        for index, value in enumerate(inputs):
            if isinstance(value, Lazy) and not (value.has_value or value.has_error):
                assigned = self.inputs.get(node.id)[index]
                if isinstance(assigned, EdgeInput) and self.node_cache.has(assigned.id):
                    self.node_cache.get(assigned.id)

    async def __resolve_node_input(self, node_input: Input) -> object:
        if isinstance(node_input, EdgeInput):
            return await self.__get_node_output(node_input.id, node_input.index)
        else:
            return node_input.value

    async def __gather_inputs(
        self, node: Node, unfinished: list[Future[typing.Any]]
    ) -> list[object]:
        """
        Returns the list of input values for the given node. Its lazy inputs' reads that
        end before their value came go to `unfinished` (__wait_lazy).
        """
        ignore: set[int] = set()
        if isinstance(node, CollectorNode):
            iterable_input = node.data.single_iterable_input
            for input_index, i in enumerate(node.data.inputs):
                if i.id in iterable_input.inputs:
                    ignore.add(input_index)
        lazy: set[int] = set()
        for input_index, i in enumerate(node.data.inputs):
            if i.lazy:
                lazy.add(input_index)
        assigned_inputs = self.inputs.get(node.id)
        assert len(assigned_inputs) == len(node.data.inputs)

        async def get_input_value(input_index: int, node_input: Input):
            if input_index in ignore:
                return None
            if input_index in lazy:
                return Lazy.from_coroutine(
                    self.__resolve_node_input(assigned_inputs[input_index]),
                    self.loop,
                    functools.partial(self.__wait_lazy, unfinished),
                )
            return await self.__resolve_node_input(node_input)

        inputs: list[object] = []
        ready = []
        for input_index, node_input in enumerate(assigned_inputs):
            parallel = (
                input_index not in ignore
                and input_index not in lazy
                and isinstance(node_input, EdgeInput)
                and pure_cpu_dependency(self, node_input.id)
            )
            if parallel:
                ready.append((input_index, node_input))
                if len(ready) < 4:
                    continue
            if ready:
                inputs.extend(
                    await owned_gather(
                        [get_input_value(index, value) for index, value in ready]
                    )
                )
                ready.clear()
            if not parallel:
                inputs.append(await get_input_value(input_index, node_input))
        if ready:
            inputs.extend(
                await owned_gather(
                    [get_input_value(index, value) for index, value in ready]
                )
            )
        return inputs

    async def __gather_collector_inputs(self, node: CollectorNode) -> list[object]:
        """
        Returns the input values to be consumed by `Collector.on_iterate`.
        """
        iterable_input = node.data.single_iterable_input
        assigned_inputs = self.inputs.get(node.id)
        assert len(assigned_inputs) == len(node.data.inputs)
        inputs = []
        for input_index, node_input in enumerate(assigned_inputs):
            i = node.data.inputs[input_index]
            if i.id in iterable_input.inputs:
                inputs.append(await self.__resolve_node_input(node_input))
        return inputs

    def __new_node_context(self, node: Node) -> _ExecutorNodeContext:
        package_id = registry.get_package(node.data.schema_id).id
        settings = self.options.get_package_settings(package_id)
        return _ExecutorNodeContext(
            self.progress,
            settings,
            self._storage_dir,
            node.id,
            run_warning(self.loop, self.queue),
            run_progress(self.loop, self.queue, node.id),
        )

    def __get_node_context(self, node: Node) -> _ExecutorNodeContext:
        context = self.__context_cache.get(node.id, None)
        if context is None:
            context = self.__new_node_context(node)
            self.__context_cache[node.id] = context
        context.chosen = self.__chosen_items.get(node.id)
        run = self.__run
        if (
            run is not None
            and isinstance(node, FunctionNode)
            and node.id in run.iterated
        ):
            # The loop runs this node on its commit path.
            context.iteration_view = self.__iteration_view(run, node, context)
        return context

    def __decide_pure(self, node: Node, inputs: list[object]) -> Exception | None:
        """
        At a temporal node on a pure schema's first run on its loop run's commit path: whether it runs as a pure one for the rest of the run (pure_in_loop, from its gate input's value), remembered in the run. The predicate's error is returned, as this run's failure, once: the node then needs its view for the rest of the run.
        """
        run = self.__run
        if (
            run is None
            or node.id not in run.iterated
            or node.id in run.pure
            or not temporal_pure(node.data)
        ):
            return None
        gate = node.data.temporal_if
        value = None if gate is None else inputs[gate_position(node.data)]
        try:
            run.pure[node.id] = pure_in_loop(node, self.inputs, run.iterated, value)
        except Exception as error:
            run.pure[node.id] = False
            return error
        return None

    async def __process(
        self, node: Node, perform_cache: bool = True
    ) -> NodeOutput | CollectorOutput:
        """
        Process a single node.

        In the case of generator and collectors, it will only run the node itself,
        not the actual iteration or collection.
        """
        logger.debug(f"node: {node}")
        logger.debug(f"Running node {node.id}")
        unfinished: list[Future[typing.Any]] = []
        inputs = await self.__gather_inputs(node, unfinished)
        if isinstance(node, CollectorNode):
            self.__collector_directories[node.id] = {
                index: inputs[index]
                for index, i in enumerate(node.data.inputs)
                if i.kind == "directory" or _relative_path_input(i)
            }
        context = self.__get_node_context(node)

        def get_lazy_evaluation_time():
            return sum(i.evaluation_time for i in inputs if isinstance(i, Lazy))

        await self.progress.suspend()
        self.__send_node_start(node)
        await self.progress.suspend()
        lazy_time_before = get_lazy_evaluation_time()
        window = self.__window
        refused = self.__decide_pure(node, inputs)
        if refused is not None:
            failed: asyncio.Future[tuple[NodeOutput | CollectorOutput, float]] = (
                self.loop.create_future()
            )
            failed.set_exception(refused)
            pending = failed
        elif window is not None and window.holds(node.id):
            # The window's job for this node runs run_node (_ItemPlanner.job), so it returns what the pool job would.
            pending = typing.cast(
                typing.Awaitable[tuple[NodeOutput | CollectorOutput, float]],
                window.take_node(node.id),
            )
        else:
            prepared = None if window is None else await window.take_prepared(node.id)
            pending = await_owned_node(
                self.loop.run_in_executor(
                    self.pool,
                    timed_supplier(
                        functools.partial(
                            run_node, node.data, context, inputs, node.id, prepared
                        )
                    ),
                )
            )
        try:
            output, execution_time = await pending
            await self.progress.suspend()
        finally:
            try:
                await self.__finish_lazy_reads(unfinished)
            finally:
                run_cleanup_functions(
                    context.node_cleanup_fns,
                    preserve_error=sys.exc_info()[0] is not None,
                )
        self.__drop_unread(node, inputs)
        lazy_time_after = get_lazy_evaluation_time()
        execution_time -= lazy_time_after - lazy_time_before
        if isinstance(output, RegularOutput):
            run = self.__run
            if run is None or run.slot >= run.first or node.id not in run.iterated:
                # A context item's outputs are discarded: the UI never sees them.
                await self.__send_node_broadcast(node, output.output)
            self.__send_node_finish(node, execution_time)
        elif isinstance(output, GeneratorOutput):
            await self.__send_node_broadcast(
                node, output.partial_output, generators=[output.generator]
            )
        if perform_cache and (not isinstance(output, CollectorOutput)):
            strategy = self.cache_strategy[node.id]
            if node.id in self.__looped and not strategy.static:
                # Its loop reads it too, after any reader that comes first (a collector's
                # construction, another generator, an earlier loop); the loop's end
                # restores it with its own strategy for later readers.
                strategy = CacheStrategy(strategy.hits_to_live + 1)
            self.node_cache.set(node.id, output, strategy)
        if self.__item_bytes is not None and isinstance(output, RegularOutput):
            self.__item_bytes[node.id] = output_nbytes(output.output)
        await self.progress.suspend()
        return output

    def __get_iterated_nodes(
        self, node: GeneratorNode
    ) -> tuple[list[CollectorNode], list[FunctionNode], set[Node]]:
        """
        Returns all collector and output nodes iterated by the given generator node, in the order the chain's edges reach them
        """
        collectors: list[CollectorNode] = []
        output_nodes: list[FunctionNode] = []
        seen: set[Node] = {node}

        def visit(n: Node):
            if n in seen:
                return
            seen.add(n)
            if isinstance(n, CollectorNode):
                collectors.append(n)
            elif isinstance(n, GeneratorNode):
                raise ValueError("Nested sequences are not supported")
            else:
                assert isinstance(n, FunctionNode)
                if n.has_side_effects():
                    output_nodes.append(n)
                for edge in self.chain.edges_from(n.id):
                    target_node = self.chain.nodes[edge.target.id]
                    visit(target_node)

        iterable_output = node.data.single_iterable_output
        for edge in self.chain.edges_from(node.id):
            if edge.source.output_id in iterable_output.outputs:
                target_node = self.chain.nodes[edge.target.id]
                visit(target_node)
        return (collectors, output_nodes, seen)

    def __generator_fill_partial_output(
        self, node: GeneratorNode, partial_output: Output, values: object
    ) -> Output:
        iterable_output = node.data.single_iterable_output
        values_list: list[object] = []
        if len(iterable_output.outputs) == 1:
            values_list.append(values)
        else:
            assert isinstance(values, (tuple, list))
            values_list.extend(values)
        assert len(values_list) == len(iterable_output.outputs)
        output: Output = partial_output.copy()
        for index, o in enumerate(node.data.outputs):
            if o.id in iterable_output.outputs:
                output[index] = o.enforce_(values_list.pop(0))
        return output

    def __describe(
        self, generator_supplier: typing.Iterator[Output | Exception]
    ) -> tuple[object] | None:
        """
        The generator's next item as a token: the 1-tuple of the values it yields, so a yielded None stays an item; None once it is exhausted. A yielded error is raised.
        """
        present, values = advance_iterator(generator_supplier)
        if not present:
            return None
        if isinstance(values, Exception):
            raise values
        return (values,)

    def __materialize(
        self, node: GeneratorNode, generator_output: GeneratorOutput, token: object
    ) -> RegularOutput:
        """
        The output of the item __describe named: its enforce (an image's normalize) runs here.
        """
        (values,) = typing.cast(tuple[object], token)
        return RegularOutput(
            self.__generator_fill_partial_output(
                node, generator_output.partial_output, values
            )
        )

    def __advance(
        self,
        node: GeneratorNode,
        generator_supplier: typing.Iterator[Output | Exception],
        generator_output: GeneratorOutput,
    ) -> RegularOutput | None:
        if hasattr(generator_supplier, "describe"):
            # A split iterator's item in its two pieces, as the window and the engine
            # take it: its error names the item's file (loop_node_error).
            iterator = typing.cast(_SplitIterator, generator_supplier)
            described = self.__describe_split(iterator)
            return (
                None
                if described is None
                else self.__materialize_split(
                    node, iterator, generator_output, described
                )
            )
        token = self.__describe(generator_supplier)
        return (
            None if token is None else self.__materialize(node, generator_output, token)
        )

    def __describe_split(self, iterator: _SplitIterator) -> object | None:
        """
        The token of a split iterator's next item (Load Images: its path and index), which nothing loads yet; None once it is exhausted.
        """
        try:
            return iterator.describe()
        except StopIteration:
            return None

    def __materialize_split(
        self,
        node: GeneratorNode,
        iterator: _SplitIterator,
        generator_output: GeneratorOutput,
        token: object,
    ) -> RegularOutput:
        """
        The output of the item a split iterator named: its load (a decode), whose error is raised, then its enforce (an image's normalize).
        """
        values = iterator.materialize(token)
        if isinstance(values, Exception):
            raise loop_node_error(node, self.inputs.get(node.id), values, token)
        return RegularOutput(
            self.__generator_fill_partial_output(
                node, generator_output.partial_output, values
            )
        )

    def __resolved_folder_input(
        self, writer: Node, index: int, node_input: EdgeInput
    ) -> object:
        """
        A writer's folder input for writer_directories: a collector's from its construction, a regular writer's from the cache (None when missing).
        """
        if isinstance(writer, CollectorNode):
            return self.__collector_directories[writer.id][index]
        cached = self.node_cache.peek(node_input.id)
        return None if cached is None else _output_value(cached, node_input.index)

    async def __open_window(
        self,
        generator_nodes: list[GeneratorNode],
        generator_outputs: dict[NodeId, GeneratorOutput],
        output_nodes: list[FunctionNode],
        collectors: list[tuple[Collector, _Timer, CollectorNode]],
        all_iterated_nodes: set[NodeId],
        generator_suppliers: dict[NodeId, typing.Iterator[Output | Exception]],
        first: int,
        end: int | None,
        context_nodes: frozenset[NodeId],
        stops: int,
    ) -> tuple[ItemWindow | None, str | None]:
        """
        Decides the group's item window K at the range's second slot (slot 1 without context items) and logs it.

        K = 1 builds no window and resolves no path. Else the writers' folder nodes not computed yet run first (folder_sources). Returns the window and the read-ahead blocker that set K = 1, if one did. The window learns the context items (before `first`, from `end` on), where it computes ahead only `context_nodes`.
        """
        # The measurement is kept until the decision is made: a decision that raises (a
        # look-ahead before the range's second slot decides early) is made again there.
        measured = self.__item_bytes
        k, reason = window_size(affinity_cpus())
        blocker = None
        if k > 1 and self.__pool_size is None:
            k, reason = (1, "pool size unknown")
        if k > 1:
            # A folder its writer's gather would compute at the writer's turn (after a
            # look-ahead at item 0 decides): computed now, so the check knows it.
            for source in folder_sources(
                self.chain,
                self.inputs,
                output_nodes,
                all_iterated_nodes,
                self.node_cache.has,
            ):
                await self.process(source)
            writers: list[Node] = [
                *output_nodes,
                *(collector_node for _, _, collector_node in collectors),
            ]
            sources = [
                (node.id, generator_outputs[node.id].generator.source_paths)
                for node in generator_nodes
            ]
            directories, targets = writer_directories(
                self.chain,
                self.inputs,
                writers,
                generator_outputs,
                all_iterated_nodes,
                self.__resolved_folder_input,
            )
            blocker = await await_owned_node(
                self.loop.run_in_executor(
                    self.pool,
                    functools.partial(read_ahead_blocker, sources, directories),
                )
            )
            if blocker is not None:
                k, reason = (1, blocker)
            escaping = await await_owned_node(
                self.loop.run_in_executor(
                    self.pool, functools.partial(escaping_targets, targets)
                )
            )
            for writer, folder in escaping:
                self.queue.put(escaping_target_warning(self.chain, writer, folder))
        logger.info("item window K=%d (%s)", k, reason)
        self.__item_bytes = None
        if k == 1:
            return None, blocker
        assert self.__pool_size is not None and measured is not None
        estimate = sum(
            size for node_id, size in measured.items() if node_id in all_iterated_nodes
        )
        if self.__budget is None:
            self.__budget = memory_budget()
        generators = [
            SlotGenerator(
                node.id,
                functools.partial(
                    self.__advance,
                    node,
                    generator_suppliers[node.id],
                    generator_outputs[node.id],
                ),
                node.data.schema_id == "chainner:utility:range",
                generator_outputs[node.id].generator.fail_fast,
            )
            for node in generator_nodes
        ]
        if (
            len(generators) == 1
            and (not generators[0].inline)
            and (not serial_producer())
        ):
            # The group's single generator, Range aside, is described serially and materialized as engine jobs. A split iterator (Load Images) names each item, then loads it; any other yields it at describe, and its enforce (an image's normalize) is the materialize.
            node = generator_nodes[0]
            supplier, output = (
                generator_suppliers[node.id],
                generator_outputs[node.id],
            )
            if hasattr(supplier, "describe"):
                iterator = typing.cast(_SplitIterator, supplier)
                generators[0] = replace(
                    generators[0],
                    describe=functools.partial(self.__describe_split, iterator),
                    materialize=functools.partial(
                        self.__materialize_split, node, iterator, output
                    ),
                    split=True,
                )
            else:
                generators[0] = replace(
                    generators[0],
                    describe=functools.partial(self.__describe, supplier),
                    materialize=functools.partial(self.__materialize, node, output),
                )
        sinks: list[Node] = [
            *output_nodes,
            *(collector_node for _, _, collector_node in collectors),
        ]
        run = self.__run
        assert run is not None
        planner = self.__planner = _ItemPlanner(
            self.chain,
            self.inputs,
            self.node_cache,
            {node.id for node in generator_nodes},
            all_iterated_nodes,
            sinks,
            {node.id: collector for collector, _, node in collectors},
            self.__pure_ahead_context,
            self.__ahead_context,
            all(generator.fail_fast for generator in generators),
            run.pure,
        )
        window = ItemWindow(
            size=k,
            generators=generators,
            stops=stops,
            loop=self.loop,
            pool=self.pool,
            jobs=engine_jobs(self.__pool_size),
            budget=self.__budget,
            estimate=estimate,
            planner=planner,
            progress=self.progress,
            first=first,
            end=end,
            context_nodes=context_nodes,
        )
        return window, None

    async def __close_window(self) -> None:
        self.__item_bytes = None
        self.__planner = None
        window, self.__window = (self.__window, None)
        if window is not None:
            await window.close()
            logger.info(
                "item window K=%d: %d items, %d ahead, %d jobs, %d materialized, %d replayed, %d awaited, %d discarded",
                window.size,
                window.items,
                window.advanced_ahead,
                window.jobs_started,
                window.materialized,
                window.replayed,
                window.awaited,
                window.discarded,
            )

    def __end_loop(self) -> None:
        """
        The loop run ended, failed or was cancelled: its nodes' iteration views are detached and their states dropped. Idempotent.
        """
        run, self.__run = (self.__run, None)
        if run is not None:
            for view in run.views.values():
                view.close()
            run.views.clear()

    def __pure_ahead_context(self, node: Node) -> _ExecutorNodeContext:
        """
        The context of a pure node computed ahead at a later item: a new one, without an iteration view, with node cleanups of its own, which its job runs after it (with_node_cleanups), and its own context's chain cleanups, which the run's end runs.
        """
        shared = self.__context_cache.get(node.id)
        if shared is None:
            shared = self.__context_cache[node.id] = self.__new_node_context(node)
        context = self.__new_node_context(node)
        context.chain_cleanup_fns = shared.chain_cleanup_fns
        return context

    def __ahead_context(self, node: FunctionNode, slot: int) -> _ExecutorNodeContext:
        """
        The context of a serial node computed ahead at a later item: its own context (settings, chain cleanups) with its iteration view at that item, sharing its state slot, and node cleanups of its own, which its job runs after it (with_node_cleanups): the commit path's run after the node's commit at its own item.
        """
        context = self.__get_node_context(node)
        view = context.iteration_view
        ahead = copy.copy(context)
        ahead.node_cleanup_fns = set()
        ahead.iteration_view = None if view is None else view.at(slot, ahead)
        return ahead

    def __iteration_view(
        self, run: _LoopRun, node: FunctionNode, context: _ExecutorNodeContext
    ) -> _IterationView:
        """
        The node's iteration view for this loop run, made when the loop first runs it.
        """
        view = run.views.get(node.id)
        if view is None:
            lineages = [self.__input_lineage(run, i) for i in self.inputs.get(node.id)]
            assert any(lineage is not None for lineage in lineages), (
                f"Iterated node {node.id} has no lineage"
            )
            view = run.views[node.id] = _IterationView(
                run,
                context,
                node,
                tuple(
                    None
                    if lineage is None
                    else self.chain.nodes[lineage].data.schema_id
                    for lineage in lineages
                ),
                functools.partial(self.__lookahead, run, node.id),
                functools.partial(self.__peek, run, node.id),
                self.loop,
            )
        return view

    def __lineage(self, run: _LoopRun, node_id: NodeId) -> NodeId | None:
        """
        The generator whose sequence iterates this node of the loop, as the UI's lineage: that of its first iterated input, inputs in order.
        """
        if node_id not in run.lineages:
            lineage = None
            for i in self.inputs.get(node_id):
                lineage = self.__input_lineage(run, i)
                if lineage is not None:
                    break
            run.lineages[node_id] = lineage
        return run.lineages[node_id]

    def __input_lineage(self, run: _LoopRun, i: Input) -> NodeId | None:
        """
        The generator whose sequence iterates this input: the generator of an iterated output, or the lineage of an iterated node; None for an input that is not iterated.
        """
        if not isinstance(i, EdgeInput):
            return None
        source = self.chain.nodes[i.id]
        if isinstance(source, GeneratorNode):
            iterable_output = source.data.single_iterable_output
            if source.data.outputs[i.index].id in iterable_output.outputs:
                return i.id
        elif isinstance(source, FunctionNode) and i.id in run.iterated:
            return self.__lineage(run, i.id)
        return None

    async def __look_ahead_window(self, run: _LoopRun) -> ItemWindow | None:
        """
        The group's item window for a look-ahead. The first look-ahead before the range's second slot (during slot 0 without context items) decides it there, once, and the window starts at that slot; the producer then continues from the next one. None, deciding nothing, with a pool of one thread: the waiting node holds it, and the decision's path check and the window's reads need another.
        """
        if self.__run is not run or self.__pool_size is None or self.__pool_size < 2:
            return None
        if not run.decided:
            if run.deciding is None:
                run.deciding = self.loop.create_task(self.__decide_early(run))
            await await_owned_node(run.deciding)
        return self.__window

    async def __decide_early(self, run: _LoopRun) -> None:
        # A slot up to the range's first runs: no generator stopped (context items come
        # with a group's only generator, whose end would have ended the loop).
        window, run.blocker = await run.open_window(0)
        run.decided = True
        if window is not None:
            window.start(run.slot)
        self.__window = window

    def __look_ahead_source(
        self, run: _LoopRun, node_id: NodeId, index: int
    ) -> EdgeInput | None:
        """
        The edge feeding this input when a look-ahead can be served: from an iterated output of one of the group's generators, or from a node the window computes ahead when the pool has the threads its waits need.
        """
        i = self.inputs.get(node_id)[index]
        if not isinstance(i, EdgeInput):
            return None
        source = self.chain.nodes[i.id]
        if isinstance(source, GeneratorNode):
            iterable_output = source.data.single_iterable_output
            iterated = source.data.outputs[i.index].id in iterable_output.outputs
            return i if iterated and i.id in run.generators else None
        planner = self.__planner
        if planner is None or not planner.ahead(i.id):
            return None
        # A pool too small for the waits it may hold serves nothing (the node falls back).
        assert self.__pool_size is not None
        return i if self.__pool_size >= planner.threads(i.id) else None

    async def __lookahead(self, run: _LoopRun, node_id: NodeId, index: int) -> int:
        window = await self.__look_ahead_window(run)
        if window is None:
            self.__warn_look_ahead_off(run, node_id)
        if window is None or self.__look_ahead_source(run, node_id, index) is None:
            return 0
        return window.size - 1

    def __warn_look_ahead_off(self, run: _LoopRun, node_id: NodeId) -> None:
        """
        A node looks ahead in a loop whose read-ahead blocker set K = 1, so it runs without its look-ahead: a run warning tells the UI why, once per node and loop run.
        """
        if self.__run is not run or run.blocker is None or node_id in run.warned:
            return
        run.warned.add(node_id)
        self.queue.put(look_ahead_off_warning(self.chain, node_id, run.blocker))

    async def __peek(
        self, run: _LoopRun, node_id: NodeId, origin: int, index: int, offset: int
    ) -> object | None:
        window = await self.__look_ahead_window(run)
        if window is None or offset >= window.size:
            return None
        source = self.__look_ahead_source(run, node_id, index)
        if source is None:
            return None
        # origin: the node's own item, a later one when it is computed ahead.
        output = await window.peek(
            origin + offset, source.id, None if origin == run.slot else origin
        )
        if output is None:
            return None
        return _output_value(typing.cast(NodeOutput, output), source.index)

    async def __temporal_nodes(
        self, iterated: set[NodeId], sinks: list[Node], generator: Generator
    ) -> tuple[list[FunctionNode], frozenset[NodeId], frozenset[NodeId]]:
        """
        The loop's temporal nodes whose context applies (temporal_nodes, temporal_applies), in the chain's sink-first order. When the generator reads context items (reads_context), a known gate's input is resolved first, so its source (Load Model) runs before the loop's first item. Also the loop's nodes that run for a context item and the temporal nodes none of those reads (context_nodes).
        """
        temporal = temporal_nodes(self.chain, iterated, sinks)
        gates = {
            node.id: await self.__resolve_node_input(self.inputs.get(node.id)[position])
            for node, position in (
                known_gates(self.chain, self.inputs, iterated, sinks, temporal)
                if reads_context(generator)
                else []
            )
        }
        temporal = [
            node for node in temporal if temporal_applies(node, gates, self.inputs)
        ]
        needed, unread = context_nodes(self.chain, iterated, temporal)
        return temporal, needed, unread

    async def __iterate_generator_nodes(self, generator_nodes: list[GeneratorNode]):
        await self.progress.suspend()
        num_generators = len(generator_nodes)
        expected_lengths = {}
        collectors: list[tuple[Collector, _Timer, CollectorNode]] = []
        collector_ids: set[NodeId] = set()
        output_nodes: list[FunctionNode] = []
        all_iterated_nodes: set[NodeId] = set()
        generator_suppliers: dict[NodeId, typing.Iterator[Output | Exception]] = {}
        generator_outputs: dict[NodeId, GeneratorOutput] = {}
        iter_timers: dict[NodeId, _IterationTimer] = {}
        # Context items (a group's only generator reads them for its temporal nodes):
        # `before` items before the range, run through the temporal nodes and the nodes
        # they need (context_nodes) alone, and `after` items after it, only served to
        # look-aheads.
        temporal_nodes: list[FunctionNode] = []
        context_nodes: frozenset[NodeId] = frozenset()
        unread: frozenset[NodeId] = frozenset()
        before = after = 0
        for node in generator_nodes:
            generator_output = await self.process_generator_node(node)
            generator_outputs[node.id] = generator_output
            generator = generator_output.generator
            contextual = num_generators == 1 and generator.context is not None
            if not contextual:
                generator_suppliers[node.id] = generator.supplier().__iter__()
            collector_nodes, __output_nodes, __all_iterated_nodes = (
                self.__get_iterated_nodes(node)
            )
            for iterated_node in __all_iterated_nodes:
                all_iterated_nodes.add(iterated_node.id)
            for o_node in __output_nodes:
                if o_node not in output_nodes:
                    output_nodes.append(o_node)
            if len(collector_nodes) == 0 and len(output_nodes) == 0:
                return
            if num_generators == 1 and generator.chosen_item is not None:
                for iterated_node in __all_iterated_nodes:
                    self.__chosen_items[iterated_node.id] = generator.chosen_item
            if contextual:
                assert generator.context is not None
                temporal_nodes, context_nodes, unread = await self.__temporal_nodes(
                    all_iterated_nodes, [*output_nodes, *collector_nodes], generator
                )
                wanted = context_wanted(self.chain, all_iterated_nodes, temporal_nodes)
                if wanted == (0, 0) or generator.expected_length == 0:
                    generator_suppliers[node.id] = generator.supplier().__iter__()
                else:
                    items, before, after = generator.context(*wanted)
                    assert 0 <= before <= wanted[0] and 0 <= after <= wanted[1], (
                        f"{node.data.name} read ({before}, {after}) context items"
                        f" for the {wanted} asked"
                    )
                    generator_suppliers[node.id] = items.__iter__()
            for collector_node in collector_nodes:
                if collector_node.id in collector_ids:
                    continue
                collector_ids.add(collector_node.id)
                await self.progress.suspend()
                timer = _Timer()
                with timer.run():
                    collector_output = await self.process_collector_node(collector_node)
                assert isinstance(collector_output, CollectorOutput)
                collectors.append((collector_output.collector, timer, collector_node))
            expected_length = generator_output.generator.expected_length
            expected_lengths[node.id] = expected_length
            iter_times = _IterationTimer(self.progress)
            iter_timers[node.id] = iter_times
            self.__send_node_progress(node, [], 0, expected_length)
        exact_lengths = {
            expected_lengths[node_id]
            for node_id, output in generator_outputs.items()
            if output.generator.length_is_exact
        }
        if len(exact_lengths) > 1:
            raise AssertionError(
                "Expected all connected iterators to have the same length"
            )
        # Before the first item, in both run engines: no two items may give a writer one path.
        check_loop_outputs(
            self.chain,
            [(node, generator_outputs[node.id]) for node in generator_nodes],
            all_iterated_nodes,
        )
        total_stopiters = 0
        # The generators whose items ran out
        ended: set[NodeId] = set()
        deferred_errors: list[str] = []
        self.__item_bytes = {}
        # The range's slots: `before` context items come first; with items after it, the
        # loop ends after its last one (they only serve look-aheads), else as always, when
        # the generators are exhausted.
        first = before
        end = None if after == 0 else before + expected_lengths[generator_nodes[0].id]
        bases = {output.generator.index_base for output in generator_outputs.values()}
        run = self.__run = _LoopRun(
            {node.id for node in generator_nodes},
            all_iterated_nodes,
            functools.partial(
                self.__open_window,
                generator_nodes,
                generator_outputs,
                output_nodes,
                collectors,
                all_iterated_nodes,
                generator_suppliers,
                first,
                end,
                context_nodes,
            ),
            first=first,
            base=bases.pop() if len(bases) == 1 else 0,
            unread=unread,
        )
        slot = -1
        while True:
            slot += 1
            if slot == end:
                break
            run.slot = slot
            is_context = slot < first
            if slot == first and (not run.decided):
                # The window's estimate measures the range's first item: a context item
                # runs the temporal nodes alone.
                self.__item_bytes = {}
            if slot == first + 1 and (not run.decided):
                run.decided = True
                self.__window, run.blocker = await run.open_window(total_stopiters)
            window = self.__window
            if window is not None:
                window.begin(slot)
            generator_output = None
            try:
                for node in generator_nodes:
                    await self.progress.suspend()
                    generator_output = generator_outputs[node.id]
                    generator_supplier = generator_suppliers[node.id]
                    try:
                        if window is not None:
                            # The producer runs this generator's __advance, or its describe and materialize pieces, so its outcome is what the loop's advance returns.
                            if native_profile.enabled():
                                # The commit path's wait for the producer, recorded outside the nesting: other timed calls run on this thread while it waits. It includes take_advance's own _wanted/_pump work and every loop callback run while it is suspended.
                                started = time.perf_counter_ns()
                                try:
                                    iter_output = typing.cast(
                                        RegularOutput | None,
                                        await window.take_advance(node.id),
                                    )
                                finally:
                                    native_profile.record(
                                        "window.take_advance",
                                        time.perf_counter_ns() - started,
                                    )
                            else:
                                iter_output = typing.cast(
                                    RegularOutput | None,
                                    await window.take_advance(node.id),
                                )
                        elif node.data.schema_id == "chainner:utility:range":
                            iter_output = self.__advance(
                                node, generator_supplier, generator_output
                            )
                        else:
                            iter_output = await await_owned_node(
                                self.loop.run_in_executor(
                                    self.pool,
                                    functools.partial(
                                        self.__advance,
                                        node,
                                        generator_supplier,
                                        generator_output,
                                    ),
                                )
                            )
                    except Exception as error:
                        failure = loop_node_error(node, self.inputs.get(node.id), error)
                        if failure is error:
                            raise
                        raise failure from error
                    if iter_output is None:
                        ended.add(node.id)
                        raise StopIteration
                    if self.__item_bytes is not None:  # K is not decided yet
                        self.__item_bytes[node.id] = output_nbytes(iter_output.output)
                    await self.progress.suspend()
                    self.node_cache.set(node.id, iter_output, StaticCaching)
                    if not is_context:
                        await self.__send_node_broadcast(node, iter_output.output)
                if is_context:
                    # A context item updates the temporal nodes' state: they run, with the
                    # nodes they need, and nothing else. Sources first: a temporal node another
                    # one reads is cached for it, not run again after the reader took it.
                    for temporal_node in reversed(temporal_nodes):
                        await self.process_regular_node(temporal_node)
                for output_node in () if is_context else output_nodes:
                    await self.process_regular_node(output_node)
                for collector, timer, collector_node in (
                    () if is_context else collectors
                ):
                    await self.progress.suspend()
                    iterate_inputs = await self.__gather_collector_inputs(
                        collector_node
                    )
                    await self.progress.suspend()
                    with timer.run():
                        prepared = (
                            None
                            if window is None
                            else await window.take_prepared(collector_node.id)
                        )
                        iterate = functools.partial(
                            run_collector_iterate,
                            collector_node,
                            iterate_inputs,
                            collector,
                            prepared,
                        )
                        if (
                            collector_node.data.schema_id
                            == "chainner:utility:accumulate"
                        ):
                            iterate()
                        else:
                            await await_owned_node(
                                self.loop.run_in_executor(self.pool, iterate)
                            )
                self.node_cache.delete_many(all_iterated_nodes)
                if window is not None:
                    window.release(slot)
                await self.progress.suspend()
                for node in () if is_context else generator_nodes:
                    iter_times = iter_timers[node.id]
                    iter_times.add()
                    iterations = iter_times.iterations
                    self.__send_node_progress(
                        node,
                        iter_times.times,
                        iterations,
                        max(expected_lengths[node.id], iterations),
                    )
                await asyncio.sleep(0)
                await self.progress.suspend()
            except Aborted:
                raise
            except StopIteration:
                if window is not None:
                    window.release(slot)
                total_stopiters = total_stopiters + 1
                if total_stopiters >= num_generators:
                    # Items that end early (a truncated video): one warning for each
                    # generator that ran out
                    for node in () if deferred_errors else generator_nodes:
                        if node.id not in ended:
                            continue
                        message = generator_outputs[
                            node.id
                        ].generator.shortfall_warning(
                            node.data.name, iter_timers[node.id].iterations
                        )
                        if message is not None:
                            logger.warning(message)
                            self.queue.put(
                                {
                                    "event": "execution-warning",
                                    "data": {"message": message},
                                }
                            )
                    break
            except Exception as e:
                if generator_output and generator_output.generator.fail_fast:
                    raise e
                else:
                    deferred_errors.append(str(e))
                    self.node_cache.delete_many(all_iterated_nodes)
                    if window is not None:
                        window.release(slot)
                    await asyncio.sleep(0)
                    await self.progress.suspend()
        self.__end_loop()
        await self.__close_window()
        await self.flush_broadcasts()
        self.node_cache.delete_many(all_iterated_nodes)
        for node in generator_nodes:
            await self.progress.suspend()
            generator_output = generator_outputs[node.id]
            self.node_cache.set(node.id, generator_output, self.cache_strategy[node.id])
            await self.__send_node_broadcast(node, generator_output.partial_output)
            self.__send_node_progress_done(node, iter_timers[node.id].iterations)
            self.__send_node_finish(node, iter_timers[node.id].get_time_since_start())
        for collector, timer, collector_node in collectors:
            await self.progress.suspend()
            with timer.run():

                def complete(
                    collector: Collector = collector,
                    collector_node: CollectorNode = collector_node,
                ):
                    return run_collector_complete(
                        collector_node, collector, self.inputs.get(collector_node.id)
                    )

                collector_output = await await_owned_node(
                    self.loop.run_in_executor(self.pool, complete)
                )
            await self.__send_node_broadcast(collector_node, collector_output.output)
            self.__send_node_finish(collector_node, timer.duration)
            self.node_cache.set(
                collector_node.id,
                collector_output,
                self.cache_strategy[collector_node.id],
            )
        if len(deferred_errors) > 0:
            error_string = "- " + "\n- ".join(deferred_errors)
            raise Exception(f"Errors occurred during iteration:\n{error_string}")

    async def __process_nodes(self):
        refuse_reads_across_loops(self.chain)
        self.__send_chain_start()
        gens_by_outs: dict[NodeId, set[NodeId]] = {}
        for node_id in self.chain.topological_order():
            node = self.chain.nodes[node_id]
            if isinstance(node, GeneratorNode):
                collector_nodes, output_nodes, _ = self.__get_iterated_nodes(node)
                for sink in (*collector_nodes, *output_nodes):
                    gens_by_outs.setdefault(sink.id, set()).add(node.id)
        self.__looped = {
            generator
            for generators in gens_by_outs.values()
            for generator in generators
        }
        for nodes_to_run in generator_loops(self.chain, gens_by_outs):
            try:
                await self.__iterate_generator_nodes(nodes_to_run)
            finally:
                self.__end_loop()
                await self.__close_window()
        # A seek-bar generator that feeds no sink runs for what it shows (shown_alone).
        for node in self.chain.nodes.values():
            if isinstance(node, GeneratorNode) and shown_alone(
                self.chain, node, self.__looped
            ):
                await self.progress.suspend()
                await self.process_generator_node(node)
        non_iterable_output_nodes = [
            node
            for node, iter_node in self.chain.get_parent_iterator_map().items()
            if iter_node is None and node.has_side_effects()
        ]
        for output_node in non_iterable_output_nodes:
            await self.progress.suspend()
            await self.process_regular_node(output_node)

    async def flush_broadcasts(self):
        await self.__broadcasts.flush()

    async def run(self):
        logger.debug("Running executor %s", self.id)
        try:
            await self.__process_nodes()
        except Exception as error:
            moved = on_failing_node(error, self.chain.nodes)
            if moved is error:
                raise
            raise moved from moved.__cause__
        finally:
            preserve_error = sys.exc_info()[0] is not None
            try:
                try:
                    await self.flush_broadcasts()
                except BaseException:
                    if not preserve_error:
                        raise
                    logger.exception("Broadcast failed while execution was exiting")
            finally:
                self.node_cache.clear()
                self.__run_cleanups(preserve_error or sys.exc_info()[0] is not None)

    def __run_cleanups(self, preserve_error: bool) -> None:
        run_cleanup_groups(
            (
                callbacks
                for context in self.__context_cache.values()
                for callbacks in (
                    context.node_cleanup_fns,
                    context.chain_cleanup_fns,
                )
            ),
            preserve_error=preserve_error,
        )

    async def chain_cleanups(self, preserve_error: bool) -> None:
        """
        The chain cleanups of nodes run without run(), which ends with them (a folder's shared nodes, folder_videos.py).
        """
        self.__run_cleanups(preserve_error)

    def resume(self):
        logger.debug(f"Resuming executor {self.id}")
        self.progress.resume()

    def pause(self):
        logger.debug(f"Pausing executor {self.id}")
        self.progress.pause()

    def kill(self):
        logger.debug(f"Killing executor {self.id}")
        self.progress.abort()

    def __send_chain_start(self):
        nodes = set(self.chain.nodes.keys())
        nodes.difference_update(self.node_cache.keys())
        self.queue.put({"event": "chain-start", "data": {"nodes": list(nodes)}})

    def __send_node_start(self, node: Node):
        self.queue.put({"event": "node-start", "data": {"nodeId": node.id}})

    def __send_node_progress(
        self, node: Node, times: Sequence[float], index: int, length: int
    ):

        def get_eta(times: Sequence[float]) -> float:
            avg_time = 0
            if len(times) > 0:
                times = times[-100:]
                weights = [max(1 / i, 0.9**i) for i in range(len(times), 0, -1)]
                avg_time = sum((t * w for t, w in zip(times, weights))) / sum(weights)
            remaining = max(0, length - index)
            return avg_time * remaining

        self.queue.put(
            {
                "event": "node-progress",
                "data": {
                    "nodeId": node.id,
                    "progress": 1 if length == 0 else index / length,
                    "index": index,
                    "total": length,
                    "eta": get_eta(times),
                },
            }
        )

    def __send_node_progress_done(self, node: Node, length: int):
        self.queue.put(
            {
                "event": "node-progress",
                "data": {
                    "nodeId": node.id,
                    "progress": 1,
                    "index": length,
                    "total": length,
                    "eta": 0,
                },
            }
        )

    async def __send_node_broadcast(
        self, node: Node, output: Output, generators: Iterable[Generator] | None = None
    ):

        def compute_broadcast_data():
            if self.progress.aborted:
                return None
            foo = compute_broadcast(output, node.data.outputs)
            if generators is None:
                return (*foo, {}, {})
            return (
                *foo,
                *compute_sequence_broadcast(generators, node.data.iterable_outputs),
            )

        async def send_broadcast():
            result = await await_owned_node(
                self.loop.run_in_executor(self.pool, compute_broadcast_data)
            )
            if result is None or self.progress.aborted:
                return
            data, types, sequence_types, item_types = result
            for output_id, type in item_types.items():
                types[output_id] = type
            evant_data: NodeBroadcastData = {
                "nodeId": node.id,
                "data": data,
                "types": types,
                "sequenceTypes": sequence_types,
            }
            self.queue.put({"event": "node-broadcast", "data": evant_data})

        if self.send_broadcast_data and len(node.data.outputs) > 0:
            if generators is not None:
                await self.flush_broadcasts()
                await send_broadcast()
            else:
                self.__broadcasts.submit(node.id, send_broadcast)

    def __send_node_finish(self, node: Node, execution_time: float):
        self.queue.put(
            {
                "event": "node-finish",
                "data": {"nodeId": node.id, "executionTime": execution_time},
            }
        )
