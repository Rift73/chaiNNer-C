from __future__ import annotations

import asyncio
import functools
import sys
import time
import typing
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, List, Literal, Mapping, NewType, Sequence, Union

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
    IteratorOutputInfo,
    IterOutputId,
    Lazy,
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
from events import EventConsumer, InputsDict, NodeBroadcastData
from nodes.impl import native_profile
from nodes.impl.execution_cleanup import (
    await_owned_node,
    run_cleanup_functions,
    run_cleanup_groups,
)
from nodes.impl.execution_scheduler import (
    LatestBroadcasts,
    NodeFlights,
    advance_iterator,
    owned_gather,
    pure_cpu_dependency,
)
from nodes.impl.item_window import (
    Candidate,
    ItemWindow,
    Phases,
    Prepared,
    SlotGenerator,
    Stored,
    affinity_cpus,
    engine_jobs,
    memory_budget,
    node_phases,
    output_nbytes,
    read_ahead_blocker,
    serial_producer,
    window_size,
)
from progress_controller import Aborted, ProgressController, ProgressToken
from util import combine_sets, timed_supplier

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
        output[i] = o.enforce(output[i])
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
            partial[i] = o.enforce(rest.pop(0))
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


def registered_phases(node: NodeData) -> Phases:
    phases = node_phases(node.schema_id)
    assert phases is not None, f"{node.schema_id} has no prepare and commit phases"
    return phases


def phase_inputs(node: NodeData) -> list[int]:
    """
    The indices of the inputs a prepare phase receives (SP3-P9): a collector's iterated inputs, as on_iterate does, else all of them.
    """
    if node.kind == "collector":
        return [
            index
            for index, i in enumerate(node.inputs)
            if i.id in node.single_iterable_input.inputs
        ]
    return list(range(len(node.inputs)))


def prepare_node(node: NodeData, inputs: list[object]) -> object | None:
    """
    A side-effect node's prepare phase (SP3-P9): its inputs enforced as the node would receive them, a lazy input as the value it produces.
    """
    return registered_phases(node).prepare(
        [
            node.inputs[index].enforce_(value)
            for index, value in zip(phase_inputs(node), inputs)
        ]
    )


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

    enforced_inputs: list[object] = []
    try:
        for i in node.data.inputs:
            if i.id in iterable_input.inputs:
                enforced_inputs.append(i.enforce_(inputs[len(enforced_inputs)]))
    except Exception as e:
        input_dict = collect_input_information(
            node.data, get_partial_inputs(inputs), enforced=False
        )
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
        input_dict = collect_input_information(
            node.data, get_partial_inputs(enforced_inputs)
        )
        raise NodeExecutionError(node.id, node.data, str(e), input_dict) from e


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
            logger.error(f"Error broadcasting output: {e}")
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
            logger.error(f"Error broadcasting output: {e}")
    return (sequence_types, item_types)


class NodeExecutionError(Exception):
    def __init__(
        self, node_id: NodeId, node_data: NodeData, cause: str, inputs: InputsDict
    ):
        super().__init__(cause)
        self.node_id: NodeId = node_id
        self.node_data: NodeData = node_data
        self.inputs: InputsDict = inputs


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


class _ExecutorNodeContext(NodeContext):
    def __init__(
        self,
        progress: ProgressToken,
        settings: SettingsParser,
        storage_dir: Path,
        node_id: NodeId,
    ) -> None:
        super().__init__()
        self.progress = progress
        self.__settings = settings
        self._storage_dir = storage_dir
        self._node_id = node_id
        self.chain_cleanup_fns: set[Callable[[], None]] = set()
        self.node_cleanup_fns: set[Callable[[], None]] = set()

    @property
    def aborted(self) -> bool:
        return self.progress.aborted

    @property
    def paused(self) -> bool:
        time.sleep(0)
        return self.progress.paused

    def set_progress(self, progress: float) -> None:
        self.check_aborted()

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

    def add_cleanup(
        self, fn: Callable[[], None], after: Literal["node", "chain"] = "chain"
    ) -> None:
        if after == "chain":
            self.chain_cleanup_fns.add(fn)
        elif after == "node":
            self.node_cleanup_fns.add(fn)
        else:
            raise ValueError(f"Unknown cleanup type: {after}")


@dataclass(frozen=True)
class _ItemCache:
    """
    The cache test of one later item (spec 4.4): its generator values and stored results, and the shared static cache for nodes that are not iterated. The live cache's iterated entries belong to the committing item.
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


# Into Directory resolves the on-disk spelling of a folder that earlier items' saves may create, so running it ahead could change its output (SP3 Task 4 audit).
_NOT_SPECULATED: frozenset[str] = frozenset({"chainner:utility:into_directory"})


class _ItemPlanner:
    """
    Which iterated nodes of a generator group run ahead for a later item, and their jobs (spec 4.4); then the prepare phases of its sinks that registered them (spec 4.7).
    """

    def __init__(
        self,
        chain: Chain,
        inputs: InputMap,
        node_cache: OutputCache[NodeOutput],
        generators: set[NodeId],
        iterated: set[NodeId],
        sinks: list[Node],
        new_context: Callable[[Node], _ExecutorNodeContext],
    ) -> None:
        self.__chain = chain
        self.__inputs = inputs
        self.__cache = node_cache
        self.__generators = generators
        self.__iterated = iterated
        self.__new_context = new_context
        ancestors: set[NodeId] = set()
        pending = [node.id for node in sinks]
        while pending:
            for edge in chain.edges_to(pending.pop()):
                if edge.source.id not in ancestors:
                    ancestors.add(edge.source.id)
                    pending.append(edge.source.id)

        def depends(sources: Iterable[Input]) -> frozenset[NodeId]:
            return frozenset(
                i.id
                for i in sources
                if isinstance(i, EdgeInput)
                and i.id in iterated
                and (i.id not in generators)
            )

        self.__candidates = [
            Candidate(node_id, depends(inputs.get(node_id)))
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

    @property
    def candidates(self) -> Sequence[Candidate]:
        return self.__candidates

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
        if self.__chain.nodes[node_id].data.schema_id in _NOT_SPECULATED:
            return False
        available = {*values, *(n for n, s in store.items() if s.error is None)}
        view = _ItemView(
            self.__chain,
            self.__inputs,
            _ItemCache(available, self.__iterated, self.__cache),
        )
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
            return timed_supplier(functools.partial(prepare_node, node.data, inputs))
        return timed_supplier(
            functools.partial(
                run_node, node.data, self.__new_context(node), inputs, node_id
            )
        )

    def nbytes(self, value: object) -> int:
        return output_nbytes(
            value.output if isinstance(value, RegularOutput) else [value]
        )


class _SplitIterator(typing.Protocol):
    """
    Load Images' native iterator (SP3b spec 4.1): describe() names its next item and raises StopIteration at the end; materialize(token) loads that item and returns an error as its value, as __next__ yields it.
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
        self.cache_strategy: dict[NodeId, CacheStrategy] = get_cache_strategies(chain)
        self._storage_dir = storage_dir
        self.__window: ItemWindow | None = None
        self.__collector_directories: dict[NodeId, dict[int, object]] = {}
        self.__item_bytes: dict[NodeId, int] | None = None
        self.__budget: int | None = None

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

    async def __resolve_node_input(self, node_input: Input) -> object:
        if isinstance(node_input, EdgeInput):
            return await self.__get_node_output(node_input.id, node_input.index)
        else:
            return node_input.value

    async def __gather_inputs(self, node: Node) -> list[object]:
        """
        Returns the list of input values for the given node.
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
                    self.__resolve_node_input(assigned_inputs[input_index]), self.loop
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
        return _ExecutorNodeContext(self.progress, settings, self._storage_dir, node.id)

    def __get_node_context(self, node: Node) -> _ExecutorNodeContext:
        context = self.__context_cache.get(node.id, None)
        if context is None:
            context = self.__new_node_context(node)
            self.__context_cache[node.id] = context
        return context

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
        inputs = await self.__gather_inputs(node)
        if isinstance(node, CollectorNode):
            self.__collector_directories[node.id] = {
                index: inputs[index]
                for index, i in enumerate(node.data.inputs)
                if i.kind == "directory"
            }
        context = self.__get_node_context(node)

        def get_lazy_evaluation_time():
            return sum(i.evaluation_time for i in inputs if isinstance(i, Lazy))

        await self.progress.suspend()
        self.__send_node_start(node)
        await self.progress.suspend()
        lazy_time_before = get_lazy_evaluation_time()
        window = self.__window
        if window is not None and window.holds(node.id):
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
            run_cleanup_functions(
                context.node_cleanup_fns, preserve_error=sys.exc_info()[0] is not None
            )
        lazy_time_after = get_lazy_evaluation_time()
        execution_time -= lazy_time_after - lazy_time_before
        if isinstance(output, RegularOutput):
            await self.__send_node_broadcast(node, output.output)
            self.__send_node_finish(node, execution_time)
        elif isinstance(output, GeneratorOutput):
            await self.__send_node_broadcast(
                node, output.partial_output, generators=[output.generator]
            )
        if perform_cache and (not isinstance(output, CollectorOutput)):
            self.node_cache.set(node.id, output, self.cache_strategy[node.id])
        if self.__item_bytes is not None and isinstance(output, RegularOutput):
            self.__item_bytes[node.id] = output_nbytes(output.output)
        await self.progress.suspend()
        return output

    def __get_iterated_nodes(
        self, node: GeneratorNode
    ) -> tuple[set[CollectorNode], set[FunctionNode], set[Node]]:
        """
        Returns all collector and output nodes iterated by the given generator node
        """
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
                assert isinstance(n, FunctionNode)
                if n.has_side_effects():
                    output_nodes.add(n)
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
                output[index] = o.enforce(values_list.pop(0))
        return output

    def __describe(
        self, generator_supplier: typing.Iterator[Output | Exception]
    ) -> tuple[object] | None:
        """
        The generator's next item as a token (SP3b P1): the 1-tuple of the values it yields, so a yielded None stays an item; None once it is exhausted. A yielded error is raised.
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
            raise values
        return RegularOutput(
            self.__generator_fill_partial_output(
                node, generator_output.partial_output, values
            )
        )

    def __writer_directories(
        self,
        writers: list[Node],
        generator_outputs: dict[NodeId, GeneratorOutput],
        all_iterated_nodes: set[NodeId],
    ) -> list[tuple[NodeId, Path | None]]:
        """
        Returns the base directory of every directory input of the given writers (SP3-P4).

        Nothing is computed: an input fed by an iterated output or node, a value missing
        from the cache, or a value that is not a path is unresolved (None).
        """
        directories: list[tuple[NodeId, Path | None]] = []
        for node in sorted(writers, key=lambda n: n.id):
            assigned_inputs = self.inputs.get(node.id)
            for index, i in enumerate(node.data.inputs):
                if i.kind != "directory":
                    continue
                node_input = assigned_inputs[index]
                if not isinstance(node_input, EdgeInput):
                    value = node_input.value
                elif node_input.id in generator_outputs:
                    source = self.chain.nodes[node_input.id]
                    iterated = (
                        source.data.outputs[node_input.index].id
                        in source.data.single_iterable_output.outputs
                    )
                    value = (
                        None
                        if iterated
                        else _output_value(
                            generator_outputs[node_input.id], node_input.index
                        )
                    )
                elif node_input.id in all_iterated_nodes:
                    value = None
                elif isinstance(node, CollectorNode):
                    value = self.__collector_directories[node.id][index]
                else:
                    cached = self.node_cache.peek(node_input.id)
                    value = (
                        None
                        if cached is None
                        else _output_value(cached, node_input.index)
                    )
                directories.append(
                    (node.id, Path(value) if isinstance(value, (str, Path)) else None)
                )
        return directories

    async def __open_window(
        self,
        generator_nodes: list[GeneratorNode],
        generator_outputs: dict[NodeId, GeneratorOutput],
        output_nodes: set[FunctionNode],
        collectors: list[tuple[Collector, _Timer, CollectorNode]],
        all_iterated_nodes: set[NodeId],
        generator_suppliers: dict[NodeId, typing.Iterator[Output | Exception]],
        stops: int,
    ) -> ItemWindow | None:
        """
        Decides the group's item window K at slot 1 and logs it (spec 4.2, SP3-P1).

        K = 1 builds no window and resolves no path.
        """
        measured, self.__item_bytes = (self.__item_bytes, None)
        k, reason = window_size(affinity_cpus())
        if k > 1 and self.__pool_size is None:
            k, reason = (1, "pool size unknown")
        if k > 1:
            writers: list[Node] = [
                *output_nodes,
                *(collector_node for _, _, collector_node in collectors),
            ]
            sources = [
                (node.id, generator_outputs[node.id].generator.source_paths)
                for node in generator_nodes
            ]
            directories = self.__writer_directories(
                writers, generator_outputs, all_iterated_nodes
            )
            blocker = await await_owned_node(
                self.loop.run_in_executor(
                    self.pool,
                    functools.partial(read_ahead_blocker, sources, directories),
                )
            )
            if blocker is not None:
                k, reason = (1, blocker)
        logger.info("item window K=%d (%s)", k, reason)
        if k == 1:
            return None
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
            # The group's single generator, Range aside, is described serially and materialized as engine jobs (SP3b spec 4.2). A split iterator (Load Images) names each item, then loads it; any other yields it at describe, and its enforce (an image's normalize) is the materialize.
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
        planner = _ItemPlanner(
            self.chain,
            self.inputs,
            self.node_cache,
            {node.id for node in generator_nodes},
            all_iterated_nodes,
            sinks,
            self.__new_node_context,
        )
        return ItemWindow(
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
        )

    async def __close_window(self) -> None:
        self.__item_bytes = None
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

    async def __iterate_generator_nodes(self, generator_nodes: list[GeneratorNode]):
        await self.progress.suspend()
        num_generators = len(generator_nodes)
        expected_lengths = {}
        collectors: list[tuple[Collector, _Timer, CollectorNode]] = []
        output_nodes: set[FunctionNode] = set()
        all_iterated_nodes: set[NodeId] = set()
        generator_suppliers: dict[NodeId, typing.Iterator[Output | Exception]] = {}
        generator_outputs: dict[NodeId, GeneratorOutput] = {}
        iter_timers: dict[NodeId, _IterationTimer] = {}
        for node in generator_nodes:
            generator_output = await self.process_generator_node(node)
            generator_outputs[node.id] = generator_output
            generator_suppliers[node.id] = (
                generator_output.generator.supplier().__iter__()
            )
            collector_nodes, __output_nodes, __all_iterated_nodes = (
                self.__get_iterated_nodes(node)
            )
            for iterated_node in __all_iterated_nodes:
                all_iterated_nodes.add(iterated_node.id)
            for o_node in __output_nodes:
                output_nodes.add(o_node)
            if len(collector_nodes) == 0 and len(output_nodes) == 0:
                return
            for collector_node in collector_nodes:
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
        if not len(set(expected_lengths.values())) <= 1:
            raise AssertionError(
                "Expected all connected iterators to have the same length"
            )
        total_stopiters = 0
        deferred_errors: list[str] = []
        self.__item_bytes = {}
        slot = -1
        while True:
            slot += 1
            if slot == 1:
                self.__window = await self.__open_window(
                    generator_nodes,
                    generator_outputs,
                    output_nodes,
                    collectors,
                    all_iterated_nodes,
                    generator_suppliers,
                    total_stopiters,
                )
            window = self.__window
            if window is not None:
                window.begin(slot)
            generator_output = None
            try:
                for node in generator_nodes:
                    await self.progress.suspend()
                    generator_output = generator_outputs[node.id]
                    generator_supplier = generator_suppliers[node.id]
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
                                RegularOutput | None, await window.take_advance(node.id)
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
                    if iter_output is None:
                        raise StopIteration
                    if slot == 0:
                        self.__item_bytes[node.id] = output_nbytes(iter_output.output)
                    await self.progress.suspend()
                    self.node_cache.set(node.id, iter_output, StaticCaching)
                    await self.__send_node_broadcast(node, iter_output.output)
                for output_node in output_nodes:
                    await self.process_regular_node(output_node)
                for collector, timer, collector_node in collectors:
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
                for node in generator_nodes:
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
                    return enforce_output(collector.on_complete(), collector_node.data)

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
        self.__send_chain_start()
        generator_nodes: list[GeneratorNode] = []
        gens_by_outs: dict[NodeId, set[NodeId]] = {}
        for node_id in self.chain.topological_order():
            node = self.chain.nodes[node_id]
            if isinstance(node, GeneratorNode):
                generator_nodes.append(node)
                collector_nodes, output_nodes, __all_iterated_nodes = (
                    self.__get_iterated_nodes(node)
                )
                for collector in collector_nodes:
                    if gens_by_outs.get(collector.id, None) is not None:
                        gens_by_outs[collector.id].add(node.id)
                    else:
                        gens_by_outs[collector.id] = {node.id}
                for out_node in output_nodes:
                    if gens_by_outs.get(out_node.id, None) is not None:
                        gens_by_outs[out_node.id].add(node.id)
                    else:
                        gens_by_outs[out_node.id] = {node.id}
        groups: list[set[NodeId]] = list(gens_by_outs.values())
        combined_groups = combine_sets(groups)
        for group in combined_groups:
            nodes_to_run: list[GeneratorNode] = []
            for node_id in group:
                generator_node = self.chain.nodes[node_id]
                if isinstance(generator_node, GeneratorNode):
                    nodes_to_run.append(generator_node)
            try:
                await self.__iterate_generator_nodes(nodes_to_run)
            finally:
                await self.__close_window()
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
                try:
                    self.node_cache.clear()
                    run_cleanup_groups(
                        (
                            callbacks
                            for context in self.__context_cache.values()
                            for callbacks in (
                                context.node_cleanup_fns,
                                context.chain_cleanup_fns,
                            )
                        ),
                        preserve_error=preserve_error or sys.exc_info()[0] is not None,
                    )
                finally:
                    pass

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
