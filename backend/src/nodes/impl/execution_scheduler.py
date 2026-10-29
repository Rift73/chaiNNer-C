"""Bounded, owned scheduling for the retained node executor."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterator
from typing import Literal, Protocol, TypeVar, cast

from api import NodeData, NodeId
from chain.chain import Chain
from chain.input import EdgeInput, Input, InputMap, ValueInput

from .execution_cleanup import await_owned_node


class CacheView(Protocol):
    def has(self, node_id: NodeId) -> bool: ...


class DependencyExecutor(Protocol):
    @property
    def chain(self) -> Chain: ...

    @property
    def inputs(self) -> InputMap: ...

    @property
    def node_cache(self) -> CacheView: ...


T = TypeVar("T")


def advance_iterator(iterator: Iterator[T]) -> tuple[bool, T | None]:
    # StopIteration cannot cross an asyncio Future boundary.
    try:
        return True, next(iterator)
    except StopIteration:
        return False, None


async def owned_gather(coroutines: list[Awaitable[T]]) -> list[T]:
    """Drain every started branch before cleanup, including on cancellation.

    A lone branch is awaited directly, like an input that is not gathered: it has
    no sibling to drain or to order errors against. Draining its own node stays
    the branch's job (NodeFlights and await_owned_node).
    """
    if len(coroutines) == 1:
        return [await coroutines[0]]

    async def capture(
        coroutine: Awaitable[T],
    ) -> tuple[Literal[True], T] | tuple[Literal[False], BaseException]:
        try:
            return True, await coroutine
        except BaseException as error:
            return False, error

    tasks = [asyncio.create_task(capture(coroutine)) for coroutine in coroutines]
    pending = asyncio.gather(*tasks)
    results = await await_owned_node(pending)
    # Stable input order determines which error is reported, not worker timing.
    for result in results:
        if result[0] is False:
            try:
                raise result[1]
            finally:
                # Like await_owned_node's future: the error's traceback holds this
                # frame, so the frame drops every local that holds the error.
                del tasks, pending, results, result
    return [cast(T, value) for _, value in results]


# Nodes outside the image/utility CPU rule that may still run ahead of earlier items
# or beside other branches: free of side effects, they use their context only per
# call (settings, node id, progress) and serialize their device work under their own
# lock (native/reports/sp3/thread-safety-audit.md, "Locked device nodes"; PyTorch,
# ONNX and NCNN on their device's thread, nodes/impl/device_thread.py).
LOCKED_DEVICE_NODES: frozenset[str] = frozenset(
    {
        "chainner:tensorrt:upscale_image",
        "chainner:migraphx:upscale_image",
        "chainner:pytorch:upscale_image",
        "chainner:onnx:upscale_image",
        "chainner:ncnn:upscale_image",
    }
)

# The lazy inputs their node always reads, so the item window computes ahead behind them
# as behind any input: per schema id, each with the boolean input that, on or
# connected, may leave it unread. Every other lazy input may go unread (a
# Conditional's If True and If False, Logic Operation's B), and the window computes no
# locked-device node behind it for an item that may never read it.
ALWAYS_READ_LAZY_INPUTS: dict[str, dict[int, int]] = {
    # The image, unless Skip existing files finds the file
    "chainner:image:save": {0: 1000},
}


def may_go_unread(data: NodeData, sources: list[Input], index: int) -> bool:
    """Whether the node may leave its lazy input data.inputs[index] unread for an item:
    one not in ALWAYS_READ_LAZY_INPUTS, or one whose boolean input is on or connected.
    sources: the node's inputs (InputMap.get)."""
    switch = ALWAYS_READ_LAZY_INPUTS.get(data.schema_id, {}).get(data.inputs[index].id)
    if switch is None:
        return True
    source = next(s for i, s in zip(data.inputs, sources) if i.id == switch)
    return not isinstance(source, ValueInput) or bool(source.value)


def pure_cpu_node(data: NodeData) -> bool:
    """An eager, side-effect-free CPU node (image or utility, without a context), or an
    audited locked-device node: one that may run beside others and ahead of earlier items."""
    locked_device = data.schema_id in LOCKED_DEVICE_NODES
    return not (
        data.kind != "regularNode"
        or data.side_effects
        or (data.node_context and not locked_device)
        or not (
            locked_device
            or data.schema_id.startswith(("chainner:image:", "chainner:utility:"))
        )
        or any(item.lazy for item in data.inputs)
    )


def pure_cpu_dependency(
    executor: DependencyExecutor, node_id: NodeId, visiting: set[NodeId] | None = None
) -> bool:
    """Parallelize only eager, side-effect-free CPU subgraphs with ready roots.

    A cached node is ready; an uncached pure node is if all its edge inputs are, read in
    order up to the first that is not. The walk keeps its own stack: a chain may have up
    to MAX_CHAIN_DEPTH pure nodes in a row, beyond Python's recursion limit."""
    visiting = set() if visiting is None else visiting
    known: dict[NodeId, bool] = {}

    def settled(current: NodeId) -> bool | None:
        """The node's answer without walking its inputs, None when they decide it."""
        if current in known:
            return known[current]
        if executor.node_cache.has(current):
            return True
        if not pure_cpu_node(executor.chain.nodes[current].data):
            return False
        if current in visiting:
            return False
        return None

    answer = settled(node_id)
    if answer is not None:
        return answer
    # The nodes being walked, each with its inputs not read yet.
    stack: list[tuple[NodeId, Iterator[Input]]] = []

    def enter(current: NodeId) -> None:
        visiting.add(current)
        stack.append((current, iter(executor.inputs.get(current))))

    enter(node_id)
    try:
        while stack:
            current, sources = stack[-1]
            # answer: the last input's, after an input that was walked; a False one
            # decides the node without its other inputs.
            if answer is not False:
                answer = True
                for value in sources:
                    if isinstance(value, EdgeInput):
                        answer = settled(value.id)
                        if answer is None:
                            enter(value.id)
                            break
                        if not answer:
                            break
                if answer is None:
                    continue
            known[current] = answer
            visiting.remove(current)
            stack.pop()
        return answer is True
    finally:
        for current, _ in stack:
            visiting.discard(current)


class NodeFlights:
    """Share only in-flight work; completed values belong to the counted cache."""

    def __init__(self):
        self.pending: dict[str, asyncio.Future] = {}

    async def run(self, key: str, run: Callable[[], Awaitable[T]]) -> T:
        task = self.pending.get(key)
        if task is None:
            task = asyncio.ensure_future(run())
            self.pending[key] = task
        try:
            return await await_owned_node(task)
        finally:
            if task.done() and self.pending.get(key) is task:
                del self.pending[key]
            # A failed task holds its error, whose traceback holds this frame.
            del task


class LatestBroadcasts:
    """One active computation and one latest pending output per node."""

    def __init__(self):
        self.pending: OrderedDict[str, Callable[[], Awaitable[None]]] = OrderedDict()
        self.task: asyncio.Task[None] | None = None
        self.error: BaseException | None = None

    def submit(self, key: str, send: Callable[[], Awaitable[None]]) -> None:
        if self.error is not None:
            raise self.error
        # Replacing a queued preview releases the superseded frame immediately.
        self.pending[key] = send
        if self.task is None:
            self.task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        try:
            while self.pending:
                _, send = self.pending.popitem(last=False)
                await send()
        except BaseException as error:
            self.error = error
            self.pending.clear()
        finally:
            self.task = None

    async def flush(self) -> None:
        while self.task is not None:
            await await_owned_node(self.task)
        if self.error is not None:
            error, self.error = self.error, None
            raise error
