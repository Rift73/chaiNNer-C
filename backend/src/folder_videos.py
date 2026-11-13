"""A folder of videos in one /run: Load Videos.

The part of the chain that Load Videos feeds runs whole, on the run's executor, once per file, as the chain with a Load
Video on that file would: the same plan, item window, Save Video and cleanups, and no second path through the engine.
Everything else runs once: the nodes feeding that part (a model's Load Model, so its engines stay loaded from one file to
the next) first, as a preview does, and their outputs reach each file's run as a preview's would; then the rest of the
chain; then the files. To the UI, this is one run: one chain-start, and Load Videos' progress is the whole folder's.
"""

from __future__ import annotations

import asyncio
import functools
import gc
import itertools
import os
import sys
import time
from collections.abc import Callable, Container, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from sanic.log import logger

from api import NodeId, OutputId
from chain.cache import OutputCache
from chain.chain import Chain, CollectorNode, FunctionNode, GeneratorNode
from chain.input import EdgeInput, InputMap
from events import Event, EventConsumer
from native_executor import after_ending_runs, ending_runs
from nodes.utils.utils import split_file_path
from process import (
    ExecutionId,
    NodeExecutionError,
    NodeOutput,
    RegularOutput,
    _colliding_files,
    _collision_message,
    _LoopItems,
    _output_value,
    _reached,
    _upstream,
    enforce_inputs,
    loop_node_error,
)
from progress_controller import Aborted

LOAD_VIDEOS = "chainner:image:load_videos"
LOAD_VIDEO = "chainner:image:load_video"
T = TypeVar("T")
# Read from sys.modules: a chain with its node has it loaded, and the server starts before the nodes load.
NODE_MODULE = "packages.chaiNNer_standard.image.video_frames.load_videos"


def run_executor(executor_class: Callable[..., Any], **arguments: Any) -> Any:
    """The executor of a /run: the chain's own, or a FolderVideosRun over it for a chain with Load Videos."""
    chain: Chain = arguments["chain"]
    if chain.nodes_with_schema_id(LOAD_VIDEOS):
        return FolderVideosRun(executor_class, arguments)
    return executor_class(**arguments)


def _downstream(chain: Chain, start: Iterable[NodeId]) -> set[NodeId]:
    return _reached(start, lambda n: [e.target.id for e in chain.edges_from(n)])


def _eager_upstream(
    chain: Chain, start: Iterable[NodeId], computed: Container[NodeId]
) -> set[NodeId]:
    """`start` and the nodes they need, up to those already computed: not the sources of a lazy input, which its node
    may leave unread."""

    def eager_sources(node_id: NodeId) -> list[NodeId]:
        if node_id in computed:
            return []
        lazy = {i.id for i in chain.nodes[node_id].data.inputs if i.lazy}
        return [
            e.source.id
            for e in chain.edges_to(node_id)
            if e.target.input_id not in lazy
        ]

    return _reached(start, eager_sources)


def _sub_chain(chain: Chain, keep: set[NodeId]) -> Chain:
    """The chain's nodes in `keep`, with their inputs and the connections between them, in the chain's order."""
    sub = Chain()
    for node_id, node in chain.nodes.items():
        if node_id in keep:
            sub.add_node(node)
            sub.inputs.inputs[node_id] = dict(chain.inputs.inputs.get(node_id, {}))
    for node_id in chain.nodes:
        if node_id in keep:
            for edge in chain.edges_from(node_id):
                if edge.target.id in keep:
                    sub.add_edge(edge)
    return sub


@dataclass(frozen=True)
class _Folder:
    node: GeneratorNode
    # The nodes that run once per file: what Load Videos feeds, and the generators zipped with it with what they feed.
    per_file: set[NodeId]
    # The nodes outside them that a file's run reads as computed.
    shared: set[NodeId]
    # A file's chain: those, and what feeds the nodes it reads through a lazy input, which its nodes may leave unread.
    chain_nodes: set[NodeId]


def _plan_folder(chain: Chain, node: GeneratorNode) -> _Folder:
    def feeders(node_id: NodeId) -> list[NodeId]:
        # A file's Load Video does not read what feeds Load Videos: only the file list does.
        return (
            [] if node_id == node.id else [e.source.id for e in chain.edges_to(node_id)]
        )

    for edge in chain.edges_to(node.id):
        for source in _upstream(chain, [edge.source.id]):
            if isinstance(chain.nodes[source], CollectorNode):
                label = next(
                    i.label for i in node.data.inputs if i.id == edge.target.input_id
                )
                raise NodeExecutionError(
                    node.id,
                    node.data,
                    f"{node.data.name}' {label} reads {chain.nodes[source].data.name},"
                    " which ends with a loop of its own, and the file list comes before"
                    f" any loop runs. Give {label} a value, or a node that needs no loop.",
                    {},
                )
    per_file = _downstream(chain, [node.id])
    while True:
        generators = {
            node_id
            for node_id in _reached(per_file, feeders) - per_file
            if isinstance(chain.nodes[node_id], GeneratorNode)
        }
        if not generators:
            break
        per_file |= _downstream(chain, generators)
    for other in per_file:
        if other != node.id and chain.nodes[other].schema_id == LOAD_VIDEOS:
            raise NodeExecutionError(
                other,
                chain.nodes[other].data,
                "Two Load Videos cannot feed one node, nor one feed the other: each"
                " runs its nodes once per file of its own folder.",
                {},
            )
    shared: set[NodeId] = set()
    read_lazily: set[NodeId] = set()
    for node_id in per_file - {node.id}:
        lazy = {i.id for i in chain.nodes[node_id].data.inputs if i.lazy}
        for edge in chain.edges_to(node_id):
            if edge.source.id not in per_file:
                (read_lazily if edge.target.input_id in lazy else shared).add(
                    edge.source.id
                )
    return _Folder(
        node, per_file, shared, per_file | shared | _upstream(chain, read_lazily)
    )


@dataclass(frozen=True)
class _Listing:
    """A folder's files, as its run goes through them."""

    folder: _Folder
    paths: list[Path]
    weights: list[int]
    directory: Path
    fail_fast: bool


def _video_items(listing: _Listing) -> _LoopItems:
    """A folder's files: Load Videos' outputs that follow from a file's path, and frame 0's Index, as every file has
    one."""
    module = sys.modules[NODE_MODULE]

    def named(_position: int, path: Path) -> dict[OutputId, object]:
        directory, name, _ = split_file_path(path)
        return {
            module.INDEX: 0,
            module.VIDEO_DIRECTORY: directory,
            module.NAME: name,
            module.SUBDIRECTORY: module.subdirectory_of(path, listing.directory),
            module.VIDEO_COUNT: len(listing.paths),
        }

    return _LoopItems(
        listing.folder.node, listing.folder.per_file, listing.paths, named
    )


def _file_chain(
    chain: Chain, folder: _Folder, path: Path, directory: Path, count: int
) -> Chain:
    """The chain of one file's run: its Load Video stands where Load Videos stands, with the same id, read options and
    the outputs they share. Load Videos' own outputs are constants for this file."""
    module = sys.modules[NODE_MODULE]
    single = _sub_chain(chain, folder.chain_nodes)
    node_id = folder.node.id
    single.nodes[node_id] = GeneratorNode(node_id, LOAD_VIDEO)
    for edge in list(single.edges_to(node_id)):
        single.remove_edge(edge)
    single.inputs.inputs[node_id].update(
        {module.VIDEO_FILE: str(path), **module.WHOLE_VIDEO}
    )
    constants = {
        module.SUBDIRECTORY: module.subdirectory_of(path, directory),
        module.VIDEO_COUNT: count,
    }
    for output_id, value in constants.items():
        for edge in list(single.edges_from(node_id, output_id)):
            single.remove_edge(edge)
            single.inputs.set(edge.target.id, edge.target.input_id, value)
    return single


def _file_failure(error: Exception, path: Path) -> Exception:
    """A file's run's error, as Stop on first error reports it: naming the file, as a loop's item does."""
    message = f"{path}: {error}"
    if isinstance(error, NodeExecutionError):
        return NodeExecutionError(error.node_id, error.node_data, message, error.inputs)
    return RuntimeError(message)


def _listing(module: Any, inputs: list[object]) -> tuple[list[Path], list[int]]:
    """The folder's files, each with its share of the folder's progress: its size, a stat away where its frame count
    takes a probe. A file that cannot be read weighs one byte: its run reports why."""

    def size(path: Path) -> int:
        try:
            return max(1, os.stat(path).st_size)
        except OSError:
            return 1

    paths = module.list_videos(*inputs)
    return paths, [size(path) for path in paths]


class _FolderProgress:
    """Load Videos' progress bar over a folder: the video it is on of how many, the share of the folder's bytes done, and
    the time left at the pace so far."""

    def __init__(
        self, node_id: NodeId, weights: list[int], seconds: Callable[[], float]
    ) -> None:
        self.node_id = node_id
        self.__weights = weights
        # Before each file, and the folder's at the end.
        self.__before = [0, *itertools.accumulate(weights)]
        self.__seconds = seconds
        self.__started = seconds()
        self.file = 0

    @property
    def elapsed(self) -> float:
        return self.__seconds() - self.__started

    def event(self, share: float) -> Event:
        """The folder's progress with this much of the current file done."""
        done = (
            self.__before[self.file] + self.__weights[self.file] * share
        ) / self.__before[-1]
        return {
            "event": "node-progress",
            "data": {
                "nodeId": self.node_id,
                "progress": done,
                "index": self.file + 1,
                "total": len(self.__weights),
                "eta": self.elapsed * (1 - done) / done if done > 0 else 0,
            },
        }


class _RunEvents(EventConsumer):
    """A file's run's events as the one run's: only this run's own chain-start goes on, and Load Videos starts once, ends
    once and shows the whole folder's progress."""

    def __init__(self, queue: EventConsumer) -> None:
        self.queue = queue
        self.folder: _FolderProgress | None = None
        # The generators read before a later run starts them: that run's events are theirs.
        self.quiet: set[NodeId] = set()

    def put(self, event: Event) -> None:
        kind = event["event"]
        if kind == "chain-start":
            return
        data: Any = event["data"]
        if isinstance(data, dict) and data.get("nodeId") in self.quiet:
            return
        folder = self.folder
        if folder is not None and kind in (
            "node-start",
            "node-finish",
            "node-progress",
        ):
            if data["nodeId"] == folder.node_id:
                if kind == "node-progress":
                    total = data["total"]
                    share = data["index"] / total if total > 0 else 1
                    self.queue.put(folder.event(share))
                    return
                if kind == "node-finish" or folder.file > 0:
                    return
        self.queue.put(event)


def _log_left_job_error(done: asyncio.Future[object]) -> None:
    """Reads the result of a job that Stop left running, which nothing else does: its error is logged, never dropped."""
    error = None if done.cancelled() else done.exception()
    if error is not None:
        logger.error("A stopped run's job failed: %s", error, exc_info=error)


class FolderVideosRun:
    """process.Executor's interface over a chain's folders of videos (server.py's /run, /pause, /resume, /kill and
    /status). `arguments`: the executor's, one executor per file."""

    def __init__(
        self, executor_class: Callable[..., Any], arguments: dict[str, Any]
    ) -> None:
        self.__executor_class = executor_class
        self.__arguments = arguments
        self.__chain: Chain = arguments["chain"]
        self.__queue: EventConsumer = arguments["queue"]
        self.__events = _RunEvents(self.__queue)
        self.__current: Any = None
        # The executors of the shared nodes, whose chain cleanups wait for the run's end.
        self.__computing: list[Any] = []
        self.__runs = 0
        self.__shared: dict[NodeId, NodeOutput] = {}
        self.__aborted = False
        # Set by kill(), on any thread.
        self.__stopped: asyncio.Future[None] = arguments["loop"].create_future()
        self.__paused_since: float | None = None
        self.__time_paused = 0.0

    @property
    def progress(self) -> FolderVideosRun:
        """process.Executor's progress controller, whose aborted and paused server.py reads."""
        return self

    @property
    def aborted(self) -> bool:
        return self.__aborted

    @property
    def paused(self) -> bool:
        return self.__paused_since is not None

    def pause(self) -> None:
        if self.__paused_since is None:
            self.__paused_since = time.monotonic()
        if self.__current is not None:
            self.__current.pause()

    def resume(self) -> None:
        if self.__paused_since is not None:
            self.__time_paused += time.monotonic() - self.__paused_since
            self.__paused_since = None
        if self.__current is not None:
            self.__current.resume()

    def kill(self) -> None:
        self.__aborted = True
        if self.__current is not None:
            self.__current.kill()
        self.__arguments["loop"].call_soon_threadsafe(self.__stop)

    def __stop(self) -> None:
        if not self.__stopped.done():
            self.__stopped.set_result(None)

    async def __on_pool(self, job: Callable[[], T]) -> T:
        """The result of the run's own job on the pool (a listing, which a slow disk can make long), which holds neither
        the loop nor Stop: the host's /kill allows seconds. Stopped, the run ends as Aborted and the job finishes in the
        background: it writes nothing, and its result goes nowhere."""
        job_done = self.__arguments["loop"].run_in_executor(
            self.__arguments["pool"], job
        )
        try:
            await asyncio.wait(
                (job_done, self.__stopped), return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            if not job_done.done():
                job_done.add_done_callback(_log_left_job_error)
        if not job_done.done():
            raise Aborted()
        return job_done.result()

    def __seconds(self) -> float:
        """The time the run was not paused, from any origin."""
        now = time.monotonic() if self.__paused_since is None else self.__paused_since
        return now - self.__time_paused

    def __executor(self, chain: Chain) -> Any:
        if self.__aborted:
            raise Aborted()
        self.__runs += 1
        executor = self.__executor_class(
            **{
                **self.__arguments,
                "id": ExecutionId(f"{self.__arguments['id']} {self.__runs}"),
                "chain": chain,
                "queue": self.__events,
                "parent_cache": OutputCache(static_data=self.__shared),
            }
        )
        self.__current = executor
        if self.paused:
            executor.pause()
        return executor

    async def __compute(self, node_id: NodeId) -> NodeOutput:
        """A node's output as a preview gives it, with what feeds it."""
        node = self.__chain.nodes[node_id]
        executor = self.__executor(
            _sub_chain(self.__chain, _upstream(self.__chain, [node_id]))
        )
        self.__computing.append(executor)
        try:
            if isinstance(node, GeneratorNode):
                return await executor.process_generator_node(node)
            assert isinstance(node, FunctionNode)
            return await executor.process_regular_node(node)
        finally:
            self.__current = None
            await executor.flush_broadcasts()

    async def __run_chain(self, chain: Chain) -> None:
        executor = self.__executor(chain)
        try:
            await executor.run()
        finally:
            self.__current = None

    async def __list_folder(
        self, folder: _Folder, computed: dict[NodeId, NodeOutput]
    ) -> _Listing:
        node = folder.node
        module = sys.modules[NODE_MODULE]
        assigned = InputMap.from_chain(self.__chain).get(node.id)
        values = [
            _output_value(computed[i.id], i.index)
            if isinstance(i, EdgeInput)
            else i.value
            for i in assigned
        ]
        enforced = await self.__on_pool(
            functools.partial(enforce_inputs, values, node.data, node.id, [])
        )
        directory = enforced[0]
        assert isinstance(directory, Path)
        fail_fast = enforced[module.LISTING_INPUTS]
        assert isinstance(fail_fast, bool)
        try:
            paths, weights = await self.__on_pool(
                functools.partial(_listing, module, enforced[: module.LISTING_INPUTS])
            )
        except Aborted:
            raise
        except Exception as error:
            failure = loop_node_error(node, assigned, error)
            raise failure from error
        return _Listing(folder, paths, weights, directory, fail_fast)

    async def __run_files(self, listing: _Listing) -> list[str]:
        """Runs the folder's files; returns the errors of those that failed, as Load Images defers its items'."""
        folder, paths = listing.folder, listing.paths
        node = folder.node
        progress = _FolderProgress(node.id, listing.weights, self.__seconds)
        self.__events.folder = progress
        errors: list[str] = []
        for index, path in enumerate(paths):
            progress.file = index
            single = _file_chain(
                self.__chain, folder, path, listing.directory, len(paths)
            )
            failed = False
            try:
                await self.__run_chain(single)
            except Aborted:
                raise
            except Exception as error:
                if listing.fail_fast:
                    raise _file_failure(error, path) from error.__cause__
                logger.error("%s failed: %s", path, error, exc_info=error)
                errors.append(f"{path}: {error}")
                failed = True
            if failed:
                # What a failed run's error held (server.py collect_failed_run), before the next file.
                gc.collect()
        self.__events.folder = None
        self.__queue.put(progress.event(1))
        self.__queue.put(
            {
                "event": "node-finish",
                "data": {"nodeId": node.id, "executionTime": progress.elapsed},
            }
        )
        return errors

    async def run(self) -> None:
        try:
            await self.__run_chains()
        finally:
            preserve_error = sys.exc_info()[0] is not None
            if ending_runs():
                # A node Stop left running may still use what they free: after it, as a run's own.
                after_ending_runs(functools.partial(self.__chain_cleanups, True))
            else:
                await self.__chain_cleanups(preserve_error)

    async def __chain_cleanups(self, preserve_error: bool) -> None:
        """Only run() ends with a chain's cleanups, and the shared nodes ran without it."""
        for executor in self.__computing:
            await executor.chain_cleanups(preserve_error)

    async def __run_chains(self) -> None:
        chain = self.__chain
        folders = [
            _plan_folder(chain, node)
            for node in chain.nodes_with_schema_id(LOAD_VIDEOS)
            if isinstance(node, GeneratorNode) and chain.edges_from(node.id)
        ]
        per_file: set[NodeId] = set().union(*(f.per_file for f in folders))
        parent: OutputCache[NodeOutput] | None = self.__arguments.get("parent_cache")
        start = set(chain.nodes)
        if parent is not None:
            start -= parent.keys()
            for node_id in parent.keys() - per_file:
                cached = parent.peek(node_id)
                if cached is not None:
                    self.__shared[node_id] = cached
        self.__queue.put({"event": "chain-start", "data": {"nodes": list(start)}})
        wanted = {
            edge.source.id
            for folder in folders
            for edge in chain.edges_to(folder.node.id)
        }.union(*(f.shared for f in folders))
        rest = _upstream(chain, set(chain.nodes) - _upstream(chain, per_file))
        file_chains: set[NodeId] = set().union(*(f.chain_nodes for f in folders))
        needed = _eager_upstream(chain, wanted, self.__shared)
        self.__events.quiet = {
            node_id
            for node_id in needed & (rest | file_chains)
            if isinstance(chain.nodes[node_id], GeneratorNode)
        }
        computed: dict[NodeId, NodeOutput] = {}
        # What they need first, each once with what it read cached. A generator's output reaches no later run: it is
        # computed only for the file list, or by what reads it.
        for node_id in reversed(chain.topological_order()):
            node = chain.nodes[node_id]
            if node_id in wanted or (
                node_id in needed - per_file and isinstance(node, FunctionNode)
            ):
                cached = self.__shared.get(node_id)
                output = cached if cached is not None else await self.__compute(node_id)
                if node_id in wanted:
                    computed[node_id] = output
                if isinstance(output, RegularOutput) and node_id not in per_file:
                    self.__shared[node_id] = output
        self.__events.quiet = set()
        # Before anything writes: what the folders hold, and whether their outputs can be told apart.
        listings = [await self.__list_folder(folder, computed) for folder in folders]
        colliding = await self.__on_pool(
            functools.partial(
                _colliding_files,
                chain,
                [_video_items(listing) for listing in listings],
                computed,
            )
        )
        if colliding is not None:
            writer = colliding[0]
            raise NodeExecutionError(
                writer,
                chain.nodes[writer].data,
                _collision_message(
                    chain,
                    colliding,
                    "Connect Load Videos' Name to its file name, and its Subdirectory Path or"
                    " Video Directory to its folder. Of two videos with one name in one folder,"
                    " choose one with a glob expression; two Load Videos need an output folder"
                    " each.",
                ),
                {},
            )
        if rest:
            await self.__run_chain(_sub_chain(chain, rest))
        # The files' runs read these; a run frees the others after their last read.
        self.__shared = {k: v for k, v in self.__shared.items() if k in file_chains}
        errors: list[str] = []
        for listing in listings:
            errors += await self.__run_files(listing)
        if errors:
            error_string = "- " + "\n- ".join(errors)
            raise Exception(f"Errors occurred during iteration:\n{error_string}")
