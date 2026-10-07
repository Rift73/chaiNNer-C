"""Item window (SP3): when a generator group may read items ahead of its commit, and
the engine that computes later items' pure CPU nodes ahead into per-item stores."""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar

import numpy as np
import psutil

from api import NodeId
from nodes.impl.execution_cleanup import await_owned_node
from progress_controller import ProgressToken

# Developer-only override of K (spec 4.2): 1 = sequential; unset or invalid = auto.
ITEM_WINDOW_VARIABLE = "CHAINNER_C_ITEM_WINDOW"
# Developer-only (SP3b spec 4.6): 1 = every group keeps the serial producer.
SERIAL_PRODUCER_VARIABLE = "CHAINNER_C_SERIAL_PRODUCER"


def serial_producer() -> bool:
    """Whether every group keeps the serial producer: only the value "1" asks for it,
    and any other value is ignored (SP3b spec 4.6)."""
    return os.environ.get(SERIAL_PRODUCER_VARIABLE) == "1"


def engine_jobs(pool_size: int) -> int:
    """P: concurrent engine pool jobs, the producer's advance included (spec 4.2).

    Two pool threads always stay free for the commit path; the advance it waits for
    may take one of them.
    """
    return max(1, pool_size - 2)


def memory_budget() -> int:
    """B: 25 % of the memory available now (spec 4.2)."""
    return psutil.virtual_memory().available // 4


def output_nbytes(values: Iterable[object]) -> int:
    """The bytes of the NumPy arrays and bytes objects among a node's output values."""
    return sum(
        value.nbytes if isinstance(value, np.ndarray) else len(value)
        for value in values
        if isinstance(value, (np.ndarray, bytes))
    )


def window_size(cpus: int) -> tuple[int, str]:
    """The window size K and its reason: a valid override, else clamp(cpus // 4, 6, 8).

    The bounds are the K-sweep's K* (the smallest K within 2 % of the best geometric
    mean) at 8 CPUs (6) and at 32 CPUs (8) (spec 5.4).
    """
    value = os.environ.get(ITEM_WINDOW_VARIABLE, "")
    if value.isascii() and value.isdigit() and int(value) >= 1:
        k = int(value)
        return (1, "forced") if k == 1 else (k, "override")
    return min(max(cpus // 4, 6), 8), "auto"


def affinity_cpus() -> int:
    """The CPUs this process may run on (its affinity mask), as the native pool counts.

    Without an affinity API (psutil on macOS), every CPU of the machine.
    """
    if not hasattr(psutil.Process, "cpu_affinity"):
        return os.cpu_count() or 1
    return len(psutil.Process().cpu_affinity())


@dataclass(frozen=True)
class Stored:
    """A node computed ahead for a later item: its output and time, or its error."""

    output: object
    seconds: float
    error: BaseException | None = None


@dataclass(frozen=True)
class Candidate:
    """A node that may run ahead, and its iterated, non-generator inputs.

    prepare: the candidate is the node's prepare phase (SP3-P9), not the node itself;
    the commit path takes its result with take_prepared.
    """

    node_id: NodeId
    depends: frozenset[NodeId]
    prepare: bool = False


@dataclass(frozen=True)
class Phases:
    """A side-effect node's two phases (SP3-P9).

    prepare receives the inputs the node would, enforced, a lazy input as the value it
    produces (a collector's: its iterated inputs, as on_iterate receives them). It is
    pure, so it may run ahead, and returns the value to commit with, or None to run the
    node whole; a prepare that raises also leaves the node to run whole, so any error
    is the node's own, raised where it is today. commit is the node itself with that
    value: commit(inputs, prepared) for a regular node, commit(collector, inputs,
    prepared) for a collector's iteration; it calls prepared.result() where the node
    would compute the value itself.
    """

    prepare: Callable[..., object | None]
    commit: Callable[..., object]


_PHASES: dict[str, Phases] = {}


def register_phases(
    schema_id: str,
    prepare: Callable[..., object | None],
    commit: Callable[..., object],
) -> None:
    """Declare a side-effect node's prepare and commit phases; its module calls this
    once at import (SP3-P9)."""
    _PHASES[schema_id] = Phases(prepare, commit)


def node_phases(schema_id: str) -> Phases | None:
    """The phases registered for a node schema, if any."""
    return _PHASES.get(schema_id)


@dataclass
class Prepared:
    """A prepare's value, taken for its node's commit (SP3-P9)."""

    value: object
    waited: bool = False  # the commit path awaited the prepare in flight
    used: bool = False  # the commit called result()

    def result(self) -> object:
        """The prepared value, for the commit where the node would compute it."""
        self.used = True
        return self.value


class ItemPlanner(Protocol):
    """Which nodes run ahead for a later slot, and their jobs (spec 4.4).

    candidates: the nodes that may run ahead, in topological order, then the prepare
    phases of side-effect nodes (spec 4.7).
    eligible: whether a node may run ahead for the slot whose generator values and
    store these are.
    job: the pool job computing that node (or its prepare) for that slot; it returns
    the output and its time, as the commit path's pool job does.
    nbytes: the bytes a stored output, a prepared value or an advance value holds.
    """

    @property
    def candidates(self) -> Sequence[Candidate]: ...

    def eligible(
        self,
        node_id: NodeId,
        values: Mapping[NodeId, object],
        store: Mapping[NodeId, Stored],
    ) -> bool: ...

    def job(
        self,
        node_id: NodeId,
        values: Mapping[NodeId, object],
        store: Mapping[NodeId, Stored],
    ) -> Callable[[], tuple[object, float]]: ...

    def nbytes(self, value: object) -> int: ...


@dataclass(frozen=True)
class SlotGenerator:
    """One generator of the group, as the producer advances it (spec 4.3).

    advance is the loop's own advance: it returns the item's output, returns None
    when the generator is exhausted, and raises the item's error. An inline
    generator (Range) is advanced on the loop thread, as the loop does.

    describe and materialize split advance in two phases (SP3b spec 4.2): describe
    returns the item's token, returns None when the generator is exhausted, or raises
    the item's error; materialize(token) returns the item's output or raises its
    error. A window over this generator alone, not inline, with both, describes
    serially and materializes as engine jobs. split: describe only names the item and
    materialize decodes it (Load Images), holding transient buffers too (spec 4.4).
    """

    node_id: NodeId
    advance: Callable[[], object | None]
    inline: bool
    fail_fast: bool
    describe: Callable[[], object | None] | None = None
    materialize: Callable[[object], object] | None = None
    split: bool = False


@dataclass(frozen=True)
class _Outcome:
    """One generator's advance in one slot: its value (None = exhausted) or error."""

    value: object = None
    error: BaseException | None = None


T = TypeVar("T")


class _Call(Generic[T]):
    """What the pool runs for an engine job (a materialize included) or a producer
    advance (or describe).

    The call is dropped as it starts, and its result reaches the loop through take(),
    not through the pool's future, which carries None. A pool thread still finishing
    its own bookkeeping after the loop moved on therefore holds neither the call's
    inputs nor its result. The result stays here until it is taken; the only take
    that can be skipped is the commit path's, when it is cancelled while taking a
    job, and that leaves only the item in progress here until this object is freed.
    """

    def __init__(self, call: Callable[[], T]) -> None:
        self._call: Callable[[], T] | None = call
        self._results: list[T] = []

    def __call__(self) -> None:
        call, self._call = self._call, None
        assert call is not None, "a pool call runs once"
        self._results.append(call())

    def take(self) -> T:
        """The call's result, once its future is done without error (on the loop)."""
        assert len(self._results) == 1, "a pool call's result is taken once"
        return self._results.pop()


@dataclass
class _Flight:
    """An engine job in flight: its future and its call."""

    future: asyncio.Future[None]
    call: _Call[tuple[object, float]]
    taken: bool = False  # take_node took it: the result or error is the commit path's


@dataclass
class _Item:
    """One produced slot: its generator values, store, jobs in flight and bytes."""

    # Every generator's value once the slot is produced, if each yielded one; only
    # then is the slot speculated.
    values: dict[NodeId, object] | None = None
    # Finished jobs' results and errors; the engine never runs a stored node again.
    store: dict[NodeId, Stored] = field(default_factory=dict)
    running: dict[NodeId, _Flight] = field(default_factory=dict)
    # Prepared values the commit path took; counted when the slot is released.
    prepared: list[Prepared] = field(default_factory=list)
    advance_bytes: int = 0
    store_bytes: int = 0
    peak: int = 0  # the most advance plus store bytes held, taken results included
    # The two-phase producer (SP3b): the slot's token from its describe until its
    # materialize starts, that materialize's pool future, and whether the slot's
    # outcomes are resolved (by its describe or its materialize).
    token: object = None
    materializing: asyncio.Future[None] | None = None
    resolved: bool = False


class ItemWindow:
    """Per-item stores of a generator group's later items (spec 4.1).

    From the first slot it begins, one serialized producer advances the group's
    generators up to size - 1 slots ahead of the committing slot, in the loop's
    per-slot pattern (spec 4.3, SP3-P2); the commit path takes each advance from
    it. A group of one generator that is not inline and carries describe and
    materialize is produced in two phases instead (SP3b spec 4.2): the producer
    describes slots serially, and each described slot's materialize is an engine
    job, the slot's first; the slot's outcome is its materialize's, or its
    describe's error or exhaustion. For the slots after the committing one, the
    engine computes the planner's eligible nodes into per-slot stores (spec 4.4).
    At most `jobs` (P) engine pool jobs run at once, the producer's advance (or
    describe) and materializations included, plus the advance (or describe, then
    materialize) the commit path waits for; the oldest slot's jobs go first. A slot
    beyond the one the commit path waits for is produced only while the in-flight
    bytes plus the per-item estimate stay within the budget (spec 4.2, SP3-P5);
    with a split generator, a slot counts twice the estimate until its outcome is
    resolved, and so does the admission (SP3b spec 4.4). The commit path takes a
    node's result from the store of the slot it commits, or awaits its job in
    flight, instead of running its pool job. For a side-effect node with a prepare
    phase, it takes the prepared value to commit with (spec 4.7).

    While the run is paused or stopped, nothing starts but the slot the commit path
    waits for; work in flight finishes (spec 4.5).
    """

    def __init__(
        self,
        *,
        size: int,
        generators: Sequence[SlotGenerator],
        stops: int,
        loop: asyncio.AbstractEventLoop,
        pool: concurrent.futures.Executor,
        jobs: int,
        budget: int,
        estimate: int,
        planner: ItemPlanner,
        progress: ProgressToken,
    ) -> None:
        self._size = size
        self._generators = tuple(generators)
        # The two-phase producer's generator (SP3b P3), or None: the serial producer.
        only = self._generators[0] if len(self._generators) == 1 else None
        self._phased = (
            only
            if only is not None
            and not only.inline
            and only.describe is not None
            and only.materialize is not None
            else None
        )
        self._split = self._phased is not None and self._phased.split
        self._loop = loop
        self._pool = pool
        self._jobs = jobs
        self._budget = budget
        self._estimate = estimate  # E: bytes one item holds, raised by released slots
        self._planner = planner
        self._prepares = frozenset(c.node_id for c in planner.candidates if c.prepare)
        self._progress = progress  # the run's pause and Stop
        self._slots: dict[int, _Item] = {}  # produced (or being produced), unreleased
        self._outcomes: dict[int, asyncio.Future[dict[NodeId, _Outcome]]] = {}
        self._commit = 0
        self._next = 0  # the next slot to produce, from the first begin
        self._wanted = -1  # the slot the commit path waits for
        self._advancing: asyncio.Task[None] | None = None  # the running producer
        # Engine pool jobs in flight: the producer and materializations included.
        self._running = 0
        self._tasks: set[asyncio.Future[Any]] = set()  # started and not done yet
        self._ended = False
        self._closing = False
        self._stops = stops  # slots that ended in exhaustion, as the loop counts them
        self._items = 0
        self._advanced_ahead = 0
        self._jobs_started = 0
        self._materialized = 0
        self._replayed = 0
        self._awaited = 0
        self._discarded = 0

    @property
    def size(self) -> int:
        """The window size K."""
        return self._size

    @property
    def items(self) -> int:
        """Slots the commit path began with the window open (SP3-P11)."""
        return self._items

    @property
    def advanced_ahead(self) -> int:
        """Slots whose production started beyond the committing slot (SP3-P11)."""
        return self._advanced_ahead

    @property
    def jobs_started(self) -> int:
        """Engine node jobs submitted (SP3-P11)."""
        return self._jobs_started

    @property
    def materialized(self) -> int:
        """Materialize jobs started (SP3b P7)."""
        return self._materialized

    @property
    def replayed(self) -> int:
        """take_node calls served by a completed result, and prepared values their
        commit used that were complete when taken (SP3-P11)."""
        return self._replayed

    @property
    def awaited(self) -> int:
        """take_node calls that waited for a job in flight, and prepared values their
        commit used that it waited for (SP3-P11)."""
        return self._awaited

    @property
    def discarded(self) -> int:
        """Results never taken: dropped by release or close, or late; and prepared
        values never used: None, or left by their commit (SP3-P11)."""
        return self._discarded

    def begin(self, slot: int) -> None:
        """The commit path starts this slot."""
        if not self._items:
            self._next = slot  # the producer continues the loop from here
        self._items += 1
        self._commit = slot
        self._pump()

    async def take_advance(self, node_id: NodeId) -> object | None:
        """This generator's outcome in the committing slot, as the loop's advance.

        Returns its value, returns None when it is exhausted, or raises its error
        (the same object). A slot not produced yet is produced now: described, then
        materialized, on the two-phase producer. Every producer in flight (an
        advance, a describe or a materialize) resolves its slot, closing or not: a
        describe that returns once closing began, when nothing is materialized any
        more, resolves it with the RuntimeError below. So a take started before or
        after closing never waits forever. A slot that can no longer be produced
        (produced and dropped, past the producer's end, or closing began with nothing
        in flight for it) raises RuntimeError at once.

        A slot abandoned on Stop or close (see _produce) lacks its later generators'
        outcomes, and they are never taken: the commit path runs a suspend() between
        every two takes, which raises Aborted once Stop was set, and close() follows
        its last take.
        """
        slot = self._commit
        outcomes = self._slot_outcomes(slot)
        if not outcomes.done():
            self._wanted = slot
            self._pump()
            # In flight (P10): the slot's serial advance, its describe, or its
            # materialize, which the pump started unless closing. Every in-flight
            # producer resolves its slot, closing or not (a describe that returns
            # once closing began, with this RuntimeError), so the take waits for it.
            producing = (self._advancing is not None and slot == self._next - 1) or (
                (item := self._slots.get(slot)) is not None
                and item.materializing is not None
            )
            if (
                not outcomes.done()
                and not producing
                and (slot < self._next or self._ended or self._closing)
            ):
                raise RuntimeError(f"item window: slot {slot} cannot be produced")
        outcome = (await await_owned_node(outcomes))[node_id]
        if outcome.error is not None:
            raise outcome.error
        return outcome.value

    def holds(self, node_id: NodeId) -> bool:
        """A stored result or an in-flight job exists for this node in this slot.

        A prepare phase's result is never the node's own (see take_prepared).
        """
        item = self._slots.get(self._commit)
        return (
            item is not None
            and node_id not in self._prepares
            and (node_id in item.store or node_id in item.running)
        )

    async def take_node(self, node_id: NodeId) -> tuple[object, float]:
        """This slot's result for the node, as its pool job would return or raise it.

        A stored result is replayed and a stored error raised, so the node runs once
        (spec 4.5). A job in flight, or done before its callback stored it, is taken
        from its slot and its own future gives the result or raises the error (the
        object a store would hold), so nothing is read from a store the callback has
        not filled yet.

        A take that has nothing to wait for still makes one pool round trip, as the
        pool job it replaces does, so it finishes no sooner than that job could: the
        nodes around it (parallel-gathered siblings, nodes whose own flight starts a
        loop step later) interleave with it as with that job at K = 1. That order is
        pool timing at any K; Python 3.14 can finish a pool job, this round trip
        included, in the loop step that started it (asyncio.futures._chain_future).
        """
        slot = self._commit
        item = self._slots.get(slot)
        if item is None or (node_id not in item.store and node_id not in item.running):
            raise RuntimeError(f"item window: slot {slot} cannot be produced")
        stored = item.store.pop(node_id, None)
        if stored is not None:
            item.store_bytes -= self._planner.nbytes(stored.output)
            self._replayed += 1
            await self._round_trip()
            if stored.error is not None:
                raise stored.error
            return stored.output, stored.seconds
        flight = item.running.pop(node_id)
        flight.taken = True  # its done callback leaves the result to this take
        if flight.future.done():
            self._replayed += 1
            await self._round_trip()
        else:
            self._awaited += 1
        await await_owned_node(flight.future)  # a failed job raises its error here
        output, seconds = flight.call.take()
        # It never entered the store, yet the slot held it: it counts toward E.
        held = item.advance_bytes + item.store_bytes + self._planner.nbytes(output)
        item.peak = max(item.peak, held)
        return output, seconds

    async def take_prepared(self, node_id: NodeId) -> Prepared | None:
        """This slot's prepared value for the node, to commit with (spec 4.7).

        None runs the node whole: no prepare phase, nothing prepared for this slot (not
        eligible, or not started before the commit path came), a prepare that returned
        None (DDS), or one that raised. A failed prepare is nothing prepared: the node
        then fails, or succeeds, exactly as it does today. A stored value is taken at
        once; a prepare in flight is awaited, as take_node awaits a job. The commit runs
        as the node's pool job, so no round trip is added. A value counts once its slot
        is released: replayed or awaited when the commit used it, else discarded; a None
        or failed prepare counts as discarded at the take.
        """
        item = self._slots.get(self._commit)
        if item is None or node_id not in self._prepares:
            return None
        waited = False
        stored = item.store.pop(node_id, None)
        if stored is not None:
            item.store_bytes -= self._planner.nbytes(stored.output)
        elif (flight := item.running.pop(node_id, None)) is not None:
            flight.taken = True  # its done callback leaves the result to this take
            waited = not flight.future.done()
            try:
                await await_owned_node(flight.future)
            except Exception as error:  # the prepare failed (a cancellation is not one)
                stored = Stored(None, 0.0, error)
            else:
                stored = Stored(*flight.call.take())
                held = (
                    item.advance_bytes
                    + item.store_bytes
                    + self._planner.nbytes(stored.output)
                )
                item.peak = max(item.peak, held)
        else:
            return None
        if stored.error is not None and not isinstance(stored.error, Exception):
            raise stored.error  # as take_node raises it (a cancelled job)
        if stored.error is not None or stored.output is None:
            self._discarded += 1  # nothing prepared: the node runs whole
            return None
        prepared = Prepared(stored.output, waited)
        item.prepared.append(prepared)
        return prepared

    def release(self, slot: int) -> None:
        """Free the slot's store and outcomes once its item is dropped.

        Its jobs still in flight stay awaited by close(); their results are dropped
        when they arrive. The slot's peak bytes raise the per-item estimate.
        """
        item = self._slots.pop(slot, None)
        if item is not None:
            self._discarded += len(item.store)
            self._count_prepared(item)
            self._estimate = max(self._estimate, item.peak)
        self._outcomes.pop(slot, None)
        self._pump()

    async def close(self) -> None:
        """Stop producing, wait for the producer, every started materialization and
        every started job, released slots' jobs included, then free every store (spec
        4.5); a queued materialization never starts. Idempotent.

        A job's error belongs to its slot, where the commit path meets it, so the
        drain only waits, and it holds no result: later items' outputs are freed
        before the run's cleanups. A cancelled caller still waits for all of them.
        """
        self._closing = True
        try:
            if self._tasks:
                await await_owned_node(
                    asyncio.gather(*(_finished(task) for task in self._tasks))
                )
        finally:
            for item in self._slots.values():
                self._discarded += len(item.store)
                self._count_prepared(item)
            self._slots.clear()
            self._outcomes.clear()

    def _count_prepared(self, item: _Item) -> None:
        """Count the slot's taken prepared values; their commits have finished."""
        for prepared in item.prepared:
            if not prepared.used:
                self._discarded += 1  # e.g. a save that skipped an existing file
            elif prepared.waited:
                self._awaited += 1
            else:
                self._replayed += 1
        item.prepared.clear()

    async def _round_trip(self) -> None:
        """One pool round trip that carries nothing: a pool job's shortest wait."""
        await await_owned_node(self._loop.run_in_executor(self._pool, _nothing))

    def _slot_outcomes(self, slot: int) -> asyncio.Future[dict[NodeId, _Outcome]]:
        """The slot's outcomes, resolved once the producer recorded them."""
        outcomes = self._outcomes.get(slot)
        if outcomes is None:
            outcomes = self._outcomes[slot] = self._loop.create_future()
        return outcomes

    def _pump(self) -> None:
        """Admission (spec 4.2): start the advance the commit path waits for (or its
        describe, then its materialize), then advances (describes), materializations
        and jobs while fewer than P run.

        Called after begin, release, a wanted take_advance and every finished
        advance, describe, materialize or job, so whatever may start does. The slot
        the commit path waits for is always produced, paused or stopped too
        (liveness), and even while P jobs run: that advance is the commit path's own
        work, on a thread reserved for it. Nothing starts once closing has begun.
        After a resume, admission continues at the loop's next begin, take_advance or
        release.
        """
        if not self._closing:
            if (
                self._wanted == self._next
                and self._advancing is None
                and not self._ended
            ):
                self._start_advance()  # the commit path waits for this slot
            elif (
                wanted := self._slots.get(self._wanted)
            ) is not None and wanted.token is not None:
                self._start_materialize(self._wanted)  # and then for its materialize
        while not self._closing and self._running < self._jobs:
            if (ready := self._oldest_ready_job()) is not None:
                slot, node_id = ready
                if node_id is None:
                    self._start_materialize(slot)
                else:
                    self._start_job(slot, node_id)
            elif self._may_admit():
                self._start_advance()
            else:
                return

    def _may_admit(self) -> bool:
        """A later slot may be produced: neither paused nor stopped, within the window
        and the memory budget (twice the estimate with a split generator)."""
        return (
            self._advancing is None
            and not self._ended
            and not self._progress.paused
            and not self._progress.aborted
            and self._next <= self._commit + self._size - 1
            and self._in_flight() + (2 if self._split else 1) * self._estimate
            <= self._budget
        )

    def _in_flight(self) -> int:
        """Bytes of the produced, unreleased slots, each counted at least at E; with a
        split generator, at 2E until its outcome is resolved (SP3b spec 4.4)."""
        return sum(
            2 * self._estimate
            if self._split and not item.resolved
            else max(item.advance_bytes + item.store_bytes, self._estimate)
            for item in self._slots.values()
        )

    def _oldest_ready_job(self) -> tuple[int, NodeId | None] | None:
        """The first job that may start, None while paused or stopped: slots after the
        committing one, ascending; a described slot's materialize first (node None);
        then candidates in order; not stored (result or error) or running; every
        input it depends on stored without error (a failed input's dependents are
        never computed, so the commit path meets the error where K = 1 does); and
        eligible."""
        if self._progress.paused or self._progress.aborted:
            return None
        for slot in sorted(slot for slot in self._slots if slot > self._commit):
            item = self._slots[slot]
            if item.token is not None:
                return slot, None
            if item.values is None:
                continue
            for candidate in self._planner.candidates:
                node_id = candidate.node_id
                if node_id in item.store or node_id in item.running:
                    continue
                if all(
                    node in item.store and item.store[node].error is None
                    for node in candidate.depends
                ) and self._planner.eligible(node_id, item.values, item.store):
                    return slot, node_id
        return None

    def _start_advance(self) -> None:
        slot = self._next
        self._next += 1
        if slot > self._commit:
            self._advanced_ahead += 1
        self._slots[slot] = _Item()
        self._running += 1
        producer = self._produce if self._phased is None else self._describe
        self._advancing = self._loop.create_task(producer(slot))
        self._track(self._advancing)

    def _start_materialize(self, slot: int) -> None:
        """Materialize a described slot as an engine job (SP3b spec 4.2)."""
        assert self._phased is not None and self._phased.materialize is not None
        item = self._slots[slot]
        token, item.token = item.token, None  # the call holds it from here
        call = _Call(functools.partial(self._phased.materialize, token))
        item.materializing = self._loop.run_in_executor(self._pool, call)
        self._running += 1
        self._materialized += 1
        self._track(item.materializing)
        item.materializing.add_done_callback(
            functools.partial(self._materialize_done, slot, call)
        )

    def _start_job(self, slot: int, node_id: NodeId) -> None:
        item = self._slots[slot]
        assert item.values is not None
        call = _Call(self._planner.job(node_id, item.values, item.store))
        flight = _Flight(self._loop.run_in_executor(self._pool, call), call)
        item.running[node_id] = flight
        self._running += 1
        self._jobs_started += 1
        self._track(flight.future)
        flight.future.add_done_callback(
            functools.partial(self._job_done, slot, node_id, flight)
        )

    def _track(self, future: asyncio.Future[Any]) -> None:
        """close() awaits every started advance (describe), materialize and job until
        it is done."""
        self._tasks.add(future)
        future.add_done_callback(self._tasks.discard)

    def _job_done(
        self,
        slot: int,
        node_id: NodeId,
        flight: _Flight,
        future: asyncio.Future[None],
    ) -> None:
        """Store a finished job's result, or what it raised (spec 4.5: only the failing
        node stores its error), in its slot, unless the commit path took the job. A
        result for a released slot is dropped."""
        self._running -= 1
        if not flight.taken:  # else take_node awaits the job and takes it itself
            if future.cancelled():  # as awaiting the job would raise
                error: BaseException | None = asyncio.CancelledError()
            else:
                error = future.exception()
            if error is None:
                stored = Stored(*flight.call.take())
            else:
                stored = Stored(None, 0.0, error)
            item = self._slots.get(slot)
            if item is None:
                self._discarded += 1  # late: its slot was released
            else:
                del item.running[node_id]
                item.store[node_id] = stored
                item.store_bytes += self._planner.nbytes(stored.output)
                item.peak = max(item.peak, item.advance_bytes + item.store_bytes)
        self._pump()

    def _materialize_done(
        self,
        slot: int,
        call: _Call[object],
        future: asyncio.Future[None],
    ) -> None:
        """Resolve the slot with its materialize's output, or what it raised."""
        self._running -= 1
        if future.cancelled():  # as awaiting the call would raise
            error: BaseException | None = asyncio.CancelledError()
        else:
            error = future.exception()
        self._resolve(
            slot,
            _Outcome(value=call.take()) if error is None else _Outcome(error=error),
        )
        self._pump()

    async def _produce(self, slot: int) -> None:
        """Advance the generators for one slot, as the loop's per-slot pattern does."""
        outcomes: dict[NodeId, _Outcome] = {}
        for position, source in enumerate(self._generators):  # the loop's order
            if position and (self._closing or self._progress.aborted):
                # Abandoned on Stop or close, not pause. The commit path takes no later
                # generator here: Stop is monotonic, so its suspend() before the next
                # take raises Aborted first, and close() follows its last take.
                break
            try:
                if source.inline:  # Range: on the loop thread, as today
                    value = source.advance()
                else:  # its value reaches the loop as a node job's result does
                    call = _Call(source.advance)
                    await await_owned_node(self._loop.run_in_executor(self._pool, call))
                    value = call.take()
            except BaseException as error:
                # Yielded or raised: later generators skip this slot.
                outcomes[source.node_id] = _Outcome(error=error)
                if source.fail_fast or not isinstance(error, Exception):
                    # The commit path raises here; nothing later is read.
                    self._ended = True
                break
            outcomes[source.node_id] = _Outcome(value=value)
            if value is None:  # exhausted: the slot ends as StopIteration does
                self._stops += 1
                if self._stops >= len(self._generators):
                    self._ended = True  # the commit loop breaks at this slot
                break
        item = self._slots.get(slot)
        if item is not None:
            item.advance_bytes = sum(
                self._planner.nbytes(outcome.value) for outcome in outcomes.values()
            )
            item.peak = max(item.peak, item.advance_bytes + item.store_bytes)
            if len(outcomes) == len(self._generators) and all(
                outcome.error is None and outcome.value is not None
                for outcome in outcomes.values()
            ):
                item.values = {
                    node: outcome.value for node, outcome in outcomes.items()
                }
        self._slot_outcomes(slot).set_result(outcomes)
        self._advancing = None
        self._running -= 1
        self._pump()

    async def _describe(self, slot: int) -> None:
        """Describe one slot (SP3b spec 4.2): its token queues the slot's materialize;
        exhaustion or an error is the slot's outcome, and the slot has none. A token
        that arrives once closing began, or for a released slot, is never materialized:
        the slot's outcome is then take_advance's RuntimeError. So a describe always
        resolves its slot or queues its materialize, as _produce always resolves it."""
        source = self._phased
        assert source is not None and source.describe is not None
        try:
            call = _Call(source.describe)
            await await_owned_node(self._loop.run_in_executor(self._pool, call))
            token = call.take()
        except BaseException as error:
            self._resolve(slot, _Outcome(error=error))
        else:
            if token is None:
                self._resolve(slot, _Outcome())  # exhausted
            elif self._closing or (item := self._slots.get(slot)) is None:
                error = RuntimeError(f"item window: slot {slot} cannot be produced")
                self._resolve(slot, _Outcome(error=error))
            else:
                item.token = token
        self._advancing = None
        self._running -= 1
        self._pump()

    def _resolve(self, slot: int, outcome: _Outcome) -> None:
        """Record the two-phase generator's outcome in the slot, as _produce records
        an advance's (spec 4.3). _ended stops describes only: a described slot is
        still materialized unless Stop or close."""
        source = self._phased
        assert source is not None
        if outcome.error is not None:
            if source.fail_fast or not isinstance(outcome.error, Exception):
                self._ended = True  # the commit path raises here
        elif outcome.value is None:  # exhausted: the slot ends as StopIteration does
            self._stops += 1
            if self._stops >= len(self._generators):
                self._ended = True  # the commit loop breaks at this slot
        item = self._slots.get(slot)
        if item is not None:
            item.resolved = True
            item.advance_bytes = self._planner.nbytes(outcome.value)
            item.peak = max(item.peak, item.advance_bytes + item.store_bytes)
            if outcome.error is None and outcome.value is not None:
                item.values = {source.node_id: outcome.value}
        self._slot_outcomes(slot).set_result({source.node_id: outcome})


def _nothing() -> None:
    """The pool call of a replay's round trip."""


async def _finished(task: asyncio.Future[Any]) -> None:
    """Wait for a started advance or job without taking its result or error."""
    await asyncio.wait([task])


def _normalized(path: Path) -> str:
    text = str(path)  # resolve() keeps an extended-length prefix it was given
    if text[:8].upper() == "\\\\?\\UNC\\":  # \\?\UNC\server\share -> \\server\share
        text = "\\\\" + text[8:]
    elif text.startswith("\\\\?\\"):  # \\?\C:\dir -> C:\dir
        text = text[4:]
    return os.path.normcase(text)  # lowercases on Windows only


def _contains(directory: str, folder: str) -> bool:
    """directory is folder itself or one of its ancestors."""
    return folder == directory or folder.startswith(directory.rstrip("\\/") + os.sep)


def read_ahead_blocker(
    sources: Sequence[tuple[NodeId, Sequence[Path] | None]],
    directories: Sequence[tuple[NodeId, Path | None]],
) -> str | None:
    """The first reason the group must not read ahead, or None when it may (spec 4.3).

    sources: each generator with the files it reads per item (None = unknown).
    directories: each file-writing node with its base directory (None = unresolved).
    Each declared file's parent folder (once per unique folder; the files are never
    resolved, which opens them on Windows) and each base directory are compared after
    Path.resolve(); a source lies under a directory that is its folder or an ancestor.
    """
    parents: set[Path] = set()
    for generator, paths in sources:
        if paths is None:
            return f"sources unknown: {generator}"
        parents.update(Path(path).parent for path in paths)
    writers: list[tuple[NodeId, Path]] = []
    for writer, directory in directories:
        if directory is None:
            return f"output directory unresolved: {writer}"
        writers.append((writer, directory))
    if not parents:
        return None  # nothing is read per item, so no write can reach a source
    try:
        folders = {_normalized(parent.resolve()) for parent in parents}
        for writer, directory in writers:
            base = _normalized(directory.resolve())
            if any(_contains(base, folder) for folder in folders):
                return f"source under output directory: {writer}"
    except (OSError, ValueError, RuntimeError) as error:
        return f"path check failed: {error}"
    return None
