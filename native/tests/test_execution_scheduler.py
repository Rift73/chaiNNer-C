"""CPU-only scheduling ownership and final UI state without timing benchmarks."""

from __future__ import annotations

import asyncio
import copy
import gc
import sys
import weakref
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend/src"))

from nodes.impl.event_queue import LatestEventQueue
from nodes.impl.execution_scheduler import (
    LatestBroadcasts,
    NodeFlights,
    advance_iterator,
    owned_gather,
)


def run(coroutine):
    async def bounded():
        # Deadlock guard only; no elapsed time is measured or compared.
        async with asyncio.timeout(5):
            return await coroutine

    return asyncio.run(bounded())


def event(kind, node="node", **data):
    return {"event": kind, "data": {"nodeId": node, **data}}


def drain(queue):
    result = []
    while not queue.empty():
        result.append(queue.get_nowait())
        queue.task_done()
    return result


@pytest.mark.parametrize("kind", ["node-start", "node-finish", "node-progress"])
def test_latest_event_is_bounded_and_retains_last_occurrence_order(kind):
    queue = LatestEventQueue()
    first = event(kind, "a", index=0)
    other = event(kind, "b", index=8)
    queue.put_nowait(first)
    queue.put_nowait(other)
    for index in range(1000):
        queue.put_nowait(event(kind, "a", index=index))
    assert queue.qsize() == 2
    assert drain(queue) == [other, event(kind, "a", index=999)]
    run(queue.join())


def test_start_finish_cycles_keep_final_event_order():
    queue = LatestEventQueue()
    for kind in ("node-start", "node-finish", "node-start"):
        queue.put_nowait(event(kind))
    assert [item["event"] for item in drain(queue)] == ["node-finish", "node-start"]


@pytest.mark.parametrize("before", [None, {}, {"0": "sequence"}])
@pytest.mark.parametrize("after", [None, {}, {"1": "latest"}])
def test_broadcast_merges_sparse_output_fields_without_mutating_inputs(before, after):
    queue = LatestEventQueue()
    first = event(
        "node-broadcast",
        data={"0": "old", "1": "keep"},
        types={"0": "old-type", "1": "keep-type"},
        sequenceTypes=before,
    )
    latest = event(
        "node-broadcast",
        data={"0": None, "2": "new"},
        types={"0": None},
        sequenceTypes=after,
    )
    originals = copy.deepcopy([first, latest])
    queue.put_nowait(first)
    queue.put_nowait(latest)
    assert queue.qsize() == 1
    assert drain(queue) == [
        event(
            "node-broadcast",
            data={"0": None, "1": "keep", "2": "new"},
            types={"0": None, "1": "keep-type"},
            sequenceTypes={**(before or {}), **(after or {})},
        )
    ]
    assert [first, latest] == originals
    run(queue.join())


@pytest.mark.parametrize(
    "control",
    [
        "execution-error",
        "chain-start",
        "backend-started",
        "backend-status",
        "package-install-status",
    ],
)
def test_control_events_are_order_barriers(control):
    queue = LatestEventQueue()
    first = event("node-progress", index=1)
    barrier = {"event": control, "data": {"message": "keep me"}}
    last = event("node-progress", index=2)
    for value in (first, barrier, last):
        queue.put_nowait(value)
    assert drain(queue) == [first, barrier, last]
    run(queue.join())


def test_consumed_event_is_not_mutated_or_merged_with_new_iteration():
    queue = LatestEventQueue()
    first = event("node-broadcast", data={"0": "first"}, types={}, sequenceTypes=None)
    queue.put_nowait(first)
    sent = queue.get_nowait()
    second = event("node-broadcast", data={"1": "next"}, types={}, sequenceTypes=None)
    queue.put_nowait(second)
    queue.task_done()
    assert sent is first
    assert drain(queue) == [second]
    run(queue.join())


def test_coalescing_preserves_queue_join_accounting_and_wakes_getters():
    async def scenario():
        queue = LatestEventQueue()
        waiting = asyncio.create_task(queue.get())
        await asyncio.sleep(0)
        await queue.put(event("node-progress", index=0))
        assert (await waiting)["data"]["index"] == 0
        for index in range(1, 20):
            await queue.put(event("node-progress", index=index))
        assert queue.qsize() == 1
        joining = asyncio.create_task(queue.join())
        await asyncio.sleep(0)
        assert not joining.done()
        queue.task_done()
        assert (await queue.get())["data"]["index"] == 19
        assert not joining.done()
        queue.task_done()
        await joining
        with pytest.raises(ValueError):
            queue.task_done()

    run(scenario())


def test_owned_gather_returns_exception_objects_as_legitimate_values():
    async def scenario():
        values = [
            None,
            ValueError("value"),
            asyncio.CancelledError("value"),
            KeyboardInterrupt("value"),
            SystemExit("value"),
        ]

        async def value(item):
            return item

        result = await owned_gather([value(item) for item in values])
        assert all(
            actual is expected for actual, expected in zip(result, values, strict=True)
        )
        assert await owned_gather([]) == []

    run(scenario())


def test_owned_gather_waits_for_all_errors_and_reports_input_order():
    async def scenario():
        release = asyncio.Event()
        later_failed = asyncio.Event()
        done = []
        first, second = ValueError("first"), RuntimeError("second")

        async def earlier():
            await release.wait()
            done.append("first")
            raise first

        async def later():
            done.append("second")
            later_failed.set()
            raise second

        task = asyncio.create_task(owned_gather([earlier(), later()]))
        await later_failed.wait()
        assert not task.done()
        release.set()
        with pytest.raises(ValueError) as caught:
            await task
        assert caught.value is first
        assert done == ["second", "first"]

    run(scenario())


@pytest.mark.parametrize("error_after_cancel", [False, True])
def test_owned_gather_cancellation_drains_siblings_before_return(error_after_cancel):
    async def scenario():
        starts = [asyncio.Event(), asyncio.Event()]
        release = asyncio.Event()
        finished = []

        async def work(index):
            starts[index].set()
            await release.wait()
            finished.append(index)
            if error_after_cancel and index == 0:
                raise ValueError("error while draining")
            return index

        task = asyncio.create_task(owned_gather([work(0), work(1)]))
        await asyncio.gather(*(item.wait() for item in starts))
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and finished == []
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert sorted(finished) == [0, 1]

    run(scenario())


def test_owned_gather_awaits_a_lone_branch_in_the_callers_task():
    async def scenario():
        async def lone():
            await asyncio.sleep(0)
            return asyncio.current_task()

        assert await owned_gather([lone()]) == [asyncio.current_task()]

    run(scenario())


def test_owned_gather_cancellation_reaches_a_lone_branch():
    # A lone branch is awaited like an input that is not gathered: the branch
    # owns its draining (NodeFlights and await_owned_node below it).
    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()
        seen = []

        async def lone():
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                seen.append("cancelled")
                raise

        task = asyncio.create_task(owned_gather([lone()]))
        await started.wait()
        task.cancel()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        observed = list(seen)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert observed == ["cancelled"]

    run(scenario())


def test_cancelled_lone_branch_still_drains_its_node_through_node_flights():
    # The composition the executor relies on: owned_gather hands the cancellation
    # to a lone branch, and NodeFlights keeps the node's work running to its end
    # before the cancellation leaves the branch.
    async def scenario():
        flights = NodeFlights()
        started, release = asyncio.Event(), asyncio.Event()
        finished = []

        async def node():
            started.set()
            await release.wait()
            finished.append("node")
            return 1

        async def branch():
            return await flights.run("node", node)

        task = asyncio.create_task(owned_gather([branch()]))
        await started.wait()
        task.cancel()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not task.done() and finished == []
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished == ["node"]

    run(scenario())


def test_node_flights_deduplicates_only_active_work_and_preserves_identity():
    async def scenario():
        flights = NodeFlights()
        started, release = asyncio.Event(), asyncio.Event()
        calls = []
        value = object()

        async def work():
            calls.append("run")
            started.set()
            await release.wait()
            return value

        tasks = [asyncio.create_task(flights.run("node", work)) for _ in range(8)]
        await started.wait()
        assert len(flights.pending) == 1 and calls == ["run"]
        release.set()
        assert all(item is value for item in await asyncio.gather(*tasks))
        assert flights.pending == {}
        assert await flights.run("node", work) is value
        assert calls == ["run", "run"]

    run(scenario())


def test_node_flights_canceled_waiter_does_not_cancel_shared_owner():
    async def scenario():
        flights = NodeFlights()
        started, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def work():
            calls.append("run")
            started.set()
            await release.wait()
            return 42

        a = asyncio.create_task(flights.run("same", work))
        b = asyncio.create_task(flights.run("same", work))
        await started.wait()
        a.cancel()
        await asyncio.sleep(0)
        assert not a.done() and not b.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await a
        assert await b == 42
        assert calls == ["run"] and flights.pending == {}

    run(scenario())


def test_node_flights_shares_failure_then_allows_retry():
    async def scenario():
        flights = NodeFlights()
        release = asyncio.Event()
        started = asyncio.Event()
        failure = ValueError("shared failure")
        calls = []

        async def work():
            calls.append("run")
            started.set()
            await release.wait()
            raise failure

        tasks = [asyncio.create_task(flights.run("same", work)) for _ in range(4)]
        await started.wait()
        release.set()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert all(item is failure for item in results)
        assert calls == ["run"] and flights.pending == {}
        with pytest.raises(ValueError) as caught:
            await flights.run("same", work)
        assert caught.value is failure and calls == ["run", "run"]

    run(scenario())


def test_latest_broadcast_releases_superseded_frames_and_keeps_final_state():
    class Payload:
        pass

    async def scenario():
        broadcasts = LatestBroadcasts()
        started, release = asyncio.Event(), asyncio.Event()
        sent = []

        async def active():
            started.set()
            await release.wait()
            sent.append("active")

        broadcasts.submit("a", active)
        await started.wait()
        discarded = Payload()
        weak = weakref.ref(discarded)

        async def pending(value=discarded):
            sent.append(value)

        broadcasts.submit("a", pending)
        del discarded, pending
        for index in range(1000):

            async def latest(value=index):
                sent.append(value)

            broadcasts.submit("a", latest)
        gc.collect()
        assert weak() is None
        assert len(broadcasts.pending) == 1
        assert sent == []
        release.set()
        await broadcasts.flush()
        assert sent == ["active", 999]
        assert broadcasts.task is None and not broadcasts.pending

    run(scenario())


def test_latest_broadcast_keys_progress_and_finish_flushes_all():
    async def scenario():
        broadcasts = LatestBroadcasts()
        sent = []

        async def send(key):
            sent.append(key)

        for index in range(4):
            broadcasts.submit(str(index), lambda index=index: send(index))
        await broadcasts.flush()
        assert sent == list(range(4))
        await broadcasts.flush()
        assert sent == list(range(4))

    run(scenario())


def test_latest_broadcast_error_is_observed_and_owned_work_cleared():
    async def scenario():
        broadcasts = LatestBroadcasts()
        failure = ValueError("broadcast failed")

        async def fail():
            raise failure

        broadcasts.submit("a", fail)
        with pytest.raises(ValueError) as caught:
            await broadcasts.flush()
        assert caught.value is failure
        assert broadcasts.task is None and not broadcasts.pending
        observed = []

        async def good():
            observed.append("recovered")

        broadcasts.submit("b", good)
        await broadcasts.flush()
        assert observed == ["recovered"]

    run(scenario())


def test_latest_broadcast_canceled_flush_waits_for_active_send():
    async def scenario():
        broadcasts = LatestBroadcasts()
        started, release = asyncio.Event(), asyncio.Event()
        done = []

        async def send():
            started.set()
            await release.wait()
            done.append(True)

        broadcasts.submit("node", send)
        await started.wait()
        task = asyncio.create_task(broadcasts.flush())
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and not done
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert done == [True]
        await broadcasts.flush()
        assert broadcasts.task is None

    run(scenario())


def test_iterator_advance_distinguishes_none_from_exhaustion_and_keeps_errors():
    iterator = iter([None, 3])
    assert advance_iterator(iterator) == (True, None)
    assert advance_iterator(iterator) == (True, 3)
    assert advance_iterator(iterator) == (False, None)
    failure = ValueError("supplier error")

    def broken():
        raise failure
        yield

    with pytest.raises(ValueError) as caught:
        advance_iterator(broken())
    assert caught.value is failure
