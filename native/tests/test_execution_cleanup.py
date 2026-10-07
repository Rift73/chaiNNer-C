"""Executor cleanup on real async success, failure, abort and worker cancellation."""

from __future__ import annotations

import asyncio
import functools
import gc
import sys
import threading
import weakref
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend/src"))
import process
from api.lazy import Lazy
from nodes.impl import execution_cleanup
from nodes.impl.execution_cleanup import (
    await_owned_node,
    run_cleanup_functions,
    run_cleanup_groups,
)
from nodes.impl.execution_scheduler import LatestBroadcasts, NodeFlights, owned_gather
from process import ExecutionId, Executor, RegularOutput
from progress_controller import Aborted


def context(node=(), chain=()):
    return SimpleNamespace(node_cleanup_fns=set(node), chain_cleanup_fns=set(chain))


def raise_error(error):
    raise error


@pytest.mark.parametrize(
    "error",
    [
        None,
        ValueError("failed"),
        Aborted(),
        asyncio.CancelledError(),
        KeyboardInterrupt(),
    ],
)
def test_executor_drains_each_context_once_on_every_exit(error, monkeypatch):
    events = []
    contexts = {
        str(i): context(
            [lambda i=i: events.append((i, "node"))],
            [lambda i=i: events.append((i, "chain"))],
        )
        for i in range(3)
    }
    executor = object.__new__(Executor)
    executor.id = ExecutionId("cleanup-test")
    monkeypatch.setattr(executor, "_Executor__context_cache", contexts, raising=False)
    monkeypatch.setattr(
        executor, "_Executor__broadcasts", LatestBroadcasts(), raising=False
    )
    monkeypatch.setattr(
        executor, "node_cache", SimpleNamespace(clear=lambda: None), raising=False
    )

    async def body():
        events.append("body")
        if error is not None:
            raise error

    monkeypatch.setattr(executor, "_Executor__process_nodes", body)
    # The shipped executor does not collect on exit; any call would show up below.
    monkeypatch.setattr(gc, "collect", lambda: events.append("gc"))

    async def run():
        if error is None:
            await executor.run()
        else:
            with pytest.raises(type(error)) as caught:
                await executor.run()
            assert caught.value is error
        assert events == [
            "body",
            *[(i, kind) for i in range(3) for kind in ("node", "chain")],
        ]
        events.clear()
        try:
            await executor.run()
        except BaseException:
            pass
        assert events == ["body"]

    asyncio.run(run())


def test_success_drains_owned_broadcast_before_cleanup(monkeypatch):
    events = []
    executor = object.__new__(Executor)
    executor.id = ExecutionId("success-test")
    monkeypatch.setattr(
        executor,
        "_Executor__context_cache",
        {"one": context(chain=[lambda: events.append("cleanup")])},
        raising=False,
    )
    monkeypatch.setattr(
        executor, "_Executor__send_chain_start", lambda: events.append("start")
    )
    monkeypatch.setattr(
        executor,
        "chain",
        SimpleNamespace(topological_order=list, get_parent_iterator_map=dict),
        raising=False,
    )
    monkeypatch.setattr(
        executor,
        "node_cache",
        SimpleNamespace(clear=lambda: events.append("cache")),
        raising=False,
    )
    monkeypatch.setattr(gc, "collect", lambda: events.append("gc"))

    async def run():
        async def broadcast():
            events.append("broadcast")

        broadcasts = LatestBroadcasts()
        monkeypatch.setattr(
            executor, "_Executor__broadcasts", broadcasts, raising=False
        )
        broadcasts.submit("one", broadcast)
        await executor.run()

    asyncio.run(run())
    assert events == ["start", "broadcast", "cache", "cleanup"]


@pytest.mark.parametrize("preserve", [False, True])
def test_all_groups_release_even_if_callback_raises_base_exception(preserve):
    events = []
    interrupt = KeyboardInterrupt("cleanup")
    groups: list[set[Callable[[], None]]] = [
        {lambda: raise_error(interrupt)},
        {lambda: events.append("released")},
    ]
    if preserve:
        run_cleanup_groups(groups, preserve_error=True)
    else:
        with pytest.raises(KeyboardInterrupt) as caught:
            run_cleanup_groups(groups)
        assert caught.value is interrupt
    assert events == ["released"]
    assert groups == [set(), set()]


def test_cleanup_set_can_mutate_and_reenter_without_duplicate_callbacks():
    events = []
    functions = set()

    def first():
        events.append("first")
        run_cleanup_functions(functions)
        functions.add(second)

    def second():
        events.append("second")

    functions.add(first)
    run_cleanup_functions(functions)
    assert events == ["first"]
    run_cleanup_functions(functions)
    assert events == ["first", "second"]
    assert not functions


def test_ordinary_cleanup_error_is_logged_and_other_callbacks_run(monkeypatch):
    errors = []
    events = []
    monkeypatch.setattr(
        execution_cleanup.logger, "error", lambda *args: errors.append(args)
    )
    functions = {
        lambda: raise_error(ValueError("failed cleanup")),
        lambda: events.append("released"),
    }
    run_cleanup_functions(functions)
    assert events == ["released"] and not functions
    assert len(errors) == 1 and str(errors[0][1]) == "failed cleanup"


@pytest.mark.parametrize("worker_error", [None, RuntimeError("worker failed")])
def test_cancelled_executor_waits_for_owned_worker_before_cleanup(
    worker_error, monkeypatch
):
    events = []
    release = threading.Event()
    monkeypatch.setattr(gc, "collect", lambda: events.append("gc"))

    async def run():
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        ctx = context()
        executor = object.__new__(Executor)
        executor.id = ExecutionId("cancel-worker-test")
        monkeypatch.setattr(
            executor, "_Executor__context_cache", {"one": ctx}, raising=False
        )
        monkeypatch.setattr(
            executor, "_Executor__broadcasts", LatestBroadcasts(), raising=False
        )
        monkeypatch.setattr(
            executor, "node_cache", SimpleNamespace(clear=lambda: None), raising=False
        )

        def worker():
            events.append("worker start")
            loop.call_soon_threadsafe(started.set)
            assert release.wait(5), "Test release was never signalled"
            ctx.chain_cleanup_fns.add(lambda: events.append("cleanup"))
            events.append("worker end")
            if worker_error is not None:
                raise worker_error
            return 42

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = loop.run_in_executor(pool, worker)

            async def body():
                return await await_owned_node(future)

            monkeypatch.setattr(executor, "_Executor__process_nodes", body)
            task = asyncio.create_task(executor.run())
            await started.wait()
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()  # Repeated cancellation must not release a live worker.
            await asyncio.sleep(0)
            assert not task.done() and events == ["worker start"]
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert future.done() and not future.cancelled()

    try:
        asyncio.run(run())
    finally:
        release.set()
    assert events == ["worker start", "worker end", "cleanup"]


@pytest.mark.parametrize("mode", ["success", "worker-error", "abort-after-worker"])
def test_actual_node_method_cleans_node_scope_without_mutating_iteration(
    mode, monkeypatch
):
    events = []
    ctx = context()
    error = ValueError("node failed")

    async def run():
        executor = object.__new__(Executor)
        executor.loop = asyncio.get_running_loop()

        async def gather(_node):
            return []

        monkeypatch.setattr(executor, "_Executor__gather_inputs", gather)
        monkeypatch.setattr(executor, "_Executor__get_node_context", lambda _node: ctx)
        monkeypatch.setattr(executor, "_Executor__window", None, raising=False)
        monkeypatch.setattr(executor, "_Executor__item_bytes", None, raising=False)
        monkeypatch.setattr(executor, "_Executor__send_node_start", lambda _node: None)

        async def broadcast(*_):
            events.append("broadcast")

        monkeypatch.setattr(executor, "_Executor__send_node_broadcast", broadcast)
        monkeypatch.setattr(executor, "_Executor__send_node_finish", lambda *_: None)
        suspends = 0

        async def suspend():
            nonlocal suspends
            suspends += 1
            if mode == "abort-after-worker" and suspends == 3:
                raise Aborted()

        monkeypatch.setattr(
            executor, "progress", SimpleNamespace(suspend=suspend), raising=False
        )

        def node(*_):
            ctx.node_cleanup_fns.update(
                {lambda: events.append("clean-one"), lambda: events.append("clean-two")}
            )
            if mode == "worker-error":
                raise error
            return RegularOutput([17])

        monkeypatch.setattr(process, "run_node", node)
        with ThreadPoolExecutor(max_workers=1) as pool:
            executor.pool = pool
            call = executor._Executor__process(  # pyright: ignore[reportAttributeAccessIssue] -- pyright has no model of name mangling outside the class; the test drives the real private method
                SimpleNamespace(id="one", data=None), perform_cache=False
            )
            if mode == "success":
                assert (await call).output == [17]
            else:
                with pytest.raises(ValueError if mode == "worker-error" else Aborted):
                    await call

    asyncio.run(run())
    assert sorted(events[:2]) == ["clean-one", "clean-two"]
    assert not ctx.node_cleanup_fns
    assert events[2:] == (["broadcast"] if mode == "success" else [])


FAILURE = "Cropped area would result in an image with no height"


class Block:
    """A failed node's input array, held only by the failing job's inputs."""


def failing_job(inputs):
    raise AssertionError(FAILURE)


def block_alive_after_failure(body):
    """Runs body(pool, inputs) to its failure with the collector off.

    The inputs hold the only reference to a Block. Returns whether it is alive once
    the failure is caught and dropped, and then after one gc.collect().
    """
    gc.collect()
    gc.disable()
    try:
        block = Block()
        alive = weakref.ref(block)
        failures = []

        async def main():
            with ThreadPoolExecutor(max_workers=2) as pool:
                try:
                    await body(pool, [block])
                except AssertionError as error:
                    failures.append(str(error))

        asyncio.run(main())
        del block
        assert failures == [FAILURE]
        after_failure = alive() is not None
        gc.collect()
        return after_failure, alive() is not None
    finally:
        gc.enable()


def test_failed_owned_job_frees_its_inputs_without_a_collection():
    # Executor.process awaits a node through NodeFlights.run, and __process awaits
    # its job through await_owned_node. A failed task or future holds the error,
    # whose traceback holds the awaiting frame, so neither frame may keep it.
    async def failing_owned(pool, inputs):
        loop = asyncio.get_running_loop()

        async def process_node():
            await await_owned_node(
                loop.run_in_executor(pool, functools.partial(failing_job, inputs))
            )

        await NodeFlights().run("crop", process_node)

    assert block_alive_after_failure(failing_owned) == (False, False)


@pytest.mark.parametrize("outcome", ["result", "error"])
def test_finished_owned_job_costs_no_loop_turn(outcome):
    # The commit path awaits flights that are already finished; like shield's fast
    # path, that must not yield to the loop (a loop turn and a cancellation point).
    async def main():
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        if outcome == "result":
            future.set_result(17)
        else:
            future.set_exception(ValueError("node failed"))
        turns = []
        loop.call_soon(turns.append, "turn")
        if outcome == "result":
            assert await await_owned_node(future) == 17
        else:
            with pytest.raises(ValueError, match="node failed"):
                await await_owned_node(future)
        assert turns == []

    asyncio.run(main())


def test_finished_failed_owned_job_frees_its_inputs_without_a_collection():
    # The fast path raises the finished job's error from the same frame, which may
    # not keep it either.
    async def finished(future):
        await asyncio.wait((future,))
        return future

    async def failing_finished(pool, inputs):
        loop = asyncio.get_running_loop()
        await await_owned_node(
            await finished(
                loop.run_in_executor(pool, functools.partial(failing_job, inputs))
            )
        )

    assert block_alive_after_failure(failing_finished) == (False, False)


def test_cancelled_owned_job_frees_its_inputs_when_it_then_fails(monkeypatch):
    # Sanic cancels the /run handler when the UI drops the request, and
    # await_owned_node keeps waiting for the running job. When that job then fails,
    # its error crosses the waiting frame, which logs it and may not keep it.
    logged = []
    monkeypatch.setattr(
        execution_cleanup.logger, "error", lambda _, error: logged.append(str(error))
    )
    release = threading.Event()

    def blocked_failing_job(inputs):
        assert release.wait(5), "Test release was never signalled"
        failing_job(inputs)

    gc.collect()
    gc.disable()
    try:
        block = Block()
        alive = weakref.ref(block)
        cancelled = []

        async def main():
            loop = asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=1) as pool:
                job = functools.partial(blocked_failing_job, [block])
                task = asyncio.create_task(
                    await_owned_node(loop.run_in_executor(pool, job))
                )
                await asyncio.sleep(0)
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done()
                release.set()
                try:
                    await task
                except asyncio.CancelledError:
                    cancelled.append(True)

        asyncio.run(main())
        del block
        assert cancelled == [True] and logged == [FAILURE]
        assert alive() is None
    finally:
        release.set()
        gc.enable()


def test_failed_gathered_branch_frees_its_inputs_without_a_collection():
    # owned_gather raises the first failed branch's error, whose traceback then
    # holds its frame, so its locals may not keep that error.
    async def failing_gathered(pool, inputs):
        loop = asyncio.get_running_loop()

        async def branch(job):
            return await loop.run_in_executor(pool, job)

        await owned_gather(
            [branch(functools.partial(failing_job, inputs)), branch(int)]
        )

    assert block_alive_after_failure(failing_gathered) == (False, False)


def test_failed_lazy_read_keeps_its_inputs_until_a_collection():
    # Upstream's edge, characterized: a Lazy keeps the error as its value, and the
    # Lazy.value frame on that error's traceback holds the Lazy. Only a collection
    # frees the failed job's inputs.
    async def failing_lazy_read(pool, inputs):
        loop = asyncio.get_running_loop()

        async def upstream():
            return await loop.run_in_executor(
                pool, functools.partial(failing_job, inputs)
            )

        image = Lazy.from_coroutine(upstream(), loop)
        # Read in a pool job, as Save Image reads its lazy image input.
        await loop.run_in_executor(pool, lambda: image.value)

    assert block_alive_after_failure(failing_lazy_read) == (True, False)


def test_failed_plain_job_frees_its_inputs_at_once():
    # The control: a job awaited directly, as the installed chaiNNer awaits its
    # run_node job, leaves no cycle.
    async def failing_plain(pool, inputs):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(pool, functools.partial(failing_job, inputs))

    assert block_alive_after_failure(failing_plain) == (False, False)
