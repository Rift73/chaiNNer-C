"""A failed or stopped /run schedules one collection; a successful one never collects.

A failed run's arrays sit in reference-cycle garbage (the error's traceback holds the
run's frames, and the futures and Lazy results that hold the error are reachable from
them). After the handler returns, collect_failed_run frees them with one gc.collect(),
then releases the NumPy pool unless a new /run executes. collect_failed_run is
compiled alone out of server.py, and the scheduling sites are read from its AST;
nothing here starts a worker.
"""

from __future__ import annotations

import ast
import gc
import weakref
from types import SimpleNamespace

from server_source import compiled, definition, source

SCHEDULE = "asyncio.get_running_loop().call_soon(collect_failed_run, ctx)"
# run's handlers that schedule one collection each: a failed run's, a stopped run's
# Aborted and the cancellation of a stopped run's /run (the app's Stop drops the
# request before /kill).
SCHEDULING = ("Exception", "Aborted", "asyncio.CancelledError")


class Block:
    """A failed run's input array, held only by cyclic garbage."""


def handlers(function, name):
    """The except handlers in function whose type is name."""
    return [
        handler
        for node in ast.walk(function)
        if isinstance(node, ast.Try)
        for handler in node.handlers
        if handler.type is not None and ast.unparse(handler.type) == name
    ]


def test_collect_failed_run_frees_cyclic_garbage_then_releases_when_idle():
    gc.collect()
    gc.disable()
    try:
        block = Block()
        alive = weakref.ref(block)
        cycle: list[object] = [block]
        cycle.append(cycle)
        del block, cycle
        # Unreachable, but only a collection frees it.
        assert alive() is not None
        released = []
        collect = compiled(
            "collect_failed_run",
            gc=gc,
            numpy_pool=SimpleNamespace(release=lambda: released.append(alive())),
            return_device_memory=lambda collect=True: None,
        )
        collect(SimpleNamespace(executor=None))
        assert alive() is None
        # One release, which found the block already freed: it follows the collection,
        # so the blocks the cycle held are idle by then.
        assert released == [None]
    finally:
        gc.enable()


def test_collect_failed_run_never_releases_while_a_run_executes():
    # A new /run started before the callback ran: the failed run's garbage is still
    # collected, but the new run's warm pool is left to its own end-of-run release.
    events = []
    collect = compiled(
        "collect_failed_run",
        gc=SimpleNamespace(collect=lambda: events.append("collect")),
        numpy_pool=SimpleNamespace(release=lambda: events.append("release")),
    )
    collect(SimpleNamespace(executor=object()))
    assert events == ["collect"]


def test_collect_failed_run_returns_device_memory_when_idle():
    # The collection frees the failed run's tensors into PyTorch's cache, which keeps
    # them reserved (measured: a failed 128 px ESRGAN run held 130 MiB until a node was
    # deleted). They go back to the driver after the collection, without a second one.
    events = []
    collect = compiled(
        "collect_failed_run",
        gc=SimpleNamespace(collect=lambda: events.append("collect")),
        numpy_pool=SimpleNamespace(release=lambda: events.append("release")),
        return_device_memory=lambda collect=True: events.append(("return", collect)),
    )
    collect(SimpleNamespace(executor=None))
    assert events == ["collect", "release", ("return", False)]


def test_a_failed_run_schedules_one_collection_after_its_error_event():
    # The error event is queued first and the 500 is returned after: call_soon runs the
    # collection on a later loop iteration, once the handler has returned and dropped
    # the exception, so neither waits for it. The 500 carries the header that asks the
    # host to restart a worker whose CUDA context is broken (restart_headers).
    (handler,) = handlers(
        definition(ast.parse(source("server.py")), "run"), "Exception"
    )
    *_, queued, scheduled, returned = handler.body
    assert ast.unparse(queued) == (
        "ctx.queue.put({'event': 'execution-error', 'data': error})"
    )
    assert ast.unparse(scheduled) == SCHEDULE
    assert ast.unparse(returned) == (
        "return json(error_response('Error running nodes!', exception), status=500, "
        "headers=restart_headers())"
    )
    calls = [n for n in ast.walk(handler) if isinstance(n, ast.Call)]
    assert [ast.unparse(n) for n in calls].count(SCHEDULE) == 1


def test_gc_is_used_only_by_collect_failed_run_return_device_memory_and_the_freeze():
    # One full collection in each and nothing else: no threshold tuning, no
    # generation-targeted collection, anywhere in server.py. return_device_memory
    # runs only when a cleared node's GPU memory goes back outside a run, so a
    # successful run still never collects. The one exception: NodeLoading's
    # gc.collect(); gc.freeze() once after the node loading, which logs the frozen
    # count (gate ruling R1, native/reports/fable-decisions.md "instant-exec gate R1").
    tree = ast.parse(source("server.py"))
    names = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == "gc"]
    assert len(names) == 5
    for function in ("collect_failed_run", "return_device_memory"):
        inside = set(ast.walk(definition(tree, function)))
        assert sum(n in inside for n in names) == 1, function
    freeze = definition(tree, "NodeLoading", "freeze_objects")
    inside = set(ast.walk(freeze))
    assert sum(n in inside for n in names) == 3
    assert [
        [ast.unparse(s) for s in n.body]
        for n in ast.walk(freeze)
        if isinstance(n, ast.If)
    ].count(["gc.collect()", "gc.freeze()"]) == 1
    uses = [
        ast.unparse(n)
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and ast.unparse(n.func).startswith("gc.")
    ]
    assert sorted(uses) == [
        "gc.collect()",
        "gc.collect()",
        "gc.collect()",
        "gc.freeze()",
        "gc.get_freeze_count()",
    ]
    imports = [
        ast.unparse(n)
        for n in ast.walk(tree)
        if (isinstance(n, ast.Import) and any(a.name == "gc" for a in n.names))
        or (isinstance(n, ast.ImportFrom) and n.module == "gc")
    ]
    assert imports == ["import gc"]
    assert any(ast.unparse(n) == "import gc" for n in tree.body)


def test_no_finally_schedules_a_collection():
    # A finally runs on every exit, a successful run's included, and so can any code
    # outside run's SCHEDULING handlers. collect_failed_run is named only by its
    # definition and by the call in each of those handlers.
    tree = ast.parse(source("server.py"))
    in_finally = [
        ast.unparse(s)
        for t in ast.walk(tree)
        if isinstance(t, ast.Try)
        for s in t.finalbody
        if any(
            isinstance(n, ast.Name) and n.id == "collect_failed_run"
            for n in ast.walk(s)
        )
    ]
    assert in_finally == []
    run = definition(tree, "run")
    sites = [
        statement
        for name in SCHEDULING
        for handler in handlers(run, name)
        for statement in handler.body
        if ast.unparse(statement) == SCHEDULE
    ]
    assert len(sites) == len(SCHEDULING)
    in_sites = {n for s in sites for n in ast.walk(s)}
    named = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Name) and n.id == "collect_failed_run"
    ]
    assert len(named) == len(SCHEDULING) and all(n in in_sites for n in named)


def test_a_stopped_run_schedules_one_collection():
    # Stop raises Aborted out of executor.run(), and the controller's Stop probe found
    # a stopped run's arrays still held after the next /runs (D). The app's Stop
    # usually drops its /run request first, so the run ends cancelled instead: that
    # handler schedules the same one collection and lets the cancellation go on. The
    # inner finally is unchanged; test_native_profile pins it.
    run = definition(ast.parse(source("server.py")), "run")
    (aborted,) = handlers(run, "Aborted")
    assert [ast.unparse(s) for s in aborted.body] == [SCHEDULE]
    (cancelled,) = handlers(run, "asyncio.CancelledError")
    assert [ast.unparse(s) for s in cancelled.body] == [SCHEDULE, "raise"]
