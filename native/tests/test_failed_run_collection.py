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

SCHEDULE = "app.loop.call_soon(collect_failed_run, ctx)"


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


def test_a_failed_run_schedules_one_collection_after_its_error_event():
    # The error event is queued first and the 500 is returned after: call_soon runs the
    # collection on a later loop iteration, once the handler has returned and dropped
    # the exception, so neither waits for it.
    (handler,) = handlers(
        definition(ast.parse(source("server.py")), "run"), "Exception"
    )
    *_, queued, scheduled, returned = handler.body
    assert ast.unparse(queued) == (
        "ctx.queue.put({'event': 'execution-error', 'data': error})"
    )
    assert ast.unparse(scheduled) == SCHEDULE
    assert ast.unparse(returned) == (
        "return json(error_response('Error running nodes!', exception), status=500)"
    )
    calls = [n for n in ast.walk(handler) if isinstance(n, ast.Call)]
    assert [ast.unparse(n) for n in calls].count(SCHEDULE) == 1


def test_gc_is_used_only_by_collect_failed_run():
    # One full collection and nothing else: no threshold tuning, no gc.freeze, no
    # generation-targeted collection, anywhere in server.py.
    tree = ast.parse(source("server.py"))
    inside = set(ast.walk(definition(tree, "collect_failed_run")))
    names = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == "gc"]
    assert len(names) == 1 and names[0] in inside
    uses = [
        ast.unparse(n)
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and ast.unparse(n.func).startswith("gc.")
    ]
    assert uses == ["gc.collect()"]
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
    # outside run's two handlers. collect_failed_run is named only by its definition
    # and by the call in run's except Exception and except Aborted handlers.
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
        for name in ("Exception", "Aborted")
        for handler in handlers(run, name)
        for statement in handler.body
        if ast.unparse(statement) == SCHEDULE
    ]
    assert len(sites) == 2
    in_sites = {n for s in sites for n in ast.walk(s)}
    named = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Name) and n.id == "collect_failed_run"
    ]
    assert len(named) == 2 and all(n in in_sites for n in named)


def test_a_stopped_run_schedules_one_collection():
    # Stop raises Aborted out of executor.run(), and the controller's Stop probe found
    # a stopped run's arrays still held after the next /runs (D). The inner finally is
    # unchanged; test_native_profile pins it.
    (handler,) = handlers(definition(ast.parse(source("server.py")), "run"), "Aborted")
    assert [ast.unparse(s) for s in handler.body] == [SCHEDULE]
