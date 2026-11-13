"""The host's setup events (/setup-sse): every listener gets them, and a late one learns
the backend's state first."""

from __future__ import annotations

import asyncio
import importlib
import sys
from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest
from sanic.exceptions import RequestCancelled, ServerError

from events import BackendEvent, BackendEventBroadcast, Event, EventQueue


def status(message: str) -> BackendEvent:
    return {
        "event": "backend-status",
        "data": {"message": message, "progress": 0.5, "statusProgress": None},
    }


def drain(listener: EventQueue) -> list[BackendEvent]:
    async def get_all() -> list[BackendEvent]:
        events = []
        while not listener.queue.empty():
            events.append(await listener.get())
        return events

    return asyncio.run(get_all())


READY: BackendEvent = {"event": "backend-ready", "data": None}
NODES_LOADED: BackendEvent = {"event": "backend-nodes-loaded", "data": None}
INSTALL: BackendEvent = {
    "event": "package-install-status",
    "data": {"message": "Collecting torch...", "progress": 0.1, "statusProgress": None},
}


def test_every_listener_gets_every_event_and_no_put_waits():
    events = BackendEventBroadcast()
    # no listener: nothing waits, nothing fails
    events.put({"event": "backend-started", "data": None})

    main_process, window = events.listen(), events.listen()
    events.put(status("Importing nodes..."))
    events.put(INSTALL)
    events.put(READY)

    expected = [status("Importing nodes..."), INSTALL, READY]
    assert drain(main_process) == expected
    assert drain(window) == expected


def test_a_late_listener_gets_the_latest_status_and_readiness_first():
    events = BackendEventBroadcast()
    events.put(status("Importing nodes..."))
    events.put(status("Loading Nodes..."))
    events.put(INSTALL)

    assert drain(events.listen()) == [status("Loading Nodes...")]

    events.put(READY)
    assert drain(events.listen()) == [status("Loading Nodes..."), READY]


def test_a_listener_that_stopped_gets_nothing_more():
    events = BackendEventBroadcast()
    listener = events.listen()
    events.stop_listening(listener)
    events.put(READY)
    assert drain(listener) == []


def test_a_late_listener_learns_that_the_nodes_have_loaded():
    events = BackendEventBroadcast()
    events.put(READY)
    events.put(NODES_LOADED)
    assert drain(events.listen()) == [READY, NODES_LOADED]


def test_the_host_announces_the_nodes_the_worker_registers_while_it_loads(
    monkeypatch: pytest.MonkeyPatch,
):
    # The app sends a node's preview once its node is registered, and fetches its
    # node list, which waits for every node, after backend-nodes-loaded: a request that
    # waited in the worker would hold one of the window's 6 connections to the host,
    # which the previews of the nodes that have loaded need meanwhile.
    monkeypatch.setattr(sys, "argv", ["server_host.py"])
    server_host = importlib.import_module("server_host")
    events = BackendEventBroadcast()
    listener = events.listen()
    # What the worker answers: the light nodes, the same again, then PyTorch's
    loading = [
        (["chainner:image:load"], False),
        (["chainner:image:load"], False),
        (["chainner:image:load", "chainner:pytorch:load_model"], True),
    ]
    ready_while_loading: list[bool] = []

    class Worker:
        async def get_packages(self) -> list[object]:
            return []

        async def wait_for_ready(self) -> None:
            pass

        async def registered_nodes(self) -> tuple[list[str], bool]:
            ready_while_loading.append(ctx.is_ready)
            return loading.pop(0)

    config = SimpleNamespace(
        install_builtin_packages=False,
        error_on_failed_node=False,
        close_after_start=False,
    )
    ctx = SimpleNamespace(
        config=config,
        get_worker_unmanaged=Worker,
        setup_events=events,
        is_ready=False,
    )
    monkeypatch.setattr(server_host.AppContext, "get", lambda _app: ctx)

    async def start() -> list[tuple[object, object]]:
        await server_host.setup(None)
        return [
            (event["event"], (event.get("data") or {}).get("schemaIds"))
            async for event in pending(listener)
        ]

    assert asyncio.run(start()) == [
        ("backend-started", None),
        ("backend-status", None),
        ("backend-status", None),
        ("backend-ready", None),
        ("backend-nodes-registered", ["chainner:image:load"]),
        # an unchanged list is not announced again
        (
            "backend-nodes-registered",
            ["chainner:image:load", "chainner:pytorch:load_model"],
        ),
        ("backend-nodes-loaded", None),
    ]
    assert ready_while_loading == [True, True, True]


async def pending(listener: EventQueue) -> AsyncIterator[Event]:
    while not listener.queue.empty():
        yield await listener.get()


# How a stream ends when its client has gone (the app's window closed, a reset
# connection): Sanic's RequestCancelled (a CancelledError) for a closing transport, its
# ServerError once the stream has ended, or the socket's own error.
CLOSED = [
    RequestCancelled(),
    ServerError(
        "Response stream was ended, no more response data is allowed to be sent."
    ),
    ConnectionResetError(10054, "An existing connection was forcibly closed"),
    ConnectionAbortedError(10053, "The connection was aborted"),
    BrokenPipeError(32, "Broken pipe"),
]
FAILURES = [*CLOSED, RuntimeError("the stream failed")]


@pytest.mark.parametrize("failure", FAILURES, ids=lambda e: type(e).__name__)
def test_a_failing_setup_stream_is_logged_and_a_closed_one_ends_quietly(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure: BaseException,
):
    closed = failure is not FAILURES[-1]
    monkeypatch.setattr(sys, "argv", ["server_host.py"])
    server_host = importlib.import_module("server_host")
    events = BackendEventBroadcast()
    events.put(READY)
    ctx = SimpleNamespace(setup_events=events)
    stopped: list[EventQueue] = []
    stop_listening = events.stop_listening

    def stop(listener: EventQueue) -> None:
        stopped.append(listener)
        stop_listening(listener)

    monkeypatch.setattr(events, "stop_listening", stop)
    monkeypatch.setattr(server_host.AppContext, "get", lambda _app: ctx)

    class Response:
        async def send(self, _data: str) -> None:
            raise failure

    async def respond(**_kwargs: object) -> Response:
        return Response()

    request = SimpleNamespace(app=None, respond=respond)
    if isinstance(failure, RequestCancelled):
        with pytest.raises(RequestCancelled):
            asyncio.run(server_host.setup_sse(request))
    else:
        asyncio.run(server_host.setup_sse(request))
    failures = [str(r.exc_info[1]) for r in caplog.records if r.exc_info]
    assert failures == ([] if closed else ["the stream failed"])
    assert len(stopped) == 1


@pytest.mark.parametrize("failure", FAILURES, ids=lambda e: type(e).__name__)
def test_the_host_s_event_stream_ends_quietly_only_for_a_closed_client(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure: BaseException,
):
    monkeypatch.setattr(sys, "argv", ["server_host.py"])
    server_host = importlib.import_module("server_host")

    class Worker:
        async def get_sse(self, _request: object) -> AsyncIterator[bytes]:
            yield b"event: x\n\n"

    async def get_worker() -> Worker:
        return Worker()

    monkeypatch.setattr(
        server_host.AppContext,
        "get",
        lambda _app: SimpleNamespace(get_worker=get_worker),
    )

    class Response:
        async def send(self, _data: bytes) -> None:
            raise failure

    async def respond(**_kwargs: object) -> Response:
        return Response()

    request = SimpleNamespace(app=None, respond=respond)
    if isinstance(failure, RequestCancelled):
        with pytest.raises(RequestCancelled):
            asyncio.run(server_host.sse(request))
    else:
        asyncio.run(server_host.sse(request))
    failures = [str(r.exc_info[1]) for r in caplog.records if r.exc_info]
    assert failures == ([] if failure in CLOSED else ["the stream failed"])
