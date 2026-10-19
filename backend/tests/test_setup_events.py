"""The host's setup events (/setup-sse): every listener gets them, and a late one learns
the backend's state first."""

from __future__ import annotations

import asyncio
import importlib
import sys
from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest

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


def test_the_host_is_ready_before_the_worker_has_loaded_its_nodes(
    monkeypatch: pytest.MonkeyPatch,
):
    # The app fetches its node list, which waits for every node, after
    # backend-nodes-loaded: a request that waited would hold one of the window's 6
    # connections to the host, which the previews of the light nodes need meanwhile.
    monkeypatch.setattr(sys, "argv", ["server_host.py"])
    server_host = importlib.import_module("server_host")
    events = BackendEventBroadcast()
    listener = events.listen()
    nodes_loaded = asyncio.Event()

    class Worker:
        async def get_packages(self) -> list[object]:
            return []

        async def wait_for_ready(self) -> None:
            pass

        async def wait_for_nodes(self) -> None:
            await nodes_loaded.wait()

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

    async def start() -> list[str]:
        setup = asyncio.create_task(server_host.setup(None, asyncio.get_running_loop()))
        await asyncio.sleep(0.05)
        assert ctx.is_ready
        before = [event["event"] async for event in pending(listener)]
        nodes_loaded.set()
        await asyncio.wait_for(setup, 1)
        after = [event["event"] async for event in pending(listener)]
        return [*before, "|", *after]

    assert asyncio.run(start()) == [
        "backend-started",
        "backend-status",
        "backend-status",
        "backend-ready",
        "|",
        "backend-nodes-loaded",
    ]


async def pending(listener: EventQueue) -> AsyncIterator[Event]:
    while not listener.queue.empty():
        yield await listener.get()
