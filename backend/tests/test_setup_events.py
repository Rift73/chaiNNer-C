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
        await server_host.setup(None, asyncio.get_running_loop())
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
