"""The host's setup events (/setup-sse): every listener gets them, and a late one learns
the backend's state first."""

from __future__ import annotations

import asyncio

from events import BackendEvent, BackendEventBroadcast, EventQueue


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
