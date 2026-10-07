"""Coalesce unsent UI state while retaining control and error events."""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any


class LatestEventQueue(asyncio.Queue):
    def __init__(self):
        super().__init__()
        # asyncio.Queue's hooks (_put, _get, qsize, empty) read self._queue, which
        # typeshed does not declare. Queue.__init__ has just made it an empty deque
        # and nothing is queued yet, so binding a fresh one here changes nothing.
        self._queue: deque[Any] = deque()
        self._latest: dict[tuple[str, str], dict] = {}

    @staticmethod
    def _key(event: Any):
        if event.get("event") in {
            "node-start",
            "node-finish",
            "node-progress",
            "node-broadcast",
        }:
            return event["event"], event["data"]["nodeId"]
        return None

    def put_nowait(self, item: Any):
        event = item
        key = self._key(event)
        if key is None:
            # Never fold state across a reliable control/error boundary.
            self._latest.clear()
        previous = self._latest.get(key) if key is not None else None
        if previous is not None:
            assert key is not None
            # Keep the order of the last occurrence of each state event. Merely
            # replacing in place could put a new start before an older finish.
            # Equal payloads may belong to earlier runs, before a control
            # barrier. Remove precisely the queued object owned by this key.
            for index, queued in enumerate(self._queue):
                if queued is previous:
                    del self._queue[index]
                    break
            else:
                raise RuntimeError("Pending UI event lost its queue ownership")
            if event["event"] == "node-broadcast":
                data = dict(event["data"])
                for field in ("data", "types", "sequenceTypes"):
                    data[field] = {
                        **(previous["data"].get(field) or {}),
                        **(data.get(field) or {}),
                    }
                event = {**event, "data": data}
            self._queue.append(event)
            self._latest[key] = event
            return
        super().put_nowait(event)

    def _put(self, item: Any):
        event = item
        key = self._key(event)
        if key is not None:
            self._latest[key] = event
        self._queue.append(event)

    def _get(self):
        event = self._queue.popleft()
        key = self._key(event)
        if key is not None and self._latest.get(key) is event:
            del self._latest[key]
        return event
