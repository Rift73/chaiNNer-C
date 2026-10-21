from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Literal, TypedDict, Union, cast

import navi
from api import BroadcastData, ErrorValue, InputId, IterOutputId, NodeId, OutputId
from nodes.impl.event_queue import LatestEventQueue


class BackendStatusData(TypedDict):
    message: str
    progress: float
    statusProgress: float | None


class BackendStatusEvent(TypedDict):
    event: Literal["backend-status", "package-install-status"]
    data: BackendStatusData


class BackendStateEvent(TypedDict):
    event: Literal["backend-started", "backend-ready", "backend-nodes-loaded"]
    data: None


class BackendNodesRegisteredData(TypedDict):
    schemaIds: list[str]


class BackendNodesRegisteredEvent(TypedDict):
    event: Literal["backend-nodes-registered"]
    data: BackendNodesRegisteredData


BackendEvent = Union[BackendStatusEvent, BackendStateEvent, BackendNodesRegisteredEvent]
InputsDict = Dict[InputId, ErrorValue]


class ExecutionErrorSource(TypedDict):
    nodeId: NodeId
    schemaId: str
    inputs: InputsDict


class ExecutionErrorData(TypedDict):
    message: str
    exception: str
    exceptionTrace: str
    source: ExecutionErrorSource | None


class ExecutionErrorEvent(TypedDict):
    event: Literal["execution-error"]
    data: ExecutionErrorData


class ExecutionWarningData(TypedDict):
    message: str


class ExecutionWarningEvent(TypedDict):
    event: Literal["execution-warning"]
    data: ExecutionWarningData


class ChainStartData(TypedDict):
    nodes: list[str]


class ChainStartEvent(TypedDict):
    event: Literal["chain-start"]
    data: ChainStartData


class NodeStartData(TypedDict):
    nodeId: NodeId


class NodeStartEvent(TypedDict):
    event: Literal["node-start"]
    data: NodeStartData


class NodeProgressData(TypedDict):
    nodeId: NodeId
    progress: float
    "A number between 0 and 1"
    index: int
    total: int
    eta: float


class NodeProgressUpdateEvent(TypedDict):
    event: Literal["node-progress"]
    data: NodeProgressData


class NodeBroadcastData(TypedDict):
    nodeId: NodeId
    data: dict[OutputId, BroadcastData | None]
    types: dict[OutputId, navi.ExpressionJson | None]
    sequenceTypes: dict[IterOutputId, navi.ExpressionJson] | None


class NodeBroadcastEvent(TypedDict):
    event: Literal["node-broadcast"]
    data: NodeBroadcastData


class NodeFinishData(TypedDict):
    nodeId: NodeId
    executionTime: float


class NodeFinishEvent(TypedDict):
    event: Literal["node-finish"]
    data: NodeFinishData


ExecutionEvent = Union[
    ExecutionErrorEvent,
    ExecutionWarningEvent,
    ChainStartEvent,
    NodeStartEvent,
    NodeProgressUpdateEvent,
    NodeBroadcastEvent,
    NodeFinishEvent,
]
Event = Union[ExecutionEvent, BackendEvent]


class EventConsumer(ABC):
    @abstractmethod
    def put(self, event: Event) -> None: ...

    @staticmethod
    def filter(queue: EventConsumer, allowed: set[str]) -> EventConsumer:
        return _FilteredEventConsumer(queue, allowed)


@dataclass
class _FilteredEventConsumer(EventConsumer):
    queue: EventConsumer
    allowed: set[str]

    def put(self, event: Event) -> None:
        if event["event"] in self.allowed:
            self.queue.put(event)


class EventQueue(EventConsumer):
    def __init__(self):
        self.queue = LatestEventQueue()

    async def get(self) -> Event:
        return await self.queue.get()

    def put(self, event: Event) -> None:
        self.queue.put_nowait(event)


class BackendEventBroadcast:
    """The host's setup events (BackendEvent), sent to every /setup-sse listener; a put
    never waits for one.

    The app's main process and its window each listen, and either may connect after
    an event: a new listener first gets the latest backend status, then backend-ready,
    the latest backend-nodes-registered and backend-nodes-loaded once they have
    happened.
    """

    def __init__(self) -> None:
        self._listeners: set[EventQueue] = set()
        self._status: BackendEvent | None = None
        self._ready: BackendEvent | None = None
        self._nodes_registered: BackendEvent | None = None
        self._nodes_loaded: BackendEvent | None = None

    def put(self, event: BackendEvent) -> None:
        if event["event"] == "backend-status":
            self._status = event
        elif event["event"] == "backend-ready":
            self._ready = event
        elif event["event"] == "backend-nodes-registered":
            self._nodes_registered = event
        elif event["event"] == "backend-nodes-loaded":
            self._nodes_loaded = event
        for listener in self._listeners:
            listener.put(event)

    def listen(self) -> EventQueue:
        listener = EventQueue()
        for event in (
            self._status,
            self._ready,
            self._nodes_registered,
            self._nodes_loaded,
        ):
            if event is not None:
                listener.put(event)
        self._listeners.add(listener)
        return listener

    def stop_listening(self, listener: EventQueue) -> None:
        self._listeners.discard(listener)


class PreviewUrl(str):
    """A LargeImage preview's data URL (base64): JSON leaves it as it is, so /sse splices it into an event's text
    instead of json.dumps scanning it (5 MB for a 1080p preview). Only the two preview encoders' URLs are wrapped:
    the engine's (image_preview.cpp) and numpy_outputs' (graph().image_io_previews, else _previews), which make
    `data:image/<format>;base64,<base64>` alone (test_preview_cache.py checks it); a quote, a backslash or a
    non-ASCII character in one would break the spliced JSON."""


_PLACEHOLDER = "\x00preview\x00"
_PLACEHOLDER_JSON = json.dumps(_PLACEHOLDER)


def event_text(message: Event) -> str:
    """An event as /sse sends it: its name and its data as JSON (json.dumps), a broadcast's PreviewUrl previews
    spliced in whole."""
    data: object = message["data"]
    urls: list[str] = []
    if message["event"] == "node-broadcast":
        data = _without_preview_urls(cast("dict[str, object]", data), urls)
    text = json.dumps(data)
    if urls:
        pieces = text.split(_PLACEHOLDER_JSON)
        if len(pieces) == len(urls) + 1:
            spliced = [pieces[0]]
            for url, piece in zip(urls, pieces[1:], strict=True):
                spliced += ['"', url, '"', piece]
            text = "".join(spliced)
        else:  # a string of the event equals the placeholder
            text = json.dumps(message["data"])
    return f"event: {message['event']}\ndata: {text}\n\n"


def _without_preview_urls(
    data: dict[str, object], urls: list[str]
) -> dict[str, object]:
    """The broadcast's data with its PreviewUrls replaced by the placeholder (copies; the event is not changed), the
    URLs collected in json.dumps's order."""
    outputs = data.get("data")
    if not isinstance(outputs, dict):
        return data
    replaced: dict[object, object] = {}
    for key, value in cast("dict[object, object]", outputs).items():
        previews = value.get("previews") if isinstance(value, dict) else None
        if not (isinstance(value, dict) and isinstance(previews, list)):
            replaced[key] = value
            continue
        entries: list[object] = []
        for preview in cast("list[object]", previews):
            url = preview.get("url") if isinstance(preview, dict) else None
            if isinstance(preview, dict) and isinstance(url, PreviewUrl):
                urls.append(url)
                entries.append(
                    {**cast("dict[str, object]", preview), "url": _PLACEHOLDER}
                )
            else:
                entries.append(preview)
        replaced[key] = {**cast("dict[str, object]", value), "previews": entries}
    return {**data, "data": replaced}
