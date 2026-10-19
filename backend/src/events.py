from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Literal, TypedDict, Union

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
    event: Literal["backend-started", "backend-ready"]
    data: None


BackendEvent = Union[BackendStatusEvent, BackendStateEvent]
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
    an event: a new listener first gets the latest backend status and, once the backend
    is ready, backend-ready.
    """

    def __init__(self) -> None:
        self._listeners: set[EventQueue] = set()
        self._status: BackendEvent | None = None
        self._ready: BackendEvent | None = None

    def put(self, event: BackendEvent) -> None:
        if event["event"] == "backend-status":
            self._status = event
        elif event["event"] == "backend-ready":
            self._ready = event
        for listener in self._listeners:
            listener.put(event)

    def listen(self) -> EventQueue:
        listener = EventQueue()
        for event in (self._status, self._ready):
            if event is not None:
                listener.put(event)
        self._listeners.add(listener)
        return listener

    def stop_listening(self, listener: EventQueue) -> None:
        self._listeners.discard(listener)
