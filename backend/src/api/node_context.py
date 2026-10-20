from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Literal

from .settings import SettingsParser
from .types import NodeId


class Aborted(Exception):
    pass


class NodeAttributedError(Exception):
    """An error that the run reports on the node `node_id` of the chain rather than on
    the node that raised it, e.g. the node whose part of a script failed in a child
    process. With no node_id, or one that is not in the chain, it is reported as usual."""

    def __init__(self, message: str, node_id: str | None = None) -> None:
        super().__init__(message)
        self.node_id = node_id


class Progress(ABC):
    @property
    @abstractmethod
    def aborted(self) -> bool:
        """
        Returns whether the current operation was aborted.
        """

    @property
    @abstractmethod
    def paused(self) -> bool:
        """
        Returns whether the current operation was paused.
        """

    def check_aborted(self) -> None:
        """
        Raises an `Aborted` exception if the current operation was aborted. Does nothing otherwise.
        """

        if self.aborted:
            raise Aborted()

    def suspend(self) -> None:
        """
        If the operation was aborted, this method will throw an `Aborted` exception.
        If the operation is paused, this method will wait until the operation is resumed or aborted.
        """

        while True:
            self.check_aborted()
            if not self.paused:
                break
            time.sleep(0.1)

    @abstractmethod
    def set_progress(self, progress: float) -> None:
        """
        Sets the progress of the current node execution. `progress` must be a value between 0 and 1.

        Raises an `Aborted` exception if the current operation was aborted.
        """

    def sub_progress(self, offset: float, length: float) -> Progress:
        """
        Returns a new `NodeProgress` object that represents a sub-progress of the current operation.

        The progress range of the sub-progress is defined by `offset` and `length`. `offset` must be a value between 0
        and 1, and `length` must be a positive value such that `offset + length <= 1`.

        The real progress of the sub-progress is calculated as `offset + progress * length`, where `progress` is the
        progress value passed to `set_progress` of the sub-progress.
        """
        return _SubProgress(self, offset, length)

    @staticmethod
    def noop_progress() -> Progress:
        """
        Returns a `Progress` object that does nothing. It is never paused or aborted and does not report any progress.
        """
        return _NoopProgress()


class _NoopProgress(Progress):
    @property
    def aborted(self) -> Literal[False]:
        return False

    @property
    def paused(self) -> Literal[False]:
        return False

    def check_aborted(self) -> None:
        pass

    def suspend(self) -> None:
        pass

    def set_progress(self, progress: float) -> None:
        pass

    def sub_progress(self, offset: float, length: float) -> Progress:
        return _NoopProgress()


class _SubProgress(Progress):
    def __init__(self, parent: Progress, offset: float, length: float):
        self._parent = parent
        self._offset = offset
        self._length = length

    @property
    def aborted(self) -> bool:
        return self._parent.aborted

    @property
    def paused(self) -> bool:
        return self._parent.paused

    def check_aborted(self) -> None:
        self._parent.check_aborted()

    def suspend(self) -> None:
        self._parent.suspend()

    def set_progress(self, progress: float) -> None:
        self._parent.set_progress(self._offset + progress * self._length)

    def sub_progress(self, offset: float, length: float) -> _SubProgress:
        return _SubProgress(
            self._parent,
            offset=self._offset + offset * self._length,
            length=length * self._length,
        )


class IterationView(ABC):
    """
    What a node sees of the generator loop that runs it once per item (its context's
    `iteration`). Items reach the node one at a time, in order.
    """

    @property
    @abstractmethod
    def index(self) -> int:
        """
        The index of the item the node runs for: the loop's index base for its first item
        (the generators' shared Generator.index_base, else 0: Load Video's Range starts at
        In), then one more per item, an item that failed (a generator that defers errors)
        included. Context items before the range (see `context`) come first, numbered
        base - before .. base - 1, so the node sees one sequence through them.
        """

    @property
    def context(self) -> bool:
        """
        True on a context item before the loop's range: a neighbour a temporal node needs
        (NodeData.temporal_context) that the generator reads only for it. Only the loop's
        temporal nodes and the nodes they need run for it; its outputs are discarded, so a
        node only updates its state. Never saved, collected, broadcast or counted in the
        progress. The items after the range are never committed: they are only served to
        `peek`.
        """
        return False

    @property
    @abstractmethod
    def source(self) -> str:
        """
        The schema id of the generator whose sequence iterates the node: the lineage of its
        first iterated input (inputs in order), as the "iterated" condition sees an input's.
        """

    @abstractmethod
    def iterated_by(self, input_id: int) -> str | None:
        """
        The schema id of the generator whose sequence iterates this input, as the "iterated"
        condition sees it, or None when the input is not iterated (a constant, or a value
        from outside the loop). Raises ValueError for an unknown input.
        """

    @property
    @abstractmethod
    def state(self) -> object | None:
        """
        The node's state slot for this loop run: None at the loop's start, kept from one
        item to the next, and dropped (back to None, the view detached) when the loop ends,
        fails or is cancelled. Never shared between loop runs or nodes.
        """

    @state.setter
    @abstractmethod
    def state(self, value: object | None) -> None: ...

    @abstractmethod
    def lookahead(self, input_id: int) -> int:
        """
        The largest offset `peek` serves for this input in this loop run, or 0 when it serves
        none; the same at every item of the run.

        It is K - 1, K being the item window of the loop's generator group, when the group
        reads ahead (K > 1: CHAINNER_C_ITEM_WINDOW is not 1, and when a generator reads
        files per item, no writer's output directory is unresolved or holds one of them;
        Load Video reads none), the executor's pool has two threads or more (the server's
        has 8), and the input is connected to one of these:
        - an iterated output of one of the group's generators;
        - a node the window computes ahead: a pure CPU node (an image or utility node without
          side effects, context or lazy input) or an audited locked-device node, not Into
          Directory, whose own iterated inputs are generator outputs or such nodes.

        Otherwise 0: no window (K = 1), an unconnected input, an input fed by a node outside
        the loop or by a generator's non-iterated output, or by a node the window never
        computes ahead (a GPU node, a node with side effects or a context, ...).

        The window's K is decided once per loop run: at the range's second item (item 1
        without context items), or when a node of the group first looks ahead (lookahead or
        peek) before it, during a context item or the range's first item. A writer whose
        output directory comes from a node that has not run yet by then counts as
        unresolved, so K is 1 when a generator of the group reads files per item.

        Like `peek`, call it from the node's own call, never from the event loop's thread.
        """

    @abstractmethod
    def peek(self, input_id: int, offset: int) -> object | None:
        """
        The value this input receives at item `index + offset` (offset >= 1), enforced as the
        node receives it: the very object (or the same view of it) the node gets when that
        item comes. It blocks until the value is there: the window produces that item and
        computes the nodes that feed the input now if it has not yet, beyond its memory budget
        and while paused, as for the item the loop waits for, never more than K - 1 items
        ahead. Nothing else changes: no event, side effect or extra node run.

        None when `offset > lookahead(input_id)`, when the item does not exist (the sequence
        ends before it), or when it fails before reaching this input: a generator raised for
        it or, failing fast, for an earlier item; zipped generators did not all yield; a node
        that feeds the input failed for it; or the input's own check rejects the value. The
        error itself is raised at its item, as without the look-ahead.

        Raises Aborted once the run was stopped, ValueError for an offset below 1 or an
        unknown input, and RuntimeError on the event loop's thread.
        """


class NodeContext(Progress, ABC):
    """
    The execution context of the current node.
    """

    @property
    def iteration(self) -> IterationView | None:
        """
        The generator loop that runs the node once per item, while it runs an item on the
        loop's commit path; None outside a generator loop, for generators and collectors,
        and for a node the item window computes ahead of the loop.
        """
        return None

    @property
    def chosen_item(self) -> int | None:
        """
        The item's number when the node's generator loop runs exactly one chosen item
        (Generator.chosen_item: Load Video's "Run this frame"), for the loop's nodes and
        collectors, as long as the run lasts; None otherwise, for a loop of several zipped
        generators, and outside a loop.
        """
        return None

    @property
    @abstractmethod
    def settings(self) -> SettingsParser:
        """
        Returns the settings of the current node execution.
        """

    @property
    @abstractmethod
    def node_id(self) -> NodeId:
        """
        The id of the current node. `/clear-cache/individual` clears by this id.
        """

    @property
    @abstractmethod
    def storage_dir(self) -> Path:
        """
        The path of a directory where nodes can store files.

        This directory persists between node executions, and its contents are shared between different nodes.
        """

    @abstractmethod
    def add_cleanup(
        self, fn: Callable[[], None], after: Literal["node", "chain"] = "chain"
    ) -> None:
        """
        Registers a function that will be called when the chain execution is finished (if set to chain mode) or after node execution is finished (node mode).

        Registering the same function (object) twice will only result in the function being called once.
        """
