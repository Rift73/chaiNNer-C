from __future__ import annotations

from typing import Any, Generic, Literal, Mapping, TypeVar

import navi

from .types import InputId, OutputId

OutputKind = Literal["large-image", "tagged", "generic"]
BroadcastData = Mapping[str, object]

T = TypeVar("T")


class BaseOutput(Generic[T]):
    def __init__(
        self,
        output_type: navi.ExpressionJson,
        label: str,
        kind: OutputKind = "generic",
        has_handle: bool = True,
        associated_type: Any = None,
    ):
        self.output_type: navi.ExpressionJson = output_type
        self.label: str = label
        self.id: OutputId = OutputId(-1)
        self.never_reason: str | None = None
        self.kind: OutputKind = kind
        self.has_handle: bool = has_handle
        self.passthrough_of: InputId | None = None
        self.nullable: bool = False

        self.associated_type: Any = associated_type

        # Optional documentation
        self.description: str | None = None
        self.should_suggest: bool = False

    def to_dict(self):
        return {
            "id": self.id,
            "type": self.output_type,
            "label": self.label,
            "neverReason": self.never_reason,
            "kind": self.kind,
            "hasHandle": self.has_handle,
            "passthroughOf": self.passthrough_of,
            "description": self.description,
            "suggest": self.should_suggest,
        }

    def with_id(self, output_id: OutputId | int):
        self.id = OutputId(output_id)
        return self

    def with_never_reason(self, reason: str):
        self.never_reason = reason
        return self

    def with_docs(self, *description: str):
        self.description = "\n\n".join(description)
        return self

    def suggest(self):
        self.should_suggest = True
        return self

    def as_passthrough_of(self, input_id: InputId | int):
        self.passthrough_of = InputId(input_id)
        return self

    def null_unless(self, input_id: InputId | int, variants: navi.ExpressionJson):
        """Makes this output exist only while the dropdown `input_id` has one of
        `variants` (its options' types, e.g. "Mode::Planes | Mode::Both"). Under the other
        options its type is `null`, which the frontend does not connect, and the node
        returns None for it."""
        self.output_type = navi.match(
            f"Input{input_id}", (variants, None, self.output_type), default="null"
        )
        self.nullable = True
        return self

    def get_broadcast_data(self, _value: T) -> BroadcastData | None:
        return None

    def get_broadcast_type(self, _value: T) -> navi.ExpressionJson | None:
        return None

    def enforce(self, value: object) -> T:
        assert value is not None
        return value  # type: ignore

    # This is the method that should be called by the processing code
    def enforce_(self, value: object) -> T | None:
        if self.nullable and value is None:
            return None
        return self.enforce(value)
