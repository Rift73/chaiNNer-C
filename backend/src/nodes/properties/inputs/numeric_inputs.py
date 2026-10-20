from __future__ import annotations

import math
from typing import Literal, Union

import navi
from api import BaseInput, InputConversion, InputId, InputKind, OutputId

from ...utils.utils import round_half_up
from .label import LabelStyle, get_default_label_style

Precision = Union[int, Literal["unlimited"]]


def _get_step(precision: Precision) -> float:
    if precision == "unlimited":
        return 1
    return 10**-precision


def _is_int(precision: Precision) -> bool:
    return precision == 0


def clamp_number(
    value: float | int,
    precision: Precision,
    min_value: float | int | None,
    max_value: float | int | None,
) -> float | int:
    # Convert proper number type
    if precision != "unlimited":
        value = round_half_up(value) if precision == 0 else round(value, precision)

    # Clamp to max and min, correcting for max/min not aligning with offset + n * step
    if max_value is not None:
        value = min(value, max_value)
    if min_value is not None:
        value = max(value, min_value)

    # guarantee integers
    if _is_int(precision):
        return int(value)
    else:
        return float(value)


def get_number_type(
    min_value: float | int | None,
    max_value: float | int | None,
    precision: Precision,
) -> navi.ExpressionJson:
    if not _is_int(precision):
        # step is not an integer
        return navi.interval(min_value, max_value)
    return navi.int_interval(min_value, max_value)


class NumberInput(BaseInput):
    """Input a number"""

    def __init__(
        self,
        label: str,
        *,
        precision: Precision = 0,
        step: float | int | None = None,
        default: float | int = 0,
        min: float | int | None = 0,
        max: float | int | None = None,
        unit: str | None = None,
        note_expression: str | None = None,
        kind: InputKind = "number",
        hide_trailing_zeros: bool = True,
        label_style: LabelStyle | None = None,
        has_handle: bool = True,
    ):
        super().__init__("number", label, kind=kind, has_handle=has_handle)
        self.precision: int | Literal["unlimited"] = precision
        # controls_step is for increment/decrement arrows.
        self.step: float | int = step if step is not None else _get_step(precision)
        self.default = default
        self.min = min
        self.max = max
        self.unit = unit
        self.note_expression = note_expression
        self.hide_trailing_zeros = hide_trailing_zeros
        self.label_style: LabelStyle = label_style or get_default_label_style(label)

        self.associated_type = float if not _is_int(precision) else int

        self.input_type = get_number_type(self.min, self.max, self.precision)
        if self.precision == 0:
            self.input_conversions = [InputConversion("number", "round(Input)")]

    def to_dict(self):
        return {
            **super().to_dict(),
            "min": self.min,
            "max": self.max,
            "noteExpression": self.note_expression,
            "def": self.default,
            "precision": 100 if self.precision == "unlimited" else self.precision,
            "controlsStep": self.step,
            "unit": self.unit,
            "hideTrailingZeros": self.hide_trailing_zeros,
            "labelStyle": self.label_style,
            "hasHandle": self.has_handle,
        }

    def make_optional(self):
        raise ValueError("NumberInput and SliderInput cannot be made optional")

    def enforce(self, value: object):
        assert isinstance(value, (int, float))

        if math.isnan(value):
            raise ValueError("NaN is not a valid number")

        return clamp_number(value, self.precision, self.min, self.max)


class SliderInput(NumberInput):
    """Input for integer number via slider"""

    def __init__(
        self,
        label: str,
        *,
        precision: Precision = 0,
        step: float | int | None = None,
        slider_step: float | int | None = None,
        min: float | int = 0,
        max: float | int = 100,
        default: float | int = 50,
        unit: str | None = None,
        note_expression: str | None = None,
        ends: tuple[str | None, str | None] = (None, None),
        hide_trailing_zeros: bool = False,
        gradient: list[str] | None = None,
        scale: Literal["linear", "log", "log-offset", "sqrt"] = "linear",
        has_handle: bool = True,
    ):
        super().__init__(
            label,
            precision=precision,
            step=step,
            default=default,
            min=min,
            max=max,
            unit=unit,
            note_expression=note_expression,
            kind="slider",
            hide_trailing_zeros=hide_trailing_zeros,
            has_handle=has_handle,
        )
        self.ends = ends
        self.slider_step = (
            slider_step
            if slider_step is not None
            else (step if step is not None else _get_step(precision))
        )
        self.gradient = gradient
        self.scale = scale

    def to_dict(self):
        return {
            **super().to_dict(),
            "ends": self.ends,
            "sliderStep": self.slider_step,
            "gradient": self.gradient,
            "scale": self.scale,
        }


class VideoSeekInput(NumberInput):
    """A video node's seek bar: the number of a frame in the video (0 is the first).

    The app's seek bar spans the node's Frame Count output and steps by its FPS output (both
    as the node's last run reported them), and its In and Out buttons set the node's frame
    range: `frames_input` (an enum with the values "all" and "range") to "range" and
    `in_input` or `out_input` to the frame. It has no handle."""

    def __init__(
        self,
        label: str,
        *,
        frames_input: InputId | int,
        in_input: InputId | int,
        out_input: InputId | int,
        frame_count_output: OutputId | int,
        fps_output: OutputId | int,
    ):
        super().__init__(
            label,
            default=0,
            min=0,
            max=None,
            kind="video-seek",
            has_handle=False,
        )
        self.frames_input = InputId(frames_input)
        self.in_input = InputId(in_input)
        self.out_input = InputId(out_input)
        self.frame_count_output = OutputId(frame_count_output)
        self.fps_output = OutputId(fps_output)

    def to_dict(self):
        return {
            **super().to_dict(),
            "framesInput": self.frames_input,
            "inInput": self.in_input,
            "outInput": self.out_input,
            "frameCountOutput": self.frame_count_output,
            "fpsOutput": self.fps_output,
        }
