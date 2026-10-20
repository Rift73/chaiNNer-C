"""VideoSeekInput: the seek bar's schema (the ids the app's seek bar reads) and its values."""

from __future__ import annotations

import pytest

# chaiNNer's inputs load the native backend, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

from nodes.properties.inputs import VideoSeekInput


def seek_input() -> VideoSeekInput:
    return VideoSeekInput(
        "Seek",
        frames_input=10,
        in_input=11,
        out_input=12,
        frame_count_output=12,
        fps_output=4,
    ).with_id(13)


def test_schema_names_the_inputs_and_outputs_the_seek_bar_uses():
    schema = seek_input().to_dict()

    assert schema["kind"] == "video-seek"
    assert schema["id"] == 13
    assert schema["def"] == 0
    assert schema["min"] == 0
    assert schema["max"] is None
    assert schema["precision"] == 0
    assert schema["hasHandle"] is False
    assert schema["framesInput"] == 10
    assert schema["inInput"] == 11
    assert schema["outInput"] == 12
    assert schema["frameCountOutput"] == 12
    assert schema["fpsOutput"] == 4


def test_value_is_a_frame_number():
    seek = seek_input()

    assert seek.associated_type is int
    assert seek.enforce(41) == 41
    assert seek.enforce(41.6) == 42
    assert seek.enforce(-3) == 0
    with pytest.raises(ValueError, match="NaN"):
        seek.enforce(float("nan"))
