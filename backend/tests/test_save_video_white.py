"""Save Video's gbrp16le code of white: by the -pix_fmt given, else by what the encoder
takes (FFmpeg keeps RGB on an encoder that takes it and converts to YUV on one that does
not)."""

from __future__ import annotations

from typing import Any, cast

import pytest

from packages.chaiNNer_standard.image.video_frames.save_video import (
    VideoEncoder,
    VideoFormat,
    Writer,
)


def writer(encoder: VideoEncoder | None, params: dict[str, Any]) -> Writer:
    return Writer(
        container=VideoFormat.MKV,
        encoder=encoder,
        fps=24,
        audio=None,
        audio_settings=cast(Any, None),
        save_path="out.mkv",
        output_params=params,
        global_params=[],
        ffmpeg_env=cast(Any, None),
        target=None,
    )


@pytest.mark.parametrize(
    ("encoder", "params", "white"),
    [
        (VideoEncoder.H264, {}, 65280.0),
        (None, {}, 65280.0),
        (VideoEncoder.FFV1, {}, 65535.0),
        (VideoEncoder.VP9, {}, 65535.0),
        (VideoEncoder.H264, {"pix_fmt": "yuv420p"}, 65280.0),
        (VideoEncoder.FFV1, {"pix_fmt": "gray10le"}, 65280.0),
        (VideoEncoder.FFV1, {"pix_fmt": "gbrp16le"}, 65535.0),
        (VideoEncoder.FFV1, {"pix_fmt": "bgr0"}, 65535.0),
    ],
)
def test_white_follows_the_pix_fmt_else_the_encoder(
    encoder: VideoEncoder | None, params: dict[str, Any], white: float
):
    assert writer(encoder, params).rgb_white == white
