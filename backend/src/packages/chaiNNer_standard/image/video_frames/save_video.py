from __future__ import annotations

# `x as x` imports: the native mirror reads these names in native/src/video_io.cpp
import os as os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from subprocess import Popen
from typing import Any, Literal

import ffmpeg as ffmpeg
import numpy as np
from sanic.log import logger as logger

from api import Collector, IteratorInputInfo, KeyInfo, NodeContext
from nodes.groups import Condition, if_enum_group, if_group
from nodes.impl.ffmpeg import FFMpegEnv
from nodes.impl.image_utils import to_uint8
from nodes.impl.item_window import Prepared, register_phases
from nodes.impl.native_graph import graph
from nodes.properties.inputs import (
    DirectoryInput,
    EnumInput,
    ImageInput,
    RelativePathInput,
    SliderInput,
    TextInput,
)
from nodes.properties.inputs.generic_inputs import AudioStreamInput
from nodes.properties.inputs.numeric_inputs import NumberInput
from nodes.utils.utils import get_h_w_c as get_h_w_c

from .. import video_frames_group


class VideoFormat(Enum):
    MKV = "mkv"
    MP4 = "mp4"
    MOV = "mov"
    WEBM = "webm"
    AVI = "avi"
    GIF = "gif"

    @property
    def ext(self) -> str:
        return self.value

    @property
    def encoders(self) -> tuple[VideoEncoder, ...]:
        return graph().video_container_encoders(globals(), self)


class VideoEncoder(Enum):
    H264 = "libx264"
    H265 = "libx265"
    VP9 = "libvpx-vp9"
    FFV1 = "ffv1"

    @property
    def formats(self) -> tuple[VideoFormat, ...]:
        return graph().video_encoder_formats(globals(), self)


class VideoPreset(Enum):
    ULTRA_FAST = "ultrafast"
    SUPER_FAST = "superfast"
    VERY_FAST = "veryfast"
    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"
    SLOWER = "slower"
    VERY_SLOW = "veryslow"


class AudioSettings(Enum):
    AUTO = "auto"
    COPY = "copy"
    TRANSCODE = "transcode"


class Simplicity(Enum):
    SIMPLE = 0
    ADVANCED = 1


class SimpleVideoFormat(Enum):
    MP4_H264 = "mp4_h264"
    MP4_H265 = "mp4_h265"
    WEBM = "webm"
    GIF = "gif"


def get_simple_format(
    simple_video_format: SimpleVideoFormat, quality: int
) -> tuple[VideoFormat, VideoEncoder, VideoPreset, int]:
    return graph().video_simple_format(globals(), simple_video_format, quality)


PARAMETERS: dict[VideoEncoder, list[Literal["preset", "crf"]]] = {
    VideoEncoder.H264: ["preset", "crf"],
    VideoEncoder.H265: ["preset", "crf"],
    VideoEncoder.VP9: ["crf"],
    VideoEncoder.FFV1: [],
}


@dataclass
class Writer:
    container: VideoFormat
    encoder: VideoEncoder | None
    fps: float
    audio: object | None
    audio_settings: AudioSettings
    save_path: str
    output_params: dict[str, str | float]
    global_params: list[str]
    ffmpeg_env: FFMpegEnv
    out: Popen | None = None

    def start(self, width: int, height: int):
        # Create the writer and run process
        graph().video_writer_start(globals(), self, width, height)

    def write_frame(self, img: np.ndarray, prepared: Prepared | None = None):
        # Create the writer and run process
        graph().video_writer_frame(globals(), self, img, prepared)

    def close(self):
        graph().video_writer_close_installed(globals(), self)


@dataclass
class VideoCollector(Collector[np.ndarray, None]):
    """Save Video's collector; its commit phase writes prepared frames to `writer`."""

    writer: Writer


@video_frames_group.register(
    schema_id="chainner:image:save_video",
    name="Save Video",
    description=[
        "Combines an iterable sequence into a video, which it saves to a file.",
        "Uses FFMPEG to write video files.",
        "This iterator is much slower than just using FFMPEG directly, so if you are doing a simple conversion, just use FFMPEG outside chaiNNer instead.",
    ],
    icon="MdVideoCameraBack",
    inputs=[
        ImageInput("Image Sequence", channels=3),
        DirectoryInput(must_exist=False),
        RelativePathInput("Video Name"),
        EnumInput(Simplicity, default=Simplicity.SIMPLE, preferred_style="tabs")
        .with_id(16)
        .with_docs(
            "Simple mode offers a more user-friendly interface, while advanced mode allows for more customization of the underlying FFMPEG options."
        ),
        if_enum_group(16, Simplicity.ADVANCED)(
            EnumInput(
                VideoFormat,
                label="Format",
                label_style="inline",
                option_labels={
                    VideoFormat.MKV: "mkv",
                    VideoFormat.MP4: "mp4",
                    VideoFormat.MOV: "mov",
                    VideoFormat.WEBM: "WebM",
                    VideoFormat.AVI: "avi",
                    VideoFormat.GIF: "GIF",
                },
            ).with_id(4),
            EnumInput(
                VideoEncoder,
                label="Encoder",
                label_style="inline",
                option_labels={
                    VideoEncoder.H264: "H.264 (AVC)",
                    VideoEncoder.H265: "H.265 (HEVC)",
                    VideoEncoder.VP9: "VP9",
                    VideoEncoder.FFV1: "FFV1",
                },
                conditions={
                    VideoEncoder.H264: Condition.enum(4, VideoEncoder.H264.formats),
                    VideoEncoder.H265: Condition.enum(4, VideoEncoder.H265.formats),
                    VideoEncoder.VP9: Condition.enum(4, VideoEncoder.VP9.formats),
                    VideoEncoder.FFV1: Condition.enum(4, VideoEncoder.FFV1.formats),
                },
            )
            .with_id(3)
            .wrap_with_conditional_group(),
            if_enum_group(3, (VideoEncoder.H264, VideoEncoder.H265))(
                EnumInput(
                    VideoPreset,
                    label="Preset",
                    label_style="inline",
                    default=VideoPreset.MEDIUM,
                )
                .with_docs(
                    "For more information on presets, see [here](https://trac.ffmpeg.org/wiki/Encode/H.264#Preset)."
                )
                .with_id(8),
            ),
            if_enum_group(3, (VideoEncoder.H264, VideoEncoder.H265, VideoEncoder.VP9))(
                SliderInput(
                    "CRF",
                    min=0,
                    max=51,
                    default=23,
                    ends=("Best", "Worst"),
                )
                .with_docs(
                    "For more information on CRF, see [here](https://trac.ffmpeg.org/wiki/Encode/H.264#crf)."
                )
                .with_id(9),
            ),
            TextInput(
                "Additional parameters",
                multiline=True,
                allow_empty_string=True,
                has_handle=False,
            )
            .make_optional()
            .with_docs(
                "Allow adding extra FFmpeg parameters the same way you would in CLI. [Link to FFmpeg documentation](https://ffmpeg.org/documentation.html).",
                hint=True,
            )
            .with_id(13),
        ),
        if_enum_group(16, Simplicity.SIMPLE)(
            EnumInput(
                SimpleVideoFormat,
                label="Video Format",
                option_labels={
                    SimpleVideoFormat.MP4_H264: "MP4 (H.264/AVC)",
                    SimpleVideoFormat.MP4_H265: "MP4 (H.265/HEVC)",
                    SimpleVideoFormat.WEBM: "WebM",
                    SimpleVideoFormat.GIF: "GIF",
                },
            ).with_id(17),
            if_group(~Condition.enum(17, SimpleVideoFormat.GIF))(
                SliderInput("Quality", min=0, max=100, default=75).with_id(18),
            ),
        ),
        NumberInput("FPS", default=30, min=1, step=1, precision=4).with_id(14),
        if_group(~Condition.enum(4, VideoFormat.GIF))(
            AudioStreamInput().make_optional().with_id(15).suggest(),
            if_group(Condition.type(15, "AudioStream"))(
                EnumInput(
                    AudioSettings,
                    label="Audio",
                    default=AudioSettings.AUTO,
                    conditions={
                        AudioSettings.COPY: ~Condition.enum(4, VideoFormat.WEBM)
                    },
                    label_style="inline",
                )
                .with_docs(
                    "The first audio stream can be discarded, copied or transcoded at 320 kb/s."
                    " Some audio formats are not supported by selected container, thus copying the audio may fail."
                    " Some players may not output the audio stream if its format is not supported."
                    " If it isn't working for you, verify compatibility or use FFMPEG to mux the audio externally."
                )
                .with_id(10)
            ),
        ),
    ],
    iterator_inputs=IteratorInputInfo(inputs=0),
    outputs=[],
    key_info=KeyInfo.enum(4),
    kind="collector",
    side_effects=True,
    node_context=True,
)
def save_video_node(
    node_context: NodeContext,
    _: None,
    save_dir: Path,
    video_name: str,
    simplicity: Simplicity,
    container: VideoFormat,
    encoder: VideoEncoder,
    video_preset: VideoPreset,
    crf: int,
    additional_parameters: str | None,
    simple_video_format: SimpleVideoFormat,
    quality: int,
    fps: float,
    audio: Any,
    audio_settings: AudioSettings,
) -> Collector[np.ndarray, None]:
    return graph().video_save(
        globals(),
        node_context,
        _,
        save_dir,
        video_name,
        simplicity,
        container,
        encoder,
        video_preset,
        crf,
        additional_parameters,
        simple_video_format,
        quality,
        fps,
        audio,
        audio_settings,
    )


def prepare_video_frame(inputs: list[np.ndarray]) -> np.ndarray:
    """The uint8 frame Save Video would write for this iteration (SP3-P9)."""
    return to_uint8(inputs[0], normalized=True)


def commit_video_frame(
    collector: VideoCollector, inputs: list[np.ndarray], prepared: Prepared
) -> None:
    """Save Video's iteration with its prepared frame as the pipe payload."""
    collector.writer.write_frame(inputs[0], prepared)


register_phases("chainner:image:save_video", prepare_video_frame, commit_video_frame)
