from __future__ import annotations

from pathlib import Path

from api import NodeContext
from nodes.impl.vapoursynth.script import VideoClip
from nodes.properties.inputs import DirectoryInput, RelativePathInput

from ..chains import named_clip_inputs, named_clips, write_script
from . import clips_group


@clips_group.register(
    schema_id="chainner:vapoursynth:save_script",
    name="Save Script",
    description=[
        (
            "Saves the VapourSynth script of up to 16 clips as a .vpy file, for vspipe,"
            " vsview or any other VapourSynth program. Clip 1 is output 0, the next"
            " connected clip output 1, and so on; each output has its name in vsview."
        ),
        (
            "The script runs on this computer: it loads chaiNNer's own helpers from this"
            " installation, and a clip from Frames to Clip reads a temporary file, which"
            " the next run of that node replaces and a start of chaiNNer deletes once"
            " it is a day old."
        ),
    ],
    icon="MdSave",
    inputs=[
        DirectoryInput(must_exist=False),
        RelativePathInput("Script Name"),
        *named_clip_inputs(),
    ],
    outputs=[],
    side_effects=True,
    node_context=True,
)
def save_script_node(
    context: NodeContext, directory: Path, name: str, *clips: VideoClip | str | None
) -> None:
    path = (directory / f"{name}.vpy").resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_script(context, named_clips(clips), path)
