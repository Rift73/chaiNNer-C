from __future__ import annotations

from api import NodeContext
from nodes.impl.vapoursynth.runner import clip_info, launch_preview
from nodes.impl.vapoursynth.script import VideoClip

from ..chains import chain_dir, named_clip_inputs, named_clips, write_script
from . import clips_group


@clips_group.register(
    schema_id="chainner:vapoursynth:preview_clips",
    name="Preview Clips",
    description=[
        "Shows up to 16 clips in vsview, each as an output under its name.",
        (
            "The first run opens a vsview window. While the window stays open, a run"
            " rewrites its script without opening another window: press Reload Script"
            " (Ctrl+R) in vsview to see the changes. After you close the window, the next"
            " run opens a new one."
        ),
        (
            "Each run first checks the script in VapourSynth: an error shows on the node"
            " that caused it, and no window opens."
        ),
    ],
    icon="BsEyeFill",
    inputs=named_clip_inputs(),
    outputs=[],
    side_effects=True,
    node_context=True,
)
def preview_clips_node(context: NodeContext, *clips: VideoClip | str | None) -> None:
    script = write_script(
        context, named_clips(clips), chain_dir(context) / "preview.vpy"
    )
    # vspipe evaluates the whole script: a failing statement raises its node's error
    # here, in chaiNNer, instead of only inside vsview.
    clip_info(script, 0)
    launch_preview(script, context.node_id)
