from __future__ import annotations

from api import NodeContext
from nodes.impl.vapoursynth.runner import VSScriptError, clip_info, launch_preview
from nodes.impl.vapoursynth.script import VideoClip
from system import glibc_version

from .. import vsview_unavailable_reason
from ..chains import chain_dir, named_clip_inputs, named_clips, write_script
from . import clips_group


@clips_group.register(
    schema_id="chainner:vapoursynth:preview_clips",
    name="Preview Clips",
    description=[
        (
            "Shows up to 16 clips in vsview, a separate viewer window, each as an"
            " output under its name."
        ),
        (
            "Run does not render the clips: it writes the script, checks that it loads"
            " and opens vsview, which computes each frame when you scrub to it. The"
            " first Run on a new video can take a while, as BestSource reads the whole"
            " file once to index it; later opens are quick."
        ),
        (
            "While the window stays open, Run again and the open viewer reloads by"
            " itself; if it does not, press Reload Script (Ctrl+R). After you close the"
            " window, the next Run opens a new one."
        ),
        (
            "Each run first checks the script in VapourSynth: an error shows on the node"
            " that caused it, no window opens, and an open window keeps the last script"
            " that worked."
        ),
    ],
    icon="BsEyeFill",
    inputs=named_clip_inputs(
        "A clip to show in vsview, as a tab under its name. vsview computes each frame"
        " when you scrub to it. Run again and the open viewer reloads by itself; if it"
        " does not, press Reload Script (Ctrl+R).",
        by_method=True,
    ),
    keywords=["Preview", "Viewer", "Scrub", "vsview"],
    outputs=[],
    side_effects=True,
    node_context=True,
)
def preview_clips_node(context: NodeContext, *clips: VideoClip | str | None) -> None:
    unavailable = vsview_unavailable_reason(glibc_version())
    if unavailable is not None:
        raise VSScriptError(unavailable)
    outputs = named_clips(clips, by_method=True)
    folder = chain_dir(context)
    # vspipe evaluates the whole script: a failing statement raises its node's error
    # here, in chaiNNer, instead of only inside vsview. A copy is checked, so that an
    # open vsview keeps the last script that worked (its Reload would show the error).
    check = write_script(context, outputs, folder / "preview.check.vpy")
    try:
        clip_info(check, 0, context)
    finally:
        check.unlink(missing_ok=True)
    script = write_script(context, outputs, folder / "preview.vpy")
    title = f"{', '.join(name for _, name in outputs)} - chaiNNer-C"
    launch_preview(script, context.node_id, title)
