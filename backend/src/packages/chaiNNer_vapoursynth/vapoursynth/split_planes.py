from __future__ import annotations

from api import NodeContext
from nodes.impl.vapoursynth.script import VideoClip, clip_var, one_clip
from nodes.properties.inputs import VideoClipInput
from nodes.properties.outputs import VideoClipOutput

from . import clips_group


@clips_group.register(
    schema_id="chainner:vapoursynth:split_planes",
    name="Split Planes",
    description=[
        (
            "Splits a clip into its three planes, each a gray clip: Y, U and V for a YUV"
            " clip, R, G and B for an RGB clip."
        ),
        (
            "The planes of a YUV clip with subsampled chroma (4:2:0, say) have different"
            " sizes."
        ),
    ],
    icon="MdCallSplit",
    inputs=[VideoClipInput("Clip")],
    outputs=[
        VideoClipOutput("Y"),
        VideoClipOutput("U"),
        VideoClipOutput("V"),
    ],
    node_context=True,
)
def split_planes_node(
    context: NodeContext, clip: VideoClip
) -> tuple[VideoClip, VideoClip, VideoClip]:
    planes = [clip_var(context.node_id, output) for output in range(3)]
    # One statement binds all three planes (vstools' split returns a list of clips).
    # Module-qualified, as every call: a user file split.py cannot shadow it.
    statement = f"{', '.join(planes)} = vstools.split({one_clip(clip, 'Clip')})"
    y, u, v = (
        VideoClip(
            var=var,
            stmt=statement,
            deps=(clip,),
            imports=frozenset({"import vstools"}),
            node_id=context.node_id,
        )
        for var in planes
    )
    return y, u, v
