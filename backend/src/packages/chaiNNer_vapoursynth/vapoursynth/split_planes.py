from __future__ import annotations

import numpy as np

from api import NodeContext
from nodes.impl.vapoursynth.script import VideoClip, clip_var, one_clip
from nodes.properties.inputs import VideoClipInput
from nodes.properties.outputs import VideoClipOutput
from nodes.utils.utils import get_h_w_c

from ..chains import render_images
from . import clips_group

PLANE_TYPE = (
    "match Input0 { Image as image => Image { width: image.width, height: image.height,"
    " channels: 1 }, _ => VideoClip }"
)


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
        "An RGB image gives its R, G and B planes as gray images.",
    ],
    icon="MdCallSplit",
    inputs=[VideoClipInput("Clip").take_images("VideoClip | Image { channels: 3 }")],
    outputs=[
        VideoClipOutput("Y", PLANE_TYPE, images=True),
        VideoClipOutput("U", PLANE_TYPE, images=True),
        VideoClipOutput("V", PLANE_TYPE, images=True),
    ],
    node_context=True,
)
def split_planes_node(
    context: NodeContext, clip: VideoClip | np.ndarray
) -> tuple[VideoClip | np.ndarray, VideoClip | np.ndarray, VideoClip | np.ndarray]:
    def split(source: VideoClip) -> tuple[VideoClip, VideoClip, VideoClip]:
        planes = [clip_var(context.node_id, output) for output in range(3)]
        # One statement binds all three planes (vstools' split returns a list of clips).
        # Module-qualified, as every call: a user file split.py cannot shadow it.
        statement = f"{', '.join(planes)} = vstools.split({one_clip(source, 'Clip')})"
        y, u, v = (
            VideoClip(
                var=var,
                stmt=statement,
                deps=(source,),
                imports=frozenset({"import vstools"}),
                node_id=context.node_id,
            )
            for var in planes
        )
        return y, u, v

    if isinstance(clip, np.ndarray):
        channels = get_h_w_c(clip)[2]
        if channels != 3:
            raise ValueError(
                f"Split Planes takes an RGB image, and this one has {channels} channels."
            )
        r, g, b = render_images(context, [clip], lambda clips: split(clips[0]))
        return r, g, b
    return split(clip)
