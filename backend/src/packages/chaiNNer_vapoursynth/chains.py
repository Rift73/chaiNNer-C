"""What the VapourSynth I/O nodes share: each node's folder in chaiNNer's storage, the
chain's compiled script, and the named clips of Preview Clips and Save Script."""

from __future__ import annotations

import ast
import os
import re
import stat
import time
import uuid
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from threading import Lock
from typing import Callable, Mapping, Sequence, cast

import numpy as np
from sanic.log import logger

from api import MB, BaseInput, NestedGroup, NodeContext
from nodes.condition import Condition
from nodes.groups import if_group, optional_list_group
from nodes.impl.native_graph import graph
from nodes.impl.vapoursynth.catalog_types import Method
from nodes.impl.vapoursynth.runner import VSScriptError, clip_props, save_frames
from nodes.impl.vapoursynth.schema_gen import MAIN_ID
from nodes.impl.vapoursynth.script import (
    ClipShown,
    VideoClip,
    VSFilter,
    compile_script,
    image_clip,
)
from nodes.properties.inputs import TextInput, VideoClipInput
from nodes.utils.utils import file_identity, get_h_w_c

from .settings import get_settings

# Preview Clips and Save Script take up to this many clips, each with a name.
NAMED_CLIPS = 16
# Frames to Clip's temporary files older than this go when the package loads.
OLD_FILE_HOURS = 24
# The family whose methods move pixels: an image's alpha goes along with them.
GEOMETRY_FAMILY = "Resize"
# So does a resize's source window, in any family, at the same size too.
SOURCE_WINDOW = re.compile(r"\bsrc_(left|top|width|height)\s*=")


def chain_dir(context: NodeContext) -> Path:
    """The node's own folder for its scripts and temporary files:
    <storage>/vapoursynth/chains/<node id>."""
    key = re.sub(r"[^0-9A-Za-z_-]", "_", context.node_id)
    folder = context.storage_dir.resolve() / "vapoursynth" / "chains" / key
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def remove_old_videos(storage: Path | None) -> None:
    """Deletes Frames to Clip's temporary videos and BestSource's indexes of them,
    `chains/*/frames-*` in the VapourSynth storage folder, older than OLD_FILE_HOURS.
    A node's next run deletes its own; these are of nodes deleted or not run again.
    Scripts and logs stay. Never raises, so that the package loads; a file it cannot
    delete (one that an open vsview reads) is logged and kept."""
    if storage is None:
        return
    cutoff = time.time() - OLD_FILE_HOURS * 3600
    count = size = 0
    try:
        for path in (storage / "chains").glob("*/frames-*"):
            if path.suffix in (".vpy", ".log"):
                continue
            try:
                info = path.lstat()
                if not stat.S_ISREG(info.st_mode) or info.st_mtime >= cutoff:
                    continue
                path.unlink()
            except OSError as error:
                logger.warning(
                    "Could not delete the old temporary file %s: %s", path, error
                )
                continue
            count += 1
            size += info.st_size
    except Exception:
        logger.exception("Could not clean up Frames to Clip's old temporary files.")
    if count:
        logger.info(
            "Deleted %d temporary files of Frames to Clip older than %d hours"
            " (%.1f MB).",
            count,
            OLD_FILE_HOURS,
            size / MB,
        )


def write_script(
    context: NodeContext, outputs: Sequence[tuple[VideoClip, str]], path: Path
) -> Path:
    """Writes the script of `outputs` (output i = the i-th (clip, name) pair) to path,
    through a temporary file in its folder (native/include/atomic_file.hpp): the bytes
    path.write_text(script, encoding="utf-8") wrote, an equal file left untouched."""
    script = compile_script(outputs, cache_mb=get_settings(context).cache_mb)
    graph().atomic_write(path, script.replace("\n", os.linesep).encode("utf-8"))
    return path


# The field orders of _FieldBased's values, as a clip's node shows them.
FIELD_ORDERS = {0: "Progressive", 1: "BFF", 2: "TFF"}


# shown_clip's results of the last clips, by _shown_key, as video.py keeps its scans:
# a clip whose statement and files are unchanged is not evaluated again.
SHOWN: OrderedDict[tuple, ClipShown] = OrderedDict()
SHOWN_LOCK = Lock()
SHOWN_KEPT = 64


def _shown_key(clip: VideoClip) -> tuple:
    """What shown_clip's result depends on: the statements up to the clip (the loader
    and its options), their imports, and the version (file_identity) of each file a
    statement names."""
    records: list[VideoClip | VSFilter] = [clip]
    texts: list[str] = []
    for record in records:
        texts.append(record.stmt if isinstance(record, VideoClip) else record.expr)
        records.extend(record.deps)
    names = {
        node.value
        for text in texts
        for node in ast.walk(ast.parse(text))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    files = tuple(file_identity(Path(n)) for n in sorted(names) if Path(n).is_file())
    return (tuple(texts), clip.imports, files)


def shown_clip(context: NodeContext, clip: VideoClip) -> VideoClip:
    """The clip with its size, frame rate and field order, for its node to show (Load
    Clip's): one evaluation of the script up to it, which reads its first frame, unless
    an unchanged clip was evaluated before (SHOWN)."""
    key = _shown_key(clip)
    with SHOWN_LOCK:
        kept = SHOWN.get(key)
        if kept is not None:
            SHOWN.move_to_end(key)
            return replace(clip, shown=kept)
    script = write_script(context, [(clip, "Clip")], chain_dir(context) / "shown.vpy")
    info, props, _ = clip_props(script, 0, context)
    shown = ClipShown(
        width=info.width,
        height=info.height,
        fps=info.fps_num / info.fps_den if info.fps_num else 0.0,
        field_order=FIELD_ORDERS.get(props.get("_FieldBased", -1), ""),
    )
    with SHOWN_LOCK:
        SHOWN[key] = shown
        while len(SHOWN) > SHOWN_KEPT:
            SHOWN.popitem(last=False)
    return replace(clip, shown=shown)


def render_images(
    context: NodeContext,
    images: Sequence[np.ndarray],
    build: Callable[[Sequence[VideoClip]], Sequence[VideoClip]],
    route: str = "rgb",
) -> list[np.ndarray]:
    """Images through one VapourSynth evaluation: each becomes a clip of one frame, read
    from a temporary NumPy file in the node's folder, `build` writes the statements on
    those clips, and each clip it gives comes back as an image."""
    for image in images:
        channels = get_h_w_c(image)[2]
        if channels not in (1, 3):
            raise ValueError(
                f"A VapourSynth clip takes RGB or gray images, and this one has {channels}"
                " channels: remove its alpha first."
            )
    folder = chain_dir(context)
    stem = f"frames-{uuid.uuid4().hex[:12]}"
    script = folder / f"{stem}.vpy"
    sources = [folder / f"{stem}-in{i}.npy" for i in range(len(images))]
    results: list[Path] = []
    try:
        for path, image in zip(sources, images):
            np.save(path, image, allow_pickle=False)
        clips = [
            image_clip(path, context.node_id, index, route)
            for index, path in enumerate(sources)
        ]
        outputs = build(clips)
        results = [folder / f"{stem}-out{i}.npy" for i in range(len(outputs))]
        source = compile_script(
            [(clip, f"Image {i}") for i, clip in enumerate(outputs)],
            cache_mb=get_settings(context).cache_mb,
        )
        save_frames(script, source, results, context)
        return [np.load(path, allow_pickle=False) for path in results]
    finally:
        for path in (script, *sources, *results):
            path.unlink(missing_ok=True)


def filter_image(
    context: NodeContext,
    images: Sequence[np.ndarray],
    build: Callable[[Sequence[VideoClip]], VideoClip],
    route: str = "rgb",
    moves_pixels: bool = False,
) -> np.ndarray:
    """The image `images[0]` through the function `build` writes, with `images[1:]` as
    its other clips. A clip has no alpha: the function runs on the colour, and the alpha
    stays as it is, unless the function moves pixels or changes the size: then it runs
    on the alpha too, as a gray image. A gray result has no alpha."""

    def run(sources: Sequence[np.ndarray]) -> np.ndarray:
        return render_images(context, sources, lambda c: [build(c)], route)[0]

    height, width, channels = get_h_w_c(images[0])
    if channels != 4:
        return run(images)
    colour = run([images[0][:, :, :3], *images[1:]])
    out_height, out_width, out_channels = get_h_w_c(colour)
    if out_channels == 1:
        return colour
    alpha = images[0][:, :, 3]
    if moves_pixels or (out_height, out_width) != (height, width):
        if len(images) > 1:
            raise ValueError(
                "This method moves pixels, and its alpha cannot go along with a second"
                " image: remove the alpha first."
            )
        try:
            alpha = run([alpha])
        except VSScriptError as error:
            raise VSScriptError(
                "This method also runs on the image's alpha, as a gray image, and that"
                f" failed: remove the alpha first. {error}",
                error.node_id,
            ) from error
        if get_h_w_c(alpha)[:2] != (out_height, out_width):
            raise ValueError(
                "The alpha came out in another size than the colour: remove the alpha"
                " first."
            )
        # A method that makes RGB of what it gets (a Resize to RGBS) has three equal planes.
        if alpha.ndim == 3:
            alpha = alpha[:, :, 0]
    return np.dstack((colour, alpha))


def image_node_output(
    context: NodeContext,
    method: Method,
    values: Mapping[int, object],
    write: Callable[[Mapping[int, object]], VideoClip | VSFilter],
) -> np.ndarray:
    """A family node's result on the image at its Clip input: `write` is the node's
    statement from its input values, and every image among them becomes a clip."""
    if not method.frame_route:
        raise ValueError(
            f"{method.label} takes clips only, not images: it looks at other frames, or"
            " its result is more than an image holds."
        )
    main = cast(np.ndarray, values[MAIN_ID])
    height, width = get_h_w_c(main)[:2]
    if min(height, width) < method.frame_min_side:
        side = method.frame_min_side
        raise ValueError(
            f"{method.label} needs an image of at least {side}x{side} pixels, and this one"
            f" is {width}x{height}."
        )
    inputs = [i for i, v in values.items() if isinstance(v, np.ndarray)]
    inputs.remove(MAIN_ID)
    inputs.insert(0, MAIN_ID)

    def build(clips: Sequence[VideoClip]) -> VideoClip:
        statement = write({**values, **dict(zip(inputs, clips))})
        assert isinstance(statement, VideoClip)
        return statement

    images = [values[i] for i in inputs]
    written = build(
        [
            image_clip(Path(), context.node_id, index, method.frame_route)
            for index in range(len(images))
        ]
    )
    return filter_image(
        context,
        cast("list[np.ndarray]", images),
        build,
        method.frame_route,
        method.family == GEOMETRY_FAMILY or bool(SOURCE_WINDOW.search(written.stmt)),
    )


def named_clip_inputs(
    clip_help: str, by_method: bool = False
) -> list[BaseInput | NestedGroup]:
    """Clip 1 and its name, then clips 2 to NAMED_CLIPS with theirs, which the user adds
    a pair at a time; `clip_help` is each clip's pop-up, `by_method` as named_clips'.
    New inputs each call:
    registering a node numbers its inputs, so pairs added later take new ids and the
    earlier ones keep theirs (saved chains)."""

    def pair(number: int) -> tuple[BaseInput, BaseInput]:
        clip = VideoClipInput(f"Clip {number}").with_docs(clip_help, hint=True)
        empty = (
            f"the method's name, such as Denoise, strong (BM3D); Clip {number} if two"
            " clips share it"
            if by_method
            else f"Clip {number}"
        )
        name = TextInput(
            f"Name {number}",
            placeholder="Method's name" if by_method else f"Clip {number}",
        ).with_docs(
            f"The clip's name in vsview; empty: {empty}. Example: Filtered", hint=True
        )
        return (clip if number == 1 else clip.make_optional()), name.make_optional()

    return [
        *pair(1),
        optional_list_group(
            # A group that always shows: one item of the list, a clip with its name.
            *[
                if_group(Condition.const(True))(*pair(number))
                for number in range(2, NAMED_CLIPS + 1)
            ]
        ),
    ]


def named_clips(
    values: Sequence[VideoClip | str | None], by_method: bool = False
) -> list[tuple[VideoClip, str]]:
    """The connected clips of named_clip_inputs' values, in order, each with its name,
    else (`by_method`) its method's label if no other unnamed clip has it, else
    "Clip <number>": the i-th becomes output i of the script."""
    pairs = [
        (number, clip, name)
        for number, (clip, name) in enumerate(
            zip(values[::2], values[1::2], strict=True), 1
        )
        if isinstance(clip, VideoClip)
    ]
    labels = [clip.label for _, clip, name in pairs if not name]
    named: list[tuple[VideoClip, str]] = []
    for number, clip, name in pairs:
        if isinstance(name, str) and name:
            named.append((clip, name))
        elif by_method and clip.label and labels.count(clip.label) == 1:
            named.append((clip, clip.label))
        else:
            named.append((clip, f"Clip {number}"))
    return named
