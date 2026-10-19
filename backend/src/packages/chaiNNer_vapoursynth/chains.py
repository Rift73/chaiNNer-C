"""What the VapourSynth I/O nodes share: each node's folder in chaiNNer's storage, the
chain's compiled script, and the named clips of Preview Clips and Save Script."""

from __future__ import annotations

import os
import re
import stat
import time
from pathlib import Path
from typing import Sequence

from sanic.log import logger

from api import MB, BaseInput, NestedGroup, NodeContext
from nodes.condition import Condition
from nodes.groups import if_group, optional_list_group
from nodes.impl.native_graph import graph
from nodes.impl.vapoursynth.script import VideoClip, compile_script
from nodes.properties.inputs import TextInput, VideoClipInput

from .settings import get_settings

# Preview Clips and Save Script take up to this many clips, each with a name.
NAMED_CLIPS = 16
# Frames to Clip's temporary files older than this go when the package loads.
OLD_FILE_HOURS = 24


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


def named_clip_inputs() -> list[BaseInput | NestedGroup]:
    """Clip 1 and its name, then clips 2 to NAMED_CLIPS with theirs, which the user adds
    a pair at a time. New inputs each call: registering a node numbers its inputs, so
    pairs added later take new ids and the earlier ones keep theirs (saved chains)."""

    def pair(number: int) -> tuple[BaseInput, BaseInput]:
        clip = VideoClipInput(f"Clip {number}")
        name = TextInput(f"Name {number}", placeholder=f"Clip {number}")
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
    values: Sequence[VideoClip | str | None],
) -> list[tuple[VideoClip, str]]:
    """The connected clips of named_clip_inputs' values, in order, each with its name
    (or "Clip <number>"): the i-th becomes output i of the script."""
    named: list[tuple[VideoClip, str]] = []
    pairs = zip(values[::2], values[1::2], strict=True)
    for number, (clip, name) in enumerate(pairs, 1):
        if isinstance(clip, VideoClip):
            named.append(
                (clip, name if isinstance(name, str) and name else f"Clip {number}")
            )
    return named
