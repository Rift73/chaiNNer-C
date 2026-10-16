"""Runs compiled VapourSynth scripts in child processes: clip info and frames in vspipe,
previews in vsview. The worker never imports VapourSynth; a plugin crash or a script
error costs only the child, and its traceback names the node whose statement failed."""

from __future__ import annotations

import importlib.util
import io
import os
import re
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Generator, Iterator

import numpy as np

from api import NodeAttributedError, NodeContext

from .script import node_for_line

# How many of vspipe's last stderr lines an error message keeps.
STDERR_TAIL_LINES = 20
# How often a wait for vspipe checks for Stop.
STOP_POLL_SECONDS = 0.05

# R81 says on every run that it refuses API 3 plugins, and vs-jetpack's extras install
# one (bm3dcpu): the line says nothing about the script.
_API3_NOTICE = re.compile(
    r"^\w+: Plugin .* uses API 3, which is no longer supported\.$"
)
_TRACEBACK_FILE = re.compile(r'^\s*File "(?P<file>.+)", line (?P<line>\d+), in ')
_FPS = re.compile(r"^(\d+)/(\d+)")
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass(frozen=True)
class ClipInfo:
    width: int
    height: int
    num_frames: int
    # 0/1 for a variable frame rate; width and height are 0 when they vary.
    fps_num: int
    fps_den: int
    # VapourSynth's name, e.g. "YUV420P8", "RGBS", "Gray16" or "Variable".
    format_name: str


class VSScriptError(NodeAttributedError, RuntimeError):
    """A failed VapourSynth run. node_id is the node whose statement failed, when the
    traceback names one: the run reports the error on that node."""


def vs_python() -> Path:
    """The Python that runs VapourSynth's children (scan, vsview): this one."""
    return Path(sys.executable)


def vspipe() -> Path:
    """vspipe of this Python's VapourSynth, by absolute path: a bare `vspipe` can
    find another VapourSynth on PATH."""
    spec = importlib.util.find_spec("vapoursynth")
    if spec is None or not spec.submodule_search_locations:
        raise VSScriptError(
            "VapourSynth is not installed. Install the VapourSynth package in the"
            " dependency manager."
        )
    name = "vspipe.exe" if sys.platform == "win32" else "vspipe"
    return Path(spec.submodule_search_locations[0]) / name


def _inherited_env() -> dict[str, str]:
    env = dict(os.environ)
    # The only way a child would load another VapourSynth's plugins.
    env.pop("VAPOURSYNTH_EXTRA_PLUGIN_PATH", None)
    return env


@cache
def _venv_appdata() -> str | None:
    # vsscript reads its config from %APPDATA%\vapoursynth; a venv (development) needs
    # one, kept inside the venv so nothing goes to the user's Roaming folder. A
    # runtime's own Python is configured implicitly.
    if sys.prefix == sys.base_prefix:
        return None
    appdata = Path(sys.prefix) / "vsconfig"
    if not (appdata / "vapoursynth" / "vapoursynth.toml").is_file():
        subprocess.run(
            [str(vs_python()), "-I", "-B", "-m", "vapoursynth", "config"],
            env={**_inherited_env(), "APPDATA": str(appdata)},
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=True,
            creationflags=_NO_WINDOW,
        )
    return str(appdata)


def vs_env() -> dict[str, str]:
    """The environment of every VapourSynth child process."""
    env = _inherited_env()
    appdata = _venv_appdata()
    if appdata is not None:
        env["APPDATA"] = appdata
    return env


def script_error(script: Path, stderr: str) -> VSScriptError:
    """The error of a failed run of `script`: the tail of vspipe's stderr, on the node
    of the traceback's last line in the script that is a node's statement."""
    lines = [line for line in stderr.splitlines() if not _API3_NOTICE.match(line)]
    target = os.path.normcase(os.path.abspath(script))
    try:
        text = script.read_text(encoding="utf-8")
    except OSError:
        text = ""
    node_id = None
    for line in reversed(lines):
        match = _TRACEBACK_FILE.match(line)
        if match and os.path.normcase(os.path.abspath(match["file"])) == target:
            node_id = node_for_line(text, int(match["line"]))
            if node_id is not None:
                break
    message = "\n".join(lines[-STDERR_TAIL_LINES:]).strip()
    return VSScriptError(message or "vspipe failed without a message.", node_id)


def _run_vspipe(script: Path, args: list[str], context: NodeContext | None) -> str:
    # Files, not pipes: no output volume can block vspipe while Stop is polled (an
    # evaluation can index a long video for minutes; the host restarts a worker that
    # does not stop within seconds).
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        with subprocess.Popen(
            [str(vspipe()), *args],
            env=vs_env(),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            creationflags=_NO_WINDOW,
        ) as process:
            try:
                while True:
                    try:
                        returncode = process.wait(timeout=STOP_POLL_SECONDS)
                        break
                    except subprocess.TimeoutExpired:
                        if context is not None:
                            context.check_aborted()
            finally:
                if process.poll() is None:
                    process.kill()
        if returncode != 0:
            stderr.seek(0)
            raise script_error(script, stderr.read().decode("utf-8", "replace"))
        stdout.seek(0)
        return stdout.read().decode("utf-8", "replace")


def _video_output(
    script: Path, output: int, context: NodeContext | None
) -> tuple[ClipInfo, dict[str, str]]:
    """The clip info of an output and all of vspipe's fields for it."""
    # --info evaluates the script and lists every output, whatever --outputindex says.
    text = _run_vspipe(script, ["--info", str(script), "-"], context)
    for block in text.replace("\r\n", "\n").split("\n\n"):
        fields = dict(
            line.split(": ", 1) for line in block.splitlines() if ": " in line
        )
        if fields.get("Output Index") != str(output):
            continue
        if fields.get("Type") != "Video":
            raise VSScriptError(f"Output {output} of the script is not a video clip.")
        fps = _FPS.match(fields["FPS"])
        info = ClipInfo(
            width=int(fields["Width"]) if fields["Width"].isdigit() else 0,
            height=int(fields["Height"]) if fields["Height"].isdigit() else 0,
            num_frames=int(fields["Frames"]),
            fps_num=int(fps[1]) if fps else 0,
            fps_den=int(fps[2]) if fps else 1,
            format_name=fields["Format Name"],
        )
        return info, fields
    raise VSScriptError(f"The script has no output {output}.")


def clip_info(
    script: Path, output: int, context: NodeContext | None = None
) -> ClipInfo:
    """Size, length, frame rate and format of a script's output, from `vspipe --info`;
    Stop of the run of `context` ends vspipe."""
    return _video_output(script.resolve(), output, context)[0]


def _float_output(info: ClipInfo, fields: dict[str, str], output: int) -> str | None:
    """Script lines that turn the output into float32 RGB (RGBS) or gray (GRAYS) frames
    without alpha, or None when it already is such a clip."""
    family = fields["Color Family"]
    if family == "Gray":
        to = "format=vapoursynth.GRAYS"
    elif family == "RGB":
        to = "format=vapoursynth.RGBS"
    else:
        # The _Matrix frame property, when set, beats matrix_in. Else Load Video's
        # guess for untagged video: BT.709 for HD (width >= 1280 or height > 576),
        # else SMPTE 170M.
        matrix = 1 if info.width >= 1280 or info.height > 576 else 6
        to = f"format=vapoursynth.RGBS, matrix_in={matrix}"
    float32 = fields["Sample Type"] == "Float" and fields["Bits"] == "32"
    if float32 and family != "YUV":
        if fields["Alpha"] == "No":
            return None
        to = ""
    resize = f".resize.Bicubic({to})" if to else ""
    return (
        "\n# chaiNNer reads float32 RGB or gray frames.\n"
        "import vapoursynth\n"
        f"vapoursynth.get_output({output}).clip{resize}.set_output({output})\n"
    )


def _read_into(stream: io.BufferedReader, buffer: memoryview) -> bool:
    done = 0
    while done < len(buffer):
        count = stream.readinto(buffer[done:])
        if not count:
            return False
        done += count
    return True


def _kill_on_stop(
    process: subprocess.Popen[bytes], context: NodeContext, done: threading.Event
) -> None:
    # A read waits until vspipe has rendered the frame, which can take long (an AI
    # model in the script): Stop ends vspipe instead of waiting for it.
    while not done.wait(STOP_POLL_SECONDS):
        if context.aborted:
            process.kill()
            return


def _vspipe_frames(
    script: Path,
    output: int,
    start: int,
    end: int,
    shape: tuple[int, int, int],
    context: NodeContext | None,
) -> Iterator[np.ndarray]:
    channels, height, width = shape
    command = [str(vspipe()), "-o", str(output), "-s", str(start), "-e", str(end - 1)]
    planes: np.ndarray | None = None
    count = 0
    with tempfile.TemporaryFile() as stderr:
        with subprocess.Popen(
            [*command, str(script), "-"],
            env=vs_env(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=stderr,
            creationflags=_NO_WINDOW,
        ) as process:
            stdout = process.stdout
            assert isinstance(stdout, io.BufferedReader)
            done = threading.Event()
            if context is not None:
                threading.Thread(
                    target=_kill_on_stop,
                    args=(process, context, done),
                    name="vspipe stop",
                    daemon=True,
                ).start()
            try:
                while count < end - start:
                    # Gray frames are views of their own buffer (no copy); RGB frames
                    # are interleaved copies, so their planes' buffer is reused.
                    if planes is None or channels == 1:
                        planes = np.empty(shape, np.float32)
                    if not _read_into(stdout, memoryview(planes).cast("B")):
                        break
                    count += 1
                    if channels == 1:
                        yield planes.reshape(height, width, 1)
                    else:
                        # vspipe writes RGB planes as G, B, R (ffmpeg's gbrp order).
                        yield np.stack((planes[1], planes[0], planes[2]), axis=2)
                stdout.read()
                returncode = process.wait()
            finally:
                done.set()
                # The consumer stopped early, or reading failed.
                if process.poll() is None:
                    process.kill()
        if context is not None:
            context.check_aborted()
        if returncode != 0 or count < end - start:
            stderr.seek(0)
            raise script_error(script, stderr.read().decode("utf-8", "replace"))


def _info_fields(info: ClipInfo) -> dict[str, str]:
    """The --info fields read_frames uses, from a ClipInfo's format name ("RGBS",
    "YUV420P10", "GrayH": family prefix; a final S is float32, H float16). A ClipInfo
    does not say whether the output has an alpha clip, so it counts as having one: a
    float32 RGB or gray output is then set again without it, not read as it is."""
    name = info.format_name
    fields = {"Alpha": "Yes", "Sample Type": "Integer", "Bits": ""}
    family = next((f for f in ("Gray", "RGB", "YUV") if name.startswith(f)), None)
    if family is not None:
        fields["Color Family"] = family
    bits = {"S": "32", "H": "16"}.get(name[-1:])
    if bits is not None:
        fields["Sample Type"], fields["Bits"] = "Float", bits
    return fields


def read_frames(
    script: Path,
    output: int,
    start: int,
    end: int,
    *,
    info: ClipInfo | None = None,
    context: NodeContext | None = None,
) -> Generator[np.ndarray]:
    """Frames start..end-1 of a script's output, rendered by vspipe, as float32 HWC BGR
    arrays (HxWx1 for gray clips). Other formats are converted in the child: YUV with
    the _Matrix frame property, else BT.709 for HD sizes, else SMPTE 170M. `info`,
    the output's clip_info from this same script, saves an evaluation of the script.
    Stop of the run of `context` ends vspipe, also in the middle of a frame."""
    script = script.resolve()
    if info is None:
        info, fields = _video_output(script, output, context)
    else:
        fields = _info_fields(info)
    if not 0 <= start <= end <= info.num_frames:
        raise ValueError(
            f"Frames {start} to {end} are outside the {info.num_frames} frames of"
            f" output {output}."
        )
    if not info.width or not info.height or "Color Family" not in fields:
        raise VSScriptError(
            f"Output {output} changes size or format between frames. Add a Resize or"
            " Convert Format node so that every frame has the same size and format."
        )
    if start == end:
        return
    shape = (1 if fields["Color Family"] == "Gray" else 3, info.height, info.width)
    lines = _float_output(info, fields, output)
    if lines is None:
        yield from _vspipe_frames(script, output, start, end, shape, context)
        return
    # A copy next to the script, so that the script finds what it finds there.
    handle, name = tempfile.mkstemp(
        prefix=f"{script.stem}.", suffix=".frames.vpy", dir=script.parent
    )
    converted = Path(name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            file.write(script.read_text(encoding="utf-8") + lines)
        yield from _vspipe_frames(converted, output, start, end, shape, context)
    finally:
        converted.unlink(missing_ok=True)


# The vsview of each preview key (a Preview Clips node), kept while its window is open.
_viewers: dict[str, subprocess.Popen[bytes]] = {}
_viewers_lock = threading.Lock()
# vsview's own process group (session elsewhere): the window stays open after the run
# and gets none of the worker's console signals.
_NEW_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)


def _open_viewer(script: Path) -> subprocess.Popen[bytes]:
    if importlib.util.find_spec("vsview") is None:
        raise VSScriptError(
            "vsview is not installed. Install the vsview package in the dependency"
            " manager."
        )
    # vsview's output goes to a log beside the script.
    with open(script.parent / "vsview.log", "wb") as log:
        return subprocess.Popen(
            [str(vs_python()), "-m", "vsview", str(script)],
            cwd=script.parent,
            env=vs_env(),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=_NO_WINDOW | _NEW_GROUP,
            start_new_session=True,
        )


def launch_preview(script: Path, key: str) -> None:
    """Shows `script` in vsview, one window per key: while the key's window is open, it
    stays (the user reloads the rewritten script there); once closed, a new one opens."""
    with _viewers_lock:
        viewer = _viewers.get(key)
        if viewer is None or viewer.poll() is not None:
            _viewers[key] = _open_viewer(script)
