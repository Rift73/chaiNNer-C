"""Runs compiled VapourSynth scripts in child processes: clip info and frames in vspipe,
previews in vsview. The worker never imports VapourSynth; a plugin crash or a script
error costs only the child, and its traceback names the node whose statement failed."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import re
import signal
import subprocess
import sys
import sysconfig
import tempfile
import threading
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Callable, Generator, Iterator, TypeVar

import numpy as np

from api import NodeAttributedError, NodeContext

from ..colour_info import high_definition, pixel_layout
from ..native_framework_images import planes_to_interleaved
from .colour_props import PROPS, vs_pix_fmt
from .script import node_for_line

T = TypeVar("T")

# How many of vspipe's last stderr lines an error message keeps.
STDERR_TAIL_LINES = 20
# How often a wait for vspipe checks for Stop.
STOP_POLL_SECONDS = 0.05
# How long a new vsview (Linux) has to stay up before its window counts as opened.
VIEWER_GRACE_SECONDS = 3
# The environment variables with which vsview (Qt) finds a window system on Linux.
DISPLAY_VARIABLES = ("DISPLAY", "WAYLAND_DISPLAY", "QT_QPA_PLATFORM")
# The colour frame properties clip_props reads.
COLOUR_PROPS = tuple(prop for prop, _ in PROPS.values())

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


def _libpython() -> Path:
    """This Python's shared libpython, which vsscript loads to run a script."""
    name = sysconfig.get_config_var("INSTSONAME")
    folders = (
        Path(sys.base_prefix) / "lib",
        Path(sys.base_prefix) / "lib64",
        Path(sysconfig.get_config_var("LIBDIR") or "."),
    )
    for folder in folders:
        if name and (folder / name).is_file():
            return folder / name
    raise VSScriptError(
        "VapourSynth needs a Python with a shared libpython, and this one has none."
    )


@cache
def vsscript_config_home() -> str:
    # On Linux vsscript reads $XDG_CONFIG_HOME/vapoursynth/vapoursynth.toml, whatever
    # the Python: a line per libvsscript.so (by its real path) with the executable and
    # the libpython to load and the executable's modification time, which must all be
    # right. `vapoursynth config` writes the line, but cannot find the libpython of a
    # Python whose sysconfig names the machine that built it, and takes the path of a
    # library reached through a link as it is. So the line is written here, in a folder
    # inside the runtime (nothing goes to the user's config), and kept current for a
    # runtime that was moved or whose Python was replaced.
    library = os.path.realpath(vspipe().parent / "libvsscript.so")
    key = library.replace("/lib64/", "/lib/")
    mtime = int(os.stat(sys.executable).st_mtime)
    values = [sys.executable, str(_libpython()), str(mtime)]
    quoted = [json.dumps(text, ensure_ascii=False) for text in (key, *values)]
    line = f"{quoted[0]} = [{','.join(quoted[1:])}]\n"
    home = Path(sys.prefix) / "vsconfig"
    config = home / "vapoursynth" / "vapoursynth.toml"
    if not config.is_file() or config.read_text(encoding="utf-8") != line:
        config.parent.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(dir=config.parent, suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as file:
                file.write(line)
            os.replace(name, config)
        except BaseException:
            Path(name).unlink(missing_ok=True)
            raise
    return str(home)


def vs_env() -> dict[str, str]:
    """The environment of every VapourSynth child process."""
    env = _inherited_env()
    if sys.platform.startswith("linux"):
        env["XDG_CONFIG_HOME"] = vsscript_config_home()
    else:
        appdata = _venv_appdata()
        if appdata is not None:
            env["APPDATA"] = appdata
    return env


def script_error(
    script: Path, stderr: str, returncode: int | None = None
) -> VSScriptError:
    """The error of a failed run of `script`: the tail of vspipe's stderr, on the node
    of the traceback's last line in the script that is a node's statement; without
    stderr, vspipe's exit code (a plugin that crashes it leaves no message)."""
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
    if not message and returncode and returncode < 0:
        # POSIX: a child that a signal ended has the signal's number, negated, as code.
        signal_name = signal.strsignal(-returncode) or "unknown signal"
        return VSScriptError(
            f"vspipe was ended by signal {-returncode} ({signal_name}) and left no"
            " message (a plugin crashed).",
            node_id,
        )
    if not message and returncode:
        # Windows' crash codes read as hex: 0xC0000005 is an access violation.
        code = (
            f"0x{returncode & 0xFFFFFFFF:08X}"
            if not 0 < returncode < 256
            else returncode
        )
        message = f"vspipe ended with code {code} and no message (a plugin crashed)."
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
            text = stderr.read().decode("utf-8", "replace")
            raise script_error(script, text, returncode)
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


def _script_copy(script: Path, lines: str, suffix: str) -> Path:
    """The script with `lines` appended, in a new file next to it, so that it finds
    what the script finds there; the caller deletes it."""
    handle, name = tempfile.mkstemp(
        prefix=f"{script.stem}.", suffix=suffix, dir=script.parent
    )
    copy = Path(name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            file.write(script.read_text(encoding="utf-8") + lines)
    except BaseException:
        copy.unlink(missing_ok=True)
        raise
    return copy


def clip_props(
    script: Path, output: int, context: NodeContext | None = None
) -> tuple[ClipInfo, dict[str, int]]:
    """clip_info of a script's output and the colour frame properties (COLOUR_PROPS)
    of its first frame that are set, in one evaluation of the script. A first frame
    that fails gives none: the run of the frames fails on it and reports the error."""
    script = script.resolve()
    handle, name = tempfile.mkstemp(suffix=".json")
    os.close(handle)
    props = Path(name)
    lines = (
        "\n# chaiNNer reads the colour properties of the first frame.\n"
        "import json\n"
        "import vapoursynth\n"
        "_props = {}\n"
        f"if {output} in vapoursynth.get_outputs():\n"
        "    try:\n"
        f"        _props = vapoursynth.get_output({output}).clip.get_frame(0).props\n"
        "    except vapoursynth.Error:\n"
        "        # The run of the frames fails on this frame again, and reports it.\n"
        "        pass\n"
        f"with open({str(props)!r}, 'w', encoding='utf-8') as _file:\n"
        f"    json.dump({{k: int(_props[k]) for k in {COLOUR_PROPS!r} if k in _props}},"
        " _file)\n"
    )
    copy = _script_copy(script, lines, ".props.vpy")
    try:
        info = _video_output(copy, output, context)[0]
        return info, json.loads(props.read_text(encoding="utf-8"))
    finally:
        copy.unlink(missing_ok=True)
        props.unlink(missing_ok=True)


def _yuv_to_rgbs(info: ClipInfo) -> str:
    # The _Matrix frame property, when set, beats matrix_in. Else Load Video's guess
    # for untagged video: BT.709 for HD, else SMPTE 170M.
    matrix = 1 if high_definition(info.width, info.height) else 6
    return f"format=vapoursynth.RGBS, matrix_in={matrix}"


def _float_output(info: ClipInfo, fields: dict[str, str], output: int) -> str | None:
    """Script lines that turn the output into float32 RGB (RGBS) or gray (GRAYS) frames
    without alpha, or None when it already is such a clip."""
    family = fields["Color Family"]
    if family == "Gray":
        to = "format=vapoursynth.GRAYS"
    elif family == "RGB":
        to = "format=vapoursynth.RGBS"
    else:
        to = _yuv_to_rgbs(info)
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
    shape: tuple[int, ...],
    unpack: Callable[[np.ndarray], T],
    views: bool,
    context: NodeContext | None,
) -> Iterator[T]:
    """Frames start..end-1 of an output that vspipe writes as float32 arrays of `shape`,
    each made an item by `unpack`. Items that copy what they take from the array let
    the next frame reuse it; items that are `views` of it get a new array per frame."""
    command = [str(vspipe()), "-o", str(output), "-s", str(start), "-e", str(end - 1)]
    buffer: np.ndarray | None = None
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
                    if buffer is None or views:
                        buffer = np.empty(shape, np.float32)
                    if not _read_into(stdout, memoryview(buffer).cast("B")):
                        break
                    count += 1
                    yield unpack(buffer)
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
            text = stderr.read().decode("utf-8", "replace")
            raise script_error(script, text, returncode)


def _gray_frame(planes: np.ndarray) -> np.ndarray:
    return planes.reshape(*planes.shape[1:], 1)


def _rgb_frame(planes: np.ndarray) -> np.ndarray:
    # vspipe writes RGB planes as G, B, R (ffmpeg's gbrp order).
    return planes_to_interleaved((planes[1], planes[0], planes[2]))


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


def _check_range(info: ClipInfo, output: int, start: int, end: int) -> None:
    if not 0 <= start <= end <= info.num_frames:
        raise ValueError(
            f"Frames {start} to {end} are outside the {info.num_frames} frames of"
            f" output {output}."
        )


def _varying(output: int) -> VSScriptError:
    return VSScriptError(
        f"Output {output} changes size or format between frames. Add a Resize or"
        " Convert Format node so that every frame has the same size and format."
    )


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
    _check_range(info, output, start, end)
    if not info.width or not info.height or "Color Family" not in fields:
        raise _varying(output)
    if start == end:
        return
    gray = fields["Color Family"] == "Gray"
    # Gray frames are views of their own array (no copy); RGB frames are interleaved
    # copies.
    shape = (1 if gray else 3, info.height, info.width)
    unpack = _gray_frame if gray else _rgb_frame
    lines = _float_output(info, fields, output)
    if lines is None:
        yield from _vspipe_frames(
            script, output, start, end, shape, unpack, gray, context
        )
        return
    converted = _script_copy(script, lines, ".frames.vpy")
    try:
        yield from _vspipe_frames(
            converted, output, start, end, shape, unpack, gray, context
        )
    finally:
        converted.unlink(missing_ok=True)


def read_planes(
    script: Path,
    output: int,
    start: int,
    end: int,
    *,
    info: ClipInfo,
    rgb: bool,
    context: NodeContext | None = None,
) -> Generator[tuple[np.ndarray | None, np.ndarray, np.ndarray, np.ndarray]]:
    """Frames start..end-1 of a script's YUV output, rendered by vspipe, as their Y, U
    and V planes at their stored sizes: float32 arrays of the stored values (integer
    codes, or float samples, chroma centred on 0, as VapourSynth has them). With
    `rgb`, each comes with the frame as read_frames gives it (None without), from the
    same render. `info` is the output's clip_info from this same script."""
    script = script.resolve()
    _check_range(info, output, start, end)
    if not info.width or not info.height or info.format_name == "Variable":
        raise _varying(output)
    layout = pixel_layout(vs_pix_fmt(info.format_name))
    if layout.kind != "yuv":
        raise VSScriptError(f"Output {output} is {info.format_name}, not YUV.")
    if start == end:
        return
    width, height = info.width, info.height
    chroma_width = width >> layout.subsampling[0]
    chroma_height = height >> layout.subsampling[1]
    # One float32 frame of every plane stacked, padded to the luma's width: one render
    # of the chain gives both the planes and RGB. std.Expr keeps the integer codes.
    planes = (
        "[_clip.std.ShufflePlanes(planes=i, colorfamily=vapoursynth.GRAY)"
        ".std.Expr('x', format=vapoursynth.GRAYS) for i in range(3)]"
    )
    to_rgb = ""
    if rgb:
        # G, B and R (vspipe's order of RGB planes) of the clip read_frames reads.
        to_rgb = f"_rgb = _clip.resize.Bicubic({_yuv_to_rgbs(info)})\n"
        planes = (
            "[_rgb.std.ShufflePlanes(planes=i, colorfamily=vapoursynth.GRAY)"
            f" for i in (1, 2, 0)] + {planes}"
        )
    lines = (
        "\n# chaiNNer reads the planes' values as float32, stacked.\n"
        "import vapoursynth\n"
        f"_clip = vapoursynth.get_output({output}).clip\n"
        f"{to_rgb}"
        f"_planes = {planes}\n"
        "vapoursynth.core.std.StackVertical("
        "[p.std.AddBorders(right=_clip.width - p.width) for p in _planes]"
        f").set_output({output})\n"
    )
    rows = (3 * height if rgb else 0) + height + 2 * chroma_height

    def unpack(
        stack: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, np.ndarray]:
        top = 3 * height if rgb else 0
        frame = None
        if rgb:
            frame = _rgb_frame(stack[:top].reshape(3, height, width))
        y = stack[top : top + height].copy()
        top += height
        u = stack[top : top + chroma_height, :chroma_width].copy()
        top += chroma_height
        v = stack[top : top + chroma_height, :chroma_width].copy()
        return frame, y, u, v

    stacked = _script_copy(script, lines, ".frames.vpy")
    try:
        yield from _vspipe_frames(
            stacked, output, start, end, (rows, width), unpack, False, context
        )
    finally:
        stacked.unlink(missing_ok=True)


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
    on_linux = sys.platform.startswith("linux")
    # A QT_QPA_PLATFORM names the platform (offscreen, say) whatever the display.
    if on_linux and not any(os.environ.get(name) for name in DISPLAY_VARIABLES):
        raise VSScriptError(
            "vsview needs a graphical session: neither DISPLAY nor WAYLAND_DISPLAY is"
            " set."
        )
    # vsview's output goes to a log beside the script.
    log_path = script.parent / "vsview.log"
    with open(log_path, "wb") as log:
        viewer = subprocess.Popen(
            [str(vs_python()), "-m", "vsview", str(script)],
            cwd=script.parent,
            # It runs the script in its own process, which needs no vsscript
            # configuration: on Linux its Qt and its settings keep the user's config
            # folder.
            env=_inherited_env() if on_linux else vs_env(),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=_NO_WINDOW | _NEW_GROUP,
            start_new_session=True,
        )
    if on_linux:
        # Qt ends the process within a second or two when it cannot open a window (no
        # X11 or Wayland connection, libxcb-cursor0 missing).
        try:
            returncode = viewer.wait(VIEWER_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            return viewer
        raise VSScriptError(
            f"vsview ended right after it started (code {returncode}). Its log,"
            f" {log_path}, says why; on Linux Qt needs a graphical session and"
            " libxcb-cursor0."
        )
    return viewer


def launch_preview(script: Path, key: str) -> None:
    """Shows `script` in vsview, one window per key: while the key's window is open, it
    stays (the user reloads the rewritten script there); once closed, a new one opens."""
    with _viewers_lock:
        viewer = _viewers.get(key)
        if viewer is None or viewer.poll() is not None:
            _viewers[key] = _open_viewer(script)
