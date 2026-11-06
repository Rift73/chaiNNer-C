"""Measures which VapourSynth functions look only at the current frame, for the table
backend/src/nodes/impl/vapoursynth/frame_methods.json: the methods whose Clip input also
takes an image. A function is frame-only when, on a clip of different frames, frame n's
output (pixels and properties) stays the same while every other frame changes, the
clip keeps its length, the function runs on a clip of one frame, and that frame makes an
image (chainner_vs.save_frame). One frame that gives what a clip's last frame gives, unlike
its other frames, is not frame-only. A function that gives the same result whatever green and
blue are reads red where vs-jetpack means luma, and runs on YUV. A method that switches
other frames on through a parameter is frame-only with that parameter off; through a
Mode, only if its default does something (else its Modes are all it has).

Run again for a new vs-jetpack or VapourSynth version. Three steps:

    python vs_frame_probe.py cases <catalog.json> <cases.json>   # any chaiNNer Python
    python vs_frame_probe.py measure <cases.json> <raw.jsonl>    # VapourSynth's Python
    python vs_frame_probe.py table <cases.json> <raw.jsonl> <frame_methods.json> <vs-jetpack> <VapourSynth>

`cases` writes the statement each family node writes, with its default inputs and
trivial values for what has no default. `measure` runs each in a child process, so that
a plugin's crash costs one case (run it where a crash leaves nothing behind: Linux,
with no GPU: a method with a CPU backend is measured on it, and one that needs a GPU to
run counts as clips only).
Only the standard library, NumPy and VapourSynth are imported outside `cases`."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

WIDTH = 128
HEIGHT = 96
# What refuses the size above: CCD's automatic scale is under 1 on it.
BIG_PICTURES = {"vsdenoise.ccd": (640, 480), "zsmooth.CCD": (640, 480)}
LENGTH = 7
# The frame compared: not the middle, where a clip's reverse keeps it.
TARGET = 2
CASE_SECONDS = 90
SOURCE_VAR = "n_src_0"
# What a second clip input gets: another picture of the same frames (the first clip again
# would make a function that compares them, or takes their difference, do nothing).
OTHER_VAR = "n_other_0"
# The formats an image can be; the integer ones count only for a function that refuses
# float.
FLOAT_FORMATS = ("RGBS", "GRAYS", "YUV444PS")
INTEGER_FORMATS = {"RGBS": "RGB48", "GRAYS": "GRAY16", "YUV444PS": "YUV444P16"}
# Parameters that can switch a function's temporal use off, with the value that does.
SWITCHES = {
    "tr": 0,
    "temporal_radius": 0,
    "temporal": "(0.0, 0)",
    "radius": 0,
    "d": 0,
    "tbsize": 1,
}
# A node whose point is the frames themselves: clips only, whatever one call shows.
CLIP_ONLY_FAMILIES = ("Cut and Splice",)
# A node where reading one plane is what the user asked for: a plane number means R, G or B.
PLANE_FAMILIES = ("Planes",)
# A node whose methods make masks: what would run on YUV and give YUV has masked each plane,
# which reads as colours; it runs on luma alone.
MASK_FAMILIES = ("Masks",)
# Results or second clips centred on 0, whose values below 0 an image cannot hold.
SIGNED_VALUES = (
    "atwt.ExtractFrequency",
    "atwt.ReplaceFrequency",
    "bm3d.RGB2OPP",
    "std.MakeDiff",
    "std.MakeFullDiff",
    "std.MergeDiff",
    "std.MergeFullDiff",
    "vstools.get_u",
    "vstools.get_v",
)
# A setting leaves the result unwritten on float samples (noise.Add's Poisson type), which its
# default call does not show: an image would come back as what memory held.
UNWRITTEN_ON_FLOAT = ("noise.Add",)
# Their plugins crash on images smaller than this, at a side (found on 5x3 and 1x1 images).
# vsrgtools' remove_grain and repair call the same zsmooth functions.
TINY_CRASHES = (
    "vsrgtools.median_blur",
    "vsrgtools.remove_grain",
    "vsrgtools.repair",
    "zsmooth.InterQuartileMean",
    "zsmooth.Median",
    "zsmooth.RemoveGrain",
    "zsmooth.Repair",
    "zsmooth.SmartMedian",
)
MIN_SIDE = 4
# What only ever names other frames: a method that already looks at one frame by default
# gets it set too, so that a value a user typed means the same on every such method.
TEMPORAL_ONLY = ("tr", "temporal_radius")
# The choices of a `backend` input that run on the CPU.
CPU_BACKENDS = ("CPU", "OLD", "ISPC")
# Errors that say the function wanted a GPU, not that it refused the clip.
GPU_ERRORS = ("device memory", "cuInit", "pyopencl", "Vulkan", "vulkan")
# Trivial values for inputs without a default, by parameter name, then by kind.
VALUES_BY_NAME: dict[str, object] = {
    "matrix": "[1, 2, 1, 2, 4, 2, 1, 2, 1]",
    # More than one, or it looks at no other frame (AverageFrames).
    "weights": "[1, 1, 1]",
    "width": WIDTH // 2,
    "height": HEIGHT // 2,
    "expr": "x",
    "shifts_or_top": "(0.25, 0.25)",
    "format": "vs.RGBS",
    "colorfamily": "vs.GRAY",
    "planes": "0",
    # Equal, they divide by 0.
    "thr_lo": 0.2,
    "thr_hi": 0.8,
    # DCTFilter reads eight, however many it gets.
    "factors": "[1, 1, 1, 1, 0, 0, 0, 0]",
}
# Functions that only make a picture bigger, for the width and height above.
UPSCALES = tuple(
    f"descale.{kernel}"
    for kernel in (
        "Bicubic",
        "Bilinear",
        "Lanczos",
        "Point",
        "Spline16",
        "Spline36",
        "Spline64",
    )
)
VALUES_BY_KIND: dict[str, object] = {
    "int": 1,
    "float": 1.0,
    "bool": False,
    "text": "x",
    "sequence": "1",
    "extra": "1",
    "ranges": "(0, 1)",
    "fraction": "24/1",
    "format": "vs.RGBS",
    "colorfamily": "vs.GRAY",
}


# Step 1: the statements, through the code that writes them in a run.


def _record(var: str) -> Any:
    from nodes.impl.vapoursynth.script import VideoClip

    return VideoClip(var, "", (), frozenset(), var)


def _fill(slot: Any, param: Any) -> object:
    """A value for an input that has none, or None when no trivial value exists."""
    from nodes.impl.vapoursynth.script import VSFilter

    kind = slot.adapter.kind
    if kind in ("clip", "optional_clip"):
        return _record(OTHER_VAR)
    if kind == "clip_or_filter":
        if "VideoNode" in param.annotation and "Callable" not in param.annotation:
            return _record(OTHER_VAR)
        return VSFilter("(lambda clip: clip)", (), frozenset(), "filter")
    if slot.adapter.choices:
        return slot.adapter.choices[0][1]
    return VALUES_BY_NAME.get(param.name, VALUES_BY_KIND.get(kind))


def _written(statement: Any) -> tuple[str, list[str], bool]:
    """The statement's text with its result named, its imports, whether it gives several
    clips; the clips it takes all are the source."""
    imports = {*statement.imports}
    return statement.stmt, sorted(imports), statement.many


def _family_cases(family: str, methods: list[Any], enums: Any) -> Iterable[dict]:
    from nodes.impl.vapoursynth.schema_gen import (
        FILTER_ID,
        MAIN_ID,
        METHOD_ID,
        MODE_ID,
        _Family,
        _flat,
        _Run,
        _statement,
    )

    builder = _Family(family, methods, enums)
    builder.build()
    inputs = _flat(builder.inputs())
    defaults = {i.id: i.to_dict().get("def") for i in inputs}
    for method in builder.methods:
        plan = builder.plans[method.id]
        for member in method.members or (method,):
            entry = plan.members[member.choice]
            values: dict[int, object] = {
                **defaults,
                MAIN_ID: _record(SOURCE_VAR),
                METHOD_ID: method.id,
                MODE_ID: "simple",
                FILTER_ID: False,
            }
            if plan.class_id is not None:
                values[plan.class_id] = member.choice
            params = (*member.init_params, *member.params)
            by_name = {}
            for param in params:
                if param is entry.main:
                    continue
                init = any(param is p for p in member.init_params)
                by_name[param.name] = entry.slots[init, param.name]
                slot = by_name[param.name]
                variadic = param.kind in ("var_keyword", "var_positional")
                # A required number starts at 0, which no function takes as a size.
                if not values.get(slot.input_id) and param.default is None:
                    values[slot.input_id] = None if variadic else _fill(slot, param)
            for at, more in enumerate(plan.more_clips):
                # A second clip: Stack and the like show what they do with two.
                values[more] = _record(OTHER_VAR) if at == 0 else None
            variants: list[tuple[dict[str, object], dict[int, object]]] = [({}, values)]
            for name, value in SWITCHES.items():
                if name in by_name:
                    slot = by_name[name]
                    try:
                        held = slot.adapter.start_value(value)
                    except ValueError:
                        continue
                    variants.append(({name: value}, {**values, slot.input_id: held}))
            # A Mode that uses other frames decides too (Clense's default does nothing).
            modes: list[tuple[str, dict[int, object]]] = []
            if "mode" in by_name:
                modes = [
                    (source, {**values, by_name["mode"].input_id: source})
                    for _, source in by_name["mode"].adapter.choices
                ]
            backends = [("", None)]
            if "backend" in by_name:
                backends += [
                    (source, by_name["backend"].input_id)
                    for _, source in by_name["backend"].adapter.choices
                    if source.rpartition(".")[2] in CPU_BACKENDS
                ]
            for (switched, base), (backend, backend_id) in (
                (v, b) for v in variants for b in backends
            ):
                chosen = base if backend_id is None else {**base, backend_id: backend}
                enforced: dict[Any, object] = {
                    i.id: _enforced(i, chosen.get(i.id)) for i in inputs if i.id != 0
                }
                enforced[MAIN_ID] = _record(SOURCE_VAR)
                case = {
                    "id": member.id,
                    "method": method.id,
                    "family": family,
                    "switch": switched,
                    "backend": backend,
                    "size": BIG_PICTURES.get(member.id, (WIDTH, HEIGHT)),
                }
                run = _Run(family, enforced)
                try:
                    stmt, imports, many = _written(
                        _statement(plan, run, "probe", False)
                    )
                except ValueError as error:
                    yield {**case, "unwritable": str(error)}
                    continue
                yield {**case, "stmt": stmt, "imports": imports, "many": many}
            for mode, chosen in modes:
                enforced: dict[Any, object] = {
                    i.id: _enforced(i, chosen.get(i.id)) for i in inputs if i.id != 0
                }
                enforced[MAIN_ID] = _record(SOURCE_VAR)
                case = {
                    "id": member.id,
                    "method": method.id,
                    "family": family,
                    "switch": {},
                    "backend": "",
                    "mode": mode,
                    "size": BIG_PICTURES.get(member.id, (WIDTH, HEIGHT)),
                }
                try:
                    stmt, imports, many = _written(
                        _statement(plan, _Run(family, enforced), "probe", False)
                    )
                except ValueError as error:
                    yield {**case, "unwritable": str(error)}
                    continue
                yield {**case, "stmt": stmt, "imports": imports, "many": many}


def _enforced(item: Any, value: object) -> object:
    from nodes.impl.vapoursynth.script import VideoClip, VSFilter

    if value is None or isinstance(value, (VideoClip, VSFilter)):
        return value
    return item.enforce_(value)


def _any_cases(methods: list[Any], enums: Any) -> Iterable[dict]:
    """Any Function's: without parameter switches, whose values it could not set."""
    from nodes.impl.vapoursynth.schema_gen import (
        ANY_ROW_IDS,
        ANY_SOCKET_IDS,
        ANY_SOCKET_NAME_IDS,
        MAIN_ID,
        _any_methods,
        _any_statement,
        _member_params,
    )

    for method in _any_methods(methods):
        values: dict[int, object] = {MAIN_ID: _record(SOURCE_VAR)}
        rows = iter(ANY_ROW_IDS)
        sockets = iter(zip(ANY_SOCKET_IDS, ANY_SOCKET_NAME_IDS, strict=True))
        main, _, params = _member_params(method, enums, True)
        case = {
            "id": method.id,
            "method": method.id,
            "family": "Any Function",
            "switch": {},
            "backend": "",
            "size": BIG_PICTURES.get(method.id, (WIDTH, HEIGHT)),
        }
        problem = "" if main is not None else "it takes no clip"
        for _init, param, adapter in params:
            if param.default is not None or param.kind in (
                "var_keyword",
                "var_positional",
            ):
                continue
            slot = SimpleNamespace(adapter=adapter)
            value = _fill(slot, param)
            if method.id in UPSCALES and param.name in ("width", "height"):
                value = {"width": WIDTH, "height": HEIGHT}[param.name] * 2
            if adapter.kind in ("clip", "optional_clip", "clip_or_filter"):
                socket = next(sockets, None)
                if socket is None:
                    problem = "more clips than sockets"
                    break
                values[socket[0]], values[socket[1]] = value, param.name
            elif value is None:
                problem = f"no trivial value for {param.name}"
                break
            elif adapter.kind == "sequence":
                values[next(rows)] = f"{param.name} = {value}"
            else:
                values[next(rows)] = f"{param.name} = {value!r}"
        if problem:
            yield {**case, "unwritable": problem}
            continue
        try:
            stmt, imports, many = _written(_any_statement(method, values, "probe"))
        except (ValueError, StopIteration) as error:
            yield {**case, "unwritable": str(error)}
            continue
        yield {**case, "stmt": stmt, "imports": imports, "many": many}


def write_cases(catalog_path: Path, out: Path) -> None:
    from nodes.impl.vapoursynth.catalog import load_catalog, overlay_paths
    from nodes.impl.vapoursynth.schema_gen import ANY_FUNCTION, EXCLUDED, LOAD_CLIP

    catalog = load_catalog(catalog_path, overlay_paths(catalog_path.parent))
    by_family: dict[str, list[Any]] = {}
    for method in catalog.methods:
        if method.id not in EXCLUDED and method.family != LOAD_CLIP:
            by_family.setdefault(method.family, []).append(method)
    cases: list[dict] = []
    for family, methods in by_family.items():
        if family == ANY_FUNCTION:
            cases.extend(_any_cases(methods, catalog.enums))
        else:
            cases.extend(_family_cases(family, methods, catalog.enums))
    out.write_text(json.dumps(cases, indent=0), encoding="utf-8")
    print(f"{len(cases)} cases, {sum('unwritable' in c for c in cases)} unwritable")


# Step 2: the measurement. Nothing here imports chaiNNer.


def _pictures(seed: int, other_picture: bool, size: tuple[int, int]) -> Any:
    """Three float32 planes: a picture with detail, hard edges at other places in each plane
    (a mask with thresholds stays blank on smooth detail alone), and noise of this seed."""
    import numpy as np

    width, height = size
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((3, height, width), dtype=np.float32) * 0.01
    if other_picture:
        x, y = width - x, y * 2
    planes = [
        0.5 + 0.2 * np.sin(x / 7) * np.cos(y / 5),
        0.4 + 0.3 * (x / width) * np.sin(y / 3),
        0.6 - 0.3 * (y / height) * np.cos(x / 9),
    ]
    edges = [
        0.2 * (((x // size + y // (size + 3)).astype(np.int32) % 2) * 2 - 1)
        for size in (11, 17, 23)
    ]
    return np.clip(np.stack(planes) + np.stack(edges) + noise, 0, 1).astype(np.float32)


def _source(frames: list[Any], fmt: str, vs: Any) -> Any:
    """The frames as a clip of `fmt`, tagged as an image's clip is."""
    import numpy as np

    core = vs.core
    height, width = frames[0].shape[1:]
    blank = core.std.BlankClip(
        format=vs.RGBS, width=width, height=height, length=len(frames), keep=True
    )

    def fill(n: int, f: Any) -> Any:
        out = f.copy()
        for plane in range(3):
            np.copyto(np.asarray(out[plane]), frames[n][plane])
        return out

    clip = core.std.ModifyFrame(blank, blank, fill)
    clip = clip.std.SetFrameProps(
        _Transfer=13, _Primaries=1, _Range=1, _Matrix=0, _FieldBased=0
    )
    if fmt in ("GRAYS", "GRAY16"):
        clip = clip.std.ShufflePlanes(0, vs.GRAY).std.RemoveFrameProps("_Matrix")
        return clip if fmt == "GRAYS" else clip.resize.Point(format=vs.GRAY16)
    if fmt.startswith("YUV"):
        return clip.resize.Point(format=getattr(vs, fmt), matrix=1)
    return clip if fmt == "RGBS" else clip.resize.Point(format=getattr(vs, fmt))


def _props(frame: Any, vs: Any) -> dict[str, str]:
    return {
        key: repr(value)
        for key, value in frame.props.items()
        if not isinstance(value, (vs.VideoFrame, vs.VideoNode, vs.AudioNode))
    }


def _same_pixels(a: Any, b: Any) -> bool:
    import numpy as np

    if a.format.id != b.format.id or (a.width, a.height) != (b.width, b.height):
        return False
    return all(
        np.array_equal(
            np.asarray(a[p]), np.asarray(b[p]), equal_nan=a.format.sample_type == 1
        )
        for p in range(a.format.num_planes)
    )


def _changed_planes(out: Any, source: Any) -> list[int] | None:
    """Which planes differ from the source's frame; None for another format or size."""
    import numpy as np

    if out.format.id != source.format.id or out.width != source.width:
        return None
    if out.height != source.height:
        return None
    return [
        p
        for p in range(out.format.num_planes)
        if not np.array_equal(np.asarray(out[p]), np.asarray(source[p]))
    ]


def _other_planes(frames: list[Any], others: list[Any], *planes: int) -> list[Any]:
    """The frames with these planes of `others`."""
    mixed = [frame.copy() for frame in frames]
    for frame, other in zip(mixed, others, strict=True):
        for plane in planes:
            frame[plane] = other[plane]
    return mixed


def _as_image(clip: Any) -> dict[str, Any]:
    """What the image route makes of the clip's only frame."""
    import numpy as np

    from nodes.impl.vapoursynth import chainner_vs

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "image.npy"
        try:
            chainner_vs.save_frame(clip, str(path))
        except chainner_vs.NotFiniteError:
            return {"finite": False}
        image = np.load(path, allow_pickle=False)
    return {"image": list(image.shape), "finite": True}


def _measure_format(case: dict, fmt: str, vs: Any, namespace: dict) -> dict:
    """The function on `fmt`: "error", or whether frame TARGET depends on other frames."""
    size = case.get("size", (WIDTH, HEIGHT))
    frames = [_pictures(i, False, size) for i in range(LENGTH)]
    similar = [_pictures(100 + i, False, size) for i in range(LENGTH)]
    different = [_pictures(200 + i, True, size) for i in range(LENGTH)]
    similar[TARGET] = different[TARGET] = frames[TARGET]

    def apply(source: Any) -> Any:
        scope = {
            **namespace,
            SOURCE_VAR: source,
            OTHER_VAR: source.std.Expr("x 0.9 *"),
        }
        exec(case["stmt"], scope)
        out = scope[case["stmt"].split(" = ", 1)[0]]
        if not isinstance(out, vs.VideoNode):
            raise TypeError(f"it gives {type(out).__name__}, not a clip")
        return out

    source = _source(frames, fmt, vs)
    out = apply(source)
    result: dict[str, Any] = {
        "out": f"{out.format.name} {out.width}x{out.height}",
        "length": out.num_frames,
    }
    if out.num_frames != LENGTH:
        return {**result, "kind": "length"}
    frame = out.get_frame(TARGET)
    result["planes"] = _changed_planes(frame, source.get_frame(TARGET))
    # From a new source: a plugin may read memory that is not its input.
    again = apply(_source(frames, fmt, vs)).get_frame(TARGET)
    if not _same_pixels(frame, again):
        return {**result, "kind": "random"}
    for name, others in (("similar", similar), ("different", different)):
        other = apply(_source(others, fmt, vs)).get_frame(TARGET)
        if not _same_pixels(frame, other):
            return {**result, "kind": "temporal", "by": f"{name} frames: pixels"}
        if _props(frame, vs) != _props(other, vs):
            return {**result, "kind": "temporal", "by": f"{name} frames: properties"}
    if fmt.startswith("RGB"):
        # Other pictures in every frame, the one compared too.
        pictures = [_pictures(300 + i, True, size) for i in range(LENGTH)]
        result["ignores_gb"] = _same_pixels(
            frame,
            apply(_source(_other_planes(frames, pictures, 1, 2), fmt, vs)).get_frame(
                TARGET
            ),
        )
        result["ignores_r"] = _same_pixels(
            frame,
            apply(_source(_other_planes(frames, pictures, 0), fmt, vs)).get_frame(
                TARGET
            ),
        )
    single = _source([frames[TARGET]], fmt, vs)
    try:
        alone = apply(single)
        if alone.num_frames != 1:
            return {**result, "kind": "length", "by": "on one frame"}
        result["alone_equal"] = _same_pixels(frame, alone.get_frame(0))
    except vs.Error as error:
        return {**result, "kind": "needs_frames", "by": str(error)[:200]}
    except Exception as error:
        return {**result, "kind": "needs_frames", "by": repr(error)[:200]}
    if not result["alone_equal"]:
        # One frame is a clip's last frame. Where that one alone gives another result, the
        # others look at the frame after them (vsaa's SangNom weaves a frame with the next
        # one's field). Grain changes with the frame number or the length: not that.
        copies = apply(_source([frames[TARGET]] * LENGTH, fmt, vs))
        last = copies.get_frame(LENGTH - 1)
        if _same_pixels(alone.get_frame(0), last) and not _same_pixels(
            copies.get_frame(TARGET), last
        ):
            by = "one frame gives another result than the same frame among others"
            return {**result, "kind": "needs_frames", "by": by}
    return {**result, **_as_image(alone), "kind": "frame"}


def measure_case(case: dict, vs: Any) -> dict:
    """The function's result on every format an image can become; a function that
    refuses a float format is tried on its 16-bit one."""
    # A statement may call chaiNNer's helpers, and the image route does.
    backend = str(Path(__file__).resolve().parents[2] / "backend" / "src")
    if backend not in sys.path:
        sys.path.append(backend)
    namespace: dict[str, Any] = {"vs": vs, "core": vs.core}
    for line in case["imports"]:
        exec(line, namespace)
    results: dict[str, dict] = {}
    for fmt in FLOAT_FORMATS:
        for tried in (fmt, INTEGER_FORMATS[fmt]):
            try:
                results[tried] = _measure_format(case, tried, vs, namespace)
                break
            except Exception as error:
                results[tried] = {"kind": "error", "by": repr(error)[:300]}
    return results


def _child(cases_path: str, events_path: str) -> None:
    import vapoursynth as vs

    vs.core.num_threads = 4
    vs.core.max_cache_size = 1024
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    with open(events_path, "a", encoding="utf-8") as events:
        for index, case in enumerate(cases):
            events.write(json.dumps({"start": index}) + "\n")
            events.flush()
            record = {"index": index, "results": measure_case(case, vs)}
            events.write(json.dumps(record) + "\n")
            events.flush()


def _run_batch(pending: list[tuple[int, dict]], work: Path, out: Any) -> set[int]:
    """Runs the cases in one child and writes each record as it comes; a case it crashes
    or hangs on gets a record that says so, and ends the batch. The cases' indices done."""
    batch = work / "batch.json"
    events_path = work / "events.jsonl"
    batch.write_text(json.dumps([case for _, case in pending]), encoding="utf-8")
    events_path.write_text("", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-B", __file__, "child", str(batch), str(events_path)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    done: set[int] = set()

    def write(position: int, results: dict) -> None:
        index = pending[position][0]
        out.write(json.dumps({"index": index, "results": results}) + "\n")
        out.flush()
        done.add(index)

    offset = 0
    current: int | None = None
    started = time.monotonic()
    while True:
        code = process.poll()
        with open(events_path, "rb") as file:
            file.seek(offset)
            data = file.read()
        end = data.rfind(b"\n") + 1
        offset += end
        for line in data[:end].decode("utf-8").splitlines():
            event = json.loads(line)
            if "start" in event:
                current, started = event["start"], time.monotonic()
            else:
                write(event["index"], event["results"])
                current = None
        if code is not None:
            if current is not None:
                write(current, {"-": {"kind": "crash", "by": f"exit code {code}"}})
            elif code != 0:
                raise RuntimeError(f"a child ended with {code} before its first case")
            return done
        if current is not None and time.monotonic() - started > CASE_SECONDS:
            process.kill()
            process.wait()
            write(current, {"-": {"kind": "crash", "by": f"over {CASE_SECONDS} s"}})
            return done
        time.sleep(0.05)


def measure_all(cases_path: Path, raw: Path) -> None:
    """Appends one JSON line per case to `raw`, skipping those it already has: a job
    that is stopped goes on where it was when run again."""
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    finished: set[int] = set()
    if raw.exists():
        finished = {
            json.loads(line)["index"]
            for line in raw.read_text(encoding="utf-8").splitlines()
        }
    pending = [
        (i, c)
        for i, c in enumerate(cases)
        if i not in finished and "stmt" in c and not c.get("many")
    ]
    print(f"{len(pending)} of {len(cases)} cases to run", flush=True)
    with open(raw, "a", encoding="utf-8") as out:
        while pending:
            finished = _run_batch(pending, raw.parent, out)
            pending = [p for p in pending if p[0] not in finished]
            print(f"{len(pending)} left", flush=True)


# Step 3: the table.


def _reads_only_red(rgb: dict) -> bool:
    """The function changes only the first plane, or its result follows red alone."""
    return rgb.get("planes") == [0] or (
        rgb.get("ignores_gb", False) and not rgb.get("ignores_r", True)
    )


def verdict(results: dict[str, dict], family: str = "") -> tuple[str, str]:
    """A case's kind over its formats, and its route: "rgb" runs an RGB image as RGBS;
    "yuv" as YUV444PS (the function refuses RGB, or reads only the first plane, which is
    red there, where vs-jetpack means luma; in the Planes node that is what was asked)."""
    ran = {f: r for f, r in results.items() if r["kind"] not in ("error", "crash")}
    if not ran:
        texts = " ".join(r.get("by", "") for r in results.values())
        if any(r["kind"] == "crash" for r in results.values()):
            return "crash", ""
        return ("gpu" if any(e in texts for e in GPU_ERRORS) else "fails"), ""
    for kind in ("temporal", "length", "needs_frames", "random"):
        if any(r["kind"] == kind for r in ran.values()):
            return kind, ""
    rgb = ran.get("RGBS")
    if rgb is None and "YUV444PS" not in ran:
        # Integer formats only, or gray only: an image cannot go in as it is.
        return "fails", ""
    if rgb is not None and family in PLANE_FAMILIES:
        return "frame", "rgb"
    route = "yuv" if rgb is None or _reads_only_red(rgb) else "rgb"
    if route == "yuv" and "YUV444PS" not in ran:
        return "fails", ""
    if (
        route == "yuv"
        and family in MASK_FAMILIES
        and ran["YUV444PS"].get("out", "").startswith("YUV")
    ):
        return "frame", "luma"
    return "frame", route


def _member_verdict(variants: list[dict]) -> tuple[str, str, dict[str, object]]:
    """A member's kind, route and the arguments an image's run sets: the default call
    (on a CPU backend when it needs one); a temporal one that a parameter switches off
    is frame-only with that parameter."""

    def best(switch: dict[str, object]) -> tuple[str, str]:
        found = [v for v in variants if v["switch"] == switch and not v["mode"]]
        ran = [v for v in found if v["kind"] not in ("gpu", "crash", "fails")]
        pick = (ran or found)[0]
        return pick["kind"], pick["route"]

    kind, route = best({})
    # A method whose default does nothing has only its Modes to go by: Clense, whose Modes
    # are temporal. A blur with a temporal Mode among spatial ones (Median Blur) stays.
    idle = all(v["idle"] for v in variants if not v["mode"] and not v["switch"])
    if idle and any(v["mode"] and v["kind"] == "temporal" for v in variants):
        return "temporal", "", {}
    if kind == "temporal":
        for switch in (v["switch"] for v in variants if v["switch"] and not v["mode"]):
            other, other_route = best(switch)
            if other == "frame":
                return other, other_route, switch
    if kind == "frame":
        for switch in (v["switch"] for v in variants if v["switch"] and not v["mode"]):
            if next(iter(switch)) in TEMPORAL_ONLY and best(switch)[0] == "frame":
                return kind, route, switch
    return kind, route, {}


def write_table(
    cases_path: Path, raw: Path, out: Path, vsjetpack: str, vapoursynth: str
) -> None:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    records = {
        r["index"]: r["results"]
        for r in map(json.loads, raw.read_text(encoding="utf-8").splitlines())
    }
    by_method: dict[str, dict[str, list[dict]]] = {}
    families: dict[str, str] = {}
    for index, case in enumerate(cases):
        families[case["method"]] = case["family"]
        members = by_method.setdefault(case["method"], {})
        if case.get("many"):
            found = ("many", "")
        elif index in records:
            found = verdict(records[index], case["family"])
        else:
            found = ("unwritable", "")
        members.setdefault(case["id"], []).append(
            {
                "switch": case["switch"],
                "mode": case.get("mode", ""),
                "kind": found[0],
                "route": found[1],
                "idle": all(
                    r.get("planes") == []
                    for r in records.get(index, {}).values()
                    if r["kind"] == "frame"
                ),
            }
        )
    frame: dict[str, dict[str, Any]] = {}
    clip_only: dict[str, str] = {}
    for method, members in sorted(by_method.items()):
        if families[method] in CLIP_ONLY_FAMILIES:
            clip_only[method] = "cuts"
            continue
        if method in SIGNED_VALUES:
            clip_only[method] = "signed"
            continue
        if method in UNWRITTEN_ON_FLOAT:
            clip_only[method] = "unwritten"
            continue
        verdicts = [_member_verdict(variants) for variants in members.values()]
        kinds = [kind for kind, _, _ in verdicts]
        # A group's members that cannot run here do not decide; one that looks at other
        # frames does.
        blocking = next(
            (
                k
                for k in ("temporal", "length", "needs_frames", "random", "many")
                if k in kinds
            ),
            None,
        )
        if blocking is not None:
            clip_only[method] = blocking
        elif "frame" in kinds:
            entry: dict[str, Any] = {}
            routes = {route for _, route, _ in verdicts}
            if "luma" in routes:
                entry["route"] = "luma"
            elif "yuv" in routes:
                entry["route"] = "yuv"
            args = {k: v for _, _, a in verdicts for k, v in a.items()}
            if args:
                entry["args"] = args
            if method in TINY_CRASHES:
                entry["min_side"] = MIN_SIDE
            frame[method] = entry
        else:
            clip_only[method] = kinds[0]
    table = {
        "vsjetpack": vsjetpack,
        "vapoursynth": vapoursynth,
        "frame": frame,
        "clip_only": clip_only,
    }
    text = json.dumps(table, indent=1, sort_keys=True)
    out.write_text(text + "\n", encoding="utf-8", newline="\n")
    print(f"{len(frame)} frame-only, {len(clip_only)} clips only")


def main(argv: list[str]) -> None:
    step = argv[0]
    if step == "cases":
        write_cases(Path(argv[1]), Path(argv[2]))
    elif step == "measure":
        measure_all(Path(argv[1]), Path(argv[2]))
    elif step == "child":
        _child(argv[1], argv[2])
    elif step == "table":
        write_table(Path(argv[1]), Path(argv[2]), Path(argv[3]), argv[4], argv[5])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
