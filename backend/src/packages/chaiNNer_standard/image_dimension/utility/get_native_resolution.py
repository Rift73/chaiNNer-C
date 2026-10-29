from __future__ import annotations

import json
import threading
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np
from sanic.log import logger

import navi
from api import NodeContext
from nodes.groups import if_enum_group, required
from nodes.impl import getnative_media as media
from nodes.impl.ffmpeg import FFMpegEnv
from nodes.impl.native_descale import descale
from nodes.impl.native_getnative import (
    FrameFacts,
    SearchOptions,
    Session,
    linear_plane,
)
from nodes.properties.inputs import (
    BoolInput,
    DirectoryInput,
    EnumInput,
    ImageInput,
    NumberInput,
    TextInput,
    VideoFileInput,
)
from nodes.properties.outputs import ImageOutput, NumberOutput, TextOutput

from .. import utility_group


class NativeResolutionSource(Enum):
    IMAGE = "image"
    VIDEO = "video"


class NativeResolutionSampling(Enum):
    KEYFRAMES = "keyframes"
    EVEN = "even"
    SCENES = "scenes"
    TIMES = "manual"


class NativeResolutionKernels(Enum):
    ALL = 0
    BILINEAR = 1
    BICUBIC = 2
    LANCZOS = 3


# Descale's kernel numbers (DescaleKernel) by Vernier's family names.
DESCALE_KERNELS = {
    "bilinear": 1,
    "bicubic": 2,
    "lanczos": 3,
    "spline16": 4,
    "spline36": 5,
    "spline64": 6,
}
DESCALE_EXTEND = 2
LIGHT_LABELS = {
    "encoded": "As stored",
    "linear-gamma24": "Gamma 2.4",
    "linear-srgb": "sRGB",
    "linear-bt709": "BT.709 camera",
    "linear-gamma22": "Gamma 2.2",
}
STILL_ASSUMPTIONS = (
    "Full-range RGB code values from the still image; ICC profiles are not transformed."
    " The selected transfer function is a tested hypothesis."
)
PLOT_WIDTH, PLOT_HEIGHT = 1024, 256


def auto_bounds(height: int) -> tuple[int, int]:
    """Vernier's validated 500-1000 for 1080 lines, scaled to the source height."""
    return round(height * 500 / 1080), round(height * 1000 / 1080)


def search_bounds(height: int, min_height: int, max_height: int) -> tuple[int, int]:
    low, high = auto_bounds(height)
    low, high = min_height or low, max_height or high
    if low < 16 or low > high or high >= height:
        raise ValueError(
            f"The height search {low}-{high} does not fit a {height}-line source: it must"
            " start at 16 or more and end below the source height. Set Min Height and Max"
            " Height."
        )
    return low, high


def plot(sizes: np.ndarray, scores: np.ndarray, winner: int) -> np.ndarray:
    """The best score per native size as a log curve (dips are good fits), the carried
    size marked in red."""
    image = np.ones((PLOT_HEIGHT, PLOT_WIDTH, 3), np.float32)
    if sizes.size < 2:
        return image
    low, high = float(sizes[0]), float(sizes[-1])
    columns = low + np.arange(PLOT_WIDTH) * (high - low) / (PLOT_WIDTH - 1)
    values = np.interp(columns, sizes, np.log10(np.maximum(scores, 1e-300)))
    span = max(values.max() - values.min(), 1e-12)
    rows = np.round(8 + (values.max() - values) / span * (PLOT_HEIGHT - 17)).astype(int)
    marker = round((winner - low) / max(high - low, 1) * (PLOT_WIDTH - 1))
    if 0 <= marker < PLOT_WIDTH:
        image[:, marker] = (0.15, 0.15, 0.85)
    previous = rows[0]
    for x, row in enumerate(rows):
        image[min(row, previous) : max(row, previous) + 1, x] = (0.55, 0.3, 0.1)
        previous = row
    return image


def _watch(context: NodeContext, session: Session, done: threading.Event) -> None:
    while not done.wait(0.05):
        if context.aborted:
            session.cancel()
            return


@utility_group.register(
    schema_id="chainner:image:get_native_resolution",
    name="Get Native Resolution",
    description=[
        "Find the resolution an image or video was upscaled from, and the kernel it was upscaled with, for Descale.",
        "It fits every native size in the search range with each kernel and checks the best fits on both axes, on the whole frame and on withheld pixels. The verdict says how far the evidence goes: a supported candidate, an ambiguous kernel or grid, a search boundary, or inconclusive.",
        "For a video it analyzes up to Max Frames frames and lets them vote on the size; the frames on that size then decide the kernel together, which needs at least 7 of them favouring one preset consistently, else the smallest kernel in the ambiguity band is carried. Status names the rule that chose the kernel. The outputs carry the stored (As stored) analysis; the light models only add evidence to Status and Report.",
        "The Descaled output is the image through Descale with the result, Border set to Extend and Linear Light off (the outputs describe the stored values). Set Descale's Border to Extend and keep its Linear Light off to reproduce it.",
        "Based on [getnative](https://github.com/Infiziert90/getnative) (MIT) and its C++ successor Vernier 0.2, whose analysis core it runs.",
    ],
    icon="MdOutlinePhotoSizeSelectSmall",
    inputs=[
        EnumInput(
            NativeResolutionSource,
            "Source",
            default=NativeResolutionSource.IMAGE,
            preferred_style="tabs",
        ).with_id(0),
        if_enum_group(0, NativeResolutionSource.IMAGE)(
            required()(ImageInput().make_optional().with_id(1)),
        ),
        if_enum_group(0, NativeResolutionSource.VIDEO)(
            VideoFileInput().make_optional().with_id(2),
            EnumInput(
                NativeResolutionSampling,
                "Sampling",
                default=NativeResolutionSampling.KEYFRAMES,
                preferred_style="tabs",
            )
            .with_id(3)
            .with_docs(
                "Which frames are candidates. Keyframes: the codec's keyframes (the least"
                " compressed frames). Even: equal time steps. Scenes: the middle of each shot"
                " FFmpeg's scene detection finds. Times: the times you type."
            ),
            NumberInput("Max Frames", min=1, max=200, default=50)
            .with_id(4)
            .with_docs(
                "At most this many frames are analyzed, spread evenly over the candidates."
                " Fewer candidates give fewer frames."
            ),
            if_enum_group(3, NativeResolutionSampling.SCENES)(
                NumberInput(
                    "Scene Threshold",
                    precision=2,
                    min=0.01,
                    max=0.99,
                    default=0.3,
                    step=0.05,
                ).with_id(5),
                NumberInput(
                    "Min Shot", precision=2, min=0, max=60, default=0.5, unit="s"
                )
                .with_id(6)
                .with_docs("Shots shorter than this are skipped."),
            ),
            if_enum_group(3, NativeResolutionSampling.TIMES)(
                TextInput("Times", default="", allow_empty_string=True)
                .with_id(7)
                .with_docs(
                    "Seconds or HH:MM:SS, separated by commas or spaces, for example"
                    " 90, 5:00, 0:12:30.5."
                ),
            ),
        ),
        BoolInput("Gamma 2.4", default=False)
        .with_id(8)
        .with_docs(
            "Light models. The stored values are always analyzed and carried by the"
            " outputs; each model checked here also analyzes the frames in linear light"
            " (2 s or more per frame) and adds that evidence to Status and Report."
        ),
        BoolInput("sRGB", default=False).with_id(9),
        BoolInput("BT.709 camera", default=False).with_id(10),
        BoolInput("Gamma 2.2", default=False).with_id(11),
        NumberInput("Min Height", min=0, max=8192, default=0, unit="px")
        .with_id(12)
        .with_docs(
            "The native heights searched. 0 is automatic: 500-1000 for a 1080-line source,"
            " scaled to other heights."
        ),
        NumberInput("Max Height", min=0, max=8192, default=0, unit="px").with_id(13),
        NumberInput("Min Width", min=0, max=8192, default=0, unit="px")
        .with_id(14)
        .with_docs("The native widths searched. 0 follows the heights and the aspect."),
        NumberInput("Max Width", min=0, max=8192, default=0, unit="px").with_id(15),
        EnumInput(
            NativeResolutionKernels,
            "Kernels",
            default=NativeResolutionKernels.ALL,
            option_labels={
                NativeResolutionKernels.ALL: "All 13",
                NativeResolutionKernels.BICUBIC: "Bicubic",
                NativeResolutionKernels.LANCZOS: "Lanczos 3",
            },
        )
        .with_id(16)
        .with_docs(
            "The kernels tried: all 13 presets, or one family. Bicubic tries Catmull-Rom"
            " and fits free B and C coefficients."
        ),
        NumberInput("Lines", min=4, max=512, default=24)
        .with_id(17)
        .with_docs("Rows and columns sampled per axis for the size scan."),
        NumberInput("Refinement Lines", min=4, max=512, default=128)
        .with_id(18)
        .with_docs("Rows and columns used to refine the best fits."),
        BoolInput("Refine", default=True)
        .with_id(19)
        .with_docs(
            "Refine fractional sizes and phases, and fit free bicubic coefficients."
        ),
        NumberInput("Workers", min=0, max=64, default=0)
        .with_id(20)
        .with_docs("Threads. 0 is half the logical processors, at most 16."),
        NumberInput("Candidate", min=0, max=200, default=0)
        .with_id(21)
        .with_docs(
            "0 carries the automatic choice. N carries the Nth ranked candidate of the"
            " selected frame instead (recorded as a manual override)."
        ),
        DirectoryInput("Export Folder", must_exist=True)
        .make_optional()
        .with_id(22)
        .with_docs(
            "Optional: a new timestamped folder in it receives analysis.json,"
            " parameters.json, summary.json and curves.csv for the carried candidate."
        ),
    ],
    outputs=[
        NumberOutput("Width", output_type="uint"),
        NumberOutput("Height", output_type="uint"),
        NumberOutput("B"),
        NumberOutput("C"),
        NumberOutput("Taps", output_type="uint"),
        NumberOutput("Source Left"),
        NumberOutput("Source Top"),
        NumberOutput("Source Width"),
        NumberOutput("Source Height"),
        TextOutput("Kernel"),
        TextOutput("Light"),
        TextOutput("Status"),
        TextOutput("Report"),
        ImageOutput(
            "Height Plot",
            image_type=navi.Image(width=PLOT_WIDTH, height=PLOT_HEIGHT),
            channels=3,
        ),
        ImageOutput(
            "Width Plot",
            image_type=navi.Image(width=PLOT_WIDTH, height=PLOT_HEIGHT),
            channels=3,
        ),
        ImageOutput("Descaled").null_unless(0, "NativeResolutionSource::Image"),
    ],
    node_context=True,
    side_effects=True,
)
def get_native_resolution_node(
    context: NodeContext,
    source: NativeResolutionSource,
    image: np.ndarray | None,
    video_path: Path | None,
    sampling: NativeResolutionSampling,
    max_frames: int,
    scene_threshold: float,
    minimum_shot: float,
    times: str,
    gamma24: bool,
    srgb: bool,
    bt709: bool,
    gamma22: bool,
    min_height: int,
    max_height: int,
    min_width: int,
    max_width: int,
    kernels: NativeResolutionKernels,
    lines: int,
    refinement_lines: int,
    refine: bool,
    workers: int,
    candidate: int,
    export_folder: Path | None,
) -> tuple[
    int,
    int,
    float,
    float,
    int,
    float,
    float,
    float,
    float,
    str,
    str,
    str,
    str,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
]:
    models = [
        domain
        for domain, checked in (
            ("linear-gamma24", gamma24),
            ("linear-srgb", srgb),
            ("linear-bt709", bt709),
            ("linear-gamma22", gamma22),
        )
        if checked
    ]
    still = source is NativeResolutionSource.IMAGE
    if still:
        assert image is not None
        height = image.shape[0]
    else:
        if video_path is None:
            raise ValueError("Choose a video file.")
        info = media.probe(video_path, context.check_aborted)
        height = info.height
    low, high = search_bounds(height, min_height, max_height)
    session = Session(
        SearchOptions(
            low,
            high,
            min_width,
            max_width,
            lines,
            refinement_lines,
            workers,
            refine,
            kernels.value,
        )
    )
    done = threading.Event()
    watcher = threading.Thread(
        target=_watch, args=(context, session, done), daemon=True
    )
    watcher.start()
    try:
        if still:
            assert image is not None
            planned, completed, error, sampling_json = _analyze_image(
                context, session, image, models
            )
        else:
            assert video_path is not None
            planned, completed, error, sampling_json = _analyze_video(
                context,
                session,
                video_path,
                info,
                sampling,
                max_frames,
                times,
                scene_threshold,
                minimum_shot,
                models,
            )
    except RuntimeError:
        context.check_aborted()
        raise
    finally:
        done.set()
        watcher.join()
    report = session.summary(
        planned,
        completed,
        1 + len(models),
        sampling_json,
        error,
        candidate=candidate - 1,
    )
    summary = json.loads(report)
    return _outputs(session, summary, report, image if still else None, export_folder)


def _analyze_image(
    context: NodeContext, session: Session, image: np.ndarray, models: list[str]
) -> tuple[int, int, str, None]:
    """The frames planned and analyzed, the error (none) and the sampling (none)."""
    plane, provenance = media.still_plane(image)
    session.add_frame(plane, FrameFacts("", "Still image", provenance))
    if models:
        rgb = image[:, :, :3] if image.ndim == 3 and image.shape[2] >= 3 else None
        if rgb is None:
            gray = image.reshape(image.shape[:2])
            rgb = np.dstack((gray, gray, gray))
        for k, domain in enumerate(models):
            context.set_progress((k + 1) / (len(models) + 1))
            session.add_frame(
                linear_plane(rgb, domain, media.BT709_WEIGHTS),
                FrameFacts(
                    "",
                    "Still image",
                    media.linear_provenance(
                        domain, media.BT709_WEIGHTS, STILL_ASSUMPTIONS
                    ),
                    domain,
                    STILL_ASSUMPTIONS,
                ),
            )
    return 1, 1, "", None


def _analyze_video(
    context: NodeContext,
    session: Session,
    path: Path,
    info: media.VideoInfo,
    sampling: NativeResolutionSampling,
    max_frames: int,
    times: str,
    threshold: float,
    minimum_shot: float,
    models: list[str],
) -> tuple[int, int, str, dict[str, object]]:
    """The frames planned and analyzed, the error naming the frames that could not be
    decoded (skipped, each logged with its traceback; with none decoded the run fails),
    and the sampling report."""

    def tools() -> FFMpegEnv:
        return FFMpegEnv.get_integrated(context.storage_dir)

    plan = media.discover(
        path,
        info,
        sampling.value,
        max_frames,
        media.parse_times(times) if sampling is NativeResolutionSampling.TIMES else [],
        threshold,
        minimum_shot,
        tools,
        context.check_aborted,
    )
    colour = (
        media.colour_model(path, info, tools, context.check_aborted) if models else None
    )
    selected = media.capped_indices(len(plan.candidates), max_frames)
    failures: list[str] = []
    for k, index in enumerate(selected):
        context.set_progress(k / len(selected))
        point = plan.candidates[index]
        try:
            plane, provenance, bgr = media.decode_frame(
                path,
                info,
                point,
                tools,
                context.check_aborted,
                colour.rgb if colour else None,
            )
        except Exception as error:
            # an abort stops the run; any other failure skips the frame
            context.check_aborted()
            logger.exception(
                f"Get Native Resolution skips the frame at {point.seconds:.3f} s of"
                f" {path}: it could not be decoded"
            )
            failures.append(f"{point.seconds:.3f} s: {type(error).__name__}: {error}")
            continue
        sample = f"{point.kind} · {point.seconds:.3f} s"
        keyframe = point.kind == media.KEYFRAME
        session.add_frame(
            plane,
            FrameFacts(
                plan.path,
                sample,
                provenance,
                requested_seconds=point.seconds,
                keyframe_only=keyframe,
            ),
        )
        for domain in models:
            assert colour is not None and bgr is not None
            session.add_frame(
                linear_plane(bgr, domain, colour.weights),
                FrameFacts(
                    plan.path,
                    sample,
                    media.linear_provenance(domain, colour.weights, colour.assumptions),
                    domain,
                    colour.assumptions,
                    point.seconds,
                    colour.weights,
                    keyframe,
                ),
            )
    if len(failures) == len(selected):
        raise ValueError(
            f"None of the {len(selected)} frames could be decoded. The first:"
            f" {failures[0]}"
        )
    error = (
        f"{len(failures)} of {len(selected)} frames could not be decoded: "
        + "; ".join(failures)
        if failures
        else ""
    )
    return (
        len(selected),
        len(selected) - len(failures),
        error,
        media.sampling_report(plan, max_frames, threshold, minimum_shot),
    )


def _status(summary: dict, selection: dict | None) -> str:
    analyses = summary["analyses"]
    encoded = [a for a in analyses if a["light_domain"] == "encoded"]
    lines: list[str] = []
    if selection is None:
        first = encoded[0]
        lines.append(
            f"{first['verdict'].upper()}: no candidate. {first['explanation']}"
        )
    else:
        chosen = analyses[selection["analysis_id"]]
        width, height = selection["consensus_grid"]
        group = next(g for g in summary["groups"] if g["light_domain"] == "encoded")
        if len(encoded) == 1:
            lines.append(
                f"{chosen['verdict'].upper()}: {width}x{height}, {selection['kernel']}."
                f" {chosen['explanation']}"
            )
        else:
            votes = next(
                (
                    v["votes"]
                    for v in group["grid_votes"]
                    if [v["width"], v["height"]] == [width, height]
                ),
                0,
            )
            lines.append(
                f"{group['state'].upper().replace('_', ' ')}: {width}x{height}, {selection['kernel']}."
                f" {votes} of {group['analyzed']} frames vote for it ({group['supported']}"
                f" supported, {group['abstained']} abstained)."
            )
            lines.append(
                f"Kernel table: {group['kernel']['status']}; 8 % band:"
                f" {', '.join(group['kernel']['ambiguity_band']) or 'none'}."
            )
            lines.append(
                f"Carried frame: analysis {selection['analysis_id']} at"
                f" {chosen['requested_seconds']:.3f} s ({chosen['verdict']})."
            )
        lines.append(f"Kernel: {selection['rule']}.")
        free = selection.get("free_bicubic")
        if free is not None:
            lines.append(
                f"The free bicubic (b {free['b']:.3f}, c {free['c']:.3f}) fits"
                f" {free['advantage']:.1%} better than {selection['kernel']}: Candidate"
                f" {free['rank']} carries it."
            )
        if chosen["kernel_status"] != "supported_model" or len(encoded) > 1:
            lines.append(
                f"Frame kernel status: {chosen['kernel_status']} (gap to the next fit"
                f" {chosen['kernel_separation']:.2%})."
            )
        decision = summary.get("kernel_decision") or {}
        if decision.get("status") not in (None, "not_measured"):
            lines.append(
                f"Kernel decision: {decision['status']} by the {decision['rule']} rule"
                f" over {decision['frames']} frame(s); band:"
                f" {', '.join(decision['band']) or 'none'}."
            )
    planned = summary["planned_frames"]
    failed = planned - summary["completed_frames_all_domains"]
    if failed:
        lines.insert(
            1,
            f"{failed} of {planned} frames could not be decoded and were skipped (the"
            " Report's error names them).",
        )
    lines.append("Light: As stored (the stored values) carried by the outputs.")
    for group in summary["groups"]:
        if group["light_domain"] == "encoded":
            continue
        grid = group.get("leading_grid")
        label = LIGHT_LABELS[group["light_domain"]]
        verdicts = sorted(
            {
                a["verdict"]
                for a in analyses
                if a["light_domain"] == group["light_domain"]
            }
        )
        differs = (
            selection is not None
            and grid is not None
            and grid != selection["consensus_grid"]
        )
        lines.append(
            f"{label} model: {', '.join(verdicts)}; leading grid"
            f" {'x'.join(map(str, grid)) if grid else 'none'}"
            f"{' (differs from As stored)' if differs else ''}."
        )
    lines.append(
        "Set Descale's Border to Extend and keep its Linear Light off to reproduce"
        " the Descaled output."
    )
    return "\n".join(lines)


def _outputs(
    session: Session,
    summary: dict,
    report: str,
    image: np.ndarray | None,
    export_folder: Path | None,
):
    selection = summary["selection"]
    status = _status(summary, selection)
    if selection is None:
        analysis = summary["analyses"][0]
        width, height = analysis["source_width"], analysis["source_height"]
        empty = plot(np.empty(0), np.empty(0), 0)
        return (
            width,
            height,
            0.0,
            0.5,
            3,
            0.0,
            0.0,
            float(width),
            float(height),
            "None",
            LIGHT_LABELS["encoded"],
            status,
            report,
            empty,
            empty,
            image,
        )
    frame, index = selection["analysis_id"], selection["candidate_index"]
    parameters_text = session.parameters(frame, index)
    parameters = json.loads(parameters_text)
    width, height = parameters["native"]["width"], parameters["native"]["height"]
    geometry = parameters["geometry"]
    family = parameters["kernel"]["family"]
    # Vernier's kernel defaults where the family has no such parameter.
    kernel = {"b": 0.0, "c": 0.5, "taps": 3, **parameters["kernel"]["params"]}
    b, c, taps = float(kernel["b"]), float(kernel["c"]), int(kernel["taps"])
    descaled = None
    if image is not None:
        descaled = descale(
            image,
            width,
            height,
            DESCALE_KERNELS[family],
            taps,
            b,
            c,
            1.0,
            (geometry["offset_x"], geometry["offset_y"]),
            (geometry["extent_x"], geometry["extent_y"]),
            DESCALE_EXTEND,
        )
    if export_folder is not None:
        folder = export_folder / f"Vernier-{datetime.now():%Y%m%d-%H%M%S-%f}"[:-3]
        folder.mkdir()
        (folder / "analysis.json").write_text(session.report(frame), encoding="utf-8")
        (folder / "parameters.json").write_text(parameters_text, encoding="utf-8")
        (folder / "summary.json").write_text(report, encoding="utf-8")
        (folder / "curves.csv").write_text(session.curves_csv(frame), encoding="utf-8")
    heights = session.curve(frame, False)
    widths = session.curve(frame, True)
    return (
        width,
        height,
        b,
        c,
        taps,
        float(geometry["offset_x"]),
        float(geometry["offset_y"]),
        float(geometry["extent_x"]),
        float(geometry["extent_y"]),
        selection["kernel"],
        LIGHT_LABELS["encoded"],
        status,
        report,
        plot(*heights, height),
        plot(*widths, width),
        descaled,
    )
