"""example.vpy's lines 18-58 and 112-129 built from the real nodes as one chain, against the same
lines written by hand with vs-jetpack: frame- and property-exact in vspipe, on BlankClip-based clips
(no video files, no models).

test_vapoursynth_families.test_reference_lines_match_in_vspipe checks each of these operations alone
(with a stand-in for Split Planes and short frame lists). These tests check what only the composition
shows:
- two Split Planes nodes in one script, each of their three outputs feeding its own nodes (one split
  statement per node, calling `vstools.split` of the one `import vstools`), every plane to RGB
  float (lines 46-58), y0 = depth(y1, 16), and the planes joined back;
- the scene replacement of lines 112-125 with the script's own frame lists, typed as text: overlapping
  and repeated ranges, minus1 (Delete frames) and plus1 (Blank clip, then Splice) shifting by one
  frame, three Replace frames nodes and SAR 1:1 per clip, for two clips sharing minus1 and plus1;
- the denoise of lines 127-129 on that result;
- Save Script's file of those clips: output order, names, every statement once."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, cast

import pytest

# chaiNNer's inputs load the native backend, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

from test_vapoursynth_families import SAME, nodes_of, run, source

from api import NodeContext, NodeData, SettingsParser
from nodes.impl.vapoursynth import catalog
from nodes.impl.vapoursynth.catalog import load_catalog
from nodes.impl.vapoursynth.runner import clip_info
from nodes.impl.vapoursynth.script import VideoClip, VSFilter, compile_script
from packages.chaiNNer_vapoursynth.vapoursynth.save_script import save_script_node
from packages.chaiNNer_vapoursynth.vapoursynth.split_planes import split_planes_node

SHIPPED = sorted(catalog.SHIPPED_OVERLAYS.glob("*.toml"))

# The test sources: a progressive PAL-like clip and an interlaced NTSC-like one with X, Y and N
# patterns (as in test_vapoursynth_families), so that every filter has work.
PAL = (
    "core.std.BlankClip(format=vs.YUV420P8, width=96, height=72, length=30, fpsnum=25,"
    " fpsden=1).akarin.Expr(['X 2 * Y + N 11 * + 256 %', 'X Y 3 * + N 3 * + 256 %',"
    " 'X Y + N 7 * + 256 %'])"
)
NTSC = (
    "core.std.BlankClip(format=vs.YUV420P8, width=96, height=72, length=30,"
    " fpsnum=30000, fpsden=1001).akarin.Expr(['X Y + N 7 * + 256 %',"
    " 'X 3 * Y + N + 256 %', 'X Y 2 * + N 5 * + 256 %']).std.SetFrameProps("
    "_FieldBased=2)"
)


def _clip16(pattern: str) -> str:
    """A YUV444P16 clip like example.vpy's clip_a, clip_a1 and sharp (24 frames, 24000/1001)."""
    return (
        "core.std.BlankClip(format=vs.YUV444P16, width=96, height=72, length=24,"
        f" fpsnum=24000, fpsden=1001).akarin.Expr({pattern})"
    )


CLIP_A = _clip16(
    "['X Y + N 977 * + 257 * 65536 %', 'X 512 * 32768 +', 'Y 512 * N 9 * +']"
)
CLIP_A1 = _clip16("['X 3 * Y + 991 * N 101 * + 65536 %', 'Y 600 *', 'X 700 * N +']")
SHARP = _clip16("['X Y * N 4099 * + 65536 %', 'N 2000 * X +', 'Y 300 * 9000 +']")

# example.vpy's frame lists, as reference.vpy maps them onto 30-frame clips: detel (lines 38-39),
# range1, range2 and range3 (lines 115-117).
DETEL = [(3, 5), (14, 15), 17, (19, 20)]
RANGE1 = [
    *[(1, 1), (1, 2), (2, 2), (2, 2), (2, 2), (2, 2), (3, 3), (3, 3), (3, 3), (3, 3)],
    *[(3, 3), (3, 4), (4, 4), (4, 4), (4, 4), (4, 4), (5, 5), (5, 5), (5, 5), (7, 7)],
    *[(7, 7), (7, 7), (8, 8), (8, 8), (8, 8), (9, 9), (9, 9), (10, 10), (10, 10)],
    *[(10, 11), (11, 11), (11, 11), (11, 11), (12, 12), (13, 13), (13, 14), (14, 14)],
    *[(14, 14), (14, 14), (14, 15), (15, 16), (16, 16), (19, 20), (20, 20), (20, 20)],
    *[(20, 20), (20, 20), (21, 21), (21, 21), (21, 22), (22, 22), (22, 22)],
]
RANGE2 = [1, 3, 5, 10, 10, 11, 14, 15, 16, 19, 20, 21]
RANGE3 = [1, 4, 4, 9, 12, 14, 20]

REF_IMPORTS = [
    "from fractions import Fraction",
    "from vsaa import NNEDI3",
    "from vsdeinterlace import vdecimate, vfm",
    "from vsdenoise import DFTTest, MVToolsPreset, bm3d, mc_degrain, nl_means",
    "from vstools import change_fps, depth, join, replace_ranges, split",
]


def as_text(frames: Sequence[int | tuple[int, int]]) -> str:
    """A frame list as a user types it into Replace frames: "3-5, 14-15, 17"."""
    return ", ".join(
        f"{f[0]}-{f[1]}" if isinstance(f, tuple) else str(f) for f in frames
    )


class Context:
    """The NodeContext members Split Planes and Save Script use."""

    def __init__(self, storage: Path, node_id: str) -> None:
        self.storage_dir = storage
        self.node_id = node_id
        self.settings = SettingsParser({"cache_mb": 1024})


@pytest.fixture(scope="module")
def real_nodes(tmp_path_factory: pytest.TempPathFactory) -> dict[str, NodeData]:
    """The family nodes of the venv's real catalog, with the shipped overlays."""
    pytest.importorskip("vapoursynth")
    path = catalog.ensure_catalog(tmp_path_factory.mktemp("vapoursynth"))
    assert path is not None
    return nodes_of(load_catalog(path, SHIPPED))


class Chain:
    """Runs the nodes of one chain: family nodes by (family, method, values), with
    node ids s1, s2..."""

    def __init__(self, nodes: dict[str, NodeData], storage: Path) -> None:
        self.nodes = nodes
        self.storage = storage
        self.count = 0

    def _id(self) -> str:
        self.count += 1
        return f"s{self.count}"

    def context(self) -> NodeContext:
        return cast(NodeContext, Context(self.storage, self._id()))

    def __call__(
        self, family: str, method: str, clip: VideoClip | None = None, **values: object
    ) -> VideoClip:
        out = run(self.nodes[family], self._id(), method, values, clip=clip)
        assert isinstance(out, VideoClip)
        return out

    def filter(self, family: str, method: str, **values: object) -> VSFilter:
        out = run(self.nodes[family], self._id(), method, values, as_filter=True)
        assert isinstance(out, VSFilter)
        return out

    def props(self, clip: VideoClip, sar_num: str, sar_den: str) -> VideoClip:
        return self(
            "Clip Properties",
            "std.SetFrameProps",
            clip,
            _Matrix="6",
            _Transfer="6",
            _Primaries="6",
            _SARNum=sar_num,
            _SARDen=sar_den,
        )

    def to_float(self, plane: VideoClip) -> VideoClip:
        """Lines 46-58: resize.Point to RGBS, then depth 32."""
        rgbs = self("Convert Format", "resize.Point", plane, format="vs.RGBS")
        return self("Convert Format", "vstools.depth", rgbs, bitdepth="32")


def evaluate(script: str, reference: list[str], checks: list[str], path: Path) -> None:
    """The chain's script, then the hand-written lines and the comparisons, in one vspipe
    evaluation (a failed comparison fails it)."""
    path.write_text(
        "\n".join([script, *REF_IMPORTS, *reference, SAME, *checks, ""]),
        encoding="utf-8",
    )
    assert clip_info(path, 0).num_frames > 0


def test_split_planes_compose_with_the_families(
    real_nodes: dict[str, NodeData], tmp_path: Path
) -> None:
    """Lines 18-22 and 28-29 (sources), 33-39 (IVTC), 44-58 (planes), and the planes joined
    back."""
    c = Chain(real_nodes, tmp_path)
    src = c.props(source("n_pal_0", PAL), "16", "15")
    src = c("Clip Properties", "std.AssumeFPS", src, fpsnum="25", fpsden="1")
    src = c("Cut and Splice", "std.Trim", src, first="3")
    src = c("Clip Properties", "vstools.change_fps", src, fps="24000/1001")
    src2 = c.props(source("n_ntsc_0", NTSC), "8", "9")
    nnedi3 = {"pscrn": "0", "tff": "True", "double_rate": 0}
    post = c.filter("Deinterlace", "vsaa.NNEDI3.deinterlace", **nnedi3)
    iv = c("Deinterlace", "vsdeinterlace.vfm", src2, postprocess=post)
    iv = c("Deinterlace", "vsdeinterlace.vdecimate", iv)
    pp = c("Deinterlace", "vsdeinterlace.vfm", src2)
    pp = c("Deinterlace", "vsdeinterlace.vdecimate", pp)
    pp = c("Deinterlace", "vsaa.NNEDI3.deinterlace", pp, **nnedi3)
    iv = c(
        "Cut and Splice",
        "vstools.replace_ranges",
        iv,
        clip_b=pp,
        ranges=as_text(DETEL),
    )
    y, u, v = split_planes_node(c.context(), src)
    y1, u1, v1 = split_planes_node(c.context(), iv)
    y0 = c("Convert Format", "vstools.depth", y1, bitdepth="16")
    floats = [c.to_float(plane) for plane in (y, u, v, y1, u1, v1)]
    u16 = c("Convert Format", "vstools.depth", u1, bitdepth="16")
    v16 = c("Convert Format", "vstools.depth", v1, bitdepth="16")
    joined = c("Planes", "vstools.join", y0, plane1_or_family=u16, plane2=v16)

    outputs = [*floats, y0, joined]
    script = compile_script([(clip, clip.var) for clip in outputs], cache_mb=1024)
    lines = script.splitlines()
    # One statement per Split Planes node, each plane used by its own nodes.
    for planes in ((y, u, v), (y1, u1, v1)):
        assert lines.count(f"{planes[0].stmt}  # node:{planes[0].node_id}") == 1
    assert "import vstools" in lines and "from vstools import split" not in lines

    reference = [
        "r_src = core.std.SetFrameProps(" + PAL + ", _Matrix=6, _Transfer=6,"
        " _Primaries=6, _SARNum=16, _SARDen=15)",
        "r_src = core.std.AssumeFPS(clip=r_src, fpsnum=25, fpsden=1)",
        "r_src = r_src[3:]",
        "r_src = change_fps(r_src, Fraction(24000, 1001))",
        "r_src2 = core.std.SetFrameProps(" + NTSC + ", _Matrix=6, _Transfer=6,"
        " _Primaries=6, _SARNum=8, _SARDen=9)",
        (
            "r_iv = vfm(r_src2, postprocess=NNEDI3(pscrn=0, tff=True,"
            " double_rate=False).deinterlace)"
        ),
        "r_iv = vdecimate(r_iv)",
        "r_pp = vfm(r_src2)",
        "r_pp = vdecimate(r_pp)",
        "r_pp = NNEDI3(pscrn=0, tff=True, double_rate=False).deinterlace(r_pp)",
        f"r_iv = replace_ranges(r_iv, r_pp, {DETEL!r})",
        # example.vpy 44-58, in its order.
        "y, u, v = split(r_src)",
        "y1, u1, v1 = split(r_iv)",
        "u1_plane, v1_plane = u1, v1",
        "y = core.resize.Point(y, format=vs.RGBS)",
        "y0 = depth(y1, 16)",
        "y1 = core.resize.Point(y1, format=vs.RGBS)",
        "y = depth(y, 32)",
        "y1 = depth(y1, 32)",
        "u = core.resize.Point(u, format=vs.RGBS)",
        "u1 = core.resize.Point(u1, format=vs.RGBS)",
        "u = depth(u, 32)",
        "u1 = depth(u1, 32)",
        "v = core.resize.Point(v, format=vs.RGBS)",
        "v1 = core.resize.Point(v1, format=vs.RGBS)",
        "v = depth(v, 32)",
        "v1 = depth(v1, 32)",
        "joined = join(y0, depth(u1_plane, 16), depth(v1_plane, 16))",
    ]
    names = ["y", "u", "v", "y1", "u1", "v1", "y0", "joined"]
    checks = [
        f"_same({clip.var}, {name}, {name!r})"
        for clip, name in zip(outputs, names, strict=True)
    ]
    # The formats lines 46-58 make; U and V differ, so a swapped plane fails its
    # comparison.
    checks += [
        (
            "assert (y.format.name, u.width, joined.format.name) =="
            " ('RGBS', 48, 'YUV420P16'), (y.format.name, u.width, joined.format.name)"
        ),
        "assert bytes(u.get_frame(0)[0]) != bytes(v.get_frame(0)[0])",
    ]
    evaluate(script, reference, checks, tmp_path / "planes.vpy")


def test_scene_replacement_composes(
    real_nodes: dict[str, NodeData], tmp_path: Path
) -> None:
    """Lines 112-125 with the script's frame lists, then 127-129, saved by Save Script."""
    c = Chain(real_nodes, tmp_path)
    clip_a = source("n_clip_a_0", CLIP_A)
    clip_a1 = source("n_clip_a1_0", CLIP_A1)
    sharp = source("n_sharp_0", SHARP)
    minus1 = c("Cut and Splice", "std.DeleteFrames", clip_a, frames="0")
    plus1 = c("Cut and Splice", "std.BlankClip", clip_a, length="1")
    plus1 = c("Cut and Splice", "std.Splice", plus1, **{"clips#2": clip_a})

    def decen(base: VideoClip) -> VideoClip:
        clip = base
        for other, frames in ((clip_a, RANGE1), (minus1, RANGE2), (plus1, RANGE3)):
            clip = c(
                "Cut and Splice",
                "vstools.replace_ranges",
                clip,
                clip_b=other,
                ranges=as_text(frames),
            )
        return c.props(clip, "1", "1")

    decen1, decen2 = decen(clip_a1), decen(sharp)
    dft = c.filter(
        "Denoise",
        "vsdenoise.DFTTest.denoise",
        backend="vsdenoise.DFTTest.Backend.CPU",
        sloc="{0.0: 4, 0.2: 9, 1.0: 15}",
    )
    ref = c(
        "Denoise",
        "vsdenoise.mc_degrain",
        decen2,
        prefilter=dft,
        preset="vsdenoise.MVToolsPreset.HQ_SAD",
        thsad="100",
    )
    denoise = c(
        "Denoise",
        "vsdenoise.bm3d",
        decen2,
        sigma="0.8",
        tr="4",
        profile="vsdenoise.bm3d.Profile.HIGH",
        ref=ref,
        planes="[0]",
        backend="vsdenoise.bm3d.Backend.OLD",
    )
    denoise = c(
        "Denoise",
        "vsdenoise.nl_means",
        denoise,
        h="0.2",
        tr="4",
        ref=ref,
        planes="[1, 2]",
        backend="vsdenoise.nl_means.Backend.ISPC",
    )

    # Save Script: clip 1 is output 0; a pair left empty closes up.
    named = [(denoise, "denoised"), (decen1, "upscaled"), (None, ""), (decen2, "")]
    values = [value for pair in named for value in pair]
    values += [None] * (32 - len(values))
    save_script_node(c.context(), tmp_path / "saved", "example", *values)
    script = (tmp_path / "saved" / "example.vpy").read_text(encoding="utf-8")
    lines = script.splitlines()
    assert [line for line in lines if line.startswith("set_output(")] == [
        f"set_output({denoise.var}, 0, 'denoised')",
        f"set_output({decen1.var}, 1, 'upscaled')",
        f"set_output({decen2.var}, 2, 'Clip 4')",
    ]
    # minus1, plus1 and the denoise reference: one statement each, used twice.
    for record in (minus1, plus1, ref):
        assert lines.count(f"{record.stmt}  # node:{record.node_id}") == 1
    assert f"ranges={RANGE1!r}" in script and f"ranges={RANGE2!r}" in script

    reference = [
        f"clip_a = {CLIP_A}",
        f"clip_a1 = {CLIP_A1}",
        f"sharp = {SHARP}",
        # example.vpy 112-125.
        "minus1 = core.std.DeleteFrames(clip_a, 0)",
        "plus1 = core.std.BlankClip(clip_a, length=1)",
        "plus1 = plus1 + clip_a",
        f"range1 = {RANGE1!r}",
        f"range2 = {RANGE2!r}",
        f"range3 = {RANGE3!r}",
        "decen1 = replace_ranges(clip_a1, clip_a, range1)",
        "decen1 = replace_ranges(decen1, minus1, range2)",
        "decen1 = replace_ranges(decen1, plus1, range3)",
        (
            "decen1 = core.std.SetFrameProps(decen1, _Matrix=6, _Transfer=6, _Primaries=6,"
            " _SARNum=1, _SARDen=1)"
        ),
        "decen2 = replace_ranges(sharp, clip_a, range1)",
        "decen2 = replace_ranges(decen2, minus1, range2)",
        "decen2 = replace_ranges(decen2, plus1, range3)",
        (
            "decen2 = core.std.SetFrameProps(decen2, _Matrix=6, _Transfer=6, _Primaries=6,"
            " _SARNum=1, _SARDen=1)"
        ),
        # example.vpy 127-129, with reference.vpy's CPU backends.
        (
            "ref = mc_degrain(decen2, prefilter=DFTTest(sloc={0.0: 4, 0.2: 9, 1.0: 15},"
            " backend=DFTTest.Backend.CPU).denoise, preset=MVToolsPreset.HQ_SAD, thsad=100)"
        ),
        (
            "denoise = bm3d(decen2, sigma=0.8, tr=4, profile=bm3d.Profile.HIGH, ref=ref,"
            " planes=0, backend=bm3d.Backend.OLD)"
        ),
        (
            "denoise = nl_means(denoise, 0.2, tr=4, ref=ref, planes=[1, 2],"
            " backend=nl_means.Backend.ISPC)"
        ),
    ]
    checks = [
        f"_same({record.var}, {name}, {name!r})"
        for record, name in (
            (minus1, "minus1"),
            (plus1, "plus1"),
            (decen1, "decen1"),
            (decen2, "decen2"),
            (ref, "ref"),
            (denoise, "denoise"),
        )
    ]
    # The shift by one frame, frame by frame: 0 and 6 are in no list; 1 is in range3
    # (plus1, the frame before); 3 is in range2 (minus1, the frame after); 7 is in
    # range1 only.
    checks += [
        "_plane = lambda clip, n: bytes(clip.get_frame(n)[0])",
        f"assert _plane({decen1.var}, 0) == _plane(clip_a1, 0)",
        f"assert _plane({decen1.var}, 1) == _plane(clip_a, 0)",
        f"assert _plane({decen1.var}, 3) == _plane(clip_a, 4)",
        f"assert _plane({decen1.var}, 7) == _plane(clip_a, 7)",
        f"assert _plane({decen1.var}, 6) == _plane(clip_a1, 6)",
    ]
    evaluate(script, reference, checks, tmp_path / "scenes.vpy")
