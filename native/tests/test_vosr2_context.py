"""Scene windows must never copy context pixels into the restored output."""

import itertools
from contextlib import nullcontext
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch

from api import Progress
from nodes.impl.pytorch import (
    vosr2_context,
    vosr2_memory,
    vosr2_scene,
    vosr2_trt_restore,
)
from nodes.impl.pytorch.vosr2 import (
    MemoryPlan,
    VosrBundle,
    VosrColorCorrection,
    VosrMemoryMode,
    tile_positions,
)
from nodes.impl.pytorch.vosr2_context import context_denoise, pad_image
from nodes.impl.pytorch.vosr2_scene import (
    _context_position,
    _context_strengths,
    detail_energy,
)


@pytest.mark.parametrize("size", [(1, 1), (16, 32), (512, 512), (528, 720)])
def test_minimum_window_padding_preserves_input(size):
    height, width = size
    value = torch.arange(height * width).float().reshape(1, 1, height, width)
    actual = pad_image(value)
    assert actual.shape[-2:] == (max(512, height), max(512, width))
    torch.testing.assert_close(actual[:, :, :height, :width], value, atol=0, rtol=0)
    assert torch.isfinite(actual).all()
    if height > 1 and height < 512:
        torch.testing.assert_close(actual[:, :, height, :width], value[:, :, -2])


def test_context_selection_is_input_derived_and_phase_aligned():
    image = torch.zeros(1, 3, 544, 720)
    image[:, :, :, 320:384:2] = 1
    image[:, :, :, 321:384:2] = -1
    y, x = _context_position(detail_energy(image))
    assert (y, x) == (32, 320)
    assert y % 16 == x % 16 == 0


@pytest.mark.parametrize("shape", [(64, 64), (66, 98), (96, 130)])
@pytest.mark.parametrize("stream_blocks", [False, True])
@torch.inference_mode()
def test_appended_context_never_escapes_into_output(shape, stream_blocks):
    class Dino(torch.nn.Module):
        def forward_intermediate_layer(self, image, layer):
            assert image.shape == (1, 3, 448, 448) and layer == 17
            return torch.zeros(1, 1024, 2)

    class Dit(torch.nn.Module):
        def forward_flexible(self, inputs, time, **kwargs):
            assert inputs.shape in ((1, 32, 64, 64), (1, 32, 64, 72))
            assert time.item() == 1
            kwargs["check_cancel"]()
            # A perfect conditional identity flow. Any leaked context, wrong
            # noise coordinates, uncovered output or crop shift breaks parity.
            return inputs[:, 16:] - inputs[:, :16]

    height, width = shape
    generator = torch.Generator().manual_seed(123)
    latent = torch.randn(1, 16, height, width, generator=generator)
    image = torch.rand(1, 3, height * 8, width * 8, generator=generator) * 2 - 1
    before = latent.clone()
    rng = torch.random.get_rng_state().clone()
    bundle = cast(
        VosrBundle, SimpleNamespace(dino=Dino(), dit=Dit(), dtype=torch.float32)
    )
    plan = MemoryPlan(7_000_000_000, 128 * 1024**2, 256, stream_blocks)
    result = context_denoise(
        bundle,
        latent,
        image,
        torch.device("cpu"),
        4,
        plan,
        Progress.noop_progress(),
        lambda: None,
    )
    torch.testing.assert_close(result, before, atol=1e-6, rtol=1e-5)
    assert torch.equal(latent, before)
    assert torch.equal(torch.random.get_rng_state(), rng)


@torch.inference_mode()
def test_disagreeing_windows_crossfade_without_a_step():
    class Dino(torch.nn.Module):
        def forward_intermediate_layer(self, image, layer):
            return torch.zeros(1, 1024, 2)

    class Dit(torch.nn.Module):
        window = 0

        def forward_flexible(self, inputs, time, **kwargs):
            result = inputs[:, 16:] + self.window
            self.window += 1
            return result

    bundle = cast(
        VosrBundle, SimpleNamespace(dino=Dino(), dit=Dit(), dtype=torch.float32)
    )
    result = context_denoise(
        bundle,
        torch.zeros(1, 16, 64, 120),
        torch.zeros(1, 3, 512, 960),
        torch.device("cpu"),
        0,
        MemoryPlan.for_mode(VosrMemoryMode.DEFAULT),
        Progress.noop_progress(),
        lambda: None,
    )
    line = -result[0, 0, 32]
    overlap = line[56:64]
    assert overlap[0] < 0.02 and overlap[-1] > 0.98
    assert bool(torch.all(torch.diff(overlap) > 0))
    # No single pixel may carry a majority of the unit jump between windows.
    assert torch.diff(line).abs().max() < 0.5
    torch.testing.assert_close(overlap + overlap.flip(0), torch.ones(8))


def test_context_activity_transition_preserves_sharp_windows():
    energy = np.concatenate(
        [np.full((512, 512), v, dtype=np.float32) for v in (1, 0.15, 0)], axis=1
    )
    result = _context_strengths(energy, [(0, 0), (0, 64), (0, 128)], (0, 0))
    assert result == pytest.approx([0, 0.5, 1], abs=1e-5)


def test_flat_input_does_not_invent_a_sharpness_reference():
    energy = detail_energy(torch.ones(1, 3, 512, 960))
    assert _context_strengths(energy, [(0, 0), (0, 56)], (0, 0)) == [
        0,
        0,
    ]


@torch.inference_mode()
def test_context_preserves_local_pixels_noise_and_dino_view(monkeypatch):
    generator = torch.Generator().manual_seed(23)
    latent = torch.randn(1, 16, 64, 176, generator=generator)
    image = torch.rand(1, 3, 512, 1408, generator=generator) * 2 - 1
    noise = torch.randn(latent.shape, generator=torch.Generator().manual_seed(4))
    inputs = torch.cat((latent, noise), dim=1)
    calls = []
    dino_calls = []
    monkeypatch.setattr(
        vosr2_context,
        "scene_routing",
        lambda pixels, windows: (
            vosr2_scene.Layout([(0, 0), (0, 56), (0, 112)], []),
            (0, 16),
            [0, 0.5, 1],
        ),
    )

    class Dino(torch.nn.Module):
        def forward_intermediate_layer(self, pixels, layer):
            dino_calls.append(pixels.clone())
            return torch.zeros(1, 1024, 2)

    class Dit(torch.nn.Module):
        def forward_flexible(self, value, time, **kwargs):
            calls.append(value.clone())
            return value[:, 16:] - value[:, :16] + (2 if value.shape[-1] == 72 else 0)

    bundle = cast(
        VosrBundle, SimpleNamespace(dino=Dino(), dit=Dit(), dtype=torch.float32)
    )
    result = context_denoise(
        bundle,
        latent,
        image,
        torch.device("cpu"),
        4,
        MemoryPlan.for_mode(VosrMemoryMode.DEFAULT),
        Progress.noop_progress(),
        lambda: None,
    )
    assert [value.shape[-1] for value in calls] == [64, 64, 72, 72]
    for value, start in zip(calls, (0, 56, 56, 112), strict=True):
        torch.testing.assert_close(
            value[:, :, :, :64], inputs[:, :, :, start : start + 64]
        )
        if value.shape[-1] == 72:
            torch.testing.assert_close(value[:, :, :, 64:], inputs[:, :, :, 2:10])
    for actual, start in zip(dino_calls, (0, 56, 112), strict=True):
        crop = image[:, :, :, start * 8 : start * 8 + 512]
        expected = torch.nn.functional.interpolate(
            crop * 0.5 + 0.5, size=(448, 448), mode="bicubic", align_corners=False
        ).clamp_(0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
        torch.testing.assert_close(actual, (expected - mean) / std)
    # Only the context prediction changes; its contribution is 0, 1/2, then 1.
    for column, change in ((16, 0), (88, 1), (160, 2)):
        torch.testing.assert_close(
            result[:, :, :, column], latent[:, :, :, column] - change
        )


def memory_mock(monkeypatch, used=2_000_000_000, previous=0.4):
    state = SimpleNamespace(used=used, fraction=previous, history=[])
    gpu = SimpleNamespace(
        index=0,
        get_current_vram_usage=lambda: SimpleNamespace(
            total=32_000_000_000, used=state.used
        ),
    )
    monkeypatch.setattr(vosr2_memory, "nvidia", SimpleNamespace(devices=[gpu]))
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(total_memory=32_000_000_000),
    )
    monkeypatch.setattr(
        torch.cuda, "get_per_process_memory_fraction", lambda _: state.fraction
    )

    def set_fraction(value, device):
        assert device == torch.device("cuda:0")
        state.fraction = value
        state.history.append(value)

    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", set_fraction)
    return state


@pytest.mark.parametrize("failure", [None, "cancel", "oom", "external"])
def test_vram_quota_restores_allocator_after_every_exit(monkeypatch, failure):
    state = memory_mock(monkeypatch)

    def work():
        with vosr2_memory.vram_budget(
            torch.device("cuda:0"), VosrMemoryMode.LOW, Progress.noop_progress()
        ) as check:
            assert state.fraction == pytest.approx(4_200_000_000 / 32_000_000_000)
            if failure == "cancel":
                raise InterruptedError("cancelled")
            if failure == "oom":
                raise torch.OutOfMemoryError("test allocator cap")
            if failure == "external":
                state.used = 6_950_000_000
            check()

    if failure:
        expected = InterruptedError if failure == "cancel" else RuntimeError
        with pytest.raises(expected):
            work()
    else:
        work()
    assert state.fraction == 0.4
    assert len(state.history) == 2


def test_vram_quota_keeps_existing_stricter_limit_and_app_budget(monkeypatch):
    state = memory_mock(monkeypatch, previous=0.05)
    with vosr2_memory.vram_budget(
        torch.device("cuda:0"), VosrMemoryMode.DEFAULT, Progress.noop_progress(), 1
    ):
        assert state.fraction == pytest.approx(1024**3 / 32_000_000_000)
    assert state.fraction == 0.05


def test_default_profile_has_no_ceiling_below_the_device(monkeypatch):
    state = memory_mock(monkeypatch, previous=1.0)
    with vosr2_memory.vram_budget(
        torch.device("cuda:0"), VosrMemoryMode.DEFAULT, Progress.noop_progress()
    ) as check:
        # The whole device minus other use and the fixed headroom; no 30 GB cap.
        assert state.fraction == pytest.approx(29_200_000_000 / 32_000_000_000)
        state.used = 31_000_000_000
        check()
        state.used = 31_950_000_000
        with pytest.raises(RuntimeError, match="safety boundary"):
            check()
    assert state.fraction == 1.0


def test_vram_quota_rejects_busy_device_before_mutation(monkeypatch):
    state = memory_mock(monkeypatch, used=6_300_000_000)
    with pytest.raises(RuntimeError, match="Not enough free"):
        with vosr2_memory.vram_budget(
            torch.device("cuda:0"), VosrMemoryMode.LOW, Progress.noop_progress()
        ):
            pytest.fail("Busy device admitted")
    assert state.history == []
    state.used = 2_000_000_000
    with vosr2_memory.vram_budget(
        torch.device("cuda:0"), VosrMemoryMode.LOW, Progress.noop_progress()
    ) as check:
        check()


def test_cpu_profile_never_mutates_cuda_allocator(monkeypatch):
    def forbidden(*args):
        pytest.fail("CPU inference touched the CUDA allocator")

    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", forbidden)
    with vosr2_memory.vram_budget(
        torch.device("cpu"), VosrMemoryMode.LOW, Progress.noop_progress()
    ) as check:
        check()


def test_context_selection_on_flat_input_has_stable_tie_break():
    image = torch.from_numpy(np.zeros((1, 3, 512, 512), dtype=np.float32))
    assert _context_position(detail_energy(image)) == (0, 448)


def hand_overs(positions):
    return [a + 32 + (b - a) // 2 for a, b in itertools.pairwise(positions)]


@pytest.mark.parametrize("length", [64, 120, 270, 480, 961])
def test_window_positions_cover_the_axis_with_bounded_steps(length):
    count = len(tile_positions(length, 64, 8))
    cost = np.random.default_rng(length).random(length)
    positions = vosr2_scene.window_positions(cost, count)
    assert len(positions) == count and positions[0] == 0
    assert positions[-1] == length - 64
    steps = np.diff(positions)
    assert np.all(steps[:-1] >= vosr2_scene.MIN_STEP) if len(steps) > 1 else True
    assert np.all(steps >= 1) and np.all(steps <= vosr2_scene.MAX_STEP)


def test_hand_overs_avoid_a_detailed_band_and_flat_axes_stay_even():
    flat = vosr2_scene.window_positions(np.zeros(480), 9)
    assert list(np.diff(flat)) == [52] * 8
    cost = np.zeros(480)
    cost[100:120] = 1  # detail on the even layout's second and fifth hand-overs
    cost[255:275] = 1
    positions = vosr2_scene.window_positions(cost, 9)
    assert not any(100 <= c < 120 or 255 <= c < 275 for c in hand_overs(positions))


def test_scene_windows_keep_a_face_inside_one_window_and_cover_everything():
    height, width = 135, 240
    energy = np.zeros((height * 8, width * 8), np.float32)
    # A face across the regular grid's second hand-over (latent 116, x 928).
    energy[200:700, 910:1000] = 0.05
    layout = vosr2_scene.scene_windows(energy, height, width)
    grid = layout.grid
    assert len(grid) == len(regular(height, width)) and layout.faces == []
    covered = np.zeros((height, width), bool)
    for y, x in grid:
        covered[y : y + 64, x : x + 64] = True
    assert covered.all()
    for y in sorted({y for y, _ in grid}):
        if y * 8 < 700 and (y + 64) * 8 > 200:
            row = sorted(x for yy, x in grid if yy == y)
            assert not any(910 <= c * 8 < 1000 for c in hand_overs(row))


def test_a_video_keeps_its_layout_within_a_shot_and_replans_at_a_cut():
    plans = []

    def plan(name):
        def make():
            plans.append(name)
            return vosr2_scene.Layout([(0, len(plans))], [])

        return make

    windows = vosr2_scene.WindowPlan()
    ramp = np.linspace(0, 1, 480, dtype=np.float32)
    shot = np.repeat(np.tile(ramp, (270, 1))[:, :, None], 3, axis=2)
    windows.begin_frame(shot)
    first = [windows.layout(plan("tile 1")), windows.layout(plan("tile 2"))]
    windows.begin_frame(np.clip(shot + 0.01, 0, 1))  # the same shot, a little brighter
    assert [windows.layout(plan("x")), windows.layout(plan("y"))] == first
    assert plans == ["tile 1", "tile 2"]
    windows.begin_frame(1 - shot)  # a cut
    assert windows.layout(plan("cut")).grid == [(0, 3)]
    for step in range(1, 6):  # a slow change (a pan, a fade) never re-plans mid-shot
        windows.begin_frame(np.clip(1 - shot + 0.05 * step, 0, 1))
        assert windows.layout(plan("drift")).grid == [(0, 3)]
    windows.begin_frame(np.zeros((300, 480, 3), np.float32))  # another size
    assert windows.layout(plan("size")).grid == [(0, 4)]
    windows.begin_frame(np.zeros((300, 484, 3), np.float32))  # same 1/16 size, 1 latent
    assert windows.layout(plan("wider")).grid == [(0, 5)]
    assert plans == ["tile 1", "tile 2", "cut", "size", "wider"]


def test_routing_with_a_video_plan_reuses_the_shot_layout():
    windows = vosr2_scene.WindowPlan()
    image = (
        torch.rand(1, 3, 1080, 1920, generator=torch.Generator().manual_seed(2)) * 2 - 1
    )
    frame = (image[0].permute(1, 2, 0).numpy() * 0.5 + 0.5).astype(np.float32)
    windows.begin_frame(frame)
    layout, _, _ = vosr2_scene.scene_routing(image, windows)
    assert layout == vosr2_scene.scene_routing(image)[0]
    moved = torch.roll(image, 40, dims=3)
    windows.begin_frame(frame)
    assert vosr2_scene.scene_routing(moved, windows)[0] is layout


def regular(height: int, width: int) -> vosr2_scene.Grid:
    """The regular grid with 8 latents of overlap."""
    return [
        (y, x)
        for y in tile_positions(height, 64, 8)
        for x in tile_positions(width, 64, 8)
    ]


def test_a_face_split_by_a_seam_gets_a_window_centred_on_it():
    grid = regular(135, 240)  # hand-overs at latents x 60, 116, 172; y 60
    split = vosr2_scene.Face(1250, 200, 300, 400)  # across x 172 and y 60
    whole = vosr2_scene.Face(150, 150, 100, 150)  # inside the first window
    assert vosr2_scene.face_windows([split, whole], grid, 135, 240) == [(18, 143)]
    assert vosr2_scene.face_windows([whole], grid, 135, 240) == []


def test_a_face_window_needs_a_middle_it_can_hold_alone_inside_the_image():
    grid = regular(135, 240)
    # Split by hand-overs, but its middle (500 px) is wider than a face window holds.
    wide = vosr2_scene.Face(900, 0, 1000, 1000)
    outside = vosr2_scene.Face(-400, -400, 300, 300)
    assert vosr2_scene.face_windows([wide, outside], grid, 135, 240) == []


def test_face_windows_are_capped_largest_first_and_kept_apart():
    grid = regular(135, 240)
    faces = [
        vosr2_scene.Face(x - 100, 280, 200, 200)
        for x in (480, 928, 1376, 1632)  # each across a vertical hand-over
    ]
    faces.append(vosr2_scene.Face(1376 - 150, 280, 300, 300))  # largest
    added = vosr2_scene.face_windows(faces, grid, 135, 240)
    assert len(added) == int(vosr2_scene.SEAM_SHARE * len(grid)) == 3
    assert added[0] == (22, 140)  # the largest face first
    centres = [(y + 32, x + 32) for y, x in added]
    for (ay, ax), (by, bx) in itertools.combinations(centres, 2):
        assert max(abs(ay - by), abs(ax - bx)) >= 2 * vosr2_scene.SEAM_REACH - 1


def test_faces_are_found_on_a_quarter_size_copy_and_mapped_back(monkeypatch, tmp_path):
    sizes = []

    def detect(image):
        assert image.dtype == np.uint8 and image.shape[:2] == (135, 240)
        assert (image[0, 0] == [0, 0, 255]).all()  # red, in YuNet's BGR order
        landmarks = [10, 20, 30, 20, 20, 30, 12, 40, 28, 40]
        return 1, np.array([[5, 10, 30, 40, *landmarks, 0.9]], np.float32)

    detector = SimpleNamespace(setInputSize=sizes.append, detect=detect)

    monkeypatch.setattr(vosr2_scene, "_DETECTORS", {})
    monkeypatch.setattr(
        vosr2_scene.cv2,
        "FaceDetectorYN",
        SimpleNamespace(create=lambda *args: detector),
    )
    image = torch.full((1, 3, 540, 960), -1.0)
    image[:, 0] = 1  # red
    faces = vosr2_scene.find_faces(image, tmp_path / "faces.onnx")
    assert sizes == [(240, 135)]
    assert faces == [vosr2_scene.Face(20, 40, 120, 160)]
    # OpenCV returns None, not an empty array, when it finds no face.
    detector.detect = lambda image: (0, None)
    assert vosr2_scene.find_faces(image, tmp_path / "faces.onnx") == []


def test_the_face_feather_holds_a_400_px_middle_and_falls_gently():
    feather = vosr2_scene.face_feather()[0, 0]
    torch.testing.assert_close(feather, feather.flip(0, 1), rtol=0, atol=0)
    torch.testing.assert_close(feather, feather.T, rtol=0, atol=0)
    row = feather[32]
    held = torch.nonzero(row == 1).flatten()
    assert (held.min(), held.max()) == (7, 56)  # 50 latents
    assert row[0] == row[63] == 0
    # A hand-over between two grid windows changes their shares by up to 0.33 a latent.
    assert row.diff().abs().max() < 0.27


# Grid windows 1 to 4, and face windows 5 and 6 on the first and third seams.
FACE_LAYOUT = vosr2_scene.Layout(
    [(0, 0), (0, 56), (0, 112), (0, 168)], [(0, 28), (0, 140)]
)


def assert_face_windows_take_their_middle(velocity: np.ndarray) -> None:
    """`velocity` along latent row 32, where each window predicted its number."""
    # Each face window alone holds 50 latents, then hands over to the grid within
    # FEATHER latents of either end.
    np.testing.assert_allclose(velocity[35:85], 5, atol=1e-4)
    np.testing.assert_allclose(velocity[147:197], 6, atol=1e-4)
    for x, value in ((20, 1), (28, 1), (91, 2), (100, 2), (140, 3), (203, 4)):
        assert velocity[x] == pytest.approx(value, abs=1e-4)
    # No hard edge: a share changes by at most 0.26 a latent (here of up to 4).
    assert np.abs(np.diff(velocity)).max() < 0.27 * 4


@torch.inference_mode()
def test_face_windows_take_their_middle_and_hand_over_smoothly(monkeypatch):
    class Dino(torch.nn.Module):
        def forward_intermediate_layer(self, image, layer):
            return torch.zeros(1, 1024, 2)

    class Dit(torch.nn.Module):
        window = 0

        def forward_flexible(self, inputs, time, **kwargs):
            self.window += 1
            return torch.full_like(inputs[:, 16:], float(self.window))

    monkeypatch.setattr(
        vosr2_context,
        "scene_routing",
        lambda pixels, windows: (FACE_LAYOUT, (0, 0), [0.0] * 6),
    )
    bundle = cast(
        VosrBundle, SimpleNamespace(dino=Dino(), dit=Dit(), dtype=torch.float32)
    )
    result = context_denoise(
        bundle,
        torch.zeros(1, 16, 64, 232),
        torch.zeros(1, 3, 512, 1856),
        torch.device("cpu"),
        0,
        MemoryPlan.for_mode(VosrMemoryMode.DEFAULT),
        Progress.noop_progress(),
        lambda: None,
    )
    noise = torch.randn(result.shape, generator=torch.Generator().manual_seed(0))
    assert_face_windows_take_their_middle((noise - result)[0, 0, 32].numpy())


@torch.inference_mode()
def test_tensorrt_blends_a_face_window_the_same_way(monkeypatch):
    class Session:
        device = torch.device("cpu")
        stream = None

        def __init__(self):
            self.constants = {}
            self.window = 0
            self.decoder_input = torch.empty(0)
            # One output buffer, as TensorRT's: rewritten by every prediction.
            self.output = torch.empty(1, 16, 64, 64)

        def begin_image(self, height, width, seed):
            pass

        def activate(self, check):
            pass

        def constant(self, key, value):
            return self.constants.setdefault(key, value)

        def vae(self, name, image, check):
            if name == "encoder":
                return torch.zeros(1, 16, image.shape[-2] // 8, image.shape[-1] // 8)
            self.decoder_input = image
            return torch.zeros(1, 3, image.shape[-2] * 8, image.shape[-1] * 8)

        def begin_window(self):
            self.window += 1

        def condition(self, image):
            return torch.zeros(1)

        def predict(self, latent, condition):
            return self.output.fill_(float(self.window))

    monkeypatch.setattr(
        vosr2_trt_restore,
        "scene_routing",
        lambda pixels, windows: (FACE_LAYOUT, (0, 0), [0.0] * 6),
    )
    # No CUDA events on the CPU.
    monkeypatch.setattr(
        vosr2_trt_restore,
        "BoundedQueue",
        lambda: SimpleNamespace(advance=lambda check: None),
    )
    session = Session()
    vosr2_trt_restore._restore_tile(
        vosr2_trt_restore.LatentStatistics(
            torch.zeros(1, 16, 1, 1), torch.ones(1, 16, 1, 1)
        ),
        torch.zeros(1, 3, 512, 1856),
        np.zeros((512, 1856, 3), np.float32),
        0,
        VosrColorCorrection.NONE,
        Progress.noop_progress(),
        lambda: None,
        cast(vosr2_trt_restore.FullSession, session),
        None,
    )
    latent = session.decoder_input
    noise = torch.randn(latent.shape, generator=torch.Generator().manual_seed(0))
    assert_face_windows_take_their_middle((noise - latent)[0, 0, 32].numpy())
