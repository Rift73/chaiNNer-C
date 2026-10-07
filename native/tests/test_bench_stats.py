from __future__ import annotations

import math
import random
import statistics

import bench_stats
import pytest

LABELS = ("A", "B")


def trial(case, label, pair, seconds):
    return {
        "case": case,
        "backend": label,
        "pair": pair,
        "seconds": seconds,
        "count": 32,
        "unit": "images/s",
    }


def trials_from(ratios, case="g"):
    """A takes `ratio` seconds and B takes 1 second in each measured pair."""
    trials = [trial(case, "A", 0, 5.0), trial(case, "B", 0, 5.0)]
    for pair, ratio in enumerate(ratios, start=1):
        trials += [trial(case, "A", pair, ratio), trial(case, "B", pair, 1.0)]
    return trials


def cpu_trials(pairs, case="g"):
    """Both trees take 1 second; A and B use the given job CPU seconds per pair.

    The warm-ups record no job CPU: they are not measured, so they never decide
    whether a case has a CPU summary.
    """
    trials = [
        {**trial(case, label, 0, 1.0), "job_cpu_seconds": 0.0} for label in LABELS
    ]
    for pair, (a, b) in enumerate(pairs, start=1):
        trials += [
            {**trial(case, "A", pair, 1.0), "job_cpu_seconds": a},
            {**trial(case, "B", pair, 1.0), "job_cpu_seconds": b},
        ]
    return trials


def test_summarize_pairs_by_index_even_past_nine_pairs():
    trials = [trial("g", "A", 0, 9.0), trial("g", "B", 0, 9.0)]
    for pair in range(1, 13):
        trials += [
            trial("g", "A", pair, 2.0 + pair),
            trial("g", "B", pair, (2.0 + pair) / 2),
        ]
    random.Random(1).shuffle(trials)
    item = bench_stats.summarize(trials, LABELS)["g"]
    assert item["pairs"] == 12
    assert item["paired_ratios"] == pytest.approx([2.0] * 12)
    assert item["b_faster_pairs"] == 12
    assert item["median_throughput"]["B"] == pytest.approx(
        2 * item["median_throughput"]["A"]
    )


def test_summarize_ignores_warmups():
    item = bench_stats.summarize(trials_from([1.0, 1.0]), LABELS)["g"]
    assert item["pairs"] == 2
    assert item["median_seconds"]["A"] == pytest.approx(1.0)


def test_summarize_skips_incomplete_case():
    trials = [trial("g", "A", 1, 1.0), trial("g", "B", 1, 1.0), trial("g", "A", 2, 1.0)]
    assert bench_stats.summarize(trials, LABELS) == {}


def test_summarize_pairs_cpu_per_item_like_seconds():
    item = bench_stats.summarize(cpu_trials([(4.0, 2.0)] * 6), LABELS)["g"]
    # A's CPU per item over B's: above 1 means B used less CPU.
    assert item["cpu"]["paired_ratios"] == pytest.approx([2.0] * 6)
    assert item["cpu"]["median_ratio"] == pytest.approx(2.0)
    assert (item["cpu"]["pairs"], item["cpu"]["b_faster_pairs"]) == (6, 6)
    assert item["cpu"]["b_slower_pairs"] == 0
    assert item["cpu"]["median_cpu_per_item"] == pytest.approx(
        {"A": 0.125, "B": 0.0625}
    )
    assert item["median_ratio"] == pytest.approx(1.0)  # Throughput is unchanged.


@pytest.mark.parametrize("cpu", [None, 0.0])  # key missing, or a zero
def test_summarize_has_no_cpu_summary_without_job_cpu(cpu):
    trials = cpu_trials([(4.0, 2.0)] * 6)
    (odd,) = (t for t in trials if t["backend"] == "B" and t["pair"] == 3)
    if cpu is None:
        del odd["job_cpu_seconds"]
    else:
        odd["job_cpu_seconds"] = cpu
    item = bench_stats.summarize(trials, LABELS)["g"]
    assert item["cpu"] is None
    assert item["pairs"] == 6  # The throughput summary still stands.


def test_quad_ratios_are_geometric_means_of_consecutive_pairs():
    quads = bench_stats.quad_ratios([1.04, 0.96, 1.02, 0.98])
    assert quads == pytest.approx(
        [
            statistics.geometric_mean([1.04, 0.96]),
            statistics.geometric_mean([1.02, 0.98]),
        ]
    )
    assert bench_stats.quad_ratios([]) == []


@pytest.mark.parametrize("count", [1, 3, 5])
def test_quad_ratios_refuse_an_odd_count(count: int):
    with pytest.raises(ValueError, match=f"even number of ratios, got {count}"):
        bench_stats.quad_ratios([1.0] * count)


def test_noise_floor_is_largest_absolute_log_of_the_position_balanced_means():
    # Quads: sqrt(1.02 * 0.96), sqrt(1.10 * 1.00), sqrt(0.99 * 1.00). The middle
    # one is the largest, so the floor is neither the first nor the last quad.
    summary = bench_stats.summarize(
        trials_from([1.02, 0.96, 1.10, 1.00, 0.99, 1.00]), LABELS
    )
    assert bench_stats.noise_floors(summary)["g"] == pytest.approx(0.5 * math.log(1.10))


def test_noise_floor_cancels_a_pure_position_effect():
    # B-first pairs read 1.05 and A-first pairs 0.95: only the order differs.
    ratios = [1.05, 0.95] * 3
    summary = bench_stats.summarize(trials_from(ratios), LABELS)
    assert bench_stats.noise_floors(summary)["g"] < 0.002
    # The rule this replaces would have called the same launch ~5% noisy.
    assert max(abs(math.log(r)) for r in ratios) == pytest.approx(0.051, abs=1e-3)


def test_noise_floor_pairs_neighbours_in_launch_order():
    # (1.05, 0.95) and (0.95, 1.05) each cancel; pairing (0.95, 0.95) would not.
    summary = bench_stats.summarize(trials_from([1.05, 0.95, 0.95, 1.05]), LABELS)
    assert bench_stats.noise_floors(summary)["g"] == pytest.approx(
        0.5 * abs(math.log(1.05 * 0.95))
    )


def test_noise_floor_keeps_a_uniform_launch_bias():
    summary = bench_stats.summarize(trials_from([1.03] * 6), LABELS)
    assert bench_stats.noise_floors(summary)["g"] == pytest.approx(math.log(1.03))


def test_noise_floor_refuses_an_odd_number_of_pairs():
    summary = bench_stats.summarize(trials_from([1.0] * 5), LABELS)
    with pytest.raises(ValueError, match="even number of ratios"):
        bench_stats.noise_floors(summary)


@pytest.mark.parametrize(
    ("ratios", "floor", "expected"),
    [
        ([1.2, 1.3, 1.25, 1.1, 1.2, 1.3], 0.05, "win"),
        ([0.8, 0.85, 0.9, 0.8, 0.82, 0.88], 0.05, "regression"),
        ([0.6, 0.6, 0.6, 1.01, 1.01, 1.01], 0.05, "regression"),
        ([0.98] * 6, 0.05, "neutral"),
        ([1.03, 1.02, 1.04, 1.03, 1.01, 1.02], 0.05, "neutral"),
        ([1.5, 1.5, 1.5, 0.9, 0.9, 0.9], 0.05, "neutral"),
        ([1.2] * 6, None, "unknown"),
    ],
)
def test_verdict(ratios, floor, expected):
    item = bench_stats.summarize(trials_from(ratios), LABELS)["g"]
    assert bench_stats.verdict(item, floor) == expected


@pytest.mark.parametrize(
    ("ratio", "shared", "current", "expected"),
    [
        (1.04, 0.02, 0.06, "neutral"),  # Beyond the shared floor only.
        (1.07, 0.02, 0.06, "win"),
        (1.04, 0.06, 0.02, "neutral"),  # The shared floor is the larger one.
        (1.04, 0.02, None, "win"),  # Without a current floor, today's rule.
    ],
)
def test_win_must_beat_the_larger_floor(ratio, shared, current, expected):
    item = bench_stats.summarize(trials_from([ratio] * 6), LABELS)["g"]
    assert bench_stats.verdict(item, shared, current) == expected


@pytest.mark.parametrize(
    ("ratio", "shared", "current", "expected"),
    [
        (0.96, 0.02, 0.06, "regression (within current A/A)"),  # Between the floors.
        (0.93, 0.02, 0.06, "regression"),  # Beyond both.
        (0.96, 0.02, None, "regression"),  # Without a current floor, today's rule.
        (0.97, 0.02, 0.01, "regression"),  # A current floor below the shared one.
        (0.96, 0.06, 0.02, "neutral"),  # Flagged only beyond the shared floor.
    ],
)
def test_regression_is_flagged_by_the_shared_floor_and_blocks_beyond_the_current(
    ratio, shared, current, expected
):
    item = bench_stats.summarize(trials_from([ratio] * 6), LABELS)["g"]
    assert bench_stats.verdict(item, shared, current) == expected
    assert bench_stats.WITHIN_CURRENT == "regression (within current A/A)"


def test_table_marks_missing_floor_unknown():
    summary = bench_stats.summarize(trials_from([1.2] * 6), LABELS)
    text = bench_stats.table(summary, LABELS, {}, {}, {})
    assert "| g | images/s |" in text
    assert "| 6/6 | n/a | n/a | unknown | n/a | n/a | unknown |" in text
    assert "+20.0%" in text


def test_table_renders_numeric_floor_as_both_bounds():
    summary = bench_stats.summarize(trials_from([1.02] * 6), LABELS)
    text = bench_stats.table(summary, LABELS, {"g": 0.08103464423609148}, {}, {})
    # Upward expm1(f) = +8.44%; downward -(1 - exp(-f)) = -7.78%.
    assert "| 6/6 | +8.4/-7.8% | n/a | neutral | n/a | n/a | unknown |" in text


def test_table_shows_the_current_floor_column():
    # +5.5% beats the shared floor (+5.1%) but not the current floor (+6.2%).
    summary = bench_stats.summarize(trials_from([1.055] * 6), LABELS)
    text = bench_stats.table(summary, LABELS, {"g": 0.05}, {"g": 0.06}, {})
    assert text.endswith(
        "| g | images/s | 30.3 | 32.0 | +5.5% | 6/6 | +5.1/-4.9% | +6.2/-5.8% "
        "| neutral | n/a | n/a | unknown |"
    )
    text = bench_stats.table(summary, LABELS, {"g": 0.05}, {}, {})
    assert text.endswith(
        "| +5.5% | 6/6 | +5.1/-4.9% | n/a | win | n/a | n/a | unknown |"
    )


def test_table_shows_cpu_change_floor_and_verdict():
    header = (
        "| Case | Unit | A | B | Paired change | B faster | Noise floor "
        "| Current floor | Verdict | CPU/item change | CPU floor | CPU verdict |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: "
        "| --- |\n"
    )
    summary = bench_stats.summarize(cpu_trials([(4.0, 2.0)] * 6), LABELS)
    text = bench_stats.table(summary, LABELS, {}, {}, {"g": 0.05})
    assert text.startswith(header)
    assert text.endswith(
        "| g | images/s | 32.0 | 32.0 | +0.0% | 0/6 | n/a | n/a | unknown "
        "| +100.0% | +5.1/-4.9% | win |"
    )
    # Without a CPU summary there is no CPU change, floor or verdict to show.
    summary = bench_stats.summarize(trials_from([1.2] * 6), LABELS)
    text = bench_stats.table(summary, LABELS, {}, {}, {"g": 0.05})
    assert text.endswith(
        "| g | images/s | 26.7 | 32.0 | +20.0% | 6/6 | n/a | n/a | unknown "
        "| n/a | n/a | unknown |"
    )
