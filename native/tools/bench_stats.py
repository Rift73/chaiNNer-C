"""Paired-trial statistics, A/A noise floors and verdicts for benchmark runs.

A pair runs one case once on tree A and once on tree B, back to back. Its ratio
is seconds_A / seconds_B, so values above 1 mean B was faster. Consecutive pairs
run in opposite orders, so their geometric mean (a quad) cancels the first-vs-
second position effect. A case's noise floor for one launch is the largest
|ln(quad)| observed when A and B are identical trees.

Throughput verdicts take two floors: the shared noise floor, which flags a
change, and the current floor: a win must beat the larger of the two, and a
regression must exceed the current floor to block (see bench.py, Noise floors).

CPU per item is paired the same way: a pair's CPU ratio is A's job CPU seconds per
item over B's, so values above 1 mean B used less CPU. The same quads, floors and
verdicts apply to it, with its CPU floor alone; its verdicts are information only.
"""

from __future__ import annotations

import math
import statistics

# A regression beyond the shared floor but not beyond the current floor: flagged
# and confirmed like a regression, but it does not block.
WITHIN_CURRENT = "regression (within current A/A)"


def paired(ratios: list[float]) -> dict:
    """The paired-ratio fields that verdict() and noise_floors() read."""
    return {
        "paired_ratios": ratios,
        "median_ratio": statistics.median(ratios),
        "pairs": len(ratios),
        "b_faster_pairs": sum(r > 1 for r in ratios),
        "b_slower_pairs": sum(r < 1 for r in ratios),
    }


def summarize(trials: list[dict], labels: tuple[str, str]) -> dict[str, dict]:
    """Summarize measured (pair > 0) trials per case, pairing A and B by index.

    A case's "cpu" pairs CPU per item (job_cpu_seconds / count) like seconds. It is
    None unless every measured trial of the case has job_cpu_seconds > 0, so a
    missing or zero job CPU reading never yields a CPU verdict.
    """
    a, b = labels
    result = {}
    for case in dict.fromkeys(t["case"] for t in trials):
        measured = [t for t in trials if t["case"] == case and t["pair"] > 0]
        runs = {
            label: sorted(
                (t for t in measured if t["backend"] == label),
                key=lambda t: t["pair"],
            )
            for label in labels
        }
        pairs = [t["pair"] for t in runs[a]]
        if not pairs or pairs != [t["pair"] for t in runs[b]]:
            continue  # Never compare unpaired trials.
        ratios = [
            x["seconds"] / y["seconds"] for x, y in zip(runs[a], runs[b], strict=True)
        ]
        count = runs[a][0]["count"]
        medians = {
            label: statistics.median(t["seconds"] for t in runs[label])
            for label in labels
        }
        cpu = None
        if all((t.get("job_cpu_seconds") or 0) > 0 for t in measured):
            per_item = {
                label: [t["job_cpu_seconds"] / t["count"] for t in runs[label]]
                for label in labels
            }
            cpu = {
                **paired(
                    [x / y for x, y in zip(per_item[a], per_item[b], strict=True)]
                ),
                "median_cpu_per_item": {
                    label: statistics.median(per_item[label]) for label in labels
                },
            }
        result[case] = {
            "unit": runs[a][0]["unit"],
            "items_per_request": count,
            "median_seconds": medians,
            "median_throughput": {label: count / medians[label] for label in labels},
            **paired(ratios),
            "cpu": cpu,
        }
    return result


def quad_ratios(ratios: list[float]) -> list[float]:
    """Geometric means of consecutive, non-overlapping pairs of paired ratios.

    Consecutive pairs run in opposite orders (pair 1 B-first, pair 2 A-first, and
    so on), so each geometric mean cancels the first-vs-second position effect.
    """
    if len(ratios) % 2:
        raise ValueError(
            f"Position balance needs an even number of ratios, got {len(ratios)}"
        )
    return [math.sqrt(ratios[i] * ratios[i + 1]) for i in range(0, len(ratios), 2)]


def noise_floors(summary: dict[str, dict]) -> dict[str, float]:
    """Per case, one launch's floor: the largest |ln(quad)| of an A/A comparison.

    Quads are position-balanced, so the floor holds only the noise left after the
    first-vs-second position effect cancels. Needs an even number of pairs. The
    caller keeps several launches and takes the maximum (see bench.merge_floors).
    """
    return {
        case: max(abs(math.log(q)) for q in quad_ratios(item["paired_ratios"]))
        for case, item in summary.items()
    }


def verdict(item: dict, floor: float | None, current: float | None = None) -> str:
    """Classify a case as win, regression, WITHIN_CURRENT, neutral or unknown.

    The two directions are deliberately asymmetric. A win needs the median shift
    beyond the noise floor AND a majority of pairs faster. A regression needs only
    the median shift beyond the floor, so a bimodal slowdown cannot slip past a gate.
    Without a noise floor the verdict is unknown. current is the current floor:
    a win must also beat it, and a regression not beyond it is WITHIN_CURRENT.
    Without one, the noise floor alone decides.
    """
    if floor is None:
        return "unknown"
    shift = math.log(item["median_ratio"])
    upper = floor if current is None else max(floor, current)
    if shift > upper and item["b_faster_pairs"] > item["pairs"] / 2:
        return "win"
    if shift < -floor:
        return "regression" if current is None or shift < -current else WITHIN_CURRENT
    return "neutral"


def bounds(floor: float | None) -> str:
    """A floor f as the paired changes it corresponds to in each direction.

    +expm1(f) upward and -(1 - exp(-f)) downward, e.g. "+8.4/-7.8%"; "n/a" without
    a floor.
    """
    if floor is None:
        return "n/a"
    return f"+{100 * math.expm1(floor):.1f}/-{-100 * math.expm1(-floor):.1f}%"


def table(
    summary: dict[str, dict],
    labels: tuple[str, str],
    floors: dict[str, float],
    current_floors: dict[str, float],
    cpu_floors: dict[str, float],
) -> str:
    """Markdown: per case the throughput columns, then the CPU-per-item columns.

    Throughput: median throughputs, median paired change, agreement, noise floor,
    current floor and the verdict of both floors. CPU per item: median paired
    change, its floor (cpu_floors) and its verdict. Floors are shown as bounds(). A
    case without a CPU summary shows n/a for its CPU change and floor, and an
    unknown CPU verdict.
    """
    a, b = labels
    rows = [
        (
            f"| Case | Unit | {a} | {b} | Paired change | {b} faster | Noise floor "
            "| Current floor | Verdict | CPU/item change | CPU floor | CPU verdict |"
        ),
        (
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: "
            "| --- |"
        ),
    ]
    for case, item in summary.items():
        floor = floors.get(case)
        current = current_floors.get(case)
        cpu = item["cpu"]
        cpu_floor = cpu_floors.get(case)
        cpu_cells = (
            "n/a | n/a | unknown"
            if cpu is None
            else f"{100 * (cpu['median_ratio'] - 1):+.1f}% | {bounds(cpu_floor)} "
            f"| {verdict(cpu, cpu_floor)}"
        )
        rows.append(
            f"| {case} | {item['unit']} "
            f"| {item['median_throughput'][a]:.1f} "
            f"| {item['median_throughput'][b]:.1f} "
            f"| {100 * (item['median_ratio'] - 1):+.1f}% "
            f"| {item['b_faster_pairs']}/{item['pairs']} | {bounds(floor)} "
            f"| {bounds(current)} | {verdict(item, floor, current)} | {cpu_cells} |"
        )
    return "\n".join(rows)
