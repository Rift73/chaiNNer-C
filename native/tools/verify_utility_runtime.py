"""Verify all twelve utility nodes through isolated oracle/portable HTTP hosts.

This exercises registered defaults and options, output broadcasts, error events,
exact rendered PNGs, and repeated execution. No GPU, models, or benchmark runs.
The default validates only the oracle's fixtures (verify_runtime.oracle_source),
on the provisioned runtime the package was copied from; --include-port compares
the released package, on the package's runtime.
Both original backend trees and user profiles stay untouched.
"""

from __future__ import annotations

import argparse
import json
import shutil
import traceback
from datetime import UTC, datetime
from pathlib import Path

from verify_framework_runtime import compare_runs, run_backend, source_hashes
from verify_runtime import (
    METADATA_COMPARISON,
    ORACLE,
    PROJECT,
    SSE_EVENT_COMPARISON,
    edge,
    make_node,
    oracle_repeat_flags,
    oracle_source,
    write_json,
)

UTILITY_IDS = {
    "chainner:utility:" + name
    for name in (
        "back_directory",
        "math",
        "math_round",
        "derive_seed",
        "random_number",
        "regex_find",
        "regex_replace",
        "text_padding",
        "text_pattern",
        "text_replace",
        "text_slice",
        "parse_number",
    )
}


def fixture_graphs(schemas: dict, _assets: Path, output: Path):
    def node(name: str, schema: str, values: dict):
        return make_node(schemas, name, schema, values)

    def case(
        name: str, schema: str, values: dict, *, error: bool = False, seed: bool = False
    ):
        nodes = [node("utility", "chainner:utility:" + schema, values)]
        result = "utility"
        if seed:
            nodes.append(
                node(
                    "random",
                    "chainner:utility:random_number",
                    {0: -100000, 1: 100000, 2: edge(result)},
                )
            )
            result = "random"
        nodes.extend(
            [
                node(
                    "pattern",
                    "chainner:utility:text_pattern",
                    {0: "Result: {1}", 1: edge(result)},
                ),
                node(
                    "render",
                    "chainner:image:text_as_image",
                    {0: edge("pattern"), 5: 320, 6: 64},
                ),
                node(
                    "save",
                    "chainner:image:save",
                    {
                        0: edge("render"),
                        1: str(output / name),
                        2: None,
                        3: "image",
                        4: "png",
                        15: "u8",
                    },
                ),
            ]
        )
        return {
            "name": name,
            "nodes": nodes,
            "expected_error": error,
            "files": [] if error else ["image.png"],
        }

    for operation in (
        "add",
        "sub",
        "mul",
        "div",
        "pow",
        "log",
        "max",
        "min",
        "mod",
        "percent",
    ):
        yield case("math-" + operation, "math", {0: 2.5, 1: operation, 2: 7.75})
    for name, values in (
        ("divide-zero", {0: 1, 1: "div", 2: 0}),
        ("power-complex", {0: -1, 1: "pow", 2: 0.5}),
        ("log-domain", {0: 1, 1: "log", 2: 2}),
    ):
        yield case("math-" + name, "math", values, error=True)
    for operation in ("Round down", "Round up", "Round"):
        for scale in ("Integer", "Multiple of...", "Power of..."):
            for value in (-2.5, 7.5) if scale != "Power of..." else (7.5,):
                name = f"round-{operation}-{scale}-{value}".replace(" ", "-").replace(
                    "...", ""
                )
                yield case(
                    name,
                    "math_round",
                    {0: value, 1: operation, 2: scale, 3: 0.25, 4: 2},
                )
    for base in range(2, 37):
        yield case(f"parse-base-{base}", "parse_number", {0: " -1_01 ", 1: base})
    yield case("parse-unicode", "parse_number", {0: "\u2003+１２３٤\u2003", 1: 10})
    yield case("parse-prefix", "parse_number", {0: "0xFA_CE", 1: 16})
    yield case("parse-invalid", "parse_number", {0: "123z", 1: 10}, error=True)
    for name, values in (
        ("default", {}),
        ("equal", {0: 7, 1: 7, 2: 73}),
        ("negative", {0: -1025, 1: -1, 2: -79}),
        ("cross-zero", {0: -1000000, 1: 1000000, 2: 123456789}),
    ):
        yield case("random-" + name, "random_number", values)
    for name, values in (
        ("none", {0: 123}),
        ("text", {0: 123, 1: "é🙂𐐀"}),
        ("holes", {0: 123, 2: "second", 9: "ninth"}),
        ("numbers", {0: 123, 1: -128, 2: 1.25, 3: 0}),
        ("order-a", {0: 123, 1: "a", 2: "bc"}),
        ("order-b", {0: 123, 1: "bc", 2: "a"}),
    ):
        yield case("derive-" + name, "derive_seed", values, seed=True)
    for alignment in ("start", "end", "center"):
        for width in (2, 12):
            yield case(
                f"padding-{alignment}-{width}",
                "text_padding",
                {0: "é🙂ab", 1: width, 2: "_", 3: alignment},
            )
    for mode in (0, 1):
        yield case(
            f"replace-{mode}", "text_replace", {0: "é🙂é🙂é", 1: "é", 2: "字", 3: mode}
        )
        yield case(
            f"replace-empty-{mode}",
            "text_replace",
            {0: "abc abc", 1: "abc", 2: "", 3: mode},
        )
    for name, values in (
        ("holes", {0: "{9}|{1}|{3}|{1}", 1: "é🙂", 3: "third", 9: "ninth"}),
        ("escaped", {0: "{{x}}={1}; unmatched { tail", 1: "value"}),
    ):
        yield case("pattern-" + name, "text_pattern", values)
    yield case("pattern-missing", "text_pattern", {0: "{2}", 1: "one"}, error=True)
    for mode in (0, 1, 2):
        for start in (-20, -2, 2):
            for alignment in ("start", "end") if mode == 2 else ("start",):
                yield case(
                    f"slice-{mode}-{start}-{alignment}",
                    "text_slice",
                    {0: "é🙂abcdef𐐀", 1: mode, 2: start, 3: 3, 4: 4, 5: alignment},
                )
    yield case("slice-zero", "text_slice", {0: "abcd", 1: 2, 4: 0})
    for mode in (0, 1):
        yield case(
            f"regex-find-{mode}",
            "regex_find",
            {
                0: "prefix é🙂42 and é🙂99",
                1: "(?P<letter>é)🙂(\\d+)",
                2: mode,
                3: "{letter}|{2}|{0}",
            },
        )
        yield case(
            f"regex-replace-{mode}",
            "regex_replace",
            {
                0: "é🙂42 é🙂99",
                1: "(?P<letter>é)🙂(\\d+)",
                2: "[{letter}:{2}]",
                3: mode,
            },
        )
        yield case(
            f"regex-zero-width-{mode}",
            "regex_replace",
            {0: "é🙂ab", 1: "^|$", 2: "_", 3: mode},
        )
    yield case(
        "regex-optional", "regex_replace", {0: "b ab", 1: "(a)?b", 2: "[{1}]", 3: 0}
    )
    yield case(
        "regex-no-match", "regex_replace", {0: "abcdef", 1: "z+", 2: "{0}", 3: 0}
    )
    yield case(
        "regex-find-no-match", "regex_find", {0: "abcdef", 1: "z+", 2: 0}, error=True
    )
    yield case(
        "regex-invalid",
        "regex_replace",
        {0: "abc", 1: "(?=a)", 2: "x", 3: 0},
        error=True,
    )
    yield case(
        "regex-invalid-replacement",
        "regex_replace",
        {0: "abc", 1: "z+", 2: "{}", 3: 0},
        error=True,
    )
    for amount in (0, 1, 2):
        name = f"directory-up-{amount}"
        directory = output / name
        source = directory.joinpath(*("child" for _ in range(amount)))
        yield {
            "name": name,
            "nodes": [
                node(
                    "utility",
                    "chainner:utility:back_directory",
                    {0: str(source), 1: amount},
                ),
                node(
                    "render",
                    "chainner:image:text_as_image",
                    {0: "Directory result", 5: 320, 6: 64},
                ),
                node(
                    "save",
                    "chainner:image:save",
                    {
                        0: edge("render"),
                        1: edge("utility"),
                        2: None,
                        3: "image",
                        4: "png",
                        15: "u8",
                    },
                ),
            ],
            "expected_error": False,
            "files": ["image.png"],
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=PROJECT / "out/chaiNNer-C")
    parser.add_argument("--include-port", action="store_true")
    parser.add_argument("--startup-timeout", type=float, default=180)
    args = parser.parse_args()
    package = args.package.resolve(strict=True)
    manifest = json.loads(
        (package / "chainner-c-package.json").read_text(encoding="utf-8")
    )
    if (
        manifest.get("state") != "complete"
        or manifest["identity"]["destination"] != str(package)
        or manifest["identity"]["project"] != str(PROJECT)
    ):
        raise ValueError(
            "A completed package manifest matching this project/location is required"
        )
    python = package / "python/python/python.exe"
    if not python.is_file() or not (package / "portable").is_file():
        raise ValueError("Independent portable Python runtime missing")
    source, oracle_python, oracle = oracle_source(manifest, ORACLE)
    sources = {"baseline": source}
    pythons = {"baseline": oracle_python, "converted": python}
    if args.include_port:
        sources["converted"] = package / "resources/src"
    root = (
        PROJECT
        / "native/reports"
        / ("utility-runtime-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    )
    root.mkdir(parents=True, exist_ok=False)
    protected = {label: source_hashes(path) for label, path in sources.items()}
    report: dict = {
        "utc": datetime.now(UTC).isoformat(),
        "package": str(package),
        "oracle": oracle,
        "success": False,
        "mode": "oracle-versus-port"
        if args.include_port
        else "baseline-fixture-validation-only",
        "cpu_only": True,
        "performance_benchmark": False,
        "model_inference": False,
        "limitations": [
            "These HTTP fixtures exercise registered inputs. Independent frozen-oracle unit tests cover direct helper protocols and invalid values outside schemas.",
            "The installed Regex Find contract has one text output and raises on no match; the clone's different two-output contract is covered separately.",
            "Independent graph events are compared without order, timestamps or execution durations, by the rule that follows.",
            SSE_EVENT_COMPARISON,
            METADATA_COMPARISON,
            "Text glyph rasterization retains the installed font engines; every decoded PNG component must match exactly.",
        ],
        "runs": [],
    }
    try:
        for label, source in sources.items():
            copied = root / (label + "-src")
            shutil.copytree(
                source, copied, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
            )
            report["runs"].append(
                run_backend(
                    label,
                    copied,
                    pythons[label],
                    root,
                    args.startup_timeout,
                    required_schema_ids=UTILITY_IDS,
                    graphs=fixture_graphs,
                    synthetic_model_inference=False,
                )
            )
        if args.include_port:
            report["comparison"] = compare_runs(
                report["runs"][0], report["runs"][1], python
            )
            report["success"] = report["comparison"]["success"]
        else:
            report["success"] = report["runs"][0]["success"]
    except Exception as error:
        report["error"], report["traceback"] = str(error), traceback.format_exc()
        report["runs"] = [
            json.loads(path.read_text(encoding="utf-8"))
            for label in sources
            if (path := root / label / "result.json").is_file()
        ]
    finally:
        report["original_source_trees_unchanged"] = {
            label: protected[label] == source_hashes(path)
            for label, path in sources.items()
        }
        report["all_owned_processes_stopped"] = all(
            run.get("owned_process_exited") and run.get("owned_job_closed")
            for run in report["runs"]
        )
        report["success"] = (
            report["success"]
            and all(report["original_source_trees_unchanged"].values())
            and report["all_owned_processes_stopped"]
        )
        # Flagged, not failed (Consult D-35): the oracle's own attempts ended on
        # different final values of an iterated field.
        report["oracle_repeat_final_values_differ"] = (
            oracle_repeat_flags(report["runs"][0]) if report["runs"] else {}
        )
        write_json(root / "report.json", report)
        print(
            json.dumps(
                {
                    "report": str(root / "report.json"),
                    "success": report["success"],
                    "error": report.get("error"),
                    "oracle_repeat_final_values_differ": sorted(
                        report["oracle_repeat_final_values_differ"]
                    ),
                }
            ),
            flush=True,
        )
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
