"""The AMD kit's report lines (native/AMD.md), as Markdown on stdout:

  packages PACKAGES NODES INSTALLED   each package's state after the dependency manager,
                                      its node count and its dependencies' installed
                                      versions (the backend's /packages, /nodes and
                                      /installed-dependencies)
  tests EXPECTED_FILES JUNIT_XML...   pass/fail counts, failing and skipped tests (on an
                                      AMD machine a skip is a finding), and the expected
                                      files no test came from
  chains ROWS_TSV LOGS                each chain's result and time, its output's size
                                      and PSNR against its reference output, and the
                                      last error lines of each failed chain's log

Runs in the app's Python (NumPy and OpenCV come with it)."""

from __future__ import annotations

import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def packages(packages_json: Path, nodes_json: Path, installed_json: Path) -> None:
    listed = json.loads(packages_json.read_text(encoding="utf-8"))
    installed = json.loads(installed_json.read_text(encoding="utf-8"))
    schema_ids = [
        node["schemaId"]
        for node in json.loads(nodes_json.read_text(encoding="utf-8"))["nodes"]
    ]
    print("| Package | State | Nodes | Dependencies (installed version) |")
    print("|---|---|---|---|")
    for package in listed:
        if not package["dependencies"]:
            continue
        prefix = f"chainner:{package['id'].removeprefix('chaiNNer_')}:"
        count = sum(schema_id.startswith(prefix) for schema_id in schema_ids)
        state = (
            f"disabled: {package['disabledReason']}"
            if package["disabled"]
            else "enabled"
        )
        dependencies = ", ".join(
            f"{d['pypiName']} {installed.get(d['pypiName'], 'MISSING')}"
            for d in package["dependencies"]
        )
        print(f"| {package['name']} | {state} | {count} | {dependencies} |")


def test_id(case: ET.Element) -> tuple[str, str]:
    """A JUnit test case's file and pytest ID. Its classname is the module's dotted path,
    then any test class; a module that failed or skipped at collection has no classname
    and its dotted path as name."""
    classname = case.get("classname", "")
    dotted, test = (
        (classname, case.get("name", ""))
        if classname
        else (
            case.get("name", ""),
            "(collection)",
        )
    )
    parts = dotted.split(".")
    module = next(
        (i for i, part in enumerate(parts) if part.startswith("test_")), len(parts) - 1
    )
    file = "/".join(parts[: module + 1]) + ".py"
    return file, "::".join([file, *parts[module + 1 :], test])


def tests(expected: Path, reports: list[Path]) -> int:
    counts = {"passed": 0, "failed": 0, "error": 0, "skipped": 0}
    problems: list[str] = []
    skips: list[str] = []
    files: set[str] = set()
    for report in reports:
        if not report.is_file():
            problems.append(f"- {report.name}: no report (pytest did not finish)")
            continue
        for case in ET.parse(report).getroot().iter("testcase"):
            file, name = test_id(case)
            files.add(file)
            outcome = "passed"
            message = ""
            for child in case:
                if child.tag in ("failure", "error", "skipped"):
                    outcome = {"failure": "failed"}.get(child.tag, child.tag)
                    message = (child.get("message") or child.text or "").strip()
                    message = message.splitlines()[0][:200] if message else ""
                    break
            counts[outcome] += 1
            if outcome in ("failed", "error"):
                problems.append(f"- {outcome.upper()} {name}: {message}")
            elif outcome == "skipped":
                skips.append(f"- {name}: {message}")
    print(
        f"{counts['passed']} passed, {counts['failed']} failed, {counts['error']}"
        f" errors, {counts['skipped']} skipped"
    )
    if problems:
        print("\nFailures and errors:\n")
        print("\n".join(problems))
    if skips:
        print("\nSkipped (each is a check that did not run on this AMD machine):\n")
        print("\n".join(skips))
    missing = [
        name
        for name in expected.read_text(encoding="utf-8").split()
        if name not in files
    ]
    if missing:
        print("\nExpected test files that gave no amd_gpu test:\n")
        print("\n".join(f"- {name}" for name in missing))
    return 1 if problems or missing else 0


def psnr(reference: np.ndarray, output: np.ndarray) -> float:

    difference = reference.astype(np.float64) - output.astype(np.float64)
    mse = float(np.mean(difference**2))
    peak = 65535.0 if reference.dtype == np.uint16 else 255.0
    return math.inf if mse == 0 else 10 * math.log10(peak**2 / mse)


def chains(rows: Path, logs: Path) -> None:
    import cv2

    print("| Chain | Model | Result | Seconds | Output | PSNR vs reference (dB) |")
    print("|---|---|---|---|---|---|")
    failed: list[str] = []
    for line in rows.read_text(encoding="utf-8").splitlines():
        name, role, result, seconds, output, reference = line.split("\t")
        size = quality = ""
        if result.startswith("FAILED"):
            failed.append(name)
        elif result == "passed":
            image = cv2.imread(output, cv2.IMREAD_UNCHANGED)
            expected = cv2.imread(reference, cv2.IMREAD_UNCHANGED)
            if image is None:
                size = "unreadable"
            else:
                size = f"{image.shape[1]}x{image.shape[0]}"
                if output == reference:
                    quality = "reference"
                elif expected is None:
                    quality = "no reference"
                elif expected.shape != image.shape:
                    quality = "other size"
                else:
                    quality = f"{psnr(np.asarray(expected), np.asarray(image)):.2f}"
        print(f"| {name} | {role} | {result} | {seconds} | {size} | {quality} |")
    for name in failed:
        log = (logs / f"chain-{name}.log").read_text(encoding="utf-8", errors="replace")
        errors = [
            text
            for text in log.splitlines()
            if "rror" in text or "failed" in text.lower()
        ]
        print(f"\n{name}, last error lines of chain-{name}.log:\n")
        print("```")
        print("\n".join(errors[-5:] or log.splitlines()[-5:]))
        print("```")


def main() -> int:
    command, *arguments = sys.argv[1:]
    if command == "packages":
        packages(Path(arguments[0]), Path(arguments[1]), Path(arguments[2]))
    elif command == "tests":
        return tests(Path(arguments[0]), [Path(a) for a in arguments[1:]])
    elif command == "chains":
        chains(Path(arguments[0]), Path(arguments[1]))
    else:
        raise SystemExit(f"unknown command {command}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
