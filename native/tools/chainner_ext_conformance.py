"""Conformance of chainner_ext_c against the real chainner_ext 0.3.10 (spec 2 and 4).

Usage (native/.venv-py311, -B, PYTHONDONTWRITEBYTECODE=1, CUDA_VISIBLE_DEVICES=-1;
the tool pins itself and its children to --affinity, default 0xFF00 = CPUs 8-15, so
it runs directly and its exit status is real):
  python -B native/tools/chainner_ext_conformance.py record-regex
  python -B native/tools/chainner_ext_conformance.py regex-probe [--probe EXE]
      [--modes default,nfa,backtrack,dfa] [--prefilter on,off] [--only PREFIX]
  python -B native/tools/chainner_ext_conformance.py regex-module [--candidate real|c]
  python -B native/tools/chainner_ext_conformance.py record
  python -B native/tools/chainner_ext_conformance.py images [--candidate real|c]
      [--only PREFIX]
  python -B native/tools/chainner_ext_conformance.py tie-proof [--processes N]

record-regex runs every candidate case of extract_regex_corpus.build_cases() on the
real module and writes, under native/tests/chainner_ext/:
- regex_corpus.jsonl: the probe input, {"id", "pattern", "ops"} per line;
- regex_expected.jsonl: the real results, {"id", "results"} per line, in the probe's
  output format (match_json());
- regex_module_cases.jsonl: the module-level cases the probe cannot express
  (module_cases()), each with its recorded outcomes.
It also binary-searches the CompiledTooBig flip of each TOOBIG_FAMILIES pattern on
the real module and adds flip-1, flip and flip+1 to the corpus. A probe op that
raises on the real module stops the recording (it belongs to the module cases).

regex-probe runs regex_probe.exe once per mode cell over the corpus and diffs its
output against the expected file, per case and op; '#midchar' cases are counted
apart (see extract_regex_corpus). regex-module runs the module cases and the corpus
through the Python API of the real module and of the candidate (chainner_ext_c from
native/build/harness, or the real module against itself) in one process and diffs
them.

record runs the image and clipboard cases (build_image_cases()) on the real module
and writes manifest.json: the SCHEMA, the rules, the regex files' hashes and counts,
the API surface, the per-cell counts and floors, the exclusions with reasons and
evidence, and one line per case (recipe, class, outcome). A cell is an item x
channel count x mode; its floor is NEW (50) on the inventory's gap paths and COVERED
(10) where the 14,960 existing cases reach. Inputs are recipes (make_input()); an
output is the SHA-256 of its bytes after every NaN becomes 0x7fc00000. classify()
gives each case its class: "compare" (a NaN palette colour on the R-tree included:
its bulk load panics); "tie" (upstream's AHashSet order decides: palette colours
with equal sort keys, or grid queries on the R-tree), counted, not compared;
"corrected" (the 300+-colour grayscale palette upstream's R-tree panics on;
non-finite queries on the R-tree, which its traversal decides and chaiNNer-C
answers by the sub-300 linear rule, Consult 8 D-5; the image clipboard failures
upstream leaks memory on), each with upstream's recorded outcome as its evidence.
Clipboard cases run under ClipboardCapture (the import table of
the binary that calls the Win32 clipboard is redirected; the live clipboard is never
touched) and add the captured payload hashes and calls.

images runs every manifest case on the real module (drift against the manifest) and
on the candidate, which must equal the manifest for "compare" and give equal results
twice for the others. tie-proof runs the tie and corrected cases on the real module
in fresh processes and reports the cases whose results differ between processes.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import extract_regex_corpus as corpus
import golden_kernels
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "native/tests/chainner_ext"
CORPUS = OUT_DIR / "regex_corpus.jsonl"
EXPECTED = OUT_DIR / "regex_expected.jsonl"
MODULE_CASES = OUT_DIR / "regex_module_cases.jsonl"
TOOBIG = OUT_DIR / "regex_toobig.json"
EXCLUSIONS = OUT_DIR / "nfa_boundary_exclusions.json"
HARNESS_DIR = REPO / "native/build/harness"
NATIVE_DLL_DIR = REPO / "backend/src/nodes/impl"
PROBES = (
    REPO / "native/build-regex/regex_probe.exe",
    REPO / "native/build/regex_probe.exe",
)
MODES = ("default", "nfa", "backtrack", "dfa")
PREFILTERS = ("on", "off")
# Exceptions compared by type only (spec 2: pyo3's argument-conversion errors).
TYPE_ONLY = frozenset(
    {"builtins.TypeError", "builtins.OverflowError", "builtins.UnicodeEncodeError"}
)

Json = Any


# ---------------------------------------------------------------------------
# Module loading


def load_real() -> ModuleType:
    return importlib.import_module("chainner_ext")


_DLL_DIRECTORIES: list[Any] = []  # add_dll_directory handles, kept open


def load_c() -> ModuleType:
    """chainner_ext_c from native/build/harness (X0/X3's harness copy), with
    backend/src/nodes/impl on the DLL search path for chainner_native.dll."""
    path = HARNESS_DIR / "chainner_ext_c.pyd"
    if not path.exists():
        raise FileNotFoundError(f"{path} is not built yet")
    if not _DLL_DIRECTORIES:
        _DLL_DIRECTORIES.append(os.add_dll_directory(str(NATIVE_DLL_DIR)))
    sys.path.insert(0, str(HARNESS_DIR))
    try:
        return importlib.import_module("chainner_ext_c")
    finally:
        sys.path.remove(str(HARNESS_DIR))


def load_candidate(name: str) -> ModuleType:
    return load_real() if name == "real" else load_c()


# ---------------------------------------------------------------------------
# Outcomes


def exc_name(exc: BaseException) -> str:
    cls = type(exc)
    return f"{cls.__module__}.{cls.__qualname__}"


def outcome(fn: Callable[[], Json]) -> Json:
    """{"ok": value} or {"raise": {"type", "message"}}; BaseException is caught,
    since pyo3's PanicException derives from it."""
    try:
        return {"ok": fn()}
    except BaseException as exc:  # PanicException is a BaseException
        if isinstance(exc, KeyboardInterrupt | SystemExit):
            raise
        return {"raise": {"type": exc_name(exc), "message": str(exc)}}


def same_outcome(a: Json, b: Json) -> bool:
    if isinstance(a, dict) and isinstance(b, dict) and "raise" in a and "raise" in b:
        ra: dict[str, str] = a["raise"]
        rb: dict[str, str] = b["raise"]
        if ra["type"] in TYPE_ONLY:
            return ra["type"] == rb["type"]
        return ra == rb
    return a == b


# ---------------------------------------------------------------------------
# Probe-format results


def span(group: Any) -> list[int] | None:
    return None if group is None else [group.start, group.end]


def sorted_groupindex(regex: Any) -> dict[str, int]:
    gi: dict[str, int] = dict(regex.groupindex)
    return dict(sorted(gi.items(), key=lambda kv: kv[1]))


def match_json(m: Any, groups: int, names: Iterable[str]) -> Json:
    if m is None:
        return None
    return {
        "start": m.start,
        "end": m.end,
        "len": m.len,
        "groups": [span(m.get(i)) for i in range(groups + 1)],
        "names": {name: span(m.get_by_name(name)) for name in names},
    }


PANIC = "pyo3_runtime.PanicException"


def _run_op(regex: Any, op: dict[str, Any]) -> Json:
    groups: int = regex.groups
    names = list(sorted_groupindex(regex))
    kind = op["op"]
    if kind == "search":
        return match_json(regex.search(op["text"], op["pos"]), groups, names)
    if kind == "findall":
        return [match_json(m, groups, names) for m in regex.findall(op["text"])]
    if kind == "split":
        return list(regex.split(op["text"]))
    if kind == "split_without_captures":
        return list(regex.split_without_captures(op["text"]))
    raise ValueError(f"unknown op {kind}")


def run_probe_case(module: ModuleType, line: dict[str, Any]) -> list[Json]:
    """The probe's results for one corpus line, through module's Python API: an op
    that panics gives {"panic": message}; other exceptions, apart from the compile
    ValueError, propagate."""
    results: list[Json] = []
    regex: Any = None
    for op in line["ops"]:
        if op["op"] == "compile":
            try:
                regex = module.RustRegex(line["pattern"])
            except ValueError as exc:
                results.append({"ok": False, "error": str(exc)})
                regex = None
            else:
                results.append(
                    {
                        "ok": True,
                        "groups": regex.groups,
                        "groupindex": sorted_groupindex(regex),
                    }
                )
            continue
        if regex is None:
            results.append(None)
            continue
        try:
            results.append(_run_op(regex, op))
        except BaseException as exc:  # PanicException is a BaseException
            if exc_name(exc) != PANIC:
                raise
            results.append({"panic": str(exc)})
    return results


# ---------------------------------------------------------------------------
# Module-level cases (what the probe cannot express)

_BIG = (1 << 64) - 1


def _group_calls(regex_names: list[str], groups: int) -> dict[str, Any]:
    misses = ["", "nope", "0", "1", *[n.upper() + "_" for n in regex_names[:1]]]
    return {
        "get": [-1, *range(groups + 3), 1 << 32, _BIG, 1 << 64],
        "get_by_name": [*regex_names, *misses],
    }


def module_cases(cases: list[corpus.Case]) -> list[Json]:
    """Recipes of the module-level cases: attributes, get/get_by_name hits, misses
    and out-of-range indices, MatchGroup.len, the default pos, argument-conversion
    errors and the type surface."""
    out: list[Json] = []
    by_id = {c.id: c for c in cases}
    for case_id in (
        "names/00",
        "names/01",
        "names/03",
        "names/04",
        "names/09",
        "names/10",
    ):
        case = by_id[case_id]
        calls: list[Json] = [{"call": "attrs"}]
        for text in case.texts:
            calls.append({"call": "search", "text": text, "groups_calls": True})
            calls.append({"call": "findall", "text": text, "groups_calls": True})
        out.append({"id": f"module/{case_id}", "pattern": case.pattern, "calls": calls})
    conversions: list[Json] = [
        {"call": "search", "text": "abc", "pos": -1},
        {"call": "search", "text": "abc", "pos": 1 << 64},
        {"call": "search", "text": "abc", "pos": _BIG},
        {"call": "search", "text": "abc", "pos": 1 << 63},
        {"call": "search", "text": "abc", "pos": None},
        {"call": "search", "text": "abc", "pos": 1.0},
        {"call": "search", "text": "abc", "pos": True},
        {"call": "search", "text": {"$bytes": "616263"}},
        {"call": "search", "text": None},
        {"call": "search", "text": "a\ud800b"},
        {"call": "findall", "text": "a\udfffb"},
        {"call": "split", "text": 5},
        {"call": "split_without_captures", "text": {"$bytes": "61"}},
        {"call": "search", "text": "abc", "groups_calls": True},
        {"call": "search_kw", "text": "abc", "pos": 1},
    ]
    out.append(
        {"id": "conversions/abc", "pattern": "(?P<n>b)(c)?", "calls": conversions}
    )
    out.append({"id": "construct", "pattern": None, "calls": [
        {"call": "construct", "arg": "a("},
        {"call": "construct", "arg": 5},
        {"call": "construct", "arg": {"$bytes": "61"}},
        {"call": "construct", "arg": "a\ud800"},
        {"call": "construct", "arg": None},
        {"call": "construct_kw", "arg": "a"},
        {"call": "surface"},
    ]})  # fmt: skip
    return out


def _match_details(m: Any, regex: Any) -> Json:
    if m is None:
        return None
    names = list(sorted_groupindex(regex))
    calls = _group_calls(names, regex.groups)

    def group(g: Any) -> Json:
        return None if g is None else [g.start, g.end, g.len]

    return {
        "start": m.start,
        "end": m.end,
        "len": m.len,
        "get": [outcome(lambda i=i: group(m.get(i))) for i in calls["get"]],
        "get_by_name": [
            outcome(lambda n=n: group(m.get_by_name(n))) for n in calls["get_by_name"]
        ],
    }


def _surface(module: ModuleType) -> Json:
    out: dict[str, Json] = {}
    for name in ("RustRegex", "RegexMatch", "MatchGroup"):
        cls = getattr(module, name)
        out[name] = {
            "module": cls.__module__,
            "name": cls.__name__,
            "qualname": cls.__qualname__,
            "doc": cls.__doc__,
            "members": sorted(n for n in dir(cls) if not n.startswith("__")),
            "text_signature": getattr(cls, "__text_signature__", None),
        }
        for member in out[name]["members"]:
            attr = getattr(cls, member)
            out[name][f"{member}.doc"] = getattr(attr, "__doc__", None)
            out[name][f"{member}.text_signature"] = getattr(
                attr, "__text_signature__", None
            )

    def subclass() -> Json:
        type("Sub", (module.RustRegex,), {})
        return "subclassable"

    out["subclass"] = outcome(subclass)
    regex = module.RustRegex("a")
    out["setattr"] = outcome(lambda: setattr(regex, "x", 1))
    out["match_new"] = outcome(module.RegexMatch)
    out["group_new"] = outcome(module.MatchGroup)
    out["repr"] = repr(regex).split(" object at ")[0]
    return out


def _arg(value: Json) -> Any:
    """Decodes a JSON argument; {"$bytes": hex} stands for a bytes object."""
    if isinstance(value, dict) and "$bytes" in value:
        return bytes.fromhex(value["$bytes"])
    return value


def run_module_call(
    module: ModuleType, pattern: str | None, call: dict[str, Any]
) -> Json:
    kind = call["call"]
    if kind == "construct":
        return outcome(lambda: module.RustRegex(_arg(call["arg"])).pattern)
    if kind == "construct_kw":
        return outcome(lambda: module.RustRegex(pattern=call["arg"]).pattern)
    if kind == "surface":
        return outcome(lambda: _surface(module))
    built = outcome(lambda: module.RustRegex(pattern))
    if "raise" in built:
        return built
    regex = built["ok"]
    if kind == "attrs":
        return {
            "pattern": regex.pattern,
            "groups": regex.groups,
            "groupindex": sorted_groupindex(regex),
        }
    text = _arg(call["text"])
    details = call.get("groups_calls", False)
    if kind == "search":
        if "pos" in call:
            found = outcome(lambda: regex.search(text, call["pos"]))
        else:
            found = outcome(lambda: regex.search(text))
        if "ok" in found:
            m = found["ok"]
            if details:
                return {"ok": _match_details(m, regex)}
            return {"ok": match_json(m, regex.groups, sorted_groupindex(regex))}
        return found
    if kind == "search_kw":
        found = outcome(lambda: regex.search(text=text, pos=call["pos"]))
        if "ok" in found:
            return {
                "ok": match_json(found["ok"], regex.groups, sorted_groupindex(regex))
            }
        return found
    if kind == "findall":
        found = outcome(lambda: regex.findall(text))
        if "ok" in found and details:
            return {"ok": [_match_details(m, regex) for m in found["ok"]]}
        if "ok" in found:
            return {
                "ok": [
                    match_json(m, regex.groups, sorted_groupindex(regex))
                    for m in found["ok"]
                ]
            }
        return found
    if kind in ("split", "split_without_captures"):
        return outcome(lambda: list(getattr(regex, kind)(text)))
    raise ValueError(f"unknown call {kind}")


def run_module_case(module: ModuleType, case: dict[str, Any]) -> list[Json]:
    return [run_module_call(module, case["pattern"], c) for c in case["calls"]]


# CPython 3.13 appends " and no __dict__ for setting new attributes" to the
# AttributeError of a new attribute on an object without __dict__; the recording
# is 3.11's, so this interpreter's suffix is added to it.
SETATTR_SUFFIX = (
    " and no __dict__ for setting new attributes" if sys.version_info >= (3, 13) else ""
)


def module_case_expected(case: dict[str, Any]) -> list[Json]:
    """The case's recorded outcomes as this interpreter words them: the surface
    call's setattr message gains SETATTR_SUFFIX. Exact, no prefix matching."""
    out = json.loads(json.dumps(case["expected"]))
    for call, want in zip(case["calls"], out, strict=True):
        if call["call"] == "surface" and "ok" in want:
            setattr_outcome = want["ok"]["setattr"]
            if (
                setattr_outcome.get("raise", {}).get("type")
                == "builtins.AttributeError"
            ):
                setattr_outcome["raise"]["message"] += SETATTR_SUFFIX
    return out


# ---------------------------------------------------------------------------
# record-regex


def toobig_flip(module: ModuleType, family: str) -> tuple[int, str]:
    """The smallest n whose pattern fails with CompiledTooBig, and its message."""

    def fails(n: int) -> str | None:
        try:
            module.RustRegex(corpus.TOOBIG_FAMILIES[family].format(n=n))
        except ValueError as exc:
            return str(exc)
        return None

    lo, hi = 1, 2
    while fails(hi) is None:
        lo, hi = hi, hi * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if fails(mid) is None:
            lo = mid
        else:
            hi = mid
    message = fails(hi)
    assert message is not None
    return hi, message


def _write_jsonl(path: Path, rows: Iterable[Json], ascii_only: bool = False) -> int:
    """ascii_only escapes every non-ASCII char, so lone surrogates survive."""
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=ascii_only) + "\n")
            count += 1
    return count


def result_kinds(lines: list[Json], expected: list[Json]) -> Counter[str]:
    """Counts of the recorded results by position class, op and outcome ("skipped":
    an op after a failed compile)."""
    kinds: Counter[str] = Counter()
    for line, row in zip(lines, expected, strict=True):
        cls = class_of(line["id"])
        compiled = row["results"][0]["ok"]
        kinds[f"{cls}:compile_ok" if compiled else f"{cls}:compile_error"] += 1
        for op, result in zip(line["ops"][1:], row["results"][1:], strict=True):
            if not compiled:
                kinds[f"{cls}:skipped"] += 1
            elif isinstance(result, dict) and "panic" in result:
                kinds[f"{cls}:{op['op']}_panic"] += 1
            elif result is None:
                kinds[f"{cls}:{op['op']}_none"] += 1
            else:
                kinds[f"{cls}:{op['op']}_value"] += 1
    return kinds


def record_regex() -> int:
    real = load_real()
    cases = corpus.build_cases()
    flips: dict[str, Json] = {}
    for family, template in corpus.TOOBIG_FAMILIES.items():
        flip, message = toobig_flip(real, family)
        flips[family] = {"pattern": template, "flip": flip, "message": message}
        print(f"toobig {family}: flip at n={flip}: {message!r}")
        cases += [corpus.toobig_case(family, n) for n in (flip - 1, flip, flip + 1)]
    lines: list[Json] = [line for case in cases for line in corpus.probe_lines(case)]
    expected: list[Json] = []
    failures: list[str] = []
    for line in lines:
        try:
            expected.append({"id": line["id"], "results": run_probe_case(real, line)})
        except BaseException as exc:  # PanicException is a BaseException
            if isinstance(exc, KeyboardInterrupt | SystemExit):
                raise
            failures.append(f"{line['id']}: {exc_name(exc)}: {exc}")
    if failures:
        print(f"{len(failures)} probe cases raise on the real module:", file=sys.stderr)
        for f in failures[:50]:
            print("  " + f, file=sys.stderr)
        return 1
    mod_cases = module_cases(cases)
    for case in mod_cases:
        case["expected"] = run_module_case(real, case)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_corpus = _write_jsonl(CORPUS, lines)
    _write_jsonl(EXPECTED, expected)
    n_module = _write_jsonl(MODULE_CASES, mod_cases, ascii_only=True)
    TOOBIG.write_text(
        json.dumps(flips, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    n_ops = sum(len(line["ops"]) for line in lines)
    print(f"corpus: {n_corpus} cases, {n_ops} ops; module cases: {n_module}")
    for family, count in sorted(
        Counter(family_of(line["id"]) for line in lines).items()
    ):
        print(f"  {family}: {count}")
    kinds = result_kinds(lines, expected)
    print("  results: " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())))
    return 0


def family_of(case_id: str) -> str:
    head = case_id.split("/", 1)[0]
    for cls in ("#midchar", "#past"):
        if case_id.endswith(cls):
            return head + cls
    return head


# ---------------------------------------------------------------------------
# regex-probe


def read_jsonl(path: Path) -> list[Json]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def find_probe() -> Path:
    for path in PROBES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "regex_probe.exe is not built: " + ", ".join(map(str, PROBES))
    )


POS_CLASSES = ("boundary", "midchar", "past")
# Consult 4 (binding): the position classes each mode is compared with the real
# module on; nfa == backtrack is asserted on every class (pair_splits()).
COMPARED_CLASSES = {
    "default": POS_CLASSES,
    "nfa": ("boundary",),
    "backtrack": ("boundary",),
    "dfa": ("boundary",),
}


# The controller's refinement of Consult 4: boundary cases where regex 1.8.4's own
# PikeVM and backtracker disagree with its default DFA path (mechanisms A and B in
# the file). Excluded from the nfa and backtrack cells' comparison with the real
# module only; default and dfa compare them, and nfa == backtrack still holds.
NFA_EXCLUDED_MODES = ("nfa", "backtrack")


def load_exclusions() -> dict[str, Json]:
    """The exclusion list by case id, checked against the corpus: each id is a
    boundary case with that pattern, and the counts add up."""
    data = json.loads(EXCLUSIONS.read_text(encoding="utf-8"))
    cases: dict[str, Json] = {c["id"]: c for c in data["cases"]}
    if len(cases) != data["count"]["cases"]:
        raise ValueError("the exclusion list's case count is wrong")
    if sum(c["ops"] for c in cases.values()) != data["count"]["ops"]:
        raise ValueError("the exclusion list's op count is wrong")
    patterns = {line["id"]: line["pattern"] for line in read_jsonl(CORPUS)}
    for case_id, case in cases.items():
        if class_of(case_id) != "boundary" or patterns.get(case_id) != case["pattern"]:
            raise ValueError(
                f"exclusion {case_id} is not that boundary case of the corpus"
            )
        if not set(case["mechanisms"]) <= set(data["mechanisms"]):
            raise ValueError(f"exclusion {case_id} names an unknown mechanism")
    return cases


def class_of(case_id: str) -> str:
    """The case's position class: its '#midchar' or '#past' suffix, else boundary."""
    for cls in ("midchar", "past"):
        if case_id.endswith("#" + cls):
            return cls
    return "boundary"


def diff_results(
    expected: dict[str, list[Json]],
    got: dict[str, list[Json]],
    ops: dict[str, list[Json]],
    classes: Iterable[str],
    excluded: frozenset[str] = frozenset(),
) -> tuple[Counter[str], list[str]]:
    """Counts per family and per position class ("class:<cls>:..."); the cases of
    the classes not in `classes` are counted as not compared, and the `excluded`
    ids as excluded, with their differing ops counted ("class:<cls>:excluded_op_diffs")."""
    counts: Counter[str] = Counter()
    details: list[str] = []
    compared = set(classes)
    for case_id, want in expected.items():
        family = family_of(case_id)
        cls = class_of(case_id)
        have = got.get(case_id)
        counts[f"class:{cls}:cases"] += 1
        if cls not in compared:
            counts[f"class:{cls}:not_compared"] += 1
            continue
        if case_id in excluded:
            counts[f"class:{cls}:excluded"] += 1
            if have is not None:
                counts[f"class:{cls}:excluded_op_diffs"] += sum(
                    1 for k, w in enumerate(want) if k >= len(have) or have[k] != w
                )
            continue
        counts[f"{family}:cases"] += 1
        if have is None:
            counts[f"{family}:missing"] += 1
            counts[f"class:{cls}:missing"] += 1
            details.append(f"{case_id}: missing from the probe output")
            continue
        bad = False
        for k, w in enumerate(want):
            h = have[k] if k < len(have) else "<absent>"
            counts[f"{family}:ops"] += 1
            counts[f"class:{cls}:ops"] += 1
            if h != w:
                bad = True
                counts[f"{family}:op_diffs"] += 1
                counts[f"class:{cls}:op_diffs"] += 1
                if len(details) < 200:
                    op = json.dumps(ops[case_id][k], ensure_ascii=False)
                    details.append(
                        f"{case_id} op {k} {op}\n    want {json.dumps(w, ensure_ascii=False)}"
                        f"\n    got  {json.dumps(h, ensure_ascii=False)}"
                    )
        if len(have) != len(want):
            bad = True
            details.append(f"{case_id}: {len(have)} results, want {len(want)}")
        if bad:
            counts[f"{family}:case_diffs"] += 1
            counts[f"class:{cls}:case_diffs"] += 1
    for case_id in got.keys() - expected.keys():
        counts["unknown:cases"] += 1
        details.append(f"{case_id}: not in the expected file")
    return counts, details


def pair_splits(
    a: dict[str, list[Json]], b: dict[str, list[Json]], ops: dict[str, list[Json]]
) -> tuple[Counter[str], list[str]]:
    """Op-by-op differences between two cells' outputs, per position class."""
    counts: Counter[str] = Counter()
    details: list[str] = []
    for case_id in sorted(a.keys() | b.keys()):
        cls = class_of(case_id)
        ra, rb = a.get(case_id), b.get(case_id)
        if ra is None or rb is None or len(ra) != len(rb):
            counts[f"{cls}:splits"] += 1
            details.append(f"{case_id}: present or sized differently in the two cells")
            continue
        for k, (x, y) in enumerate(zip(ra, rb, strict=True)):
            counts[f"{cls}:ops"] += 1
            if x != y:
                counts[f"{cls}:splits"] += 1
                if len(details) < 40:
                    op = json.dumps(ops[case_id][k], ensure_ascii=False)
                    details.append(
                        f"{case_id} op {k} {op}\n    one   {json.dumps(x, ensure_ascii=False)}"
                        f"\n    other {json.dumps(y, ensure_ascii=False)}"
                    )
    return counts, details


def run_probe_cell(
    probe: Path, mode: str, prefilter: str, payload: bytes
) -> tuple[dict[str, list[Json]] | None, float, str]:
    """One mode cell: (results by id, or None when the probe failed; wall; note)."""
    start = time.perf_counter()
    proc = subprocess.run(
        [str(probe), "--mode", mode, "--prefilter", prefilter],
        input=payload,
        capture_output=True,
        check=False,
    )
    wall = time.perf_counter() - start
    # 0: ran; 3: ran with a stub compile ("unimplemented: ..."), still diffed;
    # 2: malformed input; anything else: a crash.
    if proc.returncode not in (0, 3):
        tail = proc.stderr.decode("utf-8", "replace")[-4000:]
        return None, wall, f"probe exit {proc.returncode}\n{tail}"
    got: dict[str, list[Json]] = {}
    # "\n" only: strings hold raw U+0085, U+2028 and U+2029, which
    # str.splitlines() would also split on.
    for raw in proc.stdout.decode("utf-8").split("\n"):
        if raw.strip():
            row = json.loads(raw)
            got[row["id"]] = row["results"]
    return got, wall, "probe exit 3: a stub compile ran" if proc.returncode == 3 else ""


def regex_probe(
    probe: Path, modes: list[str], prefilters: list[str], only: str | None
) -> int:
    """Each mode cell against the real module on its COMPARED_CLASSES (the nfa and
    backtrack cells without the exclusion list's ids), then nfa == backtrack on every
    op of every class, per prefilter; counts per class per cell."""
    exclusions = frozenset(load_exclusions())
    lines = read_jsonl(CORPUS)
    expected_rows = read_jsonl(EXPECTED)
    if only:
        lines = [line for line in lines if line["id"].startswith(only)]
    keep = {line["id"] for line in lines}
    expected = {r["id"]: r["results"] for r in expected_rows if r["id"] in keep}
    ops = {line["id"]: line["ops"] for line in lines}
    payload = "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines)
    total_diffs = 0
    first_cell = True  # the per-family table is printed in full once
    outputs: dict[tuple[str, str], dict[str, list[Json]]] = {}
    for mode in modes:
        for prefilter in prefilters:
            cell = f"mode={mode} prefilter={prefilter}"
            got, wall, note = run_probe_cell(
                probe, mode, prefilter, payload.encode("utf-8")
            )
            if note:
                print(f"[{cell}] {note}")
            if got is None:
                total_diffs += 1
                continue
            outputs[mode, prefilter] = got
            classes = COMPARED_CLASSES[mode]
            excluded = exclusions if mode in NFA_EXCLUDED_MODES else frozenset[str]()
            counts, details = diff_results(expected, got, ops, classes, excluded)
            diffs = sum(
                v for k, v in counts.items()
                if not k.startswith("class:") and k.endswith((":case_diffs", ":missing"))
            )  # fmt: skip
            diffs += counts["unknown:cases"]
            total_diffs += diffs
            print(
                f"[{cell}] compared on {', '.join(classes)}: {diffs} cases differing;"
                f" probe wall {wall:.3f} s"
            )
            for cls in POS_CLASSES:
                if cls in classes:
                    excluded_note = ""
                    if counts[f"class:{cls}:excluded"]:
                        excluded_note = (
                            f"  excluded {counts[f'class:{cls}:excluded']} cases"
                            f" ({counts[f'class:{cls}:excluded_op_diffs']} ops differ)"
                        )
                    print(
                        f"    {cls:9s} cases {counts[f'class:{cls}:cases']:5d}"
                        f"  ops {counts[f'class:{cls}:ops']:6d}"
                        f"  case diffs {counts[f'class:{cls}:case_diffs']:4d}"
                        f"  op diffs {counts[f'class:{cls}:op_diffs']:5d}"
                        f"  missing {counts[f'class:{cls}:missing']:4d}" + excluded_note
                    )
                else:
                    print(
                        f"    {cls:9s} cases {counts[f'class:{cls}:cases']:5d}  not compared"
                    )
            families = sorted(
                {
                    k.split(":")[0]
                    for k in counts
                    if not k.startswith(("class:", "unknown:"))
                }
            )
            if not first_cell:
                families = [
                    f
                    for f in families
                    if counts[f + ":case_diffs"] or counts[f + ":missing"]
                ]
            first_cell = False
            for family in families:
                print(
                    f"      {family:24s} cases {counts[family + ':cases']:5d}"
                    f"  ops {counts[family + ':ops']:6d}"
                    f"  case diffs {counts[family + ':case_diffs']:4d}"
                    f"  op diffs {counts[family + ':op_diffs']:5d}"
                    f"  missing {counts[family + ':missing']:4d}"
                )
            for d in details[:40]:
                print("  " + d)
    for prefilter in prefilters:
        if ("nfa", prefilter) in outputs and ("backtrack", prefilter) in outputs:
            counts, details = pair_splits(
                outputs["nfa", prefilter], outputs["backtrack", prefilter], ops
            )
            splits = sum(counts[f"{cls}:splits"] for cls in POS_CLASSES)
            total_diffs += splits
            print(
                f"[nfa == backtrack, prefilter={prefilter}] "
                + "; ".join(
                    f"{cls} {counts[f'{cls}:splits']} splits in {counts[f'{cls}:ops']} ops"
                    for cls in POS_CLASSES
                )
            )
            for d in details[:20]:
                print("  " + d)
    print("regex-probe: " + ("PASS" if total_diffs == 0 else f"FAIL ({total_diffs})"))
    return 0 if total_diffs == 0 else 1


# ---------------------------------------------------------------------------
# regex-module


def regex_module(candidate_name: str) -> int:
    real = load_real()
    cand = load_candidate(candidate_name)
    diffs = 0
    shown = 0
    lines = read_jsonl(CORPUS)
    expected = {r["id"]: r["results"] for r in read_jsonl(EXPECTED)}
    for line in lines:
        want = expected[line["id"]]
        have = outcome(lambda line=line: run_probe_case(cand, line))
        if have != {"ok": want}:
            diffs += 1
            if shown < 30:
                shown += 1
                print(
                    f"{line['id']}: want {json.dumps(want, ensure_ascii=False)[:400]}"
                )
                print(f"    got {json.dumps(have, ensure_ascii=False)[:400]}")
    n_calls = 0
    for case in read_jsonl(MODULE_CASES):
        want_rows = module_case_expected(case)
        live_rows = run_module_case(real, case)
        have_rows = run_module_case(cand, case)
        for k, (w, live, h) in enumerate(
            zip(want_rows, live_rows, have_rows, strict=True)
        ):
            n_calls += 1
            call = case["calls"][k]
            if not same_outcome(live, w):
                diffs += 1
                print(
                    f"{case['id']} call {k}: the real module moved from the recording"
                )
            if not same_outcome(h, w):
                diffs += 1
                if shown < 60:
                    shown += 1
                    print(
                        f"{case['id']} call {k} {json.dumps(call, ensure_ascii=False)[:200]}"
                    )
                    print(f"    want {json.dumps(w, ensure_ascii=False)[:600]}")
                    print(f"    got  {json.dumps(h, ensure_ascii=False)[:600]}")
    print(f"regex-module ({candidate_name}): {len(lines)} corpus cases, {n_calls} module calls,"
          f" {diffs} differing")  # fmt: skip
    return 0 if diffs == 0 else 1


# ---------------------------------------------------------------------------
# Image and clipboard cases: recipes

CHANNELS = ("2d", 1, 2, 3, 4)
SPECIAL_BITS = (
    "0x7fc00000", "0x7f800000", "0xff800000", "0x80000000", "0x00000001",
    "0x807fffff", "0x7f7fffff", "0x7fc12345", "0xffc54321", "0x3f7fffff",
    "0x3f800001", "0x00800000",
)  # fmt: skip
MAPPINGS = ("unit", "unit", "signed", "grid", "wide", "unit", "tiny")
LAYOUTS = ("c", "c", "c", "f", "strided", "reversed")
ALPHAS = ("levels", "levels", "sparse", "opaque", "transparent")
SHAPES = ((1, 1), (1, 7), (7, 1), (2, 2), (3, 5), (8, 8), (13, 17), (31, 29), (64, 64))
DIFFUSIONS = (
    "FloydSteinberg", "JarvisJudiceNinke", "Stucki", "Atkinson", "Burkes", "Sierra",
    "TwoRowSierra", "SierraLite",
)  # fmt: skip
FILTERS = (
    "Nearest", "Box", "Linear", "Hermite", "CubicCatrom", "CubicMitchell",
    "CubicBSpline", "Hamming", "Hann", "Lanczos", "Lagrange", "Gauss",
)  # fmt: skip
PIXEL_ART = (
    ("adv_mame", 2), ("adv_mame", 3), ("adv_mame", 4), ("eagle", 2), ("eagle", 3),
    ("super_eagle", 2), ("sai", 2), ("super_sai", 2), ("hqx", 2), ("hqx", 3),
    ("hqx", 4),
)  # fmt: skip
NEW, COVERED = 50, 10  # spec 4's floors: new paths, paths the 14,960 cases cover
REJECTED = 2  # a channel count the item rejects with one fixed message


def _seed(*parts: object) -> int:
    text = ":".join(str(p) for p in (corpus.SEED, *parts))
    return int(hashlib.sha256(text.encode()).hexdigest()[:15], 16)


def f32(value: float) -> float:
    """value rounded to float32, as the bindings' `as f32` would."""
    return float(np.float32(value))


def array_recipe(
    rng: random.Random,
    case_id: str,
    role: str,
    channels: object,
    k: int,
    shape: tuple[int, int] | None = None,
    alpha: bool = False,
    specials: bool | None = None,
) -> Json:
    """A seeded array recipe: shape, PCG64 mapping, specials, alpha pattern, layout."""
    if shape is None:
        shape = (
            SHAPES[k] if k < len(SHAPES) else (rng.randint(1, 64), rng.randint(1, 64))
        )
    dims = list(shape) if channels == "2d" else [*shape, int(str(channels))]
    count = math.prod(dims)
    if specials is None:
        specials = k % 5 == 3
    special_list: list[list[Json]] = []
    if specials and count:
        picks = rng.sample(range(count), min(count, rng.randint(1, 6)))
        special_list = [[i, rng.choice(SPECIAL_BITS)] for i in sorted(picks)]
    return {
        "shape": dims,
        "seed": _seed(case_id, role),
        "mapping": MAPPINGS[k % len(MAPPINGS)] if k >= 2 else "unit",
        "specials": special_list,
        "alpha": rng.choice(ALPHAS) if alpha else None,
        "layout": LAYOUTS[k % len(LAYOUTS)],
    }


def make_input(recipe: Json) -> np.ndarray:
    """The recipe's array (golden_kernels.make_array's mappings, plus "wide": unit
    values times 2**e for e in [-145, 122)), then the alpha pattern on the last
    channel, the specials as float32 bits, and the layout (C order, Fortran order,
    every other column of a wider array, or a reversed view); the values never
    depend on the layout."""
    shape = tuple(recipe["shape"])
    mapping = recipe["mapping"]
    dtype = recipe.get("dtype", "float32")
    if mapping == "literal":
        words = np.array([int(v, 16) for v in recipe["values"]], np.uint32)
        return words.view(np.float32).reshape(shape)
    base = golden_kernels.make_array(
        {
            "shape": shape,
            "seed": recipe["seed"],
            "mapping": "unit" if mapping == "wide" else mapping,
            "specials": [],
            "dtype": dtype,
        }
    )
    if mapping == "wide":
        raw = np.random.PCG64(recipe["seed"] + 1).random_raw(base.size)
        exponent = (raw % np.uint64(267)).astype(np.int32) - 145
        base = np.ldexp(base - np.float32(0.5), exponent.reshape(shape)).astype(
            np.float32
        )
    alpha = recipe.get("alpha")
    if alpha is not None and base.size:
        plane = base[..., -1] if base.ndim == 3 else base
        if alpha == "levels":
            raw = np.random.PCG64(recipe["seed"] + 2).random_raw(plane.size)
            levels = np.array([0, 0.04, 0.05, 0.5, 1], np.float32)
            plane[...] = levels[(raw % np.uint64(5)).astype(np.intp)].reshape(
                plane.shape
            )
        elif alpha == "sparse":
            plane[...] = 0
            for y, x in ((0, 0), (-1, -1), (plane.shape[0] // 2, plane.shape[1] // 2)):
                plane[y, x] = 1
        else:
            plane[...] = 1 if alpha == "opaque" else 0
    if recipe["specials"]:
        bits = base.reshape(-1).view(np.uint32)
        for index, value in recipe["specials"]:
            bits[index] = int(value, 16)
    layout = recipe.get("layout", "c")
    if layout == "f":
        return np.asfortranarray(base)
    if layout == "strided" and base.ndim >= 2:
        wide = np.zeros((base.shape[0], base.shape[1] * 2, *base.shape[2:]), base.dtype)
        wide[:, ::2] = base
        return wide[:, ::2]
    if layout == "reversed" and base.ndim >= 2:
        return base[::-1, ::-1].copy()[::-1, ::-1]
    return base


@dataclass
class PaletteArg:
    """A materialized {"$palette"}: the array PaletteQuantization is built from."""

    array: np.ndarray


def materialize(value: Json) -> Any:
    """A case argument with its recipes made into arrays, before any module call (a
    recipe error is the harness's, never an outcome): plain JSON, or {"$img"},
    {"$palette"} (a PaletteArg), {"$tuple"}, {"$float": "nan"|"inf"|"-inf"}; the
    module-dependent {"$uniform"} and {"$enum": [class, member]} stay for construct()."""
    if not isinstance(value, dict):
        return value
    if "$img" in value:
        return make_input(value["$img"])
    if "$palette" in value:
        return PaletteArg(make_input(value["$palette"]))
    if "$tuple" in value:
        return tuple(value["$tuple"])
    if "$float" in value:
        return float(value["$float"])
    if "$uniform" in value or "$enum" in value:
        return value
    raise ValueError(f"unknown argument {value}")


def construct(module: ModuleType, value: Any) -> Any:
    """The module objects of a materialized argument (inside the outcome: their
    constructors' errors are outcomes)."""
    if isinstance(value, PaletteArg):
        return module.PaletteQuantization(value.array)
    if isinstance(value, dict) and "$uniform" in value:
        return module.UniformQuantization(value["$uniform"])
    if isinstance(value, dict) and "$enum" in value:
        cls, member = value["$enum"]
        return getattr(getattr(module, cls), member)
    return value


def img(recipe: Json) -> Json:
    return {"$img": recipe}


# ---------------------------------------------------------------------------
# Cells


@dataclass
class ImageCase:
    id: str
    cell: str
    call: str
    args: list[Json]
    kwargs: dict[str, Json] = field(default_factory=dict)
    failure: str | None = None  # a ClipboardCapture failure mode


def _cell(
    cases: list[ImageCase],
    floors: dict[str, int],
    cell: str,
    floor: int,
    make: Callable[[random.Random, str, int], tuple[str, list[Json]]],
) -> None:
    floors[cell] = floor
    for k in range(floor):
        case_id = f"{cell}/{k:03d}"
        rng = random.Random(f"{corpus.SEED}:{case_id}")
        call, args = make(rng, case_id, k)
        cases.append(ImageCase(case_id, cell, call, args))


def _pick(
    rng: random.Random, k: int, edges: Iterable[float], low: float, high: float
) -> float:
    edges = list(edges)
    return f32(edges[k] if k < len(edges) else rng.uniform(low, high))


def fill_alpha_cells(cases: list[ImageCase], floors: dict[str, int]) -> None:
    thresholds = (0.05, 0.5, 0.0, 1.0, 0.04)
    for ch in CHANNELS:
        floor = REJECTED if ch == 2 else COVERED if ch == 4 else NEW

        def fragment(
            rng: random.Random, cid: str, k: int, ch: object = ch
        ) -> tuple[str, list[Json]]:
            iterations = (0, 1, 2, 6, 8, 33)[k] if k < 6 else rng.randint(0, 10)
            count = (1, 2, 5, 8, 31, 255)[k] if k < 6 else rng.randint(1, 255)
            return "fill_alpha_fragment_blur", [
                img(array_recipe(rng, cid, "img", ch, k, alpha=True)),
                _pick(rng, k, thresholds, 0, 1), iterations, count,
            ]  # fmt: skip

        def extend(
            rng: random.Random, cid: str, k: int, ch: object = ch
        ) -> tuple[str, list[Json]]:
            iterations = (0, 1, 2, 6, 8, 33)[k] if k < 6 else rng.randint(0, 40)
            return "fill_alpha_extend_color", [
                img(array_recipe(rng, cid, "img", ch, k, alpha=True)),
                _pick(rng, k, thresholds, 0, 1), iterations,
            ]  # fmt: skip

        _cell(cases, floors, f"fill_alpha_fragment_blur/ch{ch}", floor, fragment)
        _cell(cases, floors, f"fill_alpha_extend_color/ch{ch}", floor, extend)
        for aa in (False, True):

            def nearest(
                rng: random.Random, cid: str, k: int, ch: object = ch, aa: bool = aa
            ) -> tuple[str, list[Json]]:
                radius = (0, 1, 2, 5, 20)[k] if k < 5 else rng.randint(0, 30)
                return "fill_alpha_nearest_color", [
                    img(array_recipe(rng, cid, "img", ch, k, alpha=True)),
                    _pick(rng, k, thresholds, 0, 1), radius, aa,
                ]  # fmt: skip

            _cell(
                cases,
                floors,
                f"fill_alpha_nearest_color/ch{ch}/aa={aa}",
                floor,
                nearest,
            )


def threshold_cells(cases: list[ImageCase], floors: dict[str, int]) -> None:
    for ch in CHANNELS:
        for aa in (False, True):

            def make(
                rng: random.Random, cid: str, k: int, ch: object = ch, aa: bool = aa
            ) -> tuple[str, list[Json]]:
                args: list[Json] = [
                    img(array_recipe(rng, cid, "img", ch, k)),
                    _pick(rng, k, (0.5, 0.0, 1.0, 0.25), 0, 1),
                    aa,
                ]
                if k % 3 != 0:
                    args.append(_pick(rng, k, (0.0, 0.5, 1.0, 2.0), 0, 2))
                return "binary_threshold", args

            floor = COVERED if aa else NEW
            _cell(cases, floors, f"binary_threshold/ch{ch}/aa={aa}", floor, make)


def esdf_cells(cases: list[ImageCase], floors: dict[str, int]) -> None:
    for ch in ("2d", 1):
        for pre in (False, True):
            for post in (False, True):

                def make(
                    rng: random.Random, cid: str, k: int, ch: object = ch,
                    pre: bool = pre, post: bool = post,
                ) -> tuple[str, list[Json]]:  # fmt: skip
                    return "esdf", [
                        img(array_recipe(rng, cid, "img", ch, k)),
                        _pick(rng, k, (1.0, 4.0, 0.5, 20.0), 0.25, 24),
                        _pick(rng, k, (0.5, 0.0, 1.0, 0.25), 0, 1),
                        pre, post,
                    ]  # fmt: skip

                floor = COVERED if (not pre and post) else NEW
                _cell(cases, floors, f"esdf/ch{ch}/pre={pre}/post={post}", floor, make)

    def wrong(rng: random.Random, cid: str, k: int) -> tuple[str, list[Json]]:
        return "esdf", [
            img(array_recipe(rng, cid, "img", (2, 3, 4)[k % 3], k)),
            2.0,
            0.5,
            False,
            True,
        ]

    _cell(cases, floors, "esdf/channel_errors", COVERED, wrong)


def pixel_art_cells(cases: list[ImageCase], floors: dict[str, int]) -> None:
    for algorithm, scale in PIXEL_ART:
        for ch in ("2d", 1, 3, 4):

            def make(
                rng: random.Random, cid: str, k: int, ch: object = ch,
                algorithm: str = algorithm, scale: int = scale,
            ) -> tuple[str, list[Json]]:  # fmt: skip
                shape = SHAPES[k] if k < 7 else (rng.randint(1, 32), rng.randint(1, 32))
                return "pixel_art_upscale", [
                    img(array_recipe(rng, cid, "img", ch, k, shape=shape)),
                    algorithm,
                    scale,
                ]

            _cell(
                cases,
                floors,
                f"pixel_art_upscale/{algorithm}{scale}/ch{ch}",
                COVERED,
                make,
            )

    bad = (
        ("hqx", 5),
        ("eagle", 4),
        ("sai", 3),
        ("nope", 2),
        ("", 2),
        ("HQX", 2),
        ("hqx", 0),
        ("adv_mame", 1),
    )

    def errors(rng: random.Random, cid: str, k: int) -> tuple[str, list[Json]]:
        algorithm, scale = bad[k % len(bad)]
        ch: object = (3, 2, 5)[k // len(bad)] if k < 3 * len(bad) else 2
        return "pixel_art_upscale", [
            img(array_recipe(rng, cid, "img", ch, k)),
            algorithm,
            scale,
        ]

    _cell(cases, floors, "pixel_art_upscale/errors", 3 * len(bad), errors)


def gamma_cells(cases: list[ImageCase], floors: dict[str, int]) -> None:
    for ch in (*CHANNELS, 5):

        def make(
            rng: random.Random, cid: str, k: int, ch: object = ch
        ) -> tuple[str, list[Json]]:
            gamma = _pick(rng, k, (1.0, 2.2, 1 / 2.2, 0.5, 0.0, -1.0), 0.1, 5)
            return "fast_gamma", [img(array_recipe(rng, cid, "img", ch, k)), gamma]

        _cell(cases, floors, f"fast_gamma/ch{ch}", COVERED, make)


def _palette_recipe(
    rng: random.Random, cid: str, channels: object, k: int, colors: int | None = None
) -> Json:
    width = colors if colors is not None else rng.randint(1, 40)
    recipe = array_recipe(
        rng, cid, "palette", channels, k, shape=(1, width), specials=False
    )
    recipe["mapping"] = "unit" if k % 4 else "signed"
    recipe["layout"] = "c"
    return recipe


def _uniform(rng: random.Random, k: int) -> int:
    return (2, 3, 4, 7, 16, 256, 0xFFFFFFFF)[k] if k < 7 else rng.randint(2, 64)


def dither_cells(cases: list[ImageCase], floors: dict[str, int]) -> None:
    uniform_channels = (*CHANNELS, 5)
    for ch in uniform_channels:
        new = ch in (2, 5)

        def quantize_u(
            rng: random.Random, cid: str, k: int, ch: object = ch
        ) -> tuple[str, list[Json]]:
            return "quantize", [
                img(array_recipe(rng, cid, "img", ch, k)),
                {"$uniform": _uniform(rng, k)},
            ]

        _cell(
            cases,
            floors,
            f"quantize/uniform/ch{ch}",
            NEW if new else COVERED,
            quantize_u,
        )
        for size in (1, 2, 4, 8, 16, 32, 64):

            def ordered(
                rng: random.Random, cid: str, k: int, ch: object = ch, size: int = size
            ) -> tuple[str, list[Json]]:
                return "ordered_dither", [
                    img(array_recipe(rng, cid, "img", ch, k)),
                    {"$uniform": _uniform(rng, k)},
                    size,
                ]

            floor = NEW if new or size in (1, 32, 64) else COVERED
            _cell(cases, floors, f"ordered_dither/map{size}/ch{ch}", floor, ordered)
        for algorithm in DIFFUSIONS:

            def diffusion_u(
                rng: random.Random,
                cid: str,
                k: int,
                ch: object = ch,
                algorithm: str = algorithm,
            ) -> tuple[str, list[Json]]:
                return "error_diffusion_dither", [
                    img(array_recipe(rng, cid, "img", ch, k)),
                    {"$uniform": _uniform(rng, k)},
                    {"$enum": ["DiffusionAlgorithm", algorithm]},
                ]

            floor = REJECTED if new else COVERED
            _cell(
                cases,
                floors,
                f"error_diffusion_dither/uniform/{algorithm}/ch{ch}",
                floor,
                diffusion_u,
            )

        def riemersma_u(
            rng: random.Random, cid: str, k: int, ch: object = ch
        ) -> tuple[str, list[Json]]:
            history = (2, 3, 16, 31, 257)[k] if k < 5 else rng.randint(2, 64)
            decay = (
                f32(1 / history)
                if k % 3 == 0
                else _pick(rng, k, (0.5, 0.01, 0.99), 0.001, 0.999)
            )
            return "riemersma_dither", [
                img(array_recipe(rng, cid, "img", ch, k)),
                {"$uniform": _uniform(rng, k)},
                history,
                decay,
            ]

        floor = REJECTED if new else NEW
        _cell(cases, floors, f"riemersma_dither/uniform/ch{ch}", floor, riemersma_u)

    for ch in ("2d", 1, 3, 4):
        pal_ch: object = 1 if ch == "2d" else ch

        def quantize_p(
            rng: random.Random,
            cid: str,
            k: int,
            ch: object = ch,
            pal_ch: object = pal_ch,
        ) -> tuple[str, list[Json]]:
            return "quantize", [
                img(array_recipe(rng, cid, "img", ch, k)),
                {"$palette": _palette_recipe(rng, cid, pal_ch, k)},
            ]

        _cell(cases, floors, f"quantize/palette/ch{ch}", COVERED, quantize_p)
        for algorithm in DIFFUSIONS:

            def diffusion_p(
                rng: random.Random, cid: str, k: int, ch: object = ch,
                pal_ch: object = pal_ch, algorithm: str = algorithm,
            ) -> tuple[str, list[Json]]:  # fmt: skip
                return "error_diffusion_dither", [
                    img(array_recipe(rng, cid, "img", ch, k)),
                    {"$palette": _palette_recipe(rng, cid, pal_ch, k)},
                    {"$enum": ["DiffusionAlgorithm", algorithm]},
                ]

            _cell(
                cases,
                floors,
                f"error_diffusion_dither/palette/{algorithm}/ch{ch}",
                COVERED,
                diffusion_p,
            )

        def riemersma_p(
            rng: random.Random,
            cid: str,
            k: int,
            ch: object = ch,
            pal_ch: object = pal_ch,
        ) -> tuple[str, list[Json]]:
            history = (2, 3, 16, 31, 257)[k] if k < 5 else rng.randint(2, 64)
            decay = (
                f32(1 / history)
                if k % 3 == 0
                else _pick(rng, k, (0.5, 0.01, 0.99), 0.001, 0.999)
            )
            return "riemersma_dither", [
                img(array_recipe(rng, cid, "img", ch, k)),
                {"$palette": _palette_recipe(rng, cid, pal_ch, k)},
                history, decay,
            ]  # fmt: skip

        _cell(cases, floors, f"riemersma_dither/palette/ch{ch}", NEW, riemersma_p)

        if ch in (3, 4):

            def large(
                rng: random.Random, cid: str, k: int, ch: object = ch
            ) -> tuple[str, list[Json]]:
                mode = k % 3
                palette = {
                    "$palette": _palette_recipe(
                        rng, cid, ch, k, colors=rng.randint(300, 700)
                    )
                }
                image = img(
                    array_recipe(rng, cid, "img", ch, k, specials=False)
                    | {"mapping": "unit"}
                )
                if mode == 0:
                    return "quantize", [image, palette]
                if mode == 1:
                    algorithm = DIFFUSIONS[k % len(DIFFUSIONS)]
                    return "error_diffusion_dither", [
                        image,
                        palette,
                        {"$enum": ["DiffusionAlgorithm", algorithm]},
                    ]
                return "riemersma_dither", [image, palette, 16, f32(1 / 16)]

            _cell(cases, floors, f"palette_tree/ch{ch}", COVERED, large)

    def mismatch(rng: random.Random, cid: str, k: int) -> tuple[str, list[Json]]:
        pairs = (
            (3, 4),
            (4, 3),
            (1, 3),
            (3, 1),
            (2, 2),
            (5, 3),
            ("2d", 3),
            (3, "2d"),
            (2, 1),
            (4, 2),
        )
        ch, pal_ch = pairs[k % len(pairs)]
        pal_ch = 1 if pal_ch == "2d" else pal_ch
        image = img(array_recipe(rng, cid, "img", ch, k))
        palette = {"$palette": _palette_recipe(rng, cid, pal_ch, k)}
        mode = (k // len(pairs)) % 3
        if mode == 0:
            return "quantize", [image, palette]
        if mode == 1:
            return "error_diffusion_dither", [
                image,
                palette,
                {"$enum": ["DiffusionAlgorithm", "Atkinson"]},
            ]
        return "riemersma_dither", [image, palette, 16, f32(1 / 16)]

    _cell(cases, floors, "palette/channel_rules", NEW, mismatch)


def resize_cells(cases: list[ImageCase], floors: dict[str, int]) -> None:
    sizes = ((1, 1), (2, 3), (64, 64), (7, 5), (128, 96), (3, 70))
    for name in FILTERS:
        for gamma in (False, True):
            for ch in CHANNELS:

                def make(
                    rng: random.Random, cid: str, k: int, ch: object = ch,
                    name: str = name, gamma: bool = gamma,
                ) -> tuple[str, list[Json]]:  # fmt: skip
                    size = (
                        sizes[k]
                        if k < len(sizes)
                        else (rng.randint(1, 96), rng.randint(1, 96))
                    )
                    return "resize", [
                        img(array_recipe(rng, cid, "img", ch, k)),
                        {"$tuple": list(size)},
                        {"$enum": ["ResizeFilter", name]},
                        gamma,
                    ]

                _cell(
                    cases, floors, f"resize/{name}/gamma={gamma}/ch{ch}", COVERED, make
                )


# Argument edges: validation errors and panics (message compared), conversions
# (type compared), keyword calls.
def _argument_cases() -> list[tuple[str, list[Json], dict[str, Json]]]:
    rng = random.Random(f"{corpus.SEED}:args")

    def rgba(k: int = 5, ch: object = 4) -> Json:
        return img(array_recipe(rng, f"args/{k}", "img", ch, k, alpha=True))

    def gray(k: int = 5) -> Json:
        return img(array_recipe(rng, f"args/{k}", "img", 1, k))

    f64 = img(
        {
            "shape": [4, 5, 3],
            "seed": 7,
            "mapping": "bits",
            "specials": [],
            "dtype": "float64",
        }
    )
    u8 = img(
        {
            "shape": [4, 5, 3],
            "seed": 7,
            "mapping": "ramp",
            "specials": [],
            "dtype": "uint8",
        }
    )
    i32 = img(
        {
            "shape": [4, 5],
            "seed": 7,
            "mapping": "ramp",
            "specials": [],
            "dtype": "int32",
        }
    )
    flat = img({"shape": [12], "seed": 7, "mapping": "unit", "specials": []})
    four_d = img({"shape": [2, 3, 4, 1], "seed": 7, "mapping": "unit", "specials": []})
    empty = img({"shape": [0, 3, 3], "seed": 7, "mapping": "unit", "specials": []})
    empty_w = img({"shape": [3, 0, 4], "seed": 7, "mapping": "unit", "specials": []})
    nan = {"$float": "nan"}
    inf = {"$float": "inf"}
    out: list[tuple[str, list[Json], dict[str, Json]]] = []
    for count in (0, 256, 1000):
        out.append(("fill_alpha_fragment_blur", [rgba(), 0.05, 2, count], {}))
    out += [
        ("fill_alpha_fragment_blur", [rgba(), 0.05, -1, 5], {}),
        ("fill_alpha_fragment_blur", [rgba(), 0.05, 2, 1 << 32], {}),
        ("fill_alpha_fragment_blur", [rgba(), "0.5", 2, 5], {}),
        ("fill_alpha_fragment_blur", [f64, 0.05, 2, 5], {}),
        ("fill_alpha_fragment_blur", [empty, 0.05, 2, 5], {}),
        ("fill_alpha_fragment_blur", [rgba(), nan, 2, 5], {}),
        ("fill_alpha_extend_color", [rgba(), inf, 3], {}),
        ("fill_alpha_extend_color", [empty_w, 0.5, 3], {}),
        ("fill_alpha_extend_color", [rgba(), 0.5, True], {}),
        ("fill_alpha_nearest_color", [rgba(), 0.5, 3, 1], {}),
        ("fill_alpha_nearest_color", [rgba(), 0.5, 3, "yes"], {}),
        ("fill_alpha_nearest_color", [rgba(), 0.5, 3, None], {}),
        ("fill_alpha_nearest_color", [rgba(ch=5), 0.5, 3, True], {}),
        ("fill_alpha_nearest_color", [empty, 0.5, 3, True], {}),
        ("binary_threshold", [gray(), 0.5, True, None], {}),
        ("binary_threshold", [gray(), 0.5, False], {"extra_smoothness": 1.0}),
        ("binary_threshold", [gray(), nan, True, 0.5], {}),
        ("binary_threshold", [u8, 0.5, True], {}),
        ("binary_threshold", [four_d, 0.5, True], {}),
        ("binary_threshold", [flat, 0.5, True], {}),
        ("binary_threshold", [empty, 0.5, False], {}),
        ("esdf", [gray(), -1.0, 0.5, True, True], {}),
        ("esdf", [gray(), 0.0, 0.5, False, False], {}),
        ("esdf", [gray(), inf, 0.5, True, True], {}),
        ("esdf", [gray(), 3.0, nan, True, True], {}),
        ("esdf", [empty, 3.0, 0.5, True, True], {}),
        ("esdf", [i32, 3.0, 0.5, True, True], {}),
        ("pixel_art_upscale", [gray(), "hqx", -1], {}),
        ("pixel_art_upscale", [gray(), 5, 2], {}),
        ("pixel_art_upscale", [empty, "hqx", 2], {}),
        ("pixel_art_upscale", [empty_w, "eagle", 3], {}),
        ("fast_gamma", [gray(), nan], {}),
        ("fast_gamma", [empty, 2.2], {}),
        ("fast_gamma", [f64, 2.2], {}),
        ("quantize", [gray(), {"$uniform": 1}], {}),
        ("quantize", [gray(), {"$uniform": 0}], {}),
        ("quantize", [gray(), {"$uniform": -2}], {}),
        ("quantize", [gray(), 7], {}),
        ("quantize", [gray(), {"$palette": {"shape": [2, 3, 1], "seed": 1, "mapping": "unit", "specials": []}}], {}),
        ("quantize", [gray(), {"$palette": {"shape": [1, 0, 1], "seed": 1, "mapping": "unit", "specials": []}}], {}),
        ("quantize", [gray(), {"$palette": {"shape": [1, 4, 5], "seed": 1, "mapping": "unit", "specials": []}}], {}),
        ("quantize", [gray(), {"$palette": {"shape": [1, 4], "seed": 1, "mapping": "unit", "specials": []}}], {}),
        ("quantize", [empty, {"$uniform": 4}], {}),
        ("ordered_dither", [gray(), {"$uniform": 4}, 0], {}),
        ("ordered_dither", [gray(), {"$uniform": 4}, 3], {}),
        ("ordered_dither", [gray(), {"$uniform": 4}, 12], {}),
        ("ordered_dither", [gray(), {"$uniform": 4}, -4], {}),
        ("ordered_dither", [gray(), {"$palette": {"shape": [1, 4, 1], "seed": 1, "mapping": "unit", "specials": []}}, 4], {}),
        ("error_diffusion_dither", [gray(), {"$uniform": 4}, 3], {}),
        ("error_diffusion_dither", [gray(), {"$uniform": 4}, "FloydSteinberg"], {}),
        ("error_diffusion_dither", [rgba(ch=5), {"$uniform": 4}, {"$enum": ["DiffusionAlgorithm", "Burkes"]}], {}),
        ("riemersma_dither", [gray(), {"$uniform": 4}, 1, 0.5], {}),
        ("riemersma_dither", [gray(), {"$uniform": 4}, 0, 0.5], {}),
        ("riemersma_dither", [rgba(ch=5), {"$uniform": 4}, 4, 0.5], {}),
    ]  # fmt: skip
    for decay in (0.0, 1.0, -0.5, 1.5, nan, inf, f32(1e-30)):
        out.append(("riemersma_dither", [gray(), {"$uniform": 4}, 16, decay], {}))
        out.append(("riemersma_dither", [rgba(ch=3), {"$palette": {"shape": [1, 5, 3], "seed": 3, "mapping": "unit", "specials": []}}, 16, decay], {}))  # fmt: skip
    for size in ((0, 5), (5, 0), (0, 0), (1 << 32, 1), (-1, 3), (1, 2, 3)):
        out.append(("resize", [gray(), {"$tuple": list(size)}, {"$enum": ["ResizeFilter", "Linear"]}, False], {}))  # fmt: skip
    out += [
        ("resize", [rgba(ch=5), {"$tuple": [3, 3]}, {"$enum": ["ResizeFilter", "Box"]}, False], {}),
        ("resize", [rgba(ch=5), {"$tuple": [3, 3]}, {"$enum": ["ResizeFilter", "Box"]}, True], {}),
        ("resize", [gray(), [3, 3], {"$enum": ["ResizeFilter", "Box"]}, False], {}),
        ("resize", [gray(), {"$tuple": [3, 3]}, 1, False], {}),
        ("resize", [gray(), {"$tuple": [3, 3]}, {"$enum": ["DiffusionAlgorithm", "Burkes"]}, False], {}),
        ("resize", [empty, {"$tuple": [3, 3]}, {"$enum": ["ResizeFilter", "Lanczos"]}, False], {}),
        ("resize", [gray(), {"$tuple": [4, 4]}, {"$enum": ["ResizeFilter", "Hermite"]}, 1], {}),
        ("resize", [], {"img": gray(), "new_size": {"$tuple": [5, 2]}, "filter": {"$enum": ["ResizeFilter", "Gauss"]}, "gamma_correction": True}),
        ("fill_alpha_extend_color", [], {"img": rgba(), "threshold": 0.5, "iterations": 4}),
        ("esdf", [gray()], {"radius": 3.0, "cutoff": 0.5, "pre_process": True, "post_process": False}),
        ("quantize", [gray()], {"quant": {"$uniform": 5}}),
        ("riemersma_dither", [gray(), {"$uniform": 4}], {"history_length": 8, "decay_ratio": 0.25}),
        ("pixel_art_upscale", [gray()], {"algorithm": "hqx", "scale": 2}),
        ("fast_gamma", [gray()], {"gamma": 2.0}),
        ("binary_threshold", [gray()], {"threshold": 0.5, "anti_aliasing": True}),
    ]  # fmt: skip
    return out


def argument_cells(cases: list[ImageCase], floors: dict[str, int]) -> None:
    rows = _argument_cases()
    floors["arguments"] = len(rows)
    for k, (call, args, kwargs) in enumerate(rows):
        cases.append(ImageCase(f"arguments/{k:03d}", "arguments", call, args, kwargs))


CLIPBOARD_TEXTS = (
    "", "a", "hello world", "äöü", "😀 emoji", "line\r\nbreak\nlf\rcr", "nul\x00inside",
    "\t tabs ", "中文字", "x" * 5000, "\u2028sep\u0085", "a\ud7ffb", "\U0010ffff",
)  # fmt: skip
CLIPBOARD_FAILURES = ("open", "empty", "set", "allocate", "lock")


def clipboard_cells(cases: list[ImageCase], floors: dict[str, int]) -> None:
    def text(rng: random.Random, cid: str, k: int) -> tuple[str, list[Json]]:
        if k < len(CLIPBOARD_TEXTS):
            return "Clipboard.write_text", [CLIPBOARD_TEXTS[k]]
        return "Clipboard.write_text", [corpus.random_text(rng, 40)]

    _cell(cases, floors, "clipboard/text", NEW, text)
    for ch in ("2d", 1, 3, 4):
        for fmt in ("RGB", "BGR"):

            def image(
                rng: random.Random, cid: str, k: int, ch: object = ch, fmt: str = fmt
            ) -> tuple[str, list[Json]]:
                recipe = array_recipe(rng, cid, "img", ch, k)
                if k % 4 == 1:
                    recipe["mapping"] = "signed"
                return "Clipboard.write_image", [img(recipe), fmt]

            _cell(cases, floors, f"clipboard/image/ch{ch}/{fmt}", NEW, image)

    def errors(rng: random.Random, cid: str, k: int) -> tuple[str, list[Json]]:
        formats = ("RGBA", "rgb", "", "BGRA", "RGB")
        ch: object = (2, 5, 3, 3, 3)[k % 5]
        return "Clipboard.write_image", [
            img(array_recipe(rng, cid, "img", ch, k)),
            formats[k % 5],
        ]

    _cell(cases, floors, "clipboard/errors", 2 * COVERED, errors)
    floors["clipboard/failures"] = 2 * len(CLIPBOARD_FAILURES)
    rng = random.Random(f"{corpus.SEED}:clipboard/failures")
    for k, failure in enumerate(CLIPBOARD_FAILURES):
        cases.append(
            ImageCase(f"clipboard/failures/{2 * k:03d}", "clipboard/failures", "Clipboard.write_text", ["fail"], failure=failure)
        )  # fmt: skip
        recipe = array_recipe(rng, f"clipboard/failures/{k}", "img", 3, k)
        cases.append(
            ImageCase(f"clipboard/failures/{2 * k + 1:03d}", "clipboard/failures", "Clipboard.write_image", [img(recipe), "RGB"], failure=failure)
        )  # fmt: skip


def tie_and_corrected_cells(cases: list[ImageCase], floors: dict[str, int]) -> None:
    """Palettes whose upstream answer is not defined (equal sort keys, grid inputs
    on the R-tree) and the 300+-colour grayscale palette upstream panics on."""

    def ties(rng: random.Random, cid: str, k: int) -> tuple[str, list[Json]]:
        ch = 3 if k % 2 else 4
        x = rng.choice((0.5, 0.25, 0.75, 1.0))
        colors = [[x, 0.0, 0.0], [-x, 0.0, 0.0], [0.0, x, 0.0], [0.0, -x, 0.0]]
        if ch == 4:
            colors = [[*c, 1.0] for c in colors]
        palette = np.array([colors[: 2 + k % 3]], np.float32)
        palette_recipe = {"shape": list(palette.shape), "seed": 0, "mapping": "literal", "specials": [],
                          "values": [f"0x{int(v):08x}" for v in palette.reshape(-1).view(np.uint32)]}  # fmt: skip
        image = array_recipe(rng, cid, "img", ch, k, specials=False)
        image["mapping"] = "grid"
        mode = k % 3
        quant = {"$palette": palette_recipe}
        if mode == 0:
            return "quantize", [img(image), quant]
        if mode == 1:
            return "error_diffusion_dither", [
                img(image),
                quant,
                {"$enum": ["DiffusionAlgorithm", DIFFUSIONS[k % 8]]},
            ]
        return "riemersma_dither", [img(image), quant, 8, f32(1 / 8)]

    _cell(cases, floors, "palette/ties", COVERED, ties)

    def grayscale(rng: random.Random, cid: str, k: int) -> tuple[str, list[Json]]:
        ch: object = "2d" if k % 2 else 1
        palette = _palette_recipe(rng, cid, 1, k, colors=(300, 301, 1024)[k % 3])
        image = array_recipe(rng, cid, "img", ch, k, specials=False)
        mode = k % 3
        if mode == 0:
            return "quantize", [img(image), {"$palette": palette}]
        if mode == 1:
            return "error_diffusion_dither", [
                img(image),
                {"$palette": palette},
                {"$enum": ["DiffusionAlgorithm", "FloydSteinberg"]},
            ]
        return "riemersma_dither", [img(image), {"$palette": palette}, 16, f32(1 / 16)]

    _cell(cases, floors, "palette/grayscale_tree", COVERED, grayscale)


NAN_BITS = ("0x7fc00000", "0xffc00001", "0x7f800001", "0x7fbfffff")


def nan_tree_cells(cases: list[ImageCase], floors: dict[str, int]) -> None:
    """NaN colours in palettes of 300+ colours that reach upstream's R-tree (3 or 4
    channels, a 1- or 3-channel palette expanded included): its bulk load panics,
    after the channel and shape checks and before Riemersma's base assertion, for an
    empty image too."""

    def nan_tree(rng: random.Random, cid: str, k: int) -> tuple[str, list[Json]]:
        ch = 3 if k % 2 else 4
        source = 1 if k % 5 == 4 else 3 if k % 7 == 6 and ch == 4 else ch
        colors = (300, 301, 307, 512, 1024)[k % 5]
        palette = _palette_recipe(rng, cid, source, k, colors=colors)
        picks = rng.sample(range(colors * source), rng.randint(1, 3))
        palette["specials"] = [[i, rng.choice(NAN_BITS)] for i in sorted(picks)]
        shape = (0, 3) if k % 11 == 10 else None
        image = array_recipe(rng, cid, "img", ch, k, shape=shape, specials=False)
        quant = {"$palette": palette}
        mode = k % 3
        if mode == 0:
            return "quantize", [img(image), quant]
        if mode == 1:
            return "error_diffusion_dither", [
                img(image),
                quant,
                {"$enum": ["DiffusionAlgorithm", DIFFUSIONS[k % 8]]},
            ]
        # A decay giving base >= 1: the NaN panic still comes first.
        decay = f32(2.0) if k % 4 == 0 else f32(1 / 8)
        return "riemersma_dither", [img(image), quant, 8, decay]

    _cell(cases, floors, "palette/nan_tree", NEW, nan_tree)


INF_BITS = ("0x7f800000", "0xff800000")


def nonfinite_tree_cells(cases: list[ImageCase], floors: dict[str, int]) -> None:
    """Consult 8 D-5: +-inf palette colours, NaN or +-inf pixels, or both, with 300+
    colours that reach upstream's R-tree (3 or 4 channels, a 1- or 3-channel palette
    expanded included). Upstream returns traversal-decided colours for those queries;
    chaiNNer-C applies the sub-300 linear rule ("corrected")."""

    def nonfinite_tree(rng: random.Random, cid: str, k: int) -> tuple[str, list[Json]]:
        ch = 3 if k % 2 else 4
        source = 1 if k % 5 == 4 else 3 if k % 7 == 6 and ch == 4 else ch
        colors = (300, 301, 307, 512, 1024)[k % 5]
        kind = k % 3  # 0: the palette, 1: the pixels, 2: both
        palette = _palette_recipe(rng, cid, source, k, colors=colors)
        if kind != 1:
            picks = rng.sample(range(colors * source), rng.randint(1, 3))
            palette["specials"] = [[i, rng.choice(INF_BITS)] for i in sorted(picks)]
        image = array_recipe(rng, cid, "img", ch, k, specials=False)
        image["mapping"] = "unit"
        if kind != 0:
            pixels = math.prod(image["shape"])
            picks = rng.sample(range(pixels), min(pixels, rng.randint(1, 6)))
            image["specials"] = [
                [i, rng.choice((*NAN_BITS, *INF_BITS))] for i in sorted(picks)
            ]
        quant = {"$palette": palette}
        mode = k // 3 % 3
        if mode == 0:
            return "quantize", [img(image), quant]
        if mode == 1:
            return "error_diffusion_dither", [
                img(image),
                quant,
                {"$enum": ["DiffusionAlgorithm", DIFFUSIONS[k % 8]]},
            ]
        return "riemersma_dither", [img(image), quant, 8, f32(1 / 8)]

    _cell(cases, floors, "palette/nonfinite_tree", NEW, nonfinite_tree)


def build_image_cases() -> tuple[list[ImageCase], dict[str, int]]:
    cases: list[ImageCase] = []
    floors: dict[str, int] = {}
    for builder in (
        fill_alpha_cells, threshold_cells, esdf_cells, pixel_art_cells, gamma_cells,
        dither_cells, resize_cells, argument_cells, clipboard_cells, tie_and_corrected_cells,
        nan_tree_cells, nonfinite_tree_cells,
    ):  # fmt: skip
        builder(cases, floors)
    ids = [c.id for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate image case ids")
    return cases, floors


# ---------------------------------------------------------------------------
# Classification: compared, tie (counted, not compared) or corrected


def _sort_keys(colors: np.ndarray) -> np.ndarray:
    """extract_unique_const's float32 sort keys."""
    n = colors.shape[1]
    if n == 1:
        return colors[:, 0]
    if n in (3, 4):
        r, g, b = colors[:, 0], colors[:, 1], colors[:, 2]
        key = (
            r * r * np.float32(0.2126)
            + g * g * np.float32(0.7152)
            + b * b * np.float32(0.0722)
        )
        if n == 4:
            key = key + colors[:, 3] * np.float32(10.0)
        return key
    return colors.sum(axis=1, dtype=np.float32)


def _unique_colors(palette: np.ndarray) -> np.ndarray:
    n = 1 if palette.ndim == 2 else palette.shape[2]
    flat = np.ascontiguousarray(palette).reshape(-1, n)
    bits = np.unique(flat.view(np.uint32), axis=0)
    return bits.view(np.float32)


def classify(case: ImageCase) -> tuple[str, str]:
    """("compare" | "tie" | "corrected", reason) from the recipe alone."""
    if case.call == "Clipboard.write_image" and case.failure in ("lock", "set"):
        return (
            "corrected",
            (
                "upstream leaks the image's HGLOBAL after a failed GlobalLock or "
                "SetClipboardData (and calls DeleteObject on it); chaiNNer-C frees it "
                "with GlobalFree (Consult 6 D-1)"
            ),
        )
    palettes = [
        a["$palette"] for a in case.args if isinstance(a, dict) and "$palette" in a
    ]
    if not palettes or case.call not in (
        "quantize",
        "error_diffusion_dither",
        "riemersma_dither",
    ):
        return "compare", ""
    image_recipe = case.args[0]["$img"]
    palette = make_input(palettes[0])
    image = make_input(image_recipe)
    pal_ch = 1 if palette.ndim == 2 else palette.shape[2]
    img_ch = 1 if image.ndim == 2 else image.shape[2]
    if palette.ndim not in (2, 3) or palette.shape[0] != 1 or img_ch not in (1, 3, 4):
        return "compare", ""  # rejected before any lookup
    # A 1-channel palette, or a 3-channel one for a 4-channel image, is expanded.
    expanded = pal_ch != img_ch
    if expanded and pal_ch != 1 and (pal_ch, img_ch) != (3, 4):
        return "compare", ""  # rejected before any lookup
    colors = _unique_colors(palette)
    if len(colors) == 0:
        return "compare", ""
    if len(colors) >= 300:
        if img_ch == 1:
            return (
                "corrected",
                "upstream's R-tree panics on 1-D points (300+ grayscale colours)",
            )
        if np.isnan(colors).any():
            return "compare", ""  # the R-tree's bulk load panics on any NaN coordinate
        if not (np.isfinite(colors).all() and np.isfinite(image).all()):
            return (
                "corrected",
                (
                    "upstream's R-tree traversal decides non-finite queries; chaiNNer-C "
                    "applies the sub-300 linear rule (Consult 8 D-5)"
                ),
            )
        if image_recipe["mapping"] == "grid" or palettes[0]["mapping"] == "grid":
            return "tie", "R-tree order (from AHashSet) decides equidistant queries"
        return "compare", ""
    if expanded:
        return "compare", ""
    keys = _sort_keys(colors)
    nan = np.isnan(keys)
    finite_keys = keys[~nan]
    if nan.sum() > 1 or len(np.unique(finite_keys)) != len(finite_keys):
        return (
            "tie",
            "two palette colours share a sort key; AHashSet order decides between them",
        )
    return "compare", ""


# ---------------------------------------------------------------------------
# Execution


def array_outcome(value: Any) -> Json:
    if not isinstance(value, np.ndarray):
        return {"type": type(value).__name__, "repr": repr(value)}
    array = value
    if array.dtype not in (np.float32, np.float64):
        return {"shape": list(array.shape), "dtype": str(array.dtype),
                "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}  # fmt: skip
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": golden_kernels.digest(array, "mixed"),
    }


def clipboard_binary(module: ModuleType) -> Path:
    """The loaded binary whose import table holds the clipboard calls."""
    inner = getattr(module, "chainner_ext", None)
    own = inner.__file__ if isinstance(inner, ModuleType) else module.__file__
    if own is None:
        raise RuntimeError(f"{module.__name__} has no file")
    candidates = [Path(own)]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
    kernel.GetModuleHandleW.restype = ctypes.c_void_p
    kernel.GetModuleFileNameW.argtypes = [
        ctypes.c_void_p,
        ctypes.c_wchar_p,
        ctypes.c_uint32,
    ]
    handle = kernel.GetModuleHandleW("chainner_native.dll")
    if handle:
        buffer = ctypes.create_unicode_buffer(32768)
        kernel.GetModuleFileNameW(handle, buffer, len(buffer))
        candidates.append(Path(buffer.value))
    for path in candidates:
        capture = _capture_class()(path)
        if "OpenClipboard" in capture.import_slots():
            return path
    raise RuntimeError(f"no loaded binary of {module.__name__} imports OpenClipboard")


def _capture_class() -> Any:
    tests = str(REPO / "native/tests")
    if tests not in sys.path:
        sys.path.append(tests)
    return importlib.import_module("clipboard_capture").ClipboardCapture


def run_image_case(
    module: ModuleType, case: ImageCase, clipboard_binary: Path | None = None
) -> Json:
    """The case's outcome on module: {"ok": output digest} or {"raise": ...}, plus
    "input_changed" when the call wrote to an input array; a clipboard case adds the
    captured payloads and Win32 calls."""
    args = [materialize(a) for a in case.args]
    kwargs = {k: materialize(v) for k, v in case.kwargs.items()}
    arrays = [a for a in [*args, *kwargs.values()] if isinstance(a, np.ndarray)]
    arrays += [a.array for a in [*args, *kwargs.values()] if isinstance(a, PaletteArg)]
    before = [a.tobytes() for a in arrays]

    def call() -> Json:
        built = [construct(module, a) for a in args]
        built_kw = {k: construct(module, v) for k, v in kwargs.items()}
        if case.call.startswith("Clipboard."):
            method = getattr(
                module.Clipboard.create_instance(), case.call.split(".")[1]
            )
            return array_outcome(method(*built, **built_kw))
        return array_outcome(getattr(module, case.call)(*built, **built_kw))

    def finish(result: Json) -> Json:
        if [a.tobytes() for a in arrays] != before:
            result["input_changed"] = True
        return result

    if not case.call.startswith("Clipboard."):
        return finish(outcome(call))
    if clipboard_binary is None:
        raise ValueError("a clipboard case needs its module's binary")
    with _capture_class()(clipboard_binary, case.failure) as sink:
        result = finish(outcome(call))
        if sink.callback_errors:
            raise RuntimeError(f"clipboard capture errors: {sink.callback_errors}")
        result["clipboard"] = {
            "data": {
                str(k): hashlib.sha256(v).hexdigest()
                for k, v in sorted(sink.data.items())
            },
            "calls": [list(c) for c in sink.calls],
            "leaked": len(sink.allocations - sink.transferred),
        }
    return result


def case_json(case: ImageCase) -> Json:
    row: dict[str, Json] = {
        "id": case.id,
        "cell": case.cell,
        "call": case.call,
        "args": case.args,
    }
    if case.kwargs:
        row["kwargs"] = case.kwargs
    if case.failure:
        row["failure"] = case.failure
    return row


def case_from_json(row: Json) -> ImageCase:
    return ImageCase(
        row["id"],
        row["cell"],
        row["call"],
        row["args"],
        row.get("kwargs", {}),
        row.get("failure"),
    )


def _hashable(value: object) -> str:
    hash(value)  # raises TypeError for an unhashable value
    return "hashable"


def surface_probe(module: ModuleType) -> Json:
    """Names, docs, signatures and enum behaviour of the image and clipboard API;
    each key holds an outcome, so a missing name is recorded, not fatal."""
    out: dict[str, Json] = {}
    functions = (
        "fill_alpha_fragment_blur", "fill_alpha_extend_color", "fill_alpha_nearest_color",
        "binary_threshold", "esdf", "pixel_art_upscale", "fast_gamma", "quantize",
        "ordered_dither", "error_diffusion_dither", "riemersma_dither", "resize",
    )  # fmt: skip
    for name in functions:

        def function(name: str = name) -> Json:
            fn = getattr(module, name)
            return {"doc": fn.__doc__, "text_signature": getattr(fn, "__text_signature__", None),
                    "type": type(fn).__name__}  # fmt: skip

        out[name] = outcome(function)
    classes = (
        "UniformQuantization",
        "PaletteQuantization",
        "DiffusionAlgorithm",
        "ResizeFilter",
        "Clipboard",
    )
    for name in classes:

        def class_(name: str = name) -> Json:
            cls = getattr(module, name)
            members = sorted(n for n in dir(cls) if not n.startswith("__"))
            return {"module": cls.__module__, "qualname": cls.__qualname__, "doc": cls.__doc__,
                    "text_signature": getattr(cls, "__text_signature__", None), "members": members}  # fmt: skip

        out[name] = outcome(class_)
    for name, members in (
        ("DiffusionAlgorithm", DIFFUSIONS),
        ("ResizeFilter", FILTERS),
    ):
        for member in members:

            def enum_member(name: str = name, member: str = member) -> Json:
                cls = getattr(module, name)
                m = getattr(cls, member)
                return {
                    "int": outcome(lambda: int(m)),
                    "eq_int": outcome(lambda: m == int(m)),
                    "eq_self": outcome(lambda: m == getattr(cls, member)),
                    "hash": outcome(lambda: _hashable(m)),
                    "repr": repr(m),
                    "str": str(m),
                    "value": outcome(lambda: m.value),
                    "name": outcome(lambda: m.name),
                    "type": type(m).__qualname__,
                }

            out[f"{name}.{member}"] = outcome(enum_member)
        out[f"{name}.construct"] = outcome(
            lambda name=name: repr(getattr(module, name)(0))
        )
        out[f"{name}.ne_other"] = outcome(
            lambda name=name, members=members: (
                getattr(getattr(module, name), members[0])
                != getattr(getattr(module, name), members[1])
            )
        )
    for n in (2, 3, 256, 1, 0, -1, 1 << 32, 2.5, "4"):
        out[f"UniformQuantization({n!r})"] = outcome(
            lambda n=n: module.UniformQuantization(n).colors_per_channel
        )
    out["UniformQuantization.eq"] = outcome(
        lambda: module.UniformQuantization(4) == module.UniformQuantization(4)
    )
    out["UniformQuantization.setattr"] = outcome(
        lambda: setattr(module.UniformQuantization(4), "colors_per_channel", 3)
    )
    rng = random.Random(f"{corpus.SEED}:surface")
    for k, ch in enumerate((1, 3, 4, 2, "2d", 5)):
        recipe = array_recipe(
            rng, f"surface/{k}", "palette", ch, k, shape=(1, 6), specials=False
        )
        pal = make_input(recipe)

        def palette(pal: np.ndarray = pal) -> Json:
            q = module.PaletteQuantization(pal)
            return {
                "channels": outcome(lambda: q.channels),
                "colors": outcome(
                    lambda: (
                        q.colors if not callable(q.colors) else ["callable", q.colors()]
                    )
                ),
            }

        out[f"PaletteQuantization(ch{ch})"] = outcome(palette)
    duplicates = np.zeros((1, 9, 3), np.float32)
    out["PaletteQuantization(duplicates)"] = outcome(
        lambda: module.PaletteQuantization(duplicates).colors
    )
    out["Clipboard()"] = outcome(lambda: type(module.Clipboard()).__name__)
    out["create_instance.type"] = outcome(
        lambda: type(module.Clipboard.__dict__["create_instance"]).__name__
    )
    return out


# ---------------------------------------------------------------------------
# record (the manifest) and images (the comparison)

SCHEMA = "chaiNNer-C/chainner_ext-conformance/v1"
MANIFEST = OUT_DIR / "manifest.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def regex_summary() -> Json:
    lines = read_jsonl(CORPUS)
    expected = read_jsonl(EXPECTED)
    families = Counter(family_of(line["id"]) for line in lines)
    return {
        "files": {
            p.name: {"sha256": sha256_file(p)}
            for p in (CORPUS, EXPECTED, MODULE_CASES, TOOBIG, EXCLUSIONS)
        },
        "cases": len(lines),
        "ops": sum(len(line["ops"]) for line in lines),
        "module_cases": len(read_jsonl(MODULE_CASES)),
        "families": dict(sorted(families.items())),
        "results": dict(sorted(result_kinds(lines, expected).items())),
        "toobig": json.loads(TOOBIG.read_text(encoding="utf-8")),
    }


def _write_manifest(manifest: dict[str, Json], rows: list[Json]) -> None:
    """The manifest as JSON with one image case per line."""
    head = json.dumps(manifest, indent=1, ensure_ascii=False)
    assert head.endswith("\n}")
    body = ",\n".join("  " + json.dumps(r, ensure_ascii=False) for r in rows)
    text = head[:-2] + ',\n "cases": [\n' + body + "\n ]\n}\n"
    MANIFEST.write_text(text, encoding="utf-8", newline="\n")


def record() -> int:
    """Runs every image case on the real module and writes manifest.json (with the
    regex files' hashes and counts)."""
    real = load_real()
    cases, floors = build_image_cases()
    binary = clipboard_binary(real)
    rows: list[Json] = []
    cells: dict[str, Counter[str]] = {}
    exclusions: list[Json] = []
    for case in cases:
        kind, reason = classify(case)
        result = run_image_case(real, case, binary)
        if kind == "tie" and "ok" in result:
            # upstream's answer differs between runs: keep its form, not its digest
            result["ok"] = {k: v for k, v in result["ok"].items() if k != "sha256"}
        row = case_json(case) | {"class": kind, "outcome": result}
        rows.append(row)
        count = cells.setdefault(case.cell, Counter())
        count["cases"] += 1
        count[kind] += 1
        count["raises" if "raise" in result else "returns"] += 1
        if kind != "compare":
            exclusions.append(
                {"id": case.id, "class": kind, "reason": reason, "evidence": result}
            )
    manifest = {
        "schema": SCHEMA,
        "reference": {
            "module": "chainner_ext 0.3.10",
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "rules": {
            "nan": "every NaN is mapped to 0x7fc00000 before hashing (golden_kernels.digest 'mixed')",
            "compare": "outputs by SHA-256 of their bytes; raises by type and message, "
            "type only for " + ", ".join(sorted(TYPE_ONLY)),
            "tie": "counted, not compared; the candidate must give equal results twice",
            "corrected": "palettes: the candidate must return (not raise) and give equal "
            "results twice; clipboard image failures: the recorded outcome with the "
            "corrected tail (no DeleteObject, GlobalFree before CloseClipboard, leaked 0), "
            "twice",
        },
        "regex": regex_summary(),
        "surface": surface_probe(real),
        "cells": {
            cell: {"floor": floors[cell], **dict(sorted(count.items()))}
            for cell, count in cells.items()
        },
        "exclusions": exclusions,
    }
    _write_manifest(manifest, rows)
    totals = Counter[str]()
    for count in cells.values():
        totals.update(count)
    print(f"manifest: {len(rows)} image cases in {len(cells)} cells; " +
          ", ".join(f"{k} {v}" for k, v in sorted(totals.items())))  # fmt: skip
    return 0


def load_manifest() -> Json:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _same_image_outcome(a: Json, b: Json) -> bool:
    if set(a) != set(b):
        return False
    core_a = {k: v for k, v in a.items() if k != "clipboard"}
    core_b = {k: v for k, v in b.items() if k != "clipboard"}
    return same_outcome(core_a, core_b) and a.get("clipboard") == b.get("clipboard")


def check_candidate_case(
    cand: ModuleType, row: Json, binary: Path | None
) -> str | None:
    """None when the candidate meets the row's class rule, else the reason. A
    clipboard case without a binary to capture is not run (the live clipboard is
    never touched) and fails."""
    case = case_from_json(row)
    if case.call.startswith("Clipboard.") and binary is None:
        return "not run: no loaded binary of the module imports OpenClipboard"
    have = run_image_case(cand, case, binary)
    kind = row["class"]
    if kind == "compare":
        if _same_image_outcome(have, row["outcome"]):
            return None
        return f"want {json.dumps(row['outcome'])[:500]}\n    got  {json.dumps(have)[:500]}"
    again = run_image_case(cand, case, binary)
    if again != have:
        return f"{kind}: two runs differ"
    if kind == "corrected" and case.call.startswith("Clipboard."):
        want = corrected_clipboard(row["outcome"])
        if _same_image_outcome(have, want):
            return None
        return f"corrected: want {json.dumps(want)[:500]}\n    got  {json.dumps(have)[:500]}"
    if kind == "corrected" and "raise" in have:
        return f"corrected: the candidate raises {have['raise']}"
    return None


def corrected_clipboard(outcome: Json) -> Json:
    """Consult 6 D-1: upstream's recorded failed image transfer with the corrected
    tail: no DeleteObject on the HGLOBAL, GlobalFree before CloseClipboard, nothing
    leaked."""
    calls = [call for call in outcome["clipboard"]["calls"] if call != ["DeleteObject"]]
    if calls[-1] != ["CloseClipboard"]:
        raise ValueError(f"not a failed image transfer: {calls}")
    clipboard = outcome["clipboard"] | {
        "calls": [*calls[:-1], ["GlobalFree"], calls[-1]],
        "leaked": 0,
    }
    return outcome | {"clipboard": clipboard}


def images(candidate_name: str, only: str | None) -> int:
    """The real module live against the manifest (drift), and the candidate against
    the manifest under each case's class rule; counts per cell."""
    manifest = load_manifest()
    real = load_real()
    cand = load_candidate(candidate_name)
    rows = [r for r in manifest["cases"] if not only or r["id"].startswith(only)]
    real_binary = clipboard_binary(real)
    cand_binary: Path | None = None
    try:
        cand_binary = clipboard_binary(cand)
    except RuntimeError as exc:
        print(f"clipboard cases fail: {exc}")
    per_cell: dict[str, Counter[str]] = {}
    shown = 0
    for row in rows:
        count = per_cell.setdefault(row["cell"], Counter())
        count["cases"] += 1
        count[row["class"]] += 1
        live = run_image_case(real, case_from_json(row), real_binary)
        if row["class"] == "compare" and not _same_image_outcome(live, row["outcome"]):
            count["real_drift"] += 1
            if shown < 40:
                shown += 1
                print(f"{row['id']}: the real module moved from the manifest")
        if candidate_name == "real" and row["class"] != "compare":
            continue  # upstream's own ties and panics: the evidence, not a rule
        problem = check_candidate_case(cand, row, cand_binary)
        if problem is not None:
            count["diffs"] += 1
            if shown < 40:
                shown += 1
                print(f"{row['id']}: {problem}")
    surface_diffs = 0
    have_surface = surface_probe(cand)
    for key, want in manifest["surface"].items():
        if have_surface.get(key) != want:
            surface_diffs += 1
            if shown < 60:
                shown += 1
                print(
                    f"surface {key}:\n    want {json.dumps(want)[:400]}\n    got  {json.dumps(have_surface.get(key))[:400]}"
                )
    total = Counter[str]()
    for cell, count in sorted(per_cell.items()):
        total.update(count)
        if count["diffs"] or count["real_drift"]:
            print(
                f"  {cell:56s} cases {count['cases']:4d} diffs {count['diffs']:4d} drift {count['real_drift']:3d}"
            )
    print(f"images ({candidate_name}): {total['cases']} cases in {len(per_cell)} cells: "
          f"compared {total['compare']}, ties {total['tie']}, corrected {total['corrected']}; "
          f"diffs {total['diffs']}, real drift {total['real_drift']}, surface diffs {surface_diffs}")  # fmt: skip
    return 0 if not (total["diffs"] or total["real_drift"] or surface_diffs) else 1


def tie_proof(processes: int) -> int:
    """Runs the tie and corrected cases of the manifest on the real module in
    `processes` fresh processes and reports the cases whose digests differ."""
    runs: list[dict[str, Json]] = []
    for _ in range(processes):
        proc = subprocess.run(
            [sys.executable, "-B", __file__, "tie-run"], capture_output=True, check=True
        )
        runs.append(json.loads(proc.stdout.decode("utf-8")))
    ids = sorted(runs[0])
    differing = [i for i in ids if any(run[i] != runs[0][i] for run in runs[1:])]
    print(f"tie-proof: {len(ids)} tie/corrected cases, {processes} processes, "
          f"{len(differing)} differing across processes: {differing[:20]}")  # fmt: skip
    return 0


def tie_run() -> int:
    real = load_real()
    binary = clipboard_binary(real)
    out = {
        row["id"]: run_image_case(real, case_from_json(row), binary)
        for row in load_manifest()["cases"]
        if row["class"] != "compare"
    }
    sys.stdout.write(json.dumps(out))
    return 0


def pin_cpus(mask: int) -> None:
    """Pins this process (and so its children, the probe included) to mask, so the
    tool runs directly and its exit status stays real (a `start /affinity` wrapper
    reports 0)."""
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), mask):
        raise ctypes.WinError(ctypes.get_last_error())


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--affinity",
        type=lambda s: int(s, 0),
        default=0xFF00,
        help="CPU mask (CPUs 8-15)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("record-regex")
    probe = sub.add_parser("regex-probe")
    probe.add_argument("--probe", type=Path)
    probe.add_argument("--modes", default=",".join(MODES))
    probe.add_argument("--prefilter", default=",".join(PREFILTERS))
    probe.add_argument("--only")
    module = sub.add_parser("regex-module")
    module.add_argument("--candidate", choices=("real", "c"), default="c")
    sub.add_parser("record")
    image = sub.add_parser("images")
    image.add_argument("--candidate", choices=("real", "c"), default="c")
    image.add_argument("--only")
    proof = sub.add_parser("tie-proof")
    proof.add_argument("--processes", type=int, default=3)
    sub.add_parser("tie-run")
    args = parser.parse_args()
    pin_cpus(args.affinity)
    if args.command == "record-regex":
        return record_regex()
    if args.command == "regex-probe":
        return regex_probe(
            args.probe or find_probe(),
            args.modes.split(","),
            args.prefilter.split(","),
            args.only,
        )
    if args.command == "regex-module":
        return regex_module(args.candidate)
    if args.command == "record":
        return record()
    if args.command == "images":
        return images(args.candidate, args.only)
    if args.command == "tie-proof":
        return tie_proof(args.processes)
    return tie_run()


if __name__ == "__main__":
    sys.exit(main())
