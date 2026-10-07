"""Paired CPU-only benchmark of two backend trees, gated on exact outputs.

Usage, from the project root:
  native\\.venv\\Scripts\\python.exe -B native\\tools\\bench.py
      [--a TREE] [--b TREE] [--cases CASE ...] [--repeats N] [--record-noise]
      [--env-a NAME=VALUE] [--env-b NAME=VALUE] [--profile]
      [--python-a PATH] [--python-b PATH] [--png-compare {sha256,decoded}]
      [--cross-stack]

Defaults: A is frozen baseline B0 and B is the packaged backend. Each case runs
one warm-up pair (AB), then N measured pairs in alternating order (BA, AB, ...).
Outputs and final UI state must be identical across every trial of a case (of
one side, under --cross-stack; see Cross-stack runs), or the run fails and
result.json keeps the failing trial. Records go to
native/reports/bench-<UTC>/ and media to F:\\chaiNNer-C-diagnostics\\bench-<UTC>\\; a
successful run keeps only each trial's JSON there, a failed run keeps everything.

Pairs: N (--repeats) must be even, so A runs first in exactly half the pairs and
order effects cancel. Verdicts and noise floors need at least MIN_PAIRS = 6 pairs;
with fewer, every verdict is "unknown".

Verdicts compare the median paired ratio (A seconds / B seconds) with two floors
from bench-data/noise-floor.json (see Noise floors): the case's shared noise
floor, which flags a change, and its current floor, beyond which a regression
blocks:
  win         B faster beyond both floors, in a majority of the pairs
  regression  B slower beyond both floors (no majority needed, so a bimodal
              slowdown cannot slip through)
  regression (within current A/A)
              B slower beyond the noise floor but not beyond the current floor:
              flagged and confirmed like a regression, but it does not block;
              the run warns with both floors, for the stand-in
  neutral     anything else: within the noise floor, or faster but not a win
  unknown     no usable noise floor: none recorded, fewer than MIN_LAUNCHES
              launches recorded, or fewer than MIN_PAIRS pairs
Without a current floor, the noise floor alone decides.

Confirmation: win and regression in a single run's summary.md are provisional,
and every such summary.md says so. Each backend launch carries its own small,
constant speed bias, which more pairs within that launch never average away, so
over many cases a false win or regression is expected now and then. Unless
--record-noise or --profile is set, the harness therefore automatically re-runs
the flagged cases (a win or either regression; CPU verdicts never flag a case)
in a confirmation run on freshly started backends, with the same trees, repeats
and per-side environments; that run's summary.md names the run it confirms. A
missed regression is the costlier error, so the rules are asymmetric: a
regression is final when 2 of the 3 runs flag it; wins need the confirmation run
to repeat them.
  win          stays win only if the confirmation run is also win. If the
               confirmation run flags a regression instead, the case goes to the
               tie-break; any other outcome is unconfirmed, with no tie-break
  regression   either label stays as it is if the confirmation run gives the
               same label. If the confirmation run gives the other label, says
               anything else or lacks the case, the case goes to the tie-break
  tie-break    ONE more fresh run, labelled as the tie-break in its summary.md,
               covering just the cases sent to it, together; each is a
               regression if 2 of the 3 runs flagged one, otherwise unconfirmed
  unconfirmed  anything else that was flagged: inconclusive, NOT neutral. An
               unconfirmed regression is not cleared.
A final regression is "regression", which blocks, when 2 of the 3 runs put it
beyond the current floor, and otherwise "regression (within current A/A)", which
does not block and goes to the stand-in with both floors (the warnings of the
runs that gave it name them).
Neutral and unknown verdicts stay as the first run gave them. confirmation.json
and confirmation.md go into the first run's directory, with each run's verdict
and paired change per flagged case, and the same table is appended to the first
run's summary.md under "Confirmed verdicts"; the harness prints each flagged
case's final verdict. Only the final column counts as a claim. Always quote the
confirmation run's paired change, never the first run's: the first run picked the
case for looking large, so its magnitudes are inflated by selection. Re-running
whole comparisons until a win confirms counts each attempt and raises the
false-claim rate. The runs share machine conditions (thermal state, persistent
background load), so replication removes per-launch bias but not those. A
--record-noise or --profile run is never confirmed.

Noise floors: a floor is built from at least MIN_LAUNCHES = 3 A/A launches on
identical trees, each with an even number of at least MIN_PAIRS pairs. A launch
stores, per measured case, its largest |ln q| over position-balanced quads q:
the geometric mean of a B-first pair and the A-first pair after it, which
cancels the first-vs-second position effect. A case's floor is the maximum of
its launch floors over all recorded launches, kept in bench-data/
noise-floor.json with each launch's run, repeats, floor and background load.
Until a case has 3 launches its floor stays unusable and its verdict "unknown".
The shared throughput floors ("floors") stay as recorded through SP3, so that
throughput verdicts remain comparable across sub-projects: the harness never
writes them, and re-basing them is the stand-in's decision, with no harness
flag. --record-noise records CPU floors instead (see CPU per item). A case's
current floor, by contrast, follows today's machine: the largest throughput
floor among its A/A launches recorded for the run's K setting, which keep it
beside their CPU floors in cpu_floors.<setting> (see current_floors). It applies
once the case has MIN_LAUNCHES launches there, and only where the shared floor
is usable; result.json records it as current_floors and summary.md shows it
beside the noise floor. A --record-noise run reads it before its own launch
merges into those launches. Recording a run name that is already stored replaces
that launch, and the first launch recorded for a case replaces an old-format
entry (one max-pair floor and no "launches"), which is not carried over. A
--record-noise run's CPU verdicts use the merged launches, its own included;
they are not guaranteed neutral and do not confirm the floors. A floor is
confirmed only by a later holdout A/A run without --record-noise, whose verdicts
should all be neutral. Judge the holdout by the FIRST run's summary verdicts,
not by confirmation.md: a win or regression there counts against the floors even
when the confirmation runs do not repeat it.

CPU per item: a trial's job_cpu_seconds (the owned backend job, FFmpeg children
included) over its item count. A pair's CPU ratio is A's CPU per item over B's,
so a positive CPU/item change means B used less CPU. It is judged like
throughput, by the same quads, verdict rule, MIN_PAIRS and MIN_LAUNCHES, against
its CPU floor alone (no current floor); a case whose measured trials do not all
have job_cpu_seconds > 0 has no CPU summary and CPU verdict "unknown". CPU
verdicts are information only: Job-object CPU is charged per scheduler tick and
inflated by helper wake-ups, so the A/A CPU floors are too wide to judge a
kernel. They stay in summary.md and result.json's cpu_verdicts and never send a
case to the confirmation run. CPU floors depend on K, so they are kept per K
setting: "k<n>" when CHAINNER_C_ITEM_WINDOW in the harness's environment forces
K = n (read as the item window reads it), else "auto". A run's CPU verdicts use
the CPU floors of its own K setting. --record-noise merges this launch's CPU
floors into noise-floor.json cpu_floors.<setting>, each launch storing its
throughput floor beside its CPU floor, and never writes the shared floors; a
launch whose throughput floor exceeds a case's shared floor prints a WARNING
with its background load, for the stand-in.

Page faults: a trial also records job_page_faults, the page faults (soft and hard)
of every process in the owned backend job, FFmpeg children included, over the
same span as job_cpu_seconds. summary.md shows one line per run, the median of
a case's measured trials of job_page_faults over the item count for A and B,
n/a for a case whose measured trials do not all have the field (a result.json
from before it). It is information only: never a verdict, a floor or a gate
input, and never sent to the confirmation run.

Background load: each trial records background_busy_percent, the machine's busy
CPU minus the owned backend jobs' CPU (backend plus FFmpeg children). A WARNING
is printed and kept in result.json and summary.md when the machine is over 10%
busy before the run, and when a case's median background load exceeds the highest
load any launch of one of its floors was recorded under by more than
BACKGROUND_MARGIN = 5 points: checked separately against its shared throughput
floor and against its CPU floor of the run's K setting, and naming the floor.
The current floor comes from the CPU floor's launches, so the second check
covers it too and says so. Such verdicts are less reliable; rerun on a quieter
machine.

Item window: each trial records in result.json the item-window decisions its
backend logged during the request (item_windows, one per generator group; B0
logs none), each with the window's counts when K > 1 (items, ahead, jobs,
replayed, awaited, discarded). An empty item_windows on a candidate trial
means K is not proven, never K = 1. CHAINNER_C_ITEM_WINDOW in the harness's
environment reaches both backends and forces K (1 = sequential);
environment.affinity_cpus is the CPU count the harness and its backends may use.

Per-side environment: --env-a NAME=VALUE and --env-b NAME=VALUE, each
repeatable, set a variable in one backend's environment only, applied last
(after the isolation and private-directory variables); the confirmation and
tie-break runs keep them. Each backend's own variables are recorded in
result.json as backends.<label>.env and listed in summary.md. Names are
upper-cased, as Windows reads them without case, and refused when given twice
or reserved: CHAINNER_C_ITEM_WINDOW and CHAINNER_C_PROFILE (the harness controls
them for both sides), PYTHONPATH and PYTHONHOME (dropped from both), and the
variables the backend sets for its isolation and private directories.
--record-noise refuses both options: a noise floor needs both sides launched
alike. CHAINNER_C_ISA in the harness's environment, whatever its value, refuses
the start: it would reach both backends; set it per side. A side that sets
CHAINNER_C_ISA must log its ISA line (see ISA record) within 10 s of becoming
ready, or the run fails, since a tree that ignores the variable would run the
same code on both sides.

Per-side interpreter: --python-a PATH and --python-b PATH run one backend's
run.py under another interpreter, such as an installed chaiNNer's own Python;
both default to the packaged port's (bench_backend.PYTHON). A side whose
interpreter, given or default, is not an existing python.exe refuses the start
before anything is created. Every backend runs with -B and
PYTHONDONTWRITEBYTECODE=1, so a read-only tree or interpreter gets no bytecode,
and with PIP_REQUIRE_VIRTUALENV=1: a host's dependency installer (pip or
chainner_pip install, at startup) then exits before installing, so a needed
install fails the start instead of writing into the interpreter, while pip list
still works. The tree is hashed before and after (backend_unchanged); the
interpreter's Lib\\site-packages is fingerprinted by its top-level entries' names
and mtimes (interpreter_unchanged); a change in either fails the run. The
interpreters are recorded in result.json as pythons (and backends.<label>.python)
and listed in summary.md; the confirmation and tie-break runs keep them, and
--record-noise refuses them, since noise floors need both sides launched alike.

Oracle side (parity runs): when A is make_oracle.py's tree (native/build/oracle/
src), the runtime verifiers' guard applies before anything starts. The oracle's
four chainner_ext hashes must equal the candidate's: the package manifest's when
B is a package's resources/src, B's own files otherwise. A runs on the
provisioned runtime: the one the package was copied from (python_stack.runtime,
while the project's lock is the one it recorded), else this project's
native/runtime/cpython-<the lock's interpreter>. That is --python-a's default;
another --python-a is refused, and the oracle as B is refused. result.json
records the oracle's record as oracle (None for any other A).

PNG comparison: --png-compare sha256, the default, requires every output PNG's
bytes to match. --png-compare decoded requires its pixels plus colour/orientation
chunks to match instead: each PNG is read with cv2.imread(path, IMREAD_UNCHANGED)
and compared by shape, dtype and every byte of the array, with no tolerance, and
by the type and payload SHA-256 of each gAMA, cHRM, sRGB, iCCP, sBIT, tRNS or eXIf
chunk it holds, so two encoders of the same pixels and metadata agree
(bench_oracle.png_outputs). Video outputs keep their decoded-frame oracle in both
modes. result.json records the mode as png_compare and summary.md
names it; the confirmation and tie-break runs keep it.

Cross-stack runs: --cross-stack compares two Python stacks, such as a package on
CPython 3.11 against one on 3.14, whose outputs need not match. It refuses the
start unless the two sides' interpreters report different sys.version_info[:2],
each probed by running its python.exe (-I -B) before anything is created, and it
refuses --record-noise. The schema_sha256 gate stays hard. Within a side, every
trial must equal that side's first trial, in outputs and final UI state, or the
run fails as usual. Across the sides, each case records cross_stack.cases.<case>
in result.json: {outputs_equal, files: [{name, equal, max_abs,
differing_pixels}], final_state_equal, first_difference}, from the two sides'
first trials, compared decoded. A PNG is equal when bench_oracle.png_outputs'
decoded records match (pixels plus colour/orientation chunks); max_abs and
differing_pixels are 0 for an equal file and, for a differing PNG on both
sides, verify_runtime.compare_images' maximum channel difference and count of
differing pixels, decoded in the harness's interpreter; otherwise None. A video
is equal when its decoded-frame snapshot matches. first_difference names the
first differing output, else where the final UI state differs, else None. A
difference between A and B fails nothing; a final UI state that differs is
also a WARNING, so summary.md flags it, and summary.md lists every case's
comparison under "Cross-stack". Verdicts are judged as usual and the floors are
only read. The run is labelled "cross-stack <A's version> → <B's version>,
recorded, not gating" in result.json (cross_stack.label, beside the probed
versions) and summary.md; an ordinary run records cross_stack as None. The
confirmation and tie-break runs keep the mode.

Profile runs: --profile sets CHAINNER_C_PROFILE=1 on both sides. A tree with the
timer (it holds nodes/impl/native_profile.py) then logs one line per /run after
its executor finishes, "native profile: <json>", the JSON with sorted keys and no
spaces, mapping each timed name to {"calls", "inclusive_ns", "exclusive_ns",
"max_ns"}: inclusive is the elapsed time, exclusive the elapsed time minus timed
children on the same thread, max the largest exclusive time of one call. A tree
whose timer attaches the DLL's pool counters adds "pool", {"dispatches", "wakes",
"empty_wakes"}: their deltas over the /run, not a timed name. Each
trial records that table as native_profile, beside item_windows, parsed after the
SSE wait and outside the timed region; the harness does not interpret its keys.
A trial of a tree with the timer must log exactly one table, waited for up to
10 s; a tree without it, such as B3, records None. A table in a trial that is
not profiled (no --profile, or a tree without the timer) fails the run: the
timer was on where it must be off. --profile refuses --record-noise and needs at
least one tree with the timer. Every timing carries the timer's overhead, so a
profile run applies no floors (every throughput and CPU verdict is "unknown",
and no floor's background load is compared), is never confirmed and is never a
gate; result.json records "profile": true. Without --profile, CHAINNER_C_PROFILE
in the harness's environment, whatever its value, refuses the start: it would
reach both backends and turn the timer on in a gate.

ISA record: each backend's startup line "native isa=<level> (requested <x>, cpu
<max>)" is recorded in result.json as backends.<label>.native_isa, {"level",
"requested", "cpu"}, from the first such line of its backend.log; a tree that
logs none, such as B3, records None.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import statistics
import sys
import time
import traceback
from collections.abc import Mapping
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

import bench_baseline
import bench_cases
import bench_oracle
import bench_stats
import package_files
import package_manifest
import psutil
from bench_backend import (
    ISA_VARIABLE,
    ISOLATION,
    PRIVATE_DIRS,
    PROJECT,
    PYTHON,
    Backend,
    log_matches,
)
from bench_fixtures import Fixtures, pinned_output_names, prepare, verify_unchanged
from verify_framework_runtime import OPTIONS, source_hashes
from verify_runtime import (
    CHAINNER_EXT,
    ORACLE,
    check_success,
    compare_images,
    oracle_source,
    oracle_tree,
    request,
    run_owned,
    sse_contract,
    write_json,
)
from verify_video_runtime import ffmpeg_children

DEFAULT_A = PROJECT / "native/reports/baselines/B0-20261001/src"
DEFAULT_B = PROJECT / "out/chaiNNer-C/resources/src"
DIAGNOSTICS = Path(r"F:\chaiNNer-C-diagnostics")
NOISE = bench_cases.DATA / "noise-floor.json"
LABELS = ("A", "B")
# The spec requires at least 6 AB/BA pairs for noise floors and verdicts.
MIN_PAIRS = 6
# A noise floor is usable for verdicts only once its case has this many launches.
MIN_LAUNCHES = 3
# Percentage points of background load above a floor's recorded load that make
# that floor a poor yardstick for the current run.
BACKGROUND_MARGIN = 5.0
# The two regression labels: beyond both floors (it blocks), and beyond the shared
# floor only (bench_stats.WITHIN_CURRENT).
REGRESSIONS = ("regression", bench_stats.WITHIN_CURRENT)
# Verdicts that count as a claim only after they repeat on freshly started backends.
FLAGGED = ("win", *REGRESSIONS)
# The item window's developer override of K (item_window.ITEM_WINDOW_VARIABLE; that
# module is not importable here). The harness's value reaches both backends.
ITEM_WINDOW_VARIABLE = "CHAINNER_C_ITEM_WINDOW"
# The timer's switch (native_profile.VARIABLE, not importable here); only --profile
# may set it.
PROFILE_VARIABLE = "CHAINNER_C_PROFILE"
# The timer's table, one line per /run (P6).
NATIVE_PROFILE = re.compile(r"native profile: (\{.*\})$")
# Names --env-a and --env-b may not set: the variables the harness controls for both
# sides, and the ones the backend drops or sets for its isolation.
RESERVED = frozenset(
    {
        ITEM_WINDOW_VARIABLE,
        PROFILE_VARIABLE,
        "PYTHONPATH",
        "PYTHONHOME",
        *ISOLATION,
        *PRIVATE_DIRS,
    }
)
METHOD = (
    "One warm-up per tree, then N pairs in alternating order; whole warm-backend "
    "/run requests; validation and hashing outside the timer"
)
LEGEND = (
    "Paired change = median of per-pair A/B time ratios \N{MINUS SIGN} 1; "
    "positive = B faster. "
    f"Verdicts need ≥{MIN_PAIRS} pairs; win = beyond both floors with a majority "
    "of pairs, regression = beyond the noise floor (within current A/A unless also "
    "beyond the current floor). "
    "Noise floor = largest |ln| of position-balanced pair means "
    "(B-first \N{MULTIPLICATION SIGN} A-first) "
    f"across ≥{MIN_LAUNCHES} A/A launches; current floor = the same over this K "
    "setting's A/A launches; both shown as upward/downward bounds."
)
CPU_LEGEND = (
    "CPU/item change = median of per-pair A/B CPU-per-item ratios "
    "\N{MINUS SIGN} 1; positive = B uses less CPU; CPU verdicts are information "
    "only and never confirmed."
)
IN_SAMPLE_NOTE = (
    "- Recording run: CPU verdicts are in-sample (this launch is part of its own "
    "CPU floors) and do not confirm them; throughput verdicts use the shared "
    "floors, which this run leaves unchanged, and the current floors as recorded "
    "before it."
)
PROVISIONAL_NOTE = (
    "- win/regression in this table are provisional; a claim needs the Final "
    "column of the first run's confirmation.md."
)
PROFILE_NOTE = (
    f"- Profile run: {PROFILE_VARIABLE}=1 on both sides puts the timer's overhead in "
    "every timing, so no floors apply, every verdict is unknown, and the run is "
    "never confirmed or used as a gate."
)
# Closing paragraphs of confirmation.md; the module docstring explains each rule.
CONFIRMATION_NOTES = (
    (
        "Claims require the Final column: a regression counts if 2 of the 3 runs flag "
        "it, and a win counts only if the confirmation run repeated it (the tie-break "
        "run is launched only for regressions the confirmation run did not repeat with "
        "the same label and for wins it flipped to regression)."
    ),
    (
        "A regression blocks only when 2 of the 3 runs put it beyond the current floor "
        "(the A/A launches of the run's K setting); otherwise it is final as "
        f'"{bench_stats.WITHIN_CURRENT}", which does not block and goes to the '
        "stand-in with both floors, named in the warnings of the runs that gave it."
    ),
    (
        "Unconfirmed means inconclusive, not neutral; an unconfirmed regression is not "
        "cleared."
    ),
    (
        "Always quote the confirmation run's paired change, not the first run's: "
        "first-run magnitudes are inflated by selection."
    ),
    (
        "Re-running whole comparisons until a win confirms counts each attempt and "
        "raises the false-claim rate."
    ),
    (
        "The runs share machine conditions (thermal state, persistent background "
        "load); replication removes per-launch bias, not those."
    ),
)
# The executor's item-window lines in backend.log: a group's decision, and its
# window's counts when it closes (K > 1 only).
WINDOW_DECISION = re.compile(r"item window K=(\d+) \((.*)\)$")
WINDOW_SUMMARY = re.compile(r"item window K=(\d+): (.*)$")


def environment() -> dict:
    """Machine context. Background load inflates noise, so it is recorded."""
    memory = psutil.virtual_memory()
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": psutil.cpu_count(),
        "physical_cpus": psutil.cpu_count(logical=False),
        # As item_window.affinity_cpus (not importable here): macOS has no cpu_affinity.
        "affinity_cpus": (
            len(psutil.Process().cpu_affinity())
            if hasattr(psutil.Process, "cpu_affinity")
            else os.cpu_count() or 1
        ),
        "memory_total": memory.total,
        "memory_available": memory.available,
        "busy_percent_before": psutil.cpu_percent(interval=2.0),
    }


def require_identical(a: dict[str, str], b: dict[str, str]) -> None:
    """Noise floors are only meaningful when both sides run identical code."""
    differing = bench_baseline.changed_paths(a, b)
    if differing:
        raise ValueError(
            f"--record-noise needs identical trees ({len(differing)} files differ): "
            f"{differing[:10]}"
        )


def verify_if_baseline(tree: Path) -> None:
    """Refuse a frozen baseline whose files no longer match its manifest."""
    if tree.name == "src" and (tree.parent / "manifest.json").is_file():
        bench_baseline.verify(tree.parent)


def oracle_side(
    a: Path, b: Path, python_a: Path | None
) -> tuple[Path | None, dict | None]:
    """Side A's interpreter and the oracle's record when A is make_oracle.py's tree.

    The runtime verifiers' guard (verify_runtime.oracle_source) applies: the
    oracle's four chainner_ext hashes must equal the candidate's, which are the
    package manifest's when B is a package's resources/src and B's own files
    otherwise, and the oracle runs on the provisioned runtime, the one the
    package was copied from, else this project's. That interpreter is side A's
    default, and another --python-a is refused. The oracle is only ever side A;
    with any other A, python_a is returned unchanged and the record is None.
    """
    source = (ORACLE / "src").resolve()
    if b.resolve() == source:
        raise ValueError("The oracle is the reference: pass it as --a, never --b")
    if a.resolve() != source:
        return python_a, None
    package = b.parent.parent
    if (
        b.resolve() == (package / "resources/src").resolve()
        and (package / package_manifest.MANIFEST).is_file()
    ):
        manifest = package_manifest.load(package / package_manifest.MANIFEST)
        _, python, record = oracle_source(manifest, ORACLE)
    else:
        missing = sorted(name for name in CHAINNER_EXT if not (b / name).is_file())
        if missing:
            raise ValueError(f"Tree B holds no chaiNNer-C chainner_ext: {missing}")
        chainner_ext = {
            name: package_files.hash_file(b / name) for name in CHAINNER_EXT
        }
        _, record = oracle_tree(ORACLE, chainner_ext, f"tree B ({b})")
        python = package_manifest.provisioned_runtime(PROJECT) / "python.exe"
        if not python.is_file():
            raise ValueError(
                f"Oracle interpreter refused: the provisioned runtime is missing: {python}"
            )
        record = {**record, "python": str(python)}
    if python_a is not None and python_a.resolve() != python.resolve():
        raise ValueError(
            f"The oracle runs only on the provisioned runtime {python}; "
            f"--python-a {python_a} is refused"
        )
    return python, record


def k_setting(environ: Mapping[str, str]) -> str:
    """The K setting a run's CPU floors belong to: "k<n>" or "auto".

    CHAINNER_C_ITEM_WINDOW is read as item_window.window_size reads it: ASCII
    digits for an integer of at least 1 force K; anything else leaves K automatic.
    """
    value = environ.get(ITEM_WINDOW_VARIABLE, "")
    if value.isascii() and value.isdigit() and int(value) >= 1:
        return f"k{int(value)}"
    return "auto"


def parse_env(values: list[str] | None) -> dict[str, str]:
    """One side's --env-a or --env-b entries, NAME=VALUE each, as a dict.

    VALUE may hold "=". Windows reads variable names without case, so names are
    upper-cased as os.environ keeps them; a name given twice and a RESERVED name
    are refused.
    """
    env: dict[str, str] = {}
    for entry in values or []:
        name, separator, value = entry.partition("=")
        name = name.upper()
        if not separator or not name:
            raise ValueError(f"--env-a/--env-b take NAME=VALUE, got {entry!r}")
        if name in RESERVED:
            raise ValueError(
                f"--env-a/--env-b may not set {name}: the harness or the backend's "
                "isolation controls it"
            )
        if name in env:
            raise ValueError(f"--env-a/--env-b set {name} twice for one side")
        env[name] = value
    return env


def has_timer(tree: Path) -> bool:
    """Whether a backend tree holds the native timer (B3 and B0 do not)."""
    return (tree / "nodes" / "impl" / "native_profile.py").is_file()


def python_version(python: Path) -> tuple[int, int]:
    """The sys.version_info[:2] that python.exe reports, run with -I -B."""
    output = run_owned(
        [str(python), "-I", "-B", "-c", "import sys; print(*sys.version_info[:2])"],
        timeout=60,
    )
    major, minor = output.decode("ascii").split()
    return int(major), int(minor)


def cross_stack_record(a: dict, b: dict) -> dict:
    """A case's comparison of A's and B's first trials, decoded (see Cross-stack runs).

    a and b are the two sides' trial records. A video's "files" entry is already
    its decoded-frame snapshot; an image trial's PNGs are read back from its
    output directory as bench_oracle.png_outputs' decoded records, whatever the
    run's PNG mode. A differing PNG that both sides wrote is measured by
    verify_runtime.compare_images in this interpreter.
    """
    video = a["case"].startswith("video-")
    if video:
        files_a, files_b = a["files"], b["files"]
    else:
        files_a, files_b = (
            bench_oracle.png_outputs(Path(trial["output"]), "decoded")
            for trial in (a, b)
        )
    names = sorted(files_a.keys() | files_b.keys())
    differing = [name for name in names if files_a.get(name) != files_b.get(name)]
    pairs = [
        {
            "name": name,
            "baseline": str(Path(a["output"], name)),
            "converted": str(Path(b["output"], name)),
        }
        for name in differing
        if not video and name in files_a and name in files_b
    ]
    rows = (
        {row["name"]: row for row in compare_images(Path(sys.executable), pairs)}
        if pairs
        else {}
    )
    files = []
    for name in names:
        if name not in differing:
            max_abs, pixels = 0, 0
        elif name in rows:
            max_abs = rows[name]["max_channel_difference"]
            pixels = rows[name]["differing_pixels"]
        else:
            max_abs, pixels = None, None
        files.append(
            {
                "name": name,
                "equal": name not in differing,
                "max_abs": max_abs,
                "differing_pixels": pixels,
            }
        )
    final_state_equal = a["final_state"] == b["final_state"]
    if differing:
        first = f"outputs at {differing[0]}"
    elif not final_state_equal:
        where = bench_oracle.contract_difference(a["final_state"], b["final_state"])
        first = f"final UI state at {where}"
    else:
        first = None
    return {
        "outputs_equal": not differing,
        "files": files,
        "final_state_equal": final_state_equal,
        "first_difference": first,
    }


def cross_stack_table(records: dict[str, dict]) -> str:
    """Markdown: each case's cross-stack record, a differing final UI state flagged."""
    rows = [
        "| Case | Outputs | Final UI state | First difference |",
        "| --- | --- | --- | --- |",
    ]
    for case, record in records.items():
        files = record["files"]
        differing = [item for item in files if not item["equal"]]
        if differing:
            largest = max(
                (item["max_abs"] for item in differing if item["max_abs"] is not None),
                default=None,
            )
            outputs = (
                f"differ: {len(differing)} of {len(files)} files, max abs "
                f"{'n/a' if largest is None else largest}"
            )
        else:
            outputs = f"equal ({len(files)} files)"
        state = "equal" if record["final_state_equal"] else "**differs** (flagged)"
        first = record["first_difference"] or "-"
        rows.append(f"| {case} | {outputs} | {state} | {first} |")
    return "\n".join(rows)


def merge_floors(
    entries: dict[str, dict],
    floors: dict[str, float],
    run: str,
    repeats: int,
    backgrounds: dict[str, float | None],
    throughput_floors: dict[str, float],
) -> dict[str, dict]:
    """Add this launch to each measured case's launches and keep every other case.

    entries is one section of noise-floor.json, case by case (the CPU floors of
    one K setting). A case's floor is the maximum floor over its launches. Each
    launch stores its case's throughput floor, kept for the record and never
    merged, and its median background load, which later runs at that K setting
    compare against (see background_warnings). A launch of an already stored run
    name replaces that launch. A case's entry without "launches" is the pre-quad
    format, a single max-pair floor, and is discarded instead of carried over.
    """
    merged = dict(entries)
    for case, floor in floors.items():
        launches = {
            item["run"]: item for item in merged.get(case, {}).get("launches", [])
        }
        launches[run] = {
            "run": run,
            "repeats": repeats,
            "floor": floor,
            "throughput_floor": throughput_floors[case],
            "background_busy_percent": backgrounds[case],
        }
        merged[case] = {
            "floor": max(item["floor"] for item in launches.values()),
            "launches": list(launches.values()),
        }
    return merged


def recorded_noise() -> dict:
    """The stored noise-floor file, or an empty one before any floor is recorded."""
    if not NOISE.is_file():
        return {"floors": {}}
    return json.loads(NOISE.read_text(encoding="utf-8"))


def usable_floors(
    summary: dict[str, dict], entries: dict[str, dict]
) -> dict[str, float]:
    """Floors apply only to cases measured with enough pairs and recorded launches.

    A case needs at least MIN_PAIRS pairs in this run and MIN_LAUNCHES launches in
    its entry. Pre-quad entries have no launches, so they never apply.
    """
    return {
        case: entries[case]["floor"]
        for case, item in summary.items()
        if case in entries
        and item["pairs"] >= MIN_PAIRS
        and len(entries[case].get("launches", [])) >= MIN_LAUNCHES
    }


def floor_sources(floors: dict[str, float], entries: dict[str, dict]) -> str:
    """The launches behind the usable floors, as summary.md names them."""
    runs = sorted(
        {launch["run"] for case in floors for launch in entries[case]["launches"]}
    )
    return f"{', '.join(runs)} ({len(runs)} launches)" if runs else "none usable"


def current_floors(noise: dict, setting: str) -> dict[str, float]:
    """Per case, its current floor at a K setting, from a noise-floor document.

    That is the largest throughput floor among the case's launches in
    cpu_floors.<setting>, the A/A launches recorded at that setting (see
    merge_floors). A case with fewer than MIN_LAUNCHES launches there has none.
    """
    return {
        case: max(launch["throughput_floor"] for launch in entry["launches"])
        for case, entry in noise.get("cpu_floors", {}).get(setting, {}).items()
        if len(entry.get("launches", [])) >= MIN_LAUNCHES
    }


def background_percent(
    busy_seconds: float, total_seconds: float, job_seconds: float
) -> float | None:
    """Machine CPU busy outside the owned backend jobs, as % of all CPU time.

    busy_seconds and total_seconds are machine-wide deltas summed over every
    logical CPU; job_seconds is the owned job's CPU time over the same span.
    """
    if not total_seconds:
        return None
    return 100 * max(0.0, busy_seconds - job_seconds) / total_seconds


def background_by_case(trials: list[dict]) -> dict[str, float | None]:
    """Per case, the median background_busy_percent of its measured trials."""
    medians: dict[str, float | None] = {}
    for case in dict.fromkeys(t["case"] for t in trials):
        loads = [
            t["background_busy_percent"]
            for t in trials
            if t["case"] == case
            and t["pair"] > 0
            and t["background_busy_percent"] is not None
        ]
        medians[case] = statistics.median(loads) if loads else None
    return medians


def page_faults_by_case(trials: list[dict]) -> dict[str, dict[str, float] | None]:
    """Per case, each side's median job_page_faults per item of its measured trials.

    A case is None, unknown, unless every one of its measured trials has the
    field: a trial recorded before it existed never counts as zero faults.
    """
    medians: dict[str, dict[str, float] | None] = {}
    for case in dict.fromkeys(t["case"] for t in trials):
        measured = [t for t in trials if t["case"] == case and t["pair"] > 0]
        recorded = all(t.get("job_page_faults") is not None for t in measured)
        medians[case] = (
            {
                label: statistics.median(
                    t["job_page_faults"] / t["count"]
                    for t in measured
                    if t["backend"] == label
                )
                for label in LABELS
            }
            if measured and recorded
            else None
        )
    return medians


def background_warnings(
    background: dict[str, float | None], entries: dict[str, dict], cpu: bool = False
) -> list[str]:
    """Cases loaded over BACKGROUND_MARGIN points beyond their floor's recorded load.

    entries are the shared throughput floors, or with cpu the CPU floors of the
    run's K setting; the message names that floor and verdict. The recorded load
    is the highest background_busy_percent among the case's launches. Launches
    without one never warn, and neither do entries without launches: those are
    pre-quad-format entries, which are unusable. Every measured case is checked,
    also one whose verdict is "unknown", because the load context is still
    informative.
    """
    floor_name, verdict_name = (
        ("CPU floor", "CPU verdict and current floor are")
        if cpu
        else ("noise floor", "verdict is")
    )
    warnings = []
    for case, load in background.items():
        recorded = max(
            (
                launch["background_busy_percent"]
                for launch in entries.get(case, {}).get("launches", [])
                if launch["background_busy_percent"] is not None
            ),
            default=None,
        )
        if load is None or recorded is None or load - recorded <= BACKGROUND_MARGIN:
            continue
        warnings.append(
            f"{case}: median background load {load:.1f}% exceeds the {recorded:.1f}% "
            f"its {floor_name} was recorded under by more than {BACKGROUND_MARGIN:g} "
            f"points; its {verdict_name} less reliable"
        )
    return warnings


def warn(result: dict, message: str) -> None:
    """Print a warning and keep it for result.json and summary.md."""
    result["warnings"].append(message)
    print(f"WARNING {message}", flush=True)


def item_windows(log: Path, offset: int) -> tuple[list[dict], int]:
    """The item-window entries logged after offset, and the log's end offset.

    A decision line starts an entry {"k", "reason"}; a summary line adds each of
    its "<n> <name>" counts to the entry of the decision just before it. A summary
    whose decision line is missing starts its own entry, with reason None. An
    empty list on a candidate trial means K is not proven, never that K = 1.
    """
    with log.open("rb") as stream:
        stream.seek(offset)
        data = stream.read()
    windows: list[dict] = []
    decided: dict | None = None  # the latest decision's entry, until its summary
    for line in data.decode("utf-8", errors="replace").splitlines():
        if decision := WINDOW_DECISION.search(line):
            decided = {"k": int(decision[1]), "reason": decision[2]}
            windows.append(decided)
        elif summary := WINDOW_SUMMARY.search(line):
            entry = decided
            if entry is None or entry["k"] != int(summary[1]):
                entry = {"k": int(summary[1]), "reason": None}
                windows.append(entry)
            for pair in summary[2].split(", "):
                count, name = pair.split(" ", 1)
                entry[name] = int(count)
            decided = None
    return windows, offset + len(data)


def native_profile(
    log: Path, offset: int, required: bool
) -> dict[str, dict[str, int]] | None:
    """The timer's table that one /run logged after offset, or None.

    The worker logs it after the executor finishes, and it reaches backend.log
    through the host's pipe, possibly after the response, so a required table is
    waited for up to 10 s. A required table that is missing, or more than one
    table after offset, raises. Not required (no --profile, or a tree without the
    timer), a missing table is None and a table raises: the timer was on where it
    must be off. The table's keys are not interpreted.
    """
    found = log_matches(log, offset, NATIVE_PROFILE, 10 if required else 0)
    if len(found) > 1:
        raise RuntimeError(
            f"{len(found)} native profile lines in {log} after offset {offset}; "
            "one /run logs one"
        )
    if found and not required:
        raise RuntimeError(
            f"{log} has a native profile line after offset {offset} in a trial that "
            "is not profiled: the timer was on"
        )
    if not found:
        if required:
            raise RuntimeError(
                f"{log} has no native profile line after offset {offset}, though "
                "the tree has the timer"
            )
        return None
    return json.loads(found[0][1])


def run_trial(
    backend: Backend,
    case: str,
    pair: int,
    fixtures: Fixtures,
    media: Path,
    profiled: bool,
    png_compare: str,
) -> dict:
    """One /run of a case on one backend, validated, as a trial record.

    profiled: the backend runs with the timer on and its tree has it, so the
    trial must log exactly one native_profile table. png_compare is the
    bench_oracle.png_outputs mode that identifies an image case's PNGs in
    "files"; a video case's file is its decoded-frame snapshot either way.
    """
    trial = "warmup" if pair == 0 else f"trial-{pair}"
    output = media / case / trial / backend.label
    output.mkdir(parents=True)
    sse = backend.events
    if sse is None:
        raise RuntimeError(f"Backend {backend.label} has no SSE stream")
    nodes, count, unit = bench_cases.graph(
        backend.schemas, case, fixtures.assets, output
    )
    payload = {"data": nodes, "options": OPTIONS, "sendBroadcastData": True}
    write_json(output / "request.json", payload)
    log = backend.directory / "backend.log"
    offset = log.stat().st_size
    start = len(sse.events)
    cpu_start = psutil.cpu_times()
    job_start = backend.cpu_seconds()
    faults_start = backend.page_faults()
    began = time.perf_counter_ns()
    response = request(backend.port, "/run", payload, timeout=300)
    seconds = (time.perf_counter_ns() - began) / 1e9
    job_end = backend.cpu_seconds()
    faults_end = backend.page_faults()
    cpu_end = psutil.cpu_times()
    # Everything below validates the trial and is outside the timed region.
    check_success(response)
    # The generator broadcasts again after its finish (the final static-output
    # restoration, which the oracle requires); its initial broadcast alone must
    # not end the wait.
    events = sse.wait_for(
        {n["id"] for n in nodes},
        start,
        timeout=15,
        expected_broadcasts={
            n["id"] for n in nodes if backend.schemas[n["schemaId"]]["outputs"]
        },
        broadcast_after_finish={
            n["id"]
            for n in nodes
            if backend.schemas[n["schemaId"]]["kind"] == "generator"
        },
    )
    write_json(output / "events.json", events)
    if sse.error:
        raise RuntimeError(f"SSE failure: {sse.error}")
    children = ffmpeg_children(backend.job)
    if children:
        raise RuntimeError(f"FFmpeg children outlived the request: {children}")
    completion = bench_oracle.validate_record(
        nodes, events, backend.schemas, count, output
    )
    if case.startswith("video-"):
        videos = list(output.glob("video.*"))
        if len(videos) != 1:
            raise RuntimeError(f"Expected one output video, found {videos}")
        snapshot = bench_oracle.video_snapshot(
            videos[0], bench_oracle.FFMPEG, bench_oracle.FFPROBE
        )
        if snapshot["semantic"]["decoded_frame_count"] != count:
            raise RuntimeError(
                f"Decoded {snapshot['semantic']['decoded_frame_count']} of {count} frames"
            )
        files = {videos[0].name: snapshot["semantic"]}
    else:
        files = bench_oracle.png_outputs(output, png_compare)
        if len(files) != count:
            raise RuntimeError(f"Wrote {len(files)} of {count} PNGs")
        if case == "resize-real-256" and set(files) != pinned_output_names():
            raise RuntimeError(
                "resize-real-256 processed files other than the pinned inputs"
            )
    total = sum(cpu_end) - sum(cpu_start)
    busy = total - (cpu_end.idle - cpu_start.idle)
    job_seconds = job_end - job_start
    # Parsed first: the worker logs the table after the executor, and the host
    # keeps the worker's line order, so once it is in the log, so are the
    # item-window lines of this run.
    table = native_profile(log, offset, profiled)
    return {
        "backend": backend.label,
        "case": case,
        "trial": trial,
        "pair": pair,
        "seconds": seconds,
        "count": count,
        "unit": unit,
        "throughput": count / seconds,
        "machine_busy_percent": 100 * busy / total if total else None,
        "job_cpu_seconds": job_seconds,
        "job_page_faults": faults_end - faults_start,
        "background_busy_percent": background_percent(busy, total, job_seconds),
        "output": str(output),
        "files": files,
        "final_state": sse_contract(events, output),
        "completion": completion,
        "native_profile": table,
        "item_windows": item_windows(log, offset)[0],
        "response": response,
    }


def compare(
    a: Path,
    b: Path,
    cases: list[str],
    repeats: int,
    record_noise: bool,
    confirms: str | None = None,
    tiebreak: bool = False,
    env_a: dict[str, str] | None = None,
    env_b: dict[str, str] | None = None,
    profile: bool = False,
    python_a: Path | None = None,
    python_b: Path | None = None,
    png_compare: str = "sha256",
    cross_stack: bool = False,
) -> Path:
    """Run the paired comparison and return its run directory.

    confirms names the first run that this run re-checks (see confirm), and
    tiebreak marks the re-check as the tie-break run instead of the confirmation
    run. Both are kept in result.json and named in summary.md. Ordinary runs leave
    them at None and False. env_a and env_b are the sides' own variables, refused
    unless parse_env would return them, and profile makes this a profile run;
    record_noise refuses both. python_a and python_b are the sides' interpreters,
    PYTHON when None, each refused unless an existing python.exe; record_noise
    refuses either. When a is make_oracle.py's tree, oracle_side guards it and
    python_a defaults to the provisioned runtime. png_compare is the
    bench_oracle.PNG_MODES entry that output PNGs must match under. cross_stack
    makes this a cross-stack run (see Cross-stack runs): each trial must equal
    its own side's first trial, and A against B is recorded per case by
    cross_stack_record, never failed; it refuses record_noise and two
    interpreters that python_version() finds to be one Python version.
    """
    # Every refusal comes first, so a refused start creates nothing.
    if PROFILE_VARIABLE in os.environ and not profile:
        raise ValueError(
            f"{PROFILE_VARIABLE} is set in the harness environment and would turn "
            "the timer on in both backends; unset it, or pass --profile for a "
            "profile run"
        )
    if ISA_VARIABLE in os.environ:
        raise ValueError(
            f"{ISA_VARIABLE} is set in the harness environment and would cap the ISA "
            "level of both backends; unset it, and set it on one side with --env-a "
            f"or --env-b {ISA_VARIABLE}=<level>"
        )
    # A direct call could bypass parse_env's rules: a reserved name, or a name in
    # lower case that the child would read as the upper-case one.
    for env in (env_a, env_b):
        if env and parse_env([f"{name}={value}" for name, value in env.items()]) != env:
            raise ValueError(f"side environments come from parse_env; got {env}")
    if profile and record_noise:
        raise ValueError(
            "--profile refuses --record-noise: the timer's overhead would enter "
            "the noise floors"
        )
    if cross_stack and record_noise:
        raise ValueError(
            "--cross-stack refuses --record-noise: noise floors need both sides "
            "launched alike, on one Python stack"
        )
    if record_noise and (env_a or env_b):
        raise ValueError(
            "--record-noise refuses --env-a and --env-b: noise floors need both "
            "sides launched alike"
        )
    if record_noise and (python_a is not None or python_b is not None):
        raise ValueError(
            "--record-noise refuses --python-a and --python-b: noise floors need "
            "both sides launched alike, under the packaged interpreter"
        )
    if png_compare not in bench_oracle.PNG_MODES:
        raise ValueError(
            f"--png-compare takes {' or '.join(bench_oracle.PNG_MODES)}, got "
            f"{png_compare!r}"
        )
    if profile and not (has_timer(a) or has_timer(b)):
        raise ValueError(
            "--profile needs a tree with the timer (nodes/impl/native_profile.py); "
            f"neither {a} nor {b} has it"
        )
    if record_noise and repeats < MIN_PAIRS:
        raise ValueError(
            f"--record-noise needs at least {MIN_PAIRS} pairs, got {repeats}"
        )
    if repeats % 2:
        raise ValueError(
            f"AB/BA balance needs an even number of pairs, got --repeats {repeats}"
        )
    python_a, oracle = oracle_side(a, b, python_a)
    pythons = {
        label: PYTHON if python is None else python
        for label, python in zip(LABELS, (python_a, python_b), strict=True)
    }
    for label, python in pythons.items():
        if python.name.lower() != "python.exe" or not python.is_file():
            raise ValueError(
                f"side {label}'s interpreter (--python-{label.lower()}) must be an "
                f"existing python.exe, got {python}"
            )
    cross: dict | None = None
    if cross_stack:
        versions = {label: python_version(python) for label, python in pythons.items()}
        names = {
            label: f"{major}.{minor}" for label, (major, minor) in versions.items()
        }
        if versions["A"] == versions["B"]:
            raise ValueError(
                "--cross-stack needs interpreters of different Python versions; A "
                f"and B both report {names['A']} ({pythons['A']}, {pythons['B']})"
            )
        cross = {
            "label": f"cross-stack {names['A']} \N{RIGHTWARDS ARROW} {names['B']}, "
            "recorded, not gating",
            "versions": names,
            "cases": {},
        }
    for tree in (a, b):
        verify_if_baseline(tree)
    if record_noise:
        require_identical(source_hashes(a), source_hashes(b))
    setting = k_setting(os.environ)
    timer = {PROFILE_VARIABLE: "1"} if profile else {}
    sides = {
        label: {**timer, **(env or {})}
        for label, env in zip(LABELS, (env_a, env_b), strict=True)
    }
    profiled = {
        label: profile and has_timer(tree)
        for label, tree in zip(LABELS, (a, b), strict=True)
    }
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run = PROJECT / "native/reports" / f"bench-{stamp}"
    media = DIAGNOSTICS / f"bench-{stamp}"
    run.mkdir(parents=True)
    media.mkdir()
    result: dict = {
        "success": False,
        "trees": dict(zip(LABELS, (str(a), str(b)), strict=True)),
        "cases": cases,
        "repeats": repeats,
        "k_setting": setting,
        "confirms": confirms,
        "tiebreak": tiebreak,
        "profile": profile,
        "pythons": {label: str(python) for label, python in pythons.items()},
        "oracle": oracle,
        "png_compare": png_compare,
        "cross_stack": cross,
        "method": METHOD,
        "cpu_only": True,
        "ai_models": False,
        "environment": environment(),
        "media": str(media),
        "warnings": [],
        "trials": [],
    }
    busy = result["environment"]["busy_percent_before"]
    if busy > 10:
        warn(result, f"machine already {busy:.0f}% busy; timings will be noisier")
    backends: dict[str, Backend] = {}
    try:
        fixtures = prepare(media)
        result["fixtures"] = fixtures.record
        with ExitStack() as stack:
            for label, tree in zip(LABELS, (a, b), strict=True):
                backends[label] = stack.enter_context(
                    Backend(
                        label,
                        tree,
                        run,
                        bench_oracle.FFMPEG,
                        bench_oracle.FFPROBE,
                        sides[label],
                        python=pythons[label],
                    )
                )
            if (
                backends["A"].info["schema_sha256"]
                != backends["B"].info["schema_sha256"]
            ):
                raise RuntimeError("Node schemas differ: trees are not UI-compatible")
            for case in cases:
                # Each trial must equal its reference: A's warm-up for both sides,
                # or under --cross-stack the first trial of its own side.
                references: dict[str, dict] = {}
                for pair in range(repeats + 1):
                    for label in LABELS if pair % 2 == 0 else LABELS[::-1]:
                        record = run_trial(
                            backends[label],
                            case,
                            pair,
                            fixtures,
                            media,
                            profiled[label],
                            png_compare,
                        )
                        # Kept before the parity checks so a failing record reaches
                        # result.json for diagnosis.
                        result["trials"].append(record)
                        reference = references.setdefault(
                            label if cross_stack else LABELS[0], record
                        )
                        where = f"{case} {record['trial']} {label}"
                        if record["files"] != reference["files"]:
                            first = bench_baseline.changed_paths(
                                reference["files"], record["files"]
                            )[0]
                            raise AssertionError(
                                f"{where}: outputs differ, first at {first}"
                            )
                        if record["final_state"] != reference["final_state"]:
                            first = bench_oracle.contract_difference(
                                reference["final_state"], record["final_state"]
                            )
                            raise AssertionError(
                                f"{where}: final UI state differs at {first}"
                            )
                        write_json(run / "result.json", result)
                        print(
                            f"{where}: {record['seconds']:.3f}s "
                            f"{record['throughput']:.1f} {record['unit']}",
                            flush=True,
                        )
                if cross is not None:
                    # Outside every timed region; the media is still on disk.
                    side_a, side_b = references["A"], references["B"]
                    compared = cross_stack_record(side_a, side_b)
                    cross["cases"][case] = compared
                    if not compared["final_state_equal"]:
                        where = bench_oracle.contract_difference(
                            side_a["final_state"], side_b["final_state"]
                        )
                        warn(
                            result,
                            f"{case}: final UI state differs between A and B at "
                            f"{where}; recorded, not failed ({cross['label']})",
                        )
                    write_json(run / "result.json", result)
                    print(
                        f"{case} cross-stack: outputs "
                        f"{'equal' if compared['outputs_equal'] else 'differ'}, final "
                        "UI state "
                        f"{'equal' if compared['final_state_equal'] else 'differs'}",
                        flush=True,
                    )
        verify_unchanged(fixtures)
        for label, backend in backends.items():
            info = backend.info
            if not (
                info["owned_process_exited"]
                and info["owned_job_closed"]
                and info["backend_unchanged"]
                and info["interpreter_unchanged"]
            ):
                raise RuntimeError(
                    f"Backend {label} cleanup/identity check failed: {info}"
                )
        summary = bench_stats.summarize(result["trials"], LABELS)
        background = background_by_case(result["trials"])
        noise = recorded_noise()
        # The shared throughput floors are only read, never merged or written. A
        # profile run reads no floor at all: its timings carry the timer's overhead,
        # so no verdict, floor source or background warning may rest on one.
        entries = {} if profile else noise["floors"]
        floors = usable_floors(summary, entries)
        # A current floor only raises the thresholds of a case that a shared floor
        # can judge, so a profile run has none either. It is read before a
        # --record-noise launch merges into the launches it comes from.
        current = {
            case: floor
            for case, floor in current_floors(noise, setting).items()
            if case in floors
        }
        cpu_summary = {
            case: item["cpu"] for case, item in summary.items() if item["cpu"]
        }
        cpu_entries = {} if profile else noise.get("cpu_floors", {}).get(setting, {})
        # The CPU floors' loads are read before this launch is merged into them.
        for message in [
            *background_warnings(background, entries),
            *background_warnings(background, cpu_entries, cpu=True),
        ]:
            warn(result, message)
        if record_noise:
            launch_floors = bench_stats.noise_floors(summary)
            for case, floor in launch_floors.items():
                if case in floors and floor > floors[case]:
                    load = background[case]
                    warn(
                        result,
                        f"{case}: this launch's throughput floor "
                        f"+{100 * math.expm1(floor):.1f}% exceeds the shared floor "
                        f"+{100 * math.expm1(floors[case]):.1f}% (median background "
                        f"load {'n/a' if load is None else f'{load:.1f}%'}, machine "
                        f"busy before {busy:.1f}%); re-basing the shared floors is "
                        "the stand-in's decision",
                    )
            cpu_entries = merge_floors(
                cpu_entries,
                bench_stats.noise_floors(cpu_summary),
                run.name,
                repeats,
                background,
                launch_floors,
            )
            noise["cpu_floors"] = {**noise.get("cpu_floors", {}), setting: cpu_entries}
        cpu_floors = usable_floors(cpu_summary, cpu_entries)
        result["summary"] = summary
        result["background_busy_percent"] = background
        result["current_floors"] = current
        result["verdicts"] = {
            case: bench_stats.verdict(item, floors.get(case), current.get(case))
            for case, item in summary.items()
        }
        for case, verdict in result["verdicts"].items():
            if verdict == bench_stats.WITHIN_CURRENT:
                change = 100 * (summary[case]["median_ratio"] - 1)
                warn(
                    result,
                    f"{case}: regression {change:+.1f}% is beyond the shared floor "
                    f"{bench_stats.bounds(floors[case])} but within the current floor "
                    f"{bench_stats.bounds(current[case])} of the {setting} A/A "
                    "launches; it does not block and goes to the stand-in",
                )
        result["cpu_verdicts"] = {
            case: bench_stats.verdict(item["cpu"], cpu_floors.get(case))
            if item["cpu"]
            else "unknown"
            for case, item in summary.items()
        }
        loads = ", ".join(
            f"{case} {load:.1f}%" if load is not None else f"{case} n/a"
            for case, load in background.items()
        )
        faults = ", ".join(
            f"{case} {item['A']:.1f} / {item['B']:.1f}" if item else f"{case} n/a"
            for case, item in page_faults_by_case(result["trials"]).items()
        )
        role = "Tie-break" if tiebreak else "Confirmation"
        if record_noise:
            note = IN_SAMPLE_NOTE
        elif profile:
            note = PROFILE_NOTE
        else:
            note = PROVISIONAL_NOTE
        header = [
            f"# Benchmark {run.name}",
            "",
            f"A = `{a}`",
            f"B = `{b}`",
            "",
            f"- Repeats: {repeats} AB/BA pairs per case, after one warm-up",
            f"- Machine busy before the run: {busy:.1f}%",
            f"- Median background load (CPU outside the owned backends): {loads}",
            f"- Page faults per item (median of measured trials), A / B: {faults}",
            f"- Noise floors from: {floor_sources(floors, entries)}",
            (
                f"- K setting: {setting}; CPU floors from: "
                f"{floor_sources(cpu_floors, cpu_entries)}"
            ),
            f"- Interpreters: A `{pythons['A']}`, B `{pythons['B']}`",
            f"- PNG outputs compared by {bench_oracle.PNG_MODES[png_compare]}",
            *([] if cross is None else [f"- Label: {cross['label']}"]),
            *(
                f"- {label} environment: "
                + ", ".join(f"{name}={value}" for name, value in env.items())
                for label, env in sides.items()
                if env
            ),
            note,
            *([] if confirms is None else [f"- {role} run for {confirms}."]),
            *(f"- WARNING {message}" for message in result["warnings"]),
            "",
            LEGEND,
            CPU_LEGEND,
            "",
        ]
        cross_section = (
            ""
            if cross is None
            else f"\n## Cross-stack\n\n{cross_stack_table(cross['cases'])}\n"
        )
        (run / "summary.md").write_text(
            "\n".join(header)
            + "\n"
            + bench_stats.table(summary, LABELS, floors, current, cpu_floors)
            + "\n"
            + cross_section,
            encoding="utf-8",
        )
        if record_noise:
            # Written last, so a run that fails anywhere leaves the floors untouched.
            # Only this K setting's CPU section changes; "floors" is written back
            # exactly as it was read.
            write_json(NOISE, noise)
        result["success"] = True
    except BaseException:
        result["error"] = traceback.format_exc()
        raise
    finally:
        result["backends"] = {
            label: backend.info for label, backend in backends.items()
        }
        if result["success"]:
            # A successful run's outputs and fixture copies (about 2 GB for 13 cases)
            # were compared during the run; each trial's events.json and request.json
            # are evidence (sp4b-verdict.py reads them) and stay. A failed run keeps
            # everything for diagnosis.
            removed = 0
            try:
                # Deepest paths first, so a directory is emptied before it is checked.
                for path in sorted(media.rglob("*"), reverse=True):
                    if path.is_file() and path.suffix != ".json":
                        path.unlink()
                        removed += 1
                    elif path.is_dir() and not any(path.iterdir()):
                        path.rmdir()
            except OSError as error:
                result["media_cleanup_error"] = repr(error)
                print(f"WARNING media not fully removed: {error!r}", flush=True)
            result["media_files_removed"] = removed
        write_json(run / "result.json", result)
        print(f"RESULT {run} success={result['success']}", flush=True)
    return run


def final_verdicts(
    first: dict[str, str],
    second: dict[str, str],
    third: dict[str, str] | None = None,
) -> dict[str, str]:
    """Each case's verdict once its win or regression has been re-checked.

    second is the confirmation run and third the optional tie-break run. A
    regression is final when 2 of the 3 runs flagged one, of either label, and a
    win stands only when second is also a win. So a win that second flipped to
    regression becomes a regression when third also flags one. A final regression
    is "regression", which blocks, when 2 of the 3 runs put it beyond the current
    floor, and WITHIN_CURRENT otherwise. Any other flagged case, also when a run
    lacks it, becomes "unconfirmed". Every other verdict passes through unchanged.
    """
    third = third or {}

    def settle(case: str, verdict: str) -> str:
        if verdict == "win" and second.get(case) == "win":
            return "win"
        given = [run.get(case) for run in (first, second, third)]
        if given.count("regression") >= 2:
            return "regression"
        if sum(label in REGRESSIONS for label in given) >= 2:
            return bench_stats.WITHIN_CURRENT
        return "unconfirmed"

    return {
        case: settle(case, verdict) if verdict in FLAGGED else verdict
        for case, verdict in first.items()
    }


def tiebreak_cases(first: dict[str, str], second: dict[str, str]) -> list[str]:
    """The cases whose regression signal needs a third run, in the first run's order.

    These are the regressions the confirmation run did not repeat with the same
    label, and the wins it flipped to a regression of either label. A missed
    regression is the costlier error, so a flip is followed up, and so is a
    regression whose two runs disagree on whether it blocks. Every other win is
    decided by the confirmation run alone: wins need it to repeat them.
    """
    return [
        case
        for case, verdict in first.items()
        if (verdict in REGRESSIONS and second.get(case) != verdict)
        or (verdict == "win" and second.get(case) in REGRESSIONS)
    ]


def run_outcome(run: Path) -> tuple[dict[str, str], dict[str, float]]:
    """A finished run's verdicts and paired changes in % (median ratio minus 1)."""
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    changes = {
        case: 100 * (item["median_ratio"] - 1)
        for case, item in result["summary"].items()
    }
    return result["verdicts"], changes


def verdict_cell(
    verdicts: dict[str, str], changes: dict[str, float], case: str, missing: str
) -> str:
    """A table cell holding the case's verdict and paired change, or missing."""
    if case not in verdicts:
        return missing
    return f"{verdicts[case]} ({changes[case]:+.1f}%)"


def confirmation_block(
    flagged: list[str], outcomes: list[tuple[dict[str, str], dict[str, float]]]
) -> dict:
    """The flagged cases' record in confirmation.json, settled by final_verdicts.

    outcomes are the run_outcome() of the first, confirmation and tie-break runs,
    the last ({}, {}) when no tie-break run was made.
    """
    (first, first_changes), (second, second_changes), (third, third_changes) = outcomes
    return {
        "flagged": flagged,
        "first": first,
        "confirmation": second,
        "tiebreak": third,
        "final": final_verdicts(first, second, third),
        "paired_change_percent": {
            "first": first_changes,
            "confirmation": second_changes,
            "tiebreak": third_changes,
        },
    }


def confirmation_table(block: dict, tied: list[str]) -> str:
    """Markdown: each flagged case's verdict and change per run, and its final.

    A case the tie-break run did not cover shows "-" in that column.
    """
    changes = block["paired_change_percent"]
    rows = [
        "| Case | First | Confirmation | Tie-break | Final |",
        "| --- | --- | --- | --- | --- |",
    ]
    for case in block["flagged"]:
        cells = (
            verdict_cell(block["first"], changes["first"], case, "n/a"),
            verdict_cell(block["confirmation"], changes["confirmation"], case, "n/a"),
            verdict_cell(
                block["tiebreak"],
                changes["tiebreak"],
                case,
                "n/a" if case in tied else "-",
            ),
            block["final"][case],
        )
        rows.append(f"| {case} | {' | '.join(cells)} |")
    return "\n".join(rows)


def confirm(
    first_run: Path,
    a: Path,
    b: Path,
    repeats: int,
    env_a: dict[str, str] | None = None,
    env_b: dict[str, str] | None = None,
    python_a: Path | None = None,
    python_b: Path | None = None,
    png_compare: str = "sha256",
    cross_stack: bool = False,
) -> dict[str, str]:
    """Re-check the first run's win and regression cases on freshly started backends.

    A case is flagged when its throughput verdict is a win or a regression of
    either label; CPU verdicts are information only and flag nothing. compare()
    launches its own backends, so the confirmation run of every flagged case, and
    one tie-break run for the disputed regressions and the wins flipped to
    regression (see tiebreak_cases and final_verdicts), are fresh launches whose
    speed bias is independent of the first run's. Both keep the first run's
    per-side environments (env_a, env_b) and interpreters (python_a, python_b),
    its PNG mode (png_compare) and its cross-stack mode (cross_stack). The
    combined records go into first_run: confirmation.json, confirmation.md, and
    the same table appended to summary.md. Each flagged case's final verdict is
    printed, and the final verdicts of every case are returned. With no flagged
    case nothing runs and nothing is written. A failing run propagates before anything is written; the
    first run's own records already exist.
    """
    first_outcome = run_outcome(first_run)
    first = first_outcome[0]
    flagged = [case for case, verdict in first.items() if verdict in FLAGGED]
    if not flagged:
        return first
    confirmation_run = compare(
        a,
        b,
        flagged,
        repeats,
        False,
        confirms=first_run.name,
        env_a=env_a,
        env_b=env_b,
        python_a=python_a,
        python_b=python_b,
        png_compare=png_compare,
        cross_stack=cross_stack,
    )
    second = run_outcome(confirmation_run)
    tied = tiebreak_cases(first, second[0])
    tiebreak_run = None
    third: tuple[dict[str, str], dict[str, float]] = ({}, {})
    if tied:
        tiebreak_run = compare(
            a,
            b,
            tied,
            repeats,
            False,
            confirms=first_run.name,
            tiebreak=True,
            env_a=env_a,
            env_b=env_b,
            python_a=python_a,
            python_b=python_b,
            png_compare=png_compare,
            cross_stack=cross_stack,
        )
        third = run_outcome(tiebreak_run)
    block = confirmation_block(flagged, [first_outcome, second, third])
    write_json(
        first_run / "confirmation.json",
        {
            "first_run": first_run.name,
            "confirmation_run": confirmation_run.name,
            "tiebreak_run": None if tiebreak_run is None else tiebreak_run.name,
            **block,
        },
    )
    table = confirmation_table(block, tied)
    tiebreak_name = "none" if tiebreak_run is None else f"`{tiebreak_run.name}`"
    runs = (
        f"Runs: first `{first_run.name}`, confirmation `{confirmation_run.name}`, "
        f"tie-break {tiebreak_name}."
    )
    (first_run / "confirmation.md").write_text(
        "\n\n".join(
            [f"# Confirmation of {first_run.name}", table, runs, *CONFIRMATION_NOTES]
        )
        + "\n",
        encoding="utf-8",
    )
    with (first_run / "summary.md").open("a", encoding="utf-8") as summary:
        summary.write(
            f"\n## Confirmed verdicts\n\n{table}\n\n"
            "See confirmation.md for the runs, the tie-break rule and the caveats.\n"
        )
    for case in flagged:
        print(f"FINAL {case}: {block['final'][case]}", flush=True)
    return block["final"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--a",
        type=Path,
        default=DEFAULT_A,
        help="tree A, the reference backend src directory (default: frozen B0)",
    )
    parser.add_argument(
        "--b",
        type=Path,
        default=DEFAULT_B,
        help="tree B, the candidate backend src directory (default: packaged backend)",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=bench_cases.CASES,
        default=list(bench_cases.CASES),
        metavar="CASE",
        help="cases to run, in order (default: all): " + ", ".join(bench_cases.CASES),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=6,
        help="measured AB/BA pairs per case after one warm-up; must be even, and "
        f"at least {MIN_PAIRS} for verdicts and noise floors (default: 6)",
    )
    parser.add_argument(
        "--record-noise",
        action="store_true",
        help="add this launch's A/A CPU noise floor for each measured case to bench-"
        f"data/noise-floor.json under the K setting of {ITEM_WINDOW_VARIABLE} "
        f"(usable after {MIN_LAUNCHES} launches); each launch's throughput floor is "
        "kept beside it and sets the current floor; never writes the shared "
        f"throughput floors; needs identical trees and at least {MIN_PAIRS} pairs",
    )
    for label in LABELS:
        parser.add_argument(
            f"--env-{label.lower()}",
            action="append",
            metavar="NAME=VALUE",
            help=f"set NAME=VALUE in backend {label}'s environment only, applied last "
            "and recorded; repeatable; kept by the confirmation runs; refused with "
            "--record-noise and for the harness's and the isolation's own variables",
        )
    parser.add_argument(
        "--profile",
        action="store_true",
        help=f"profile run: {PROFILE_VARIABLE}=1 on both sides, and each trial of a "
        "tree with the timer records its native_profile table; needs such a tree; "
        "applies no floors and is never confirmed",
    )
    for label in LABELS:
        parser.add_argument(
            f"--python-{label.lower()}",
            type=Path,
            metavar="PATH",
            help=f"the python.exe that runs backend {label} (default: the packaged "
            f"port's, {PYTHON}"
            + (
                ", or the provisioned runtime when A is the oracle"
                if label == "A"
                else ""
            )
            + "); refused unless an existing python.exe; recorded; kept by the "
            "confirmation runs; refused with --record-noise",
        )
    parser.add_argument(
        "--png-compare",
        choices=list(bench_oracle.PNG_MODES),
        default="sha256",
        help="what output PNGs must match on: sha256, the file bytes (default), or "
        "decoded, the pixels cv2.imread(IMREAD_UNCHANGED) returns, by shape, dtype "
        "and every byte, plus colour/orientation chunks; videos keep their "
        "decoded-frame oracle; recorded; kept by the confirmation runs",
    )
    parser.add_argument(
        "--cross-stack",
        action="store_true",
        help="the sides run different Python stacks: each trial must equal its own "
        "side's first trial, and A against B is recorded per case, compared "
        "decoded, never failed; refused unless the interpreters report different "
        "sys.version_info[:2], and with --record-noise; kept by the confirmation runs",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    a, b = args.a.resolve(), args.b.resolve()
    env_a, env_b = parse_env(args.env_a), parse_env(args.env_b)
    python_a = None if args.python_a is None else args.python_a.resolve()
    python_b = None if args.python_b is None else args.python_b.resolve()
    first = compare(
        a,
        b,
        args.cases,
        args.repeats,
        args.record_noise,
        env_a=env_a,
        env_b=env_b,
        profile=args.profile,
        python_a=python_a,
        python_b=python_b,
        png_compare=args.png_compare,
        cross_stack=args.cross_stack,
    )
    if not (args.record_noise or args.profile):
        confirm(
            first,
            a,
            b,
            args.repeats,
            env_a,
            env_b,
            python_a,
            python_b,
            args.png_compare,
            args.cross_stack,
        )


if __name__ == "__main__":
    main()
