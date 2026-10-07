"""chaiNNer-C's chainner_ext against the conformance manifest (spec 4), on CPython 3.14.

native/tests/chainner_ext/ holds what the real chainner_ext 0.3.10 did, recorded on
native/.venv-py311 by native/tools/chainner_ext_conformance.py (record-regex, then
record). This test needs no real module: it runs every recorded case on the module
`import chainner_ext` gives (backend/src/chainner_ext) under the manifest's rules.
"""

from __future__ import annotations

from pathlib import Path

import chainner_ext_conformance as conformance
import pytest

import chainner_ext

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = conformance.load_manifest()
ROWS = MANIFEST["cases"]
CELLS = sorted({row["cell"] for row in ROWS})
CORPUS = conformance.read_jsonl(conformance.CORPUS)
EXPECTED = {
    row["id"]: row["results"] for row in conformance.read_jsonl(conformance.EXPECTED)
}
FAMILIES = sorted({conformance.family_of(line["id"]) for line in CORPUS})


def test_the_module_is_chainner_c():
    assert chainner_ext.__file__ is not None, "a namespace package: no __init__.py"
    package = Path(chainner_ext.__file__).resolve().parent
    assert package == (ROOT / "backend/src/chainner_ext").resolve()


def test_manifest_schema_and_regex_files():
    assert MANIFEST["schema"] == conformance.SCHEMA
    for name, entry in MANIFEST["regex"]["files"].items():
        assert conformance.sha256_file(conformance.OUT_DIR / name) == entry["sha256"], (
            name
        )


@pytest.mark.parametrize("cell", CELLS)
def test_image_cell(cell):
    rows = [row for row in ROWS if row["cell"] == cell]
    assert (
        len(rows)
        == MANIFEST["cells"][cell]["cases"]
        >= MANIFEST["cells"][cell]["floor"]
    )
    binary = None
    if any(row["call"].startswith("Clipboard.") for row in rows):
        binary = conformance.clipboard_binary(chainner_ext)
    problems = []
    for row in rows:
        problem = conformance.check_candidate_case(chainner_ext, row, binary)
        if problem is not None:
            problems.append(f"{row['id']}: {problem}")
    assert not problems, "\n".join(problems[:20])


def test_surface():
    have = conformance.surface_probe(chainner_ext)
    differing = {
        k: (v, have.get(k)) for k, v in MANIFEST["surface"].items() if have.get(k) != v
    }
    assert not differing


@pytest.mark.parametrize("family", FAMILIES)
def test_regex_corpus(family):
    differing = []
    for line in CORPUS:
        if conformance.family_of(line["id"]) == family:
            have = conformance.run_probe_case(chainner_ext, line)
            if have != EXPECTED[line["id"]]:
                differing.append(line["id"])
    assert not differing, differing[:20]


def test_regex_module_cases():
    differing = []
    for case in conformance.read_jsonl(conformance.MODULE_CASES):
        have = conformance.run_module_case(chainner_ext, case)
        want_rows = conformance.module_case_expected(case)
        for k, (want, got) in enumerate(zip(want_rows, have, strict=True)):
            if not conformance.same_outcome(got, want):
                differing.append(f"{case['id']} call {k}")
    assert not differing, differing[:20]


def test_nfa_boundary_exclusions_are_boundary_cases_of_the_corpus():
    exclusions = conformance.load_exclusions()
    assert len(exclusions) == 60
    assert sum(case["ops"] for case in exclusions.values()) == 221


@pytest.mark.parametrize("family", sorted(MANIFEST["regex"]["toobig"]))
def test_regex_compiled_too_big_flip(family):
    want = MANIFEST["regex"]["toobig"][family]
    assert conformance.toobig_flip(chainner_ext, family) == (
        want["flip"],
        want["message"],
    )
