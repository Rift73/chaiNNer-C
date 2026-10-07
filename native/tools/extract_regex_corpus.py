"""Candidate regex cases for the chainner_ext remake's conformance (spec section 4).

Usage:
  python native/tools/extract_regex_corpus.py [--src DIR] --out FILE

Writes the candidate probe cases as JSON lines, for inspection. The tracked corpus
(native/tests/chainner_ext/regex_corpus.jsonl), its expected results and the
module-level cases are written by `chainner_ext_conformance.py record-regex`, which
calls build_cases() and records every op on the real chainner_ext 0.3.10.

Sources, in this order:
- regex 1.8.4's tests/ (SOURCE_FILES): every mat!, matiter!, ismatch!, shortmat!,
  split!, splitn!, expand! and replace! with a literal pattern and text; every
  noparse!, consistent!, regex! and regex_new! with a literal pattern, paired with
  generated texts. Arguments that are not one plain string literal are skipped;
  byte-string literals are skipped.
- seeded generated cases (SEED): random patterns from a small grammar, mutated
  patterns (most fail to parse, for the error texts), hand-written error and flag
  patterns, the pos family (every position 0..bytes+2 over multibyte texts), the
  names family (named and unmatched optional groups) and the nest-limit family.

A case is {"id", "pattern", "ops"}; ops are compile, then per text: search at the
edge positions, findall, split and split_without_captures. A search position is
classed by the byte offset regex-py's to_byte_pos gives it (pos_class()), and each
class goes to its own case, so the harness counts them apart:
- "boundary": a char boundary at or before the end; in the case itself;
- "midchar": inside a code point (to_byte_pos maps pos in [chars, bytes] to
  pos + chars - bytes); in the sibling case "<id>#midchar", since which engine runs
  decides the result there (the DFA's UTF-8 prefix cannot leave a continuation
  byte, a PikeVM can);
- "past": past the end, through to_byte_pos's out-of-bounds branch (pos > bytes),
  or the usize wrap of pos + chars - bytes; in the sibling case "<id>#past". The
  real module panics there, and the expected result is {"panic": message}.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SRC = REPO / "native/runtime/download/src/regex-1.8.4/tests"
SEED = 20261006
USIZE = 1 << 64

SOURCE_FILES = (
    "fowler.rs",
    "noparse.rs",
    "unicode.rs",
    "word_boundary.rs",
    "word_boundary_unicode.rs",
    "word_boundary_ascii.rs",
    "crazy.rs",
    "regression.rs",
    "regression_fuzz.rs",
    "multiline.rs",
    "flags.rs",
    "api.rs",
    "api_str.rs",
    "misc.rs",
    "shortest_match.rs",
    "suffix_reverse.rs",
    "replace.rs",
    "crates_regex.rs",
)

# macro -> index of the pattern argument, index of the text argument (or None)
MACROS: dict[str, tuple[int, int | None]] = {
    "mat": (1, 2),
    "matiter": (1, 2),
    "ismatch": (1, 2),
    "shortmat": (1, 2),
    "split": (1, 2),
    "splitn": (1, 2),
    "expand": (1, 2),
    "replace": (2, 3),
    "noparse": (1, None),
    "consistent": (1, None),
    "regex": (0, None),
    "regex_new": (0, None),
}

PosClass = Literal["boundary", "midchar", "past"]
Op = dict[str, Any]


@dataclass
class Case:
    id: str
    pattern: str
    texts: list[str]
    # explicit positions per text (None: the edge set of edge_positions())
    positions: list[list[int] | None] = field(default_factory=list)
    # False: no split ops (findall only), to keep huge texts out of the files
    splits: bool = True


# ---------------------------------------------------------------------------
# Rust source scanning


def _skip_comment(src: str, i: int) -> int:
    if src.startswith("//", i):
        j = src.find("\n", i)
        return len(src) if j < 0 else j + 1
    depth = 0
    while i < len(src):
        if src.startswith("/*", i):
            depth += 1
            i += 2
        elif src.startswith("*/", i):
            depth -= 1
            i += 2
            if depth == 0:
                return i
        else:
            i += 1
    raise ValueError("unterminated block comment")


def _ident_before(src: str, i: int) -> bool:
    return i > 0 and (src[i - 1].isalnum() or src[i - 1] == "_")


def _string_start(src: str, i: int) -> bool:
    """True when a (byte or raw) string literal starts at i."""
    j = i
    if src[j] == "b":
        j += 1
    if j < len(src) and src[j] == "r":
        k = j + 1
        while k < len(src) and src[k] == "#":
            k += 1
        return k < len(src) and src[k] == '"' and not _ident_before(src, i)
    if j < len(src) and src[j] == '"':
        return j == i or not _ident_before(src, i)
    return False


def _decode_escape(src: str, i: int, byte_string: bool) -> tuple[str, int]:
    """Decodes the escape at src[i] == '\\'; returns (text, next index)."""
    c = src[i + 1]
    simple = {
        "n": "\n",
        "r": "\r",
        "t": "\t",
        "\\": "\\",
        "0": "\0",
        "'": "'",
        '"': '"',
    }
    if c in simple:
        return simple[c], i + 2
    if c == "x":
        value = int(src[i + 2 : i + 4], 16)
        if value > 0x7F and not byte_string:
            raise ValueError("\\x above 0x7f in a str literal")
        return chr(value), i + 4
    if c == "u":
        end = src.index("}", i)
        return chr(int(src[i + 3 : end].replace("_", ""), 16)), end + 1
    if c == "\n":
        j = i + 2
        while j < len(src) and src[j] in " \t\n\r":
            j += 1
        return "", j
    raise ValueError(f"unknown escape \\{c}")


def lex_string(src: str, i: int) -> tuple[str | None, int]:
    """Parses the string literal at i; returns (value, end). Byte strings give None."""
    byte_string = src[i] == "b"
    j = i + 1 if byte_string else i
    if src[j] == "r":
        k = j + 1
        hashes = 0
        while src[k] == "#":
            hashes += 1
            k += 1
        close = '"' + "#" * hashes
        end = src.index(close, k + 1)
        value = src[k + 1 : end]
        return (None if byte_string else value), end + len(close)
    out: list[str] = []
    k = j + 1
    while src[k] != '"':
        if src[k] == "\\":
            text, k = _decode_escape(src, k, byte_string)
            out.append(text)
        else:
            out.append(src[k])
            k += 1
    return (None if byte_string else "".join(out)), k + 1


def _skip_char_or_lifetime(src: str, i: int) -> int:
    if src[i + 1] == "\\":
        return src.index("'", i + 2) + 1
    if i + 2 < len(src) and src[i + 2] == "'":
        return i + 3
    return i + 1


@dataclass
class Invocation:
    name: str
    args: list[str]
    line: int


def _split_args(src: str, i: int) -> tuple[list[str], int]:
    """Splits the macro arguments starting after '(' at i; returns (args, end)."""
    args: list[str] = []
    depth = 0
    start = i
    while True:
        c = src[i]
        if src.startswith("//", i) or src.startswith("/*", i):
            i = _skip_comment(src, i)
        elif _string_start(src, i):
            _, i = lex_string(src, i)
        elif c == "'":
            i = _skip_char_or_lifetime(src, i)
        elif c in "([{":
            depth += 1
            i += 1
        elif c in ")]}":
            if depth == 0:
                tail = src[start:i].strip()
                if tail:
                    args.append(tail)
                return args, i + 1
            depth -= 1
            i += 1
        elif c == "," and depth == 0:
            args.append(src[start:i].strip())
            i += 1
            start = i
        else:
            i += 1


_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def invocations(src: str) -> Iterator[Invocation]:
    """Every name!( ... ) in src, nested ones included, outside comments and literals."""
    i = 0
    while i < len(src):
        c = src[i]
        if src.startswith("//", i) or src.startswith("/*", i):
            i = _skip_comment(src, i)
            continue
        if _string_start(src, i):
            _, i = lex_string(src, i)
            continue
        if c == "'":
            i = _skip_char_or_lifetime(src, i)
            continue
        m = _IDENT.match(src, i)
        if m is None or (i > 0 and (src[i - 1].isalnum() or src[i - 1] == "_")):
            i += 1
            continue
        j = m.end()
        if src.startswith("!", j) and j + 1 < len(src):
            k = j + 1
            while src[k] in " \t\n":
                k += 1
            if src[k] == "(" and m.group() in MACROS:
                args, _ = _split_args(src, k + 1)
                yield Invocation(m.group(), args, src.count("\n", 0, i) + 1)
                i = k + 1
                continue
        i = j


def literal(arg: str) -> str | None:
    """The value of arg when it is exactly one str literal, else None."""
    if not arg or not _string_start(arg, 0):
        return None
    try:
        value, end = lex_string(arg, 0)
    except ValueError:  # an escape regex's tests never use in a str
        return None
    except IndexError:  # unterminated
        return None
    return value if end == len(arg) else None


# ---------------------------------------------------------------------------
# Positions


def to_byte_pos(text: str, char_pos: int) -> int:
    """regex-py's to_byte_pos, with usize wrapping."""
    encoded = text.encode("utf-8")
    if char_pos == 0:
        return 0
    if char_pos > len(encoded):
        return char_pos
    if char_pos < len(text):
        return len(text[:char_pos].encode("utf-8"))
    return (char_pos + len(text) - len(encoded)) % USIZE


def pos_class(text: str, pos: int) -> PosClass:
    byte = to_byte_pos(text, pos)
    encoded = text.encode("utf-8")
    if byte > len(encoded):
        return "past"
    if byte == len(encoded) or (encoded[byte] & 0xC0) != 0x80:
        return "boundary"
    return "midchar"


def edge_positions(text: str) -> list[int]:
    chars = len(text)
    nbytes = len(text.encode("utf-8"))
    raw = {0, 1, chars // 2, chars - 1, chars, chars + 1, chars + 3}
    raw |= {nbytes - 1, nbytes, nbytes + 1}
    return sorted(p for p in raw if p >= 0)


def text_ops(
    text: str, positions: list[int], splits: bool = True
) -> dict[PosClass, list[Op]]:
    ops: dict[PosClass, list[Op]] = {"boundary": [], "midchar": [], "past": []}
    for pos in positions:
        ops[pos_class(text, pos)].append({"op": "search", "text": text, "pos": pos})
    ops["boundary"].append({"op": "findall", "text": text})
    if splits:
        ops["boundary"] += [
            {"op": "split", "text": text},
            {"op": "split_without_captures", "text": text},
        ]
    return ops


# ---------------------------------------------------------------------------
# Texts for pattern-only sources

ALPHABET = "abcxyzABC019 _-.,:;/\n\täÄöüßéΩж中😀"
_SKETCH = {
    r"\d": "7",
    r"\D": "x",
    r"\w": "w",
    r"\W": "!",
    r"\s": " ",
    r"\S": "s",
    r"\n": "\n",
    r"\t": "\t",
    r"\.": ".",
    r"\/": "/",
    r"\-": "-",
    r"\:": ":",
}


def sketch_text(pattern: str) -> str:
    """A rough text that pattern may match: escapes replaced, syntax dropped."""
    text = re.sub(r"\(\?P?<[^>]*>", "", pattern)
    text = re.sub(r"\(\?[a-zA-Z-]*:?", "", text)
    text = re.sub(r"\[\^?([^\]\\]?)[^\]]*\]", lambda m: m.group(1) or "q", text)
    text = re.sub(r"\{\d+(,\d*)?\}", "", text)
    for escape, replacement in _SKETCH.items():
        text = text.replace(escape, replacement)
    text = re.sub(r"\\p\{?\w+\}?", "a", text)
    text = re.sub(r"\\.", "", text)
    return "".join(c for c in text if c not in "^$()|*+?{}[]")


def random_text(rng: random.Random, max_len: int, alphabet: str = ALPHABET) -> str:
    return "".join(rng.choice(alphabet) for _ in range(rng.randint(0, max_len)))


def pattern_texts(rng: random.Random, pattern: str) -> list[str]:
    sketch = sketch_text(pattern)
    local = "".join(sorted(set(sketch))) or "a"
    return [
        sketch,
        random_text(rng, 2) + sketch + random_text(rng, 4),
        random_text(rng, 20, local + "a1 äZ😀\n"),
    ]


# ---------------------------------------------------------------------------
# Upstream tests


def upstream_cases(src_dir: Path) -> Iterator[Case]:
    rng = random.Random(f"{SEED}:upstream")
    for filename in SOURCE_FILES:
        stem = filename.removesuffix(".rs")
        src = (src_dir / filename).read_text(encoding="utf-8").replace("\r\n", "\n")
        seen: dict[str, int] = {}
        for inv in invocations(src):
            pattern_index, text_index = MACROS[inv.name]
            if len(inv.args) <= pattern_index:
                continue
            pattern = literal(inv.args[pattern_index])
            if pattern is None:
                continue
            if inv.name in ("regex", "regex_new"):
                name = f"L{inv.line}"
            else:
                name = (
                    inv.args[0]
                    if inv.args and _IDENT.fullmatch(inv.args[0])
                    else f"L{inv.line}"
                )
            key = f"{stem}/{name}"
            seen[key] = seen.get(key, 0) + 1
            if seen[key] > 1:
                key = f"{key}~{seen[key]}"
            text = None
            if text_index is not None and len(inv.args) > text_index:
                text = literal(inv.args[text_index])
            texts = [text] if text is not None else pattern_texts(rng, pattern)
            yield Case(key, pattern, texts)


# ---------------------------------------------------------------------------
# Generated patterns


_ATOMS = (
    "a", "b", "c", "ab", "x", "ä", "Ä", "ß", "😀", "中", " ", r"\.", ".", r"\d",
    r"\w", r"\s", r"\W", r"\D", r"\pL", r"\p{Greek}", r"\PL", r"\p{Lu}", "[a-c]",
    "[^a]", "[äöü]", r"[\d\s]", "[[:alpha:]]", "[[:^digit:]]", r"(?-u:\w)",
    r"(?-u:\d)", r"\x41", r"\u{e4}", r"\x{1F600}", "1", "_", r"\n", r"[a-z&&[^c]]",
    r"[\w--\d]", r"[a-c~~b-d]", "[-a]", r"[\pL\d]", "(?i:k)", "(?i:ß)", "(?i:\u03c3)",
)  # fmt: skip
_ASSERTIONS = (
    "^", "$", r"\b", r"\B", r"\A", r"\z", "(?m:^)", "(?m:$)", r"(?-u:\b)",
    r"(?-u:\B)",
)  # fmt: skip
_REPEATS = ("*", "+", "?", "*?", "+?", "??", "{2}", "{1,3}", "{0,2}?", "{2,}", "{0}")
_FLAGS = ("(?i)", "(?m)", "(?s)", "(?U)", "(?is)", "(?i-u)", "(?m-s)", "(?x)")
GEN_ALPHABET = "abcxABCX19 _\n.äÄöß😀中\u03a3\u03c3\u03c2ΩK\u212a"


class _PatternGen:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.names = 0

    def node(self, depth: int) -> str:
        rng = self.rng
        roll = rng.random()
        if depth <= 0 or roll < 0.35:
            return rng.choice(_ATOMS)
        if roll < 0.45:
            return rng.choice(_ASSERTIONS)
        if roll < 0.62:
            return "".join(self.node(depth - 1) for _ in range(rng.randint(2, 3)))
        if roll < 0.72:
            return "|".join(self.node(depth - 1) for _ in range(rng.randint(2, 3)))
        if roll < 0.88:
            inner = self.node(depth - 1)
            kind = rng.random()
            if kind < 0.45:
                group = f"({inner})"
            elif kind < 0.7:
                group = f"(?:{inner})"
            else:
                self.names += 1
                opener = "(?P<" if rng.random() < 0.6 else "(?<"
                group = f"{opener}n{self.names}>{inner})"
            if rng.random() < 0.4:
                group += rng.choice(_REPEATS)
            return group
        atom = rng.choice(_ATOMS)
        if len(atom) > 1 and not atom.startswith(("\\", "[", "(")):
            atom = f"(?:{atom})"
        return atom + rng.choice(_REPEATS)

    def pattern(self, max_depth: int = 4) -> str:
        self.names = 0
        body = self.node(self.rng.randint(1, max_depth))
        if self.rng.random() < 0.2:
            body = self.rng.choice(_FLAGS) + body
        return body


def generated_cases(
    count: int,
    mutated: int,
    tags: tuple[str, str] = ("gen", "mut"),
    seeds: tuple[str, str] = ("gen", "mutate"),
    max_depth: int = 4,
    text_len: int = 16,
) -> Iterator[Case]:
    """count random patterns (ids <tags[0]>/k) and mutated ones (<tags[1]>/k)."""
    rng = random.Random(f"{SEED}:{seeds[0]}")
    gen = _PatternGen(rng)
    patterns = [gen.pattern(max_depth) for _ in range(count)]
    for k, pattern in enumerate(patterns):
        texts = [random_text(rng, text_len, GEN_ALPHABET) for _ in range(2)]
        texts.append(sketch_text(pattern))
        yield Case(f"{tags[0]}/{k:04d}", pattern, texts)
    mut = random.Random(f"{SEED}:{seeds[1]}")
    for k in range(mutated):
        pattern = mut.choice(patterns)
        chars = list(pattern)
        action = mut.random()
        at = mut.randint(0, len(chars))
        if action < 0.5 or not chars:
            chars.insert(at, mut.choice("()[]{}*+?\\|^$-:<>P,0"))
        elif action < 0.8:
            del chars[min(at, len(chars) - 1)]
        else:
            chars[min(at, len(chars) - 1)] = mut.choice("({[\\?*")
        mutated_pattern = "".join(chars)
        yield Case(
            f"{tags[1]}/{k:04d}", mutated_pattern, [random_text(mut, 10, GEN_ALPHABET)]
        )


ERROR_PATTERNS = (
    "(", ")", "[", "[]", "[^]", "[a-", "a**", "a{2,1}", r"\p{Foo}", r"\p{Greek",
    "(?P<>a)", "(?P<a>x)(?P<a>y)", "(?<1a>x)", "(?P<a b>x)", "(?P<a", r"\8", r"\xZZ",
    r"\x{110000}", r"\x{D800}", "(?z)", "(?i-i)", "(?-)", "(?)", "a{4294967296}",
    "a{1,4294967296}", "*", "+a", "x{,3}", "[z-a]", "\\", "(?-u)\\xFF", r"(?-u:\W)",
    "(?-u).", "(?-u)[^a]", r"\141", "[[:foo:]]", r"[\d-z]", "a{", "a{1", "a{1,", "{",
    "a{}", "(?:", "(?P=a)", r"\k<a>", "(?=a)", "(?!a)", "(?<=a)", "(?<!a)", r"\1",
    r"\Z", r"\G", "a++", "a?+", r"\b{start}", r"\Qa\E", r"\cA", r"\e", r"\N{DIGIT}",
    "[a&&]", "[&&a]", "[a--]", r"\p{Script=Foo}", r"\p{gc=Lu}", r"\p{scx=Greek}",
    r"\p{Greek", r"\pZ", r"\pX", "(?x) a b # comment", "(?x)[a b]", "(?x)a\\ b",
    "(?i)ǅ", "(?s-s:.)", "[\\]]", "[]a]", "[^]a]", "[a-\\d]", "[\\d-a]", "\\u{}",
    "\\u{zz}", "\\U0001F600", "\\u00e4", "(?U)a+?", "a|*", "|", "||", "()", "(|)",
    "(?:)", "(?P<n>)", "(?P<n>)?", "[[:alpha:]-z]", "[a-[:digit:]]", "(?m)^$",
    "\\pN{2}", "x{2}{3}", "x**", "x{0,0}", "x{0}?", "(?i)[k]", "(?i)\\p{Lu}",
)  # fmt: skip

NEST_DEPTHS = (248, 249, 250, 251, 252)


def special_cases() -> Iterator[Case]:
    rng = random.Random(f"{SEED}:special")
    for k, pattern in enumerate(ERROR_PATTERNS):
        texts = [random_text(rng, 8, GEN_ALPHABET), "a b\nab]x{2}"]
        yield Case(f"special/{k:03d}", pattern, texts)
    for depth in NEST_DEPTHS:
        yield Case(f"nest/paren/{depth}", "(" * depth + "a" + ")" * depth, ["a"])
        yield Case(f"nest/noncap/{depth}", "(?:" * depth + "a" + ")" * depth, ["a"])
        yield Case(f"nest/class/{depth}", "[" * depth + "a" + "]" * depth, ["a"])


# The Unicode \b and \B ones run on the NFA engines (the DFA cannot), which step
# through positions inside a code point.
POS_PATTERNS = (
    "", "a", "x*", "(a)|b", r"\b", r"\B", "abc", "ö", ".", "$", "^", "(?m)^",
    "(?m)$", "(.)(.)", r"\w+", r"(?-u:\b)", "😀", "😀|b", "(?s).", "[^a]", r"\pL*",
    "b$", "^a", "(?P<x>ä)?(?P<y>.)", ".*", "(?U).+", "a|ä|😀", r"\z", r"\A.",
    r"\b.", r"\B.", r"(\B)(.)?", r"\b(.)?", r"\B\w*", r"(?:\b|ä)(.)?", r".?\B",
    r"\B|ö", r"(\B)|(\b)", r"(?s)\B.*", r"(?:\B)*(.)?(.)?", r"\B(?:)|b", r"(\b)?(\B)?",
)  # fmt: skip
HUGE_POSITIONS = ((1 << 32) + 1, (1 << 63), (1 << 64) - 1)
POS_TEXTS = (
    "", "a", "abc", "äöü", "😀😀", "aä😀b", "😀a😀b😀", "中文字", "a\nb\n",
    "ßẞ", "éé😀", "b",
)  # fmt: skip


def pos_cases() -> Iterator[Case]:
    for p, pattern in enumerate(POS_PATTERNS):
        positions: list[list[int] | None] = []
        for text in POS_TEXTS:
            positions.append([*range(len(text.encode("utf-8")) + 3), *HUGE_POSITIONS])
        yield Case(f"pos/{p:02d}", pattern, list(POS_TEXTS), positions)


NAME_PATTERNS = (
    "(?P<a>x)?(?P<b>y)", "(?P<a>x)|(?P<b>y)", "(a)?(b)?(c)?", "(?<first>\\w+)\\s(?<last>\\w+)",
    "(?P<outer>a(?P<inner>b)?c)", "((a)|(b))+", "(a*)*", "(a|b)*?c", "(?P<n>a)(?:(?P<m>b)|c)",
    "(?P<dup>a)|(?P<other>b)(?P<third>c)?", "(?P<ä>ä)(?P<z_1>\\d)?", "(?:(a)|b)(?:(c)|d)",
    "(?P<x>)", "()()", "(?P<long_name_with_underscores_9>.)", "(?<x>a)(?<y>b)?(?<z>c)??",
)  # fmt: skip
NAME_TEXTS = (
    "xy",
    "y",
    "x",
    "abc",
    "ac",
    "bd",
    "äz",
    "ä1",
    "ababc",
    "",
    "John Smith 😀 Ünïcode Name",
)


def name_cases() -> Iterator[Case]:
    for k, pattern in enumerate(NAME_PATTERNS):
        yield Case(f"names/{k:02d}", pattern, list(NAME_TEXTS))


TOOBIG_FAMILIES = {"pL": r"\pL{{{n}}}", "az": "[a-z]{{{n}}}"}


def toobig_case(family: str, n: int) -> Case:
    return Case(
        f"toobig/{family}/{n}", TOOBIG_FAMILIES[family].format(n=n), ["abc", "äöü" * 3]
    )


# One group per exec.rs MatchType a single pattern can take (Literal, Dfa,
# DfaAnchoredReverse, DfaSuffix, Nfa through the Unicode \b gate, anchored start),
# matched against texts with the literals planted.
STRATEGY_PATTERNS = (
    "foo", "foo|bar|baz", "(?i)hello", "(?i)straße|ǅ", "abc|abd|xyz", "[ab]cd",
    "(?i)k", "foo(bar)?", "(?i)foo|BAR", "x@example\\.com|main\\.rs",
    "foo$", r"\d+$", r"[a-z]+\.txt$", r"(?m)\w+$", "(?s).*bar$",
    r"\w+@example\.com", "[a-z]+ing", r".*\.rs", r"\d+px", r"(\w+)\s+px",
    r"\bfoo\b", r"\b\w+\b", r"(?i)\bstraße\b", r"\B\w{2}\B",
    "^foo", r"\A\d+", "^(a|b)*c", r"(?m)^\w+",
    r"[a-z]+\d+", "(a|b)*abb", "x.*y", "(?s)a.+b", r"(\d{2})[-/](\d{2})",
)  # fmt: skip
WORDS = (
    "foo", "bar", "baz", "hello", "HELLO", "straße", "STRASSE", "test.txt", "main.rs",
    "12px", "x@example.com", "walking", "ä", "😀", "ǅ", "K", "\u212a", "1999", "07-04",
    "abb", "aab", "xy", "zzz", "Ωmega", "中文", "a", "b", "c", "abd", "fooBAR",
)  # fmt: skip


def planted_text(rng: random.Random, length: int) -> str:
    """Words, spaces and newlines up to about length chars."""
    out: list[str] = []
    size = 0
    while size < length:
        word = rng.choice(WORDS)
        out.append(word + rng.choice("  \n,-"))
        size += len(word) + 1
    return "".join(out)


def strategy_cases() -> Iterator[Case]:
    rng = random.Random(f"{SEED}:strategy")
    for k, pattern in enumerate(STRATEGY_PATTERNS):
        texts = [planted_text(rng, n) for n in (0, 20, 120, 400)]
        yield Case(f"strategy/{k:02d}", pattern, texts)


# Multi-KB texts: the literal searchers' long paths and both sides of the
# backtracker's 256 KiB visited-set limit (should_exec) in the default strategy.
LONG_PATTERNS = (
    "needle", "(?i)needle\\d", r"\pL{3}\d", r"(\w+)\s(\w+)", r"[^\n]*needle[^\n]*$",
    r"(?m)^\d+$", r"\b\w{4}\b", "x@example\\.com", r"(\pL+)(\d+)?(\pL{2,4})",
    r"(?:a|b|c|ä|😀){3,}", r"\w+ing\b", r"(?s)foo.*?bar", r"(?:\w+\s){1,100}needle\d",
)  # fmt: skip


DFA_CACHE_PATTERN = "(a|b)*a(a|b){20}c"
DFA_CACHE_CHARS = 200_000


def dfa_cache_text() -> str:
    """DFA_CACHE_CHARS of a/b: PCG64(SEED) raw words, bit 63 picks b."""
    raw = np.random.PCG64(SEED).random_raw(DFA_CACHE_CHARS)
    return "".join("b" if word >> 63 else "a" for word in raw.tolist())


def dfa_cache_cases() -> Iterator[Case]:
    """The lazy DFA's cache fill (2 MiB): (a|b)*a(a|b){20}c needs up to 2**21
    states, so random a/b text fills and flushes the cache until the DFA quits to
    the NFA. /00 never matches; /01 appends "a" + "ab"*10 + "c" and matches.
    Search at 0 and findall only (the texts are 200,000 chars)."""
    text = dfa_cache_text()
    for k, sample in enumerate((text, text + "a" + "ab" * 10 + "c")):
        yield Case(
            f"dfa_cache/{k:02d}", DFA_CACHE_PATTERN, [sample], [[0]], splits=False
        )


def long_cases() -> Iterator[Case]:
    rng = random.Random(f"{SEED}:long")
    for k, pattern in enumerate(LONG_PATTERNS):
        texts = [planted_text(rng, n) + "needle7" for n in (3000, 6000)]
        yield Case(f"long/{k:02d}", pattern, texts)


def build_cases(src_dir: Path = DEFAULT_SRC) -> list[Case]:
    """Every candidate case; families after name_cases() were added later and
    leave the earlier ids and recipes unchanged."""
    cases = [
        *upstream_cases(src_dir),
        *generated_cases(400, 150),
        *special_cases(),
        *pos_cases(),
        *name_cases(),
        *generated_cases(800, 250, ("gen2", "mut2"), ("gen2", "mutate2"), 6, 24),
        *strategy_cases(),
        *long_cases(),
        *dfa_cache_cases(),
    ]
    ids = [c.id for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate case ids")
    return cases


def probe_lines(case: Case) -> list[dict[str, Any]]:
    """The probe cases of case: the case, plus '#midchar' and '#past' when it has
    searches of those classes."""
    by_class: dict[PosClass, list[Op]] = {
        "boundary": [{"op": "compile"}],
        "midchar": [{"op": "compile"}],
        "past": [{"op": "compile"}],
    }
    for k, text in enumerate(case.texts):
        explicit = case.positions[k] if k < len(case.positions) else None
        ops = text_ops(
            text,
            explicit if explicit is not None else edge_positions(text),
            case.splits,
        )
        for cls, class_ops in ops.items():
            by_class[cls] += class_ops
    lines = [{"id": case.id, "pattern": case.pattern, "ops": by_class["boundary"]}]
    for cls in ("midchar", "past"):
        if len(by_class[cls]) > 1:
            lines.append(
                {
                    "id": f"{case.id}#{cls}",
                    "pattern": case.pattern,
                    "ops": by_class[cls],
                }
            )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.out.open("w", encoding="utf-8", newline="\n") as f:
        for case in build_cases(args.src):
            for line in probe_lines(case):
                f.write(json.dumps(line, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
