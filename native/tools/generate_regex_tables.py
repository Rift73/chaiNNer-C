"""Convert regex-syntax 0.7.2's Unicode tables (src/unicode_tables/*.rs) to C.

The tables are ucd-generate 0.2.14 output for Unicode 15.0.0. This reads the pinned
crate sources in native/runtime/download/src (git-ignored, hash-verified when they
were downloaded) and writes native/src/regex/unicode_tables/: one .c per .rs except
mod.rs, plus unicode_tables.h. It understands only the literal forms those files use
(constant arrays of chars, strings and tuples); it is not a Rust parser. Every
generated file carries the Unicode licence notice. Run it with any Python 3.10+:
  python -B native/tools/generate_regex_tables.py
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "native/runtime/download/src/regex-syntax-0.7.2/src/unicode_tables"
OUTPUT = ROOT / "native/src/regex/unicode_tables"

# Files compiled out by regex 1.8.4's feature set (mod.rs: perl_decimal needs
# unicode-perl without unicode-gencat, perl_space unicode-perl without unicode-bool).
# unicode.rs then reads general_category's Decimal_Number and property_bool's
# White_Space instead, so these two get no data.
CFG_OUT = {"perl_decimal", "perl_space"}

TOKEN = re.compile(
    r"""\s+|//[^\n]*|(?P<char>'(?:\\u\{[0-9a-fA-F]+\}|\\.|[^\\'])')"""
    r"""|(?P<str>"(?:\\.|[^\\"])*")|(?P<ident>[A-Za-z_][A-Za-z0-9_]*)"""
    r"""|(?P<punct>[&\[\](),:;=<>'])""",
    re.S,
)
ESCAPES = {"0": 0, "t": 9, "n": 10, "r": 13, "\\": 92, "'": 39, '"': 34}


def unescape(body: str) -> list[int]:
    out: list[int] = []
    i = 0
    while i < len(body):
        c = body[i]
        if c != "\\":
            out.append(ord(c))
            i += 1
        elif body[i + 1] == "u":
            end = body.index("}", i)
            out.append(int(body[i + 3 : end], 16))
            i = end + 1
        elif body[i + 1] == "x":
            out.append(int(body[i + 2 : i + 4], 16))
            i += 4
        else:
            out.append(ESCAPES[body[i + 1]])
            i += 2
    return out


def tokens(text: str) -> list[tuple[str, object]]:
    out: list[tuple[str, object]] = []
    pos = 0
    while pos < len(text):
        m = TOKEN.match(text, pos)
        assert m is not None, text[pos : pos + 40]
        pos = m.end()
        if m.group("char"):
            value = unescape(m.group("char")[1:-1])
            assert len(value) == 1
            out.append(("char", value[0]))
        elif m.group("str"):
            out.append(("str", "".join(map(chr, unescape(m.group("str")[1:-1])))))
        elif m.group("ident"):
            out.append(("ident", m.group("ident")))
        elif m.group("punct"):
            out.append(("punct", m.group("punct")))
    return out


class Parser:
    def __init__(self, toks: list[tuple[str, object]]) -> None:
        self.toks = toks
        self.i = 0

    def peek(self) -> tuple[str, object]:
        return self.toks[self.i]

    def take(self, kind: str, value: object = None) -> object:
        tok = self.toks[self.i]
        assert tok[0] == kind and (value is None or tok[1] == value), (tok, kind, value)
        self.i += 1
        return tok[1]

    def value(self) -> object:
        kind, val = self.peek()
        if kind == "punct" and val == "&":
            self.i += 1
            self.take("punct", "[")
            items = []
            while self.peek() != ("punct", "]"):
                items.append(self.value())
                if self.peek() == ("punct", ","):
                    self.i += 1
            self.i += 1
            return items
        if kind == "punct" and val == "(":
            self.i += 1
            items = []
            while self.peek() != ("punct", ")"):
                items.append(self.value())
                if self.peek() == ("punct", ","):
                    self.i += 1
            self.i += 1
            return tuple(items)
        self.i += 1
        if kind == "ident":
            return ("ref", val)
        return val

    def consts(self) -> dict[str, object]:
        out: dict[str, object] = {}
        while self.i < len(self.toks):
            self.take("ident", "pub")
            self.take("ident", "const")
            name = str(self.take("ident"))
            while self.peek() != ("punct", "="):
                self.i += 1
            self.i += 1
            out[name] = self.value()
            self.take("punct", ";")
        return out


def license_notice() -> str:
    text = (SOURCE / "LICENSE-UNICODE").read_text(encoding="utf-8")
    start = text.index("COPYRIGHT AND PERMISSION NOTICE")
    return "\n".join(
        (" * " + line).rstrip() for line in text[start:].strip().splitlines()
    )


def c_string(s: str) -> str:
    assert all(32 <= ord(c) < 127 and c not in '"\\' for c in s), s
    return '"' + s + '"'


# The parse tree is untyped (Parser.value); each table checks the shape it reads.
def as_list(value: object) -> list[object]:
    assert isinstance(value, list), value
    return value


def as_pair(value: object) -> tuple[object, object]:
    assert isinstance(value, tuple) and len(value) == 2, value
    return value[0], value[1]


def as_int(value: object) -> int:
    assert isinstance(value, int), value
    return value


def as_str(value: object) -> str:
    assert isinstance(value, str), value
    return value


def as_ranges(value: object) -> list[tuple[int, int]]:
    pairs = [as_pair(item) for item in as_list(value)]
    return [(as_int(start), as_int(end)) for start, end in pairs]


def ranges_c(name: str, ranges: list[tuple[int, int]]) -> str:
    rows = [f"    {{0x{s:X}, 0x{e:X}}}," for s, e in ranges]
    return (
        f"static const chn_hir_class_unicode_range {name}[] = {{\n"
        + "\n".join(rows)
        + "\n};\n"
    )


def generate(stem: str, header_comment: str, notice: str) -> str:
    out = [
        (
            f"/* Port of regex-syntax 0.7.2 src/unicode_tables/{stem}.rs, MIT OR Apache-2.0. "
            "Unicode data: Unicode License (LICENSE-UNICODE). */"
        ),
        "/* Generated by native/tools/generate_regex_tables.py; do not edit. The Rust source says:",
        header_comment,
        " *",
        " * Unicode License (regex-syntax 0.7.2 src/unicode_tables/LICENSE-UNICODE):",
        " *",
        notice,
        " */",
        '#include "unicode_tables.h"',
        "",
    ]
    if stem in CFG_OUT:
        out.insert(
            -1,
            "/* regex 1.8.4's features compile this table out (see the generator); no data. */",
        )
        return "\n".join(out)
    text = (SOURCE / f"{stem}.rs").read_text(encoding="utf-8")
    consts = Parser(tokens(text)).consts()
    prefix = f"chn_ut_{stem}"
    body: list[str] = []
    if stem == "case_folding_simple":
        table = as_list(consts.pop("CASE_FOLDING_SIMPLE"))
        values: list[int] = []
        rows = []
        for key, mapped in map(as_pair, table):
            folded = [as_int(v) for v in as_list(mapped)]
            rows.append(f"    {{0x{as_int(key):X}, {len(values)}, {len(folded)}}},")
            values.extend(folded)
        body.append(
            f"const uint32_t {prefix}_values[] = {{\n"
            + "\n".join(
                "    " + ", ".join(f"0x{v:X}" for v in values[i : i + 8]) + ","
                for i in range(0, len(values), 8)
            )
            + "\n};\n"
        )
        body.append(
            f"const chn_ut_case_fold {prefix}[] = {{\n" + "\n".join(rows) + "\n};\n"
        )
        body.append(f"const size_t {prefix}_len = {len(table)};\n")
    elif stem == "property_names":
        table = as_list(consts.pop("PROPERTY_NAMES"))
        rows = [
            f"    {{{c_string(as_str(a))}, {c_string(as_str(b))}}},"
            for a, b in map(as_pair, table)
        ]
        body.append(
            f"const chn_ut_name_pair {prefix}[] = {{\n" + "\n".join(rows) + "\n};\n"
        )
        body.append(f"const size_t {prefix}_len = {len(table)};\n")
    elif stem == "property_values":
        table = as_list(consts.pop("PROPERTY_VALUES"))
        rows = []
        for prop, vals in map(as_pair, table):
            name = as_str(prop)
            pairs = [as_pair(v) for v in as_list(vals)]
            ident = "values_" + name.lower()
            body.append(
                f"static const chn_ut_name_pair {ident}[] = {{\n"
                + "\n".join(
                    f"    {{{c_string(as_str(a))}, {c_string(as_str(b))}}},"
                    for a, b in pairs
                )
                + "\n};\n"
            )
            rows.append(f"    {{{c_string(name)}, {ident}, {len(pairs)}}},")
        body.append(
            f"const chn_ut_property_value_set {prefix}[] = {{\n"
            + "\n".join(rows)
            + "\n};\n"
        )
        body.append(f"const size_t {prefix}_len = {len(table)};\n")
    elif stem == "perl_word":
        body.append(ranges_c("PERL_WORD", as_ranges(consts.pop("PERL_WORD"))))
        body.append(f"const chn_hir_class_unicode_range *const {prefix} = PERL_WORD;\n")
        body.append(
            f"const size_t {prefix}_len = sizeof(PERL_WORD) / sizeof(PERL_WORD[0]);\n"
        )
    else:
        by_name = as_list(consts.pop("BY_NAME"))
        for name, value in consts.items():
            body.append(ranges_c(name, as_ranges(value)))
        rows = []
        for name, ref in map(as_pair, by_name):
            kind, target = as_pair(ref)
            assert kind == "ref" and target in consts, ref
            rows.append(
                f"    {{{c_string(as_str(name))}, {target}, sizeof({target}) / sizeof({target}[0])}},"
            )
        body.append(
            f"const chn_ut_named_ranges {prefix}_by_name[] = {{\n"
            + "\n".join(rows)
            + "\n};\n"
        )
        body.append(f"const size_t {prefix}_by_name_len = {len(by_name)};\n")
        consts.clear()
    assert not consts, (stem, list(consts))
    return "\n".join(out) + "\n".join(body)


HEADER = """/* Port of regex-syntax 0.7.2 src/unicode_tables/mod.rs, MIT OR Apache-2.0. Unicode data: Unicode License (LICENSE-UNICODE). */
/* Generated by native/tools/generate_regex_tables.py; do not edit. The C shapes of
 * regex-syntax 0.7.2's Unicode 15.0.0 tables (ucd-generate 0.2.14), private to the regex
 * front end. Every array is sorted as in the Rust source: names in byte order (for
 * binary search), ranges and case folding keys ascending. */
#ifndef CHAINNER_REGEX_UNICODE_TABLES_H
#define CHAINNER_REGEX_UNICODE_TABLES_H
#include "chainner_hir.h"

/* (&'static str, &'static [(char, char)]): a BY_NAME entry. */
typedef struct chn_ut_named_ranges {
    const char *name;
    const chn_hir_class_unicode_range *ranges;
    size_t len;
} chn_ut_named_ranges;

/* (&'static str, &'static str): a normalized alias and its canonical name. */
typedef struct chn_ut_name_pair {
    const char *name;
    const char *canonical;
} chn_ut_name_pair;

/* (&'static str, &'static [(&'static str, &'static str)]): a PROPERTY_VALUES entry. */
typedef struct chn_ut_property_value_set {
    const char *property;
    const chn_ut_name_pair *values;
    size_t len;
} chn_ut_property_value_set;

/* (char, &'static [char]): a CASE_FOLDING_SIMPLE entry; its slice is
 * chn_ut_case_folding_simple_values[offset .. offset + len]. */
typedef struct chn_ut_case_fold {
    uint32_t c;
    uint32_t offset;
    uint32_t len;
} chn_ut_case_fold;

"""

MODULES = [
    ("age", "by_name"),
    ("case_folding_simple", None),
    ("general_category", "by_name"),
    ("grapheme_cluster_break", "by_name"),
    ("perl_decimal", None),
    ("perl_space", None),
    ("perl_word", None),
    ("property_bool", "by_name"),
    ("property_names", None),
    ("property_values", None),
    ("script", "by_name"),
    ("script_extension", "by_name"),
    ("sentence_break", "by_name"),
    ("word_break", "by_name"),
]


def header() -> str:
    decls = []
    for stem, kind in MODULES:
        p = f"chn_ut_{stem}"
        if kind == "by_name":
            decls.append(
                f"/* {stem}::BY_NAME */\nextern const chn_ut_named_ranges {p}_by_name[];\nextern const size_t {p}_by_name_len;"
            )
    decls.append(
        "/* case_folding_simple::CASE_FOLDING_SIMPLE */\n"
        "extern const chn_ut_case_fold chn_ut_case_folding_simple[];\n"
        "extern const size_t chn_ut_case_folding_simple_len;\n"
        "extern const uint32_t chn_ut_case_folding_simple_values[];"
    )
    decls.append(
        "/* perl_word::PERL_WORD */\n"
        "extern const chn_hir_class_unicode_range *const chn_ut_perl_word;\n"
        "extern const size_t chn_ut_perl_word_len;"
    )
    decls.append(
        "/* property_names::PROPERTY_NAMES */\n"
        "extern const chn_ut_name_pair chn_ut_property_names[];\n"
        "extern const size_t chn_ut_property_names_len;"
    )
    decls.append(
        "/* property_values::PROPERTY_VALUES */\n"
        "extern const chn_ut_property_value_set chn_ut_property_values[];\n"
        "extern const size_t chn_ut_property_values_len;"
    )
    return HEADER + "\n\n".join(decls) + "\n\n#endif\n"


def main() -> None:
    notice = license_notice()
    for stem, _ in MODULES:
        text = (SOURCE / f"{stem}.rs").read_text(encoding="utf-8")
        lines = []
        for line in text.splitlines():
            if not line.startswith("//"):
                break
            lines.append((" * " + line[2:].strip()).rstrip())
        out = generate(stem, "\n".join(lines), notice)
        (OUTPUT / f"{stem}.c").write_text(out, encoding="utf-8", newline="\n")
    (OUTPUT / "unicode_tables.h").write_text(header(), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
