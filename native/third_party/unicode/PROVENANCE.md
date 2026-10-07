# Unicode 15.0.0 data

The tables in `native/src/regex/unicode_tables/` (ages, simple case folding, general
categories, grapheme cluster, word and sentence breaks, the Perl decimal, space and word
classes, boolean properties, property names and values, scripts and script extensions) are
Unicode 15.0.0 data. `native/tools/generate_regex_tables.py` converts them
to C from regex-syntax 0.7.2's `src/unicode_tables/*.rs`, which ucd-generate 0.2.14 produced
from the Unicode Character Database 15.0.0.

- **Licence:** the Unicode, Inc. License Agreement - Data Files and Software, the licence the
  15.0.0 data was published under. `LICENSE-UNICODE` is regex-syntax 0.7.2's copy
  (`src/unicode_tables/LICENSE-UNICODE`), unmodified.
- Source archive: see `../regex-syntax/PROVENANCE.md`.
