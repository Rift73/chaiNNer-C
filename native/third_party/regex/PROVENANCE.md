# regex 1.8.4

chaiNNer-C's regex engine (`native/src/regex/`, built as `chainner_regex.lib` and linked into
`chainner_ext.pyd` and `regex_probe.exe`) is a C port of the parts of regex 1.8.4 that
chainner_ext 0.3.10 reaches: the compiler and its size check, the PikeVM, the bounded
backtracker, the lazy DFA, the exec strategy with its literal matchers, and the match and
split iteration. Each ported file opens with one line naming its Rust origin and this licence;
`native/src/regex/README.md` maps every file.

- **Source:** the crates.io archive `regex-1.8.4.crate`, SHA-256
  `d0ab3ca65655bb1e41f2a8c8cd662eb4fb035e67c3f78da1d61dffe89d07300f`, verified against
  crates.io's checksum. It is kept, unbuilt, in the git-ignored `native/runtime/download/src`.
- **Licence:** MIT OR Apache-2.0, Copyright (c) 2014 The Rust Project Developers.
  `LICENSE-MIT` and `LICENSE-APACHE` are the crate's own files, unmodified.
- Upstream project: https://github.com/rust-lang/regex
