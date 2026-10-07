# chainner_ext 0.3.10

`chainner_ext` is chaiNNer-C's C module (since 2026-10-06), a drop-in for chainner_ext 0.3.10
(chaiNNer-rs). `native/src/chainner_ext/*.c` ports its bindings (`crates/bindings`) and
regex-py's split and position code (`crates/regex-py`); its image functions bind the kernels
of `chainner_native.dll`. `backend/src/chainner_ext/__init__.py` is the wheel's re-export and
`__init__.pyi` the wheel's stub (annotations written as `dict`/`list`). Each ported file opens
with one line naming its Rust origin and this licence.

- **Source:** the PyPI sdist `chainner_ext-0.3.10.tar.gz`, SHA-256
  `77f4f67aed984ca8ccfbfdcf9dbdf41968f075d49dc929cc4d076474e7397e4b`, verified against
  PyPI's hash. It is kept, unbuilt, in the git-ignored `native/runtime/download/src`.
- **Licence:** MIT OR Apache-2.0, as its `PKG-INFO` and `Cargo.toml` declare (authors: the
  chaiNNer team). The sdist ships no licence files, so `LICENSE-MIT` (Copyright (c) 2023
  Michael Schmidt) and `LICENSE-APACHE` are the project's own, unmodified: chaiNNer-rs at
  commit `3b6687c857`, their only revision (2023-06-11, before 0.3.10), downloaded 2026-10-06.
  SHA-256: `LICENSE-MIT` `2a68c03a00b314d2c1fd0403e0d0ce9fb687d84ebd47d74106667b21b5f61749`,
  `LICENSE-APACHE` `a60eea817514531668d7e00765731449fe14d059d3249e0bc93b36de45f759f2`.
- Upstream project: https://github.com/chaiNNer-org/chaiNNer-rs
