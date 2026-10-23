# Vernier 0.2 analysis core

Source: the Vernier 0.2 C++ source archive `Vernier-0.2-Cpp-source.zip`, SHA-256
`4b60506d0bad3de5aca3ca3eb847ab88be4c5a61edb1e945995b622777a8c28d`. Vernier is a C++ successor of getnative and
keeps its MIT license (`LICENSE`, the archive's top-level file).

The files here are byte-identical copies of the archive's `cpp/src/resample.cpp`, `cpp/src/analyze.cpp`,
`cpp/src/spectrum.cpp` and their headers in `cpp/include/`, plus `LICENSE` (`.gitattributes` keeps their bytes, the
license's CRLF line ends included). No patch. Vernier's Qt layers (media, report, batch, GUI, CLI) and its CPU
affinity code are not vendored; `native/src/getnative_unit.cpp` rebuilds the report, parameter and summary JSON and
the sampling rules without Qt.

| File | SHA-256 |
|---|---|
| `LICENSE` | `5bbd8ceb3da187bbffc87793cb306cb6b52862ab70935bd24bf46e68fdfe7b99` |
| `include/analyze.hpp` | `53ebe3a580449d5e7dd69641d7845bf22654712c4d54f0e5bffed08388e62523` |
| `include/resample.hpp` | `4187172d1c57d6d63755b4562ec3d79a53dd0cccff47407dee6572db8cfb2ec8` |
| `include/spectrum.hpp` | `2ecd1415628848982d4b652d8f36eeee83e762ad643490dfaee3b09b59e108d2` |
| `src/analyze.cpp` | `1e627bab703c700c03d0c003b7acc0b2df9900444eae03e7614bf5dc7b8ba13a` |
| `src/resample.cpp` | `e0401f7c8043dcb83636605eaca19ed6c7ed3a2ae88e254a92f5019e8e891aa9` |
| `src/spectrum.cpp` | `204ac33d07707ebffd90982689493e57a9f5e0bfdb73ef92cd29ef0303b0998e` |

Built by `native/CMakeLists.txt` twice, as the static libraries `chainner_vernier_base` (baseline x86-64) and
`chainner_vernier_avx2` (chaiNNer's AVX2 level), each with `native/src/getnative_unit.cpp` and
`native/src/getnative_kernel.cpp`. Each build renames the sources' namespace on the command line
(`/Dnative=cn_vernier_base`, `/Dnative=cn_vernier_avx2`; `native` occurs in the sources only as the namespace name),
so the two link into `chainner_native.dll` side by side; `native/src/getnative_ops.cpp` picks one per session by
`cn_isa_current()` (`CHAINNER_C_ISA` caps it). Both use Vernier's own floating-point model (the compiler's default,
not chaiNNer's `/fp:strict`) with contraction off (`-ffp-contract=off`), so the two builds, and a local `-march`
build, give bit-identical results.
Vernier's `std::thread` workers are kept; no CPU affinity is changed.
