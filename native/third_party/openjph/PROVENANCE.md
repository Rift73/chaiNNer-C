# OpenJPH 0.32.0

Upstream: https://github.com/aous72/OpenJPH, tag `0.32.0` (2026-09-17), source archive
`https://github.com/aous72/OpenJPH/archive/refs/tags/0.32.0.tar.gz`, SHA-256
`5cb1ebe18e5ee1322d23ff2130b37562d447d1ad5586c3d16cf84be3ae30f719` (the project publishes no release asset or
signature). Its `src/core` is byte-identical to the copy in OpenEXR 3.5.2's signed tarball (`external/OpenJPH`), apart
from that copy's CMakeLists.txt.

`src/core/` holds unmodified copies of the archive's `src/core` sources and headers for x86-64 (the WASM, VSX and
NEON units left out), with `LICENSE` (BSD-2-Clause). Built by `native/cmake/codecs.cmake` as `chainner_openjph` with
upstream's per-file flags for clang (SSE, SSE2, SSSE3, AVX, AVX2 and AVX-512 kernels chosen by its CPUID check at
run time) for OpenEXRCore's HTJ2K and lossy HTJ2K compressions.
