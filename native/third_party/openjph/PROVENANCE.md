# OpenJPH 0.32.0

Upstream: https://github.com/aous72/OpenJPH, tag `0.32.0` (2026-09-17), source archive
`https://github.com/aous72/OpenJPH/archive/refs/tags/0.32.0.tar.gz`, SHA-256
`5cb1ebe18e5ee1322d23ff2130b37562d447d1ad5586c3d16cf84be3ae30f719` (the project publishes no release asset or
signature). Its `src/core` is byte-identical to the copy in OpenEXR 3.5.2's signed tarball (`external/OpenJPH`), apart
from that copy's CMakeLists.txt.

`src/core/` holds copies of the archive's `src/core` sources and headers for x86-64 (the WASM, VSX and NEON units left
out), unmodified but for the patch below, with `LICENSE` (BSD-2-Clause). Built by `native/cmake/codecs.cmake` as
`chainner_openjph` with upstream's per-file flags for clang (SSE, SSE2, SSSE3, AVX, AVX2 and AVX-512 kernels chosen by
its CPUID check at run time) for OpenEXRCore's HTJ2K and lossy HTJ2K compressions.

## chaiNNer-C patch: the NLT marker's one-byte points (2026-10-23)

`param_nlt::read` (`codestream/ojph_params.cpp`) read a one-byte-per-point NLT table with the size of its whole
store (`len`) into the store's last part, past the end of the allocation (a heap buffer overflow found by
`native/tests/fuzz`'s OpenEXR harness on a damaged HTJ2K chunk). The patch reads `num_points` bytes, as upstream's
master branch does (`file->read(p->rec.marker_points, p->rec.num_points) == p->rec.num_points`); the two- and
four-byte cases already read one point at a time. SHA-256 upstream / patched: `codestream/ojph_params.cpp`
`df30d31c64a8bc0b53561c6f21f0e788c96d5e21642a2f4b7493f02552f5b3d3` /
`5a74eb9c4291fd41cc39d335089ae0f768c81c99f001b8ca1cefdd39674d9e5f`. At the next OpenJPH refresh, take a release
that carries upstream's fix and drop this patch (stand-in's ruling 9, 2026-10-23).
