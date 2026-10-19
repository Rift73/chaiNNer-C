# libdeflate 1.26

Upstream: https://github.com/ebiggers/libdeflate, release tarball
`https://github.com/ebiggers/libdeflate/releases/download/v1.26/libdeflate-1.26.tar.gz` (2026-08-22), SHA-256
`125856d4656e0feab660f94842f835923410c9281fedbcee64598c918da42b5a` (the digest GitHub publishes for the asset; the
project publishes no signature). OpenEXR 3.5.2 bundles this release.

The files here are unmodified copies of the tarball's `libdeflate.h`, `common_defs.h`, the `lib/` sources and
headers of upstream's static library for x86-64 (`lib/arm` and `lib/riscv` left out), `COPYING` (MIT) and
`NEWS.md`. Built by `native/cmake/codecs.cmake` as `chainner_deflate` (upstream's default LIB_SOURCES; its x86 code
paths dispatch on CPUID at run time) for OpenEXRCore's ZIP, ZIPS and the DWA and PXR24 stages.
