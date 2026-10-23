# libdeflate 1.26

Upstream: https://github.com/ebiggers/libdeflate, release tarball
`https://github.com/ebiggers/libdeflate/releases/download/v1.26/libdeflate-1.26.tar.gz` (2026-08-22), SHA-256
`125856d4656e0feab660f94842f835923410c9281fedbcee64598c918da42b5a` (the digest GitHub publishes for the asset; the
project publishes no signature). OpenEXR 3.5.2 bundles this release.

The files here are copies of the tarball's `libdeflate.h`, `common_defs.h`, the `lib/` sources and headers of
upstream's static library for x86-64 (`lib/arm` and `lib/riscv` left out), `COPYING` (MIT) and `NEWS.md`, unmodified
but for the patch below. Built by `native/cmake/codecs.cmake` as `chainner_deflate` (upstream's default LIB_SOURCES;
its x86 code paths dispatch on CPUID at run time) for OpenEXRCore's ZIP, ZIPS and the DWA and PXR24 stages and the
PNG decoder's direct path (`native/src/codecs/png_codec.c`: inflate and IDAT CRCs), and linked into
`chainner_native.dll` for the default PNG encoder (`native/src/png_encode.c`).

## chaiNNer-C patch: non-final pieces (2026-10-21)

Purpose: the default PNG encoder compresses an image's row bands in parallel, each band with its own compressor, and
joins the pieces into one zlib stream. A piece other than the last must not end the stream: no block of it may carry
BFINAL, and it must end on a byte boundary so the next piece's first block can follow it. The patch adds
`libdeflate_deflate_compress_nonfinal()` (declared in `libdeflate.h`), which compresses exactly as
`libdeflate_deflate_compress()` does but marks no block final and ends the output with an empty non-final stored block
(the header bits, alignment, then `00 00 FF FF`), as zlib's `Z_SYNC_FLUSH` ends its output; its output takes at most
`libdeflate_deflate_compress_bound()` + 5 bytes. The probe of 2026-10-21 (MinSum filter, level 6, 32 bands) did the
same through a thread-local switch; here the switch is a field of the call's output bitstream, so nothing is shared
between calls.

The changes, all in `lib/deflate_compress.c` but the declaration:

- `struct deflate_output_bitstream` gains `bool nonfinal`; `deflate_flush_block()` writes BFINAL 0 when it is set.
- `deflate_compress_none()` (inputs of at most the compressor's pass-through size, and level 0) takes a `final` flag:
  without it, no stored block is final and an empty non-final stored block follows them.
- `libdeflate_deflate_compress()`'s body moves to `deflate_compress_ending(..., final)`, which both public functions
  call; without `final` it writes the empty stored block after the last block's bits instead of the final byte.

`libdeflate_deflate_compress()`, and with it the zlib and gzip wrappers OpenEXRCore uses, writes the same bytes as
before. SHA-256 upstream / patched: `libdeflate.h` `1bb50a3ff8c6980c1031c0e9bb324cf99df51379fdca77a358644cca841765cc`
/ `2d780d8f7733f695da1a074a10804165794f3ceea244859360bdd9f2b2704ca4`, `lib/deflate_compress.c`
`a46dae12f2020b68d20c965ae43f9710e3e792bf77f3ca4611582f8d8d0c9541` /
`245eae0a7eaf45c4f80454fb3b916ad1084f67fe3fa1ea032808ba59f4e3d675` (decision, 2026-10-21).
