# libdeflate 1.26

Upstream: https://github.com/ebiggers/libdeflate, release tarball
`https://github.com/ebiggers/libdeflate/releases/download/v1.26/libdeflate-1.26.tar.gz` (2026-08-22), SHA-256
`125856d4656e0feab660f94842f835923410c9281fedbcee64598c918da42b5a` (the digest GitHub publishes for the asset; the
project publishes no signature). OpenEXR 3.5.2 bundles this release.

The files here are copies of the tarball's `libdeflate.h`, `common_defs.h`, the `lib/` sources and headers of
upstream's static library for x86-64 (`lib/arm` and `lib/riscv` left out), `COPYING` (MIT) and `NEWS.md`, unmodified
but for the patches below. Built by `native/cmake/codecs.cmake` as `chainner_deflate` (upstream's default LIB_SOURCES;
its x86 code paths dispatch on CPUID at run time) for OpenEXRCore's ZIP, ZIPS and the DWA and PXR24 stages and the
PNG decoder's direct path (`native/src/codecs/png_codec.c` and `png_inflate.c`: inflate and IDAT CRCs), and linked into
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

## chaiNNer-C patch: decompressing non-final pieces (2026-11-04)

Purpose: the PNG decoder's direct path (`native/src/codecs/png_codec.c`) inflates the pieces of the first patch in
parallel, each into its own part of the output. The patch adds `libdeflate_deflate_decompress_nonfinal()` (declared in
`libdeflate.h`), which decompresses as `libdeflate_deflate_decompress()` does, except that it succeeds only if its
input ends, at the last byte, on an empty non-final stored block and no block of it is final: a final block anywhere,
or input running out anywhere else, is `LIBDEFLATE_BAD_DATA`. A match reaching before the output's start was an error
already, which is what makes a piece that succeeds independent of the pieces before it.

The changes:

- `struct libdeflate_decompressor` (`lib/deflate_decompress.c`) gains `bool nonfinal`, zero from the allocation; the new
  function sets it around its call to `libdeflate_deflate_decompress_ex()`, so every other call sees it false.
- `lib/decompress_template.h`, in all its instantiations: with the flag, the stored-block branch ends the piece after an
  empty non-final block at the end of the input, and a final block returns `LIBDEFLATE_BAD_DATA`.

Without the flag decompression does what it did, for OpenEXRCore too: both tests are on the stored-block branch and
after the final block, none in the Huffman loops. SHA-256 before / patched: `libdeflate.h`
`2d780d8f7733f695da1a074a10804165794f3ceea244859360bdd9f2b2704ca4` (the first patch's) /
`24b18445f7e1efdfed89890ee7c5612547cff67b9f54f1da8e8fe134ab9bf633`, `lib/deflate_decompress.c` (upstream's)
`f4299b3a688b5768c3a8790339e47e02a6e6247d3675b0fafe412a1fc9bd995f` /
`54a6b53a7e1a272139a7b7dfa2ef95cbc7584377ee2140f76d8bc09caede0a21`, `lib/decompress_template.h` (upstream's)
`ade3034c1d5c840cb31bd9738a015be804862a8dfce03c14ac2f5b3a826792b8` /
`5ce57c83310b473ad9413f9a2d6d302bfb31a52e9d25d62329e0d00761a35232`.

## chaiNNer-C patch: decoding a stretch of a stream (2026-11-04)

Purpose: the PNG decoder's parallel inflate (`native/src/codecs/png_inflate.c`) decodes one DEFLATE stream from block
boundaries found by search, on many threads. It needs the decoder to start at a bit offset, stop at a block boundary,
read matches from a history before its output, and, while that history is unknown, carry markers of its positions
through the output. Tables, bit reader, fast loop and checks stay libdeflate's.

The changes:

- `libdeflate.h` gains `struct libdeflate_range`, `libdeflate_deflate_decompress_range()` and
  `libdeflate_deflate_decompress_symbols()`.
- `lib/deflate_decompress.c`: `struct libdeflate_decompressor` gains `range` (NULL outside those two calls), which both
  set around the same decompression routine; a second compilation of the template, `DECODE_SYMBOLS`.
- `lib/decompress_template.h`: with `range`, the bit reader starts at `start_bit`, offsets are checked against the
  output plus `history`, and each block boundary is recorded and may end the call. With `DECODE_SYMBOLS`, the output is
  16-bit: literals as their values, matches copied by words that also note any marker. Literal stores say `(u8)`, which
  a byte output did implicitly.
- `lib/x86/decompress_impl.h`: the BMI2 compilation of the symbol routine.

Without `range`, the two new branches of the existing routines cost one test per call and one per block; every call
OpenEXRCore and the serial PNG path make decodes as before. SHA-256 before (the previous patches') / after: `libdeflate.h`
`24b18445f7e1efdfed89890ee7c5612547cff67b9f54f1da8e8fe134ab9bf633` /
`807b43576b11609665d1bb69949e95e93803be83fdc7c6fa1f83d57db6c3e169`, `lib/decompress_template.h`
`5ce57c83310b473ad9413f9a2d6d302bfb31a52e9d25d62329e0d00761a35232` /
`43adeaf6142a7509b1b295d8d075eda29696bfc540a71b006c823eb2d1105d4c`, `lib/deflate_decompress.c`
`54a6b53a7e1a272139a7b7dfa2ef95cbc7584377ee2140f76d8bc09caede0a21` /
`04daa6741e597a48d72832f988ab048a27d1a00f2b679b9344d203c31fc7e80e`, `lib/x86/decompress_impl.h`
`81ab9fd7d4723fdc7ae81dca82f817bcecb7baf5bdfffde0ccfbba70569ac2f1` /
`1ae72be1edd9da70fae2715e377ac071259de54da66ff664a992b2415e58d3ac`.
