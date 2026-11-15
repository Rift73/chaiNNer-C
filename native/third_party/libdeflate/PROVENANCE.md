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

## chaiNNer-C patch: compressing a piece with a history (2026-11-15)

Purpose: the default PNG encoder's higher efforts compress each band of rows with the 32 KiB of filtered rows before
it as a history its matches may reach, so that a band costs no more than the same rows in one stream would, and
optionally let the near-optimal parser fit its costs on that history before the band's first block. The patch adds
`libdeflate_deflate_compress_history()` (declared in `libdeflate.h`), which compresses as
`libdeflate_deflate_compress()` (with `final`) or `libdeflate_deflate_compress_nonfinal()` does, with up to 32768
bytes before the input as history.

The changes, all in `lib/deflate_compress.c` but the declaration:

- `struct libdeflate_compressor` gains `history_nbytes` and `history_warm`, zero and false outside the new function,
  which sets them around its call to `deflate_compress_ending()`.
- `deflate_compress_lazy_generic()` (levels 5 to 9) inserts the history into its hash chains before the input
  (`hc_matchfinder_skip_bytes`).
- `deflate_compress_near_optimal()` (levels 10 to 12) inserts it into its binary trees, one position at a time
  (`bt_matchfinder_skip_byte`); with `warm` it instead parses the history as blocks of its own, which end at the
  input's start (a long match's skip stops there) and which `deflate_optimize_and_flush_block()` (a new `emit` flag)
  optimizes but does not write.
- The fastest and greedy compressors (levels 1 to 4) ignore the history.

Every other call compresses as before: without the new function the fields are zero, the history loops run no
iteration, every block is written and the skip's bound is never reached. SHA-256 before (the previous patches') /
after: `libdeflate.h` `807b43576b11609665d1bb69949e95e93803be83fdc7c6fa1f83d57db6c3e169` / `32fc9f299bf3619981b2d3fd2d3b1311a4c5c38689ba38b286c92f63428bc6b3`,
`lib/deflate_compress.c` `245eae0a7eaf45c4f80454fb3b916ad1084f67fe3fa1ea032808ba59f4e3d675` / `14e4d04c4efcd14654c6c31b6a4e55116564a2832d6f97feab44032508d7096e`.

## chaiNNer-C patch: matches found ahead, and the minimum-cost path with AVX2 (2026-11-15)

Purpose: the default PNG encoder's top effort writes one stream at level 12 in about a third of the time: the
near-optimal compressor's match finding runs in parallel pieces of the input, and its parse, which stays on one thread,
finds each block's minimum-cost path with AVX2.

The changes, all in `lib/deflate_compress.c` but the declarations:

- `libdeflate_find_matches()` writes, for each position of a piece, the matches the binary-tree match finder returns
  there, its trees primed with the bytes before the piece (`bt_matchfinder_skip_byte`); primed with 32768 bytes it
  gives the matches of one search over the whole input (on the PNG encoder's test image, no position differs).
  `libdeflate_deflate_compress_matches()` compresses the whole input as `libdeflate_deflate_compress()` does at levels
  10 to 12, but `deflate_compress_near_optimal()` copies each position's matches from them (a new `ext` field of
  the compressor, NULL outside the call) instead of searching; with one search's matches the output is the same.
- `deflate_find_min_cost_path()` runs `min_cost_path_avx2()` where the CPU has AVX2 (`HAVE_AVX2`, libdeflate's own
  check): the same choices (the literal first, then the lowest cost, ties to the shorter length), with the costs of
  the 10 nodes after the current one kept in registers and, for a position of at most 8 matches no longer than 10, a
  path without branches in which each length lane counts the matches shorter than it and the offset slots are
  computed. Positions with more or longer matches take `min_cost_matches_avx2()`, 8 lengths at a time.

Every other call compresses as before, and the AVX2 path writes the same bytes as the scalar one (checked on the
encoder's test images and its random sweep). SHA-256 before (the previous patches') / after: `libdeflate.h`
`32fc9f299bf3619981b2d3fd2d3b1311a4c5c38689ba38b286c92f63428bc6b3` / `d6befea3e8f7f5b9fb12beab128c94cd9ce7050e27adb51d56dfd15318339f31`, `lib/deflate_compress.c`
`14e4d04c4efcd14654c6c31b6a4e55116564a2832d6f97feab44032508d7096e` / `84af80ba5bf794ef4b8d778e0c39199ea8ff5ab30c4b5b48a0be5c46dcf43846`.

## chaiNNer-C patch: matches found ahead for a piece of a stream (2026-11-16)

Purpose: the PNG encoder's top effort parses a large image as a few segments on their own threads from the matches
found once for the whole image, each segment first parsing a lead-in before it (not written) to fit its costs.
`libdeflate_deflate_compress_matches()` gains `history_nbytes`, `warm` and `final` as
`libdeflate_deflate_compress_history()` has them (the matches of the history's positions come from the same lists,
indexed below the input's start; with `warm` the history may be longer than 32768 bytes), and a position's matches that
reach past the input's end are cut there, as the search would have found them. With one segment and no history the
output is as before. SHA-256 before (the previous patch's) / after: `libdeflate.h`
`d6befea3e8f7f5b9fb12beab128c94cd9ce7050e27adb51d56dfd15318339f31` / `8962028e1eced93dd63687692d76b69a1bafc06ea2350f5d77f7713101b36f35`, `lib/deflate_compress.c`
`84af80ba5bf794ef4b8d778e0c39199ea8ff5ab30c4b5b48a0be5c46dcf43846` / `f1ba168e1c9fba69c34578b9974e23b120e0d8cd57e394398e26e2b5e7149905`.

## chaiNNer-C patch: a stream's blocks planned, refined and written in three steps (2026-11-16)

Purpose: the PNG encoder's smallest effort leaves the near-optimal parse's local optimum with the threads the serial
parse leaves idle: once the serial parse has fixed every block and the costs it came from, each block is parsed again
from those costs changed at random, on any thread, and the smallest parse of each block is written.

The changes, all in `lib/deflate_compress.c` but the declarations (`struct libdeflate_block_plan` and three functions
in `libdeflate.h`):

- `libdeflate_plan_blocks()` runs `libdeflate_deflate_compress_matches()` with a plan array set in the compressor:
  `deflate_optimize_and_flush_block()` then records each block (its place, its mode: the costs' parse, literals only or
  the static codes, the costs that gave its best parse, its size in bits) and copies its match cache, instead of
  writing it.
- `libdeflate_refine_block()` loads a block's cache into any level-10-to-12 compressor and runs restarts from its costs,
  each changed by an xorshift generator seeded by the caller's seed and the restart's number, each to convergence as
  the passes of `deflate_optimize_and_flush_block()` are, keeping the smallest true cost.
- `libdeflate_write_blocks()` writes the plans through `deflate_compress_ending()` with a compressor function that
  parses each block once from its costs (or chooses its literals, or the static codes) and flushes it.

No other call changes. SHA-256 before (the previous patch's) / after: `libdeflate.h` `8962028e1eced93dd63687692d76b69a1bafc06ea2350f5d77f7713101b36f35` / `373c6fc49b65dcd4af8108363aa3875214ff5474e75e2eaccd40a449c39bddf0`,
`lib/deflate_compress.c` `f1ba168e1c9fba69c34578b9974e23b120e0d8cd57e394398e26e2b5e7149905` / `8c14815bdbbe74c4382066c0430794c70fac2b8c2c4d37e2a68a1072b0264e40`.

## chaiNNer-C patch: libdeflate_find_matches() with its own match finder (2026-11-16)

Purpose: the PNG encoder's pieces of match finding each took a whole level-12 compressor (about 10 MB, mostly its
match cache and optimum nodes) where only the binary trees are used. `libdeflate_find_matches()` now takes the
compressor as `const` for its settings and allocates a `struct bt_matchfinder` of its own, so that one compressor
serves every piece at once; the matches are the same. SHA-256 before (the previous patch's) / after: `libdeflate.h`
`373c6fc49b65dcd4af8108363aa3875214ff5474e75e2eaccd40a449c39bddf0` / `6486dd2db70873b64019c35983479c8b0eee3ddc22b1a8c5799f8825260495ce`, `lib/deflate_compress.c` `8c14815bdbbe74c4382066c0430794c70fac2b8c2c4d37e2a68a1072b0264e40` / `a909596f87bfd4ba8c3396b94994ab378db336902d54e5a0f997321b39015b46`.

## chaiNNer-C patch: refining a block from other costs (2026-11-16)

Purpose: rounds of refinement in which each block also starts from the costs its predecessor's best parse came from,
as the serial parse's chain of blocks does. `libdeflate_refine_block()` gains `start`: when not NULL its restarts
change those costs instead of the plan's, and its restart 0 tries them unchanged. SHA-256 before (the previous
patch's) / after: `libdeflate.h` `6486dd2db70873b64019c35983479c8b0eee3ddc22b1a8c5799f8825260495ce` / `d80f77f769214c1db4a211ff173942950780524611c1f5d1bc29e60fba246df1`, `lib/deflate_compress.c` `a909596f87bfd4ba8c3396b94994ab378db336902d54e5a0f997321b39015b46` / `2258840b6ff5bef85cb8536dad683b8e6e774d6752026bf0ea46e761c20840e0`.

## chaiNNer-C patch: match lists from a window (2026-11-16)

Purpose: the PNG encoder keeps only a few chunks of match lists ahead of each parse instead of the whole image's.
`struct libdeflate_matches` gains `window` and `context`: when `window` is set, `deflate_compress_near_optimal()`
asks it for the lists of the position it reaches (positions only increase) and uses them until the end the call
gives. The lists, and so the output, are the same as from whole arrays. SHA-256 before (the previous patch's) /
after: `libdeflate.h` `d80f77f769214c1db4a211ff173942950780524611c1f5d1bc29e60fba246df1` / `a099418cac18c226cfdff5986cd8e320458615c88e1e4b801099b67fbbcf890b`, `lib/deflate_compress.c` `2258840b6ff5bef85cb8536dad683b8e6e774d6752026bf0ea46e761c20840e0` / `d64254930dc08ad20709371565eebbeabf0c010fa629fa115b73bb6b7317fecf`.

## chaiNNer-C patch: libdeflate_find_matches' allocation failure told apart (2026-11-16)

`libdeflate_find_matches()` returns SIZE_MAX, no longer 0, when its own match finder cannot be allocated, so that a
caller can tell a failure from lists that need more room. SHA-256 before (the previous patch's) / after:
`libdeflate.h` `a099418cac18c226cfdff5986cd8e320458615c88e1e4b801099b67fbbcf890b` / `09f3b05f11c1a4f4751bd38e32b74b4a357e7469b5535b6c579c936cbf80f39d`, `lib/deflate_compress.c` `d64254930dc08ad20709371565eebbeabf0c010fa629fa115b73bb6b7317fecf` / `d5f28b78e7f9f6d691e6de9cd31a8cdabdad429b054d9eca47a7818cfb7add5a`.

## chaiNNer-C patch: libdeflate_plan_blocks() allocates nothing (2026-11-16)

`libdeflate_plan_blocks()` writes no block, so it now gives `deflate_compress_ending()` a 64-byte buffer on its stack
for the stream's ending instead of allocating one and a half times the input; it can no longer fail for memory, and
its 0 means only an input too short to plan or plans and cache too small. SHA-256 before (the previous patch's) /
after: `libdeflate.h` `09f3b05f11c1a4f4751bd38e32b74b4a357e7469b5535b6c579c936cbf80f39d` / `09f3b05f11c1a4f4751bd38e32b74b4a357e7469b5535b6c579c936cbf80f39d`, `lib/deflate_compress.c` `d5f28b78e7f9f6d691e6de9cd31a8cdabdad429b054d9eca47a7818cfb7add5a` / `ed7dcc0d6757283f94e454238c8cd2d8daa004b375413966ce7f5191350ef0fd`.

## chaiNNer-C patch: the branch-free AVX2 kernel inside its guard (2026-11-16)

`min_cost_short_avx2()` moves inside the `#if defined(ARCH_X86_64) && (defined(__GNUC__) || defined(__clang__))` block
that includes `<immintrin.h>` and defines `MIN_COST_AVX2`, as its only caller already was, so that a build where the
guard is false compiles the scalar path alone. SHA-256 before (the previous patch's) / after: `libdeflate.h` `09f3b05f11c1a4f4751bd38e32b74b4a357e7469b5535b6c579c936cbf80f39d`
/ HEADER_AFTER, `lib/deflate_compress.c` `ed7dcc0d6757283f94e454238c8cd2d8daa004b375413966ce7f5191350ef0fd` / `5b1d3c51326372cc25502b75fd8a1baf55c409f35ec3824f6458f4561821ede2`.

## chaiNNer-C patch: the window's base position, and the refine's noise range (2026-11-16)

The `window` callback of `struct libdeflate_matches` gains `base`: it hands the arrays of the window's own positions
and their first position, and `deflate_compress_near_optimal()` indexes them from it, so that no caller has to form a
pointer outside its arrays. `libdeflate_refine_block()` counts a `noise` above 1000 as 1000 (an unsigned `1000 -
noise` wrapped before). SHA-256 before (the previous patch's) / after: `libdeflate.h` `09f3b05f11c1a4f4751bd38e32b74b4a357e7469b5535b6c579c936cbf80f39d` / `c10d516934cdb39a0909b47f6b1ae8c9176502f90f460e8ab6a1823b59326109`,
`lib/deflate_compress.c` `5b1d3c51326372cc25502b75fd8a1baf55c409f35ec3824f6458f4561821ede2` / `510ed6b6dbf3861e5af6cd1b3563f972211c936b0e675c3a689c94b97f006e2c`.
