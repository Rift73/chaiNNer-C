# Zstandard 1.5.7

Upstream: https://github.com/facebook/zstd, release tarball
`https://github.com/facebook/zstd/releases/download/v1.5.7/zstd-1.5.7.tar.gz`, SHA-256
`eb33e51f49a15e023950cd7825ca74a4a2b43db8354825ac24fc1b7ee09e6fa3` (matches the published `.sha256`), verified
against its signature (`.sig`, key 4EF4 AC63 455F C9F4 545D 9B7D EF8F E995 28B5 2FFD, "Zstandard Release Signing
Key <signing@zstd.net>").

`lib/` holds unmodified copies of the tarball's `lib/common`, `lib/compress` and `lib/decompress` (without
`huf_decompress_amd64.S`) and `zstd.h`, `zstd_errors.h`, `zdict.h`; `LICENSE` is the BSD licence (zstd is dual
BSD/GPLv2; the BSD terms are used). OpenEXR 3.5.2 bundles the same files (a byte-identical subset) and compiles the
same 26 units, with `ZSTD_DISABLE_ASM` on its MSVC path; `native/cmake/codecs.cmake` builds them as `chainner_zstd`
for OpenEXRCore's ZSTD compression.
