# libwebp 1.6.0

Upstream: https://chromium.googlesource.com/webm/libwebp, release tarball
`https://storage.googleapis.com/downloads.webmproject.org/releases/webp/libwebp-1.6.0.tar.gz`, SHA-256
`e4ab7009bf0629fd11982d4c2aa83964cf244cffba7347ecd39019a9e38c4564`, verified against the WebP release signature
(`.asc`, key 6B0E 6B70 976D E303 EDF2 F601 F9C3 D6BD B823 2B5D, "WebP release signing key").

The files here are unmodified copies of the tarball's: `sharpyuv/`, `src/dec`, `src/demux`, `src/dsp`, `src/enc`,
`src/mux` (without the animation encoder `anim_encode.c`/`animi.h`), `src/utils` and `src/webp`, keeping only the
x86 and generic units (the NEON, MIPS and MSA units are empty on x86-64); `COPYING` (BSD-3-Clause), `PATENTS`,
`AUTHORS` and `README.md`. OpenCV 5.0.0's `3rdparty/libwebp` is the same release (its one patch adds MSVC <= 1900
intrinsic shims) and Pillow 12.3.0 links the same release.

Built by `native/cmake/codecs.cmake` (`chainner_webp`) as OpenCV and Pillow build it: every unit, libwebp's runtime
CPU dispatch (the SSE4.1 and AVX2 units compiled for their level and run after libwebp's CPUID check) and
`WEBP_USE_THREAD`.
