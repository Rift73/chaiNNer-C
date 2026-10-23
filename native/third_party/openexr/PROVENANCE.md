# OpenEXR 3.5.2 (OpenEXRCore)

Upstream: https://openexr.com/, release tarball
`https://github.com/AcademySoftwareFoundation/openexr/releases/download/v3.5.2/openexr-3.5.2.tar.gz` (2026-10-03),
SHA-256 `e5750af57bbed556b462cbf0c39550d9063b2028c0184ec65fe42fdae458cdea` (the digest GitHub publishes for the
asset); its sigstore bundle `openexr-3.5.2.tar.gz.sigstore.json` verifies with cosign 3.1.3 (`verify-blob`, OIDC
issuer `https://token.actions.githubusercontent.com`, identity
`https://github.com/AcademySoftwareFoundation/openexr/.github/workflows/release-sign.yml@refs/tags/v3.5.2`).

`OpenEXRCore/` holds the tarball's `src/lib/OpenEXRCore` sources and headers (every unit its CMakeLists.txt
compiles, including the HTJ2K units `internal_ht.cpp` and `internal_ht_common.cpp`), with `LICENSE.md` (BSD-3-Clause)
and `PATENTS` (DreamWorks' grant for the DWA files). Not copied: OpenEXR's C++ library, the tools, and the tarball's
`external/` copies of its dependencies, which are vendored from their own releases instead:
`../libdeflate` (1.26; OpenEXR's copy is 1.26 with a sed patch for symbol visibility and an `ALIGN` rename, needed
only when it is compiled inside OpenEXRCore), `../zstd` (1.5.7; OpenEXR's copy is a byte-identical subset),
`../openjph` (0.32.0; OpenEXR's `src/core` is byte-identical) and `../imath` (3.2.3, the version OpenEXR's CMake
fetches).

One line differs from upstream. `OpenEXRCore/internal_zip.c` line 23 enables its SSE4.1 predictor for any
MSVC-compatible x64 compiler without a CPU check; clang-cl rejects those intrinsics without the target feature, and
the shipped build must run on every x86-64 CPU, so the guard gains `!defined(__clang__)`: clang-cl takes upstream's
portable scalar loop unless `__SSE4_1__` is defined (the `CHAINNER_C_MARCH` builds). Decoded bytes are the
same either way. Upstream SHA-256 `2caf5ac8adc83a0c328391259b21ffc648e84b058bf24e43d49597b55280cb1f`, patched
`fd009eac1ebaf7981d95c5601d17b43b454053adc0a757f8e75ab547a7e29e78` (decision, 2026-10-19).

`chainner/` holds the headers OpenEXR's CMake writes from `cmake/*.h.in` for a static, non-embedded Windows x64
library built with clang-cl: `OpenEXRConfig.h` (3.5.2, library version 34.3.5.2, namespaces `Imf_3_5`/`Imf`, Imath
3.2.3 / OpenJPH 0.32.0 / zstd 1.5.7, no large stack, no API visibility attributes), `IlmThreadConfig.h` (threading
on, `IlmThread_3_5`) and `OpenEXRConfigInternal.h` (external libdeflate and zstd, complete iomanip, and
`OPENEXR_IMF_HAVE_GCC_INLINE_ASM_AVX`, which OpenEXR's probe sets for clang-cl 23.1.2: it enables `_xgetbv` in the
runtime AVX/F16C check; the DWA AVX kernel itself needs an LP64 target and stays off on Windows).

Built by `native/cmake/codecs.cmake` as `chainner_openexr` and called only by `native/src/codecs/exr_codec.c`
(Load Image's `.exr`, by signature as OpenCV's imdecode). The fixtures of `native/tests/test_exr_decode.py` come from
this library's writer (`native/tests/exr/gen_exr_fixtures.c`).
