# Imath 3.2.3 (half.h only)

Upstream: https://github.com/AcademySoftwareFoundation/Imath, release tarball
`https://github.com/AcademySoftwareFoundation/Imath/releases/download/v3.2.3/Imath-3.2.3.tar.gz` (2026-08-20),
SHA-256 `72f498e7073890a2a1b870e453908a07c9725f9d1384527043a38578bfb4ef31`; its sigstore bundle verifies with
cosign 3.1.3 (identity
`https://github.com/AcademySoftwareFoundation/Imath/.github/workflows/release-sign.yml@refs/tags/v3.2.3`). This is
the version OpenEXR 3.5.2's CMake fetches when Imath is not installed.

`Imath/` holds unmodified `src/Imath/half.h` and the headers it includes (`ImathExport.h`, `ImathNamespace.h`,
`ImathPlatform.h`), with `LICENSE.md` (BSD-3-Clause). OpenEXRCore uses `half.h` for its half/float conversions and
the HTJ2K units' `half` type; nothing of Imath's library is compiled.

`chainner/Imath/ImathConfig.h` is `config/ImathConfig.h.in` as Imath's CMake writes it (3.2.3, library version
30.3.2.3, namespaces `Imath_3_2`/`Imath`, noexcept on) with `IMATH_HALF_USE_LOOKUP_TABLE` off, so `half.h` converts
with arithmetic and needs no table from Imath's library (the conversions are exact either way).
