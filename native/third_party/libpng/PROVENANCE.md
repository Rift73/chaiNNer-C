# libpng 1.6.57

Upstream: https://www.libpng.org/pub/png/libpng.html, tarball
`https://download.sourceforge.net/libpng/libpng-1.6.57.tar.xz`, SHA-256
`d10c20d7171569804cae8dfc13ba6dcd0662c41ed39d43d4d429314aafb10a80`; its sources are identical to the GitHub tag
archive `https://github.com/pnggroup/libpng/archive/refs/tags/v1.6.57.tar.gz` (SHA-256
`4cbb7b0746edc1683c9581b373365e955133b7f1243f171b7d1535b4415dfedb`) and to OpenCV 5.0.0's `3rdparty/libpng`.

The files here are unmodified copies of the tarball's library sources and headers, `intel/` (the SSE2 filters),
`LICENSE` and `README`.

`chainner/pnglibconf.h` is OpenCV 5.0.0's `3rdparty/libpng/pnglibconf.h` (libpng 1.6.45's
`scripts/pnglibconf.h.prebuilt`, SHA-256 `f2878e12e63d929ddc3e9eff4829093d2d38797f1d142b604359dd8e8cc63c98`), the
configuration OpenCV builds 1.6.57 with: it lacks 1.6.57's cLLI and mDCV chunk support, so those chunks stay
unknown chunks as in OpenCV. Built with `PNG_INTEL_SSE_OPT=1`, as OpenCV's x86 build.
