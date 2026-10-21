# libtiff 4.7.1

Upstream: https://libtiff.gitlab.io/libtiff/, tarball `https://download.osgeo.org/libtiff/tiff-4.7.1.tar.gz`
(RELEASE-DATE 20250911), SHA-256 `f698d94f3103da8ca7438d84e0344e453fe0ba3b7486e04c5bf7a9a3fabe9b69`; its detached
signature `tiff-4.7.1.tar.gz.sig` verifies with Even Rouault's key `B1FA 7D81 EEB8 E663 9917 8B97 33EB BFC4 7B3D
D87D`. Every library source in the tarball is identical to OpenCV 5.0.0's `3rdparty/libtiff` (the git blob hashes
of all 62 files there match; only OpenCV's own `CMakeLists.txt` differs).

The files here are unmodified copies of the tarball's `libtiff/` sources and headers that OpenCV's build compiles
for its codec set (core, Win32 I/O, LZW, ZIP, PackBits, CCITT/fax3, ThunderScan, NeXT, LogLuv, JPEG), and
`LICENSE.md`. Left out: `tif_ojpeg.c`, `tif_lzma.c`, `tif_jbig.c`, `tif_zstd.c`, `tif_webp.c`, `tif_pixarlog.c`
and `tif_jpeg_12.c` (OpenCV compiles them to empty units with its configuration), `tif_lerc.c` (OpenCV's list
leaves it out),
`tif_stream.cxx` (the C++ stream API), `tif_unix.c` and the tools.

`chainner/` holds the headers OpenCV 5.0.0's `3rdparty/libtiff/CMakeLists.txt` generates for its Windows x64 build
(`opencv-python` 5.0.0.93 reports "TIFF: build (ver 42 - 4.7.1)"):
- `tif_config.h` and `tiffconf.h` from the tarball's `*.cmake.in` with OpenCV's values: CCITT, PackBits, LZW,
  ThunderScan, NeXT, LogLuv, MDI and ZIP (zlib) on; JPEG on (OpenCV's libjpeg-turbo; here `chainner_jpeg`) with
  `JPEG12_FOUND FALSE`, so neither `JPEG_DUAL_MODE_8_12` nor `HAVE_JPEGTURBO_DUAL_MODE_8_12` (12-bit JPEG-in-TIFF
  fails as in OpenCV); OJPEG, JBIG, LZMA, LERC, PixarLog, WebP, ZSTD and libdeflate off; `STRIPCHOP_DEFAULT` unset
  (no strip chopping), `STRIP_SIZE_DEFAULT 8192`; `SUBIFD_SUPPORT`, `DEFAULT_EXTRASAMPLE_AS_ALPHA` and
  `CHECK_JPEG_YCBCR_SUBSAMPLING` 1; `USE_WIN32_FILEIO`; `TIFF_MAX_DIR_COUNT 1048576`; MSVC's header and type checks
  (`HAVE_ASSERT_H`, `HAVE_FCNTL_H`, `HAVE_IO_H`, `HAVE_SYS_TYPES_H`, `HAVE_SETMODE`, `SIZEOF_SIZE_T 8`, 64-bit
  `long long` types, little-endian); OpenCV sets no `PACKAGE_*` values, so they are empty.
- `tiffvers.h`: the tarball's `libtiff/tiffvers.h`, which `tiffvers.h.cmake.in` reproduces for version 4.7.1.

Built as the static target `chainner_tiff` (native/cmake/codecs.cmake) against `chainner_zlib` (zlib 1.3.2) and
`chainner_jpeg` (libjpeg-turbo 3.1.4.1, 8-bit only for libtiff).
