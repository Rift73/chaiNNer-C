# libjpeg-turbo 3.1.4.1

Upstream: https://github.com/libjpeg-turbo/libjpeg-turbo, release tarball
`https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.1.4.1/libjpeg-turbo-3.1.4.1.tar.gz`,
SHA-256 `ecae8008e2cc9ade2f2c1bb9d5e6d4fb73e7c433866a056bd82980741571a022`, verified against the project's
signature (`.sig`, key 0338 C8D8 D9FD A62C F9C4 21BD 7EC2 DBB6 F4DB F434, "The libjpeg-turbo Project").

`src/`, `simd/`, `LICENSE.md` and `README.ijg` are unmodified copies of the tarball's files that the static library
needs (the library sources of upstream's `JPEG_SOURCES`, the arithmetic codec, the x86-64 NASM SIMD and
`simd/x86_64/jsimd.c`, and every header they include). `LICENSE.md` (Modified BSD and zlib licences) and
`README.ijg` (the IJG licence) hold the licence terms.

`chainner/` is chaiNNer-C's: `jconfig.h`, `jconfigint.h` and `jversion.h` are upstream's `src/*.h.in` templates with
the values upstream's CMake writes for Pillow 12.3.0's Windows build (`WITH_JPEG8=TRUE`, static, SIMD): 
JPEG_LIB_VERSION 80, arithmetic encoding and decoding, MEM_SRCDST, WITH_SIMD, INLINE `__forceinline`, THREAD_LOCAL
`__declspec(thread)`, SIZEOF_SIZE_T 8, HAVE_INTRIN_H, BUILD "chaiNNer-C" (upstream's build date; jcmaster.c stores it in
the compressor's `jpeg_version` string, which no code reads).

Built by `native/cmake/codecs.cmake` (`chainner_jpeg`) with the repository's compile options and the SIMD units
assembled by NASM 3.01 (`-fwin64 -DWIN64 -D__x86_64__ --reproducible`). Pillow 12.3.0 decodes `.jpg` with this
version and API; OpenCV 5.0.0 decodes `.jpe` and encodes JPEG with 3.1.2 (libjpeg v7 API), which gives the same
pixels and bytes for the corpus (`native/local/instant-exec/report-codecs.md`).
