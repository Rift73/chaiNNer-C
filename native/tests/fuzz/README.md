# Fuzzing chaiNNer-C's readers of outside input

libFuzzer harnesses with AddressSanitizer, built by the project's own toolchain (clang-cl 23.1.2, lld-link) with
`CHAINNER_C_FUZZ=ON`. They are local tools, never part of a release, and the build never touches `backend/src`. The
first-party code they run (the codecs, `image_decode.cpp`, the regex engine) also carries UndefinedBehaviorSanitizer's
checks (signed overflow, shifts, float-to-integer overflow, array bounds and the like; `CMakeLists.txt` here), which stop
at their first report.

| Harness | What it runs | Checks beyond AddressSanitizer |
| --- | --- | --- |
| `fuzz_image_decode` | `image_decode.cpp`'s three routes over each input: OpenCV's by signature (BMP, JPEG, WebP, PxM, TIFF, PNG, OpenEXR), Pillow's JPEG, Pillow's straight-alpha TIFF | each route twice, into buffers filled with 0x00 and 0xFF: a sample left unwritten, or a route that succeeds once and declines once, aborts; an allocation of an impossible shape or above `CN_MAX_IMAGE_PIXELS` aborts |
| `fuzz_regex` | the regex engine: an input is a pattern, a 0xFF byte, then a text (both UTF-8) | every engine setting of `chn_regex_options` runs regex-py's operations (a search at every byte position, both iterators, both splits, the PosTranslator), compared as `chainner_ext_conformance.py regex-probe` compares them with the real module: default (prefilter on or off) equals production everywhere and dfa at character boundaries; nfa and backtrack equal each other everywhere, and production at character boundaries where the pattern has no look-around (`nfa_boundary_exclusions.json`'s mechanisms) |

## Build

The ASan runtime is a DLL next to clang's libraries, so the build directory is its own and its outputs stay outside
`backend/src` (configuration refuses otherwise):

```powershell
$ninja = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe'
cmake -S native -B native\build-fuzz -G Ninja "-DCMAKE_MAKE_PROGRAM=$ninja" -DCMAKE_BUILD_TYPE=Release `
    "-DCMAKE_TOOLCHAIN_FILE=$PWD\native\toolchain.cmake" -DCHAINNER_C_FUZZ=ON "-DCHAINNER_C_OUTPUT_DIR=$PWD\native\build-fuzz\out"
cmake --build native\build-fuzz --target fuzz_image_decode fuzz_regex -- -j 8
```

`CHAINNER_C_FUZZ` instruments every target (no ThinLTO, `/Zi`), so build only the harnesses (and `regex_probe`,
whose ASan copy runs `native/tools/chainner_ext_conformance.py regex-probe --probe <exe>`).

## Run

```powershell
$env:PATH = "C:\Executables\clang+llvm-23.1.2-x86_64-pc-windows-msvc\lib\clang\23\lib\windows;$env:PATH"
$env:ASAN_OPTIONS = 'quarantine_size_mb=32:detect_leaks=0'
native\build-fuzz\out\fuzz_image_decode.exe -rss_limit_mb=1024 -malloc_limit_mb=256 -max_total_time=900 `
    -timeout=20 -max_len=65536 <new-corpus-dir> <seed-dirs...>
native\build-fuzz\out\fuzz_regex.exe -rss_limit_mb=1024 -malloc_limit_mb=1024 -max_total_time=900 -max_len=1024 <dirs>
```

Seeds: small files of each format (the encoders of the runtime's OpenCV and Pillow, `native/tests/exr/*.exr`), and
`pattern 0xFF text` pairs from `native/tests/chainner_ext/regex_corpus.jsonl`. A crash leaves its input as
`crash-<sha1>` (or the `-artifact_prefix` given); rerun the harness with that file as its only argument to reproduce.
The harness decodes images up to 16 MiB; a larger one declines, as a failed allocation does in the application.
A header within the 1.6-gigapixel limit still makes libjpeg (progressive coefficients) or libtiff (a strip or tile
buffer) allocate whole-image buffers before that cap applies, as OpenCV and Pillow do; `-malloc_limit_mb=256` ends
such a unit at its first large allocation instead of letting the process grow (libFuzzer then reports it as
out-of-memory; it is not a defect).
