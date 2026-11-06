# Fuzzing chaiNNer-C's readers of outside input

libFuzzer harnesses with AddressSanitizer, built by the project's own toolchain (clang-cl 23.1.2, lld-link) with
`CHAINNER_C_FUZZ=ON`. They are local tools, never part of a release, and the build never touches `backend/src`. The
first-party code they run (the codecs, `image_decode.cpp`, the regex engine) also carries UndefinedBehaviorSanitizer's
checks (signed overflow, shifts, float-to-integer overflow, array bounds and the like; `CMakeLists.txt` here), which stop
at their first report.

| Harness | What it runs | Checks beyond AddressSanitizer |
| --- | --- | --- |
| `fuzz_image_decode` | `image_decode.cpp`'s three routes over each input: OpenCV's by signature (BMP, JPEG, WebP, PxM, TIFF, PNG, OpenEXR), Pillow's JPEG, Pillow's straight-alpha TIFF | each route twice, into buffers filled with 0x00 and 0xFF: a sample left unwritten, or a route that succeeds once and declines once, aborts; an allocation of an impossible shape or above `CN_MAX_IMAGE_PIXELS` aborts |
| `fuzz_png_inflate` | the PNG decoder's inflate paths; the first byte picks a zlib stream for `cn_png_inflate_parallel` (even) or a file for `cn_png_decode` (odd) | a stream's Adler-32 and a file's CRCs are made right first, or a damaged stream would never reach an inflate; where the parallel path says yes, serial libdeflate must accept the stream at that size and give the same bytes, and a size off by one or a damaged Adler-32 must be declined; a decoded file must equal libpng's, except for the codes zlib refuses and libdeflate reads, which must then inflate serially |
| `fuzz_regex` | the regex engine: an input is a pattern, a 0xFF byte, then a text (both UTF-8) | every engine setting of `chn_regex_options` runs regex-py's operations (a search at every byte position, both iterators, both splits, the PosTranslator), compared as `chainner_ext_conformance.py regex-probe` compares them with the real module: default (prefilter on or off) equals production everywhere and dfa at character boundaries; nfa and backtrack equal each other everywhere, and production at character boundaries where the pattern has no look-around (`nfa_boundary_exclusions.json`'s mechanisms) |

## Build

The ASan runtime is a DLL next to clang's libraries, so the build directory is its own and its outputs stay outside
`backend/src` (configuration refuses otherwise):

```powershell
$ninja = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe'
cmake -S native -B native\build-fuzz -G Ninja "-DCMAKE_MAKE_PROGRAM=$ninja" -DCMAKE_BUILD_TYPE=Release `
    "-DCMAKE_TOOLCHAIN_FILE=$PWD\native\toolchain.cmake" -DCHAINNER_C_FUZZ=ON "-DCHAINNER_C_OUTPUT_DIR=$PWD\native\build-fuzz\out"
cmake --build native\build-fuzz --target fuzz_image_decode fuzz_png_inflate fuzz_regex -- -j 8
```

`CHAINNER_C_FUZZ` instruments every target (no ThinLTO, `/Zi`), so build only the harnesses (and `regex_probe`,
whose ASan copy runs `native/tools/chainner_ext_conformance.py regex-probe --probe <exe>`).

## Run

```powershell
$env:PATH = "C:\Executables\clang+llvm-23.1.2-x86_64-pc-windows-msvc\lib\clang\23\lib\windows;$env:PATH"
$env:ASAN_OPTIONS = 'quarantine_size_mb=32:detect_leaks=0'
native\build-fuzz\out\fuzz_image_decode.exe -rss_limit_mb=1024 -malloc_limit_mb=256 -max_total_time=900 `
    -timeout=20 -max_len=65536 <new-corpus-dir> <seed-dirs...>
native\build-fuzz\out\fuzz_png_inflate.exe -rss_limit_mb=1500 -malloc_limit_mb=512 -max_total_time=900 -timeout=30 `
    -max_len=65536 <new-corpus-dir> <seed-dirs...>
native\build-fuzz\out\fuzz_regex.exe -rss_limit_mb=1024 -malloc_limit_mb=1024 -max_total_time=900 -max_len=1024 <dirs>
```

Seeds: small files of each format (the encoders of the runtime's OpenCV and Pillow, `native/tests/exr/*.exr`), and
`pattern 0xFF text` pairs from `native/tests/chainner_ext/regex_corpus.jsonl`. A crash leaves its input as
`crash-<sha1>` (or the `-artifact_prefix` given); rerun the harness with that file as its only argument to reproduce.
The fuzz build lowers the parallel PNG inflate's size threshold (`src/codecs/png_inflate.c`) to 12 KiB of compressed data
per stretch, and only a seed of 48 KiB to 64 KiB, compressed 3:1 to 5:1 in small dynamic blocks, reaches it within
`-max_len`: for a level-6 stream, `zlib.compressobj(6, zlib.DEFLATED, 15, 3)` gives those blocks.
`fuzz_png_inflate`'s seeds start with a byte that picks the mode: 0 and such streams of 24 KiB to 64 KiB, or 1 and PNG
files of every direct format, in one IDAT, in small ones, and as independent pieces (a full flush at each row band's end).
Four logical CPUs per process (`start /affinity F`, or `taskset`) leave a stream of that size four stretches of 32 KiB of
output each, which more threads would cut too short to chain.
The harness decodes images up to 16 MiB; a larger one declines, as a failed allocation does in the application.
A header within the 1.6-gigapixel limit still makes libjpeg (progressive coefficients) or libtiff (a strip or tile
buffer) allocate whole-image buffers before that cap applies, as OpenCV and Pillow do; `-malloc_limit_mb=256` ends
such a unit at its first large allocation instead of letting the process grow (libFuzzer then reports it as
out-of-memory; it is not a defect).

## Sanitizer runs of the test suite (Linux)

`CHAINNER_C_SANITIZE=ON` builds every native module with AddressSanitizer, without ThinLTO, and chaiNNer-C's own targets
with UndefinedBehaviorSanitizer (the checks above plus alignment). It covers what the harnesses do not: the kernels, video,
the ONNX and NCNN graphs and the executor. A checkout's own binaries stay untouched; the outputs go outside `backend/src`
and a dedicated clone links them in:

```bash
native/build_linux.sh -B "$B" -j 8 -D CHAINNER_C_PYTHON_ROOT="$RUNTIME" -D CHAINNER_C_NASM="$NASM" -D CHAINNER_C_SANITIZE=ON     -D CHAINNER_C_OUTPUT_DIR="$B/out/nodes/impl" -D CHAINNER_C_EXT_OUTPUT_DIR="$B/out/chainner_ext"
ln -sf "$B"/out/nodes/impl/*.so backend/src/nodes/impl/
ln -sf "$B"/out/chainner_ext/chainner_ext.so backend/src/chainner_ext/
ln -sf "$B"/out/nodes/impl/texconv/texconv backend/src/texconv/
```

With `CHAINNER_C_TENSORRT_ROOT` and `CHAINNER_C_CUDA_ROOT` the build also holds the TensorRT runner, whose planner, frame checks and
CPU reference run without a GPU (`backend/tests/test_tensorrt_runner.py`); link `out/nodes/impl/tensorrt/*.so` likewise.

The interpreter is not instrumented, so the runtime is preloaded, and the variables below make NumPy's buffers and Python's
objects the sanitizer's own heap blocks:

```bash
export LD_PRELOAD="$LLVM/lib/clang/23/lib/x86_64-unknown-linux-gnu/libclang_rt.asan.so" PYTHONMALLOC=malloc CHAINNER_C_NUMPY_POOL=0
export ASAN_OPTIONS=halt_on_error=0:detect_leaks=0:detect_odr_violation=0:allocator_may_return_null=1:hard_rss_limit_mb=5800:quarantine_size_mb=64:log_path=$LOGS/san
export UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=0:log_path=$LOGS/san
python -B -m pytest -q -p no:cacheprovider native/tests/test_x.py   # one process per file; two at a time; CHAINNER_C_ISA for each level
```

A report is in `$LOGS/san.<pid>`; one location reports once per process unless `suppress_equal_pcs=0` is added. Under AddressSanitizer the
scratch pool poisons an idle slot and a lease's tail, so an overrun of a reused buffer reports as `use-after-poison`. A test that reads the pool's or the build's own state (`test_numpy_pool*.py`,
`test_native_versions.py`) or depends on floating-point flags can fail in this build without a report; only the reports count.
