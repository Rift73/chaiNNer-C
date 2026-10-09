# chaiNNer-C architecture

Reference for the single-backend tree on branch `chaiNNer-C`; reasoning in [`DESIGN-DECISIONS.md`](DESIGN-DECISIONS.md),
checklists in [`MAINTAINING.md`](MAINTAINING.md). Commands: `README.md`. State, pending work and decisions: `STATUS.md`.
Pre-SP1 history is readable at tag `chainner-c-checkpoint-20261001`.

## 1. Layers

```text
Installed Electron 25 / React UI (0.25.1-nightly.2025-10-21) + pinned bundle patches
  -> owned Python/Sanic backend (backend/src): node registry, asyncio executor, HTTP + SSE
     -> Python node contracts: schemas, validation, model objects
        -> chainner_native.dll  C17 kernels, ctypes C ABI v2 (GIL released during calls)
        -> _chainner_graph.pyd  C++20/pybind11: ONNX/NCNN graphs, tiling, codecs, video, utilities
        -> chainner_ext.pyd     C, limited API: chainner_ext 0.3.10's drop-in, with the regex engine
           -> retained engines: OpenCV/IPP, NumPy BLAS/LAPACK, FFmpeg, ONNX Runtime, NCNN,
              PyTorch, FreeType/RAQM
```

| Concern | Files |
| --- | --- |
| Native build | `native/CMakeLists.txt`, `native/toolchain.cmake`, `native/Build.ps1` |
| Runtime ISA dispatch (scalar, AVX2 or AVX-512 per kernel; no CPU check beyond NumPy's x86-64-v2 baseline) | `native/src/isa.c` |
| C ABI and kernels | `native/include/chainner.h`, `native/src/*.c`; bridges `backend/src/nodes/impl/native*.py` |
| C++ modules | `native/src/graph_module.cpp` (pybind11 entry), `native/src/*.cpp`, `native/include/*.hpp` |
| Executor | `backend/src/process.py` (the only executor), `backend/src/server.py` (worker app) |
| Scheduling and lifetime | `backend/src/nodes/impl/execution_scheduler.py`, `execution_cleanup.py`, `item_window.py`, `backend/src/api/lazy.py` |
| Native CPU pool | `native/src/parallel.c`, `native/include/parallel.h` |
| Mirrors' CRT math from the process's `ucrtbase.dll` (Consult 6 D-4; groups in the header) | `native/include/cn_crt_math.h`, `native/src/crt_math.c` |
| Ownership, cache, events | `backend/src/nodes/impl/native_buffers.py`, `nodes/properties/outputs/numpy_outputs.py`, `chain/cache.py`, `nodes/impl/event_queue.py`, `nodes/impl/numpy_pool.py` (section 6) |
| Resize and PNG | `backend/src/nodes/impl/native_resample.py`, `native/src/resample_filters.c`, `native/src/image_io.cpp`, `native/third_party/fpng` |
| Video | `native/src/video_io.cpp`, `backend/src/nodes/impl/video.py` |
| Packaging | `native/tools/package_port.py`, `package_files.py`, `package_manifest.py` |
| UI patches | `native/tools/independent_ui.py`, `native/tools/input-drop.json`, `native/tools/tensorrt-types.json`, `native/tools/tensorrt-clear.json`, `verify_*.cjs` |
| Benchmark and profile | `native/tools/bench*.py`, `native/tools/bench-data/`; native-call timer `backend/src/nodes/impl/native_profile.py` (`CHAINNER_C_PROFILE`, README Bench) |
| Runtime verification | `native/tools/verify_*runtime.py`, `verify_resize_pipeline.py`, `clipboard_isolation.py` |

## 2. Shipped baseline and frontend

- `backend/src` is B0 plus SP2's changes (`nodes/impl/execution_scheduler.py`), SP3's (`api/iter.py`, `process.py`,
  `server.py`; in `nodes/impl/`: `execution_scheduler.py`, `item_window.py` (new), `native_buffers.py`; in
  `packages/chaiNNer_standard/image/`: `batch_processing/split_spritesheet.py`, `io/save_image.py`,
  `video_frames/save_video.py`), SP3b's (`api/lazy.py`, `process.py`, `server.py`; in `nodes/impl/`: `item_window.py`,
  `native_buffers.py`, `native_profile.py`; the Load Image node), SP4a's (`server.py`; in `nodes/impl/`: `native.py`,
  `native_graph.py`, `native_profile.py` (new)) and SP4b's (in `nodes/impl/`: `image_utils.py`, `native*.py`; the
  Gaussian, Box, Lens Blur, Dilate and Erode nodes) and SP4c's (`server.py`; in `nodes/impl/`: `numpy_pool.py` (new),
  `ffmpeg.py`, `native_profile.py`; `packages/chaiNNer_external/web_ui.py`), plus the rebuilt binaries, less the
  nightly's GPU lease (below): 425 files, 11.3 MiB. B0 = the installed nightly backend (upstream `d56e507f` plus the
  nightly's local delta: 4 added files `gpu_lease.py`, `GPU_LEASE.md`, `tests/test_gpu_lease.py`,
  `tests/check_gpu_lease_wsl.py`, and 10 changed files including `server.py`, `server_host.py`, `process.py`,
  `events.py`, `chain/cache.py`) plus the port's 216 files (156 replaced, 60 added). The packager ships it as-is: no
  overlays, delegates, transplants or AST pins.
- Line endings (root `.gitattributes`): every text file is LF in the index and in every checkout (`* text=auto
  eol=lf`). `backend/src/** -text` stores and checks out the shipped bytes exactly; they are LF, as upstream's GitHub
  source is (the installed app's CRLF is its Windows build's; Consult 10 R-o, lane E). `native/tests/reference_*/**`,
  `native/tests/oracle/**` and `native/tests/chainner_ext/**` are `-text` too: frozen copies and corpora keep their
  bytes (CRLF, LF and mixed alike).
- The two binaries are git-ignored (`.gitignore` names the DLL; `*.py[cod]` covers the `.pyd`): build or copy them.
- The nightly's GPU lease was removed on 2026-10-05 (owner's decision): its 4 files are gone and every spot it hooked is
  upstream again. The delta's TF32, ONNX export size 32, `gc.collect()` removals and 8-worker pool stay (owner's choice).
- UI: the package ships the installed Electron 25 bundles with pinned patches (section 9). `src/`
  is upstream `d56e507f` (Electron `^25.8.4`) with the two UI patches ported to the TypeScript, its integrated Python
  moved to CPython 3.14.8 (kept at 3.14.8 or newer) and its own root `%APPDATA%\chaiNNer-C`: the run-from-source UI (`npm start`), never part of the package. Do not
  substitute newer frontend, preload or main assets into the installed runtime.
- The package owns its backend like normal chaiNNer. A `--remote-backend` launch changes ownership
  and restart/recovery behavior, so it is not used and is not proof of parity.

## 3. Native layer

- Sources: 136 files under `native/src` and `native/include` (66 `.c`, 24 `.cpp`, 34 `.h`,
  11 `.hpp`, 1 `.inc`; 51,362 physical lines including generated tables and attributed adaptations).
- `chainner_native.dll`: C17 image kernels behind a C ABI. `backend/src/nodes/impl/native.py`
  loads it with ctypes and requires `cn_abi_version() == 2`; a load or ABI failure raises, never
  selecting a Python implementation. `cn_status` codes map to `TypeError`/`ValueError`/
  `OverflowError`/`MemoryError`/`RuntimeError`. Bridges validate dtype, contiguity and alignment.
- Scratch pool (`scratch.c`): kernels that write every scratch byte before reading it (morphology, median cut, composite,
  resample's intermediate and linear buffers) lease one buffer from 4 slots, each kept up to 16 MiB when idle (64 MiB in
  all); a larger lease grows its slot and frees it on release. A lease never waits: a busy pool gives a temporary
  `malloc`. Resample's coefficients stay in its own slots.
- `_chainner_graph.pyd`: C++20/pybind11 application logic: ONNX/NCNN graph parsing, optimization, conversion and
  TensorProto work, model application and tiling control, scalar/random/text utilities, image I/O with fpng, execution
  helpers, file sequences, video I/O, clipboard. C++ exceptions are translated at the Python boundary;
  `native/include/graph_exception.hpp` preserves CPython's handled-exception state inside native catch blocks.
- `chainner_ext.pyd` (`backend/src/chainner_ext/`) is chaiNNer-C's own C module since 2026-10-06, replacing the Rust
  chainner_ext 0.3.10: its image API binds `chainner_native.dll`'s kernels, its regex engine (`native/src/regex/`,
  regex 1.8.4 and regex-syntax 0.7.2 in C) also serves `_chainner_graph.pyd` through the `_C_API` capsule, and Rust
  parity is held by the conformance manifest (`native/tests/chainner_ext/`, `test_chainner_ext_manifest.py`).
- Generated sources, never run in the application (README Build): `native/include/onnx_converter_passes.hpp`,
  `pixel_art_tables.inc`, `tiling_control.hpp`.
- Build (owner directive 2026-10-06; Consult 9): clang-cl, lld-link and llvm-lib of LLVM 23.1.2 only, under Ninja,
  with the Build Tools' MSVC 14.44 libraries and Windows SDK 10.0.26100 (`toolchain.cmake` pins them;
  `CMakeLists.txt` refuses any other compiler). Every unit is baseline x86-64 code (the ISA units compiled for their level only, fpng for SSE4.1
  and PCLMUL behind its own CPU check; `CHAINNER_C_MARCH` adds `-march` to local builds only) with ThinLTO, `/fp:strict`
  (the kernels' FP-flag windows mirror `np.seterr` and the pool carries MXCSR, so FP operations may not move across
  them), dynamic denormals (kernels run under DAZ/FTZ too), `/Brepro` (no timestamps; no PDB in Release) and
  `-fmacro-prefix-map=<native>=.` (clang's MSVC mangling names an anonymous namespace by a hash of the main file's
  path, which reaches RTTI names), so the same sources give the same binaries wherever they are checked out.
  `/W4 /WX` on every first-party target is the strict check. The binaries run on any
  x86-64-v2 CPU, NumPy 2.5.3's own baseline (owner correction 2026-10-07; no CPU check of ours). LLVM leaves a NaN's sign and payload unspecified, so
  where upstream's NaN bits are observable they are derived from the operands as integers (the normal output, the
  palette key), never read from the arithmetic. Only Windows x64 is validated and packaged.

## 4. Threading and scheduling

Four pools. Only the native one is budgeted; the engines keep their defaults (see below):

| Pool | Behavior |
| --- | --- |
| asyncio loop + `ThreadPoolExecutor(max_workers=8)` | `backend/src/server.py:48` (`POOL_SIZE`; nightly delta; upstream has 4). Runs node functions, generator advance/iterate, collector completion and broadcast-data computation |
| Native `cn_parallel_for(count, grain)` | Windows threadpool; capacity = CPUs in the process affinity mask. `count <= grain` runs inline. A larger call runs `min(1 + (count - 1) / grain, 128)` chunks, a partition that depends on `(count, grain)` alone. The caller works on its own call; generic helpers join the active call with the fewest helpers and re-pick when the set of calls changes; a helper is admitted only while callers plus helpers fit the capacity. Every chunk runs under the caller's MXCSR. The DLL is never unloaded |
| OpenCV internal pool | OpenCV defaults |
| BLAS pool | NumPy's OpenBLAS defaults |

Graph scheduling (`process.py` input gathering): consecutive eligible inputs are awaited together, up to 4 per
`owned_gather`. Eligible means an eager, non-lazy edge whose upstream subgraph is pure CPU (`pure_cpu_dependency`:
regular node, no side effects, no node context, schema `chainner:image:*` or `chainner:utility:*`, no lazy inputs,
recursively; cached nodes count). `NodeFlights` shares in-flight ancestors; completed values belong to the counted
cache. `owned_gather` drains every started branch, even on repeated cancellation, and raises the first error in input
order. A lone eligible input is awaited directly (`owned_gather` with one branch).

SP2 (2026-10-02) replaced the greedy helper reservation, which gave every helper to the first caller and left a
concurrent one alone on its thread, and a partition that depended on the CPU count and on how many helpers were free.
All 164 call sites were read: no result depends on the partition (152 elementwise, 7 exact reductions, 5 whose code
path, not result, depends on where a chunk ends); `test_parallel_affinity.py` guards that results do not depend on the
CPU count or on overlapping callers. Engine thread counts are in no budget: OpenCV's changes output (native Canny
policy, AREA resize zero signs) and BLAS `ddot` is suspected to. A helper that releases a call's last pin wakes the
caller only after releasing the registry lock: woken under it, the caller blocks on the lock and needs a second wake,
which showed up on video-ffv1 under ffmpeg's CPU load. SP4b kept the policy: two cascading-wake variants (a helper
that claims a chunk wakes one more) failed, because empty wakes are late arrivals onto short chunks, not a full pool;
the lever is chunk length (conversions 262,144 elements, lens convolution 16,384 outputs, morphology combine 65,536).

Item window (SP3, `nodes/impl/item_window.py`). The sequential loop in `__iterate_generator_nodes` is the only commit
path: side effects, events, cache and errors happen in item order as before. From item 1 on, an engine on the same
event loop keeps up to K - 1 later items in flight. One serialized producer advances the generators (Range inline; see
Two-phase producer). Iterated `pure_cpu_dependency` nodes of later items other than Directory Go Into (`into_directory`:
it resolves on-disk folder spelling) run ahead with fresh contexts, tested against the shared static cache plus that
item's store; a static node never runs ahead. `__process` replays a stored result, or awaits its in-flight job, in place
of its pool job inside the usual start/broadcast/finish; a stored error is raised there. A replay makes one empty pool
round trip, so it finishes no sooner than the pool job it replaces could, and its siblings' events interleave with it as
at K = 1 (pool timing; section 7). Save Image (conversion and in-memory encode; DDS runs
whole) and Save Video (uint8 conversion only) register a prepare phase that runs ahead and a commit phase that keeps
today's order; a failed prepare runs the node whole. K = clamp(cpus // 4, 6, 8) from the affinity mask; at most P = pool
size − 2 engine jobs plus the advance the loop waits for, oldest item first; speculation is admitted only while
in-flight bytes plus the per-item estimate fit 25% of available memory, and the item the loop waits for is always
produced. Read-ahead needs every generator to declare its per-item files (`Generator.source_paths`) and every writer's
base directory to resolve, with none of those files' parent folders under one; an unresolved directory blocks it like a
source under one: K = 1, today's path. Not detected: an iterated subdirectory or name containing `..`, a declared file
that is a symlink into an output directory, the loopback admin-share alias of a path, a hardlink, or a case-only
spelling difference on a case-insensitive volume outside Windows (macOS by default, FAT or exFAT). The developer-only
`CHAINNER_C_ITEM_WINDOW` forces K; the decision is logged once per group as `item window K=<k> (<reason>)`, and a
window's counts (items, ahead, jobs, materialized, replayed, awaited, discarded) when it closes.

Two-phase producer (SP3b). Load Images items are produced in two phases: a serial `describe()` names the file, and a
parallel `materialize(token)` (decode and normalize) runs as an engine job that shares P with node jobs, oldest slot
first; the slot the commit path waits for bypasses P, pause and Stop. A Load Images slot counts twice the per-item
estimate (2E) until its outcome resolves, and speculation needs in-flight bytes plus 2E to fit the same 25%. Every
single generator's normalize leaves the producer (video included); zipped groups and Range stay serial. The decoders
are unchanged (Pillow stays the JPEG decoder; its plugin registry is warmed once at Load Image's import); OpenCV's
temp-file decoders (`.hdr .pic .sr .ras .exr`, or their content signatures) take one lock.
`CHAINNER_C_SERIAL_PRODUCER=1` forces the serial producer (developer setting; any other value is ignored).

## 5. Invariants

- Normalized float32 image semantics; BGR/BGRA channel order; grayscale flattening, dtype promotion, signed-zero and
  nonfinite behavior where defined; strided, reversed, unaligned and read-only inputs; explicit output ownership.
- Bit-exact float32 against the installed chaiNNer, judged (section 7; the frozen B0 and B3 are regression proxies):
  SIMD only where bit-identical (elementwise, per-output lane); no FMA contraction, reordered reductions or fast-math
  (`/fp:strict`). ISA dispatch (SP4): engine-mirror parameters (`lanes`, `fused`) choose each output's operations, the
  level only the code that runs them: `scalar`, `avx2` or `avx512`, probed once in `isa.c` (`CHAINNER_C_ISA`
  caps it), in export-free units compiled for their level only: eleven `*_avx2.c` (conversions, convolution, lens, separable filters,
  morphology with its exceptional scan, median cut and palette counts, dither, blend, resample, the spectral multiply,
  the normal output) and the kept `convolution_avx512.c` and `separable_avx512.c` (16-lane groups; spec 4.3's keep rule:
  a timer win, no throughput loss; separable: kept on the 8-CPU timer by controller ruling), which `CHAINNER_C_ISA=avx2`
  caps on parts that downclock. The multiply (`spectral_avx2.c`) and the normal output (`color_avx2.c`) share their
  per-element code with the baseline unit through `spectral_shared.h` and `color_shared.h` (static helpers); the
  other SP4b Task 8 tail kernels are baseline row loops.
- Never mutate cached graph inputs; never turn a borrowed view writable. Bridges keep every owner
  alive through native calls and asynchronous writes. Keep no-op identities where required.
- C ABI version and status checks stay; DLL/ABI failure is visible.
- C++ runtime order: the first `msvcp140.dll` a process loads serves every module, and Pillow's
  bundled 14.29 breaks newer Torch and ONNX Runtime builds (WinError 1114). The worker is safe by
  one step: `packages/chaiNNer_pytorch/__init__.py` imports torch, which loads the system copy by
  name, after `chaiNNer_standard` (no Pillow, no runtime yet) and before `load_nodes` imports any
  node module. `test_runtime_dlls.py` pins both halves. The pytest process preloads the system copy
  (`conftest.py`); child interpreters must not import Pillow before a framework.
- Preserve arithmetic order for exact reductions and per-pixel work. Tolerances and corrections are
  only those in section 7; no blanket image tolerances.
- Seeded generation and repeated identical pure operations stay reproducible. Execution Number,
  counters, clipboard, file writes and model setup are stateful: respect their side effects.
- Shared in-flight work never duplicates side effects or lets cleanup free a resource a worker
  still uses. Drain every started branch on cancellation; stable input order chooses the error.
- Counted caches consume each actual edge and keep separate reservations for side-effect roots.
  Release intermediates promptly without dropping final or static generator outputs.
- Generator-once: each generator node is constructed once per run. The executor keeps the initial
  `GeneratorOutput` with its supplier for the whole sequence and the final restoration (per-item
  reconstruction of Load Images once caused 35 directory discoveries in a 32-image probe).
- Item order of side effects and events; final broadcasts; terminal progress and completion events.
- Video: persistent FFmpeg processes per stream; frame order and shape, partial writes, EOF and
  finalization, audio and codec settings preserved; graceful EOF, then bounded targeted
  termination; owned children reaped. No unconditional hard-shutdown bound is promised.
- fpng: used only for default-option 8-bit 3/4-channel PNG within its bounds (each dimension
  <= 65,535; filtered stream below `INT32_MAX - 1 MiB`). Grayscale, 16-bit, explicit codec options
  or larger images keep the OpenCV encoder. A selected fpng failure is an error, never retried.
- File traversal order, Unicode paths, overwrite/skip options, error continuation, final UI state
  and completion events are external contracts. So are schemas, names, controls and saved chains.

## 6. Ownership, caches and events

| Area | Behavior |
| --- | --- |
| Output ownership | Weak, bounded provenance lets proven normalized read-only/native outputs and immutable bytes be borrowed (`normalized_readonly`); unknown writable aliases are still normalized or copied |
| Counted cache | `OutputCache` reservations are consumed per actual edge, including the first computation; side-effect roots reserved separately; no full GC per expiry (the nightly delta also drops per-run `gc.collect()`) |
| NumPy pool (SP4c) | `native/src/numpy_pool.cpp` in `_chainner_graph` is a NumPy data-memory handler that keeps freed blocks of 512 KiB or more warm: a request takes the smallest idle block whose capacity is in [size, 2 × size] (best fit), else a fresh `VirtualAlloc` block; smaller requests go to NumPy's default handler. Idle bytes are capped at min(512 MiB, 10 % of RAM), oldest released first (`CHAINNER_C_NUMPY_POOL`, README Bench). NumPy keeps its handler per thread, so `numpy_pool.install` runs on the worker's main thread before `app.run` and as `initializer=` of every worker `ThreadPoolExecutor` (`server.py`, `ffmpeg.py`, `web_ui.py`). Idle blocks are released at the end of `/run`, after an individual run's broadcasts are sent (a deferred task) and on `/clear-cache/individual`, the last two only while `ctx.executor is None`, so a preview or cache clear mid-run never empties the run's warm set. Load Video's frames are read into fresh NumPy buffers, so they pass through the handler too (D8). Only addresses change. A failed `/run` (and a stopped one) schedules one full `gc.collect()` after its handler returns and then releases the pool if no run executes (`collect_failed_run`); successful runs never collect. `await_owned_node` and `NodeFlights.run` drop their `future`/`task` locals in `finally`, so an eagerly read failed input frees by refcount |
| Events | `LatestBroadcasts`: one active computation and one latest pending preview per node. `LatestEventQueue` coalesces routine queued states, keeping control/error events, final state, acknowledgments, cancellation ownership and sequence metadata. `/run/individual` responds without waiting for its broadcasts, as upstream |
| Generators, collectors | Blocking reads, writes and finalization run in owned workers; cheap Range/Accumulate paths stay inline |
| Video transport | No redundant contiguous-frame copies; an owning read-only view is held until complete and short writes finish |
| NCNN/ONNX | Repeated successful GC deferred to chain cleanup; direct-call, error and OOM cleanup and allocator handling preserved |
| Model helpers | Align Image to Reference and Upscale Face take chain-scoped exclusive leases (`backend/src/nodes/impl/pytorch/resource_cache.py`); incomplete checkpoints are not cached; explicit memory budgets disable retention. Checked only with mocks and CPU contract tests (`GPU-DEFERRED.md` item 5) |
| Preparation caches | Bounded parameter data only, never image sequences: font faces (`font_cache.py`, at most 16 idle faces, 8 MiB, 512 px), regex and replacement parsers (`text_cache.py`), lens/spectral/Gaussian/normal/palette preparation |

## 7. Retained engines and exactness exceptions

Retained engines are deliberate dependencies, not undisclosed conversions:

| Engine | Where exactness depends on it |
| --- | --- |
| OpenCV 5.0.0 / IPP | Default Surface Blur (proprietary IPP bilateral); Gaussian Blur LINEAR upsampling resize; Average Color Fix CUBIC resize; DFT for Box Blur, Convolve, Image Metrics, Lens Blur, High Boost Filter and the Normal Map Generator spectral path; Distance Transform primitive; image codecs |
| NumPy 2.5.3, scipy-openblas 0.3.34 (BLAS/LAPACK) | Color Transfer covariance and eigensolvers (complex, as NumPy 2's `eig` returns); Alpha Matting/Chroma Key CG dot products through NumPy's ILP64 OpenBLAS `scipy_cblas_ddot64_` (pinned reduction order) |
| FreeType/RAQM (via Pillow) | Glyph shaping and rasterization for Add Caption and Text As Image; layout and pixel assembly are C |
| CPython, WCMatch | Integer/Unicode/hash/random primitives (MT engine); glob parsing |
| FFmpeg, codecs | Video decode/encode; OpenCV/Pillow/AVIF/DirectXTex image codecs |
| ONNX Runtime, NCNN, PyTorch | Inference and framework objects. The 8 excluded PyTorch nodes (Convert To NCNN, Convert To ONNX, Guided Upscale, Align Image to Reference, Inpaint, Load Model, Upscale Face, Upscale Image) keep their engines; only Align Image to Reference and Upscale Face changed (helper reuse) |

Native work that reproduces a retained engine's choices reads that engine's own dispatch at run time, never CPU
features alone:

- **OpenCV 5.0.0** (`native_opencv_simd.target`): the wheel dispatches AVX512-SKX over an SSE3 baseline. Here
  median_blur, box_filter, bilateral_filter, sumpixels and matmul run AVX512_SKX (16 float32 lanes); every other
  family, filter and morph included, runs AVX2 (8 lanes), and the baseline under `setUseOptimized(False)`. Median Blur
  is C for all radii and channels and follows median_blur's width for radius <= 2; Convolve's spatial path reproduces
  the filter family's accumulation.
- **NumPy 2.5.3** (`native_numpy_simd.loop_target`): its MSVC wheels build X86_V2 as the baseline and dispatch X86_V3
  only (X86_V4 is disabled under MSVC), so the float32 loops the mirrors follow (sin, cos, exp, log, minimum, maximum,
  add, multiply) run X86_V3: 8 lanes with FMA3, and MSVC's `/fp:contract` fuses their quadrant step. Add Normals and
  Lens reproduce that sine and exp. Full reductions follow NumPy's iterator (`native_numpy_reduce`): one pairwise tree
  for an unbuffered operand, else buffer blocks that restart at every outer index.
- **The C runtime** (Consult 6 D-4, `cn_crt_math.h`): mirrors of NumPy's, SciPy's and CPython's own scalar math
  (pocketfft's twiddles and `_nd_image`'s Gaussian included) call the `ucrtbase.dll` exports those libraries use,
  through the `cn_crt` table; OpenCV's mirrors keep the static CRT, as cv2.pyd does.

Wavelet Color Fix is C for the default CPU float32 domain; Torch device, autograd and non-default precision keep the
framework path.

Deterministic corrections of undefined or defective baseline behavior, and the kept deviations from the installed
chaiNNer (outputs may differ only here; the 13 bench graphs equal chaiNNer's at 32 and 8 CPUs, so none changes a pixel or frame):

- Inpaint: tiny-image invalid reads. Dither (Palette): tie and large-palette panics resolved
  deterministically. Median Blur: wide-window overflow. Dilate and Erode: a dimension above `INT_MAX` is a size
  overflow, as OpenCV's int dimensions. Other oversized arithmetic.
- Nonconstant border extension of an empty image raises `ValueError` instead of hanging in OpenCV 4.8's reflective-border loop.
- Threshold TRUNC NaN/zero ties; Specular to Metal signed-zero clamp.
- NCNN's tile reader (`ncnn/auto_split.py`) reads its input by logical index (Consult 11 D-18), where upstream's
  `Mat.from_pixels` reads the raw buffer, a permuted one as interleaved pixels. No node output changes: Upscale Image's
  cvtColor hands the reader a C copy of 3 and 4 channels, and one channel has one order, so Lens Blur's planar layout
  never reached that read through the node (VERIFIED on CPU, upstream's node and the port's).
- NCNN optimizer: lines of upstream's `optimizer.py` are corrected before translation and in the tests' oracle
  (`CORRECTIONS` in `native/tests/reference_ncnn/generate_optimizer_cpp.py`). MemoryData–Split–BinaryOp fusion starts
  from MemoryData layers only, as ncnnoptimize does; upstream's inverted test sent any other layer that feeds a Split
  and a two-input BinaryOp into it, which raised `KeyError` (`'0'`, `'1'` or `'data'`) after a partial mutation, so
  such ONNX graphs failed to convert (`d = x - mean(x); d * d`; 15 of `test_branching_chain`'s 30 graphs). The
  backward searches of Dropout, 1x1 Pooling and Split elimination and of the BinaryOp→Eltwise fusion find no producer
  for a first layer (upstream chaiNNer #2397: they ended on −2, or 1, and rewired another layer; only a hand-written
  `.param` without an Input layer starts with one). No graph that converts or loads correctly upstream changes:
  chaiNNer-C neither emits nor parses MemoryData, so that fusion only ever raised, and outside a first layer the
  searches end where upstream's do (`test_corrected_oracle_departs_from_upstream_only_as_recorded` checks the whole
  pass corpus).
- NCNN param schema: Reduction's `reduce_all` (param 1) defaults to 1, ncnn's own default, where upstream's schema
  says 0 (owner-approved 2026-10-09). Parameters equal to their default are not written, so upstream omitted the `1=0`
  of a per-axis reduction, both when Convert To NCNN made one and when a loaded `.param` (ncnn's tools write `1=0`)
  was handed to ncnn, and ncnn reduced over everything (a converted channel mean was off by 1.26). Only such
  silently wrong models change; `1=1` and an omitted `1=` stay global. The differential tests' oracle reads the
  frozen `param_schema.json` with this default corrected (`SCHEMA_CORRECTIONS` in `native/tests/test_ncnn_graph.py`;
  `backend/tests/test_ncnn_reduction.py` compares a converted model with ONNX Runtime).
- Load Image and Load Images decode 8-bit RGBA TIFFs with unassociated alpha (ExtraSamples 2, orientation 1-4) with
  Pillow, keeping straight colour, where upstream's OpenCV path premultiplies it and loses the colour under alpha 0
  (upstream chaiNNer #409; owner-approved 2026-10-09). 8-bit grey+alpha TIFFs under the same conditions load as BGRA
  with the grey in B, G and R, exactly as a grey+alpha PNG loads, where upstream's OpenCV path drops the alpha and
  returns one channel (owner-approved 2026-10-09). Every other file, associated alpha included, takes the OpenCV path
  as before (`backend/tests/test_load_image_tiff_alpha.py`).
- Load Image and Load Images turn JPEG, PNG, WebP and AVIF files upright as their EXIF Orientation (2-8) asks, as
  `ImageOps.exif_transpose` does (5-8 swap height and width), where upstream ignores it (upstream chaiNNer #1312;
  owner-approved 2026-10-09). Both read paths (`read_cv`, `read_pil` in `image_io.cpp`) apply one transform that keeps
  the channel order; `read_cv` finds the tag by walking the file's segments or chunks (a PNG's eXIf before or after
  IDAT), `read_pil` in the metadata Pillow already read. TIFFs, which OpenCV and the #409 path already orient, and
  files without the tag or with Orientation 1 load as before (`backend/tests/test_load_image_orientation.py`).
- Save Image's RGBA TIFFs (U8, U16, F32) carry ExtraSamples = 2 (unassociated alpha), which TIFF 6.0 requires and
  OpenCV's encoder omits (upstream chaiNNer #2950; owner-approved 2026-10-09): `add_extra_samples` in `image_io.cpp`
  appends a copy of the first IFD with the tag and points the header at it, so the file is OpenCV's plus one IFD;
  pixels, compression and predictor are unchanged, RGB and grey TIFFs are byte-identical
  (`backend/tests/test_save_image_tiff_extra_samples.py`).
- Save Image's lossless WebP of an RGBA image is Pillow's with `exact=True` (owner-approved 2026-10-09; side finding
  of upstream chaiNNer #2914), where OpenCV's libwebp call drops the colour under alpha 0; every sample round-trips.
  The file grows only by that colour: +0.1 % when transparent pixels are uniform, +55 % on a synthetic worst case
  (a third transparent with noisy colour). RGB lossless and every lossy WebP stay OpenCV's bytes
  (`backend/tests/test_save_image_webp_exact.py`).
- Save Video's audio mux (owner-approved, upstream chaiNNer #3331): chaiNNer's own FFmpeg muxes the audio after the
  video, as upstream v0.25.1 does, where the frozen nightly ran the `ffmpeg` on `PATH` with the audio first and only
  logged a failure, so without FFmpeg on `PATH` every saved video was silent. Auto copies the audio and, when the
  container cannot hold the copy (PCM in MP4 on the integrated FFmpeg 5.1.2), transcodes it with Transcode's options
  (AAC at 320 kb/s; upstream has no fallback). Audio the mux cannot carry fails the run instead of leaving a silent
  video, where upstream logs: Copy names the codec and container and suggests Auto or Transcode, a failed transcode
  names its encoder (mono audio into WebM's fixed 320 kb/s Opus fails so, the bitrate is the owner's call); a WebM
  Copy still raises before the mux, as in the nightly. Only a source file without audio saves the video without it.
  The temporary file never stays behind, and `os.replace` swaps the files, so a failed swap keeps the video. Video
  packets are unchanged; the stream order and, where a PATH FFmpeg differed, the audio encoder's build change. The
  tests' oracle is the frozen v0.25.1 mux with these departures (`INTENDED_MUX`, `native/tests/test_video_io.py`), and
  the video verifier compares the oracle's files video stream first (`verify_video_runtime.video_first`).
- Sibling event order (SP2): inputs awaited together (`owned_gather`, section 4) start and finish in pool-timing
  order, where upstream awaits inputs in turn (`start a, finish a, start b, finish b`). The event multiset and each
  node's final state are identical (`verify_runtime.event_contract` and `sse_contract` compare no order); item order
  and each node's dependency order are kept. Python 3.14's `asyncio.futures._chain_future` sets a pool job's future at
  once when the job finished before it was wrapped, which adds upstream's own order as one interleaving. Measured
  2026-10-07 (g0 → {a, b} → join → save, 200 runs of 6 items, CPUs 28–31): upstream 1200 of 1200 items `sa fa sb fb`;
  the port at K = 1: 891 `sa sb fa fb`, 296 `sa sb fb fa`, 13 `sa fa sb fb`.
- The normalize/enforce clamp follows `np.clip` on the tested stack (NumPy 2.5.3 `clip.cpp`: `x<min→min`, `x>max→max`; −0 and NaN payloads pass; under DAZ a denormal becomes the converted zero of its sign). On the 3.11 stack, NumPy 1.24.4 gave +0 for −0 and Task 3b1 (`602146a5`) mirrored that; U2 lane H restored B3's form when the stack moved.
- Kept for the owner's record: default PNGs are encoded with fpng, pixel-identical to chaiNNer's but about 19 % larger
  (7.97 → 9.51 MB for 32 files, VERIFIED 2026-10-04), a deliberate speed-for-size trade the owner may reverse. Composite's
  full-overlap shortcut (SP4b Task 6) allocates nothing and succeeds where B3's region mallocs failed under memory exhaustion.
- Widened annotations, annotation-only (Consult 14 D-31): where upstream annotates `ort.InferenceSession`, the
  backend's session parameters, returns and cache use the `OnnxSession` Protocol (`nodes/impl/onnx/session.py`:
  `get_inputs`, `get_outputs`, `run`), which ORT's session and the native `NativeSession` both satisfy (rembg `bg.py`,
  `session_base.py`, `session_factory.py`; onnx `auto_split.py`, `session.py`; `upscale_image.py`), so the cast to
  `ort.InferenceSession` in `create_inference_session` goes. `parse_onnx_shape` takes any dims sequence (both
  sessions' shapes are lists), and NCNN's `load_layer_weights` takes a `BinaryIO` where upstream's says
  `BufferedReader` (the native reader only calls `read(n)`; no caller uses a BufferedReader-only member). No runtime
  change.
- Bridge annotations widened to the contract the tests pin, annotation-only (Consult 15 B2): `image_fill`'s `color`
  and `second` and `chroma_key`'s `color` take `tuple[float, ...] | np.ndarray`, and `normalize_lens`'s
  `coefficients` takes `list[tuple[float, float]] | np.ndarray`; each already reads them through `np.require` or
  sequence indexing, so foreign (reversed, unaligned) arrays were always accepted. No runtime change.
- One body change of the pyright lane is a code-path removal, not an annotation (R3 M-7): `blue.py`'s removed
  `isinstance(seed, (int, np.integer))` guard was the port's own always-true native-dispatch condition (B0; the only
  caller passes an `int`), not upstream text, removed in `1dee63a3` as a root fix. Upstream's `blue.py` has no
  `isinstance`.
- Don't-cares: which NaN payload survives where two meet (table below); under DAZ|FTZ, which neither chaiNNer nor the
  port sets, Dilate and Erode may return a zero where OpenCV returns a same-sign denormal, or the reverse (equal after the clamp);
  Soft Light with a negative-denormal base gives NaN where upstream gives 0.541 (C `sqrtf` against NumPy's DAZ sqrt);
  and Pin Light returns a flushed zero where upstream's `np.where` keeps the denormal's bits (both equal B3; Consult 13).
  Nothing in the stack sets DAZ (U4-light's MXCSR readout was `0x00000000` after every in-process GPU item), and new threads start
  with the default FP state.

Test tolerances (saved-image and runtime fixtures stay exact, but for the last row):

| Computation | Allowed difference |
| --- | --- |
| Lens Blur float32 power | abs 3e-6, rel 8e-6 |
| Edge Detection hypot | abs 2e-7, rel 3e-7 |
| Binary distance | abs 2e-7 |
| Double Simplex evaluation | abs 5e-14 |
| CAS power | 2e-7 (float32), 2e-15 (float64) |
| SSIM (six decimals) | abs 1e-6 |
| Linear Histogram / Principal Color float64 BLAS | abs/rel 3e-14, before float32 conversion |
| NaN payloads where two different NaNs (a generated one, from inf·0 or inf − inf, included) can meet in one operation (SP4 forced paths, B3 goldens; e.g. a pairwise sum's horizontal tree, where x86 keeps the first operand's NaN and the compiler orders a float addition's operands freely, MSVC in NumPy's build and LLVM in ours, so source order cannot steer it) | NaN positions and every non-NaN bit; single-payload inputs bit for bit |
| Iterated items' broadcast types and values (runtime verifiers, Consult D-35) | Upstream on 3.14 broadcasts iterated items' types in timing-dependent order (CPython `_chain_future`); the UI may show either last item; the io verifier compares broadcast multisets, not last-wins, for these fields (the port's final value is one the oracle broadcast; `report.json` flags an oracle whose own attempts disagree). A same-side check that the port's final value equals its own last broadcast would be a tautology (the final state is the newest-wins merge of those broadcasts); the port's "latest pending" contract is pinned in the `LatestBroadcasts` unit tests instead (`test_execution_scheduler.py`, `test_execution_cleanup.py`; R3 M-1) |

Not claimed: unchanged global Torch RNG state when alignment/face helpers are reused (reuse skips
repeated random initialization); identical PNG bytes between fpng and OpenCV (compare decoded
pixels); video container bytes (compare decoded frames, audio and metadata).

## 8. Language policy

- Keep verified C kernels as C behind the C ABI. Use C++20 for application logic, framework integration, resource
  ownership and execution coordination (owned buffers, spans, scoped cleanup, explicit error translation).
- No C++ exception crosses a C or Python boundary.
- Never rename a source file or wrap a retained engine and count it as a conversion. Retained
  library calls are integration, not algorithm rewrites.
- Strict floating point everywhere (`/fp:strict`, which also keeps contraction off; section 3).

## 9. Packaging

`native/tools/package_port.py` (CLI and orchestration) with `package_files.py` (walk, hash,
verified copy, lock probe, atomic replace) and `package_manifest.py` (schema, identity, recovery):

1. Verify the installed shell: required files, `app_version` from `resources/app/package.json`, inventory = installed
   app minus `resources/src/**`, `*.log` and the inert root `squirrel.exe`; reparse points refused.
2. Build the UI patches in memory (`independent_ui.build_ui_overrides`). Each bundle's SHA-256 must match its pin
   (renderer `50f22a10…fd16877`, main `df979a26…26f10`) and each anchor must occur exactly once; otherwise the run
   fails closed. Both bundles force `checkForUpdatesOnStartup` off; main ignores the old `check-upd-on-strtup-2` key
   and its migration forces false, and its integrated-Python check (`gR.eq(s.version,I)`, equality with the 3.11.5
   download table) becomes `gR.gte(s.version,"3.14.0")`, so a shipped CPython 3.14.x is kept. A missing runtime, or
   one below 3.14, no longer takes upstream's branch that deletes it and downloads 3.11.5: it throws "chaiNNer-C's
   integrated Python is missing or older than 3.14; reinstall chaiNNer-C", which upstream's setup catch logs and
   offers Retry, system Python or exit (Consult 2 Q6). Main shows chaiNNer-C's version (About `chaiNNer v0.3.2`, the
   startup log `chaiNNer-C 0.3.2 (upstream 0.25.1-nightly.2025-10-21)`, system information `app.version` 0.3.2 plus
   `app.upstream`), drops the Release Notes item and leaves `app.getVersion()` upstream's, since it stamps saved chains
   (Consult 2 Q5). The renderer replaces the update header with the logo, the title and a `v0.3.2` tag (no Alpha
   badge; the manifest's `app_version` stays the upstream baseline), removes the update toggle and rejects
   release-API strings. The renderer then applies the 11 drop edits in `input-drop.json`
   (own baseline pin): directory drops with toasts, locked/connected guards for file and node drops, directory
   rejection before extension routing, and forwarding of file drags past elevated edges to the node beneath. Last,
   `tensorrt-types.json` (pinned to all three baselines, the stylesheet `index.css` `2b33220c…609c9f81` included) adds
   the TensorRT navi types (`TensorRTEngine`, `TrtPrecision`, `TrtShapeMode`, `convenientUpscaleTrt`) to the scope
   in both bundles, the TensorRT accent colour to the renderer and its light and dark tokens to the stylesheet. Then
   `tensorrt-clear.json` (same pins) adds a Clear item to the node menu of the TensorRT Upscale Image and Load Engine
   nodes and a `/clear-cache/individual` per removed node to `removeNodesById`, since TensorRT engines and sessions
   outlive chain runs (`nodes/impl/tensorrt/cache.py`).
3. Snapshot `backend/src`: every file except `__pycache__`, which must equal `git ls-files backend/src` plus the two
   binaries; provenance = git HEAD, branch, `backend_dirty`.
   The runtime is the provisioned `native/runtime/cpython-3.14.8`, which must be what `native/python-stack.lock.txt`
   describes (CPython 3.14+ with its `python3XX.dll`, the lock's interpreter, every pin, pip) and hold `chainner-pip`.
   Its copy leaves out `package_files.EXCLUDED` (eight dev-only distributions, and the oracle-only `google-re2`,
   `Sanic-Cors` and `pynvml`): every file their RECORDs list, and each top-level site-packages directory only they own. The run
   fails if an excluded name is not installed or a shipped distribution's active `Requires-Dist` names one (markers
   evaluated for the runtime, requested extras followed to a fixed point).
4. Fresh destination: manifest `copying`; copy shell and runtime with verified `copy2` in a thread pool (never the
   installed `resources/src`); write `portable`; write UI and backend files; parity check of every recorded hash;
   manifest `complete`.
5. Existing destination: every recorded shell, UI and backend file must match its hash and no unrecorded file may sit
   under `resources/src`. Same backend inventory, UI records and app version: `verified_no_op`, no writes. Otherwise
   probe each target for locks before any write, then replace via `<target>.tmp` + `os.replace`, binaries first, and
   delete a removed file only if it still matches its record. A `copying` manifest resumes from verified files.
6. Never touched after the first copy: `python\python` (recorded, not re-verified) and the profile.

At run time the worker keeps numba's cache out of the runtime: `server.py`'s `main` sets `NUMBA_CACHE_DIR` to
`<storage_dir>/numba-cache` before the node modules import numba, so PyMatting's `cache=True` kernels write their
`.nbi`/`.nbc` files there (in the package, `backend-storage/numba-cache`, a ruled root of the real-use pass), not into
`python\python`'s site-packages. A chaiNNer-C deviation: upstream leaves numba's cache beside the package sources.

Manifest v2 (`chainner-c-package.json`, sorted keys, atomic write): `schema` (`chaiNNer-C/package/v2`), `state`,
`identity` (`project`, `destination`, `installed_app`, `installed_python`), `provenance` (`utc`, `git_head`,
`git_branch`, `backend_dirty`), `app_version`, `upstream_updates`, `shell`, `ui_patches`, `backend`, `runtime`,
`python_stack` (`interpreter_version`, `build_tag`, `archive_sha256`, `lock_sha256`, `runtime`, `excluded` with each
reason, `chainner_pip_version`, `bytecode`; a refresh refuses a changed one, and ignores the retired `overlay` key a
package copied before 2026-10-07 carries). The runtime verifiers read `state` and `identity`,
the `verify_*.cjs` scripts read `identity` only; the runtime verifiers refuse a package that is not `complete` or that
another project generated.

## 10. Third-party notices

- `backend/src/nodes/impl/chainner_native.LICENSE.txt` carries every adaptation notice, file by file: chaiNNer-rs (MIT;
  2xSaI credit GPL, HQ tables LGPL), colortrans (MIT), OpenCV 4.8.0, Pillow 9.2.0, NumPy 1.24.4 (FreeBSD msun complex
  math in `ncnn_complex_math.hpp`), PyMatting (MIT), PyTorch TensorIterator stride order, resize crate 0.8.3,
  zhang_hilbert 0.1.1, avx2_mathfun, CPython 3.11.5 `random.py`, pybind11 2.13.6, fpng 1.0.6 (Unlicense).
- Vendored: `native/third_party/fpng` (`PROVENANCE.md`: hashes, Unlicense, size caps),
  `native/third_party/pybind11`, `native/third_party/onnxruntime` (C API header, `LICENSE`).
- Ported into `chainner_ext.pyd`: `native/third_party/chainner_ext`, `regex` and `regex-syntax` (MIT OR Apache-2.0) and
  `unicode` (the Unicode 15.0.0 tables' licence), each with its licence texts and `PROVENANCE.md`.
- Test references: `native/tests/reference_pixel_art/LICENSE-MIT`, `native/reference-sources-tensors/PyTorch-LICENSE`.
  The matting tests compare with the live PyMatting and keep no copy of its source.
- The repository is GPLv3 (root `LICENSE`); the bundled app and runtime keep their own notices.
