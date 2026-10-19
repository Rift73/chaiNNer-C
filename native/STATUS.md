# chaiNNer-C status

Current state, pending work and the decision log. Commands: `README.md`. Design: `ARCHITECTURE.md`.

## State

- instant-exec (2026-10-19, v0.3.7; branch `instant-exec` from `fffe9cf7`; design, plan, lane reports, the engine review
  and the owner queue in `native/local/instant-exec/`, rulings in `native/reports/fable-decisions.md`, "instant-exec";
  24 h, CPU only): outputs as at `fffe9cf7` except the sanctioned changes at the end. Startup: the window loads at once
  and the editor shows the node list cached in `<root>\ui-cache\backend-ui.json` (README Layout; keyed by the app and
  backend build, the Python and its distributions, the GPUs and drivers and `CUDA_VISIBLE_DEVICES`; the live list
  replaces it and re-checks the open chain; no cache = the splash); readiness is pushed over `/setup-sse`; the host
  spawns the worker first and reads `*.dist-info` in process (no `pip list`); the worker serves at once, registers the
  light node modules (137 nodes, 0.37 s), then imports the frameworks in the background with a readiness event per
  package; a run needing a schema not yet loaded waits for it, or fails with its package's import error (A1). A startup
  chain no longer opens "unsaved" (dirty state by content, `src/common/nodes/chainSnapshot.ts`). Headless host from
  source, warm, before -> after: host ready 9.9 -> 1.04 s, light preview 9.9 -> 1.05-1.08 s, PyTorch node 9.9 -> 4.4-5.0
  s, full node list 9.9 -> 5.4-6.4 s; the packaged app (the repo's `electron.exe`, CUDA hidden, alternated with v0.3.6
  r4, medians of launches 2-4 on a quiet machine, ms from launch; `scratchpad startup\p2`): real interface 10654 -> 1220
  (x8.7), window 902 -> 850, a Load Image result 2273 and a Load Model result on the CPU 8101 (r4 shows the interface
  only after every node loaded, 10.7 s); the first launch after an update fills the cache, 13118 -> 10534. Native run
  engine (ARCHITECTURE section 4), the default executor since `2912a49a`; `CHAINNER_C_EXECUTOR=python` runs
  process.Executor, the diagnostic oracle, never a user setting (A5). Pure nodes run on shared workers across branches
  and window items, the others on each run's driver thread in today's order; events are built in C++ and posted in
  batches; a memo cache keeps allow-listed nodes' outputs across runs inside the worker (the user setting Result
  Cache: on by default, size default min(2 GiB, 10 % of RAM), applied at each run's start;
  `CHAINNER_C_MEMO_BUDGET_MIB`, 0 = off, a diagnostic override that wins; a hit sends a computed node's events). The
  differential suite
  (`native/tools/engine_differential.py`: both executors over every bench chain and the executor scenarios; outputs,
  events, A6 order, files) is green; the concurrency review's findings (C1, C2, H1-H4, M1-M3, then N1, N2 and a native
  preview returning no output) are fixed (`review-engine.md`). In process, 32 synthetic 512x384 JPEGs, ms per item at
  auto K, Python -> native: `resize-real-256` 2.58 -> 2.09, `lens-spectral` 6.88 -> 3.41, `parallel-branches` 3.44 ->
  2.24, `palette-median` 2.36 -> 1.75, `normal-map` 3.12 -> 2.73 (x1.14 to x2.02 over ten chains; x1.29 to x2.09 after
  the review fixes, memo off); K = 1 `resize-real-256` 2.15 -> 1.44, `lens-spectral` 6.13 -> 2.93; 12 sinks of 128 MiB
  peak at 131 MiB (Python 129); `bench.py` paired against `fffe9cf7` on the frozen tree (`f4d1a58f`; CPUs 16-31, 2
  pairs, 32 images per trial, 256 for `resize-real-256`, 600 video frames), ms per item before -> after, memo off
  (`bench-20261019T080356Z`): `resize-real-256` 2.82 -> 1.37 (x2.05), `gaussian` 2.96 -> 1.89 (x1.56), `lens-spectral`
  8.32 -> 4.37 (x1.90), `morphology` 4.14 -> 3.35 (x1.24), `normal-map` 3.79 -> 3.02 (x1.25), `caption` 3.62 -> 2.40
  (x1.51), `split-channels` 2.80 -> 2.01 (x1.39), `palette-median` 2.93 -> 1.66 (x1.76), `palette-dither` 4.28 -> 2.47
  (x1.73), `parallel-branches` 4.03 -> 2.26 (x1.78), `video-ffv1` 3.78 -> 3.08 (x1.23), `video-h264` 6.47 -> 5.88
  (x1.10), `video-resize-h264` 4.29 -> 2.88 (x1.49); a rerun on a warm memo cache (`bench-20261019T080808Z`) x1.06
  (`video-h264`) to x6.87 (`lens-spectral`; 7.89 -> 1.15); outputs equal in every trial of both runs. Native nodes (A7:
  bodies of our native code plus glue only): the bench chains' nodes but Normal Map Generator and Add Caption, 63 of 81
  other image nodes and 23 of 41 utility nodes have C++ entry points, each bit-identical to its Python body, error type
  and text included (`native/tests/test_run_nodes_*.py`); a path an entry does not mirror declines to the Python body
  (`RunPython`; no decline on the 13 bench chains). Per call they are kernel-bound (Gaussian 6/6 at 1080p 2361 -> 2077
  µs); the gain is no GIL inside the engine. Codecs and writers: vendored libjpeg-turbo 3.1.4.1 (JPEG8), libpng 1.6.57,
  zlib 1.3.2, libwebp 1.6.0 and libtiff 4.7.1 (ARCHITECTURE section 5) decode JPEG, PNG, WebP, TIFF, BMP and PxM and
  encode Save Image's PNG, JPG, WebP, BMP and TIFF with today's arrays and bytes (a 4,305-file corpus and 3,024 Save
  combinations equal under every JSIMD and ISA cap); anything else declines to today's path, which
  `CHAINNER_C_NATIVE_CODECS=0` forces (the oracle). Native Load Image and Save Image entry points and LargeImage
  previews. One thread, ms: JPEG decode 512x384 1.48 -> 0.52, 1080p 17.8 -> 6.0; Load Images JPEG item 1.86 -> 0.92;
  Save JPG 1.31 -> 0.87, PNG 1.28 -> 0.94; previews 4.7 -> 3.8; PNG and WebP decode about equal (library-bound);
  `_chainner_graph.pyd` 7.7 -> 10.4 MB. Every writer (Save Image, Save Video at a successful close, DDS, the model
  savers, VapourSynth scripts, the TensorRT timing cache; the app's chain and settings saves) writes a temporary file
  beside its target and renames it (A4); a target with equal bytes is left untouched, mtime included. Hunt lanes
  (byte-identical unless listed below). Upscale CPU path: one native interleave per backend (NCHW results are ordinary
  interleaved images now), the typed caster on contiguous runs, the final clip borrowed by the output check, one-pass
  alpha helpers, a native TensorRT tile blend, the ONNX size probe per model; 1080p -> 4K tile blending 246 -> 123 ms,
  NCNN node 174 -> 32 ms, ONNX FP32 34 -> 25 ms, TensorRT tile blend 483 -> 75 ms, RGBA path 311 -> 180 ms. App: a click
  is not an edit; atomic chain and settings saves (an invalid `settings.json` is moved aside, defaults used); narrower
  re-renders; one input hash per node (a stale-hash bug fixed; 0.45 -> 0.12 ms at 31 nodes); uninstall KeyError; package
  rescan after pip; atomic FFmpeg setup; per-process temp folders; clipboard images by content hash. Video: the CLI
  reader in 1 MiB reads (4K float 430.7 -> 52.9 ms per frame), cached MKV/WebM rate scans and ffprobe results (repeat
  setup 25.3 -> 0.2 ms), native plane interleave (PyAV 1080p read 5.09 -> 2.83 ms per frame), vspipe evaluation 142.6 ->
  116.0 ms. Models: sessions follow a changed setting (they kept the old session), Convert To ONNX exports a copy and
  caches exports, no random init in spandrel builds (DRFT-L build 727 -> 249 ms), ONNX parsed once (DAT2 load 423 -> 347
  ms), a Build Engine cache (memory; the disk "Engine Cache" setting off by default), Load Models read-ahead (5 models
  3.6 -> 2.9 s). Gate: NumPy pool idle cap min(512 MiB, 10 %) -> min(2 GiB, 10 % of RAM) (R10; tiled 4x 1080p item 134.2
  -> 100.8 ms); the Engine Cache key includes its record format (R11); `run_tests.py` keeps the Numba cache per checkout
  (R12). Second wave (R13): preview rate limit inside loops (R4: per node the first at once, then one per 250 ms, the
  newest kept; a 1080p 20-item loop 20 -> 7-8 previews) and an encoded-preview cache (64 MiB, 64 entries, held weakly; a
  repeated Load Image /run 60.1 -> 14.0 ms, its event text 10.9 -> 2.4 ms); MP4 faststart (R5a), Load Video audio trim
  (R7), Load Images listing, previews of loading nodes held in the window (R13 d), `gc.freeze` (R1; a full collection
  after loading 226-253 ms -> 0). Phase-end reviews (six) and delta reviews (three), all findings fixed with tests: a
  preview race at startup (the older of two waiting previews could win and feed Run), the UI cache key covers backend
  files updated in place, File > Open and drag-and-drop wait for schemas the cached list lacks, pip runs before the
  early worker, a settings file with a BOM, quit during startup ends the backend, the editor survives a failed live
  list; native Stop answers in 0.52 s (was 4.7 s, and the host restarted the worker; e2e default chain /kill 6.31 ->
  0.51 s, rerun equal), a second Stop recovers a hung node, a never-started run leaks nothing, late errors are logged; a
  NativeExecutor reference cycle (outputs and previews kept until a collection) and PyAV's filter-graph cycle (31.7 MiB
  per video run) removed (the NumPy pool release check passes again); atomic writes replace a target open with delete
  sharing (30/200 -> 0/200 failures), short temp names for long names (P8), invalid names fail before writing (a
  wildcard name could replace a sibling), no lost encode at Save Video's close; TIFF straight alpha honours the size
  limit, lone-surrogate and NUL paths behave as `open()`, Pillow's EXIF dpi failure declines, codec allocators never
  throw; the ffprobe cache skips inputs that read other files and sees same-size rewrites (change time); flipped clips
  read through the CLI (the 8-bit one crashed the worker); the VapourSynth upscale child gets TF32; Convert To ONNX
  keeps one export per model; `auto_split` overlapping rows and host out-of-memory split; one image normalization, exact
  value types, one exception mapping, NumPy's memory error text; CPython's `float_pow` for non-finite noise ratios;
  generator outputs memoized only when they follow from their inputs (a replaced video's FPS was stale). Image size
  limit (owner, 2026-10-19): 40,000 x 40,000 = 1,600,000,000 pixels on every decode route
  (`backend/src/image_limits.py`, `CN_MAX_IMAGE_PIXELS`), no warning below, refused above (was Pillow 89,478,485 /
  178,956,970 and OpenCV 1 << 30). E2E on the frozen tree: 7 chains x 6 steps equal on both executors. Behaviour changes
  (Fable). Sanctioned output changes: H4 Q1 and R8 (an FP32, Mixed or DUAL ONNX export, and PyTorch -> NCNN's FP32
  intermediate, of an FP16-loaded model start from the file's FP32 weights when the file is unchanged and they round
  bitwise to the loaded ones, else from today's FP16 weights with a warning); H4 Q4 (ONNX keeps the loaded file's bytes,
  so a non-standard writer's file saves as it was, not re-serialised); R5a (MP4 faststart on the mux pass when audio
  follows: atom order changes, streams equal); R7 (Load Video with a limit trims the audio to the video; it copied the
  whole audio). A4: a target held open without delete sharing fails with [Errno 13], a symlink at a non-Save-Image
  writer's target is replaced, a failed or stopped Save Video leaves no file. Event order between independent nodes may
  differ (C; the kept invariants are checked). Phase-end rulings (P1-P9): the compact layout of engine images is the
  reference for float reductions downstream of a view (P7; last-digit and signed-zero differences after Crop with
  offsets or Separate RGBA, `LAYOUT_DIFFERENCES`); GFPGAN's CPU noise no longer repeats a fresh worker's first run (P3);
  a stopped node with no abort check finishes in the background and the next run waits for it (P4; the Python oracle
  still waits in /kill); a long target name uses `.cnc-<pid>-<seq>.tmp` (P8); more than 512 channels in the shared ops
  raise ValueError (P5). Informational (R1): the worker runs one `gc.collect()` and `gc.freeze()` after node loading,
  lifting the 2026-10-06 scope line against `gc.freeze` (an explicit collection 136-164 ms -> about 0); skipped, never
  retried, while a run is in flight or after a heavy-package run.

- instant-exec follow-ups (2026-10-19 evening; owner yes on the queue; shipped as v0.3.7 re-releases): v0.3.7 released
  (fork tag `v0.3.7`; re-release 1 `043317fd`: a /nodes request given up while the nodes load no longer cancels the
  loading, which left a GPU install's first start without a node list); `.exr` loads (vendored OpenEXRCore 3.5.2 with
  Imath 3.2.3, zstd 1.5.7, libdeflate 1.26 and OpenJPH 0.32.0, `native/third_party/*/PROVENANCE.md`; OpenCV's EXR
  semantics, float values clipped to 0..1 at Load Image's output like every float image (Fable); 4K RGBA half 18-36 ms);
  the Result Cache as a setting (Settings > Python, "chaiNNer_standard": on by default at min(2 GiB, 10 % of RAM), a
  size in MiB, off empties it; `CHAINNER_C_MEMO_BUDGET_MIB` still wins); Frames to Clip's FFV1 temp in 16 slices where
  valid (sides >= 7 px, <= 2^27 bits a slice; 1080p 21.8 -> 10.6 ms, 4K 82.1 -> 37.2 ms per frame, frames equal);
  spandrel 0.4.2+c7 (DRFT's fold without a per-channel loop: DRFT-XL 2447 -> 1652 ms, outputs equal); the image size
  limit 40,000 x 40,000 pixels (owner); the disk Engine Cache stays off (owner: engines are built and saved); no pip
  cache (owner). Open: the Python executor's retirement (owner, after v0.3.7's use); MAINTAINING section 10's checks
  re-based on chaiNNer-C (they compare with upstream; Fable); the hosted CI fails on 18 older pyright errors.

- VRAM after Stop, again (owner report 2026-10-19 20:09 on D:, a DUAL upscale with No Tiling overflowing VRAM): Upscale Image
  (PyTorch) checked Stop only between tiles, so the tile running at Stop kept its memory until its forward ended (the
  native engine runs it on in the background; minutes when VRAM spills into shared memory). The forward now checks
  Stop at each block (each module with children; per thread, the hooks only for the call) and synchronizes before
  Aborted, which leaves the forward's frames behind. ESRGAN 1280 px single tile, Stop mid-tile: VRAM back 1.1 s ->
  0.39 s after Stop; 1536 px in 256 px tiles 2.31 -> 2.07 s (one paired run). Also: `progress_controller.Aborted` is
  `api.Aborted`, so a node's own check_aborted ends its run as stopped, not as a NodeExecutionError, and a stopped
  run's result read within the grace is no longer reported as never retrieved. Final code on the GPU: 0.40 s, log
  clean. Released as v0.3.7 re-release 3 (owner yes; fork tag `bdb5372d`, zip `EFE50B86...3409`). In a VRAM
  overflow (owner: about 2 s with DUAL, No Tiling) the floor is the GPU work already running: a 1x ESRGAN at 6000 px
  freed its VRAM 6.5 s after Stop, the same with at most 1, 2 or 4 blocks queued ahead or none (the CPU waits about
  2 s in one CUDA call while Windows pages, then 4.4 s of running operations); ending the worker freed it after 1.9 s.
  The queue limit was dropped. Fixed meanwhile: a node clear right after Stop no longer returns the GPU's memory on
  the loop while the stopped node still runs (that wait stalled the loop 2.1 s).

- VRAM offload (owner 2026-10-20: "VRAM overflow but fast, like RamTorch but better", No Tiling kept): Upscale
  Image's forward on 2 MP or more runs under a dispatch mode (`nodes/impl/pytorch/offload.py`) with PyTorch capped
  3 GiB below the free VRAM, so Windows never spills to shared memory; past cap - 2 GiB the least recently used
  activations go to pinned RAM on a copy stream and come back when an op needs them; tensors sharing memory never
  move. The owner's 1x DUAL (C128G4D16), FP16, No Tiling: 2176 px 232.65 s -> 5.96 s first run, 4.95 s warm (4.6 s
  at the fitting rate); 1664 px (fits) 2.71 -> 2.71 s; forced to 14 GiB, 13 GiB moved, output bit-identical. The
  worker sets PYTORCH_CUDA_ALLOC_CONF to page-lock pinned RAM with 8 threads (about 1 s off a first overflow) unless
  the user set it. A 1x ESRGAN gains nothing (its peak is one op's working set). The pinned RAM goes back to Windows
  when a run ends (owner's option B; torch 2.14's internal `_host_emptyCache`): through the backend, DUAL 2176 px
  moved 14.4 GiB and the worker held 2.3 GiB after the run.

- YUV planes, colour information and high bit depth (2026-10-18, branch `feat/yuv-planes`; local design and
  rulings `native/local/yuv-planes/`): video planes are float32 gray images at their stored sizes (Y 0 = black,
  1 = white; U and V centred on 0.5; `nodes/impl/colour_info.py`, every code round-trips exactly). Load Video has
  Mode RGB / Planes / Both, RGB Precision Float (default; chroma sited as tagged) or 8-bit (as before, byte-identical;
  a chosen Matrix or Range applies, Chroma Siting does not); Save Video's output of old chains changed in bytes (RGB
  is piped as gbrp16le, bgr24 for GIF and 8-bit RGB output, and converted with explicit chroma siting and tagged
  -chroma_sample_location: better, but not byte-identical). Load Video has the Colour Info and Colour outputs and
  five Auto-first colour overrides; detection reads the stream's tags, then HDR side data, then cross-field rules,
  then mpv's size guesses, and warns on implausible tags. Save Video takes RGB or Y/U/V planes (subsampling from the
  plane sizes), Colour Info, the overrides, Bit Depth (Auto: Colour Info's, rounded up to a depth the encoder writes,
  capped for H.264 at 8 and H.265/VP9 at 10) and chroma siting correction, and writes every colour tag and HDR10
  metadata.
  New nodes Planes to RGB / RGB to Planes (zimg's maths, within 1e-6 of VapourSynth's resize). Frames to Clip writes
  16-bit (default) or 32-bit float and takes planes; Clip Frames has the same Mode. Mode-dependent outputs are typed
  null in the other modes (`BaseOutput.null_unless`). The plane maths run in C with runtime ISA dispatch (NumPy's
  results bit for bit; 4K 4:2:0 planes to RGB 259 -> 19 ms). A C++ TensorRT runner (`native/trt_runner`: N streams,
  pinned buffers, CUDA graphs, planes converted on the GPU, a VapourSynth filter and a Python binding) runs AI
  Upscale's TensorRT backend by default, with the node's Streams input (1-4, default 1; each stream about one more
  frame of VRAM); `CHAINNER_C_TRT_RUNNER=0` keeps the Python bridge, `=1` also puts TensorRT's Upscale Image node on
  it, and a build without the runner's binaries falls back to the Python paths with one log line. GPU phase 1 found
  its results bit-identical to the Python paths (ESRGAN); the short check of the default with 1 and 2 streams is
  pending (GPU busy). TensorRT is pinned at 11.4.0.106 (was 11.2.1.2; native binaries built with the 11.4 SDK and
  CUDA 13.4); engines from 11.2 must be rebuilt, VOSR's rebuild themselves.

- Full DDS support (2026-10-18, branch `feat/dds-full`, merged into `feat/yuv-planes`): texconv May2026
  (2026.5.8.1); chaiNNer hands texconv and reads back raw uncompressed DDS intermediates (8-bit, 16-bit or float by
  the format's depth) instead of PNG, and 8-bit targets write the PNG era's bytes (266 files byte-identical,
  `native/tests/test_dds.py`). Save Image writes 75 formats (every one texconv encodes but YUV/video and XR_BIAS)
  and has five new inputs: Colour Values, Premultiply Alpha, GPU Compression (BC6H/BC7 on the GPU, on by default as
  before), Mip Levels and Mip Filter. New nodes: Load DDS (any mip level, layer or cubemap face and volume slice,
  with the format, size, mip levels, layers, depth and cubemap flag), Load DDS Images (a mip level's layers, faces
  or slices) and Save DDS Texture (an array, cubemap or volume). Values: 8- and 16-bit formats store 0..1, SNORM
  maps 0..1 to -1..1 at every depth, float formats store values as they are, integer formats store the values
  themselves; Load DDS and Load DDS Images keep values outside 0-1.

- VOSR 2.0 (2026-10-13; guide `VOSR2.md`, maintainer page `VOSR2-MAINTAINING.md`): one PyTorch node, Upscale
  Image (VOSR), which downloads the official model (about 7 GB) on first use (Load VOSR Model removed), on a fixed BF16/FP32 policy with GPU Memory Default or Low and the PyTorch, TensorRT
  and TensorRT FP8 backends. In a Load Video loop the node stabilizes frames (Three Frames or Persistent Detail,
  strength 65 %) with frames ahead from the item window's new look-ahead (`NodeContext.iteration`, the `iterated`
  input condition; ARCHITECTURE section 4), else from past frames only. Gone: VOSR 2.0 Upscale, Temporal Sequence,
  Precision, Original mode and tile inputs, FP16, High. 2026-10-19 (owner): GPU Memory Low offers TensorRT and
  TensorRT FP8 (were PyTorch only): weights streamed beyond a 2 GiB activation reserve, outer tiles of 512 px;
  engines built with RUNTIME_ACTIVATION_RESIZE (every set rebuilds once), so activations follow the input instead
  of the 2176x3840 profile (VAE encoder 7.6 GiB at any size before). RTX 5090, 768^2 2x: FP8 Default 11.47 -> 6.04 GB
  above idle at 0.6 s; Low FP8 2.66 GB 1.1 s, BF16 2.84 GB 1.7 s, the same peaks at a 3072^2 output; Low PyTorch
  ran out of its budget on that case. Low tiles: 34 dB from Default (textures differ per tile).
  The TensorRT plugin `vosr2_qkv.dll` is an optional native target
  (`14dfaec0…`, four reproducible builds); the FP8 graphs are hosted (`vosr2-fp8-f9e142df5846`, 3.45 GB, SHA-256
  pinned) and compiled on first use (191 s, compile 177 s). Since 2026-10-17 (owner) the set is hosted and kept
  Zstandard-compressed (`vosr2-fp8-3ebb2423591d`, format 2, 2.68 GB; each file's `zstd` entry) and unpacked only
  for a build (10 s, deleted afterwards); engines stay keyed by the graphs' own pin (`GRAPHS_SHA256`), and other
  graph sets are deleted. The owner uploads `out\vosr2-fp8-3ebb2423591d` to Feral before this ships. The cached
  BF16 and FP8 engine sets keep Zstandard level 3 plans too (`vosr2_plan_file.py`; 74-82 %, about 5 GB -> 3.8 GB;
  streamed into TensorRT at about 1 GB/s); a set cached plain is compressed in place on its next use. Verified on the RTX 5090: PyTorch images bit-identical to
  the old node (256×256 and 640×360 → 2560×1440); a 30-frame Load Video chain bit-identical to the old Temporal Sequence
  for both methods on PyTorch; TensorRT BF16 frames 51.1 dB minimum, 52.2 dB mean PSNR from the old ones (rebuilt
  engines; one image alone 49.1 dB); FP8 from the hosted graphs 51.9 and 43.5 dB from the old FP8 outputs, which the
  old plans reproduce bit for bit. 640×360 → 2560×1440 per frame: PyTorch 4.60 s, TensorRT 1.40 s. CPU: the VOSR,
  executor and item-window files 2331 passed, only the 4 known DDS failures; pyright 0 errors; lint, runtime type check
  and vitest (136 passed) clean. Final acceptance on the merged code: FP8 from the real host (3.45 GB in 62 s, build
  157 s, later runs neither download nor build), PyTorch Three Frames through Load Video 30/30 frames bit-identical,
  BF16 build with the timing cache 166 → 91 s; full suite 111,591 passed, only the known failures. Grain pass
  (2026-10-14, owner: B x1.4, grain follows the source): synthesized output-resolution grain replaces the enlarged
  source grain (Castorice 2x: within 2 % of the approved sample's strength, neighbour correlation 0.36 instead of
  0.72); TensorRT Three Frames, 30 frames: identical to before outside grain areas, a still background's grain
  0.9997 correlated frame to frame, 1.42-1.43 s instead of 1.39 s per later frame. Window layout (2026-10-14, owner:
  placement + face windows): windows placed so seams run through low-detail areas, plus a window on each face a seam
  splits (YuNet, downloaded once), centred on the face box, laid over the grid's blend: it alone gives the face's
  middle (400 px) and fades out over 48 px (blend share step at most 0.26 a latent; grid hand-overs 0.51; the
  reviewed first version's raised blend weight cut at 1.00); the issue-13 portrait 2x, TensorRT BF16: the front face's
  right/left cheek detail gap 1.88 -> 1.08 (placement alone 1.56), 4 face windows, 2.92 -> 3.13 s; PyTorch gap 1.07;
  Rin and the texture sample (no faces) within 6 % except Rin's blurred background 12 % softer; 30-frame video: one
  layout for the shot, 1.44 s vs 1.49 s per frame, no extra flicker. Seam triggers from input detail, prediction
  differences and detail amounts could not find the face (all parked).
- Temporal Fix (2026-10-13): a PyTorch node running pifroggi/vs_temporalfix's AI model (Apache-2.0, weights fetched
  from the pinned upstream commit and checked by SHA-256) over a loop's 7-frame window; with look-ahead it equals a
  standalone reproduction of upstream's CUDA path bit for bit (30/30). After Upscale Image (VOSR) or PyTorch Upscale
  Image the executor computes that producer up to 3 items ahead, in order and once per item (side-effect-free
  producers, fail-fast loops, a pool with the threads its waits need); GPU check, Load Video -> VOSR -> a 3-ahead
  consumer, 30 frames: VOSR bit-identical to its references, served 3 ahead to the end. No device lock: GPU nodes may
  overlap as before. Elsewhere (side effects, deferred errors) it mirrors the past side.
- Descale (2026-10-16): Image > Dimensions > Resize; vapoursynth-descale Release 12 (MIT, vendored unchanged in
  `native/third_party/descale`) under `src/descale_ops.c`, which runs its row and column passes on the pool in a
  workspace the bridge takes from NumPy. Bit-identical with the Release 12 VapourSynth plugin (12/12 cases, kernels,
  shifts, fractional sizes, borders); 1080p -> 720p RGB 1.0-1.5 ms (first version 9.7-10.4 ms, fresh 52 MB per
  call). A lane-for-lane AVX-512 port (bit-identical) gave no end-to-end gain (memory bound) and is not built.
- Branch `chaiNNer-C` on upstream `d56e507f`. SP1 took `native/` from `codex/c-port` at `c081e940`; that branch is kept as-is, with the SP1 plan and its execution rulings on top.
- One backend: `backend/src` = B0 (426 files) plus SP2's, SP3's, SP3b's, SP4a's, SP4b's and SP4c's edits (ARCHITECTURE section 2) and rebuilt binaries (git-ignored);
  the sources are tracked with their shipped bytes. Until SP1 the package was the installed backend plus 216 overlays applied through delegates, transplants and AST pins; that mechanism is retired (it and prior history: tag `chainner-c-checkpoint-20261001`, `fede43d8`).
- Package `out\chaiNNer-C`: installed `0.25.1-nightly.2025-10-21` shell, two pinned UI patches,
  `backend/src`, the binaries and the CPython 3.14.8 runtime, built fresh at `d0280d17` for the Python-stack upgrade
  (U3, next line) and refreshed at `a831d6b6` (one file, `server.py`: the worker's unsupported-CPU refusal; the
  binaries unchanged); the last CPython 3.11 package is kept read-only as `out\chaiNNer-C-py311`.
- Pinned nightly restored (2026-10-18): the owner's chaiNNer update (2026-10-08) had removed the installed
  `0.25.1-nightly.2025-10-21`; upstream's official Windows zip of it (SHA-256 `8e93b1f4…ff451`, GitHub's digest) is
  extracted to the git-ignored `native\local\upstream\app-0.25.1-nightly2025-10-21` (`independent_ui.UPSTREAM_APP`,
  README Prerequisites), which `make_oracle.py`, `package_port.py`, the DDS test's `texconv.exe` and the layout gate's
  oracle read. Its UI bundles equal the pinned ones; its backend is upstream's own, without the installed copy's local
  delta (ARCHITECTURE section 2), so the oracle no longer carries the removed GPU lease.
- Portable build (owner correction 2026-10-07): the shipped binaries are baseline x86-64 again and run on any
  x86-64-v2 CPU (NumPy 2.5.3's floor), with per-file AVX2/AVX-512 ISA units behind `isa.c`'s dispatch; the
  AVX-512 CPU refusal is gone; `-march=icelake-server` is the local-only `CHAINNER_C_MARCH`. Binaries
  `chainner_native.dll` `B3F730B4…`, `_chainner_graph.pyd` `29D686AA…`, `chainner_ext.pyd` `E5994117…`.
- U3 done (the Python-stack upgrade, 2026-10-07): CPython 3.14.8 (python-build-standalone `20261003`; the stack is
  `native/python-stack.lock.txt`, which provisioning installs), built with clang-cl 23.1.2 `-march=icelake-server`,
  MSVC 14.44.35207 and SDK 10.0.26100.0. Package binaries: `chainner_native.dll` `4233A895…`, `_chainner_graph.pyd`
  `3CF95957…`, `chainner_ext.pyd` `AF59F462…`, independent of the checkout's path and line endings. Final acceptance
  on the fresh package (main `d0280d17`): runtime verifiers 4/4 (`runtime-20261006T222848Z`,
  `io-execution-runtime-20261006T222950Z`, `utility-runtime-20261006T223111Z`, `video-runtime-20261006T223231Z`),
  parity 13/13 at 32 CPUs (`bench-20261006T223415Z`), real-use pass 15/15 gates (`real-use-20261006T224735Z`). The
  cross-stack bench, 3.11 against 3.14, ran on the pre-D-16 3.14 package (D-16, `9708c9f5`: 18 producers return
  upstream's memory layout) and stands, since none of the 13 graphs uses those producers (`bench-20261006T203031Z`,
  recorded, not gating): 13 of 13 neutral, outputs equal across the stacks in 12 of 13 cases, `lens-spectral` within
  max abs 1.
- U4 (GPU, 2026-10-07, under the owner's rules: ESRGAN models only, benchmarks of about 50 iterations): the owner's
  `default.chn` (1x Saiyajin, PyTorch CUDA fp16, the owner's own settings) gives pixel-identical output on the port, the
  oracle and the installed chaiNNer, each with a VRAM peak of 1.22 GiB (`default-gpu-20261007T045807Z`). The throughput
  item and the helper leases: `GPU-DEFERRED.md`. ONNX Runtime's TensorRT and DirectML providers are not exercised; on a
  machine without NVIDIA, the ONNX package declares `onnxruntime-directml`, the untested path for those GitHub users.
- All 161 installed node schemas and the UI are preserved (native work covers 122 of them). Updater removed; drop repairs, the generator-once resize fix (user-confirmed), fpng for default PNGs and the broad optimization pass (CPU results, below) are installed.
- SP0 done: B0 frozen; tracked harness `native/tools/bench*.py`; noise floors for all 13 cases from 3 A/A launches (1.35-6.44%); auto-confirmation of flagged verdicts.
- SP2 done: helpers shared between concurrent native calls, partition from `(count, grain)` alone, lone gathered branch awaited directly; `parallel-branches` 48.6 to 84.8 img/s (gate `bench-20261002T033751Z`, confirmed by `bench-20261002T034508Z`: 11 of 13 cases neutral, 1 win, `video-resize-h264` unconfirmed, no regression). At 8 CPUs (affinity `0xFF`): 38.3 to 56.8 img/s, no regression (`bench-20261002T041903Z`, confirmed by `bench-20261002T042902Z`; floors from 32 CPUs).
- SP3 done: cross-item pipelining (item window K = clamp(cpus // 4, 6, 8), ordered commit, read-ahead only when safe, Save prepare/commit split); gates `bench-20261002T183704Z` (32 CPUs), `bench-20261002T185935Z` (8 CPUs, affinity `0xFF`) and `bench-20261002T185102Z` (32 CPUs, K = 1), confirmed by `bench-20261002T184404Z`, `bench-20261002T190906Z` and `bench-20261002T185850Z`: PNG outputs byte-identical, video decoded frames and stream metadata identical, Final win on all 13 cases at 32 and 8 CPUs, K = 1 neutral against B0 except SP2's `parallel-branches` gain; targets 15 of 18. Package check `bench-20261002T193254Z` (2 cases, K = 8 auto, engine ahead on every trial; no verdict claimed). Review fixes (`95369abd`, the advance the commit path waits for may start while P jobs run): gate `bench-20261002T204445Z` (32 CPUs, confirmed by `bench-20261002T205144Z`), Final win on all 13; against the pre-fix tree neutral on the five cases whose ratios moved most (`bench-20261002T210033Z`; both runs under 10–19% background load).
- SP4a done (measurement and dispatch): B3 frozen (`native/reports/baselines/B3-20261003`, SP3's final tree); CPU per item recorded beside throughput, with A/A floors at auto K (`bench-20261002T233958Z`, `bench-20261002T234603Z`, `bench-20261002T235213Z`) and at K = 1 (`bench-20261002T235813Z`, `bench-20261003T000600Z`, `bench-20261003T001348Z`): the CPU floors are 4–77%, unusable for most cases (Job-object CPU is charged per tick), so CPU per item is information only, and the K = 1 throughput floors are 1.5–8% for image cases; the `CHAINNER_C_PROFILE` timer (0.72 µs per call) and the profile (`native/reports/sp4/profile.md`; K = 1 `bench-20261003T023329Z`, auto `bench-20261003T023709Z`); the ISA probe and `CHAINNER_C_ISA`; the two AVX2 kernels in `separable_avx2.c`/`convolution_avx2.c` (B3 goldens and forced paths pass; the baseline objects of the DLL and pyd are VEX-free, VERIFIED with `isa_check.py`, except P12's dispatched `wmemcmp`, `distance` and fpng's `adler32_sse_16`); gate `bench-20261003T055413Z` (32 CPUs, B3 against `backend/src`, auto K, run on DLL `04dd3d85…0eb7`; the shipped DLL, rebuilt by the polish commit `d7564ce0`, has an identical `.text`, VERIFIED with `isa_check.py text`; pyd unchanged): 13 of 13 neutral, no flagged verdict, no speed-up claimed (`parallel-branches` +4.2%, inside its current floor of 9.4%); package check `bench-20261003T063802Z` (2 cases, `/run` + SSE: B logs `native isa=avx512 (requested auto, cpu avx512)`, A none, no timer table; no verdict claimed).
- SP4b done, Task 8 included: 13 of 18 kernel target rows are met (Task 8's five rows are counted apart: one met, four misses accepted), AVX-512 is kept for both families and the wake policy is unchanged. Plan `docs/superpowers/plans/2026-10-03-sp4b-kernels.md`; local evidence: targets `native/reports/sp4/targets.md`, `native/reports/sp4/task<N>-verdict.md`, final profile `native/reports/sp4/profile-final.md`. One reviewed commit per task, outputs bit-exact against the B3 goldens (Task 3b1 re-based only the −0 cases toward chaiNNer), each task gated against B3 at auto K (ids: 32 CPUs, then 8 CPUs where run). Tasks: 1 conversions, AVX2 unit, pool counters `5ae36362`, `e273c0c5`, `408f65c5` (`bench-20261003T083929Z`); 2 lens convolution and compose `54c36533`, `abd50e92`, `07c1f215` (`bench-20261003T113220Z`; its wall-clock flag on `video-ffv1` is no claim); 1d conversion grain 65536 → 262144 `e0e2df25` (`bench-20261003T134451Z`, confirmed by `bench-20261003T135040Z`; 8 CPUs `bench-20261003T135608Z`, confirmed by `bench-20261003T140346Z`); our counters close that flag: per 720p frame at auto K, wakes 62 → 20 and empty helper wakes 12.1 → 0.0 (`video-ffv1`), 21.0 → 0.1 (`video-h264`), and video wall-clock labels are no claims; 3 separable row tiles without an image-size temporary `fd4f40a8` (`bench-20261003T152633Z`, `bench-20261003T153718Z`); 3b lens finishing store and AVX2 normalized scan, outputs borrowed at enforce `071f3130` (`bench-20261003T171445Z`); 3b1 the enforce clamp gives +0 like the installed chaiNNer `602146a5`; 3b2 Gaussian and Box outputs borrowed, parity tests judged and corrected `6fc95ae3`, `47bb7d54`, `f75800dd` (`bench-20261003T192033Z`); 3c eight accumulator chains `dcc52694` (`bench-20261003T210445Z`, `bench-20261003T211538Z`); 3e lens convolution grain 16384 `5320dbfb` (`bench-20261003T235309Z`, 8 CPUs `bench-20261003T233148Z`; its `palette-median` Final regression was run-level, a paired A/B `bench-20261004T000536Z` read −0.6 %, neutral; harness fix `f8f886a9`); 4 and 4b morphology in place with the scratch pool, then grains `5f51cbbc`, `6c0ce1e7` (`bench-20261004T013456Z`; `bench-20261004T031513Z`, `bench-20261004T032650Z`: `palette-median` `regression (within current A/A)` at 8 CPUs, closed by Task 5's rewrite); 5 and 5b median cut, dither, exact radix select `503fbabd`, `d1016d02`, `b4f6787e` (`bench-20261004T050947Z`, `bench-20261004T052139Z`; `bench-20261004T065303Z`, `bench-20261004T070406Z`); 6 composite `05963492` (`bench-20261004T084802Z`, `bench-20261004T085924Z`); 7 resample `93be69b7` (`bench-20261004T101308Z`, `bench-20261004T102341Z`); 9 AVX-512 `cce5213b`, `e32e10ea` (`bench-20261004T124522Z`, `bench-20261004T125658Z`); 8 tail kernels as row loops `ba332436`, `5857987b`: row loops for `cn_spectral_tile`, `cn_spectral_store`, `cn_spectral_multiply`, `cn_normal_output` and `cn_caption_compose`; AVX2 units kept for the multiply (`spectral_avx2.c`, `spectral_shared.h`) and the normal output (`color_avx2.c`, `color_shared.h`), the tile's dropped (no timer win), caption not vectorized (≈ 157 of 200 µs is first touch of its fresh 2.6 MB output); goldens `native/tests/golden/tail.json` (179 cases from B3), bit-exact at scalar, AVX2 and AVX-512; per call at K = 1, before → after in µs (target): tile 522 → 261 (215), store 194 → 64 (100, met), multiply 108 → 77 (55), normal output 446 → 228 (215), caption compose 454 → 409 (120); the four misses are accepted by the stand-in: in isolation every kernel beats its target, and the pipeline remainder is first touch of fresh buffers; K = 1 loop, ms per item: `normal-map` 13.44 → 11.72 (≤ 12.0, met), `caption` 7.09 → 6.84; K = 1 A/B against B3 `bench-20261005T053336Z`: Final wins `normal-map` +24.1 %, `caption` +14.5 %; gates against B3: 32 CPUs `bench-20261005T053510Z` no Final regression (`normal-map` unconfirmed; `caption` neutral, and a same-session A/B against SP3b's tree found no attributable loss: a watch item), 8 CPUs `bench-20261005T054640Z` 13 of 13 Final wins; five cross-run gate flags on untouched cases were closed by a same-session span A/B (all neutral). Bench harness: `9a5f3908`, `7ce3d872` (storage), `551933ce`, `5c7a55bd` (Task 10a: per-side interpreter, decoded-PNG oracle). Kernels: 13 of 18 target rows met, each by the instrument its task was ruled on (`profile-final.md` section 3); missed `cn_convolution_spatial` +2.4 % and `cn_lens_compose` +12.8 % (inside the 25 % band), `cn_gaussian_f32` +99 % and +54 % (judged on the 1-CPU roofline, 840 → 406 µs), `cn_filter_morphology` +51 % (algorithm, parked); the conversion rows are met in isolation and bound in the pipeline by fresh-output first touch (INFERRED). AVX-512 kept for both families (lens timer 133.1 → 126.5 µs; gaussian at 8 CPUs 376.7 → 356.6 µs, a controller ruling; throughput neutral, no clock-down seen); `CHAINNER_C_ISA=avx2` caps them. Wake policy unchanged (design: ARCHITECTURE section 4): Task 3d's two cascading-wake variants failed (one lost K = 1 throughput on `parallel-branches`; the other left wakes per item unchanged and raised the empty share on `gaussian` and `parallel-branches`) and nothing was committed, so chunk length was the lever (Tasks 1d, 3e, 4b); its 7 policy-agnostic pool tests moved to `native/tests/test_parallel.py` (`31565cce`); empty wakes at auto K remain 43.4 % on `lens-spectral`, 41.1 % on `parallel-branches`, 31.7 % on `morphology`. Package check `bench-20261004T164638Z` (`gaussian`, `video-resize-h264`; B0 against the package, `/run` + SSE): `success=True`, B `native_isa` avx512 (requested auto, cpu avx512), A none, no timer table; no verdict claimed. Suite 106266 passed, 187 warnings before Task 8 (`31565cce` plus the review-fix edits), 106746 passed after it.
- SP3b done (decode off the producer): Load Images items are produced in two phases, a serial `describe()` that names the file and a parallel `materialize(token)` (decode and normalize) that runs as an engine job sharing P with node jobs, oldest slot first; the slot the commit path waits for bypasses P, pause and Stop; every single generator's normalize leaves the producer (video included, no speed target); zipped groups and Range stay serial; `CHAINNER_C_SERIAL_PRODUCER=1` forces the serial producer (developer setting; any other value is ignored). The decoders are unchanged (Pillow stays the JPEG decoder): Pillow's plugin registry is warmed once at Load Image's import, and OpenCV's temp-file decoders (`.hdr .pic .sr .ras .exr`, or their content signatures) take one lock. Opened on the serial producer (72–89% of the auto-K loop in 12 of 13 cases, `native/reports/sp4/profile-final.md` section 5, `bench-20261004T172037Z`, VERIFIED(run)). Spec `docs/superpowers/specs/2026-10-05-sp3b-decode-off-the-producer-design.md` (Outcome: section 9.1), plan `docs/superpowers/plans/2026-10-05-sp3b-decode-off-the-producer.md`; product code final at `5a0a8c18`, tooling fix `3091f190` (`verify_runtime` under generator-once; its rule, broadcasts minus starts for generator nodes, is superseded by the shared rule in the Task 6a line below; `runtime-20261005T025126Z`); local evidence `native/reports/sp3b/` (`task4-verdict.md`, `task7-verdict.md`, profiles). Exact (blocking; all MET): 13 of 13 bit-exact against the installed chaiNNer at 32 CPUs (`bench-20261004T235146Z`, confirmed by `bench-20261005T001901Z`) and at 8 CPUs (`bench-20261005T004626Z`, confirmed by `bench-20261005T011253Z`), the installed app untouched (before/after snapshots: 0 added, removed or changed); across Task 4, 22 runs and 2,496 trials with 0 output and 0 `sse_contract` mismatches; suites `106332 passed` at auto K (`bf618346`) and `106336 passed, 187 warnings` at K = 1 (`CHAINNER_C_ITEM_WINDOW=1`, `5a0a8c18`, native and backend tests). Gates against B3, Final `win` on all 13 cases at both counts, confirmation-run changes: 32 CPUs `bench-20261004T232346Z`, confirmed by `bench-20261004T232937Z`: `resize-real-256` +46.5 %, `gaussian` +18.0, `lens-spectral` +238.3, `morphology` +18.3, `normal-map` +9.9, `caption` +16.5, `split-channels` +20.0, `palette-median` +58.6, `palette-dither` +85.6, `parallel-branches` +27.6, `video-ffv1` +20.8, `video-h264` +13.6, `video-resize-h264` +12.6; 8 CPUs `bench-20261004T233546Z`, confirmed by `bench-20261004T234315Z`: `resize-real-256` +31.7 %, `gaussian` +47.0, `lens-spectral` +1124.9, `morphology` +69.2, `normal-map` +19.9, `caption` +16.9, `split-channels` +48.4, `palette-median` +72.4, `palette-dither` +118.2, `parallel-branches` +187.0, `video-ffv1` +116.0, `video-h264` +60.5, `video-resize-h264` +67.4 (video: no speed claim). Against the installed chaiNNer (10a, 32 CPUs, confirmation): ×10.8 (`lens-spectral`) to ×85.4 (`normal-map`); `resize-real-256` 4.2 → 306.1 img/s. SP3b's own gain against SP4b's final tree, auto K, confirmation figures: 32 CPUs (`bench-20261004T230511Z`, `bench-20261004T231046Z`) `resize-real-256` +19.3 %, `palette-median` +11.1 %, `parallel-branches` −7.6 % (within the A/A floor; accepted by Fable); 8 CPUs (`bench-20261005T013954Z`, `bench-20261005T014606Z`) `resize-real-256` +36.3 %, `split-channels` +26.9, `palette-median` +14.4, `gaussian` +12.8, `caption` +8.5, no regression; K = 1 (`bench-20261004T231559Z`) within noise everywhere. Same binary, two-phase against `CHAINNER_C_SERIAL_PRODUCER=1`: 32 CPUs `resize-real-256` +24.4 %, `split-channels` +10.7, `caption` +9.9 (`bench-20261004T224601Z`, `bench-20261004T225136Z`); 8 CPUs `resize-real-256` +28.4, `split-channels` +15.2, `palette-median` +42.1 (`bench-20261004T225347Z`, `bench-20261004T225955Z`). Missed (spec section 9 win targets; 32 CPUs: `resize-real-256` ≥ +40 %, `gaussian`, `caption`, `split-channels`, `palette-median` ≥ +25 %; 8 CPUs: `resize-real-256`, `split-channels`, `palette-median` ≥ +20 %): all five at 32 CPUs (`gaussian` +7.5 % and `split-channels` +2.1 %, both unconfirmed; `caption` neutral) and `palette-median` at 8 CPUs; waived as blocking by Fable, who records that the targets were mis-derived (they used the producer's busy share, but SP4b's window had already overlapped much of it, and they allowed no commit-path inflation). Measured cause (Task 7, `native/reports/sp3b/task7-verdict.md` section 6): with decoding off the producer, each item's time is set by Save's commit path, which handles one item at a time: it is 91–97 % of every image item and waiting for the producer 3–9 %; on `resize-real-256` Save's write and glue take 49 % and the hand-offs between the event-loop thread and pool threads 46 % (Save's lazy-input hop 21 %, the steps around Save's job 25 %), and the loop thread is busy under half the time, so the cost is latency, not loop CPU; decodes running beside the commit path slow Save's write by ≈ 0.4 ms per item, which cancels what moving Save's encode off the commit path saved. D2's 30 % clause: not tripped (+5.9 % against the same-session serial setting); the cap sweep is waived. Phase 2 (GIL-free copy out of Pillow's buffer): the trigger fired as written (2.656 ms; 1.513; 0.676 ms) and the build is deferred, since decode is off the critical path; revival rule, re-anchored to the re-profile after SP4c (SP3c closed without a lever): run phase 2 if `resize-real-256`'s auto-K loop is still above 2.0 ms per item and Save's like-for-like inflation is still ≥ 0.3 ms per item (early read: 0.380 ms, loop 2.99–3.11 ms). Profile-only timers added: `window.take_advance`, `lazy.resolve`, `loop.thread_cpu`.
- SP3c closed after its probe (2026-10-05, no code shipped): the commit path, SP3b's measured bound. Map: one image item's commit path is 3 + 3 loop/pool transitions and 22 loop callbacks; its idle cost is ≈ 0.19 ms against 1.44 ms under load (contention-amplified waiting). Probe (the stand-in's pre-registered rules; 14 runs, outputs and `sse_contract` identical in all): dropping the empty round trip for replays on the item's single chain (B1) gained 0.114 ms per item on `resize-real-256` at 32 CPUs (bench: neutral; the saving reappeared in Save's write); adding the NodeFlights/shield changes (C1+C2) gained 0.043 ms more; Save's pre-read on the loop (A1) failed its gate (0.057 ms per removed transition, gate 0.15) and its loop-side check costs 118–162 µs median. No lever produced a confirmed win, so SP3c closed: B1, C2, D (describe on the loop) and F (previews on the loop) are parked, C1 was not built and A1 is rejected. Evidence: `native/reports/sp3c/` (commit-path map, design questions, probe prep, probe verdict, arm diffs). Next (owner's order, 2026-10-05; SP4c is done, see the SP4c line for what follows): SP4c, output allocation (first touch of fresh buffers), allocator-level only (a bounded NumPy memory handler that keeps freed blocks warm, or pre-faulting); no per-bridge buffer reuse; per-case fresh-bytes and page-fault count before targets; a byte cap with a long-video growth test. SP3b's phase-2 revival rule is re-anchored to the re-profile after SP4c, and B1 is re-tested after it. Then SP5a (clang-cl LLVM 23.1.2 against MSVC first; Pending).
- SP4c done (output allocation, 2026-10-06): a NumPy data-memory pool in `_chainner_graph` (`native/src/numpy_pool.cpp`; ARCHITECTURE section 6; `CHAINNER_C_NUMPY_POOL` in README Bench) keeps freed blocks of 512 KiB or more warm (best fit in [size, 2 × size], idle bytes capped at min(512 MiB, 10 % of RAM)), is installed per thread and releases every idle block at the end of each run; D8 reads Load Video's frames into fresh pooled NumPy buffers. Spec `docs/superpowers/specs/2026-10-05-sp4c-output-allocation-design.md` (Outcome 2026-10-06), plan `docs/superpowers/plans/2026-10-05-sp4c-output-allocation.md`; local evidence `native/reports/sp4c/` (`task4-verdict.md`, whose D2 addendum is the record for the shipped tree, and `task5-verdict.md`). Commits: pool `f9f14413`, per-thread install and release `053b8bd2`, growth tools `a9bffbcd`, threshold 512 KiB `11de844f` (test pin `097a73a9`), D8 `4cceb734` (test fix `14176509`). Threshold: 1 MiB first, which missed `lens-spectral` at 8 CPUs (+8.9 % neutral, `bench-20261005T150927Z`); D2's retry at 512 KiB (Fable) was kept on its four pre-committed parts. Run ids are `bench-20261005T<id>`. Stage 1 (8 CPUs): profile `194053Z`, timed `194400Z` + confirmation `195006Z`. Stage 2: growth `run-20261005T195454Z`, profiles `195635Z` (K = 1) and `200104Z` (auto, 32 CPUs), timed `200354Z` + `200920Z`, gates against the pre-SP4c tree and B3 `201415Z` + `201944Z`, `202426Z` + `203031Z`, `203409Z` + `203957Z`, `204523Z` + `205055Z`, `205643Z` + `210420Z`, end to end against the installed chaiNNer `211230Z` + `214025Z`, `221603Z` + `224330Z`. Confirmed Final wins, same-tree A/B (A = pool off, auto K), confirmation-run changes: 32 CPUs `normal-map` +17.6 % and `parallel-branches` +16.5 % (`200354Z`, `200920Z`); 8 CPUs `lens-spectral` +14.8 % and `normal-map` +14.8 % (`194400Z`, `195006Z`). `resize-real-256` at 32 CPUs missed: +3.2 % neutral (`200354Z`; noise floor +4.8 %), waived by Fable as non-blocking (the mechanism's ceiling, ≈ 4.6–5.1 % INFERRED, sits at the noise floor; the 1 MiB readings were +6.8 % and +4.6 %; faults 543 → 23 per item at K = 1). Video, on our K = 1 per-frame loop timer at 600 frames, never wall-clock (`195635Z`): `video-ffv1` −24.5 % (4.206 → 3.178 ms per frame), `video-h264` −21.1 % (4.622 → 3.646; the 1 MiB tree's clean third run `193901Z` read −23.4 %), `video-resize-h264` −26.1 % (4.688 → 3.464); the targets (16, 17 and 17 %) are met. D8 built and kept (Fable), against the pre-D8 tree with the pool on (`bench-20261006T001348Z`, K = 1, 4 pairs, 600 frames): faults per frame −683 / −680 / −681, loop timer −10.7 % (3.281 → 2.929) / −5.1 % (3.789 → 3.594) / −6.8 % (3.619 → 3.372), every pair faster; identity `bench-20261006T001723Z` + confirmation `bench-20261006T003201Z`. D8 has no runtime switch (`CHAINNER_C_NUMPY_POOL=0` does not undo it): its undo is `git revert 14176509 4cceb734`; parked minor: one extra frame-sized pool request per stream at end of stream. Identity of the shipped pool tree: 13 of 13 graphs against the installed chaiNNer (decoded-PNG compare) and against B3 (SHA-256) at 32 and 8 CPUs, `sse_contract` identical in all 22 runs, no confirmed Final regression at either count; suites `106813 passed, 2 skipped` (auto K and K = 1), `106821 passed, 2 skipped` with D8. Growth, all PASS: `run-20261005T195454Z` (4 of 4) and, on the shipped D8 tree, `run-20261006T005241Z` (4 of 4; video pool requests 12,001 = 4 per frame + 1; check (e) +0.3 MiB after the next release). Check (e) was reported, not asserted, at SP4c's close: a failed `/run`'s 64 MiB array stayed live after the response (+64.1 MiB) in reference-cycle garbage, a deviation present since B0 and independent of the pool (`native/reports/error-retention/rootcause.md`). It was fixed on 2026-10-06 by `bad1dba7` and `084d640c` (Pending, first row; `native/reports/error-retention/verdict.md`); since then (e) and the new (f) are asserted (`run-20261006T042213Z`, 4 of 4). Activation (2026-10-06, refreshed at `14176509`): `refreshed`, repeat run `verified_no_op` (manifest `009a7ef5…` unchanged), both shipped binaries hash-equal in `backend/src` and the package, and the four runtime verifiers green in port mode: `runtime-20261006T020521Z` success, `io-execution-runtime-20261006T020715Z` 92/92, `utility-runtime-20261006T020810Z` 116/116, `video-runtime-20261006T020302Z` 26/26 (65 frame PNGs byte-different, pixel-identical: D8's path). Next, in order (Fable, 2026-10-06): (1) the error-retention fix (done, above); (2) the B1 re-test and SP3b's phase-2 re-check; (3) SP5a (clang-cl against MSVC).
- Parity verifiers (2026-10-06, Task 6a `62e3859c` and `8679d81a`, ruled by Fable): `io_execution_runtime`, `utility_runtime` and `video_runtime` test the package only with `--include-port` (without it they run the installed backend alone), and in port mode they compare under chaiNNer-C's three accepted divergences from the installed chaiNNer: generator-once (upstream re-runs a generator node per item, one extra node-start and node-broadcast each; the port runs it once and starts it exactly once), fpng PNG bytes (PNGs are compared by decoded pixels) and preview coalescing (`LatestBroadcasts`: per node, the port's broadcasts are a sub-multiset of the installed app's with no order compared, the final state is equal by `bench_oracle.sse_contract`'s merge, everything else exact). The rule lives in `verify_runtime.sse_event_mismatches`, shared by every verifier (`verify_framework_runtime` adopts it through `compare_runs`, UNTESTED: it loads models). Limit: a dropped restoration fails the rule only when it is the sole source of an output id; `bench_oracle.validate_record` and `test_execution_scheduler.py` pin the rest.
- SP1 acceptance (2026-10-02, plan Task 9): a fresh package equals v1 byte for byte except manifest, profile and `__pycache__` (55,028 files, `native/reports/sp1/`); the real package was converted in place to manifest v2 (sha256 `a2c68ce3…`), repeat run `verified_no_op`; suites pass on the single backend, 4684 + 95409 + 8; bench B0 against the new package `bench-20261001T220537Z`: no regression.
- TensorRT video chain (owner, 2026-10-08, 1 h budget): `tensorrt-video.chn` (375 frames of 4K HEVC, 960x540
  into a 4x compact engine) 9.9 -> 19.8 fps (`f985ff8a`: native tile conversion, no blend for one tile); then
  `e13ebdef`, `cn_planar_to_interleaved_f32` (exact, all ISA levels): the node 75 -> 23 ms per frame, frames identical
  (framemd5, 375). The chain stays at 19-20 fps: the video read path bounds it (Load Video + one Resize alone 26.7
  fps; ffmpeg decodes and converts at 132 fps to a file, the 25 MB-per-frame pipe into Python is the limit). By the
  owner's rule the work stopped there. D:\chaiNNer rebuilt from chaiNNer-C's own UI (`npm run make`) at `e13ebdef`
  with the local PGO icelake binaries; the previous build is `D:\chaiNNer-prev-20261008`.
- In-process video decoding (owner, 2026-10-08, 1 h): Load Video reads through PyAV 19.0.1 (FFmpeg 9.0.2 in-process,
  the same libswscale conversion to bgr24), the FFmpeg CLI pipe reader is the fallback (rotated or variable-rate files,
  PyAV missing, open errors; `CHAINNER_C_VIDEO_READER=cli` forces it). Read path 26.7 -> 68.1 fps; `tensorrt-video.chn`
  19-20 -> 23.5 fps; frames identical (375 on the owner's 4K HEVC clip, `test_video_pyav` on h264/ffv1/mjpeg/bt601).
- Rotated videos (`ba3ebc86`): metadata reports the displayed size (a quarter-turn display matrix swaps width and
  height) and the PyAV reader inserts the CLI's autorotate filters; both readers give identical, upright frames.
- Chain limit (2026-10-08): per 4K frame, CPU ms (wall ms) alone: decode + conversion 140 (9.3), uint8 -> float32
  87 (7.3), Resize 105 (4.9), TensorRT node 168 (17.2), Resize 64 (5.1); in the chain about 0.95 CPU-s per frame,
  25 of 32 threads busy. Window size (K 4/8/12) and a 4 GiB NumPy-pool cap are neutral. The item window runs the
  TensorRT node ahead (`d59b4b1d`, `execution_scheduler.LOCKED_DEVICE_NODES`, session under its lock; owner's call
  after the probe missed its +20 % keep rule): 23.5-24.5 -> 26.6-26.8 fps, frames identical. With identical
  output the chain is throughput-bound at about 27 fps on this PC.
- Spandrel fork (owner, 2026-10-08): the PyTorch package's Spandrel is `0.4.2+c1`, the wheel on
  github.com/Rift73/spandrel (release `v0.4.2-c1`, SHA-256 `fe884dc1…`): upstream 0.4.2 plus DRFT, DUAL and
  SST/SST-Real. `Dependency.url` installs a wheel instead of the PyPI release; a version with a local label is
  satisfied only by that exact build, so existing installs switch at the next start. The lock pins the wheel as
  pip freeze writes it (`read_lock` takes the wheel's version). The local overlay is retired by the owner (only the
  three architectures are used); the repo runtime and D:\chaiNNer carry the fork, D:\chaiNNer's old spandrel is
  in `D:\chaiNNer-spandrel-backup-20261008`.
- DUAL to TensorRT (owner, 2026-10-08, 1.5 h + 2 h): traiNNer-redux's DUAL exporter is vendored in
  `backend/src/vendor/dual_tensorrt` (canonical `dual_arch.py` SHA `a7c579c5…`, export/graph/spec/kernels, its
  contract tests and `build.py` for them; its guide is the README). Convert To ONNX gives a DUAL preset a
  fixed-size TensorRT ONNX (CPU child process, the guide's C++ plugin nodes). Build Engine runs
  `dual_engine_worker.py` in a child process: `dual_aot.lower` swaps the plugin nodes for stand-in operators
  (TensorRT 11.2's ONNX parser mixes up Python plugin instances), `attach` adds the plugins as AOT Python plugins
  (`tensorrt.plugin`): the vendored Triton kernels plus Triton Norm and barrier, Core split into three, compiled to
  PTX by triton-windows 3.5.1.post22 (a TensorRT-package dependency) for the GPU and embedded in the engine; the
  guide's builder policy. No C++ compiler, CUDA Toolkit or TensorRT SDK. AOT plugins cost ~0.3 ms host time per
  launch, so TensorRTSession replays fixed-shape engines from a CUDA graph (Light x4 256x256 per tile: 48 ms plain,
  9.0 ms replayed; C++ plugins 9.2 / 8.0 ms); through the session 11.6 vs 10.9 ms (256x256) and 72.7 vs 72.0 ms
  (540x300) against the C++ plugins. Gate 6 (fresh process without plugin registration vs folded PyTorch FP32):
  Light x4 (owner checkpoint) and XS x4 (synthetic, nonzero experts) at 256x256 pass on all six inputs, error at or
  under the BF16 floor, max abs <= 0.0116. The C++ plugin route (CMake + nvcc + MSVC, vendored TensorRT headers,
  TensorRT SDK Path) was removed in the commit after `cb735a43`; reverting that commit restores it (Fable ruling).
  TensorRT loads embedded plugins without `engine_host_code_allowed`, so no trust gate.
- DUAL speed (owner, 2026-10-09): Light x4 512x512 was ~2x the 2026-10-04 DUAL3 PU2 D16 engine (14.33 ms). Per-layer
  profiles: (1) AOT kernels compiled without Triton's 16-byte pointer-alignment hint (scalar loads; core 2.13 vs
  0.156 ms per layer), fixed in `5ca20024`; (2) the exporter moved every 3x3 after a PixelShuffle ahead of it (4x work
  for Light's 64->256 tail conv), now only the RGB readout, `c0de1572` (vendored `graph.py` now differs from
  traiNNer's). Through chaiNNer-C's build, GPU-only replay: 23.93 -> 15.64 ms; gate 6 passes at 512x512, errors
  unchanged. The remaining ~1.3 ms vs DUAL3 is DUAL3X's dynamic convolutions. Owner-postponed: dynamic-shape DUAL
  engines (ask before starting).
- DUAL dynamic engines (owner 2026-10-09, 2 h box; branch `dual-dynamic`, WIP): dynamic plugins (`e146bc73`,
  bit-identical to the fixed-size kernels) and exporter + worker profile (`7f9140a2`). Light x4, one engine
  64x64..512x1024: gate passes at 7 sizes; +11 % vs static at 512x512. Limit: above 512x1024 TensorRT has no kernel
  for the last tail conv in a dynamic build (static 576x1024 builds; a row-split try failed). Not wired into the
  nodes or TensorRTSession (per-shape CUDA graphs) yet. Waiting on the owner: continue or drop the branch.
  Update (owner chose to continue after Codex's finding): `8f7f3d18` lifts the limit (each tail conv's input
  reshaped to sizes computed from the network input; Codex) — one engine 64x64..1088x1920 passes the gate at 8
  sizes, +3.5 % vs static at 1088x1920; `7a590eba` wires it in (Convert To ONNX "TensorRT Shape", default Dynamic;
  Build Engine profile checked against the 64-px alignment; TensorRTSession: a CUDA graph per input shape, up to 4).
  Through chaiNNer's tiler: 1080p in one tile, 197 ms per image; engine vs PyTorch on the same padded input within
  the BF16 floor. Merged into `chaiNNer-C` and copied to D:\chaiNNer (owner, 2026-10-09).
  `009e9d80`: dynamic DUAL ONNX takes any multiple of 4 (input named `input_aligned_4`, the tiler pads to it),
  so 1080p runs unpadded and matches whole-image PyTorch (max 0.0022); +3.4 % vs alignment 64 at mid sizes,
  equal at 1088x1920. Session captures a shape's CUDA graph on its second run. One unexplained dynamic build
  failure in five attempts (no log kept); rebuilding worked.
  Settings that do not apply to DUAL are hidden (owner, 2026-10-09): Convert To ONNX's Data Type, Opset and
  Verify, Build Engine's Precision. Since 2026-10-16 (owner) Build Engine's Workspace (GiB, any amount; 0 =
  automatic: TensorRT's default, up to the GPU's memory, since 2026-10-17 when 4 GiB failed an FDAT 2048x2048
  build by 8 MB; 8 for DUAL), Allow TF32, Sparse Weights and Optimization Level (0-5, default 3) apply to every
  ONNX, DUAL's worker included (its defaults stay the guide's: 8 GiB, TF32 off, dense, level 3). Build Engine's
  Tiling Optimization (2026-10-18; owner 2026-10-19: keep it, for DUAL's worker too; TensorRT's L2-cache tiling,
  None/Fast/Moderate/Full, default None): 256x256, level 3, FP16, on the graphs before spandrel c5 DAT2 57.72 ->
  51.79 ms with Full (build 56 -> 301 s), HAT-L 89.20 -> 83.36 ms with Moderate; on c5's (attention laid out for
  TensorRT) Full gains little, DAT2 47.40 -> 46.62 ms (build 42 -> 195 s), HAT-L 71.80 -> 71.42 ms (53 -> 221 s).
  Save Engine's
  Compress (owner, 2026-10-16; default on) writes LZMA, or Zstandard level 19 (Compression, owner 2026-10-17),
  and appends "_compressed" to the name; Load Engine and VapourSynth's AI Upscale decompress by the xz or zstd
  header (`nodes/impl/compressed_file.py`). Measured on the owner's 1x DUAL Light engine: 47.5 MB -> 11.7 MB,
  +0.44 s per load, 6.6 s per save; zstd 13.2-13.6 MB at 35-60 ms. VOSR's FP8 ONNX weights (2026-10-17, 128 MB
  samples) compress to 71-83 % with LZMA, 78-87 % with zstd 19. ONNX Save Model has the same Compress and
  Compression (owner 2026-10-17; default on, LZMA); Load Model (and Load Models, AI Upscale) reads both forms.
  Eight large ONNX upscalers became 38-89 % of their size (HAT-L 47 %). A DUAL ONNX loads as precision "mixed" (arch DUAL in
  its type); every DUAL engine, fixed ones too, names its input `input_aligned_4`, which Load Engine reads as
  "mixed". Fixed engines built before this still read as FP32. The fork's backend-tests check failed since the
  dynamic push (the ONNX-load assertion needs the native build, which GitHub does not make); it now skips there.
- Upstream bug fixes (2026-10-09, branch `fix/upstream-issues`, decisions D1-D15 by the owner's stand-in in
  `native/reports/fable-decisions.md`): 70 open upstream bug and unlabelled issues plus 82 enhancement issues triaged
  against chaiNNer-C (reports in the session scratchpad), fixed in about 60 commits, worst first. Highlights: a fresh
  install starts again (NumPy with the server packages; a missing or broken torch costs only the PyTorch nodes;
  verified with a real first start of an `npm run make` build, where v0.3.2 dies on `numpy`); Convert To ONNX/NCNN
  export with `dynamo=False` (torch 2.14 needed onnxscript); NCNN return codes checked; Save Video audio on chaiNNer's
  FFmpeg, never silent; Save/Load Video colours by size with full tags; ONNX tile padding found by running the model;
  EXIF orientation; straight-alpha and grey+alpha TIFFs; TIFF ExtraSamples; lossless WebP `exact`; GIF palette per
  frame; CPU-only NCNN tiles by available RAM; a vitest runs the app's schema check over the real node list; CLI ends its
  backend; system-Python option removed; Python download honours the proxy; pip's errors reach the UI; Spandrel
  `0.4.2+c7` (fork release `v0.4.2-c7`, SHA-256 `030ba1a2…`; c2: fp32 LayerNorm statistics in fp16 for 11 archs, ONNX
  padding per input for 12; c3, 2026-10-16: DUAL runs in fp16, its last group's dynamic conv in fp32, 1x Light 512²
  343 → 199 ms, 2x 87 → 52 ms vs fp32+TF32; c4, 2026-10-18: HAT and DAT run in fp16, the last group's closing conv
  and the final norm in fp32 (trained 4x HAT-L writes 722k there, DAT2 121k), 70-71 dB from fp32, 256² HAT-L 476 → 250
  ms, DAT2 313 → 229 ms; c5, 2026-10-18: DAT2 fp16 256² 217 → 103 ms in PyTorch, channels-last windows, shift masks on
  the device; Convert To ONNX pads HAT heads to 8 and freezes HAT/DAT position biases, TensorRT 256² fp16 level 3
  HAT-L 89.20 → 71.56 ms, DAT2 57.72 → 47.57 ms; c6, 2026-10-19: window attention through
  scaled_dot_product_attention with heads padded to 8 (30-wide heads get no fused kernel), explicit in ONNX
  exports (SDPA's summed bias+mask cost HAT-L 72 → 87 ms in TensorRT); PyTorch fp16 256² cuDNN first HAT-L
  246.1 → 191.9 ms, DAT2 102.8 → 96.4 ms; Convert To ONNX also pads DAT's qkv heads, TensorRT DAT2 47.40 → 44.51
  ms, all 36 spatial and 84 HAT attention calls in TensorRT's fused kernel; c7, 2026-10-19: DRFT's edge-branch
  fold without a per-channel Python loop, bit-identical, CPU fold of a trained 4x DRFT-XL 2447 → 1652 ms, 4x
  DRFT-M 628 → 402 ms). PyTorch tries cuDNN attention first (owner, 2026-10-16: every model in
  fp16, not DUAL only; DUAL fp16 197.5 → 182.0 ms 1x, 52.0 → 47.0 ms 2x; Windows PyTorch has no Flash Attention).
  With the fp16 setting on, DUAL TensorRT exports start from fp16-rounded weights (68.5–88.6 dB vs exact, below the
  engine's bf16 noise; owner accepted). Deviations: ARCHITECTURE section 7.

## Last CPU comparison

2026-10-19 (instant-exec, `f4d1a58f` against `fffe9cf7`, `bench-20261019T080356Z` memo off and
`bench-20261019T080808Z` memo on): State. The comparison below is SP4b's.

2026-10-04 (UTC), SP4b Task 10 Step 3, B0 against `backend/src` (DLL `bb0fd7aa…6fee3`), auto K (K = 8 at 32 CPUs, K = 6 at 8 CPUs, affinity `0xFF` = 4 cores × 2 SMT threads): 32 CPUs `bench-20261004T164739Z` (confirmed by `bench-20261004T165415Z`), 8 CPUs `bench-20261004T170057Z` (confirmed by `bench-20261004T170910Z`); outputs identical in every trial, Final `win` on all 13 cases at both counts. Background load was 10–28 %, above most floors' recorded load (warned in the summaries), so read the magnitudes as indicative. Speedup = confirmation-run median paired ratio (> 1 = faster, the gains of SP2 to SP4b included), 32 / 8 CPUs:
`resize-real-256` 1.66/1.58, `gaussian` 2.13/2.41, `lens-spectral` 5.95/13.87, `morphology` 3.57/3.12, `normal-map` 2.70/2.47, `caption` 1.72/1.72, `split-channels` 1.82/1.64, `palette-median` 4.76/4.65, `palette-dither` 7.53/6.42, `parallel-branches` 4.74/5.39.
SP3's goals (SP3 spec section 1), image cases: 3 of 6 at 32 CPUs (≥ 2.0×; `resize-real-256`, `caption` and `split-channels` stay under, producer-bound then; SP3b took decode off the producer, and the commit path is now the bound, State) and 6 of 6 at 8 CPUs (≥ 1.4×).
Video cases are read by our per-frame timer, never by wall-clock (FFmpeg dominates; their Final `win` is no speed claim). Conversions per frame at K = 1, SP4a → final (`bench-20261004T171716Z`): `video-ffv1` 2.85 → 0.85 ms, `video-h264` 3.02 → 1.30, `video-resize-h264` 2.06 → 0.88 with resample 0.93 → 0.67; the pipeline targets (≤ 0.65 / 0.65 / 0.5 / 0.3) are missed, while the isolated kernels meet them (58.3 + 82.3 µs per frame pair, 239.3 µs for resample).
K = 1 loops, ms per item, SP4a → final (`bench-20261004T171716Z`, a profile run under 22 % background load) against `targets.md` section 3: met `lens-spectral` 59.58 → 17.80 (≤ 24.0), `parallel-branches` 13.91 → 8.30 (≤ 8.5), `morphology` 18.06 → 11.47 (≤ 11.5), `palette-median` 19.03 → 7.08 (≤ 11.5), `palette-dither` 33.94 → 15.62 (≤ 22.0); missed `gaussian` 8.30 → 6.36 (≤ 6.0; 5.36 at Task 3c) and `normal-map` 13.19 → 12.45 (≤ 12.0; 11.47 at Task 1; Task 8 meets it: 13.44 → 11.72 in its own run), session load INFERRED; producer-bound `resize-real-256` 5.98 → 4.73 (≈ 5.0 reached), `caption` 7.33 → 6.84 (≈ 6.3) and `split-channels` 6.34 → 6.84 (≈ 5.4) not reached.
End to end against the installed chaiNNer (Task 10a, `551933ce`; `5c7a55bd` adds colour-chunk comparison and the install block, and neither encoder writes such chunks, so the runs stand): A = its `resources\src` under `AppData\Roaming\chaiNNer`'s Python (read-only, unchanged), B = `backend/src`, PNGs by decoded pixels, auto K. 13 of 13 graphs are output-identical, 28 of 28 trials each, at 32 CPUs (`bench-20261004T144028Z`, confirmed by `bench-20261004T150940Z`) and at 8 CPUs (`bench-20261004T153852Z`, confirmed by `bench-20261004T160551Z`); Final `win` on all 13.
Image cases, port against installed app, confirmation-run speedup at 32 / 8 CPUs: `resize-real-256` 66.5×/65.2×, `gaussian` 26.9×/26.5×, `lens-spectral` 10.9×/11.8×, `morphology` 20.3×/15.3×, `normal-map` 77.1×/88.0×, `caption` 22.5×/26.8×, `split-channels` 23.2×/25.3×, `palette-median` 52.1×/50.8×, `palette-dither` 37.1×/39.2×, `parallel-branches` 20.5×/23.7×; the three video cases are exact, with no speed claim. SP3b re-ran this comparison (State).

## Pending

| Item | Plan |
| --- | --- |
| The error-retention fix's two recorded follow-ups, both INFERRED: a failed `/run/individual` (a preview's inputs) and the per-item errors of a non-fail-fast generator (`process.py:903` keeps `str(e)`; the run then fails, so B collects at its end). Each lets an exception through the same frames as a failed `/run`; A breaks their chaiNNer-C edge, and upstream's `Lazy` edge remains. One later commit (Fable's final review) carries `await_owned_node`'s cancel-branch `error` local, which keeps a cycle if a node fails while a cancel waits. That is reachable, because Sanic cancels the handler when the UI drops `/run`; the fix is `del error` with a cancel-branch test. The same commit carries `owned_gather`'s `results`/`tasks` locals and two comment fixes in `server.py`. Upstream's `Lazy.from_coroutine` `completion` local needs a separate decision. Root cause: `native/reports/error-retention/rootcause.md` | The fix (Fable, 2026-10-06) is two commits: A, `bad1dba7` (`await_owned_node` and `NodeFlights.run` drop their `future`/`task` locals), then B with D (`collect_failed_run`: one `gc.collect()` after a failed or stopped `/run`'s handler returns, then the pool's release if no run executes; ARCHITECTURE section 6). A failed run's arrays are freed right after it, earlier than upstream git's end of the next `/run`; successful runs never collect. Growth record: the release check (0)–(f), with (e) and (f) asserted; verdict `native/reports/error-retention/verdict.md`. OWNER REVIEW FLAG (decision log): `import gc` and that one collection; undo = `git revert` of the second commit. The B1 re-test and SP3b's phase-2 re-check follow it, then SP5a |
| `lens_blur` builds its kernels on every call (B0's unused `_lens_plans` cache went in SP4b Task 3b); that cost is still not measured; SP3b did not take it up. SP3 follow-ups: bench cases for a resumed batch with skip-existing (a second run over the same output folder; harness) and for a heavy untaken Conditional branch; a K spot-check at 4 CPUs (affinity `0xF`); count the replay round trips, then skip the round trip when a replay has no concurrent sibling; compare folder identity in the read-ahead predicate (would close the loopback-alias and non-Windows case-insensitivity residuals). SP2 native-pool carry-overs, deferred (SP3 kept the pool unchanged): the spurious WakeAll on re-pin, the `leave` comment, the `cn_parallel_dispatches` non-Windows note, measuring whether dropping the generation bump in `leave` pays. Backend and test cleanups: dead `sequence` branch in `nodes/impl/native_layout.py:154-174` (Stack Images Sequence does not ship); `# pyright: ignore` in `nodes/impl/event_queue.py`; `native_analysis.is_grayscale` has no shipped caller (retire it or test it against numpy); split the 3605-line `native/tests/test_item_window.py` by concern (read-ahead predicate, engine and executor, Save phases, fresh interpreter) | SP4b, SP3b, SP3c and SP4c are closed (State). Order (Fable, 2026-10-06): the error-retention fix (done, first row), then the B1 re-test and SP3b's phase-2 re-check, then SP5a (State, SP4c; rows below). A commit-path lever (B1 is re-tested after the error-retention fix) stays inside SP3c's boundary: outputs bit-identical to the installed chaiNNer and B3; file writes in item order, each complete before its node's finish event and before the next item's commit; events and `sse_contract` identical; the same error object, selected and surfaced at the same point relative to earlier and later items' side effects; Stop, pause and generator-once unchanged; K = 1 within noise. SP3 follow-ups and SP2 carry-overs: not yet scheduled (the SP4 attribution is done); cleanups SP4 or later |
| SP4b parked levers, sizes in `native/reports/sp4/profile-final.md` section 7: output allocation (SP4c, done, State; first touch costs 0.15–0.20 µs per page, 88–117 µs per fresh 2.36 MB output; a 'clamped-by-us' registry mark would also skip the borrow's scan and freeze, 41–65 and 26–33 µs per output); resample horizontal SIMD (≈ 200 of ≈ 220 µs per call at 1 CPU, INFERRED); resample grain (720p: 13.6 of 31 wakes empty per call) and morphology grains (`cn_filter_morphology` runs 9 chunks per dispatch; Task 4b's line-grain variants were neutral or cost 30–35 % more CPU); `cn_filter_morphology`'s algorithm (2114 µs against ≤ 1400; erode ≈ 23× its serial bound); lens convolution grain (K = 1 per call 85.7 µs at Task 3c, ≈ 123 µs now, INFERRED from Task 3e's grain 16384; a count-dependent grain is the candidate); the commit path (SP3c, closed, State): 91–97 % of every image item, by measurement the largest lever on 6 of 10 image graphs, yet SP3c's probe confirmed no win | SP4c took output allocation (State); the others stay parked, chosen by measured size, each with its own paired A/B; phase 2's revival rule (State, SP3b) is applied at the re-profile after SP4c (the re-check follows the error-retention fix) |
| SP5a-c (owner directive 2026-10-06, Consult 9): the product build is clang-cl 23.1.2 only, `-march=icelake-server` with ThinLTO and no PGO, `/fp:strict`, `/Brepro` (ARCHITECTURE section 3); MSVC and gcc left the build, and a startup check refuses a CPU below Ice Lake-SP / Zen 4. The A/B against MSVC was replaced by the switch. `isa_check.py`'s instruction-set checks (`objects`, `dll`) are retired, since every unit is Ice Lake-SP code and the ThinLTO objects are bitcode; its `text` comparison stays. Corrected 2026-10-07 (owner): baseline x86-64 with runtime AVX2/AVX-512 dispatch, no CPU refusal, `-march` local-only (State) | The old-vs-new bench (CPython 3.11 + MSVC against 3.14 + clang) is the only timing; `-mprefer-vector-width=512` is an optional measured A/B after it; CCD affinity stays optional |
| Known third-party diagnostics (Consult 12 D-20): the clang build prints vendored fpng's two `-Wunused-function` warnings, `fpng.cpp:113` `simple_swap64` and `:124` `swap64` (fpng builds at its vendor `/W3`, outside `/WX`). A vendored file is patched only for an output or safety defect, as a recorded patch with its new SHA-256 in that component's `PROVENANCE.md`, never for a diagnostic, and no suppression flag is added | Standing rule |
| NaN-bit robustness (Consult 12 D-21): `cn_spectral_multiply` stays as is (equal to OpenCV 5.0.0 at every level; its NaN bits follow operand order). Any codegen-affecting toolchain or flag change (an LLVM bump, `-mprefer-vector-width=512`, PGO) re-runs `test_packed_spectrum_all_parities` and the NaN goldens (`test_golden_kernels` tail and dither, `test_forced_paths`) at all three `CHAINNER_C_ISA` levels; a failure is fixed in the integer shape of `normal_nan_outputs` (`color_shared.h`), never by a flag | Standing rule; SP5b (PGO) applies it before landing |
| Isolated CMake outputs: SP4a added `CHAINNER_C_OUTPUT_DIR` (default unchanged: every default configuration still writes the DLL/PYD into `backend/src/nodes/impl`) and `CHAINNER_C_PDB`; reproducible default build from a clean clone; bench harness: the media root `F:\chaiNNer-C-diagnostics` is still hard-coded (`bench.py:245`), and the pyright errors at `bench_fixtures.py:84` and `image_utils.py:346` predate SP4b (found in Tasks 10a and 3b2). Closed: comparing the static CRT with `ucrtbase.dll`, FMA3 path on and off (Consult 6 D-4, commit "native: mirrors call the process's ucrtbase.dll math (D-4)"). Mirrors of a reference library's own CRT call now call the process's `ucrtbase.dll` through `cn_crt` (`native/include/cn_crt_math.h`). `native/tests/test_crt_math.py` compares every entry with the static CRT and warns on a difference: on this machine only `pow` with FMA3 differs, by 1 ulp (12 of the test's 32,786 inputs; 261 of 1,048,594 in an enlarged run). Bench note (Consult 8 R-d): no tested bits change; two untested double-`pow` paths (PyMatting's pre-round level sizes, ncnn's float64 power) now follow NumPy, as the probe test records | SP5a, before any build variant |
| Closed by `3a324fe1`: a process that loads every package and node module and then exits normally died at interpreter shutdown: a call to address 0 from `ntdll` (`0xC0000005`, crash report with module "unknown"). Seen in an import probe; `run.py --close-after-start` (`npm run type-check:runtime`) takes the same path on Windows. The fix: the NCNN session frees its cached nets, then ncnn's Vulkan instance, at exit, so a normal exit returns 0 (U4-light item 2, VERIFIED on the GPU: `server.py <port> --close-after-start` with Vulkan visible, 0xC0000005 before and 0 after; the controller's re-run of `type-check:runtime`'s command line on main `a831d6b6`, Vulkan visible, 2026-10-07: exit 0, its three variables set directly since this checkout has no `node_modules`). Each `--close-after-start` exit also prints Sanic's "Task was destroyed but it is pending!" and its Python 3.14 event-loop-policy `DeprecationWarning`s (relayed at ERROR by the host), as upstream's code does on this stack (the oracle, two runs); that policy API is removed in Python 3.16, so a 3.16 upgrade needs a Sanic release without it. The `msvcp140.dll` crash reports of 2026-09-21 to 2026-10-02 are resolved: they were Torch/ONNX Runtime import failures in test processes where Pillow's bundled 14.29 loaded first; `native/tests/conftest.py` now loads the system copy and the suite runs in one process | Closed by `3a324fe1` (controller re-run: exit 0). Leftover crash-report clones (`python.exe`, 0 threads) lock DLLs until an elevated `taskkill /F /PID` or a reboot |
| Live clipboard publication/paste (API-boundary tests only); external viewer (displayed image never observed); real Explorer/OLE drop retry (synthetic drop passed); SP4a's optional AMD uProf / WPR sampling session (elevated; rebuild `native\build\pdb` first, since the PDB DLL predates `cn_isa_get`; uProf flags VERIFIED from the installed docs, `wpr -stop <file>` unverified; local script `native\build\sp4a-sample.ps1` (ignored, not in a clean clone), with `-DryRun`; it would split the pyd copies and encode, 30–71 % of the image cases' auto-K loops, `profile-final.md` section 2); `native/tests/test_isa_dispatch.py` once on an AVX2-only machine (the AVX-512 cap is VERIFIED by disassembly, UNTESTED on this AVX-512 host); the kept AVX-512 units on a part that downclocks (UNTESTED; no clock-down seen here, `CHAINNER_C_ISA=avx2` caps them); the scalar level on an SSE2-only CPU (UNTESTED on SSE2-only hardware, forced on this machine; a Hyper-V VM in processor-compatibility mode first, an emulator download only after) | USER-GATE: manual checks |
| GPU-dependent work (the GPU lease and its unittest were removed on 2026-10-05, owner's decision; `GPU-DEFERRED.md` item 2 retired). Item 1, the GPU provider paths, was verified by U4-light on 2026-10-07 and removed: NCNN on Vulkan, ONNX Runtime's CUDA execution provider and PyTorch CUDA work on the 3.14 stack (`default.chn` on the GPU included; peak per-process VRAM 1.87 GB); ONNX Runtime's DirectML and TensorRT providers were not exercised | `GPU-DEFERRED.md`, under the owner's rule of 2026-10-07: no GPU benchmarks, at most 3 GB VRAM |
| SP0 deferred follow-ups | `docs/superpowers/plans/2026-10-01-sp0-foundation.md`, "Deferred follow-ups" |
| Upstream-parity items still listed (owner directive 4, 2026-10-04): the C1-class pins not yet checked against chaiNNer; lens power's input `maxss(+0, x)`, output-invisible and VERIFIED equal to chaiNNer's chain; compose's NaN selection, positions-level by rule (spec 4.4); suite-wide ±0-aware bitwise comparators (the suite's `assert_array_equal` treats ±0 as equal). Closed: the normalize/enforce clamp follows `np.clip` on the tested stack, −0 kept (ARCHITECTURE section 7, Consult 11 R-p) | SP5a, or the next task that touches them |
| Upstream-bug follow-ups (2026-10-09): the bundled `chainner_pip` 23.2 downloads PEP 658 wheels twice (needs pip's pypa/pip#11847 fix in the bundled wheel); SCUNet ONNX export unblocking (needs a real-model test); the ONNX size probe cannot find multiples 12/24/48 and its lock serializes ONNX upscales during a probe; the host reads the worker's output as strict UTF-8 (`backslashreplace` on that Popen); the real first-start check needs a fixed, firewall-allowed folder | `native/reports/fable-decisions.md` (D1-D15) |
| VOSR speed, PyTorch path: each window's DINO features go to the CPU and back, and every DiT prediction is synchronized and copied to the CPU (`vosr2_context.py`) | Keep them on the GPU when Default has room; paired A/B |
| VOSR cold start: the 5.6 GB FP32 DiT checkpoint is converted to BF16 at every cold load (`vosr2.py`) | A cached BF16 copy; measure the cold start |
| VOSR temporal on PyTorch: the next restoration does not overlap the CPU stabilization (about 1.1 s of 4.6 s per 2560×1440 frame); `stabilize` runs on one thread (1.0 s); the next `motion_pair` could run alongside | Paired A/B and a memory check for each |
| VOSR temporal: each item settles its background restoration, so the GPU idles while downstream nodes (Save Video's encode) run | Owner decision: letting it cross the item boundary relaxes "no GPU work escapes an item" |
| VOSR engine identities are broader than the engines: all of `vosr2.py` (a message edit rebuilds) and the whole plugin DLL (a metadata-only rebuild rebuilds) | Hash only engine-relevant code and the DLL's SASS/PTX |
| VOSR engine sets of earlier identities (BF16, the prepared FP8 `f2d7a3da…`) are never removed: 1.7 to 3.5 GB each | A "remove unused VOSR caches" step |
| VOSR engine builds: about 3 minutes without progress, one stage at a time, no timing cache | Stage count in the log; small stages in parallel; a per-GPU timing cache |
| VOSR FP8 and model downloads: one host each and no mirror | A mirror list beside the pins |
| VOSR plugin release: nothing requires `vosr2_qkv.dll` in a release, and the package manifest does not record its TensorRT and CUDA versions | A release flag that requires it; versions in `native_toolchain` |
| VOSR scene focus where a busy in-focus texture (thatch, foliage) holds the sharpest strip: in-focus faces count as partly defocused and lose 10-15 % detail (the issue-13 portrait, 2x). An edge-width reference (re-blur gradient ratio) fixed that frame but weakened Rin's defocused background at 4x. Separately, the half-soft face there was VOSR's own window seam (present with routing off), now handled by window placement and face windows (State). Owner 2026-10-14: keep today's reference and the 64 px minimum window overlap (16 px measured: 8 % faster, 3x the seam step) | A focus measure tuned on more of the owner's frames; seam windows for non-face content (anime faces, hands) need a trigger that finds them; faces wider than 800 px keep the layout's seams; in an output under 512 px on one side, a mirrored copy of an edge face in the padding can take a face window (harmless, rare) |
| VOSR plugin code: no sm_90/sm_100 SASS (compute_120 PTX cannot run there); no header dependency file for the nvcc step; a dead branch in `qkv.cu` and an `rsqrtf` worth a comment | SASS when those GPUs become targets; `-MD -MF` with `DEPFILE`; code cleanups need GPU revalidation |
| Look-ahead executor: K is decided at item 0 once a node looks ahead (a later-computed writer folder then forces K = 1 for groups reading per-item files); jobs computed ahead get a fresh context each; `ItemWindow._wake` rechecks every waiter; forced jobs bypass P; the UI's lineage takes the first iterated edge, the backend the first iterated input | A schema-level "looks ahead" flag, a per-node context cache, a per-slot waiter map and a cap, when a measured need appears |
| VOSR in the local portable package: the nightly UI's patches lack the iterated condition | A patch JSON against the restored nightly (README, Prerequisites) |
| VOSR temporal tests keep the removed generators verbatim as their oracle | Golden float32 outputs of a tiny clip |
| Loop runs allocate about 60 MB more torch memory than one call (likely a cuBLAS workspace per pool thread) | Pin a GPU node's calls to one thread |
| `tests/common/backend-nodes.test.ts` imports nodes with `error_on_failed_node=False`, so a node that fails to import disappears silently | Assert a few schema ids or import strictly |
| `Build.ps1 -Configuration Debug` fails at `chainner_ext` (needs `python314_d.lib`, which the runtime lacks; pre-existing) | Skip `chainner_ext` in Debug or provide the debug library |
| Upstream check and bug hunt (2026-10-16), queued items, each ruled by the Fable stand-in (`native/reports/fable-decisions.md`, "2026-10-16 · Upstream check and bug hunt"): `independent_ui.py` patches for 17d9331c (uninstall a disabled package) and 3f671309 (hidden experimental toggle), with the nightly-restore item (first row); Electron logs backend stdout per chunk, so bare prints are missing from `main.log` (a line-based reader keeping every level); the read-ahead predicate refinement for look-ahead nodes (compute Save Image's folder nodes before the item-0 decision, compare the full output folder; Save Image's skipped-file CPU work also runs ahead); look-ahead jobs share the node's cleanup set (latent); Descale: a multi-reflect Mirror instead of the refusal, and a schema-range sweep for degenerate Blur/Taps (Blur 0.01 Bilinear gives black); PyAV CFR resampling for variable-rate MKV/WebM (today they use the CLI reader); VOSR engine-cache self-heal (move a corrupt cache aside, rebuild); pip resolves on every backend start (a lock-vs-installed check); lazy inputs: evaluate the producer inline on the waiting thread; untaken-branch outputs cached until run end; malformed `/run` error wording; Save Image: reject `..` (owner-visible), RGBA to JPEG flatten vs drop (owner choice), AVIF 4:0:0 label, `nul` message; Riemersma History Length above about 487 M; ONNX int64-input message; Build Engine minimum 48 px; a newer-chaiNNer chain notice | Next sessions, in the order of the decisions file |
| Owner queue (2026-10-16): Electron 25 -> 43 (about 40 advisories; must land with upstream #3343/#3344), vitest 1 -> 4 (with ip-address 9 -> 10 and adm-zip 0.5 -> 0.6), upstream #3318 (OOM recovery and tile-size display, a feature), re-release v0.3.4 with today's Descale Mirror out-of-bounds fix (v0.3.4 ships the bug), and the upstream features #3140/#3191/#3187/#3222 stay out | The owner's yes or no |
| VapourSynth hunt follow-ups (2026-10-16; the v0.3.5 re-release has the 21 hunt fixes plus 3; rulings by the Fable stand-in in `native/reports/fable-decisions.md`): startup install of user modules outside the node import (the host waits 300 s for nodes, an install may take 900 s); vspipe in a job object so it ends with the worker (move `OwnedJob` to `nodes/impl/owned_job.py`, soft `assign`, vsview excluded); hide the ~36 base classes that always fail in Class dropdowns (scan hide list, vs-jetpack-version tagged) with MVTools methods needing analyze(), no-clip/tuple returns and object-only methods; `*args` clips; wrapped signatures (DFTTest.extract_freq, Morpho.black/white_hat); version pins in user_modules.toml; name a hanging user file in the scan timeout; uninstall removes transitive deps no other package needs (keep numpy, scipy and chaiNNer-C's own); uninstall KeyError; AI Upscale SAR with non-uniform ONNX scale; Preview Clips checks a temp copy before rewriting an open vsview's script; silent vsview start failure and per-worker viewers; `range` ints by sibling params; union inputs; FilePathType paths; per-plane expression lists; "*" file drag-and-drop (frontend). | Reports in the session scratchpad `vshunt\a1..a5-report.md`, rulings in `vshunt\fable-rulings.md` |
| GPU phase of the YUV/TensorRT 11.4 work, the rest (owner approval only; ESRGAN models, DUAL Light/XS) | Phase 1 done 2026-10-18 (`native/local/yuv-planes/report-gpu.md`: runner tests, Upscale Image and AI Upscale bit-identical, 11.2 vs 11.4, bridge vs runner vs vstrt, the 11.2-engine message; Tile documented, not honoured). Left: AI Upscale's runner default on a 1080p RGBS testsrc2 clip, Streams 1 and 2, against the bridge (`CHAINNER_C_TRT_RUNNER=0`) with one ESRGAN engine, headless, 50 frames after warm-up: frames bit-identical, ms/frame each; vosr2_qkv and DUAL on 11.4 |
| GPU checks deferred on 2026-10-16 (a model was training; light checks only): Build Engine Fixed 64x64 -> Get Engine Info -> Upscale on 200x150 under chaiNNer-C's TensorRT 11.2; a fixed-size ONNX engine keeps its size under 11.2; the timing cache file after the first build and "Loaded" on the second; Stop and worker kill during a real DUAL build (no child left, VRAM back); the TensorRT session-leak paths (need a CUDA fault); the VOSR CUDA checks listed in the area-7 hunt report | The owner's next GPU session |
| instant-exec, deferred by ruling (Fable, 2026-10-19): Save Video adds audio in the encode pass, dropping the second full pass (R5b); Clip Frames evaluates its script twice (a sidecar from the frames run's evaluation); Normal Map Generator native only with a native FFT, no partial port (R9); an optional row stride in the image ABI (a partial-width crop copies today where Python returned a view); Load Images as a native `ItemSource`, only if `resize-real-256` shows at least 0.1 ms per item of GIL-held glue; zlib-ng inflate for PNG and TIFF decode (read side only); the Python executor and its white-box tests retired in the first release after v0.3.7 whose full gate passes with the native default (A5) | Next sessions, each with its paired measurement; GPU items: `GPU-DEFERRED.md` rows 6-22 |
| instant-exec residuals (the lane reports' open lists, `native/local/instant-exec/report-*.md`). Engine: 16 concurrent previews 20 ms against Python's 9; a fault outside a node names no node; a context node computed ahead runs its cleanups at once (outputs equal); no memo for native generators' items; the memo counts a storage an item and its crop share twice (errs safe); a file changed between stat and decode can hit a stale entry (a same-size rewrite is seen by its change time); latent: `decide()` takes a slot after the GIL, paused time while the driver blocks outside `suspend()`; engine images are always C-contiguous (P7: the reference); `engine_differential.py` builds a chain per executor, so side-effect order across sinks is not compared exactly. Startup: the first launch after an update shows the splash once; fresh storage compiles PyMatting (about 25 s, delaying its two nodes and the full list); Run is inert until the backend takes requests (about 1 s). Nodes: a Seed through Pass Through is an equal new object; above 2^63 bytes NumPy raises ValueError where the engine raises MemoryError or OverflowError (per-node decision). Codecs: non-plain TIFF Load Image and GIF/TGA/AVIF/DDS Save Image still decline; `JSIMD_FORCENONE` changes our libjpeg-turbo, not OpenCV's (one CMYK JPEG-in-TIFF); the writer's cross-drive move is untested; `.exr` still fails (owner queue). Upscale: the size probe's disk cache and background probe at Load Model; `recover_alpha` per chunk; output-changing TensorRT balanced tiles and PyTorch tile batching (owner decision). App: pip's `~` folders and FFmpeg's `.part`/`.tmp` crash leftovers are not swept; a backend crash's `app.exit(1)` skips the settings flush; `useSettings` subscribes to the whole object; a layout-only counter, an incremental type pass and preview reuse on reopen (skipped: small gain). Video: a fused float BGR -> 16-bit planes kernel; `chainner_vs/upscale.py` `_image` still `np.stack`s; no native PyAV reader (Save Video sets the pace). Models: `size_probe.py` could use `model_facts`; the probe and the build cache hash the same bytes twice; Interpolate Models and auto_split mutate shared models; two concurrent first uses of an ONNX session make two sessions. Backend: `file_sequence.cpp` resolves every folder (`realpath`; a 1000x5 tree 227 ms against 105 ms with `scandir`) unless the second wave's listing work covers it; dead code: `server.py`'s `/packages` builds a dependency list it discards. Final wave: an 8.3 short target name is no longer expanded by the atomic writer; a module that fails on every start for a reason other than a missing dependency never writes the UI cache (that start shows the splash, as before); `gc.freeze` is skipped when a framework node's preview finished first (the owner's template), so R1's gain is for starts without one; the native codecs read OpenCV's limit variables as plain integers only; two Convert To ONNX nodes with different options on one model re-export on each run; the size probe re-probes a model that does not run on each load | Next sessions, by measured size |
| Owner queue (instant-exec, 2026-10-19; `native/local/instant-exec/owner-queue.md`): release v0.3.7 (tag, push, D: update)? The native engine becomes every user's default executor; the Python one stays reachable by `CHAINNER_C_EXECUTOR=python` until its retirement. FFV1 `-level 3 -slices 16` for Frames to Clip's internal lossless temp? That node 2.7x faster, temp bytes change, one named exception to the codec rule (R6). Build Engine's disk Engine Cache on by default? Up to 16 GiB of engines in the storage folder. A pip wheel cache for dependency installs (multi-GB, the disk rule). The memo cache as a user setting. The `.exr` loading fix (OpenCV's build has no OpenEXR). The spandrel fork wheel release with the DRFT fold speed-up (build 309 -> 210 ms). The Python executor's retirement in the first release after v0.3.7 (A5). Checks: the image size limit as read (40,000 x 40,000 = 1.6 billion pixels in total, each side up to 1,048,576); clicking a node selects it on mouse-up (was mouse-down); the packaged `out\` startup timing (one launch; a new folder may prompt the firewall); the GPU checks, `GPU-DEFERRED.md` rows 6-22 | The owner's yes or no |
| VRAM after Stop follow-ups (2026-10-19): on `CHAINNER_C_EXECUTOR=python` a dropped /run's node that ends after the request keeps its memory reserved (collect_failed_run ran before it ended: 18 GB held 25 s in the probe) | Release after the node's end on the Python executor |
| VRAM offload follow-ups (2026-10-20): a first overflow in a session pays the page-locking of its pinned RAM (2176 px: 5.96 s against 4.95 s warm), again in each run that overflows (its pinned RAM is released at the run's end); least-recently-used order brings 6.8 GiB back where a traced farthest-next-use plan would bring almost none; only Upscale Image's forward is covered (inputs of 2 MP or more) | A traced plan if the warm 7 % matters |

## Decision log

User rulings (pre-program handoff section 2, at the tag; still binding):

- Exact chaiNNer UI/UX: schemas, names, controls, saved chains, previews, errors, cancellation,
  ownership and restart. Only exceptions: updater removal and the reported drag-and-drop repairs.
- Independent edition: never check upstream releases; old settings cannot re-enable it.
- C or C++ per component; tested C kernels stay C; no renaming or wrapping counted as conversion.
- Broad optimization across workflows; fpng chosen for supported PNG encoding.
- CPU benchmarks allowed, without AI models (supersedes "no timing benchmarks"). Historical GPU
  limit: at most 2 GB VRAM, no precise GPU timing; superseded by the owner's rule of 2026-10-07: no GPU
  benchmarks, at most 3 GB VRAM.
- No computer-use or desktop automation. Ask before dependency installs or toolchain switches.
- Keep implementation, tests, packaging, runtime and performance evidence separate. Reap only
  owned processes; never kill by name.

Program decisions (spec section 2, 2026-10-01, as amended): bit-exact float32 against the installed chaiNNer as the user receives outputs (user, 2026-10-04; judged
too: where it is wrong the port does the correct thing and records a deviation, ARCHITECTURE section 7; B0 and B3 are regression proxies), SIMD only where
bit-identical, no FMA contraction, reordered reductions or fast-math; one tracked backend; checkpoint and tag, one reviewed commit per task, no `Co-Authored-By`, no
pushes unless asked; decisions as 2-4 options with a recommendation (the user does not write code); GPU work deferred; Opus for critical work and Sonnet for
zero-risk tasks, the Fable stand-in consulted in place of the user. Distribution (2026-10-02): the port ships on GitHub, so optimization is generalized and
deterministic first (same bits on every machine); superseded on 2026-10-06 by the owner's clang-cl `-march=icelake-server` build with
ThinLTO and no PGO (Consult 9; State, Pending); corrected on 2026-10-07: baseline x86-64 with runtime AVX2/AVX-512
dispatch, `-march` local-only.

SP0 rulings R1-R13: SP0 plan, "Execution rulings" (R11b noise-floor method, R12-R13 auto-confirm). SP2 rulings R1-R5 and the user's SP2 decisions (per-machine chaiNNer parity, balanced helper sharing, engines untouched): SP2 plan and spec. SP3 decisions SP3-P1 to P11 and the accepted residuals (an iterated subdirectory or name containing `..` can escape the checked base directory, Load Images never yields one; a declared file that is a symlink into an output directory, the loopback admin-share alias of a path, or a hardlink is not detected): SP3 plan and spec; owner-level SP3 decisions by the Fable stand-in (`native/reports/fable-decisions.md`). SP3b decisions: D1-D7 (SP3b spec section 3) and the Fable stand-in's rulings on the design and on the Task 4 results (`native/reports/fable-decisions.md`, 2026-10-05). SP3c: the probe's pre-registered rules and verdict (`native/reports/sp3c/`). SP4 decisions: SP4 spec, the SP4a plan's P1–P12 and the SP4b plan's Global Constraints and D1–D15 (with the owner directives of 2026-10-04: the installed chaiNNer is the reference and is judged, tests are judged, finished work is not re-audited); video (owner, 2026-10-03/04): the codecs themselves are not tuned, and video cases are judged by our per-frame timer, never by wall-clock; owner-level rulings by the Fable stand-in.

SP1 rulings:

- SP1-R1: root `.gitattributes` gains `backend/src/** -text` (bytes stored verbatim: CRLF, LF and
  mixed). The `eol=LF` fix has landed: `.gitattributes` is now `* text=auto eol=lf` (U3 lane E).
- SP1-R2: the byte-identity proof excludes `__pycache__` and the portable profile; the v2 manifest is
  cross-checked against v1, not byte-compared.
- SP1-R3: the real package is never rebuilt from scratch; it is converted in place once
  (`--adopt-v1`, removed afterwards).
- SP1-R4/R5: no shell-digest pin (the two bundle pins suffice); `Package.ps1` deleted, `README.md` gives the venv command.
- SP1-R6: `docs/superpowers/**` stays as the planning record; since 2026-10-07 (owner) it is local and git-ignored, so the paths into it in this file name local files.
- SP1-R7: `backend/tests` is the `d56e507f` set; the port's `test_lazy.py` extension stays at the tag.
- SP1-R8: `native/` comes from `codex/c-port` HEAD (SP0 included), not the bare checkpoint.
- SP1-R9: `backend/src` is frozen during SP1; backend cleanups wait for SP2 or later.
- SP1-R10 to R13: the packager v2 lands before the retirement commit; `--adopt-v1` is deleted after
  its one use; the whole-branch review covers the cleanup commits; the branch is kept, nothing
  merged; execution ruling SP1-X8 stores `native/tests/reference_*/**` verbatim too.

## Records

Local, git-ignored records (SP1-R6). Spec: `docs/superpowers/specs/2026-10-01-chainner-c-program-design.md`. Plans and their execution
rulings: `docs/superpowers/plans/` (`2026-10-01-sp0-foundation.md`, `2026-10-02-sp1-single-source.md`, `2026-10-02-sp2-cpu-budget.md`, `2026-10-02-sp3-cross-item-pipelining.md`, `2026-10-03-sp4a-measurement-dispatch.md`, `2026-10-03-sp4b-kernels.md`, `2026-10-05-sp3b-decode-off-the-producer.md`, `2026-10-05-sp4c-output-allocation.md`).
History: `git show chainner-c-checkpoint-20261001:chainner_c_handoff.md` (2026-09-26 handoff: paths, invariants,
results, 161-node catalog); the retired `native/*.md` reports, overlays and transplant tools are at the same tag.
