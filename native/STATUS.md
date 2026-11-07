# chaiNNer-C status

Current state, pending work and the decision log. Commands: `README.md`. Design: `ARCHITECTURE.md`.

## State

- 2026-11-07 **Spandrel 0.4.2+c17: HAT, ESRGAN and DUAL faster in PyTorch; HAT faster on TensorRT.** Same session, RTX 5090, FP16, before and after. HAT-L: 512
  px 775 -> 507 ms, 256 px 194 -> 128 ms, taking DAT's path through `__arch_helpers/inference.py`. ESRGAN channels last on CUDA: 1200x864 621 -> 440 ms, peak 7.95 ->
  4.00 GiB, same output. 2x DUAL Light, one product per 16x16 cell for all four window grids: 1200x864 249 -> 222 ms, 256 px 39.6 -> 32.8 ms, peak 2.11 -> 2.41
  GiB. FP16 accuracy against strict FP32 the same to 0.03 dB. HAT's FP16 export pads its 180-channel trunk to 184 in the weights (`hat_export.py`) and PatchEmbed
  is a permute: TensorRT Fixed 1200x864 1871.7 -> 1559.3 ms, Dynamic 64-512 at 512 px about 441 -> 388 ms, 256 px 78.6 -> 63.1 ms, crops the same to 0.2 dB. ESRGAN's
  and DUAL's TensorRT plans have no moves left to remove (3.8 of 215.7 ms and 1.3 of 73.3 ms at 1200x864). On ROCm, HAT and DAT keep channels first and their
  depthwise kernels, which c16's DAT path had left unused. HAT on TensorRT and DAT's planar form hold about 1,500 MHz above 512 px at about 350 W (power limit 600 W
  of 800 W, sw_power_cap active) while 256 px holds 2,600 MHz at 550 W: untested whether a higher limit helps.
- 2026-11-07 **DAT's FP16 export 17 % faster at 1200x864, less memory.** The lengths sum along each row and then the rows; the spatial attention takes
  each branch's own channels in one pass, its spatial map from the tokens as linear layers and its mix in tokens. Trained 4x DAT2, FP16, Dynamic 64-1280
  with the optimum at 1200x864: 1166.0 -> 970.8 ms (PyTorch: 1082 ms), Fixed 1141.8 -> 921.8 ms, context 14.63 -> 13.68 GiB; V3detail Fixed 256 px 47.60 ->
  40.13 ms, Dynamic 512 px 249.04 -> 220.66 ms. Crops against strict FP32 the same to 0.1 dB, 65.0 to 70.0 dB, no NaN on 2x_bndl v1.2. Keeping the channel
  attention planar as well removes more moves but is no faster: the GPU's clock falls from about 2700 to about 1600 MHz (1043.3 ms). The ONNX changes, so
  automatic engines rebuild, and a saved engine keeps the old graph until it is converted again. UNTESTED: an AMD GPU.
- 2026-11-07 **Spandrel 0.4.2+c16: PyTorch DAT 35 % faster, half the memory.** Profiled: most of the time was moving data and redoing constant
  work. Inference runs channels last, so convolutions read and write tokens in place; a shifted window that does not wrap is a plain window at
  an offset, with no roll and no mask (39 of 256 windows at 256 px take them); position biases, batch norms folded into their convolutions,
  channels-last convolution weights and heads padded in qkv's weights are kept, and dropped when the model is moved, cast or loaded. Trained
  4x DAT2, CUDA, FP16, same session: 256 px 99.91 -> 64.13 ms, 512 px 413.32 -> 269.22 ms, peak 4.28 -> 2.42 GiB; FP32 256 px 185.58 -> 138.85 ms;
  CPU FP32 64 px 284.0 -> 213.9 ms. Crops against strict FP32 71.8-73.8 dB (71.8-73.9), FP32 against c15's FP32 135.6 dB or better. ONNX
  exports byte-identical to c15. HAT is unchanged. UNTESTED: an AMD GPU.
- 2026-11-07 **DAT's FP16 channel attention takes its product by rows instead of in FP32: the old speed with the FP32 product's
  accuracy.** The per-layer profile put the cost in 18 FP32 GEMMs, 32.6 of 79.69 ms at Fixed 256 px. Scaling both sides so that no term
  is subnormal changes nothing (512 px crops 43.4 dB): what loses the image is one FP16 sum of every pixel's small term. One FP16
  product per row, each a short sum, with the rows added in FP32. 4x IllustrationJaNai DAT2 on TensorRT, level 3, TF32 off, against strict
  PyTorch FP32, plain FP16 product / FP32 product / by rows, one session: Fixed 256 px 49.60 / 79.69 / 47.60 ms, Dynamic 512 px 260.08 /
  366.72 / 249.04 ms; Dynamic 512 px crops 43.4-52.8 / 66.3-69.3 / 66.3-69.2 dB, largest error 74.63 / 1.71 / 1.83 codes. By rows on a
  Dynamic engine: crops 65.0 dB or better at 64, 128, 256, 512, 128x512 and 512x128, no NaN; 2x_bndl v1.2 no NaN, crops 63.5 dB or better.
  A saved FP16 ONNX or engine keeps the old form until converted again. UNTESTED: MIGraphX on an AMD GPU; above 512 px.
- 2026-11-07 **AMD GPUs move to ROCm 10.0.0, PyTorch 2.13.0, TorchVision 0.28.0, MIGraphX 2.17.0 and ONNX Runtime 1.31.1.** Linux and
  Windows both take AMD's builds from `stable.repo.amd.com/rocm/whl-next/` (Windows had PyTorch 2.12.0, Linux PyTorch's own 2.14.1):
  one list in `get_pytorch_rocm`, by pinned URL and SHA-256 on Windows and by that index on Linux. Linux gains PyTorch's and
  TorchVision's kernel wheels and rocm-bootstrap 0.3.0; its Triton is AMD's `triton 3.8.0+git4cff872c.rocm10.0.0`, which replaces
  `triton-rocm` and which NVIDIA's and the CPU builds replace (`triton+git`). ROCm 10.0 keeps `amdhip64_7.dll` and `libamdhip64.so.7`
  and reports HIP 7.15. AMD's Linux wheels need glibc 2.28 at most. MIGraphX 2.17.0 already has the literal fix 2.16.1 was made for; the
  HIP Graph patch needed two hunks moved, and ONNX Runtime's three patches apply as they are. Fork release `v1.31.1-rocm10.0-hipgraph`:
  Linux `3856b2f0...` (101,120,585 B), Windows `64feb4c6...` (80,406,026 B). NVIDIA and CPU stay on PyTorch 2.14.1 and ONNX Runtime
  1.30.0. Nothing here has run on an AMD GPU: checked are the builds, imports, MIGraphX's ref target and pip's resolver.
- 2026-11-07 **Spandrel 0.4.2+c15: PyTorch FP16 HAT and DAT keep the last two groups' sums in FP32.** The export's FP32 spots move into
  the model. On CUDA PyTorch, against strict FP32, trained 2x DAT2 files that went all NaN on 5 to 11 of 12 inputs (natural crops too, at
  256 px) go NaN on none; FP32 output is byte-identical on 9 model and size pairs; 4x IllustrationJaNai DAT2 and HAT-L keep their crops
  (70.8-75.6, 71.4-74.4 dB). 256 px, c14 -> c15, DAT2 100.2 -> 102.6 ms, HAT-L 203.7 -> 205.3 ms; 512 px DAT2 413.2 -> 420.9 ms, with
  builds running. BF16 in the same spots passes the same limits and is no faster (DAT2 256 px 106.0 ms, 512 px 412.9 ms; HAT-L crops
  68.9-71.0 dB), so FP32 stays. Whole-model BF16 is as fast as FP16 mixed (DAT2 99.0-100.4 against 99.0-101.0 ms, 512 px 411 against 416 ms)
  at 54.3-57.7 dB. Files saved before this that take the export route are unchanged, as its spots were the same.
- RELEASED 2026-11-06 as **v0.3.10 re-release 1** (tag `bec124491`, one commit over `5bda45d80`, tree = local `e571f5674`; win-x64 `7e9256a9`, linux-x64 `f03bbdbe`): the entry below.
- AMD: PyTorch's Upscale Image has Use MIGraphX Program (input 8's label on AMD PCs; one program per Tile Size, kept in
  the Program Cache folder; smaller images padded by reflection to the program's size and the output cropped; a failure
  falls back to PyTorch with a warning). DUAL builds on AMD Windows through `triton-windows` 3.8.0.post29 with
  `dual_migx_ops.dll`. Build Program has Use HIP Graph (experimental, off by default), a backport in
  `native/tools/migraphx/migraphx-hip-graph.patch` on pin 2487b688, in the wheels `1.30.0+rocm7.14.hipgraph` and
  `.hipgraph.dml` (fork release `v1.30.0-rocm7.14-hipgraph`); Get Program Info shows HIP Graphs. Written without an AMD
  GPU: DUAL's kernels lowered for gfx1100 on Windows, the Windows wheel imported, the Windows runner compiled against
  the new MIGraphX; nothing run on a GPU. NVIDIA PCs rebuild kept automatic engines once (the key hashes
  `auto_engine.py`). Gates on `1cf8df86b`: Windows suite 144,123 passed, 826 skipped, 2 failed (one test's file count,
  fixed; `test_streamed_full_vae_matches_direct_encode_decode` failed once at 12 workers and passed alone, cause not
  found); Linux suite 141,048 passed, 1 failed (the same count), 2,916 skipped on WSL 3.0.3, kernel 6.18.54.1; ROCm tests
  572 passed, 94 skipped; `test_convert_to_onnx.py` on Linux 391 passed; VapourSynth 618 passed; type check and ESLint
  clean. The tests copy FFmpeg from a local copy and never download it (`native/tools/scratch_storage.py`).
- RELEASED 2026-11-06 as v0.3.10 (tag `5bda45d80`, one commit over `30b7ae113`, tree = local `a15ff71fc`; win-x64 `c19d6194`, linux-x64 `3b544543`): the entries below down to "Electron alternatives researched". Gates on `80c911b1d`: Windows suite
  144,053 passed, 815 skipped, 3 failed that later commits fixed (the files changed since: Windows 805 passed, Linux 645
  passed); Linux suite 140,975 passed, 0 failed, 2,913 skipped; AMD tests 515 passed, 92 skipped; vitest 547 passed, 27
  skipped; VapourSynth files 610 passed, 19 skipped; eslint and tsc clean. AddressSanitizer and UndefinedBehaviorSanitizer on
  the merged tree: native suite 137,106 passed, backend 4,719, the node sweep on all 136 nodes, the Denoise, blur and codec
  files at AVX2 and scalar: one report, DirectXTex's known mismatch in texconv. Not run: GPU paths. Earlier, on `294223515`: Windows suite 143,386 passed, 0 failed,
  810 skipped (on `dfa889fd4`, which differs only by the Linux skip list); Linux suite 140,322 passed, 1 failed, 2,901
  skipped (`test_vosr2_streaming::test_streamed_full_vae_matches_direct_encode_decode`, which passes 5 of 5 alone and
  uses unseeded `randn`; cause inferred), its `test_convert_to_onnx.py` 378 passed; AMD tests 515 passed, 92 skipped;
  vitest 471 passed, 40 skipped; eslint and tsc clean; VapourSynth test files under the integrated Python 607 passed,
  9 skipped. Before the last merge round, on `a8b92536`: Windows 142,499 passed, 0 failed, 800 skipped, 1,363 s; Linux
  139,805 passed, 0 failed, 2,903 skipped. Full Windows runs on the way (141,594 passed, 44 failed; 141,693 passed,
  187 failed; 142,342 passed, 1 failed; 142,937 passed, 515 failed) failed on tests whose call sites crossed merges,
  one lost import, a Resize clip that made the two executors disagree (dropped) and one VOSR test that passes alone.
  Records: `native/local/night-20261106/`.
- 2026-11-06 **Use TensorRT Engine goes up to the node's Tile Size.** Every automatic engine is Dynamic, from 64 px (16 px on
  ESRGAN, Compact, SPAN and PLKSR) up to the Tile Size, one engine per size; Auto, Maximum and No Tiling take 512 px. RGT stops
  at 256 px with a warning. Through the real route: a 2x SPAN at 256, 768 and Auto builds those three, runs tiles of those sizes
  at 64.6 to 66.0 dB against PyTorch FP32, and reads the first two back from disk in a new process with identical output; RGT at
  Tile Size 512 builds 64-256, warns once per process and holds 65.23 dB.
- 2026-11-06 **DAT's FP16 channel attention takes its product in FP32; attention models measured on Dynamic engines.** DAT2
  Dynamic at 512 px: crops 41.4-57.9 -> 66.0-68.5 dB, largest error 123.2 -> 2.6 codes, at a speed cost: 234.82 -> 357.30 ms
  at 512 px (FP32 573.03), Fixed 256 px 43.83 -> 77.10 ms. FP16 Dynamic 64-512 holds on FDAT XL (crops at least 65.6 dB),
  SwinIR-M (62.5), Swin2SR (62.3), SwinIR-L (60.6; a 64 px horizontal gradient reads 45.2 dB, rising to 55.2 at 512 px) and DRCT-L
  (59.86; 251 s build, 12.46 GiB in the TensorRT build), one trained weight each. RGT on a Dynamic engine filled the card at 512 px
  (30.8 of 32.6 GB, 36 GiB resident) and its process could not be killed for several minutes, with `nvidia-smi` hanging; 64 to
  256 px hold (crops at least 63.7 dB). Windows cannot end a process stuck in such a call, so a memory limit has to act before the
  request.
- 2026-11-06 **Spandrel 0.4.2+c14**: DySample's ONNX export follows the input's size (real RealPLKSR 16-25 -> 100-141 dB;
  older exports must be made again and still warn), SPAN folds once per device and dtype (full upscale 256 px, FP16,
  10.5-11.5 -> 2.9-3.7 ms), and Flex Attention loads on first use (`import spandrel` after torch 1.45 -> 0.95 s, 10 pairs
  on the real wheel; DRFT, legacy DRFT and SST CPU outputs identical; the hybrid and Flex routes not run).
- 2026-11-06 **FP16 on TensorRT widened and held to 512 px.** `FP16_ARCHS` 20 -> 24 entries: SwinIR `medium` and `large`,
  FDAT `xl`, Swin2SR `medium`, RGT, DRCT `large`, FIGSR (each judged on one trained weight; FIGSR has only one published),
  2.2 to 3.4x faster at Fixed 256 px, FP16 engine 54.5 dB or better on smooth images and 62 dB or better on crops against
  PyTorch FP32 (FIGSR: crops 58.4 dB or better to 512 px; its smooth inputs miss 50 dB where FP32 misses too). FDAT's
  channel attention takes its product in FP32: FDAT M at 512 px 35.7-45.4 -> 66.0-68.9 dB at the same speed. FP16 Dynamic
  64-512 holds on 13 listed architectures. Convert To ONNX's check runs at 64 px or below on the CPU and passes faults that
  grow with tile size. RGT follows PyTorch at a side of 1024 px on TensorRT (137.62 dB). Opset 14 FP16 holds no float64.
  The opt-in runner's seven fixes are confirmed on the GPU against the Python session.
- 2026-11-06 **Output changes**: blurs on RGBA have no fringe from the colour under transparent pixels (Gaussian, Box, Lens
  and Unsharp Mask's blur premultiplied; Surface Blur weights taps by alpha; Median, Dilate, Erode clear colour under alpha
  0 and stay per channel); RGB, gray and opaque RGBA are bit-identical. 1 MP transparent RGBA, ms: Gaussian r1 2.83 -> 4.99,
  Box r1 6.12 -> 8.50, Lens r3 9.54 -> 10.53, Surface r4 15.84 -> 20.51; opaque pays one alpha scan, 0.10 ms per MP. Add
  Noise Poisson is shot noise with mean 0, standard deviation Amount x sqrt(value) (1 MP RGB 9.7-24.7 -> 57 ms). Dither
  keeps alpha.
- 2026-11-06 **Mixed types removed, noise plugin added.** Convert To ONNX's Mixed FP16 and BF16, Calibration Image,
  autocast() and the NVIDIA ModelOpt package are gone (about 65 MiB of install); a migration clears them from saved chains,
  and a Mixed ONNX from an earlier version still loads and builds in its own types. Deband and Grain's Grainer gets its
  noise plugin (vsnoise 1.3, GPL-3.0, upstream's wheels by hash, Windows and Linux); Poisson runs a float clip or image
  through 16 bits, as the plugin has no float path. `noise.Add` takes clips only. Placebo still needs Vulkan.
- 2026-11-06 **Four branches in one tree.** The standard nodes load without SciPy and PyMatting: idle private memory 1703 ->
  1536 MiB, first use of such a node +0.7 s. New nodes Is Grayscale and Math Expression with optional number inputs. A node
  shows its own progress during long upscales on both executors, at most 10 updates a second, outside the chain bar
  (TensorRT 152.875 -> 152.822 ms, outputs identical); PyTorch upscales report their tiles; the window shows on native
  Wayland. 14 UI languages load on first use, English by default, with 62 own strings English until translated; the one
  English change is the timer tooltip ("7.13 seconds", not "7.1299999999999955 seconds"). The chain bar still follows a
  batch's last event.
- 2026-11-06 **Faults found by sweeps over nodes, files, video, VapourSynth, types and the editor.** Resize: Hamming
  and Hann reductions keep flat areas flat (flat 0.5 read up to 1.0 before; 36 of 495 cases change, all reductions,
  identical at every ISA level); the 16 GiB limit counts the horizontal pass. Load Image: grey PNG transparency, BMP
  kinds (masked 32-bit, OS/2 colours, RLE4, 16-bit white), planar and MinIsWhite deep TIFF, a `.tif` fallback,
  grey+alpha. Save Image: clear errors for refused characters and empty names; a folder ending in a space or a dot
  saves where Windows makes it (the literal name elsewhere). Nodes: Gaussian Blur with one zero sigma blurs (it
  returned the image), tiny images, Box Blur's integer axis, Burn, Dodge, Divide, Stretch Contrast on flat and NaN
  input, Average Color Fix, Quantize axes, Generate Threshold on RGBA; wider effects: Stretch Contrast Percentile with
  equal bounds returns the image; Unsharp Mask, High Pass, Edge Detection and Normal Map Generator change on tiny
  images. Utility and API: Derive Seed Directory, Go Up stops at a root, a folder named like an image is no item,
  errors keep their text after a NUL, the `/run` reply stays JSON on infinite or NaN input. Video: Load Video took its
  rate from `r_frame_rate`, so a 25 fps phone clip read as 90000 or 2000 fps and Save Video wrote it up to 3600x too
  fast; variable-rate FPS and Frame Count, uneven TS, FLV, PS and NUT lengths, PyAV left crop, Preview past the
  estimate, elementary streams, WebM and WMA audio, cover art, odd-size GIF. VapourSynth: a NaN or infinite result
  is an error (Denoise NL-Means on the CPU gave NaN and -302 to 109 silently), alpha follows the source window, SangNom
  frame against clip, the frame table fixed, 11 wrongly refused methods allowed (table 256 -> 266). Types and ONNX:
  seven nodes declare their refusals, Log To Linear Gamma, a Blend color layer ignores a hidden position (the type
  promised 64x48x3 and the node gave 68x55x4); stored masks stay out of Swin-family exports (Swin2SR Lightweight x2:
  a crash and 52 dB -> 143.8 dB), no float64 in FP16 below opset 17, SAFMNBCIE and FFTformer refused by name; RGT
  unchanged. Editor: Ctrl+Z after File > Open no longer leads Ctrl+S to overwrite the opened file with the other
  chain; execution, preview and paste fixes; a replaced value alerts and leaves the chain unsaved. Docs: chaiNNer-C's
  own judged outputs are the reference (BUILDING, ARCHITECTURE, README); the EXR doc says Load Image clips to 0..1.
  The video, file, utility and type fixes passed the suites without a separate reading. UNTESTED: the editor fixes
  in the running app (they need an npm rebuild and a version bump at release).
- 2026-11-06 **CPU speed, outputs identical** (ms, 50 iterations, paired, 1 MP unless noted, quiet machine, bench
  slot): Denoise avx512 43.83 -> 37.32, avx2 53.08 -> 43.69, scalar 164.03 -> 119.47; Image Diff 19.09 -> 5.74;
  White Balance Gray World 12.02 -> 4.57; Hue & Saturation 4.49 -> 1.95; HSV to RGB 2.51 -> 0.85; HSL to RGB 2.39 ->
  0.85; RGB to Lab 2.72 -> 1.81; at 12 MP: WebP lossy 89.77 -> 70.79, uncompressed TIFF 28.35 -> 22.36, AVIF 150.33
  -> 116.76 (10 iterations). Lens Blur 23.51 -> 10.23 and Median Blur 15.71 -> 6.66 are ratios from a busy slot, not
  bench figures (19 cases byte-identical at every ISA level, sanitizers clean). Load Video's output check 2.0 -> 0.15
  ms a frame. White Balance on 8x8 reads 0.017 -> 0.023 ms. Left out: a JPEG encode change (9 %, under the line) and
  a Resize clip that made the executors disagree. Test shards cap ONNX Runtime's and OpenCV's threads
  (`test_convert_to_onnx.py` shard 499 -> 433 s).
- 2026-11-06 **Every architecture converts with one automatic precision.** FP16 for the architectures and sizes
  measured to hold on TensorRT (`FP16_ARCHS`, 20 entries: ESRGAN, HAT, DAT, RealESRGAN Compact, SPAN, PLKSR, OmniSR,
  RCAN, Swift-SRGAN, MoSR, DRUNet, DCTLSA, FBCNN, RealCUGAN, DnCNN, MMRealSR, MixDehazeNet, RetinexFormer, SwinIR
  `small`, FDAT `medium`), FP32 for every other, architectures added later included, with no choice shown. DUAL and
  legacy DRFT are unchanged. An FP16 export is checked on the CPU against FP32, kept in FP16 between nodes, on smooth,
  flat, gradient, drawn and textured images. A failure re-exports FP32 from the file's weights with a warning. An FP32
  export is compared too and warns if it does not match. Warnings repeat on every reuse of a kept export, Stop acts
  between the check's images, and a comparison that cannot run says so. The file records architecture and precision,
  and Build Engine and Build Program follow it and hide Precision; a file with no record keeps it. The check is a
  guard, not proof for TensorRT (HVICIDNet reads 74 dB on the CPU, 34 on TensorRT). 29 real listed files pass with no
  fallback; failures read RMS 0.063 to 0.095 on detail images. Its session creation costs about 5x FP32's, paid once
  per model and node: kept, since without it overflow inside a chain goes unseen. Conversion, before -> after, CPU,
  same bench slot, s: Compact 0.82 -> 1.05, UltraSharp 29.07 -> 34.50, SwinIR-S 7.74 -> 9.75; FDAT M 2.12 (FP32) ->
  8.27 (FP16); HAT-L's check adds 89 s on 4 threads. The list rests on one TensorRT weight per architecture and size
  tag. UNTESTED: the check on a GPU device; TensorRT after the final merges; pressing Stop in a real run; MIGraphX
  beyond stand-ins.
- 2026-11-06 **FP16 measured on TensorRT, one weight per architecture.** TensorRT 11.4, Fixed 256, level 3, TF32 off,
  FP16 engine against FP32 engine, 5 smooth inputs and 4 real crops, worst dB crop / smooth. Holds: RCAN 63.87 /
  65.50, SwiftSRGAN 63.29 / 67.92, MoSR 68.69 / 69.22, DRUNet 63.85 / 63.47, DCTLSA 61.77 / 63.96, FBCNN 64.63 /
  67.41, RealCUGAN 63.62 / 62.58, DnCNN 54.88 / 52.74, MMRealSR 55.04 / 64.03, MixDehazeNet 58.92 / 55.22. Fails:
  HVICIDNet 35.74 / 33.10, NAFNet 21.84 / 71.22, DITN 1.95 / 0.00 with NaN, MoESR 15.95 / 67.25, KBNet 6.38 / 6.04,
  RetinexFormer 16.70 / 20.99 with NaN, fixed below (their FP32 engines match PyTorch at 87 to 152 dB, MoESR 61.02 on
  one crop). Against PyTorch FP32: SwinIR-S with `swin_export.py` 59.55 / 54.11, FDAT M 50.62 / 71.60.
  - FP32 -> FP16 ms, one paired run, 50 iterations: RCAN 31.42 -> 13.07, MoSR 9.04 -> 3.99, SwiftSRGAN 3.07 -> 1.72,
    SwinIR-S 12.35 -> 6.30, FDAT M 48.11 -> 13.88, DRUNet 5.52 -> 1.58, DCTLSA 12.24 -> 7.41, FBCNN 7.04 -> 1.98,
    RealCUGAN 3.46 -> 1.47, DnCNN 1.57 -> 0.82, MMRealSR 53.49 -> 19.14, MixDehazeNet 4.18 -> 2.32, RetinexFormer
    3.97 -> 1.84. RCAN's FP16 build takes 34.0 s at Fixed 256 (FP32 23.9 s), about 52 s Dynamic 64-512: one Myelin
    compile, linear at 12 ms per layer.
  - ONNX Runtime's CPU provider cannot clear a model for TensorRT: it drops FP16 cast pairs, so overflow stays hidden.
    The gap passes 6 dB on 15 of 16 rows (HVICIDNet crops 74.59 there, 35.74 on TensorRT; sizes and references
    differ). Verify runs there and passed all six failures, and Swin2SR while it goes NaN on flat images.
  - On CPU only (a graph that holds FP16 between nodes): Swin2SR, RGT and DRCT rewrites pass, with the end of the last
    group in FP32 and the mask in FP16; they stay FP32. No export or no run: ATD, FFTformer, IPT, SAFMNBCIE, GFPGAN,
    AuraSR; refused: SCUNet, SAFMN; not an image model: LaMa. UNTESTED: SST, GRL; Swin2SR, RGT, DRCT, SwinIR-M,
    SwinIR-L, FDAT XL on TensorRT; Dynamic engines on these rows.
- 2026-11-06 **SeemoRe's export fixed, FP32 only; RetinexFormer in FP16.** A SeemoRe export kept the traced example's
  expert for every image; `traced_size_export.py` now runs the expert each image picks (FP32 36.83-47.83 ->
  135.64-138.92 dB on a crop at 32-128 px). Its FP32 engine on TensorRT reads 123.58-137.59 dB on crops against
  PyTorch FP32. In FP16 its router ties on about 1 % of natural images and one crop reads 33.42 dB with the tie
  broken, so it stays FP32. RetinexFormer with FP32 lengths and an FP32 attention product (`fp16_export.py`), Fixed
  256 on TensorRT: crops 68.44-72.75 dB, flat white 48.19 dB (RMS 0.0039, worst pixel 2 of 255), FP32 -> FP16 3.58 ->
  1.64 ms. NAFNet, MoESR and DITN stay FP32: their rounding is amplified across the model, so no island reaches 40 dB
  on hard crops. UNTESTED: RetinexFormer above 256 px on TensorRT and on Dynamic engines; SeemoRe's larger models.
- 2026-11-06 **FP16 HAT and DAT keep their last groups' sums in FP32** (`attention_export.py`, 37 to 49 Casts, placed
  by measured peaks). Trained 2x DAT2 files went all NaN on smooth tiles in the released FP16 route. ONNX Runtime CPU,
  64 px, 10 smooth inputs: 2xEvangelion 9, 2x_bndl v1.2 9, v1 8 all NaN; real crops at 128 px 1 of 4, 4 of 4; with the
  fix none, 62.7 to 90.9 dB on smooth inputs, 63.96 to 76.91 on crops, on the two files re-run; weights already finite
  move -0.01 to +0.02 dB. Nothing left in FP16 reaches 65504 on 14 files swept (worst 14,665, HAT-L; margin 4.5x).
  - TensorRT, Fixed 256, level 1, TF32 off, 2xEvangelion against its FP32 engine: flat 0, 0.02, 1 and a gradient 100 %
    NaN as released; with the fix 60.49, 58.07, 54.17, 51.56 dB, no NaN. Flat 0.5 and noise stay 36.40, 36.57 dB
    either way (a model made for DVD sources: its dB is no precision verdict).
  - TensorRT, Fixed 256, level 3, 4x_IllustrationJaNai_V3detail_DAT2_28k against its FP32 engine, fix / as released:
    no NaN either way; flat 0 73.67 / 73.44, 0.02 69.09 / 69.59, 0.5 74.58 / 74.79, 1 112.23 / 112.23, gradient 68.90
    / 68.90, noise 40.01 / 40.04; 4 real crops 56.03, 60.76, 59.00, 62.72 / 56.02, 60.73, 59.05, 62.70.
  - CPU attribution on that weight: 96 px, graph holding FP16 between nodes, 77.61 dB, 84.19 after a 16 px blur
    (4xNomos2 72.56, 80.38); brightness does not move (0.09 codes). FP32 input and first convolution for HAT and DAT:
    67.28 -> 83.85 dB, 64.31 -> 80.14 dB at 64 px on smooth images, about 0.1 code: not added. A saved FP16 ONNX or
    engine keeps the fault until converted again. Speed on TensorRT was not measured with the fix. PyTorch FP16 raises
    a clear error on those tiles: its fix needs the fork. UNTESTED: MIGraphX.
  - Also: Interpolate Models (ONNX) works on HAT and DAT (a slice's INT64_MAX went through floats); ONNX and NCNN
    Interpolate take other plane counts; inline View Image refuses 2 planes; Build Engine's IndexError below opset 5
    is gone.
- 2026-11-06 **Verify, one set of limits.** An FP16 export is checked against the model in FP32 on a smooth image at
  16 and 40 px (padded to the size requirement): RMS at most 0.005, largest difference at most 0.02, and RMS 0.015 on
  a drawn and a textured image. Any other export: RMS at most 0.001, largest at most 0.1, and RMS 0.002 on the check's
  images. No NaN or Inf where FP32 has none. Verify on refuses a mismatch, Verify off warns. Correct exports read at
  most 0.001 (FP16) and 0.00028 (FP32, real DITN); wrong ONNX files at RMS 0.0165 and up are refused. It cannot see
  rounding inside a node or on TensorRT, nor FBCNN-style size faults at its two sizes.
- 2026-11-06 **FP16 export of Compact, SPAN, PLKSR and OmniSR takes FP32 up to its first convolution**
  (`input_export.py`); ESRGAN, HAT and DAT stay as they were. FP16 rounding of the input alone cost trained models up
  to 30 dB on smooth images: RealESRGANv2 TensorRT Dynamic 32.1 -> 62.8 dB, animevideov3 44.9 -> 56.7, 4x-anifilm 45.8
  -> 61.4. No time cost on Compact: RealESRGANv2 512 px 6.37 -> 6.18 ms; UNTESTED on SPAN, PLKSR, OmniSR. No NaN or
  Inf on any tested weight. Full ESRGAN in plain FP16 on 20 smooth images, CPU, 64 px: 5 weights, worst 51.24 dB
  (RealESRGAN_x4plus), 82 more UNTESTED. Reconvert to gain it; Interpolate Models refuses an older and a new FP16
  export together. Convert To NCNN takes out a Cast between FP16 and FP32, so Compact converts again (RealESRGANv2
  NCNN against ONNX 77.76-79.38 dB in FP32, 70.77-74.67 in FP16); SPAN and PLKSR give their own refusals.
- 2026-11-06 **PLKSR and RealPLKSR convert and build in FP16 with no precision choice**. Session against PyTorch FP32
  on six trained weights, Fixed and Dynamic: 62.4 to 74.0 dB on the worst crop, no NaN or Inf. TensorRT, 64 channels
  and 28 blocks, FP32 -> FP16: 256 21.12 -> 6.20 ms, 512 76.81 -> 21.09 ms. DySample stays in FP32 inside the FP16
  graph, from weights as loaded (42 -> 71 dB at 256, 36 -> 66 dB at 512). The export has no ScatterND any more
  (`plksr_export.py`): ONNX Runtime on CUDA 108.15 -> 26.15 ms in FP16 and 133.80 -> 29.84 ms in FP32 with the same
  output; TensorRT unchanged. Older ONNX files gain only when converted again. GroupNorm on small Dynamic ranges holds
  in every precision: Build Engine writes it out in FP32 with casts, for FP16, BF16 and Mixed too (all NaN -> 66 dB in
  FP16, 67 dB in Mixed FP16; 53 dB in BF16 and Mixed BF16). Plain PLKSR shares the recipe on random weights only.
  UNTESTED: MIGraphX; LayerNorm with DySample on a trained weight; input above 512 px.
- 2026-11-06 **DySample's ONNX export follows the input's size** (`dysample_export.py`). A trace froze its grid at
  32x32, so RealPLKSR, FDAT, GaterV3, SMoSR, MoESR and MoSR exports were wrong at every other size: real RealPLKSR 16
  to 25 dB against PyTorch at 48 to 256 px, now 100 to 141. Older files need converting again. **Compact and SPAN
  convert and build in FP16 with no precision choice**: against strict FP32 at 256x256, Fixed and Dynamic, 8 crops, no
  NaN or Inf, 6 Compact weights 67.1 to 73.8 dB, 5 SPAN weights 52.5 to 65.3. Session ms, FP32 -> FP16: Compact (64nf
  16nc) 1.86-1.96 -> 1.03-1.17, SPAN x2 2.05-2.10 -> 1.06-1.20. SPAN's ONNX Runtime CPU run costs +48 %, Compact's +5
  %. `FP16_ARCHS` holds spandrel's names. FDAT, GaterV3, SMoSR and MoESR were checked on random weights only.
- 2026-11-06 **OmniSR converts and builds in FP16 with no choice.** Two sums FP16 cannot hold run in FP32 in the
  export copy (`omnisr_export.py`): the attention's vector lengths, whose squares underflowed to NaN on TensorRT, and
  the image mean, which overflowed on ONNX Runtime CUDA from 256x256. On 5 trained weights at 256 px against PyTorch
  FP32: TensorRT Fixed and Dynamic at 56.7 dB or better on crops, no NaN or Inf; ONNX Runtime CUDA at 58 dB or better.
  Before, on TensorRT: 4x-OmniSR-DF2K 40.4 dB, 2xAniScale2 all NaN on 2 of 8 crops. FP32 -> FP16 ms at 256: 2xHFA2kAVC
  10.40 -> 5.47, 4x-OmniSR-DF2K 11.64 -> 6.44, 2xAniScale2 16.44 -> 5.44. 4x-OmniSR-DF2K drops to 53.1 dB at 64x64. No
  export layout reached 15 %. An ONNX converted earlier keeps its Precision and both faults until converted again. 5
  of about 20 weights tested, one with window 16.
- 2026-11-06 **Convert To ONNX: more architectures follow the input's size or load at all**, export side, no fork
  change. GFISRv2 and FIGSR load in ONNX Runtime (`fourier_export.py`; opset 17 or newer, older refuse clearly):
  81-114 and 103-136 dB against PyTorch FP32 on the CPU. The LDA upsampler (`lda_export.py`; GFISRv2, FIGSR, GaterV3,
  FDAT), DITN, HVICIDNet, FBCNN and Uformer (`traced_size_export.py`) follow the input's size: real DITN and FBCNN
  84.9-146.7 dB; LDA on trained bodies with a random head 99.6-145.5 dB. No trained LDA, HVICIDNet or Uformer weight
  was found: those are UNTESTED on trained weights. Older exports must be converted again. Uformer through ONNX
  Upscale Image matches PyTorch on non-square images. Convert To NCNN says GFISRv2 and FIGSR are not supported. Build
  Engine takes their DFTs (Transposes put them on axis -2; TensorRT's own plugins registered at build): against
  PyTorch FP32 at 256 px, FIGSR FP32 125.3, 130.8 dB, 19.63 ms, FP16 43.17, 50.21 dB on crops; GFISRv2 FP32 109.3,
  104.7 dB, FP16 33.25, 23.48 (a manga weight drawing screentone: plumbing only). GaterV3 and SMoSR do not refold per
  call (counted: 0).
- 2026-11-06 **Older DySample ONNX files are flagged.** `nodes/impl/onnx/frozen_grid.py` finds the constant pair on
  DySample's path to GridSample; Load Model, Load Models and Build Engine warn and go on, naming the input size the
  file is right at when it can. A model right at every size is not flagged, nor a fixed-size input. New
  `NodeContext.warn` sends `execution-warning` on both executors, and individual runs now pass that event on, since
  `/run` reuses their cached output. Remade pairs through Convert To ONNX: 36 of 36 pre-fix hit, 0 of 32 post-fix. A
  local library: 1 of 72 hit, a true one (`4xHFA2k_ludvae_fp32-opset17.onnx`, frozen at 3x3, 19-42 dB beyond 3 px). AI
  Upscale and Build Program do not warn; engines already built cannot be detected.
- 2026-11-06 **Build Engine nests the dense blocks of existing FP16 ESRGAN ONNX files** (`tensorrt/dense_blocks.py`):
  a file converted before the nested export gets the new conversion's graph at build time (protobuf-equal on 4 real
  weights, 2 of them real opset-17 files). A block is matched whole, and a file with any other join stays as it is (a
  TSCUNet with dense blocks inside). FP32, BF16 and mixed stay plain. The engine key changes only for a rewritten
  file. Session ms, plain file -> rewritten (reconverted), same session, 512x512 Fixed: 4x-UltraSharp 70.08 -> 53.39
  (53.59), 1x_Saiyajin 63.16 -> 46.65 (46.86), 4x_BSRGAN 68.21 -> 54.13 (53.93), 1x-UnResize-V3 62.91 -> 47.40
  (47.34). Engines 73 -> 36 MB, builds 31-42 -> 19-23 s. Hash-equal to the reconverted route under one shared timing
  cache; cold builds differ as any two builds do. Output against plain FP16: 62.6 to 70.3 dB. MIGraphX is untouched.
- 2026-11-06 **Upscale Image (TensorRT): the GPU no longer waits for an item's host work.** Items of one tile share a
  session: only enqueues take turns, two slots keep the next tile queued, conversion and clip run beside the GPU. Per
  image in the item window, one paired run, RTX 5090, outputs bit-identical: Typescale 21.33 -> 17.13 ms, anime6B 5.20
  -> 4.35, x4plus 14.20 -> 13.26, UltraSharp 14.24 -> 13.35, MangaScaleV3 13.54 -> 13.02, Saiyajin 12.86 -> 12.37,
  UltraSharp 512 53.47 -> 51.25. Serial submit and collect stays flat, and a chain limited by decode or save gains
  less. The clip runs in place, the output scan across the pool, and BF16 engines convert natively (anime6B BF16 11.39
  -> 6.01 ms in the window, 13.53 -> 6.51 one by one). `SharedCache.use(exclusive=False)`: a session given up after an
  error closes with its last user, and the next is made only then. A session whose enqueue, copy or wait fails stops
  with a clear error, since a further enqueue can crash inside TensorRT; a refused capture leaves it on TensorRT's
  enqueue. The stream is non-blocking; captures take turns with engine and context setup and destruction
  (`gpu_setup_lock`); a thread's first tile of an image takes any error another library left. Overlap byte-exact
  against one at a time: ESRGAN FP16 3672 images, 3551 more through errors and 76 Stops, BF16 718, DUAL Light 1769.
  `bfloat16.py` removed. The native runner's GPU conversion measured no gain and a second copy stream is bounded below
  the keep rule: not adopted. UNTESTED: DRFT on a GPU, VOSR's lock on a GPU.
- 2026-11-06 **RGBA upscales skip the white pass on opaque tiles**: on an image with varying alpha spanning several
  tiles, TensorRT's node, PyTorch and ONNX reuse the black pass's output where a tile reaches the model as the same
  bits (FP16 and BF16 hold no difference over opaque pixels; an FP32-input FP16 engine counts a tile whose alpha is
  exactly 1 as the same). 4x-UltraSharp, 512 px, 256 px tiles, one corner hole: TensorRT 252.95 -> 151.78 ms (-40.0
  %), PyTorch FP16 466.84 -> 347.28 ms (-25.6 %), outputs identical. Ramp alpha, one-tile images and opaque RGBA
  (already one pass) are unchanged. Free memory is read once a second and once per image, not at every kept tile:
  RealESRGANv2 512 px with a hole 131.12 -> 46.64 ms (both passes in full 48.24).
  `backend/src/nodes/impl/upscale/white_pass.py`, `backend/tests/test_white_pass.py`. UNTESTED: reuse speed on SPAN,
  PLKSR, OmniSR.
- 2026-11-06 **Upscale Image (TensorRT) has Balanced Tiles**, off by default, bit-identical when off: same tile count,
  spread evenly, on Dynamic engines the smallest allowed size. 480x270 on 256 px engines, same session: UltraSharp
  89.29 -> 61.08 ms, anime6B 30.82 -> 20.36, DUAL Light 30.15 -> 19.61, DUAL XS 85.72 -> 49.50 (seeded weights);
  multiples of 64 limit ESRGAN elsewhere (512x512 UltraSharp 141.30 -> 141.44). Error inside the blended bands at most
  1.43x the error outside (today's layout 1.62x). `TILE_OVERLAP` stays at 16: cut 8 measured below FP16 rounding. One
  tile remains the fastest route (UltraSharp 512x512 61.05 against 141.30 ms). Frames above 512 px: arithmetic only,
  UNTESTED on a GPU.
- 2026-11-06 **Upscale Image (PyTorch, NCNN, ONNX, TensorRT) has a Tile Size output; TensorRT retries in smaller
  tiles.** The session goes on at half the size when an allocation finds no memory and keeps it for the run; Fixed
  engines and engines at their limit stop with a message saying what to do. Real CUDA refusals (an impossible size),
  both executors, a folder in flight: output bit-identical to a clean run in those tiles. Build Engine's Dynamic
  engines take activation memory by the tile: 2x ESRGAN 64-1024 px 1552 MB at any size -> 109 / 436 / 1745 MB at 256 /
  512 / 1024 px, same speed (512 px 58.41 -> 58.47 ms), bit-identical on that model; +12 to +29 % at the largest size.
  Cached Dynamic engines rebuild once. UNTESTED: a really exhausted card (Windows may spill instead of refusing), the
  flag on HAT and DAT. The native runner shows the size but does not retry; DUAL and DRFT builders do not set the
  flag.
- 2026-11-06 **PyTorch Upscale Image can run through a TensorRT engine** (Use TensorRT Engine, off by default; off is
  the old path bit for bit, 72 cases, and imports no engine code). A model on `FP16_ARCHS`, DUAL or legacy DRFT gets
  one Dynamic engine up to 512 px, built once through Convert To ONNX's and Build Engine's own code and kept by its
  weights in the Engine Cache's folder under that one 16 GiB limit, also with the Engine Cache setting off (about 43
  MB for a 4x ESRGAN). Any failure gives the image to PyTorch with a warning, bit-identical to PyTorch; an image that
  fails on PyTorch too keeps the engine; a kept file TensorRT refuses is built again once. The CPU setting is
  honoured, infinity is seen before the clip, the Tile Size output tells the engine's tiles, and ESRGAN, Compact, SPAN
  and PLKSR take images down to 16 px (RealPLKSR at 40x40 70.42 dB against PyTorch FP32). 512x512, same session, 50
  iterations, median: 4x-UltraSharp 139.8 -> 63.2 ms, NMKD-Siax 136.9 -> 59.8, 1x_Saiyajin 120.4 -> 48.3; against
  PyTorch FP32 61.1 to 61.7 dB worst of 8 crops (PyTorch FP16 66.5 to 67.7). First run about 60 s, after a restart
  1.50 s through a real backend with no export or build. Through the node function on a GPU: OmniSR, RealPLKSR, SPAN,
  Compact 68.5 to 77.9 dB. UNTESTED on a GPU: HAT, DAT, DUAL, DRFT and the entries new to the list; a real refused
  file; real infinity.
- 2026-11-06 **ONNX first run: the size probe no longer waits**: on a GPU provider the probe's own CPU session was
  38-59 % of Load Model to first tile. Its requirement is now kept in `<storage>/onnx-size-probe` by content (1024
  files; a damaged or stale file is probed again; a passing memory failure is not kept), Load Model and Upscale Image
  start the probe in the background, and a model's bytes are hashed once. Same bench job: ESRGAN FP16 on CUDA 3.81 ->
  2.97 s first ever, 1.66 s once kept; FP32 2.51 -> 1.79 / 1.51 s; TensorRT with a cached engine 2.19 -> 1.46 / 1.36
  s; same outputs. Real backend, restart then Run: 3.833 -> 1.466 s. New cost: Load Model runs each new model on the
  CPU once (ESRGAN 2.4-3.4 s, HAT 7.9 s, DAT2 15.5 s wall), also for someone who only builds an engine; the CPU
  provider starts nothing early.
- 2026-11-06 **Load Videos** (`chainner:image:load_videos`, Video Frames): a folder of videos in one run, upstream
  #3032 and #3077. `backend/src/folder_videos.py` runs the part of the chain it feeds once per file on the run's
  executor, with a Load Video in its place; everything else runs once, so a model and its engines stay loaded. Hashes
  equal to single runs. Deferred errors or Stop on first error, no partial outputs, whole-folder progress by file
  size. A run that would write two videos to one file, across folders too, fails before anything else writes: each
  file's final path is compared, running the pure scalar nodes on its values. Shared nodes' chain cleanups run at the
  run's end, after any node Stop left running. Stop does not wait for the listing or that check (behind an 8 s listing
  7.2 -> 0.012 s). A collector's output into what the file list reads is refused, naming the input. Per file, warm,
  same session: native 84 ms against 94 for single runs, Python 92 against 90. Test:
  `native/tests/test_folder_videos.py`. UNTESTED: engines staying loaded across files on a GPU.
- 2026-11-06 **Save Video gains AV1 (SVT-AV1), ProRes and H.264, H.265 and AV1 NVENC in Advanced mode.** NVENC hides
  without NVIDIA and fails with a clear message. Colour tags verified on all; HDR10 metadata goes into AV1 and H.265,
  while ProRes and NVENC warn that it is left out. ProRes 4:2:2 refuses an odd width (the encoder damages the last
  chroma column; 4444 is clean). AV1 refuses frames under 16 pixels on a side (SVT-AV1 crashes FFmpeg). NVENC refuses
  odd sizes, 10-bit RGB and above 10 bits. Tests in `backend/tests/test_save_video_encoders.py`. Only an RTX 5090 ran
  NVENC; players other than FFmpeg's decoders are UNTESTED.
- 2026-11-06 **Small image nodes**: Crop to Size, Pad to Square, Image Diff, Normal to Height (least-squares
  integration, tileable or mirrored; pixels that are no normals count as flat; slopes limited to 10; Height Range
  output in pixels; SciPy's DCT/DST, imported on first use) and White Balance (Gray World, White Patch, Neutral Color,
  Temperature and Tint, in linear light; refuses a color that would need a factor above 64). Python nodes, no native
  kernels (at most 0.32 s at 4K, single runs). Memo allow-list 21, speculation list 123.
- 2026-11-06 **Blend Images Color and Luminosity** (`cn_blend_images`, identical at every ISA level); **Resize Magic
  Kernel Sharp 2013 and 2021** (filter values 13, 14); **Save Image EXR** through OpenEXRCore (float32 or half,
  lossless codecs, round trip exact, chunks compress in parallel, ZIP 1080p RGBA 250.16 -> 26.30 ms from one CPU to
  16, same bytes). Tests: `test_blend_non_separable.py`, `test_magic_kernel.py`, `test_exr_encode.py`. EXR float32 PIZ
  at 1080p and up was read back only by the vendored library (FFmpeg refuses those); other viewers UNTESTED.
- 2026-11-06 **Colour profiles and metadata**: Load Image and Load Images have Convert To sRGB (off by default) and a
  Metadata output; Save Image takes it and writes ICC, EXIF and DPI (PNG, JPG, TIFF), ICC and EXIF (WebP, AVIF), DPI
  (BMP). Matrix and curve profiles convert at any depth within 1/255 of ImageCms; untagged and sRGB images stay
  bit-identical (650 outputs against base, 0 differ); unconnected chains keep their native paths. Tags that cannot be
  read or written warn and are left out, and stop no load and no save. Tags are read as Save Image writes, so a chain
  that overwrites its source first writes what the file then holds (the output's docs say so). 12 MP Adobe RGB convert
  752.5 ms. Code: `nodes/impl/colour_profile.py`, `nodes/impl/image_tags.py`.
- 2026-11-06 **VapourSynth nodes take images**: methods measured to look only at the current frame
  (`frame_methods.json`, `native/tools/vs_frame_probe.py`; 256 of 500, vs-jetpack 2.2.6, VapourSynth 81) take an image
  at Clip and give an image; temporal ones and those with signed results refuse at connect through a type that follows
  Method. Exact float32 round trip; alpha kept unless pixels move; temporal radius forced to 0; methods that read red
  alone run through YUV444PS, per-plane masks on luma. One process per chain run, same session: 155.7 -> 11.3 ms
  (Invert, 64x48), 285.8 -> 14.4 ms (box_blur, 512x512), 723.6 -> 48.0 ms (three nodes), 317.7 -> 49.8 ms (1080p),
  measured before the review's route fixes, not re-timed. Only inputs that take images are judged by their evaluated
  type when connecting. UNTESTED: GPU routes.
- 2026-11-06 **Video Float RGB is the exact matrix, read and write together.** libswscale's 16-bit RGB reaches white
  at 65280/65535 at 8 bits, limited YUV at every depth. White read 0.99615 and a saved 1.0 became Y 236. Load Video
  multiplies by the format's gain in the interleave pass, and Save Video scales 1.0 to 65280 for YUV or gray output
  and for H.264 or a container's default encoder, 65535 for an encoder that takes RGB. 8-bit limited against the exact
  matrix: max error 0.00388 -> 5.7e-5, interleave 3.145 -> 3.131 ms at 1080p. Every Float RGB chain changes by at most
  1 code of 255 before the encoder; decoded H.264 can differ by up to about 30 codes on rare samples (inferred: the
  encoder's decisions amplify the input change; 90.4 % of samples equal, mean signed -0.001). 8-bit video-to-video can
  change Y codes by at most 1, in pixels above white that the old read kept by accident; 4:2:0 output still moves Y by
  up to 7 codes through chroma resampling, as before. **Limited-range gray video** reads black as 0 and white as 1
  (code 16 read 0.0627, 235 read 0.9216): an offset beside the gain, exact at 8 to 16 bits through PyAV, the pinned
  CLI and FFmpeg 8.1, interleave at 1080p 2.159 -> 2.193 ms. Only gray video tagged or chosen limited changes, by up
  to 20 codes of 255. Also: Optimize and Interpolate keep pixel-unshuffle models running, gray PNG video reads in RGB,
  Get Native Resolution takes yuva video and refuses plane stacks, Optimize Model's output keeps its input's type.
- 2026-11-06 **Denoise on the CPU**: non-local means in blocks of 16 rows by about 512 columns with planar sums,
  window sums in doubling passes and a weight table that ends at its first 0; `denoise_avx2.c`, `denoise_avx512.c`
  (one and two channels), driver `denoise_block_shared.h`; Lab split and merge on the pool. Outputs identical at every
  `CHAINNER_C_ISA` level and to OpenCV. 1 MP RGB, CPUs 0-15, 50 paired iterations: avx512 373.49 -> 45.30 ms, avx2
  390.60 -> 56.92 ms, scalar 365.86 -> 164.63 ms; 64x64 7.33 -> 1.00 ms.
- 2026-11-06 **Get Native Resolution finds one-axis upscales**: 1440x1080 stretched to 1920x1080 now reads 1440x1080
  with the kernel, not a meaningless height. When the core has no supported grid and only one axis shows a grid,
  including a fold scan past the search's end, the other axis stays at the source's size and the frame is judged on
  the scaled axis (`searchUnscaledAxis`, `getnative_kernel.cpp`); the vendored core is unchanged. Held-out plants:
  one-axis right 18 of 18 (0 before), two-axis byte-identical 18 of 18, native 9 of 9. One-axis chain about +2 % net
  of a same-session floor (paired, loaded machine; the quiet gate is UNTESTED).
- 2026-11-06 **Nodes against planted truth.** Get Native: samples past the last frame's timestamp read it, samples of
  one frame count once. Normal Map: Tileable padding covers blur plus filter; Scale 0 is flat (`color_ops.c`). Frames
  to Clip: FFV1 level 1 below 7 px a side. Align Image: flow scaled on each axis (80 rows on real weights: 24 up, 14
  down, mean +0.37 dB over the 38 changed; rows that fell had the old scale cancelling the network's own shortfall),
  small images run, Wide Search skips the homography on images it cannot fit, no-homography fallback, download
  timeout, clear error for a non-zip. Guided Upscale: Stop and Pause reach the training and the prediction; prediction
  in bands of 16,384 pixels, matching the whole guide to 1e-7 (256x256 peak 1596 -> 394 MiB); a Tile Size input, 80 px
  by default up to 2048 (3 RGB images, one tile against 80: mean +2.50 dB, one loses 0.89 dB; time grows with the
  source's area, about 100 tiles at 512 px, computed; old chains now run with 80). UNTESTED: Align and Guided on a
  GPU.
- 2026-11-06 **Zipped generators and loop order**: a generator's per-run reader before its loop no longer releases it;
  the loop counts as a read on both executors, the body runs once, and the generator's object ends with its loop.
  Generator, sink and collector order inside a loop no longer follows hash seed or addresses. A loop that needs
  another loop's collector runs after it (`process.generator_loops`); orders that were already valid are unchanged.
  Loops that need each other, or a loop reading its own collector, fail with the collector message instead of crashing
  or giving an engine error. Sweep of 200 chains on both executors: 226 failing runs now pass, 174 passing identical.
  Test: `native/tests/test_zipped_generators.py` (6 seeds).
- 2026-11-06 **Stop and concurrency**: ThreadSanitizer on Linux over `parallel.c` (5.3 M calls), real chains (PNG wave
  and inflate, the NumPy pool, every decoder, video), and random Stop, Pause and server requests (5900 iterations,
  1171 rounds, 2176 real-chain runs), plus ASan and a Windows run. Two fixes: a run that failed through a Python
  reader's lazy input stayed alive for the life of the process, its error held by the engine's reader in a cycle the
  collector could not see (Save Image as TGA: 3 of 3 executors alive before, 0 after); a run's started and closed
  flags were written without its lock while Stop read them. libdeflate's lazily set dispatch pointers race harmlessly
  (upstream's pattern, left alone). UNTESTED: GPU paths.
- 2026-11-06 **Sanitizers over all native code**: `CHAINNER_C_SANITIZE=ON` (Linux, `tests/fuzz/README.md`) builds
  every module with ASan and first-party code with UBSan, for runs with the runtime preloaded. The native suite ran at
  scalar, avx2 and avx512; the backend suite and the TensorRT runner's CPU paths ran once at the default level; the
  node sweep ran at all three levels and in tiny, pairs, random, combo and colors modes. Found and fixed: kernel
  failure statuses outside `cn_status`'s range (undefined in C++ callers), and a test reading past a one-float source.
  Under ASan the scratch pool poisons idle slots and lease tails, with a positive control. Left: DirectXTex's
  `new[]`/`delete` mismatch in texconv, harmless on glibc. Not run: GPU paths.
- 2026-11-06 **Linux gaps and PNG size limits**. Skips read by reason: 68 tests without a good reason now run on Linux
  (Load Images parity on symlinks and POSIX names, rounding and denormal controls through
  `native/tools/fp_control.py`, engine video chains, colour detection, VapourSynth upscale with
  `CHAINNER_C_TEST_MODELS`). `linux_differences.json` re-checked: native results equal Windows' on 2,258 of its tests.
  Fixed: a name that is not text failed every native run and every Save Video (on Windows Save Video now says FFmpeg
  cannot take it, naming the path); a Linux record of the differences dropped the listed tests. PNG limits run at
  their exact values under ASan + UBSan (2^29 parallel 0.35 s, 2^31 pieces 1.08 s, one row over either falls back), no
  bug. New harness `fuzz_png_inflate` reaches damaged streams with checks made right: 592,602 runs clean, TSan replay
  clean. Colour, EXR, blends, Magic Kernel, Load Videos, zipped generators and Denoise (scalar, AVX2, AVX-512) pass on
  Linux. Five full runs before the last merges: no failure was a Linux product bug; AMD tests 506 passed, 92 skipped.
- 2026-11-06 **Saved and combined chains through the real `/run` and SSE**, both executors, two runs per backend: the
  four saved chains byte-identical to `4c101c73` on CPU and GPU; images, videos, engines, VapourSynth on images and
  frames, Metadata, new nodes, frozen ONNX warning all pass; idle memory shows no per-run growth. `real_use_pass.py`
  reads migration 46. UNTESTED: the app itself, Stop during TensorRT, VapourSynth on the GPU.
- 2026-11-06 **Merged tree `d73dc50f` against base `4c101c73`**: no clear slowdown or memory growth on backend start
  (ready 1.19 -> 1.18 s, /nodes 6.11 -> 6.20 s, idle private 2985 -> 2986 MiB), 13 CPU chains, or 4x-UltraSharp at 512
  (TensorRT 59.9 -> 55.4 ms, FP16 173.7 -> 173.0 ms, outputs and ONNX identical). split-channels and palette-dither
  read +5 to +9 % on 65-100 ms runs, cause not found. Video outputs differ only by the Float RGB change.
  `resize-real-256`'s saved request fixed. Merges after that head were not re-timed.
- 2026-11-06 **DUAL engines' VRAM after free: both causes proven, neither in chaiNNer-C** (RTX 5090, TensorRT
  11.4.0.106; no code change). The first 128 MiB: the CUDA driver keeps it from the first load of any module that
  calls device assert (`__assertfail`, here TensorRT's `xmma_gemm` kernels) until the process ends: a small PTX module
  in a plain process without TensorRT keeps +128.01 MiB after `cuModuleUnload` (`vprintf` +58, neither +0); the printf
  FIFO limit does not change it. The growth: TensorRT's AOT plugin path (`nvinfer_11.dll`, RVA 0x613924) loads each
  plugin layer's module on every enqueue over the old handle and unloads only the last at close: 74 modules (XS), 78
  (Light), 1.31 MiB VRAM and 3.6-4.1 MiB host memory per enqueue, kept after session, engine and runtime are freed.
  CUDA graph replays load nothing, so a uniform-size session loses 2 sets; a Dynamic engine on assorted sizes loses
  one per new size (300 images: +392 MiB VRAM, +1,216 MiB host). Every allocation of ours and the engine's own modules
  are freed (CUPTI ledger).
- 2026-11-06 **Checks, no code change.** ONNX Runtime's CUDA provider runs headless on an ESRGAN file, with and
  without torch imported, with a cuDNN for CUDA 12 on PATH; all CUDA DLLs load from PyTorch's folder (`72e2b913`).
  Frames to Clip's 16 slices on the pinned FFmpeg 9.0.2, against plain FFV1, which already picks its own slices: 360p
  2.25x, 720p 1.36x, 1080p 1.06x, 4K 0.88x, frames identical through BestSource and FFmpeg. These replace the figures
  from FFmpeg 5.1 (2.7x). Lens Blur's kernel plan is at most 1.4 % of a call, so no plan cache; its convolution grain
  is neutral at 4096-262144.
- 2026-11-06 **Electron trim**: a window launch no longer loads the command-line run, type engine, regex WebAssembly
  or React (`main.js` 5,857,501 -> 1,431,323 bytes; `run.js` only for the `run` command); ELK, mathjs and whole icon
  packs load on first use (`index.js` 8,007,901 -> 1,872,366; the 75 icon names schemas use stay at startup); no
  default menu is built. No language change: Electron's locale files and the UI's text are as they were, and no disk
  is saved. Startup savings are estimates, not measured: UNTESTED inside Electron. Updates must copy all of
  `resources\app\.vite`. Node's compile cache was skipped. The main log rotates at 64 MiB (electron-log's default of 1
  MB could lose a long batch's first lines).
- 2026-11-06 **Electron alternatives researched: stay, no rewrite now** (no code, nothing launched). No host gives
  full visual and experience parity. CEF: same engine, the shell must be rebuilt, no measured gain, downloads larger.
  WebView2 hosts: near on Windows, but the menu bar and Alt keys differ, versions drift unless a runtime over 250 MB
  is pinned, no memory win, startup 2-3x slower in two small-app benchmarks; on Linux the system web view is another
  engine. Other engines cannot run this UI. GPUI (Rust): Discord's CTO announced a move to it on 2026-11-04, no date,
  no public build, so its parity claim cannot be checked; pre-1.0, no partial redraw on Windows, about 220-255 MiB
  more VRAM on a maximized 4K window (arithmetic, not measured), a rewrite of about 38,700 lines with no pixel parity.
  Electron is 386.6 MB of the install, 88.6 % before Python, 4.2 % with it; UI hardware acceleration is off by
  default. Dragging during a run is the one open number. The startup figure of 1.22 s (real interface 1220 ms, window
  850 ms, 2026-10-19) was measured on Electron 25.8.4, before the moves to 43 and 45; no timing exists on Electron
  45.1.0.
- RELEASED 2026-11-05 as **v0.3.9 re-release 3** (fork and tag 30b7ae11; spandrel c13): the entries below down to
  PNG. Windows suite 139,132 passed, Linux 136,246 passed, AMD tests 506 passed. Installed copy updated, 197 nodes.
- 2026-11-05 **ESRGAN, HAT and DAT convert and build in FP16 with no precision choice**. Convert To
  ONNX hides Data Type and Calibration Image for them and exports FP16 whatever PyTorch's FP16 setting says; it records
  the architecture in the ONNX (`chainner_c.architecture`), so a saved file stays seamless. Build Engine and Build
  Program hide Precision for such files and ignore a value a saved chain still holds. Files without the record behave
  as before. Against strict FP32 at 256x256, TF32 off: 15 ESRGAN weights 52.9 to 66.8 dB (8 crops), HAT-L 58.7 to 60.2,
  DAT2 65.8 to 67.7 (55.2 to 59.8 on a Dynamic engine; 4 crops, one weight each); no NaN or Inf. Session ms, FP32 ->
  FP16, RTX 5090: HAT-L 333.6 -> 69.9, DAT2 123.8 -> 43.1. Screentone models (4x eula) are not precision tests: they
  draw dots on flat areas, so pixel PSNR collapses (24.5 dB) while tone holds (46.6 to 56.9 dB after a 16-pixel blur).
- 2026-11-05 **ESRGAN FP16 exports nest their dense-block joins** (`esrgan_export.py`): `cat(cat(x, x1), x2)` instead
  of `cat(x, x1, x2)`, same channels in the same order. TensorRT's concat kernels were 25 % of a plain 512x512 run and
  are gone. Session ms before -> after, same session, one paired run, 50 iterations, median: 4x-UltraSharp 512x512 68.87
  -> 52.44; at 256x256 UltraSharp 16.85 -> 15.18, RealESRGAN_x4plus 17.37 -> 14.61, 1x_Saiyajin 15.28 -> 12.46,
  8x_NMKD-Typescale 25.14 -> 21.89, 2x_MangaScaleV3 15.42 -> 14.50, x4plus_anime_6B 6.33 -> 6.08. Output against plain
  FP16: 60.0 to 69.8 dB on those weights, close to two builds of plain (about 63 dB). FP32 stays plain: nesting made one
  1x file slower (44.08 -> 59.00 ms). Measured and left out: conv after upsampling at low resolution (the tail is 5.5 %
  of the time), folding 0.2 into conv 5, and a Triton conv kernel (estimated slower than nesting across the network).
- 2026-11-05 **Channel fixes**: Separate Alpha on a 1-channel engine; ONNX models above 4 planes read as what they are,
  and AI Upscale refuses rather than drops planes; a clear message for multi-plane models; Save Image and View Image
  (External) refuse plane counts no format holds; AI Upscale takes 4->4 and 4->3 engines through its bridge; Mixed
  calibration noise has the model's planes; NCNN multi-plane input.
- 2026-11-05 **TensorRT's GroupNorm kernel is silently wrong on small Dynamic ranges**: channels per group x max height x
  max width at or below 20,736 (right from 25,088). Build Engine writes GroupNorm out for FP32 Dynamic engines up to
  65,536, under a new cache key; other builds are unchanged. A general export change cost 2 to 6 % on Fixed and more on
  Dynamic, so it stays out.
- 2026-11-05 **PNG decode and encode faster**. Pixels and file bytes unchanged. 3840x2160 RGB, Ryzen 9
  9950X3D, all 32 threads, Load Image (reader + 8-bit to float32), medians over three paired runs in fresh processes:
  own encoder 50.2 -> 11.1 ms, Pillow 50.2 -> 14.7, OpenCV 46.6 -> 14.3, Paeth rows 97.6 -> 45.6, own encoder at
  7680x4320 311.7 -> 44.0; 48 real images of a dataset, all small, 138.0 -> 96.2 ms in total. Save Image's native path
  22.75 -> 17.85 ms, its batch path 18.92 -> 17.13 ms.
  - Independent IDAT pieces (the default encoder's bands) inflate in parallel (`png_inflate_pieces`, libdeflate's
    `libdeflate_deflate_decompress_nonfinal`): a piece must end on an empty stored block at its last byte, and a match
    cannot reach before its own output, so what succeeds is what a serial decode gives.
  - One zlib stream of 1 MiB or more, compressed 5:1 or less, up to 512 MiB of rows, inflates on the pool
    (`png_inflate.c`, the pugz technique, 2 to 6 times faster on inflate alone at 4K): threads decode from block headers
    found by search, carrying markers for the unknown history, and the result stands only if their blocks chain exactly
    from the real start to the real end and the Adler-32 holds; else serial from scratch. 2.1 to 3 times the filtered
    size in added memory; a damaged stream costs 12 to 35 % more than serial alone before it falls back.
  - Unfiltering runs as a wave across rows (`png_unfilter_rows`), Paeth in AVX2 (`png_unfilter_avx2.c`, right on all 2^24
    inputs); the store step runs on the pool; the direct path's cap rose from 64 MiB to 2 GiB of filtered rows.
  - The encoder keeps compressors and band buffers between calls (released with the NumPy pool's idle blocks at a run's
    end, or by `numpy_pool.release()` with no pool); encoded files live in pool blocks, unzeroed; the native path checks
    the folder exists before `create_directories`. Level 6, the band split and filter choices are untouched: 14,856
    encoder comparisons and 346 written files equal.
  - Accepted: a crafted stream with codes zlib refuses and libdeflate reads (length symbols 286, 287, distance symbols 30,
    31, too many codes) opens where libpng fails. Already so below 64 MiB; rejecting them would change libdeflate's hot
    decode path, and no valid file is affected.
  - Tested: Windows 139,081 passed, 0 failed (`make_oracle.py` must run first after a native change); Linux 136,195
    passed in the suite and 502 in the ROCm runtime, with 1 test corrected (its dispatch assertion depended on the thread
    count, the stream's result did not). Whole-machine run of the wave's 16 workers under load, and CPUs other than Zen 5,
    not measured.
- RELEASED 2026-11-04 as **v0.3.9 re-release 2**: the entry below, and the AMD kit's docs and messages describe any
  AMD machine reached over SSH; the notes no longer say AMD support has not run on an AMD GPU. Fork branch and tag
  `v0.3.9` at `b65e73b0` (publish commit over `353b54d1`, tree = main `a5d13353`); assets win-x64 `c3429d1a…d797`
  (176,375,072 B) and linux-x64 `355154b8…8f32` (147,426,223 B), both checked against git. Gate: Windows 139,031
  passed, 0 failed. `D:\chaiNNer` updated (7 files, tuned binaries kept); 197 nodes.
- 2026-11-04 **Cross-node fusion measured, dropped.** A 3840x2160 RGB chain, Load Image (PNG) -> Brightness &
  Contrast -> Hue & Saturation -> Levels -> Unsharp Mask -> Save Image (PNG 8-bit), nodes called in-process with
  their own output enforce, Ryzen 9 9950X3D: 117.4 ms, of which Load 52.1 and Save 23.8 (65 %). Fusing the four
  processing nodes removes at most 3 whole-image passes of 3.6 ms (Invert Color's time): 9.1 % of the chain, under
  the 10 % line set before measuring (25.7 % without Load and Save). The output range check of a node without
  `assume_normalized` (Levels, Unsharp Mask) is one more read of the image, 3.6 ms at this size (Pending).
- 2026-11-04 **A replaced engine or program frees its VRAM at once** (`tensorrt/cache.py`). Upscale
  Image's session (TensorRT's context and buffers, MIGraphX's runner) outlives the run under its node id, so after Load
  Engine or Load Program took another file, or was cleared or deleted, the old engine stayed alive with that session
  until the Upscale node ran again. A cache built on another (`built_on`, keys `(value, ...)`) now has its entries on a
  closed value released with it; a session in use goes when its run returns. RTX 5090, the process's dedicated memory
  after picking engine B in Load Engine, before any run: ESRGAN 4x engines (A 23 blocks to 512 px, B 12 to 384 px)
  3,310 -> 496 MiB; DUAL-XS 1x (fixed 256 and 192 px) 814 -> 598 MiB. After the run with B the old engine was already
  freed before (2,114 -> 2,082 MiB).
- RELEASED 2026-11-01 as **v0.3.9 re-release 1**: the entry below. Fork branch and tag `v0.3.9` at `353b54d1`
  (publish commit over `925e92ec`, tree = main `565c2cd0`); assets win-x64 `4d6acc8e…e740` (176,374,621 B) and
  linux-x64 `86df1a75…fe7e` (147,425,713 B), both checked against git. `D:\chaiNNer` updated (4 files, tuned binaries
  kept); 197 nodes. Tests: the DUAL plugin tests on D:'s Python (43 passed with the GPU hidden; the attention kernel's
  21, its GPU cases included, on the RTX 5090) and MIGraphX's DUAL tests in WSL with ROCm's Triton (34 passed).
- 2026-11-01 **DUAL TensorRT plugins faster** (`dual_na_kernel.py`, `dual_aot.py`). Neighborhood attention takes its 32
  content channels and 16 positional-bias channels as two dot products into one FP32 accumulator (no padding of 48
  channels to 64) and runs 64-query programs on 4 warps (was 128 on 8), each pair keeping its 128-query parent's key
  origin, 64-key blocks and bias frame, so every row reduces in the same order; RCA's value application runs 64
  positions per program (`APPLY_BT`, was 256), each output's four-region order kept; the residual addition before a
  dynamic router's Transpose, FP32 Cast and ReduceMean is its own BF16 plugin (`router_add`), which keeps TensorRT
  from a slow fused Add/Transpose/Cast/Mean. RTX 5090, TensorRT 11.4, DUAL-XS 4x at 512x512, BF16, optimization
  level 3: 61.99 -> 58.03 ms (attention products and `APPLY_BT`), then 58.41 -> 54.81 ms (smaller attention programs
  and `router_add`), full engines bit-identical to the old ones on the synthetic inputs tried. Through chaiNNer-C's
  build (XS 1x at 256x256, synthetic weights): the same error against PyTorch FP32 as before (relative L2 0.002148,
  max 0.00579). MIGraphX's DUAL route shares the attention kernel (4 parts per tile now; compiles for gfx1100, gfx942
  and gfx1201) and keeps 256 positions per value-application program (not measured on AMD). The engine build cache
  hashes both files, so engines rebuild; engines built before keep the old kernels.
- RELEASED 2026-11-01 as **v0.3.9**: fork branch and tag `v0.3.9` at `925e92ec` (publish commit over `0b597ed0`, tree =
  main `118e7bf5`); assets `chaiNNer-C-v0.3.9-win-x64.zip` SHA-256 `cc9c8b4c…137a` (176,373,176 B) and
  `chaiNNer-C-v0.3.9-linux-x64.zip` `74e2f937…a806` (147,424,287 B), both checked against git (every backend file
  equal; the Windows binaries equal the README hashes). Gates: Windows 139,026 passed (the unseeded VOSR streaming test
  failed; it fails 2 in 7 alone), Linux 136,107 passed at `92e056a9`. `D:\chaiNNer` updated (212 files) with its
  tuned set rebuilt from this tree (DLL `745B33D7`, graph `9050BF6E`, ext `A603F216`; exports equal to the portable
  DLL) and spandrel c12; 197 nodes. `npm run make`'s Squirrel maker fails since `58e069b1` (electron-winstaller's
  install script, which puts `7z.exe` in place, is denied); the zip maker is unaffected: Pending.
- 2026-11-01 **v0.3.9**: the version label (eight files, as v0.3.8's), and everything below since v0.3.8 re-release 5.
  The release ships a Windows zip with the AMD GPU parts (`Build.ps1 -RocmRoot -MigraphxRoot`; the seven other
  binaries byte-identical to a build without them) and, for the first time, a Linux zip
  (`native/tools/build_linux_release.sh --rocm --migraphx`).
- 2026-10-30 **CPU FP32 in channels-last order; spandrel 0.4.2+c12; AMD kit checks** (not released).
  - **Channels-last.** PyTorch upscales on the CPU in FP32 take their input in channels-last order, as BF16 does:
    4x-UltraSharp 2.212 -> 1.425 s/image, 4xLSDIRCompactN 0.0476 -> 0.0391 (Ryzen 9 9950X3D, 256x256); outputs move
    by 1 code on 0.006-0.008 % of values. MixDehazeNet keeps NCHW (15 % slower in channels-last, random weights at
    128 px; every other architecture measured within 5 % or faster). GPUs and FP16 unchanged.
  - **spandrel c12.** Depthwise 3x3 convolutions and ESRGAN's dense blocks run on HIP kernels under ROCm (adapted
    from spandrel-rocm, MIT; Windows and Linux; compiled with HIPRTC for the GPU present; any failure falls back to
    PyTorch, `SPANDREL_HIP_KERNELS=0` turns them off). Compiled here for gfx1201, gfx1200, gfx1100, gfx1101, gfx1030,
    gfx942, gfx90a and gfx908; not yet run on an AMD GPU.
  - **AMD kit.** Two steps after the chains: every spandrel architecture with random weights on the GPU in FP32,
    FP16 and BF16 against the CPU (58 architectures as 61 networks; a crash or NaN fails), and the HIP kernels off
    and on for ESRGAN and DAT2 (difference and time).
- 2026-10-29 **Type check; large 16-bit grey+alpha TIFFs** (not released): `npm run type-check:py` has no errors (25
  at `e7a16a33`, 49 warnings unchanged); 16-bit grey+alpha TIFFs above 89.5 MP keep their alpha and depth since the
  image-size limit replaced Pillow's (a test pins it).
- 2026-10-29 **VapourSynth catalog and node inputs** (not released): what always fails with vs-jetpack 2.2.6 is left
  out (23 base classes, 51 by-name kernel and scaler choices, MVTools methods that need analyze(), methods needing an
  object no input gives or giving no clip: a version-keyed table in `scan.py`, measured by running every class choice
  with VapourSynth R81); `*clips` parameters and vs-jetpack's aliases (Morpho.black_hat) take their inputs; a union
  setting gets an Advanced "as a Python value" input; expression arrays take a list per plane and norm_expr plain
  text; FilePathType is a file input; a plugin's `range` is a colour range only beside `range_in`; AI Upscale scales
  the SAR with a non-uniform model; a "*" file input takes a dropped file.
- 2026-10-29 **VOSR caches: corrupt and unused engine sets, the DiT's BF16 copy** (not released).
  - **Engine sets.** A compatible BF16 set with a missing or changed plan no longer stops every later run: the
    migration moves it to `quarantine-*` and builds afresh (two compatible sets with different plans still stop the run,
    naming both). Once the current BF16, FP8 or MIGraphX set is valid, under its cache lock, it is marked used and
    the sets of other identities unused for 14 days (1.7 to 3.5 GB each; builds sharing the storage folder keep theirs),
    `quarantine-*`, `building-*` and killed migrations' `migrating-*` folders are moved aside and deleted, each logged
    with its size; a set with an open file stays whole; another GPU's set, timing caches and unknown folders stay.
  - **BF16 copy of the DiT.** The PyTorch backend's BF16 load reads `ema_model.bf16.safetensors` (2.8 GB, made by the
    first load in 5 s, its SHA-256 remembered like the official files') instead of converting the 5.6 GB FP32 weights.
    CPU, synthetic checkpoint of the real layout: cold file cache 2.09 -> 0.89 s, warm 0.86 -> 0.19 s, a repeat load
    in one process 0.83 -> 0.42 s, peak working set 9.2 -> 3.5 GB. DINOv2 (0.23 s) still converts.
- 2026-10-29 **App and backend residuals** (branch `batch-electron`; not released): a backend failure's exit writes
  the settings first (`gui/backendFailure.ts`); the backend's start removes pip's `~` folders in the integrated
  Python (none while another process runs it) and FFmpeg setups' `.part`/`.tmp` of ended processes in the storage
  folder (`leftovers.py`); the atomic writer keeps a file's long name for its 8.3 short name (`_chainner_graph.pyd`
  rebuilt); a node module failing the same way at two starts in a row no longer keeps the node list out of the UI
  cache (`load_failures.py`); Stop ends Get Native Resolution's packet-index scan; `/packages` without its dead loop.
- 2026-10-29 **Electron 43.8.3 -> 45.1.0** (branch `batch-electron`; not released): the clipboard on Electron 44's
  promise API (`src/main/clipboard.ts`, `clipboardImage.ts`). A chain is still copied as its JSON under the raw format
  `application/chainner.chain` (pastes to and from v0.3.8); the image paste keeps its alpha rule; a screenshot goes
  through Chromium's bitmap write as before. A paste is one call (`clipboard-readForPaste`: chain, else image, else
  types). Backend output is logged line by line, a log line at its own level, bare prints included (the exit
  message's last output too keeps whole lines). `allowScripts` denies the four install scripts npm 11.19 skips. Live
  clipboard checks: Pending.
- 2026-10-29 **Model conversion fixes** (batch branch `batch-convert`; not released). PyTorch's Convert To NCNN
  refuses DRFT (both kinds) and NAFNet by name: their shapes follow the image's size, which NCNN cannot express (they
  failed with the converter's bare IndexError or KeyError). ONNX's Convert To NCNN names a tensor the model computes
  where NCNN needs a constant, and raises to a scalar integer with ncnn's UnaryOp square (2) or Power layer (ncnn's
  pow gave about 2.4e38 for a negative base). Upscale Image, Inpaint and Upscale Face run a model in another precision
  or on another device as a kept copy, and Interpolate Models no longer moves its inputs, so a shared model is never
  changed in place. Convert To ONNX keeps one export per node and model (two nodes converting one model with different
  options no longer export on every run). Type check: 60 errors to 15 (the rest in files other batch lanes change),
  each fixed at its cause; a Reorg stride given as a list is refused by name. Debug builds leave chainner_ext out
  (Python.h would link python314_d.lib). Retired: stack()'s unused sequence mode and the unused grayscale check.
- 2026-10-29 **CPU BF16 for PyTorch** (branch `batch-bf16`; not released). Use BF16 Mode (CPU), off by default, after
  Use FP16 Mode: where PyTorch runs on a CPU with AVX-512 BF16 or AMX, Upscale Image (PyTorch) runs the forward under
  `torch.autocast` BF16 (weights stay FP32, channels-last input) for models spandrel marks bfloat16-capable. Other
  CPUs show the toggle disabled and fall back to FP32 with one log line (oneDNN's emulated BF16 is 1.5-2.4x and AVX2's
  generic path 6-10x slower than FP32). Ryzen 9 9950X3D, 48 images 256x256, FP32 -> BF16 s/image: 4x-UltraSharp
  2.14 -> 0.66 (PSNR 51.7 dB mean, 50.5 min, max 6 codes), 4xLSDIRCompactN 0.053 -> 0.040 (47.3 / 46.8 dB, max 20);
  through Upscale Image 2.76 -> 0.81 and 0.046 -> 0.024. Report: `native/local/batch-20261029/lane-bf16`.
  - **Get Native Resolution on rotated video.** The light models read a video's RGB as stored, as its As stored
    plane: Load Video's upright frames made a display matrix's quarter turn a size the reader refused and its half
    turn an upside-down raster (`getnative_media.py`).
  - **VOSR temporal tests.** The removed Temporal Sequence generators and the one-thread stabilize, kept verbatim as
    oracles, gave way to golden float32 outputs of tiny clips recorded from them
    (`native/tests/golden/vosr2_temporal.npz`, 730 KB; `native/tools/vosr2_temporal_golden.py`): within 1e-5 inside
    a 4-pixel margin and 1e-4 on average, as OpenCV's flow moves a few edge pixels by up to 0.01 with AVX2 off.
- 2026-10-29 **Descale and Riemersma guards** (batch, not released). Descale's Mirror border repeats its reflection
  past the target's size (the vendored solver patched, `third_party/descale/PATCHES`) instead of refusing; parameters
  that leave a target pixel without weight (Blur 0.01 Bilinear gave black) or give non-finite weights are refused,
  naming Blur or the source window; every other result is bit-identical. Riemersma's History Length is checked
  against 682541919, above which its float32 decay base is 1.
- 2026-10-29 **ONNX and TensorRT small fixes** (batch run, not released): the ONNX size probe finds multiples
  of 12, 24 and 48; pads to at least 16 px where shape inference gives no minimum (a converted OmniSR ended the
  process below 12 px, and now matches PyTorch from 3x3 px), also where the probe gets no answer; probes a model
  that does not run once per file content (a run short of memory is not kept); concurrent first uses share one
  session, and two loads of one file share a TensorRT or MIGraphX session (one engine folder); a DRFT plugin ONNX
  loads without four tracebacks; a non-float model input is named; Build Engine sizes start at 49 px, where an
  engine tiles larger images. Convert To ONNX records the model's minimum and multiple in the ONNX, which ONNX
  Upscale Image pads to (an SST with 24 px windows was 0.82 off PyTorch at 20x23), and traces RealCUGAN at 33 px,
  which keeps its crop of odd sizes (0.10-0.14 off before; now within 1e-6 at 33x35, 37x51, 41x40).
- 2026-10-29 **VapourSynth processes, user modules, uninstall** (branch `batch-vsproc`; not released). vspipe and the
  catalog scan end with their worker, also a killed one (`nodes/impl/owned_job.py`: the worker's kill-on-close job on
  Windows, a parent-death launcher on Linux; `OwnedJob` moved there); a scan past 120 s names the import it was in.
  The host installs the user modules of `user_modules.toml` at a start with the worker stopped (the node import only
  scans), each with an optional version spec. Preview Clips checks a copy before it rewrites an open window's script,
  reports a vsview that ends at once with its log, and keeps the window over a worker restart (`vsview.pid`). An
  uninstall also removes what pip pulled in that nothing staying needs (VapourSynth: 55 distributions, ~1 GB on Linux).
- 2026-10-29 **Executor fixes** (not released). A chain of more than 500 nodes in a row is refused when its run
  starts, naming its first and last node (`MAX_CHAIN_DEPTH`; before: a recursion error from about 495 pure nodes on
  the Python executor, 990 on the engine); 500 run on both executors. On `CHAINNER_C_EXECUTOR=python`, Stop during a
  lazy read now waits for the input's node, so the run, and the server's release of its memory, end after it. A lazy
  input a node did not read (a Conditional's untaken branch) frees its upstream after its last other read, not at
  the run's end (the engine also when the upstream runs later). A serial producer run ahead runs its own node
  cleanups after itself, outside its time (Python executor; the engine already did). A connection into a lazy input
  counts as 5 nodes in a row (the engine runs 124 computed Conditionals in a row; 99 are accepted). A starved pool's
  error ends a long lazy series at once (it waited 1 s per link beyond the pool). A Save Image folder computed by a
  pure node (Text Append, Directory) is computed before a look-ahead at item 0 decides the item window, so Temporal
  Fix keeps its look-ahead (both executors). An engine fault outside a node's job names the node being run, or the
  run's phase.
- 2026-10-29 **Test storage, generator shortfall, Save Image paths** (branch `batch-io`; not released). Tests keep
  to their own storage folder (`native/tools/scratch_storage.py`: `native/local/test-storage` with the pinned
  FFmpeg, which `run_tests.py` sets up; started backends get `--storage-dir`), never `%APPDATA%\chaiNNer`;
  `backend-nodes.test.ts` fails on a node module that fails to load for any reason but a dependency not installed.
  A generator whose items end early warns once at its end, on both executors: Load Video on a cut file says
  "ended after 150 of about 300 frames" (its header's count) and the run succeeds. Save Image (and Save DDS
  Texture) refuses a Subdirectory Path or name with a `..` component or an absolute value, on both executors,
  and on Windows one with a device name (NUL, CON, COM1, ...).
  AVIF's 4:0:0 is labelled Grayscale (it drops the colour). 16-bit grey+alpha TIFFs load with their alpha at 16
  bits (OpenCV gave 8-bit grey).
- 2026-10-30 **Upscale run-ahead removed** (not released): PyTorch, ONNX and NCNN Upscale Image no longer run ahead
  of earlier items; they run on the driver again, and of the upscales only TensorRT's and MIGraphX's run ahead. The
  feature (2026-10-29: one thread per device in `device_thread.py`, combined VRAM caps in `memory_cap.py`, a
  per-input unread flag in the engine's plan, a failing run's give-up for work computed ahead, a Pause/Stop check
  before each upscale) was removed on 2026-10-30 because its gain (ONNX +31 %, PyTorch +9 %, NCNN +5 % on an RTX
  5090) did not justify the code. The offload's and VOSR's VRAM caps are as before it. Its gate behind lazy inputs
  is back for TensorRT's and MIGraphX's upscales: none is computed ahead behind a lazy input its node may leave
  unread (Save Image's image with Skip existing files on or connected, a Conditional's other branch), where v0.3.8
  computed them and threw them away (`execution_scheduler.may_go_unread`, the plan's per-input unread flag).
- 2026-10-29 **TensorRT engine caches, Windows CUDA loading, installer, VapourSynth on glibc 2.28** (not released).
  - **ONNX TensorRT cache.** ONNX Runtime 1.30's engine file name hashes tensor names, never weights
    (`TRTGenerateId`), and chaiNNer creates sessions from bytes, so two models of one architecture shared an engine:
    on the RTX 5090 the second model gave the first one's output. Each model now keeps its engines in
    `<cache>/<sha256[:32]>` (`session.py`); the timing cache (tactics per layer) stays shared. Old engine files at
    the cache root are never read again; Clear Cache removes them.
  - **CUDA on Windows.** Without torch imported (a VapourSynth script's process, or a worker whose torch failed to
    import), ONNX Runtime's CUDA and TensorRT providers took cuBLAS/cuDNN from PATH (a cuDNN for CUDA 12 aborted the
    process). A CUDA or TensorRT session now first runs ONNX Runtime's `preload_dlls` from PyTorch's `torch\lib`
    when torch is installed but not imported; without the PyTorch package the error says to install it.
  - **Installer.** `dependencies/run_chainner_pip.py` runs the bundled chainner_pip 23.2 with pip's fix for
    pypa/pip#11847: wheels resolved from PEP 658 metadata (the PyTorch indexes' torch, triton, ...) download once
    (a Windows NVIDIA PyTorch install fetched the 2 GB torch wheel twice); the progress names the file's package.
    Install All and Update All count each dependency once and only what is missing or outdated.
  - **VapourSynth on glibc 2.28-2.34.** vapoursynth-akarin 1.5.0 and vapoursynth-nlm-ispc 4.1 built for
    manylinux_2_28 from the unchanged sources, released on the forks `Rift73/vapoursynth-akarin`
    (`v1.5.0-glibc2.28`) and `Rift73/vapoursynth-nlm-ispc` (`v4.1-glibc2.28`) and pinned by SHA-256 below the glibc
    PyPI's wheels need (2.35, 2.34); same frames as PyPI's builds in 28 cases on Rocky 8, Ubuntu 20.04 and Rocky 9.
    vsview is left out below 2.34; a missing libatomic is named for VapourSynth without stopping the launch.
- 2026-10-29 **DDS on Linux** (branch `prep-dds` over `chaiNNer-C` `0291226e`; not released). Linux reads and writes DDS
  through `native/texconv`, a texconv for the options chaiNNer passes (`native/src/image_io.cpp`), built on DirectXTex
  may2026, the release Windows' texconv.exe 2026.5.8.1 comes from, with DirectXMath apr2025, DirectX-Headers v1.619.5
  and the .NET runtime's `sal.h` (all MIT), fetched at configure time and pinned by SHA-256. Each step is texconv's own
  with its DirectXTex calls; what texconv.exe does through WIC (Windows' imaging component) and MSVC's `/fp:fast`
  arithmetic are rebuilt from measurements of texconv.exe: WIC's conversions for the intermediates' pairs (5:6:5,
  5:5:5:1, BGRX, grey), WIC's Fant, nearest-neighbour and linear scaler (the default mip filter, BOX, FANT, POINT, LINEAR), reciprocal division in DirectXTex and
  texconv.exe's BC4/BC5 UNORM gradient arithmetic. BC6H and BC7 compress on the CPU, spread over threads as texconv.exe
  does with OpenMP (4096x4096 BC7 with mips: 142 s, texconv.exe `-nogpu` 171 s). The zip carries `texconv` and its
  notices (`texconv.LICENSE.txt`); Windows is unchanged (`_chainner_graph.pyd` and `chainner_native.dll` rebuilt
  byte-identical).
  - **Verified.** `native/tests/test_texconv.py`: 193 cases of chaiNNer's command lines (every offered format with and
    without mips, grey and opaque sources, compression, alpha and mip filter options, cubemap, array, volume), file
    and read-back pixels against texconv.exe's digests (`golden/texconv.json`): Windows 194 passed; Linux 184 passed,
    10 listed in `linux_differences.json` (`texconv-wic-*`). A wider matrix (1,425 writes, 2,332 reads): every read
    and every write equal except WIC's cubic mip filter on 8-bit formats and its linear one on formats it scales as
    float (not rebuilt, DirectXTex's filters there: 12 % of the levels' samples differ, by up to 2 steps, more on SNORM, whose negative values WIC
    clips to 0), a colour image saved as R8_UNORM or R16_UNORM (WIC's grey: 3.6 % of samples one step apart at 8 bits, up
    to 2 steps of 65535 at 16 bits), and the mip levels of formats WIC scales as float (SNORM, 4-bit, 4:2:2, 8-bit
    integer: 0.1 to 6 % of samples, a few steps apart at most) and their `-sepalpha`; BC3 with alpha dithering differs at some
    ties on larger images (not rebuilt). `test_dds.py` runs on Linux but for texconv.exe's own options.
- 2026-10-29 **AMD GPUs: ROCm 7.14.1 and MIGraphX 2.16.1** (branch `amd` over `chaiNNer-C` `ebf96abd`; ships in
  v0.3.9). Design, decisions and the per-area reports: `native/local/amd-20261029/`. No AMD GPU ran any of it yet:
  what needs one is an `amd_gpu` test (`native/AMD.md`).
  - **What.** Where an AMD GPU with 8 or more compute units is found and no NVIDIA GPU (`gpu.amd`: KFD topology on
    Linux, HIP in a child process on Windows, HIP's device order and visibility variables), the packages install AMD
    builds: PyTorch 2.14.1+rocm7.14 (Linux) or AMD's 2.12.0 (Windows) with only that GPU's kernel wheels (the bundled
    `rocm 7.14.1+chainner` meta package needs none of the others), and ONNX Runtime 1.30.0 with MIGraphX (fork release
    `v1.30.0-rocm7.14`; Windows adds DirectML, the default there). New: the MIGraphX package (Build, Save and Load
    Program, Upscale Image, Get Program Info) on the native runner `_chainner_migraphx` (`native/migraphx_runner/`,
    MIGraphX unpatched; nodes of domain `chainner` reach custom ops through `custom_op.h`), with DRFT's kernels and
    VOSR's qkv ported to HIP (`native/src/drft_hip`, `vosr2_hip`) and DUAL's Triton kernels compiled for the GPU as
    MIGraphX ops (`native/src/dual_hip`, Linux only at the time; on Windows since v0.3.10 re-release 1, through `triton-windows`); VOSR's MIGraphX backends (FP8 by
    MIGraphX's own quantisation on gfx942/gfx950/gfx12xx); HAT's and DAT's attention in PyTorch's fused AMD kernels
    and, exported on a ROCm torch, in the layout MIGraphX fuses (spandrel `0.4.2+c11`); an installer that uninstalls
    the builds an AMD dependency replaces (`Dependency.replaces`), and the reverse: on a computer with an NVIDIA GPU or
    none, a folder from an AMD computer gets NVIDIA's or the CPU builds in place of the ROCm PyTorch, ONNX Runtime's
    MIGraphX build and Triton for ROCm, and ROCm's runtime wheels leave with them. NVIDIA and CPU machines without AMD
    builds choose, install and show what they did before.
  - **How built.** `native/cmake/rocm.cmake` with `CHAINNER_C_ROCM_ROOT` and `CHAINNER_C_MIGRAPHX_ROOT`
    (`build_linux.sh`, or `Build.ps1 -RocmRoot -MigraphxRoot`): HIP code for every target ROCm 7.14.1 ships (25 on
    Linux, 22 on Windows). MIGraphX and the ONNX Runtime wheels: `native/tools/migraphx/` (Linux; `windows/`), the
    release build's `--migraphx` must be that build. `native/tools/build_linux_release.sh --rocm --migraphx` adds the
    four AMD modules to the zip.
  - **Verified.** HIP kernels built through HIP's NVIDIA path equal the CUDA originals on the RTX 5090 (runner kernels
    byte for byte in 1884 cases, DUAL's ops bit for bit, DRFT within one BF16 step, VOSR's qkv bit for bit); MIGraphX's
    CPU reference target compiles DRFT's whole graph with its 21 custom nodes and round-trips programs; NVIDIA's HAT-L
    and DAT2 outputs and exports byte-identical to before; the existing Windows binaries byte-identical with the AMD
    parts on or off. Full suites: Windows 137,809 passed and Linux 135,384 passed (failures were test-side only and
    fixed, reruns clean); the ROCm runtime's AMD-side tests pass on the CPU. Fixed on the way: a fresh install's first
    start (`dependencies/store.py` imported `packaging` before installing it) and CPU-only PyTorch (`offload.py`
    imported a GPU-only torch call).

- 2026-10-29 **Linux x86-64** (branch `linux` over `chaiNNer-C` `225e1765`; Linux zips ship from v0.3.9). Design,
  decisions and the per-area reports: `native/local/linux-20261029/`.
  - **What.** A portable zip (`chaiNNer-linux-x64/`, started by `chaiNNer.sh`) for x86-64 Linux with glibc 2.28 or
    newer. Every node and package that works on Windows works there, on the CPU, on NVIDIA GPUs (PyTorch, TensorRT
    with the DUAL, DRFT and VOSR plugins, ONNX Runtime with its TensorRT provider) and on Vulkan (NCNN, defaulting to the CPU where every Vulkan
    device is software). The launcher probes whether Chromium's sandbox can start and starts without it only where it
    cannot (`CHAINNER_SANDBOX`), checks the system libraries Electron, OpenCV and VapourSynth link against and prints
    the install command of the distribution (`CHAINNER_SKIP_LIBRARY_CHECK`), and gives `run` a virtual display
    (`xvfb-run`) when there is none. Copy To Clipboard calls `wl-copy` or `xclip`. DDS goes through chaiNNer-C's
    texconv for Linux (`native/texconv`, below). Limits: below glibc 2.34 VapourSynth's
    Preview Clips cannot open vsview (its Qt needs 2.34); ONNX Runtime has no Intel GPU provider there (CPU; AMD: above); C-library math
    follows the system's libm, so a few results can differ from Windows in the last bit (Linux twins of the golden
    manifests, `native/tests/golden/linux/`).
  - **How built.** `native/build_linux.sh` configures `native/toolchain-linux.cmake` and `native/cmake/linux.cmake`:
    clang and `ld.lld` 23.1.2 with the Windows build's options in clang's spelling (ThinLTO, `-ffp-model=strict`,
    dynamic denormals, `-Wall -Wextra -Werror`), `libstdc++` and `libgcc` static, seven modules (`.so`, same directories
    as the Windows ones). Windows code is untouched: Linux code sits in `#else` branches and `if(NOT WIN32)` blocks.
    `native/tools/build_linux_release.sh` builds one commit in the image `cnl-build` (Ubuntu 22.04 with a Rocky 8.10
    glibc 2.28 sysroot, since LLVM's binaries need glibc 2.34; every download pinned by SHA-256), checks the modules
    with `native/tools/check_glibc.py`, and makes the zip with `npm run make-linux-zip`. The Linux Python stack is
    `native/python-stack.lock.linux.txt`, provisioned by `native/tools/provision_runtime.sh`. ONNX Runtime is the fork
    release `v1.30.0-trt11`'s `1.30.0+trt11.4` on both systems now (TensorRT 11.4.0.106; the Linux wheel is
    manylinux_2_28 and adds Hopper's architecture).
  - **Verified.** Per change: the Linux build and the tests naming the touched files, and for a shared file the Windows
    build and its tests (rebuilt binaries byte-identical to the base's, bar the nvcc-built plugin DLLs, whose bytes
    follow the checkout path). On an RTX 5090 under WSL, against PyTorch: a PyTorch ESRGAN equals the CPU at 92 dB, DUAL
    through TensorRT matches at 63 dB, a small DRFT engine at 59.6 dB and the native TensorRT runner at 101.6 dB; NCNN
    on Vulkan (software) and on the CPU equals NumPy exactly. The release build's modules need at most GLIBC_2.28 and
    load in `rockylinux:8` (glibc 2.28) and `ubuntu:20.04`; the launcher's check ran against ten bare distribution
    images (Ubuntu 20.04 and 24.04, Debian 12 and 13, Rocky 8, Fedora, Arch, openSUSE Leap 15.6 and 16.0, Tumbleweed).
    Untested: a VOSR TensorRT build end to end, NCNN on a real Vulkan GPU, AMD and Intel GPUs, the desktop clipboard and
    drag and drop outside WSLg. Full suites: Linux 135,092 passed, 0 failed (2,876 skipped, each with its reason:
    Windows-only behaviour, or a Linux NumPy or OpenCV build that rounds differently from the Windows one, listed in
    `native/tests/linux_differences.json`, where the native result was checked equal to Windows'); Windows 137,329
    passed (the unseeded VOSR2 streaming test failed once and passes alone), and all seven Windows binaries rebuilt
    byte-identical. The release zip ran a Load Image, Resize, Save Image chain headless on Ubuntu 20.04 and 24.04,
    Debian 12, Rocky 8, Fedora, Arch and Tumbleweed with byte-identical output, and started under WSLg; through
    chaiNNer-C's session code ONNX Runtime's CUDA and TensorRT providers match the CPU on the 5090. Open: VapourSynth on glibc 2.28 to
    2.34 (a glibc 2.28 build of `vapoursynth-akarin` and a PySide6 override).
- RELEASED 2026-10-25 as **v0.3.8 re-release 5**: DUAL neighborhood attention in both routes (spandrel 0.4.2+c10,
  wheel SHA-256 `24967414…b50f`, release commit `5b4ed67`; the TensorRT exporter's `DualNA` node and AOT kernel). Fork
  branch and tag `v0.3.8` at `b5ffdf03` (publish commit over `b114b776`, tree = main `bbe9ce60`); asset SHA-256
  `9B3BB1D3…1317` (zip verified against git, 0 mismatches). Gate 137,277 passed with c10 installed (matting test
  skipped; the first run's one failure, `test_streamed_full_vae_matches_direct_encode_decode`, is an unseeded test that
  fails by shard order, passed 8/8 alone and in the repeat). `D:\chaiNNer` updated (15 files; tuned binaries kept) with
  spandrel c10 installed; 197 nodes.
- RELEASED 2026-10-24 as **v0.3.8 re-release 4**: the worker leaves expandable_segments out of an inherited PyTorch
  allocator setting (a user's FIGSR and FDAT runs failed `!handles_.at(i)` after one out-of-memory tile), and spandrel
  0.4.2+c9 (released GaterV3 models load; GaterV3, SMoSR, FIGSR and GFISRv2 checked against resselt). Fork branch and
  tag `v0.3.8` at `b114b776` (publish commit over `e30ea7b2`, tree = main `63755028`); asset SHA-256 `CA52B08E…B421`
  (zip verified against git, 0 mismatches). Gate 137,269 passed with c9 installed (matting test skipped while training).
  `D:\chaiNNer` updated (4 files; tuned binaries kept) with spandrel c9 installed; 197 nodes.
- RELEASED 2026-10-24 as **v0.3.8 re-release 3**: spandrel 0.4.2+c8 (WSR, GaterV3, SMoSR, FIGSR, GFISRv2; FDAT's
  DySample without CUDA). Fork branch and tag `v0.3.8` at `e30ea7b2` (publish commit over `f426929a`, tree = main
  `a0cfcee2`); asset SHA-256 `82787155…68C0` (zip verified against git, 0 mismatches). `D:\chaiNNer` updated (2 files;
  tuned binaries kept, native code unchanged) with spandrel c8 installed; 197 nodes; the five archs load there.
- RELEASED 2026-10-24 as **v0.3.8 re-release 2**: the entry below. Fork branch and tag `v0.3.8` at `f426929a` (one
  publish commit over `89f4b2c8`, tree = main `6087e480`); asset SHA-256 `88918C1C…1F2C` (`npm run make`, Node
  24.22.1; zip verified against git, 0 mismatches; no `Squirrel.exe` in the zip). D: installs updated with backups;
  `D:\chaiNNer` has the tuned icelake + ThinLTO + PGO set rebuilt from this tree (DLL `ADA0B089`, graph
  `67E83E91`, ext `A603F216`, exports identical to the portable DLL); both installs serve 197 nodes.
- 2026-10-23 **Get Native Resolution, Linear Light, FFmpeg 9.0.2 pin, Electron 43** (five parallel branches and an
  integration branch, CPU only, 19:38-23:30 GMT+8; merged into `chaiNNer-C` after one review, released above).
  Design and reports: `native/local/getnative-20261023/`.
  - **Get Native Resolution** (Image Dimensions > Utility): Vernier 0.2's analysis core vendored unchanged
    (`native/third_party/vernier`, MIT), built per ISA level (baseline and AVX2, identical JSON; AVX2 20 % less time
    per frame); Image and Video tabs, the node opens the video itself (even / keyframes / scene midpoints / typed
    times, 1-200 frames). Same verdicts, grids, kernels and scores (1e-6) as Vernier's `nativeprobe.exe`. The
    kernel is decided across the frames on the voted grid from least-squares scores (a preset needs 7 or more
    frames consistently in its favour; the free bicubic is a nuisance model): planted set 297 frames correct and
    supported against Vernier's rule's 85, none wrong; held-out episodes E04-E11 Bilinear supported 8 of 8
    (Vernier's rule 0); grids unchanged against the earlier record. E01, Keyframes, 50 frames: 1488x837 Bilinear,
    kernel supported over 25 frames, 113 s, 0.37 GB. Detector 2.32 s per 1080p frame against nativeprobe's 2.54 s.
    The outputs carry the stored (encoded) values; light models add evidence to Status and Report only.
  - **Linear Light** on Descale and Resize: a checkbox (off) and a Transfer dropdown (Gamma 2.4, sRGB, Gamma 2.2,
    BT.709 camera; Descale defaults to Gamma 2.4, Resize to sRGB), signed and unclipped; Resize converts the colours
    before premultiplying alpha with every curve. Kernel with a polynomial power and an AVX2 unit, identical at every
    level (4.1 MP RGB, Linear Light on: Resize Lanczos 42.9 -> 28.9 ms, Descale Bilinear 44.7 -> 34.3 ms).
  - **FFmpeg 9.0.2** pinned per platform by version and SHA-256 (BtbN Windows/Linux, evermeet macOS x64,
    martin-riedl macOS arm64) with a `chainner-ffmpeg.json` marker: an older copy of ours (the 5.1.2 download, an
    older marker) stays in use while the update downloads in the background (connect and read timeouts) into
    `ffmpeg.staged`, which replaces it at a later start once nothing runs the old copy; a user-replaced or newer copy
    is kept. FFmpeg 7 and later take colour tags from the frames: Save Video now sets them with `setparams` (without
    it every YUV video written by an FFmpeg 8 or 9 build, the D: install's included, lacked its primaries and transfer
    tags, and SD output lost its chroma siting), and Auto transcodes PCM and FLAC going into MP4 as it did on 5.1.2.
    Video suites pass on 5.1.2 and 9.0.2.
  - **Electron 25.8.4 -> 43.8.3**, Electron Forge 7.11.2, Node 24 pin for `npm run make` (`.nvmrc`,
    `engines.node`), upstream #3343 (dropdown spacer) and #3344 (dialogs open in the last folder); `ipcRenderer.sendTo`
    and `File.path` replaced. 45 waits for the clipboard rewrite (Electron 44 rebuilt the clipboard API).
    `tsc`, eslint/stylelint and vitest (288 passed, 1 skipped) clean.
  - Gate (APPDATA pointed at a scratch folder holding the pinned 9.0.2; `test_matting_complete.py` not run):
    136,917 passed, 0 failed, 493 skipped (three `run_tests.py` shards, and the fourth shard's files in one process
    after the corrupt-DLL load test stopped waiting on Windows' Bad Image box); `tsc`, eslint/stylelint, vitest clean.
    Binaries `chainner_native.dll` `49cf7d54…`, `_chainner_graph.pyd` `9183c102…` (after the review's fix round).

- RELEASED 2026-10-23 as **v0.3.8 re-release 1**: the bug
  and speed hunt below. Fork branch and tag `v0.3.8` at the publish of this tree (over `f7462c74`, with a
  follow-up that drops references to people and tools from notes and comments); asset SHA-256 `7599fc12…5e47` (`npm run make`, Node 25.2.1; zip verified against git, 0
  mismatches). D: installs updated with backups; `D:\chaiNNer` has the tuned icelake + ThinLTO + PGO set rebuilt
  from this tree (DLL `1B9549A2`, graph `3ABDCB46`, ext `A603F216`); both installs serve 196 nodes.
- 2026-10-23 **bug and speed hunt** (6 parallel lanes, CPU only, 02:35-07:20 GMT+8; wrapped at 97 % weekly usage):
  112 commits merged into `chaiNNer-C` (fast-forward to `076a1c67`, not pushed or released). Full gate on the
  merged tree: 136,026 passed, 0 failed, 492 environmental skips (`test_matting_complete.py` not run: 12-20 GB in
  one process); `tsc` and 10 frontend test files clean. Native binaries rebuilt (`chainner_native.dll`
  `94d841ad…`, `_chainner_graph.pyd` `6fdbad47…`; old ones in `native\build\held\pre-hunt-20261023\`).
  Speed, outputs identical: real CPU chains median 1.90x one image at a time, 2.40x with 16 CPUs; PNG decode
  5.2 -> 3.1 ms; median 3x3 59-84 -> 1.2-4 ms; morphology ~10x; gamma 21 -> 3 ms; Wavelet Color Fix 140 -> 11 ms;
  Blue Noise 61 -> 30 s; Save Video frame prep 1080p 8 -> 0.5 ms, 4K 28.5 -> 3.6 ms; startup node load 480 ->
  386 ms; editor per edit at 501 nodes 23 -> 1.9 ms (Node-measured); upscale CPU glue -13 to -34 %, fp16 cast 7-12x.
  Bugs fixed: OpenJPH heap overflow via HTJ2K `.exr` (upstream fix backported, PROVENANCE.md), damaged ONNX
  crashing the backend, three hangs (Blue Noise 1x2, Convert To NCNN cycles, auto-split retries), Stop letting Save
  Image write, NCNN mismatched-channel models, TensorRT RGBA engines, FFmpeg 8.x planes/CLI-reader bugs (the D: install uses an FFmpeg 8-series build), many NaN/failing nodes, malformed-chain and CLI errors.
  Intended output changes (ruling 2-5, 8): Radial gradient clamp, Chroma Key no-key/all-key frames,
  Color Transfer transparent inputs, connected numbers at a decimal tie round up, plus the NaN and channel fixes.
  Report: `native/local/hunt-20261023/REPORT.md` (lane reports, the rulings file, `integration-trial.md` beside
  it). RAM incident at 02:52 (the hunt's test runs filled 64 GB and stopped a training run): see memory rule; the run continued
  under a process-tree watchdog.

- RELEASED 2026-10-22 as **v0.3.8** ("copy push rebuild release"): the three entries below (reviewed hunt
  corrections, review follow-on decisions, Guided grayscale tiling) plus the version label. Fork branch and tag
  `v0.3.8` at the replay of main `01042e33..d453ca29`; asset SHA-256 `f8ce4228…a444` (33 entries differ from v0.3.7
  r8: 31 backend files, the main and renderer bundles; every backend entry equals the release commit, the native
  binaries equal the README hashes). Release gate: 136,190 passed with 1-2 failures per run, each a different test
  whose external program crashed at startup (texconv, ffprobe, a child Python: access violation at address 0; git.exe
  crashed the same way outside the tests); every failing test passes alone and in its whole file. Both D: installs
  updated (34 files each, backups in the session scratchpad) with a tuned build from the refreshed tree and the
  existing profile (DLL `45E7487A…`, graph `C4B4B61C…`, ext `A603F216…` unchanged) and serve 196 nodes. Pending: the
  live UI checks on the installed app: PASSED (2026-10-22: a typed number kept by Ctrl+S without leaving the
  field; File > Quit then Cancel leaves the backend running for the next Run).
- Guided grayscale tiling (2026-10-22, the approved one-line follow-up; not released): the Guided callback now
  applies its existing `_as_3d` helper to each processed tile before assembly. Final grayscale outputs still become
  HxW through ImageOutput; multichannel results are unchanged. The new non-square grayscale CPU regression failed
  before the fix while its RGB control passed; all eight focused Guided CPU cases pass afterward. Scoped format/lint
  passed. The single real GPU rerun used byte-identical source/guide arrays to the previous failed run: two tiles each
  trained for 1,000 CUDA Adam updates, then returned a finite 432x768 grayscale image. Output was visually inspected
  against its guide; no obvious tile discontinuity. The prior assembly defect is closed. No full-suite rerun or native
  rebuild (explicitly waived for this one-line Python change); the previous A1 gate stands. Evidence:
  `native/local/implementation-review/guided-fix/RESULTS.md`. No benchmark, app launch, push, release or D: change;
  review owns packaging/release and the installed UI checks after handback.
- review implementation-review follow-ups (2026-10-22, tested source `3a345dee`; not released): completed F11/F12
  document-generation protection and ordered saves by canonical target, including file-link/directory-junction aliases.
  Startup Stop stays Killing; Pause/Resume apply before execution; cleared preview bodies cannot retain cache entries
  or release their Undo successor's registration. Numeric changes use the common draft-flush path; Guided retries only
  memory failures; retry-trace/video-range checks and the two reviewed nits are complete. Final concurrency review accepted.
  Portable native build, TypeScript, scoped lint and focused checks passed. Required final CPU gate on main:
  **136,190 passed / 0 failed / 0 errors / 491 skipped (109 s, 17 shards)**, superseding the previous phase's partial gate.
  Authorized GPU functional checks: the frame 4032 YUV/model graph reduced isolated bright-color candidates
  122 -> 0 (source) and 684 -> 0 (processed), with negative pre-save floats captured and the affected crops inspected;
  ONNX and NCNN each completed one GPU tile then raised Stop; trained U2NetP Remove Background passed and was visually
  inspected. Guided performed 1,000 CUDA training updates, then failed grayscale tile assembly (2D tile into 3D buffer):
  separate pre-existing defect, no complete output, awaiting review and approval. Evidence and exact limits:
  `native/local/implementation-review/IMPLEMENTATION_REVIEW_RESULTS.md`. No benchmark/speed claim, app launch, packaging,
  D: update, push or release; actual UI/installed checks remain pending. Small segmentation weights were downloaded only
  into the task folder with explicit permission. The three code commits are `d9027458`, `18e05e25`, `3a345dee`.
- review follow-on decisions (2026-10-22, source implementation; not released): F02-F10, F13-F16,
  F18-F26 and D01 implemented, plus CV3 Guided Upscale bulk patch preparation and CV4 non-matting
  background-removal assembly through the existing C kernel. Integer image exports now saturate extended RGB
  before rounding (+Inf white, -Inf black, NaN diagnostics retained); floating-point planes and float32 TIFF
  remain extended-range. Shared collectors run once per item/group, temporal radii accumulate through serial
  dependencies, and estimated video counts no longer truncate requested ranges or lookahead. Max-tiler retries
  retain row position and output scale; ONNX/NCNN Stop reaches tile callbacks; PyTorch only retries memory errors.
  Active previews stop on cache clear, numeric drafts join the existing flush registry, and backend shutdown waits
  until window closing is accepted. Type declarations and resource-failure cleanup follow the decisions.
  Portable native builds, TypeScript check and 10 type-expression cases passed. The one permitted full-gate rerun
  finished 136,162 passed / 21 failed / 0 errors / 491 skipped (181 s). All 21 failures then passed focused checks
  after explicit frozen-reference/schema expectation adapters and correction of the duplicate color quantizer.
  At that phase's close no third full gate ran; its tree had focused validation only (superseded by the review gate above). Existing
  full Pyright issues match unchanged main (30 errors / 44 warnings); the final range-contract files check clean.
  No GPU work, timing claims, app launch, release or D: installation update. Evidence is under
  `native/local/follow-on-implementation/`; the review authority remains
  the follow-on decisions file.
- Reviewed hunt corrections (2026-10-21, source implementation; not released): execution-only video range overrides
  remove incoming connections before serialization; Save Video rejects changed frame/plane geometry before writing;
  the CLI decoder selects the first video track and reports a nonzero exit at natural EOF. Corrupt videos that
  FFmpeg gives up on now fail instead of saving their readable prefix. Stop and early range completion stay distinct.
  Pending text edits flush on blur/unmount and before Save/Run/Undo/Redo; history commits the current edit before
  stepping. Preview cache publication requires the current request's ownership. Dependency walks memoize shared
  subgraphs per walk, text counts toward the result cache budget, idle NumPy blocks are reclaimed once after a failed
  fresh allocation, and Denoise reuses at most 128 preallocated scratch workspaces without changing its integer math.
  Custom High Pass accepts RGBA blurred inputs while preserving source alpha; Stack Images and Pad describe their
  actual channel counts. PSNR now uses `10*log10(1/MSE)` before rounding: old values divided by about 2.303 give the
  new decibel values, except where old MSE rounding distorted them; saved PSNR thresholds must be updated. TensorRT
  sessions select their device on reuse/teardown; PyTorch offload rejects insufficient headroom as a CUDA memory error
  and retries restoration after evicting only non-input activations. Remote backend startup has a five-minute deadline.
  Portable native build, TypeScript check, scoped lint and the approved critical checks passed. The phase gate passed
  136,125 / 0 failed / 491 skipped (117 s) on its one permitted rerun, after updating existing fakes/metadata assertions
  for the accepted contracts; an untouched, unseeded CPU VOSR numerical check failed in the first gate and passed both
  alone and in the rerun (recorded, no VOSR code/tolerance change). Evidence: `native/local/hunt-20261021/`.
  No speed gain measured, no GPU execution, no packaged UI launch, and neither D: installation updated.
- Node times (2026-10-21, v0.3.7 re-release 8; `1a68047e`): the run engine leaves a Python body's `util.waiting` out of
  its time, as `process.Executor`'s `timed_supplier` has since `46ba54ef`. TensorRT Upscale Image had shown its turn in
  the item window's queue again since the engine (v0.3.7): 4-12 s -> 1.44-1.47 s per image on Load Images -> TensorRT
  Upscale (DRFT-L x4, fixed 768) -> Resize -> Save PNG, 20 images, RTX 5090; throughput unchanged. Not a DRFT
  regression: per 768^2 tile the chaiNNer-C engine takes 1482 ms at 2512 MHz / 478 W against the TensorRT 11.2
  engine's 1619 ms at 2775 MHz / 548 W (denser math reaches the GPU's power cap at fewer watts).
  RELEASED 2026-10-21 as v0.3.7 re-release 8 (fork branch and tag `v0.3.7` at the replay of main up to `01042e33`;
  asset SHA-256 `da475ce1…cd6c`, r7 `9AEC18A7…767B` kept locally; gate 136,111 passed); both local installs updated
  (3 files each, plus their tuned `_chainner_graph.pyd` rebuilt with the existing profile) and serve 196 nodes.
- PNG encoder (2026-10-21: "use Fast option as default"; v0.3.7 re-release 7): default 8-bit PNGs (1-4
  channels; Save Image, the Save prepare step, the viewer and UI previews) use `native/src/png_encode.c`: MinSum filter
  per row (scalar + AVX2 kernel, bytes identical at every ISA level; AVX-512 dropped: no encode-level win), libdeflate
  1.26 level 6 on up to 32 row bands over the native pool (min 16 KiB / 8 rows per band; vendored libdeflate patched
  for non-final pieces that join into one stream), own Adler-32 combine, one IDAT per band. fpng removed. Before ->
  after on 6 real images (pinned, median of 5): 42.3 MB / 174 ms (fpng) -> 33.4 MB / 107 ms (79.0 %, 0.45x fpng
  SLOWER); 1080p 8.0 -> 4.9 ms; 64x64 0.011 -> 0.061 ms. Pixels identical; bytes differ (tests compare decoded
  pixels). ECT evaluated and not adopted (74.7 % at -1, ~60x slower even on all cores). Gate 136,110 passed.
  RELEASED 2026-10-21 as v0.3.7 re-release 7 (fork branch and tag `v0.3.7` at the replay of main up to `eeb4b27b`,
  the PNG test's sample folder replayed as its environment variable; asset SHA-256 `9aec18a7…767b`, r6 `8A906053…89C3`
  kept locally); both local installs updated (5 files each) and serve 196 nodes.
- video-seek (2026-10-21, v0.3.7 re-release 6; branch `feat/video-seek`; design, plan and lane reports in
  `native/local/video-seek/`, rulings in `the decision log`, "video-seek"): Load Video gains a seek
  bar (kind `video-seek`: slider, first/-1 s/-1/+1/+1 s/last, frame box, timecode, In/Out, Run this frame), a Preview
  of the seek frame, Frames All|Range (Use limit migrates to Range 0..Limit-1; Out is an upper bound), Index = the
  frame's number, Frame Count. Exact seeks (`video.py` frame index + PyAV keyframe seek with open-GOP retry; CLI
  reader by decode-and-discard; untimed packets -> CLI): frame N of a seek equals item N of a full read in RGB float,
  RGB 8-bit and Planes (matrix: H.264 B-frames and open GOP, HEVC 10-bit, edit-list MP4, open-GOP HEVC, MPEG-TS,
  MPEG-PS, rotated, VFR). Context frames for temporal nodes (`NodeData.temporal_context`: Temporal Fix 3/3, VOSR 1/5)
  in both executors; a Range/one-frame run equals the full run at those frames (Temporal Fix, VOSR Three Frames: bit
  for bit on the GPU, both executors). A one-frame run makes Save Video write `<name>_f000123.png` (never the video);
  Auto audio re-encodes a range starting after 0, Copy refuses it. "Run this frame" = `runWithOverrides` (the editor's
  values and types untouched). Before -> after: frame 5000 of a 1080p H.264 file 19.40 -> 0.22 s (one-frame run,
  fresh process; 0.016-0.07 s warm); 4K HEVC 10-bit frame 500 6.82 -> 0.34 s; preview +1 frame ~3 ms, -1 ~55 ms.
  DPID: Resize/Resize To Side filter "DPID (detail-preserving)" with Lambda (0-10, default 1),
  `native/src/dpid_ops.c` (mergian/dpid, BSD-3), float, both executors; oracle 5.1e-07; 4K->1080p 8.4 ms (Lanczos
  7.8 ms). Gate 135,970 passed / 0 failed / 490 skipped (112 s); real use 13/13 checks on both executors over
  HTTP; the dev UI checked through CDP; GPU check above. RELEASED 2026-10-21 as v0.3.7 re-release 6: fork branch and
  tag `v0.3.7` at the replay of main up to `dc19bba6`, asset SHA-256 `8a906053…89c3` (r5 `973B2F45…9307` kept
  locally); both local installs updated (26/25 files, the portable binaries replace the tuned ones) and serve 196
  nodes.
- instant-exec (2026-10-19, v0.3.7; branch `instant-exec` from `fffe9cf7`; design, plan, lane reports, the engine review
  and the open questions in `native/local/instant-exec/`, rulings in `the decision log`, "instant-exec";
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
  limit (2026-10-19): 40,000 x 40,000 = 1,600,000,000 pixels on every decode route
  (`backend/src/image_limits.py`, `CN_MAX_IMAGE_PIXELS`), no warning below, refused above (was Pillow 89,478,485 /
  178,956,970 and OpenCV 1 << 30). E2E on the frozen tree: 7 chains x 6 steps equal on both executors. Behaviour changes
  (review). Sanctioned output changes: H4 Q1 and R8 (an FP32, Mixed or DUAL ONNX export, and PyTorch -> NCNN's FP32
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

- instant-exec follow-ups (2026-10-19 evening; approved; shipped as v0.3.7 re-releases): v0.3.7 released
  (fork tag `v0.3.7`; re-release 1 `043317fd`: a /nodes request given up while the nodes load no longer cancels the
  loading, which left a GPU install's first start without a node list); `.exr` loads (vendored OpenEXRCore 3.5.2 with
  Imath 3.2.3, zstd 1.5.7, libdeflate 1.26 and OpenJPH 0.32.0, `native/third_party/*/PROVENANCE.md`; OpenCV's EXR
  semantics, float values clipped to 0..1 at Load Image's output like every float image (review); 4K RGBA half 18-36 ms);
  the Result Cache as a setting (Settings > Python, "chaiNNer_standard": on by default at min(2 GiB, 10 % of RAM), a
  size in MiB, off empties it; `CHAINNER_C_MEMO_BUDGET_MIB` still wins); Frames to Clip's FFV1 temp in 16 slices where
  valid (sides >= 7 px, <= 2^27 bits a slice; 1080p 21.8 -> 10.6 ms, 4K 82.1 -> 37.2 ms per frame, frames equal);
  spandrel 0.4.2+c7 (DRFT's fold without a per-channel loop: DRFT-XL 2447 -> 1652 ms, outputs equal); the image size
  limit 40,000 x 40,000 pixels ; the disk Engine Cache stays off (engines are built and saved); no pip
  cache . Open: the Python executor's retirement (after v0.3.7's use); MAINTAINING section 10's checks
  re-based on chaiNNer-C (they compare with upstream; review); the hosted CI fails on 18 older pyright errors.

- VRAM after Stop, again (report 2026-10-19 20:09 on D:, a DUAL upscale with No Tiling overflowing VRAM): Upscale Image
  (PyTorch) checked Stop only between tiles, so the tile running at Stop kept its memory until its forward ended (the
  native engine runs it on in the background; minutes when VRAM spills into shared memory). The forward now checks
  Stop at each block (each module with children; per thread, the hooks only for the call) and synchronizes before
  Aborted, which leaves the forward's frames behind. ESRGAN 1280 px single tile, Stop mid-tile: VRAM back 1.1 s ->
  0.39 s after Stop; 1536 px in 256 px tiles 2.31 -> 2.07 s (one paired run). Also: `progress_controller.Aborted` is
  `api.Aborted`, so a node's own check_aborted ends its run as stopped, not as a NodeExecutionError, and a stopped
  run's result read within the grace is no longer reported as never retrieved. Final code on the GPU: 0.40 s, log
  clean. Released as v0.3.7 re-release 3 (fork tag `bdb5372d`, zip `EFE50B86...3409`). In a VRAM
  overflow (about 2 s with DUAL, No Tiling) the floor is the GPU work already running: a 1x ESRGAN at 6000 px
  freed its VRAM 6.5 s after Stop, the same with at most 1, 2 or 4 blocks queued ahead or none (the CPU waits about
  2 s in one CUDA call while Windows pages, then 4.4 s of running operations); ending the worker freed it after 1.9 s.
  The queue limit was dropped. Fixed meanwhile: a node clear right after Stop no longer returns the GPU's memory on
  the loop while the stopped node still runs (that wait stalled the loop 2.1 s).

- VRAM offload (2026-10-20: "VRAM overflow but fast, like RamTorch but better", No Tiling kept): Upscale
  Image's forward on 2 MP or more runs under a dispatch mode (`nodes/impl/pytorch/offload.py`) with PyTorch capped
  3 GiB below the free VRAM, so Windows never spills to shared memory; past cap - 2 GiB the least recently used
  activations go to pinned RAM on a copy stream and come back when an op needs them; tensors sharing memory never
  move. The 1x DUAL (C128G4D16), FP16, No Tiling: 2176 px 232.65 s -> 5.96 s first run, 4.95 s warm (4.6 s
  at the fitting rate); 1664 px (fits) 2.71 -> 2.71 s; forced to 14 GiB, 13 GiB moved, output bit-identical. The
  worker's PYTORCH_CUDA_ALLOC_CONF page-locks pinned RAM with 8 threads (about 1 s off a first overflow) unless the
  user set those options (`cuda_allocator.py`, 2026-10-24: it keeps the user's other options and leaves out
  expandable_segments, whose failed mappings broke every later allocation, pytorch/pytorch#188008). A 1x ESRGAN
  gains nothing (its peak is one op's working set). The pinned RAM goes back to Windows
  when a run ends (option B; torch 2.14's internal `_host_emptyCache`): through the backend, DUAL 2176 px
  moved 14.4 GiB and the worker held 2.3 GiB after the run. Released with the Stop loop fix as v0.3.7 re-release 4 (approval;
  fork tag `bf43dbb6`, zip `923F5467...CE24`); both D: installs updated (196 nodes; the PGO-tuned install kept its binaries).

- Legacy DRFT's TensorRT route on main (2026-10-20; Convert To ONNX of a DRFT-L x4 failed with protobuf's 2 GiB
  limit): `feat/drft-trt` (GPU-accepted 2026-10-18, never merged) merged into main (`8c4ddafc`): Convert To ONNX sends a
  legacy DRFT with LayerNorm to the plugin exporter (FP32 weights from its file under PyTorch's FP16 setting; the export
  kept like DUAL's), Build Engine builds it through the engine cache (key: the plugin library's and worker's SHA-256)
  with Tiling Optimization, and a DRFT ONNX loads as mixed precision. The generic export of a legacy DRFT is unusable:
  its window routing is decided in Python per size, which the trace froze at 32x32 (20.7-21.9 dB against PyTorch at
  other sizes), and its folded attention biases (72 x 24 MiB) passed 2 GiB. On the photos the route's BF16 body
  was 25.9-37.4 dB against PyTorch FP32, PyTorch's own BF16 alike (the 10-18 acceptance image was a mild one); PyTorch
  FP16 38.7-58.6 dB. Decision: port to FP16 (`97477974`): the window neural bias, QKV pack, OCAB relative bias and final
  crop plugins take FP16 as well as BF16, the export is FP16, and its attention drops spandrel's +-50 logit clamp (192
  Clip nodes; 43-52 dB in FP32 by itself). The export traces on the loaded model's device (`8aac9486`): CPU FP16
  convolutions are ~3,000x slower than BF16 (605 s against 37 s on the CPU; 14-17 s on the GPU). GPU check: the FP16
  engine gave NaN (the trained logits reach 48.6 million, past FP16's 65504; FP32 logits filled 32 GB at 512^2; FP16
  logits / 2^16 lost the small ones to TensorRT's flush to zero, 42.0-52.8 dB; BF16 logits alone, `74547060`: 45.1 /
  49.8 / 56.5 dB but 1486 ms at 512^2). Decision: a fused attention (4 h budget, from 04:40): TensorRT fuses the plain
  attention pattern into its MHA kernels (192 `_gemm_mha_v2`; FP16: 576 ms but NaN), so `4530d511` runs the attention in
  BF16 inside the FP16 body: through Convert To ONNX (14 s on the GPU) and Build Engine (140 s), 48.1 / 51.5 / 54.6 dB
  against PyTorch FP32 on 100x130 crops of three photos (BF16 engine 26.4 / 31.8 / 39.1; PyTorch FP16 38.7 / 54.6 /
  58.6; a probe build of the same maths 48.7 / 51.3 / 59.1), 512^2 in 678 ms (BF16 engine 708, PyTorch FP16 1693 on
  10-18); Upscale Image's session and tiler at 300x450: 46.9 dB, 0.39 s. Gate green (135,245). Released as v0.3.7
  re-release 5 (fork tag `74c6e872`, the merge published as one commit; zip `973B2F45...F9307`); both D:
  installs updated (196 nodes; the PGO-tuned install kept its binaries). i-LN DRFT: out of scope .

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
  Precision, Original mode and tile inputs, FP16, High. 2026-10-19 : GPU Memory Low offers TensorRT and
  TensorRT FP8 (were PyTorch only): weights streamed beyond a 2 GiB activation reserve, outer tiles of 512 px;
  engines built with RUNTIME_ACTIVATION_RESIZE (every set rebuilds once), so activations follow the input instead
  of the 2176x3840 profile (VAE encoder 7.6 GiB at any size before). RTX 5090, 768^2 2x: FP8 Default 11.47 -> 6.04 GB
  above idle at 0.6 s; Low FP8 2.66 GB 1.1 s, BF16 2.84 GB 1.7 s, the same peaks at a 3072^2 output; Low PyTorch
  ran out of its budget on that case. Low tiles: 34 dB from Default (textures differ per tile).
  The TensorRT plugin `vosr2_qkv.dll` is an optional native target
  (`14dfaec0…`, four reproducible builds); the FP8 graphs are hosted (`vosr2-fp8-f9e142df5846`, 3.45 GB, SHA-256
  pinned) and compiled on first use (191 s, compile 177 s). Since 2026-10-17  the set is hosted and kept
  Zstandard-compressed (`vosr2-fp8-3ebb2423591d`, format 2, 2.68 GB; each file's `zstd` entry) and unpacked only
  for a build (10 s, deleted afterwards); engines stay keyed by the graphs' own pin (`GRAPHS_SHA256`), and other
  graph sets are deleted. The maintainer uploads `out\vosr2-fp8-3ebb2423591d` to Feral before this ships. The cached
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
  (2026-10-14: B x1.4, grain follows the source): synthesized output-resolution grain replaces the enlarged
  source grain (Castorice 2x: within 2 % of the approved sample's strength, neighbour correlation 0.36 instead of
  0.72); TensorRT Three Frames, 30 frames: identical to before outside grain areas, a still background's grain
  0.9997 correlated frame to frame, 1.42-1.43 s instead of 1.39 s per later frame. Window layout (2026-10-14:
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
- Descale (2026-10-16): Image > Dimensions > Resize; vapoursynth-descale Release 12 (MIT, vendored in
  `native/third_party/descale` with one patch since 2026-10-29, `PATCHES`) under `src/descale_ops.c`, which runs its
  row and column passes on the pool in a workspace the bridge takes from NumPy. Bit-identical with the Release 12
  VapourSynth plugin (12/12 cases, kernels, shifts, fractional sizes, borders); 1080p -> 720p RGB 1.0-1.5 ms (first
  version 9.7-10.4 ms, fresh 52 MB per call). A lane-for-lane AVX-512 port (bit-identical) gave no end-to-end gain
  (memory bound) and is not built.
- Branch `chaiNNer-C` on upstream `d56e507f`. SP1 took `native/` from the `c-port` branch at `c081e940`; that branch is kept as-is, with the SP1 plan and its execution rulings on top.
- One backend: `backend/src` = B0 (426 files) plus SP2's, SP3's, SP3b's, SP4a's, SP4b's and SP4c's edits (ARCHITECTURE section 2) and rebuilt binaries (git-ignored);
  the sources are tracked with their shipped bytes. Until SP1 the package was the installed backend plus 216 overlays applied through delegates, transplants and AST pins; that mechanism is retired (it and prior history: tag `chainner-c-checkpoint-20261001`, `fede43d8`).
- Package `out\chaiNNer-C`: installed `0.25.1-nightly.2025-10-21` shell, two pinned UI patches,
  `backend/src`, the binaries and the CPython 3.14.8 runtime, built fresh at `d0280d17` for the Python-stack upgrade
  (U3, next line) and refreshed at `a831d6b6` (one file, `server.py`: the worker's unsupported-CPU refusal; the
  binaries unchanged); the last CPython 3.11 package is kept read-only as `out\chaiNNer-C-py311`.
- Pinned nightly restored (2026-10-18): the installed chaiNNer update (2026-10-08) had removed the installed
  `0.25.1-nightly.2025-10-21`; upstream's official Windows zip of it (SHA-256 `8e93b1f4…ff451`, GitHub's digest) is
  extracted to the git-ignored `native\local\upstream\app-0.25.1-nightly2025-10-21` (`independent_ui.UPSTREAM_APP`,
  README Prerequisites), which `make_oracle.py`, `package_port.py`, the DDS test's `texconv.exe` and the layout gate's
  oracle read. Its UI bundles equal the pinned ones; its backend is upstream's own, without the installed copy's local
  delta (ARCHITECTURE section 2), so the oracle no longer carries the removed GPU lease.
- Portable build (correction 2026-10-07): the shipped binaries are baseline x86-64 again and run on any
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
- U4 (GPU, 2026-10-07, under the rules: ESRGAN models only, benchmarks of about 50 iterations): the
  `default.chn` (1x Saiyajin, PyTorch CUDA fp16, the own settings) gives pixel-identical output on the port, the
  oracle and the installed chaiNNer, each with a VRAM peak of 1.22 GiB (`default-gpu-20261007T045807Z`). The throughput
  item and the helper leases: `GPU-DEFERRED.md`. ONNX Runtime's TensorRT and DirectML providers are not exercised; on a
  machine without NVIDIA, the ONNX package declares `onnxruntime-directml`, the untested path for those GitHub users.
- All 161 installed node schemas and the UI are preserved (native work covers 122 of them). Updater removed; drop repairs, the generator-once resize fix (user-confirmed), fpng for default PNGs and the broad optimization pass (CPU results, below) are installed.
- SP0 done: B0 frozen; tracked harness `native/tools/bench*.py`; noise floors for all 13 cases from 3 A/A launches (1.35-6.44%); auto-confirmation of flagged verdicts.
- SP2 done: helpers shared between concurrent native calls, partition from `(count, grain)` alone, lone gathered branch awaited directly; `parallel-branches` 48.6 to 84.8 img/s (gate `bench-20261002T033751Z`, confirmed by `bench-20261002T034508Z`: 11 of 13 cases neutral, 1 win, `video-resize-h264` unconfirmed, no regression). At 8 CPUs (affinity `0xFF`): 38.3 to 56.8 img/s, no regression (`bench-20261002T041903Z`, confirmed by `bench-20261002T042902Z`; floors from 32 CPUs).
- SP3 done: cross-item pipelining (item window K = clamp(cpus // 4, 6, 8), ordered commit, read-ahead only when safe, Save prepare/commit split); gates `bench-20261002T183704Z` (32 CPUs), `bench-20261002T185935Z` (8 CPUs, affinity `0xFF`) and `bench-20261002T185102Z` (32 CPUs, K = 1), confirmed by `bench-20261002T184404Z`, `bench-20261002T190906Z` and `bench-20261002T185850Z`: PNG outputs byte-identical, video decoded frames and stream metadata identical, Final win on all 13 cases at 32 and 8 CPUs, K = 1 neutral against B0 except SP2's `parallel-branches` gain; targets 15 of 18. Package check `bench-20261002T193254Z` (2 cases, K = 8 auto, engine ahead on every trial; no verdict claimed). Review fixes (`95369abd`, the advance the commit path waits for may start while P jobs run): gate `bench-20261002T204445Z` (32 CPUs, confirmed by `bench-20261002T205144Z`), Final win on all 13; against the pre-fix tree neutral on the five cases whose ratios moved most (`bench-20261002T210033Z`; both runs under 10–19% background load).
- SP4a done (measurement and dispatch): B3 frozen (`native/reports/baselines/B3-20261003`, SP3's final tree); CPU per item recorded beside throughput, with A/A floors at auto K (`bench-20261002T233958Z`, `bench-20261002T234603Z`, `bench-20261002T235213Z`) and at K = 1 (`bench-20261002T235813Z`, `bench-20261003T000600Z`, `bench-20261003T001348Z`): the CPU floors are 4–77%, unusable for most cases (Job-object CPU is charged per tick), so CPU per item is information only, and the K = 1 throughput floors are 1.5–8% for image cases; the `CHAINNER_C_PROFILE` timer (0.72 µs per call) and the profile (`native/reports/sp4/profile.md`; K = 1 `bench-20261003T023329Z`, auto `bench-20261003T023709Z`); the ISA probe and `CHAINNER_C_ISA`; the two AVX2 kernels in `separable_avx2.c`/`convolution_avx2.c` (B3 goldens and forced paths pass; the baseline objects of the DLL and pyd are VEX-free, VERIFIED with `isa_check.py`, except P12's dispatched `wmemcmp`, `distance` and fpng's `adler32_sse_16`); gate `bench-20261003T055413Z` (32 CPUs, B3 against `backend/src`, auto K, run on DLL `04dd3d85…0eb7`; the shipped DLL, rebuilt by the polish commit `d7564ce0`, has an identical `.text`, VERIFIED with `isa_check.py text`; pyd unchanged): 13 of 13 neutral, no flagged verdict, no speed-up claimed (`parallel-branches` +4.2%, inside its current floor of 9.4%); package check `bench-20261003T063802Z` (2 cases, `/run` + SSE: B logs `native isa=avx512 (requested auto, cpu avx512)`, A none, no timer table; no verdict claimed).
- SP4b done, Task 8 included: 13 of 18 kernel target rows are met (Task 8's five rows are counted apart: one met, four misses accepted), AVX-512 is kept for both families and the wake policy is unchanged. Plan `docs/superpowers/plans/2026-10-03-sp4b-kernels.md`; local evidence: targets `native/reports/sp4/targets.md`, `native/reports/sp4/task<N>-verdict.md`, final profile `native/reports/sp4/profile-final.md`. One reviewed commit per task, outputs bit-exact against the B3 goldens (Task 3b1 re-based only the −0 cases toward chaiNNer), each task gated against B3 at auto K (ids: 32 CPUs, then 8 CPUs where run). Tasks: 1 conversions, AVX2 unit, pool counters `5ae36362`, `e273c0c5`, `408f65c5` (`bench-20261003T083929Z`); 2 lens convolution and compose `54c36533`, `abd50e92`, `07c1f215` (`bench-20261003T113220Z`; its wall-clock flag on `video-ffv1` is no claim); 1d conversion grain 65536 → 262144 `e0e2df25` (`bench-20261003T134451Z`, confirmed by `bench-20261003T135040Z`; 8 CPUs `bench-20261003T135608Z`, confirmed by `bench-20261003T140346Z`); our counters close that flag: per 720p frame at auto K, wakes 62 → 20 and empty helper wakes 12.1 → 0.0 (`video-ffv1`), 21.0 → 0.1 (`video-h264`), and video wall-clock labels are no claims; 3 separable row tiles without an image-size temporary `fd4f40a8` (`bench-20261003T152633Z`, `bench-20261003T153718Z`); 3b lens finishing store and AVX2 normalized scan, outputs borrowed at enforce `071f3130` (`bench-20261003T171445Z`); 3b1 the enforce clamp gives +0 like the installed chaiNNer `602146a5`; 3b2 Gaussian and Box outputs borrowed, parity tests judged and corrected `6fc95ae3`, `47bb7d54`, `f75800dd` (`bench-20261003T192033Z`); 3c eight accumulator chains `dcc52694` (`bench-20261003T210445Z`, `bench-20261003T211538Z`); 3e lens convolution grain 16384 `5320dbfb` (`bench-20261003T235309Z`, 8 CPUs `bench-20261003T233148Z`; its `palette-median` Final regression was run-level, a paired A/B `bench-20261004T000536Z` read −0.6 %, neutral; harness fix `f8f886a9`); 4 and 4b morphology in place with the scratch pool, then grains `5f51cbbc`, `6c0ce1e7` (`bench-20261004T013456Z`; `bench-20261004T031513Z`, `bench-20261004T032650Z`: `palette-median` `regression (within current A/A)` at 8 CPUs, closed by Task 5's rewrite); 5 and 5b median cut, dither, exact radix select `503fbabd`, `d1016d02`, `b4f6787e` (`bench-20261004T050947Z`, `bench-20261004T052139Z`; `bench-20261004T065303Z`, `bench-20261004T070406Z`); 6 composite `05963492` (`bench-20261004T084802Z`, `bench-20261004T085924Z`); 7 resample `93be69b7` (`bench-20261004T101308Z`, `bench-20261004T102341Z`); 9 AVX-512 `cce5213b`, `e32e10ea` (`bench-20261004T124522Z`, `bench-20261004T125658Z`); 8 tail kernels as row loops `ba332436`, `5857987b`: row loops for `cn_spectral_tile`, `cn_spectral_store`, `cn_spectral_multiply`, `cn_normal_output` and `cn_caption_compose`; AVX2 units kept for the multiply (`spectral_avx2.c`, `spectral_shared.h`) and the normal output (`color_avx2.c`, `color_shared.h`), the tile's dropped (no timer win), caption not vectorized (≈ 157 of 200 µs is first touch of its fresh 2.6 MB output); goldens `native/tests/golden/tail.json` (179 cases from B3), bit-exact at scalar, AVX2 and AVX-512; per call at K = 1, before → after in µs (target): tile 522 → 261 (215), store 194 → 64 (100, met), multiply 108 → 77 (55), normal output 446 → 228 (215), caption compose 454 → 409 (120); the four misses are accepted by review: in isolation every kernel beats its target, and the pipeline remainder is first touch of fresh buffers; K = 1 loop, ms per item: `normal-map` 13.44 → 11.72 (≤ 12.0, met), `caption` 7.09 → 6.84; K = 1 A/B against B3 `bench-20261005T053336Z`: Final wins `normal-map` +24.1 %, `caption` +14.5 %; gates against B3: 32 CPUs `bench-20261005T053510Z` no Final regression (`normal-map` unconfirmed; `caption` neutral, and a same-session A/B against SP3b's tree found no attributable loss: a watch item), 8 CPUs `bench-20261005T054640Z` 13 of 13 Final wins; five cross-run gate flags on untouched cases were closed by a same-session span A/B (all neutral). Bench harness: `9a5f3908`, `7ce3d872` (storage), `551933ce`, `5c7a55bd` (Task 10a: per-side interpreter, decoded-PNG oracle). Kernels: 13 of 18 target rows met, each by the instrument its task was ruled on (`profile-final.md` section 3); missed `cn_convolution_spatial` +2.4 % and `cn_lens_compose` +12.8 % (inside the 25 % band), `cn_gaussian_f32` +99 % and +54 % (judged on the 1-CPU roofline, 840 → 406 µs), `cn_filter_morphology` +51 % (algorithm, parked); the conversion rows are met in isolation and bound in the pipeline by fresh-output first touch (INFERRED). AVX-512 kept for both families (lens timer 133.1 → 126.5 µs; gaussian at 8 CPUs 376.7 → 356.6 µs, a controller ruling; throughput neutral, no clock-down seen); `CHAINNER_C_ISA=avx2` caps them. Wake policy unchanged (design: ARCHITECTURE section 4): Task 3d's two cascading-wake variants failed (one lost K = 1 throughput on `parallel-branches`; the other left wakes per item unchanged and raised the empty share on `gaussian` and `parallel-branches`) and nothing was committed, so chunk length was the lever (Tasks 1d, 3e, 4b); its 7 policy-agnostic pool tests moved to `native/tests/test_parallel.py` (`31565cce`); empty wakes at auto K remain 43.4 % on `lens-spectral`, 41.1 % on `parallel-branches`, 31.7 % on `morphology`. Package check `bench-20261004T164638Z` (`gaussian`, `video-resize-h264`; B0 against the package, `/run` + SSE): `success=True`, B `native_isa` avx512 (requested auto, cpu avx512), A none, no timer table; no verdict claimed. Suite 106266 passed, 187 warnings before Task 8 (`31565cce` plus the review-fix edits), 106746 passed after it.
- SP3b done (decode off the producer): Load Images items are produced in two phases, a serial `describe()` that names the file and a parallel `materialize(token)` (decode and normalize) that runs as an engine job sharing P with node jobs, oldest slot first; the slot the commit path waits for bypasses P, pause and Stop; every single generator's normalize leaves the producer (video included, no speed target); zipped groups and Range stay serial; `CHAINNER_C_SERIAL_PRODUCER=1` forces the serial producer (developer setting; any other value is ignored). The decoders are unchanged (Pillow stays the JPEG decoder): Pillow's plugin registry is warmed once at Load Image's import, and OpenCV's temp-file decoders (`.hdr .pic .sr .ras .exr`, or their content signatures) take one lock. Opened on the serial producer (72–89% of the auto-K loop in 12 of 13 cases, `native/reports/sp4/profile-final.md` section 5, `bench-20261004T172037Z`, VERIFIED(run)). Spec `docs/superpowers/specs/2026-10-05-sp3b-decode-off-the-producer-design.md` (Outcome: section 9.1), plan `docs/superpowers/plans/2026-10-05-sp3b-decode-off-the-producer.md`; product code final at `5a0a8c18`, tooling fix `3091f190` (`verify_runtime` under generator-once; its rule, broadcasts minus starts for generator nodes, is superseded by the shared rule in the Task 6a line below; `runtime-20261005T025126Z`); local evidence `native/reports/sp3b/` (`task4-verdict.md`, `task7-verdict.md`, profiles). Exact (blocking; all MET): 13 of 13 bit-exact against the installed chaiNNer at 32 CPUs (`bench-20261004T235146Z`, confirmed by `bench-20261005T001901Z`) and at 8 CPUs (`bench-20261005T004626Z`, confirmed by `bench-20261005T011253Z`), the installed app untouched (before/after snapshots: 0 added, removed or changed); across Task 4, 22 runs and 2,496 trials with 0 output and 0 `sse_contract` mismatches; suites `106332 passed` at auto K (`bf618346`) and `106336 passed, 187 warnings` at K = 1 (`CHAINNER_C_ITEM_WINDOW=1`, `5a0a8c18`, native and backend tests). Gates against B3, Final `win` on all 13 cases at both counts, confirmation-run changes: 32 CPUs `bench-20261004T232346Z`, confirmed by `bench-20261004T232937Z`: `resize-real-256` +46.5 %, `gaussian` +18.0, `lens-spectral` +238.3, `morphology` +18.3, `normal-map` +9.9, `caption` +16.5, `split-channels` +20.0, `palette-median` +58.6, `palette-dither` +85.6, `parallel-branches` +27.6, `video-ffv1` +20.8, `video-h264` +13.6, `video-resize-h264` +12.6; 8 CPUs `bench-20261004T233546Z`, confirmed by `bench-20261004T234315Z`: `resize-real-256` +31.7 %, `gaussian` +47.0, `lens-spectral` +1124.9, `morphology` +69.2, `normal-map` +19.9, `caption` +16.9, `split-channels` +48.4, `palette-median` +72.4, `palette-dither` +118.2, `parallel-branches` +187.0, `video-ffv1` +116.0, `video-h264` +60.5, `video-resize-h264` +67.4 (video: no speed claim). Against the installed chaiNNer (10a, 32 CPUs, confirmation): ×10.8 (`lens-spectral`) to ×85.4 (`normal-map`); `resize-real-256` 4.2 → 306.1 img/s. SP3b's own gain against SP4b's final tree, auto K, confirmation figures: 32 CPUs (`bench-20261004T230511Z`, `bench-20261004T231046Z`) `resize-real-256` +19.3 %, `palette-median` +11.1 %, `parallel-branches` −7.6 % (within the A/A floor; accepted by review); 8 CPUs (`bench-20261005T013954Z`, `bench-20261005T014606Z`) `resize-real-256` +36.3 %, `split-channels` +26.9, `palette-median` +14.4, `gaussian` +12.8, `caption` +8.5, no regression; K = 1 (`bench-20261004T231559Z`) within noise everywhere. Same binary, two-phase against `CHAINNER_C_SERIAL_PRODUCER=1`: 32 CPUs `resize-real-256` +24.4 %, `split-channels` +10.7, `caption` +9.9 (`bench-20261004T224601Z`, `bench-20261004T225136Z`); 8 CPUs `resize-real-256` +28.4, `split-channels` +15.2, `palette-median` +42.1 (`bench-20261004T225347Z`, `bench-20261004T225955Z`). Missed (spec section 9 win targets; 32 CPUs: `resize-real-256` ≥ +40 %, `gaussian`, `caption`, `split-channels`, `palette-median` ≥ +25 %; 8 CPUs: `resize-real-256`, `split-channels`, `palette-median` ≥ +20 %): all five at 32 CPUs (`gaussian` +7.5 % and `split-channels` +2.1 %, both unconfirmed; `caption` neutral) and `palette-median` at 8 CPUs; waived as blocking by review, who records that the targets were mis-derived (they used the producer's busy share, but SP4b's window had already overlapped much of it, and they allowed no commit-path inflation). Measured cause (Task 7, `native/reports/sp3b/task7-verdict.md` section 6): with decoding off the producer, each item's time is set by Save's commit path, which handles one item at a time: it is 91–97 % of every image item and waiting for the producer 3–9 %; on `resize-real-256` Save's write and glue take 49 % and the hand-offs between the event-loop thread and pool threads 46 % (Save's lazy-input hop 21 %, the steps around Save's job 25 %), and the loop thread is busy under half the time, so the cost is latency, not loop CPU; decodes running beside the commit path slow Save's write by ≈ 0.4 ms per item, which cancels what moving Save's encode off the commit path saved. D2's 30 % clause: not tripped (+5.9 % against the same-session serial setting); the cap sweep is waived. Phase 2 (GIL-free copy out of Pillow's buffer): the trigger fired as written (2.656 ms; 1.513; 0.676 ms) and the build is deferred, since decode is off the critical path; revival rule, re-anchored to the re-profile after SP4c (SP3c closed without a lever): run phase 2 if `resize-real-256`'s auto-K loop is still above 2.0 ms per item and Save's like-for-like inflation is still ≥ 0.3 ms per item (early read: 0.380 ms, loop 2.99–3.11 ms). Profile-only timers added: `window.take_advance`, `lazy.resolve`, `loop.thread_cpu`.
- SP3c closed after its probe (2026-10-05, no code shipped): the commit path, SP3b's measured bound. Map: one image item's commit path is 3 + 3 loop/pool transitions and 22 loop callbacks; its idle cost is ≈ 0.19 ms against 1.44 ms under load (contention-amplified waiting). Probe (review's pre-registered rules; 14 runs, outputs and `sse_contract` identical in all): dropping the empty round trip for replays on the item's single chain (B1) gained 0.114 ms per item on `resize-real-256` at 32 CPUs (bench: neutral; the saving reappeared in Save's write); adding the NodeFlights/shield changes (C1+C2) gained 0.043 ms more; Save's pre-read on the loop (A1) failed its gate (0.057 ms per removed transition, gate 0.15) and its loop-side check costs 118–162 µs median. No lever produced a confirmed win, so SP3c closed: B1, C2, D (describe on the loop) and F (previews on the loop) are parked, C1 was not built and A1 is rejected. Evidence: `native/reports/sp3c/` (commit-path map, design questions, probe prep, probe verdict, arm diffs). Next (order, 2026-10-05; SP4c is done, see the SP4c line for what follows): SP4c, output allocation (first touch of fresh buffers), allocator-level only (a bounded NumPy memory handler that keeps freed blocks warm, or pre-faulting); no per-bridge buffer reuse; per-case fresh-bytes and page-fault count before targets; a byte cap with a long-video growth test. SP3b's phase-2 revival rule is re-anchored to the re-profile after SP4c, and B1 is re-tested after it. Then SP5a (clang-cl LLVM 23.1.2 against MSVC first; Pending).
- SP4c done (output allocation, 2026-10-06): a NumPy data-memory pool in `_chainner_graph` (`native/src/numpy_pool.cpp`; ARCHITECTURE section 6; `CHAINNER_C_NUMPY_POOL` in README Bench) keeps freed blocks of 512 KiB or more warm (best fit in [size, 2 × size], idle bytes capped at min(512 MiB, 10 % of RAM)), is installed per thread and releases every idle block at the end of each run; D8 reads Load Video's frames into fresh pooled NumPy buffers. Spec `docs/superpowers/specs/2026-10-05-sp4c-output-allocation-design.md` (Outcome 2026-10-06), plan `docs/superpowers/plans/2026-10-05-sp4c-output-allocation.md`; local evidence `native/reports/sp4c/` (`task4-verdict.md`, whose D2 addendum is the record for the shipped tree, and `task5-verdict.md`). Commits: pool `f9f14413`, per-thread install and release `053b8bd2`, growth tools `a9bffbcd`, threshold 512 KiB `11de844f` (test pin `097a73a9`), D8 `4cceb734` (test fix `14176509`). Threshold: 1 MiB first, which missed `lens-spectral` at 8 CPUs (+8.9 % neutral, `bench-20261005T150927Z`); D2's retry at 512 KiB (review) was kept on its four pre-committed parts. Run ids are `bench-20261005T<id>`. Stage 1 (8 CPUs): profile `194053Z`, timed `194400Z` + confirmation `195006Z`. Stage 2: growth `run-20261005T195454Z`, profiles `195635Z` (K = 1) and `200104Z` (auto, 32 CPUs), timed `200354Z` + `200920Z`, gates against the pre-SP4c tree and B3 `201415Z` + `201944Z`, `202426Z` + `203031Z`, `203409Z` + `203957Z`, `204523Z` + `205055Z`, `205643Z` + `210420Z`, end to end against the installed chaiNNer `211230Z` + `214025Z`, `221603Z` + `224330Z`. Confirmed Final wins, same-tree A/B (A = pool off, auto K), confirmation-run changes: 32 CPUs `normal-map` +17.6 % and `parallel-branches` +16.5 % (`200354Z`, `200920Z`); 8 CPUs `lens-spectral` +14.8 % and `normal-map` +14.8 % (`194400Z`, `195006Z`). `resize-real-256` at 32 CPUs missed: +3.2 % neutral (`200354Z`; noise floor +4.8 %), waived by review as non-blocking (the mechanism's ceiling, ≈ 4.6–5.1 % INFERRED, sits at the noise floor; the 1 MiB readings were +6.8 % and +4.6 %; faults 543 → 23 per item at K = 1). Video, on our K = 1 per-frame loop timer at 600 frames, never wall-clock (`195635Z`): `video-ffv1` −24.5 % (4.206 → 3.178 ms per frame), `video-h264` −21.1 % (4.622 → 3.646; the 1 MiB tree's clean third run `193901Z` read −23.4 %), `video-resize-h264` −26.1 % (4.688 → 3.464); the targets (16, 17 and 17 %) are met. D8 built and kept (review), against the pre-D8 tree with the pool on (`bench-20261006T001348Z`, K = 1, 4 pairs, 600 frames): faults per frame −683 / −680 / −681, loop timer −10.7 % (3.281 → 2.929) / −5.1 % (3.789 → 3.594) / −6.8 % (3.619 → 3.372), every pair faster; identity `bench-20261006T001723Z` + confirmation `bench-20261006T003201Z`. D8 has no runtime switch (`CHAINNER_C_NUMPY_POOL=0` does not undo it): its undo is `git revert 14176509 4cceb734`; parked minor: one extra frame-sized pool request per stream at end of stream. Identity of the shipped pool tree: 13 of 13 graphs against the installed chaiNNer (decoded-PNG compare) and against B3 (SHA-256) at 32 and 8 CPUs, `sse_contract` identical in all 22 runs, no confirmed Final regression at either count; suites `106813 passed, 2 skipped` (auto K and K = 1), `106821 passed, 2 skipped` with D8. Growth, all PASS: `run-20261005T195454Z` (4 of 4) and, on the shipped D8 tree, `run-20261006T005241Z` (4 of 4; video pool requests 12,001 = 4 per frame + 1; check (e) +0.3 MiB after the next release). Check (e) was reported, not asserted, at SP4c's close: a failed `/run`'s 64 MiB array stayed live after the response (+64.1 MiB) in reference-cycle garbage, a deviation present since B0 and independent of the pool (`native/reports/error-retention/rootcause.md`). It was fixed on 2026-10-06 by `bad1dba7` and `084d640c` (Pending, first row; `native/reports/error-retention/verdict.md`); since then (e) and the new (f) are asserted (`run-20261006T042213Z`, 4 of 4). Activation (2026-10-06, refreshed at `14176509`): `refreshed`, repeat run `verified_no_op` (manifest `009a7ef5…` unchanged), both shipped binaries hash-equal in `backend/src` and the package, and the four runtime verifiers green in port mode: `runtime-20261006T020521Z` success, `io-execution-runtime-20261006T020715Z` 92/92, `utility-runtime-20261006T020810Z` 116/116, `video-runtime-20261006T020302Z` 26/26 (65 frame PNGs byte-different, pixel-identical: D8's path). Next, in order (2026-10-06): (1) the error-retention fix (done, above); (2) the B1 re-test and SP3b's phase-2 re-check; (3) SP5a (clang-cl against MSVC).
- Parity verifiers (2026-10-06, Task 6a `62e3859c` and `8679d81a`, ruled by review): `io_execution_runtime`, `utility_runtime` and `video_runtime` test the package only with `--include-port` (without it they run the installed backend alone), and in port mode they compare under chaiNNer-C's three accepted divergences from the installed chaiNNer: generator-once (upstream re-runs a generator node per item, one extra node-start and node-broadcast each; the port runs it once and starts it exactly once), fpng PNG bytes (PNGs are compared by decoded pixels) and preview coalescing (`LatestBroadcasts`: per node, the port's broadcasts are a sub-multiset of the installed app's with no order compared, the final state is equal by `bench_oracle.sse_contract`'s merge, everything else exact). The rule lives in `verify_runtime.sse_event_mismatches`, shared by every verifier (`verify_framework_runtime` adopts it through `compare_runs`, UNTESTED: it loads models). Limit: a dropped restoration fails the rule only when it is the sole source of an output id; `bench_oracle.validate_record` and `test_execution_scheduler.py` pin the rest.
- SP1 acceptance (2026-10-02, plan Task 9): a fresh package equals v1 byte for byte except manifest, profile and `__pycache__` (55,028 files, `native/reports/sp1/`); the real package was converted in place to manifest v2 (sha256 `a2c68ce3…`), repeat run `verified_no_op`; suites pass on the single backend, 4684 + 95409 + 8; bench B0 against the new package `bench-20261001T220537Z`: no regression.
- TensorRT video chain (2026-10-08, 1 h budget): `tensorrt-video.chn` (375 frames of 4K HEVC, 960x540
  into a 4x compact engine) 9.9 -> 19.8 fps (`f985ff8a`: native tile conversion, no blend for one tile); then
  `e13ebdef`, `cn_planar_to_interleaved_f32` (exact, all ISA levels): the node 75 -> 23 ms per frame, frames identical
  (framemd5, 375). The chain stays at 19-20 fps: the video read path bounds it (Load Video + one Resize alone 26.7
  fps; ffmpeg decodes and converts at 132 fps to a file, the 25 MB-per-frame pipe into Python is the limit). By the rule the work stopped there. D:\chaiNNer rebuilt from chaiNNer-C's own UI (`npm run make`) at `e13ebdef`
  with the local PGO icelake binaries; the previous build is `D:\chaiNNer-prev-20261008`.
- In-process video decoding (2026-10-08, 1 h): Load Video reads through PyAV 19.0.1 (FFmpeg 9.0.2 in-process,
  the same libswscale conversion to bgr24), the FFmpeg CLI pipe reader is the fallback (rotated or variable-rate files,
  PyAV missing, open errors; `CHAINNER_C_VIDEO_READER=cli` forces it). Read path 26.7 -> 68.1 fps; `tensorrt-video.chn`
  19-20 -> 23.5 fps; frames identical (375 on the 4K HEVC clip, `test_video_pyav` on h264/ffv1/mjpeg/bt601).
- Rotated videos (`ba3ebc86`): metadata reports the displayed size (a quarter-turn display matrix swaps width and
  height) and the PyAV reader inserts the CLI's autorotate filters; both readers give identical, upright frames.
- Chain limit (2026-10-08): per 4K frame, CPU ms (wall ms) alone: decode + conversion 140 (9.3), uint8 -> float32
  87 (7.3), Resize 105 (4.9), TensorRT node 168 (17.2), Resize 64 (5.1); in the chain about 0.95 CPU-s per frame,
  25 of 32 threads busy. Window size (K 4/8/12) and a 4 GiB NumPy-pool cap are neutral. The item window runs the
  TensorRT node ahead (`d59b4b1d`, `execution_scheduler.LOCKED_DEVICE_NODES`, session under its lock; a deliberate call
  after the probe missed its +20 % keep rule): 23.5-24.5 -> 26.6-26.8 fps, frames identical. With identical
  output the chain is throughput-bound at about 27 fps on this PC.
- Spandrel fork (2026-10-08): the PyTorch package's Spandrel is `0.4.2+c1`, the wheel on
  github.com/Rift73/spandrel (release `v0.4.2-c1`, SHA-256 `fe884dc1…`): upstream 0.4.2 plus DRFT, DUAL and
  SST/SST-Real. `Dependency.url` installs a wheel instead of the PyPI release; a version with a local label is
  satisfied only by that exact build, so existing installs switch at the next start. The lock pins the wheel as
  pip freeze writes it (`read_lock` takes the wheel's version). The local overlay is retired (only the
  three architectures are used); the repo runtime and D:\chaiNNer carry the fork, D:\chaiNNer's old spandrel is
  in `D:\chaiNNer-spandrel-backup-20261008`.
- DUAL to TensorRT (2026-10-08, 1.5 h + 2 h): traiNNer-redux's DUAL exporter is vendored in
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
  Light x4 (a trained checkpoint) and XS x4 (synthetic, nonzero experts) at 256x256 pass on all six inputs, error at or
  under the BF16 floor, max abs <= 0.0116. The C++ plugin route (CMake + nvcc + MSVC, vendored TensorRT headers,
  TensorRT SDK Path) was removed in the commit after `cb735a43`; reverting that commit restores it (ruling ruling).
  TensorRT loads embedded plugins without `engine_host_code_allowed`, so no trust gate.
- DUAL neighborhood attention (2026-10-25, the owner's handoff `DUAL_Optimized_NA_chaiNNer_Handoff.md`): the vendored
  source is traiNNer's `b73554e5…` (context layers = neighborhood attention: each query attends its own clamped 32/64
  neighborhood; weights unchanged), and spandrel 0.4.2+c10 runs it (bitwise vs the source on CPU FP32; inference
  batches whole tile rows, capped by mask size). The exporter emits `DualNA_<key>_TRT(qkv, bq, bk; window)` at each
  context site whose trunk exceeds its window (a dynamic export at every context site); `dual_aot` lowers it to the
  `na_<key>` / `na_c<C>` AOT plugins running `dual_na_kernel.na_attention` (the research kernel generalized to
  run-time H/W and small trunks; bitwise vs it where it applies; 28672 B shared on sm 86/89/120). Manifests record
  `"context": "na-v1"`; an ONNX without it still builds, with a log warning. Build Engine refuses a dynamic maximum
  beyond the kernels' 32-bit indexing. Research input 512x692 on the 5090: engine vs the owner's accepted NA image
  MAE 0.30/255, P99 2, max 7 (the research engine's numbers); legacy 18.05 / NA 19.62 / research engine 19.45 ms;
  PyTorch FP16 67.8 -> 107.3 ms. Records: `native/local/dual-na-20261025/` (design, lane reports, review, bench).
- DUAL speed (2026-10-09): Light x4 512x512 was ~2x the 2026-10-04 DUAL3 PU2 D16 engine (14.33 ms). Per-layer
  profiles: (1) AOT kernels compiled without Triton's 16-byte pointer-alignment hint (scalar loads; core 2.13 vs
  0.156 ms per layer), fixed in `5ca20024`; (2) the exporter moved every 3x3 after a PixelShuffle ahead of it (4x work
  for Light's 64->256 tail conv), now only the RGB readout, `c0de1572` (vendored `graph.py` now differs from
  traiNNer's). Through chaiNNer-C's build, GPU-only replay: 23.93 -> 15.64 ms; gate 6 passes at 512x512, errors
  unchanged. The remaining ~1.3 ms vs DUAL3 is DUAL3X's dynamic convolutions. Postponed: dynamic-shape DUAL
  engines (ask before starting).
- DUAL dynamic engines (2026-10-09, 2 h box; branch `dual-dynamic`, WIP): dynamic plugins (`e146bc73`,
  bit-identical to the fixed-size kernels) and exporter + worker profile (`7f9140a2`). Light x4, one engine
  64x64..512x1024: gate passes at 7 sizes; +11 % vs static at 512x512. Limit: above 512x1024 TensorRT has no kernel
  for the last tail conv in a dynamic build (static 576x1024 builds; a row-split try failed). Not wired into the
  nodes or TensorRTSession (per-shape CUDA graphs) yet. Open: continue or drop the branch.
  Update (continued after the finding): `8f7f3d18` lifts the limit (each tail conv's input
  reshaped to sizes computed from the network input; an earlier review) — one engine 64x64..1088x1920 passes the gate at 8
  sizes, +3.5 % vs static at 1088x1920; `7a590eba` wires it in (Convert To ONNX "TensorRT Shape", default Dynamic;
  Build Engine profile checked against the 64-px alignment; TensorRTSession: a CUDA graph per input shape, up to 4).
  Through chaiNNer's tiler: 1080p in one tile, 197 ms per image; engine vs PyTorch on the same padded input within
  the BF16 floor. Merged into `chaiNNer-C` and copied to D:\chaiNNer (2026-10-09).
  `009e9d80`: dynamic DUAL ONNX takes any multiple of 4 (input named `input_aligned_4`, the tiler pads to it),
  so 1080p runs unpadded and matches whole-image PyTorch (max 0.0022); +3.4 % vs alignment 64 at mid sizes,
  equal at 1088x1920. Session captures a shape's CUDA graph on its second run. One unexplained dynamic build
  failure in five attempts (no log kept); rebuilding worked.
  Settings that do not apply to DUAL are hidden (2026-10-09): Convert To ONNX's Data Type, Opset and
  Verify, Build Engine's Precision. Since 2026-10-16  Build Engine's Workspace (GiB, any amount; 0 =
  automatic: TensorRT's default, up to the GPU's memory, since 2026-10-17 when 4 GiB failed an FDAT 2048x2048
  build by 8 MB; 8 for DUAL), Allow TF32, Sparse Weights and Optimization Level (0-5, default 3) apply to every
  ONNX, DUAL's worker included (its defaults stay the guide's: 8 GiB, TF32 off, dense, level 3). Build Engine's
  Tiling Optimization (2026-10-18; 2026-10-19: keep it, for DUAL's worker too; TensorRT's L2-cache tiling,
  None/Fast/Moderate/Full, default None): 256x256, level 3, FP16, on the graphs before spandrel c5 DAT2 57.72 ->
  51.79 ms with Full (build 56 -> 301 s), HAT-L 89.20 -> 83.36 ms with Moderate; on c5's (attention laid out for
  TensorRT) Full gains little, DAT2 47.40 -> 46.62 ms (build 42 -> 195 s), HAT-L 71.80 -> 71.42 ms (53 -> 221 s).
  Save Engine's
  Compress (2026-10-16; default on) writes LZMA, or Zstandard level 19 (Compression, 2026-10-17),
  and appends "_compressed" to the name; Load Engine and VapourSynth's AI Upscale decompress by the xz or zstd
  header (`nodes/impl/compressed_file.py`). Measured on the 1x DUAL Light engine: 47.5 MB -> 11.7 MB,
  +0.44 s per load, 6.6 s per save; zstd 13.2-13.6 MB at 35-60 ms. VOSR's FP8 ONNX weights (2026-10-17, 128 MB
  samples) compress to 71-83 % with LZMA, 78-87 % with zstd 19. ONNX Save Model has the same Compress and
  Compression (2026-10-17; default on, LZMA); Load Model (and Load Models, AI Upscale) reads both forms.
  Eight large ONNX upscalers became 38-89 % of their size (HAT-L 47 %). A DUAL ONNX loads as precision "mixed" (arch DUAL in
  its type); every DUAL engine, fixed ones too, names its input `input_aligned_4`, which Load Engine reads as
  "mixed". Fixed engines built before this still read as FP32. The fork's backend-tests check failed since the
  dynamic push (the ONNX-load assertion needs the native build, which GitHub does not make); it now skips there.
- Upstream bug fixes (2026-10-09, branch `fix/upstream-issues`, decisions D1-D15 by review in
  `the decision log`): 70 open upstream bug and unlabelled issues plus 82 enhancement issues triaged
  against chaiNNer-C (reports in the session scratchpad), fixed in about 60 commits, worst first. Highlights: a fresh
  install starts again (NumPy with the server packages; a missing or broken torch costs only the PyTorch nodes;
  verified with a real first start of an `npm run make` build, where v0.3.2 dies on `numpy`); Convert To ONNX/NCNN
  export with `dynamo=False` (torch 2.14 needed onnxscript); NCNN return codes checked; Save Video audio on chaiNNer's
  FFmpeg, never silent; Save/Load Video colours by size with full tags; ONNX tile padding found by running the model;
  EXIF orientation; straight-alpha and grey+alpha TIFFs; TIFF ExtraSamples; lossless WebP `exact`; GIF palette per
  frame; CPU-only NCNN tiles by available RAM; a vitest runs the app's schema check over the real node list; CLI ends its
  backend; system-Python option removed; Python download honours the proxy; pip's errors reach the UI; Spandrel
  `0.4.2+c10` (fork release `v0.4.2-c10`, SHA-256 `24967414…`; c2: fp32 LayerNorm statistics in fp16 for 11 archs, ONNX
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
  DRFT-M 628 → 402 ms; c8, 2026-10-24: WSR, GaterV3, SMoSR, FIGSR and GFISRv2, each loaded from its state dict alone,
  tests from the archs' own weights since none are released; FDAT's DySample loads on CPU-only torch; c9, 2026-10-24:
  GaterV3 checkpoints without the flash buffer load as channel attention (two released 1x models matched the original
  code exactly), GaterV3 gets resselt's upsamplers, SMoSR and GFISRv2 loader fixes; SMoSR, FIGSR and GFISRv2 checked
  against resselt, 245 variants exact; resselt's inference-only differences not adopted, its FIGSR border measured
  4.8 dB worse at the edges; c10, 2026-10-25: DUAL's context layers run neighborhood attention, bitwise vs
  traiNNer's source on CPU FP32, 2x Light 690x512 fp16 68 -> 107 ms). PyTorch tries
  cuDNN attention first (2026-10-16: every model in
  fp16, not DUAL only; DUAL fp16 197.5 → 182.0 ms 1x, 52.0 → 47.0 ms 2x; Windows PyTorch has no Flash Attention).
  With the fp16 setting on, DUAL TensorRT exports start from fp16-rounded weights (68.5–88.6 dB vs exact, below the
  engine's bf16 noise; accepted). Deviations: ARCHITECTURE section 7.

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
| Use MIGraphX Program, DUAL on AMD Windows and Use HIP Graph have not run on an AMD GPU | Ask AMD users: does the first run finish, does a second skip the compile, does a small image come out at its size times the scale, and with Use HIP Graph on and off: the count, the time, whether the images match, and any out-of-memory |
| HIP Graph's capture boundaries (views placed by host values, `select_module`, `gpu::loop`) have never executed; upstream's pass tests look runnable without a GPU | Run them when the patch next changes |
| A folder moved from AMD to NVIDIA Windows keeps Triton 3.8.0.post29 (pins are floors), and TensorRT DUAL on it is untested; TensorRT's `dual_plugin_identity` reads only `triton`, so on Windows it ignores Triton's version | Check when such a folder turns up |
| `test_streamed_full_vae_matches_direct_encode_decode` failed once at 12 workers on Windows and passed alone | Look if it repeats |
| Yes/no questions from the 2026-10-23 hunt (`native/local/hunt-20261023/REPORT.md`, "Your yes/no list"): (1) Save Video FFV1 `-level 3 -slices 16` (14-46 % faster saves, frames identical, bytes change; ruling 7; may be stale: on FFmpeg 9.0.2 plain FFV1 already picks its own slices, Frames to Clip measured 2026-11-06; re-measure before deciding); (2) Save Image PNG compression level 5 (-24 % save time, +0.7 % size); (3) Blue Noise non-identical speedup (prototype 8-19x, 6 of 65,536 ranks differ; ruling 11); (4) upscale run-ahead: built 2026-10-29, removed 2026-10-30 (State) | The yes or no |
| ONNX Upscale Image on CUDA: runs headless since `72e2b913`, with and without torch imported (2026-11-06); the node body with a stub context, not the app | Check in the app |
| Electron 43 checks still to make by hand: first start of the packaged zip, a folder dropped onto inputs, dropdown group headers, dialogs remembering the folder, screenshot copy, Squirrel install and uninstall. Passed by hand on 2026-10-24 from source: first start, a `.chn` and an image dropped, chain copy and paste, image paste, Get Native Resolution's values in Number/Text nodes, Linear Light. A start listens on a socket (Vite, Electron, the backend): new program paths raise a Firewall prompt | By hand, announced first (`native/local/getnative-20261023/electron-report.md`, "Start check") |
| Electron 45 clipboard checks by hand (unit tests mock Electron's clipboard): chain copy and paste in the app, from v0.3.8 into it and from it into v0.3.8; image paste of an opaque and a transparent image (from a browser, which puts a PNG file on the clipboard, and from an app that puts only a bitmap); Export viewport to clipboard pasted into another app; `main.log` shows the backend's bare prints | By hand, at the next app start |
| `Squirrel.exe` is missing from the portable zip: Forge 7.11 runs the makers concurrently, so the zip no longer picks up the copy electron-winstaller makes (inert in a portable package; `verify_independent_ui.cjs` expects none; the installer's nupkg has it) | Accept, or copy it in `postMake` (without the chaiNNer icon) |
| Get Native Resolution's grid search at CRF 23 on x1.29 upscales: the planted 960x540 grid is supported on 0 of 288 frames (13 leading; supported frames sit on 930x523/930x540); at CRF 16, 98 of 288. Fractional plants vote 959x540 or 961x539 | A grid-core investigation (Vernier's search, not the kernel rule) |
| Get Native Resolution's fold-ratio kernel fingerprint was set aside: its statistic carries 1/(1 + 1/SNR(f)), which varies across the band, so soft or compressed content looks like a softer kernel (2 wrong supported decisions at CRF 16); the code is in history (`e03c36e4`) | A noise-corrected estimator (a noise floor from the permuted coherence, or noise in the model) before another try |
| Get Native Resolution follow-ups: `parameters.json`'s `selection` keeps Vernier's per-frame meaning on the Video tab; no Descaled output on the Video tab | Next session |
| Video written by Save Video through an FFmpeg 8.x or 9 build before the `setparams` fix (the D: install's FFmpeg 8 build included) has no primaries or transfer tags; FFmpeg 5.1.2 wrote them | Fixed for new saves; files already written keep their tags as written |
| npm 11.19 (Node 24.22.1) skips the install scripts of `@swc/core`, `core-js-pure`, `electron-winstaller` and `esbuild` (none is needed) and warns on every install; the lock pins `@electron/node-gyp` from GitHub over `git+ssh`, so `npm ci` needs GitHub access | `allowScripts` denies the four (2026-10-29); left: a registry `@electron/node-gyp` when Forge's rebuild has one |
| 2026-10-23 hunt, found not fixed: Convert To NCNN's Power layers (integer powers other than 2, exact since the batch run) run on the CPU in a Vulkan net, one round trip each (ncnn has no Vulkan Power; square or multiply layers would keep small powers on the GPU); every node re-renders on edge changes; Lens Blur's serial planar and interleaved copies (candidate, unmeasured); PNG encode is now the largest batch cost; editor speed-ups and the damaged-chain message need an in-app check | Next sessions |
| Follow-on F01/F17: concurrent Skip Existing writers and concurrent settings writers | Still deferred by review. F11/F12 were subsequently approved and implemented in `3a345dee` |
| Follow-on installed UI checks left: numeric drafts changed by stepper/arrow keys/wheel, and the backend stopping after an accepted quit | Typed numeric Ctrl+S and Quit-then-Cancel PASSED on the installed v0.3.8 (2026-10-22). Prescribed YUV/model, ONNX/NCNN Stop, background removal and the corrected Guided GPU run passed |
| Ruff: `npm run lint:py` passes (`55583e78`); `ruff check` from `native\` reports 32 in tests and `tools/amd/bench.py` (I001 x7, Q000 x7 in `test_atomic_file.py`, B905 x4, PLW0108 x3, RUF005 x3 with test_video_io.py's, ISC004 x3, RUF043 x2, RUF001 x2, C408) | A lint pass |
| Hunt O02: cold CFR Matroska/WebM rate detection and frame indexing read packets twice, then cache the result | Deferred unless a long cold load visibly stalls; preserve their different counting rules |
| Hunt O03: opt-in `CHAINNER_C_TRT_RUNNER=1` retains both Python and C++ TensorRT engine copies | Parked: the cross-boundary ownership change is riskier than the current saving |
| After the wave, run the store step inside its tiles (drops a second pool dispatch, works on cache lines just written); unmeasured | A probe with a keep rule |
| The output range check (`normalized_output`, `cn_pixels_normalized_f32`) reads every image a node returns without `assume_normalized` once more: 3.6 ms at 3840x2160 RGB on the 9950X3D, about as long as Invert Color's whole pass. Such a node's kernel could report whether every value it wrote is in range, so the engine skips the read (same outputs) | When CPU chains are next sped up, after PNG decode and encode (65 % of the measured chain) |
| Nested ESRGAN engines lose LeakyReLU fusion (about 7 % of a 512x512 run, at most about 3 % to win back); `nestfold` and `nestdeconv` unmeasured | A probe with a keep rule |
| TensorRT 11.4's AOT plugin path leaks one module per plugin layer on every enqueue (State, 2026-11-06): a long assorted-size batch on a Dynamic DUAL engine holds 1.3 MiB VRAM and 4 MiB RAM per new size until the backend exits; the driver's 128 MiB after the first device-assert module is one-off. Balanced Tiles, whole-frame mode and the PyTorch engine route on DUAL make more sizes | Not reported, by ruling; re-test on the next TensorRT; whether Dynamic ESRGAN engines also keep the 128 MiB is UNTESTED; the native runner's path was not measured |
| Hunt GPU corrections B16-B18: owning-device reuse/teardown, low-headroom split admission, activation restore recovery | CPU/source checks only; real-device confirmation deferred in `GPU-DEFERRED.md` |
| Load Video seek bar follow-ups (2026-10-21): the independent-UI patch (`native/tools/independent_ui.py`) for the `video-seek` input kind and runWithOverrides; the CLI path's Frame Count stays the probe's estimate (a run that reaches the end does not teach it); a one-frame run shares the Preview's decoded frame only in RGB float; VOSR Persistent Detail starts its history at a range's first context frame | Next sessions |
| The error-retention fix's two recorded follow-ups, both INFERRED: a failed `/run/individual` (a preview's inputs) and the per-item errors of a non-fail-fast generator (`process.py:903` keeps `str(e)`; the run then fails, so B collects at its end). Each lets an exception through the same frames as a failed `/run`; A breaks their chaiNNer-C edge, and upstream's `Lazy` edge remains. `await_owned_node`'s cancel-branch `error` local and `owned_gather`'s `results`/`tasks` locals are dropped since `8f220ab9`, with tests (checked 2026-10-29); the two `server.py` comments were rewritten since. Upstream's `Lazy.from_coroutine` `completion` local needs a separate decision. Root cause: `native/reports/error-retention/rootcause.md` | The fix (2026-10-06) is two commits: A, `bad1dba7` (`await_owned_node` and `NodeFlights.run` drop their `future`/`task` locals), then B with D (`collect_failed_run`: one `gc.collect()` after a failed or stopped `/run`'s handler returns, then the pool's release if no run executes; ARCHITECTURE section 6). A failed run's arrays are freed right after it, earlier than upstream git's end of the next `/run`; successful runs never collect. Growth record: the release check (0)–(f), with (e) and (f) asserted; verdict `native/reports/error-retention/verdict.md`. REVIEW FLAG (decision log): `import gc` and that one collection; undo = `git revert` of the second commit. The B1 re-test and SP3b's phase-2 re-check follow it, then SP5a |
| SP3 follow-ups: bench cases for a resumed batch with skip-existing (a second run over the same output folder; harness) and for a heavy untaken Conditional branch; a K spot-check at 4 CPUs (affinity `0xF`); count the replay round trips, then skip the round trip when a replay has no concurrent sibling; compare folder identity in the read-ahead predicate (would close the loopback-alias and non-Windows case-insensitivity residuals). SP2 native-pool carry-overs, deferred (SP3 kept the pool unchanged): the spurious WakeAll on re-pin, the `leave` comment, the `cn_parallel_dispatches` non-Windows note, measuring whether dropping the generation bump in `leave` pays. Test cleanup: split the 3605-line `native/tests/test_item_window.py` by concern (read-ahead predicate, engine and executor, Save phases, fresh interpreter) | SP4b, SP3b, SP3c and SP4c are closed (State). Order (2026-10-06): the error-retention fix (done, first row), then the B1 re-test and SP3b's phase-2 re-check, then SP5a (State, SP4c; rows below). A commit-path lever (B1 is re-tested after the error-retention fix) stays inside SP3c's boundary: outputs bit-identical to the installed chaiNNer and B3; file writes in item order, each complete before its node's finish event and before the next item's commit; events and `sse_contract` identical; the same error object, selected and surfaced at the same point relative to earlier and later items' side effects; Stop, pause and generator-once unchanged; K = 1 within noise. SP3 follow-ups and SP2 carry-overs: not yet scheduled (the SP4 attribution is done); cleanups SP4 or later |
| SP4b parked levers, sizes in `native/reports/sp4/profile-final.md` section 7: output allocation (SP4c, done, State; first touch costs 0.15–0.20 µs per page, 88–117 µs per fresh 2.36 MB output; a 'clamped-by-us' registry mark would also skip the borrow's scan and freeze, 41–65 and 26–33 µs per output); resample horizontal SIMD (≈ 200 of ≈ 220 µs per call at 1 CPU, INFERRED); resample grain (720p: 13.6 of 31 wakes empty per call) and morphology grains (`cn_filter_morphology` runs 9 chunks per dispatch; Task 4b's line-grain variants were neutral or cost 30–35 % more CPU); `cn_filter_morphology`'s algorithm (2114 µs against ≤ 1400; erode ≈ 23× its serial bound); the commit path (SP3c, closed, State): 91–97 % of every image item, by measurement the largest lever on 6 of 10 image graphs, yet SP3c's probe confirmed no win | SP4c took output allocation (State); the others stay parked, chosen by measured size, each with its own paired A/B; phase 2's revival rule (State, SP3b) is applied at the re-profile after SP4c (the re-check follows the error-retention fix) |
| SP5a-c (decision 2026-10-06, Consult 9): the product build is clang-cl 23.1.2 only, `-march=icelake-server` with ThinLTO and no PGO, `/fp:strict`, `/Brepro` (ARCHITECTURE section 3); MSVC and gcc left the build, and a startup check refuses a CPU below Ice Lake-SP / Zen 4. The A/B against MSVC was replaced by the switch. `isa_check.py`'s instruction-set checks (`objects`, `dll`) are retired, since every unit is Ice Lake-SP code and the ThinLTO objects are bitcode; its `text` comparison stays. Corrected 2026-10-07 : baseline x86-64 with runtime AVX2/AVX-512 dispatch, no CPU refusal, `-march` local-only (State) | The old-vs-new bench (CPython 3.11 + MSVC against 3.14 + clang) is the only timing; `-mprefer-vector-width=512` is an optional measured A/B after it; CCD affinity stays optional |
| NaN-bit robustness (Consult 12 D-21): `cn_spectral_multiply` stays as is (equal to OpenCV 5.0.0 at every level; its NaN bits follow operand order). Any codegen-affecting toolchain or flag change (an LLVM bump, `-mprefer-vector-width=512`, PGO) re-runs `test_packed_spectrum_all_parities` and the NaN goldens (`test_golden_kernels` tail and dither, `test_forced_paths`) at all three `CHAINNER_C_ISA` levels; a failure is fixed in the integer shape of `normal_nan_outputs` (`color_shared.h`), never by a flag | Standing rule; SP5b (PGO) applies it before landing |
| Isolated CMake outputs: SP4a added `CHAINNER_C_OUTPUT_DIR` (default unchanged: every default configuration still writes the DLL/PYD into `backend/src/nodes/impl`) and `CHAINNER_C_PDB`; reproducible default build from a clean clone; bench harness: the media root `F:\chaiNNer-C-diagnostics` is still hard-coded (`bench.py:245`), and the pyright errors at `bench_fixtures.py:84` and `image_utils.py:346` predate SP4b (found in Tasks 10a and 3b2). Closed: comparing the static CRT with `ucrtbase.dll`, FMA3 path on and off (Consult 6 D-4, commit "native: mirrors call the process's ucrtbase.dll math (D-4)"). Mirrors of a reference library's own CRT call now call the process's `ucrtbase.dll` through `cn_crt` (`native/include/cn_crt_math.h`). `native/tests/test_crt_math.py` compares every entry with the static CRT and warns on a difference: on this machine only `pow` with FMA3 differs, by 1 ulp (12 of the test's 32,786 inputs; 261 of 1,048,594 in an enlarged run). Bench note (Consult 8 R-d): no tested bits change; two untested double-`pow` paths (PyMatting's pre-round level sizes, ncnn's float64 power) now follow NumPy, as the probe test records | SP5a, before any build variant |
| Closed by `3a324fe1`: a process that loads every package and node module and then exits normally died at interpreter shutdown: a call to address 0 from `ntdll` (`0xC0000005`, crash report with module "unknown"). Seen in an import probe; `run.py --close-after-start` (`npm run type-check:runtime`) takes the same path on Windows. The fix: the NCNN session frees its cached nets, then ncnn's Vulkan instance, at exit, so a normal exit returns 0 (U4-light item 2, VERIFIED on the GPU: `server.py <port> --close-after-start` with Vulkan visible, 0xC0000005 before and 0 after; the controller's re-run of `type-check:runtime`'s command line on main `a831d6b6`, Vulkan visible, 2026-10-07: exit 0, its three variables set directly since this checkout has no `node_modules`). Each `--close-after-start` exit also prints Sanic's "Task was destroyed but it is pending!" and its Python 3.14 event-loop-policy `DeprecationWarning`s (relayed at ERROR by the host), as upstream's code does on this stack (the oracle, two runs); that policy API is removed in Python 3.16, so a 3.16 upgrade needs a Sanic release without it. The `msvcp140.dll` crash reports of 2026-09-21 to 2026-10-02 are resolved: they were Torch/ONNX Runtime import failures in test processes where Pillow's bundled 14.29 loaded first; `native/tests/conftest.py` now loads the system copy and the suite runs in one process | Closed by `3a324fe1` (controller re-run: exit 0). Leftover crash-report clones (`python.exe`, 0 threads) lock DLLs until an elevated `taskkill /F /PID` or a reboot |
| Live clipboard publication/paste (API-boundary tests only); external viewer (displayed image never observed); real Explorer/OLE drop retry (synthetic drop passed); SP4a's optional AMD uProf / WPR sampling session (elevated; rebuild `native\build\pdb` first, since the PDB DLL predates `cn_isa_get`; uProf flags VERIFIED from the installed docs, `wpr -stop <file>` unverified; local script `native\build\sp4a-sample.ps1` (ignored, not in a clean clone), with `-DryRun`; it would split the pyd copies and encode, 30–71 % of the image cases' auto-K loops, `profile-final.md` section 2); `native/tests/test_isa_dispatch.py` once on an AVX2-only machine (the AVX-512 cap is VERIFIED by disassembly, UNTESTED on this AVX-512 host); the kept AVX-512 units on a part that downclocks (UNTESTED; no clock-down seen here, `CHAINNER_C_ISA=avx2` caps them); the scalar level on an SSE2-only CPU (UNTESTED on SSE2-only hardware, forced on this machine; a Hyper-V VM in processor-compatibility mode first, an emulator download only after) | USER-GATE: manual checks |
| GPU-dependent work (the GPU lease and its unittest were removed on 2026-10-05, decision; `GPU-DEFERRED.md` item 2 retired). Item 1, the GPU provider paths, was verified by U4-light on 2026-10-07 and removed: NCNN on Vulkan, ONNX Runtime's CUDA execution provider and PyTorch CUDA work on the 3.14 stack (`default.chn` on the GPU included; peak per-process VRAM 1.87 GB); ONNX Runtime's DirectML and TensorRT providers were not exercised | `GPU-DEFERRED.md`, under the rule of 2026-10-07: no GPU benchmarks, at most 3 GB VRAM |
| SP0 deferred follow-ups | `docs/superpowers/plans/2026-10-01-sp0-foundation.md`, "Deferred follow-ups" |
| Upstream-parity items still listed (decision 4, 2026-10-04): the C1-class pins not yet checked against chaiNNer; lens power's input `maxss(+0, x)`, output-invisible and VERIFIED equal to chaiNNer's chain; compose's NaN selection, positions-level by rule (spec 4.4); suite-wide ±0-aware bitwise comparators (the suite's `assert_array_equal` treats ±0 as equal). Closed: the normalize/enforce clamp follows `np.clip` on the tested stack, −0 kept (ARCHITECTURE section 7, Consult 11 R-p) | SP5a, or the next task that touches them |
| Upstream-bug follow-ups (2026-10-09): SCUNet ONNX export unblocking (needs a real-model test); the real first-start check needs a fixed, firewall-allowed folder | `the decision log` (D1-D15) |
| VOSR speed, PyTorch path: each window's DINO features go to the CPU and back, and every DiT prediction is synchronized and copied to the CPU (`vosr2_context.py`) | Keep them on the GPU when Default has room; paired A/B |
| VOSR temporal on PyTorch: the next restoration does not overlap the CPU stabilization (about 1.1 s of 4.6 s per 2560×1440 frame); `stabilize` runs on one thread (1.0 s); the next `motion_pair` could run alongside | Paired A/B and a memory check for each |
| VOSR temporal: each item settles its background restoration, so the GPU idles while downstream nodes (Save Video's encode) run | Decision: letting it cross the item boundary relaxes "no GPU work escapes an item" |
| VOSR engine identities are broader than the engines: all of `vosr2.py` (a message edit rebuilds) and the whole plugin DLL (a metadata-only rebuild rebuilds) | Hash only engine-relevant code and the DLL's SASS/PTX |
| VOSR engine builds: one stage at a time, about 3 minutes (each stage's start and end and the total are logged; the per-GPU timing cache exists) | Small stages in parallel (a GPU check) |
| VOSR FP8 and model downloads: one host each and no mirror | A mirror list beside the pins |
| VOSR scene focus where a busy in-focus texture (thatch, foliage) holds the sharpest strip: in-focus faces count as partly defocused and lose 10-15 % detail (the issue-13 portrait, 2x). An edge-width reference (re-blur gradient ratio) fixed that frame but weakened Rin's defocused background at 4x. Separately, the half-soft face there was VOSR's own window seam (present with routing off), now handled by window placement and face windows (State). 2026-10-14: keep today's reference and the 64 px minimum window overlap (16 px measured: 8 % faster, 3x the seam step) | A focus measure tuned on more of the frames; seam windows for non-face content (anime faces, hands) need a trigger that finds them; faces wider than 800 px keep the layout's seams; in an output under 512 px on one side, a mirrored copy of an edge face in the padding can take a face window (harmless, rare) |
| VOSR plugin code: no sm_90/sm_100 SASS (compute_120 PTX cannot run there); a dead branch in `qkv.cu` and an `rsqrtf` worth a comment | SASS when those GPUs become targets; code cleanups need GPU revalidation |
| Look-ahead executor: K is decided at item 0 once a node looks ahead (a later-computed writer folder then forces K = 1 for groups reading per-item files); jobs computed ahead get a fresh context each; `ItemWindow._wake` rechecks every waiter; forced jobs bypass P; the UI's lineage takes the first iterated edge, the backend the first iterated input | A schema-level "looks ahead" flag, a per-node context cache, a per-slot waiter map and a cap, when a measured need appears |
| VOSR in the local portable package: the nightly UI's patches lack the iterated condition | A patch JSON against the restored nightly (README, Prerequisites) |
| Loop runs allocate about 60 MB more torch memory than one call (likely a cuBLAS workspace per pool thread) | Pin a GPU node's calls to one thread |
| Upstream check and bug hunt (2026-10-16), queued items, each ruled by the review stand-in (`the decision log`, "2026-10-16 · Upstream check and bug hunt"): `independent_ui.py` patches for 17d9331c (uninstall a disabled package) and 3f671309 (hidden experimental toggle), with the nightly-restore item (first row); the read-ahead predicate refinement for look-ahead nodes (comparing the full output folder instead of Save Image's folder lets more chains read ahead: a ruling; Save Image's skipped-file CPU work also runs ahead); PyAV CFR resampling for variable-rate MKV/WebM (today they use the CLI reader); lazy inputs: evaluate the producer inline on the waiting thread; Save Image: RGBA to JPEG flatten vs drop (choice); a newer-chaiNNer chain notice | Next sessions, in the order of the decisions file |
| Open questions (2026-10-16): vitest 1 -> 4 (with ip-address 9 -> 10 and adm-zip 0.5 -> 0.6), re-release v0.3.4 with today's Descale Mirror out-of-bounds fix (v0.3.4 ships the bug), and the upstream features #3140/#3191/#3187/#3222 stay out | The yes or no |
| VapourSynth hunt follow-ups (2026-10-16; the v0.3.5 re-release has the 21 hunt fixes plus 3; rulings by the review stand-in in `the decision log`; the process side closed 2026-10-29 on `batch-vsproc`): the catalog's hide table re-measured for a new vs-jetpack (the class sweep in `native/local/batch-20261029/lane-vscat/sweep`); frequency_merge gets one clip input (its other clip-or-number parameters keep Clip 2 to 8 away). | Reports in the session scratchpad `vshunt\a1..a5-report.md`, rulings in the local rulings file |
| GPU phase of the YUV/TensorRT 11.4 work, the rest (approval only; ESRGAN models, DUAL Light/XS) | Phase 1 done 2026-10-18 (`native/local/yuv-planes/report-gpu.md`: runner tests, Upscale Image and AI Upscale bit-identical, 11.2 vs 11.4, bridge vs runner vs vstrt, the 11.2-engine message; Tile documented, not honoured). Left: AI Upscale's runner default on a 1080p RGBS testsrc2 clip, Streams 1 and 2, against the bridge (`CHAINNER_C_TRT_RUNNER=0`) with one ESRGAN engine, headless, 50 frames after warm-up: frames bit-identical, ms/frame each; vosr2_qkv and DUAL on 11.4 |
| GPU checks deferred on 2026-10-16 (a model was training; light checks only): Build Engine Fixed 64x64 -> Get Engine Info -> Upscale on 200x150 under chaiNNer-C's TensorRT 11.2; a fixed-size ONNX engine keeps its size under 11.2; the timing cache file after the first build and "Loaded" on the second; Stop and worker kill during a real DUAL build (no child left, VRAM back); the TensorRT session-leak paths (need a CUDA fault); the VOSR CUDA checks listed in the area-7 hunt report | The next GPU session |
| instant-exec, deferred by ruling (2026-10-19): Save Video adds audio in the encode pass, dropping the second full pass (R5b); Clip Frames evaluates its script twice (a sidecar from the frames run's evaluation); Normal Map Generator native only with a native FFT, no partial port (R9); an optional row stride in the image ABI (a partial-width crop copies today where Python returned a view); Load Images as a native `ItemSource`, only if `resize-real-256` shows at least 0.1 ms per item of GIL-held glue; zlib-ng inflate for PNG and TIFF decode (read side only); the Python executor and its white-box tests retired in the first release after v0.3.7 whose full gate passes with the native default (A5) | Next sessions, each with its paired measurement; GPU items: `GPU-DEFERRED.md` rows 6-22 |
| instant-exec residuals (the lane reports' open lists, `native/local/instant-exec/report-*.md`). Engine: 16 concurrent previews 20 ms against Python's 9; no memo for native generators' items; the memo counts a storage an item and its crop share twice (errs safe); a file changed between stat and decode can hit a stale entry (a same-size rewrite is seen by its change time); latent: `decide()` takes a slot after the GIL, paused time while the driver blocks outside `suspend()`; engine images are always C-contiguous (P7: the reference); `engine_differential.py` builds a chain per executor, so side-effect order across sinks is not compared exactly. Startup: the first launch after an update shows the splash once; fresh storage compiles PyMatting (about 25 s, delaying its two nodes and the full list); Run is inert until the backend takes requests (about 1 s). Nodes: a Seed through Pass Through is an equal new object; above 2^63 bytes NumPy raises ValueError where the engine raises MemoryError or OverflowError (per-node decision). Codecs: non-plain TIFF Load Image and GIF/TGA/AVIF/DDS Save Image still decline; `JSIMD_FORCENONE` changes our libjpeg-turbo, not OpenCV's (one CMYK JPEG-in-TIFF); the writer's cross-drive move is untested; `.exr` still fails (open questions). Upscale: `recover_alpha` per chunk; PyTorch tile batching (decision). App: `useSettings` subscribes to the whole object; a layout-only counter, an incremental type pass and preview reuse on reopen (skipped: small gain). Video: a fused float BGR -> 16-bit planes kernel; `chainner_vs/upscale.py` `_image` still `np.stack`s; no native PyAV reader (Save Video sets the pace). Backend: `file_sequence.cpp` resolves every folder (`realpath`; a 1000x5 tree 227 ms against 105 ms with `scandir`) unless the second wave's listing work covers it. Final wave: `gc.freeze` is skipped when a framework node's preview finished first (the template), so R1's gain is for starts without one; the native codecs read OpenCV's limit variables as plain integers only; two Convert To ONNX nodes with different options on one model re-export on each run | Next sessions, by measured size |
| Open questions (instant-exec, 2026-10-19; the local queue file): release v0.3.7 (tag, push, D: update)? The native engine becomes every user's default executor; the Python one stays reachable by `CHAINNER_C_EXECUTOR=python` until its retirement. Build Engine's disk Engine Cache on by default? Up to 16 GiB of engines in the storage folder. A pip wheel cache for dependency installs (multi-GB, the disk rule). The memo cache as a user setting. The `.exr` loading fix (OpenCV's build has no OpenEXR). The spandrel fork wheel release with the DRFT fold speed-up (build 309 -> 210 ms). The Python executor's retirement in the first release after v0.3.7 (A5). Checks: the image size limit as read (40,000 x 40,000 = 1.6 billion pixels in total, each side up to 1,048,576); clicking a node selects it on mouse-up (was mouse-down); the packaged `out\` startup timing (one launch; a new folder may prompt the firewall); the GPU checks, `GPU-DEFERRED.md` rows 6-22 | The yes or no |
| DRFT TensorRT route (2026-10-20): released (re-release 5, State); nothing open | On request |
| VRAM offload follow-ups (2026-10-20): a first overflow in a session pays the page-locking of its pinned RAM (2176 px: 5.96 s against 4.95 s warm), again in each run that overflows (its pinned RAM is released at the run's end); least-recently-used order brings 6.8 GiB back where a traced farthest-next-use plan would bring almost none; only Upscale Image's forward is covered (inputs of 2 MP or more) | A traced plan if the warm 7 % matters |
| `ARCHITECTURE.md:10,65` says the package ships Electron 25; it ships 45.1.0 (stale) | Correct both lines |
| App checks from 2026-11-06: the looks of the new inputs, outputs and nodes, the running app's Stop, previews and toasts (nothing launched the app) | By hand, at the next app start; the local checklist in `native/local/night-20261106/` |
| Released v0.3.9 r3 has the DAT2 blank-image fault on its FP16 route (State) | Re-release, the yes or no; affected models then convert again |
| `test_vsview_of_an_earlier_worker_stays_the_window` failed once in a full Linux run, no failure text kept, then passed alone 3 of 3, under load 4 of 4 and in four full runs; `test_vosr2_streaming::test_streamed_full_vae_matches_direct_encode_decode` failed once under 13 shards, passes alone | Watch them |
| SwinIR-family models stored at `img_size` 32 freeze the mask in the trace: wrong or crashing at other sizes (reach unknown, no real file seen; not from the FP16 rewrite). | A different example size for such models |
| A renderer code cache through a custom scheme (`f7233ee8`, estimated 20-70 ms, never launched) | Keep it only if its checks pass and the real interface is at least 30 ms faster |
| Upstream check of 66 commits (to `f223edb6`): Regex Find's `Found` left out, since saved chains would change; taskbar progress during installs (about 8 lines) only with another UI rebuild; `src/common/i18n.ts:4` has a shape that never reads the English JSON, so `t()`'s defaults show | If asked |
| Save Video's Simple mode has no AV1 or NVENC entries | The yes or no |
| On Windows, refuse ONNX on CUDA or TensorRT without the PyTorch package: a cuDNN for another CUDA on PATH can abort silently; refusing costs those who run a working CUDA 13 toolkit an install. UNTESTED without the package | The yes or no |
| One trained plain PLKSR weight (none local; shares RealPLKSR's recipe, random weights only) | The yes or no: fetch one official weight and measure |
| Stress scripts and the TSan recipe stay in the run's local folder | The yes or no: move them to `native/tools` and `native/BUILDING.md` |
| Electron trim's startup savings are estimates, and nothing launched on Electron 45 | A timing trial by hand, before and after, medians of launches 2-4; to undo, `git revert 07c80704 7da3ae8d 58eebe7c ea68c9bb` and an npm rebuild |
| Dragging the chain during a run is the one open Electron number: one paired run per case of a 48-image ESRGAN TensorRT chain with a 146-node chain open at 4K 240 Hz (minimized, visible idle, stepped spinner, scripted pan, pan with acceleration on): chain time, whole-tree private memory, per-process dedicated VRAM | By hand, other jobs stopped; if the pan adds under 5 %, or a frame cap removes most of it, stop |
| Renderer code cache through a custom scheme: `file://` pages get none, so `index.js` compiles at every launch (estimate -50 to -170 ms). Steps written down, no Local Storage migration needed; built on `f7233ee8`, never launched; a saved zoom resets once | A trial, by hand: all checks in its list pass and the real interface at least 30 ms faster |
| `native/.venv` has no VapourSynth, so `test_vapoursynth_*.py` skip in the full gate; a stale call there went unseen | Run them under the integrated Python at each gate, as the last gate did |
| `run_tests.py`'s shards oversubscribe the CPU on `test_convert_to_onnx.py` (each export test runs torch and ONNX Runtime with all threads): 222 s alone, about 20 minutes in the gate; with the shard cap in, `test_convert_to_onnx_sizes.py` still took 1,708 s in the last gate against under 50 s in the first: ONNX Runtime ran 15 threads on a 2-CPU shard inside a native session, which the cap does not reach | Cap threads per shard from the worker count, the native session included; the durations file holds the files' timings |
| `native/tests/test_native_codecs.py` has 10 RUF005 findings (`lint:py` covers the backend only) | A lint pass |
| split-channels and palette-dither read +5 to +9 % on 65-100 ms runs, merged tree against base; cause not found | More images per run, if it shows |
| TensorRT out-of-memory retry: a really exhausted card is UNTESTED (Windows accepted a 120 GB device request, working set past 100 GiB and page file +92 GiB, so no test may ask for more than the card has); the native runner does not retry; DUAL and DRFT builders do not take activation memory by the tile; the flag on HAT and DAT is UNTESTED | When a card can be filled safely |
| Use TensorRT Engine: GPU runs on HAT, DAT, DUAL, DRFT, the entries new to the list and above 512 px UNTESTED; a real refused file and real infinity seen only through stand-ins; the build shows no progress and Stop waits for it (37.7 s on UltraSharp); Settings shows Clear Cache only while the Engine Cache switch is on; PyTorch's memory budget does not reach the engine; no line says that a first build has started (RCAN's first use is about 80 s: trace 19, check 8, build 52) | Next GPU session; one line when a first build starts ("Preparing a TensorRT engine, first use only") |
| Balanced Tiles: record the ONNX's size multiple in new engines for a finer grid (512x512 UltraSharp 140.80 -> 95.93 ms, 63.42 -> 63.06 dB; about 25 lines; cached engines keep the 64 grid) | The yes or no |
| HAT and DAT `GlobalAveragePool` on ONNX Runtime CUDA in FP16 may overflow as OmniSR's ReduceMean did: UNTESTED | A GPU check |
| MIGraphX: UNTESTED on the FP32 sums of HAT and DAT, OmniSR, PLKSR, Compact and SPAN in FP16, the FP32 input | Next session with that device |
| SPAN on TensorRT: 2xHFA2kSPAN 46.3 dB on a diagonal gradient (61.1 on ONNX Runtime CPU; two more blocks in FP32 reach 47.0), 2xHFA2k_LUDVAE_SPAN flat red 49.2 on the older input | Look only if it shows in real use |
| 4x-OmniSR-DF2K thin bars on black at 256 px: 39.5 dB on ONNX Runtime CPU, 37.9 on TensorRT; FP32 input and FP32 attention do not help | Look only if it shows in real use |
| RealPLKSR on flat primary colours on TensorRT reads 28 to 38 dB on two weights; in FP32, noise of 0.03 codes already gives 48.3 dB: the model's own instability, accepted. Two builds differ by about 7 dB | Accepted |
| ONNX Runtime on CUDA in FP32 reached 57 dB on the worst Text2HD crop against PyTorch FP32 (median 81 dB), FP16 62 dB; cause not looked for, TF32 suspected | Look only if it shows in real use |
| GRL's export crashed once on a tiny config, cause unknown. ATD, FFTformer, IPT, SAFMNBCIE have no working export; `is_onnx_supported` says SAFMNBCIE has | A Linux probe for GRL; a clearer refusal for the rest |
| Older RealPLKSR ONNX files gain the ONNX Runtime speedup and automatic FP16 only when converted again | Decide if asked |
| Align Image on a GPU in FP16 with mask and Wide Search, Guided Upscale on a GPU: UNTESTED | Next GPU session |
| Get Native Resolution one-axis: a fractional plant stays Inconclusive with the core's 639 for 640; JPEG blocks can lead a Lanczos one-axis plant to a 7/8 width (672 of 768); an axis scaled to within 32 pixels of the source looks unscaled; detail on one axis only stays Inconclusive | Next session |
| VapourSynth images: re-measure `frame_methods.json` with a GPU visible (21 methods are clips only because they need one; Based AA needs a model download); a new vs-jetpack or VapourSynth turns images off until `vs_frame_probe.py` runs again; AI Upscale's models stay loaded in the frame process until the run ends (VRAM unmeasured); `vsrgtools.awarpsharp` may give bad pixels | Next GPU session |
| Load Videos: two writers that differ in a non-format dropdown, or a path through a lazy node, are not compared; a Load Images feeding Directory runs its body twice (one more listing); a Range zipped with it must match every file's length, and the message names no lengths | If asked |
| AI Upscale and Build Program do not warn on a frozen DySample ONNX; AI Upscale also probes per process on GPU providers (0.9-2.3 s per script start) | Next session |
| Colour stated without an ICC profile (EXIF Interop, PNG gAMA/cHRM/cICP, AVIF nclx) counts as untagged; CMYK stays Pillow's RGB with a warning; XMP, IPTC and PNG text are not carried; upstream #1754, #3200 not built | If asked |
| libswscale's full-range RGB to YUV chroma sits about 0.5 code low, before and after the Float RGB change; 5-bit RGB video white reads 0.9726; `ffv1_options`' docstring still names FFmpeg 5.1 | Next time video is touched |
| Listed races, not fixed: a pure node calling `peek` or `lookahead` (unreachable today); `Run::fault`'s out-of-memory notify; `cn_numpy_pool_free` reading `config.granularity`; `Run::close` clearing `broadcast_order_`; Linux per-call `getenv` against `os.environ["PATH"]` in `migraphx/runner.py`. VOSR's `stored_record` hashes `str(path).encode()`, which fails on a path that is not text | If they show |
| `real_use_pass` with `default.chn` needs the DUAL weight at its saved path on a WSL share, or it skips that chain | Keep in mind at the next release |
| Temporal refusal with Stop on first error off prints once per image with no file name; a corrupt file's message gives the path twice | Low |
| Held after the last merge round, each on its own branch: a new 16 GiB limit for Pad and Merge (Resize's limit, which narrows large images on a machine that holds them); the Resize Percentage body (it disagrees with the type system at half-way values: 375 px at 9.2 % declared 35, given 34; 158 cases newly wrong) | The yes or no, each |
| Output changes found and not made: Threshold, Add, Multiply and Divide change alpha (Threshold Binary Inverted on an opaque image gives alpha 0); Rotate at 0, 90, 180 and 270 degrees goes through 8 bits; Change Color Model RGB to YUV passes V outside 0..1 unclipped (Gamma of the negative gives NaN); Blend Add, Subtract, Linear Burn and Linear Light with alpha mix the unclipped result (matches Photoshop) | The yes or no on each; Rotate as its own fix |
| Truncated JPEG files decode silently grey (up to 82 % of pixels wrong, no error); the refusal built for them allocated without bound on progressive files and was taken out. Save Image JPG at 65501x2 took 15.5 GiB before it was stopped: a possible fault, no result | A bounded check; reproduce the JPG size under a memory cap |
| Load Video: Seek to a frame the decoder never produces fails the run on damaged files; `gbr_planes_u16` checks contiguity, not alignment; Round uses floor(x + 0.5) | If asked; a guard |
| VapourSynth: `fmtc.bitdepth` on a gray image keeps a full-range tag on limited output (largest 0.0625) | The yes or no: refuse `fmtc.bitdepth` on images |
| VapourSynth: Grainer's Placebo ends the process where there is no Vulkan device | The yes or no: refuse it there |
| A Dynamic RGT engine up to 512 px or more, built by hand in Build Engine, can fill the card and hang the GPU for minutes: the automatic engine stops at 256 px, Build Engine does not warn | A warning before the build |
| Valgrind Memcheck, one pass at `CHAINNER_C_ISA=avx2` on v0.3.10 (12 codec, filter and noise test files): no uninitialised read starts in first-party code. libwebp's encoder reads its own scratch (`InitVP8Encoder`) on 13x9 images, and its files still equal OpenCV's byte for byte; libjpeg-turbo's 4:4:0 decode of a 13x9 JPEG reports rows 0 and 1 as uninitialised, while its pixels equal Pillow's at every size and do not follow heap contents. Valgrind stops at an instruction inside `cv2` (0xC5 0x79 0xD6), which cut `test_gaussian_complete` and `test_convolution_ops` short; it reports a repeated stack once, so one report can stand for more rows | Look again on a libjpeg-turbo or libwebp update; a pass with `cv2` kept out would cover the two files |
| SwinIR-L FP16 on a 64 px horizontal gradient reads 45.2 dB against FP32's 112 dB, rising with size: left on the list | If it shows |
| Three alerts added to `GlobalNodeState.tsx` ("File contains invalid values", "Chain contains invalid values", "Error") do not go through `t()`, so they stay English in every language | When they are translated |
| The opt-in runner against the Python session on RGBA with a hole, tiled, on an FP16 engine: largest difference 0.000977 on 3,549 of 3,040,000 alpha values, on the old and the new module (the runner runs the white pass in full) | If it shows |
| FIGSR's FP32 engine, Dynamic at exactly 256 px, reads 62.65-70.71 dB on crops where other sizes read 127 or better: seen, not looked into | If it shows |
| `chainner_trt_vs.dll`: frames bit-identical to the Python bridge (16 of 16, FP16 and FP32, Fixed and Dynamic); its failure paths ran only through `_chainner_trt.pyd` and the fault harness | Next GPU session |
| `verify_runtime.py`'s fixtures added-noise-poisson, dither-uniform-*, surface-rgba and fourth-combined should now differ from the oracle on purpose (read from the code, not run) | Update them when the verifier next runs |
| A number that is infinite or NaN into a text input stops the run with "cannot convert float infinity to integer" (`generic_inputs.py` keeps `int()` raising) | The error names the node and the value |
| Upstream has Spanish and German UI translations (#3171) that are missing here; existing language support is never changed | The yes or no |

## Decision log

- 2026-10-29 (DDS on Linux, stand-in ruling): a texconv lookalike on DirectXTex (native/texconv) instead of the
  error; texconv.exe's output is the reference, and what Windows decides in WIC or MSVC's arithmetic is rebuilt where
  measured exactly, otherwise listed (`linux_differences.json`, `texconv-wic-*`). Records:
  `native/local/prep-20261029/lane-dds/`.
- 2026-10-29 (Linux x86-64): Windows code stays verbatim and Linux code goes in `#else` / `if(NOT WIN32)` blocks and
  `native/toolchain-linux.cmake`; the glibc 2.28 floor comes from a Rocky 8.10 sysroot inside an Ubuntu 22.04 image;
  C-library math follows the system's libm (Linux twins of the golden manifests) except `sqrt` and `sqrtf`, which
  reproduce the static UCRT; Linux workers switch off NumPy's AVX-512 dispatch; DDS keeps its error and the zip carries
  no texconv; VapourSynth is unavailable below glibc 2.35; NCNN defaults to the CPU when every Vulkan device is
  software; Copy To Clipboard calls wl-clipboard or xclip; the sandbox is probed and `CHAINNER_SANDBOX` overrides it; no
  separate Linux release, the zips ship with the next one. Records: `native/local/linux-20261029/` (decisions.md).
- 2026-10-25 (DUAL neighborhood attention, owner handoff): every DUAL model runs it in both routes with no switch back
  (weights carry no routing); non-RFB or cosine variants keep their windows; a fixed export whose trunk fits both
  windows keeps the whole-window ops (the source's dispatch), dynamic exports always use the NA kernel; ONNX files
  exported before still build, with a log warning; GPU tests at the research input 512x692 (owner); the remaining
  0.9 % to the research engine (its NA plugin layers) parked. Records: `native/local/dual-na-20261025/decisions.md`.
- 2026-10-24 (resselt's GaterV3, SMoSR, FIGSR and GFISRv2): spandrel loads every checkpoint resselt's versions make,
  but keeps its own inference behaviour where it is better or is the trainers' (FIGSR without resselt's 4 px border,
  GFISRv2's even-size padding); GFISR v1 and MoSRv2 not added; shipped with the allocator fix as one re-release.
  Records: `native/local/resselt-20261024/` (brief, lane reports).
- 2026-10-24 (a user's FIGSR and FDAT runs failed `!handles_.at(i)` in PyTorch's allocator): the worker leaves
  expandable_segments out of an inherited allocator setting rather than keeping the user's value, and keeps the
  user's other options with the pinned RAM ones (`backend/src/cuda_allocator.py`).
- 2026-10-23 (Get Native Resolution, Linear Light, FFmpeg pin, Electron): the node opens the video itself; its
  outputs carry the stored (encoded) values, light models are evidence only; an ambiguous kernel carries the
  smallest support in the 8 % band; the kernel rule is the least-squares cross-frame rule (the fold-ratio
  fingerprint failed its pre-registered keep rule and was set aside); FFmpeg pinned at 9.0.2 on every platform;
  Electron 43.8.3 rather than 45 (clipboard rewrite in 44); Save Video sets its colour tags on the frames and Auto
  transcodes PCM and FLAC going into MP4; the GIF test judges palette error perceptually (FFmpeg 6.0's OkLab
  palettegen). Records: `native/local/getnative-20261023/` (CONTRACT.md, the rulings file, the branch and
  integration reports).
- 2026-10-22 (the guided decision): apply the one-line Guided tile-axis correction, add one
  non-square grayscale CPU test with RGB control, repeat the single Guided GPU sample with final HxW shape, commit and
  hand back. Only focused pix_transform tests, no new full suite/native rebuild; no release or D: changes by an earlier review.
- 2026-10-22 (implementation review): the implementation review arrived and the GPU was freed. Implement
  A2 F11/F12 and B1-B8, then one final main CPU gate followed by the prescribed sequential GPU functional checks;
  no benchmarks, large-DUAL/multi-GPU work, app launch, D: update, push or release. Release remains a separate decision.
  Approved: downloading small U2NetP weights into the task folder; no dependency installation.
  The newly reproduced Guided grayscale assembly defect is recorded for review and approval, not implemented.
- 2026-10-22 (implement the follow-on decisions): F26 saturates all shared integer exports, including
  normalized=True and infinities; preserve NaN and floating-point output semantics. Keep normalized parameters for
  caller compatibility. F19's zero-epoch case gets exactly the requested optimizer steps, not a forced full epoch;
  positive epochs are unchanged. D01 keeps small-image SSIM NaN and makes the type/documentation admit it.
- 2026-10-22 (follow-on validation): no GPU use during training; no app launch, deployment, dependency installation,
  push or release. Six Sol xhigh implementation lanes, bounded Sol repairs and one Astra max source reviewer; narrow critical discriminators
  and the existing one-phase CPU gate. F01/F11/F12/F17 and other performance proposals stay deferred.
- 2026-10-21 (review B13): PSNR changes to decibels from unrounded MSE, infinity only at zero; native/Python together, analytic check and saved-threshold release note.
- 2026-10-21 (review B15): Pad uses the type-only border-channel correction, including unions for connected values; runtime pixels stay unchanged.
- 2026-10-21 (review B18): protect all operation inputs and retry failed activation restoration after non-input eviction; one CPU stand-in check only, real DUAL VRAM confirmation deferred.
- 2026-10-21 (current hunt): no GPU benchmarks; functional GPU work only within 1 GB VRAM. This implementation phase uses no GPU. Delegated work uses GPT-6.1 Sol xhigh, with one GPT-6 Astra max reviewer when justified.
User rulings (pre-program handoff section 2, at the tag; still binding):

- Exact chaiNNer UI/UX: schemas, names, controls, saved chains, previews, errors, cancellation,
  ownership and restart. Only exceptions: updater removal and the reported drag-and-drop repairs.
- Independent edition: never check upstream releases; old settings cannot re-enable it.
- C or C++ per component; tested C kernels stay C; no renaming or wrapping counted as conversion.
- Broad optimization across workflows; fpng chosen for supported PNG encoding.
- CPU benchmarks allowed, without AI models (supersedes "no timing benchmarks"). Historical GPU
  limit: at most 2 GB VRAM, no precise GPU timing; superseded by the rule of 2026-10-07: no GPU
  benchmarks, at most 3 GB VRAM.
- No computer-use or desktop automation. Ask before dependency installs or toolchain switches.
- Keep implementation, tests, packaging, runtime and performance evidence separate. Reap only
  owned processes; never kill by name.

Program decisions (spec section 2, 2026-10-01, as amended): bit-exact float32 against the installed chaiNNer as the user receives outputs (user, 2026-10-04; judged
too: where it is wrong the port does the correct thing and records a deviation, ARCHITECTURE section 7; B0 and B3 are regression proxies), SIMD only where
bit-identical, no FMA contraction, reordered reductions or fast-math; one tracked backend; checkpoint and tag, one reviewed commit per task, no `Co-Authored-By`, no
pushes unless asked; decisions as 2-4 options with a recommendation (the user does not write code); GPU work deferred; Opus for critical work and Sonnet for
zero-risk tasks, the review stand-in consulted in place of the user. Distribution (2026-10-02): the port ships on GitHub, so optimization is generalized and
deterministic first (same bits on every machine); superseded on 2026-10-06 by the clang-cl `-march=icelake-server` build with
ThinLTO and no PGO (Consult 9; State, Pending); corrected on 2026-10-07: baseline x86-64 with runtime AVX2/AVX-512
dispatch, `-march` local-only.

SP0 rulings R1-R13: SP0 plan, "Execution rulings" (R11b noise-floor method, R12-R13 auto-confirm). SP2 rulings R1-R5 and the SP2 decisions (per-machine chaiNNer parity, balanced helper sharing, engines untouched): SP2 plan and spec. SP3 decisions SP3-P1 to P11 and the accepted residuals (an iterated subdirectory or name containing `..` can escape the checked base directory, Load Images never yields one; a declared file that is a symlink into an output directory, the loopback admin-share alias of a path, or a hardlink is not detected): SP3 plan and spec; owner-level SP3 decisions by the review stand-in (`the decision log`). SP3b decisions: D1-D7 (SP3b spec section 3) and the review stand-in's rulings on the design and on the Task 4 results (`the decision log`, 2026-10-05). SP3c: the probe's pre-registered rules and verdict (`native/reports/sp3c/`). SP4 decisions: SP4 spec, the SP4a plan's P1–P12 and the SP4b plan's Global Constraints and D1–D15 (with the directives of 2026-10-04: the installed chaiNNer is the reference and is judged, tests are judged, finished work is not re-audited); video (owner, 2026-10-03/04): the codecs themselves are not tuned, and video cases are judged by our per-frame timer, never by wall-clock; owner-level rulings by the review stand-in.

SP1 rulings:

- SP1-R1: root `.gitattributes` gains `backend/src/** -text` (bytes stored verbatim: CRLF, LF and
  mixed). The `eol=LF` fix has landed: `.gitattributes` is now `* text=auto eol=lf` (U3 lane E).
- SP1-R2: the byte-identity proof excludes `__pycache__` and the portable profile; the v2 manifest is
  cross-checked against v1, not byte-compared.
- SP1-R3: the real package is never rebuilt from scratch; it is converted in place once
  (`--adopt-v1`, removed afterwards).
- SP1-R4/R5: no shell-digest pin (the two bundle pins suffice); `Package.ps1` deleted, `README.md` gives the venv command.
- SP1-R6: `docs/superpowers/**` stays as the planning record; since 2026-10-07  it is local and git-ignored, so the paths into it in this file name local files.
- SP1-R7: `backend/tests` is the `d56e507f` set; the port's `test_lazy.py` extension stays at the tag.
- SP1-R8: `native/` comes from the `c-port` branch HEAD (SP0 included), not the bare checkpoint.
- SP1-R9: `backend/src` is frozen during SP1; backend cleanups wait for SP2 or later.
- SP1-R10 to R13: the packager v2 lands before the retirement commit; `--adopt-v1` is deleted after
  its one use; the whole-branch review covers the cleanup commits; the branch is kept, nothing
  merged; execution ruling SP1-X8 stores `native/tests/reference_*/**` verbatim too.
- 2026-11-06 (night run's questions): Upscale Image (TensorRT)'s item overlap stays: on images of one tile, about 1 to 5 %
  on ESRGAN-size models and about 12 % on light or 8x ones comes from the overlap itself, no extra VRAM, source +390 -170
  with its fixes; BF16's gain comes from `a2c18dc58` and stays either way. Convert To sRGB stays off: on costs +7 ms on a
  12 MP PNG and leaves the native load, old chains with tagged images change colour, and colours outside sRGB clip.
  Guided Upscale's Tile Size of 80 stays. The TensorRT plugin leak is not reported. FIGSR goes FP16 on its one published
  weight. Spandrel c14 holds DySample's root fix, SPAN's fold and Flex Attention on first use.
- 2026-11-06 (later rulings): DAT's FP32 product stays though it costs 76 % at Fixed 256 px, and a cheaper form is a probe for later.
  The automatic engine is always Dynamic and follows the Tile Size; RGT stops at 256 px; the four convolution architectures keep
  their 16 px; no architecture is excluded. FP16 engines are judged on Dynamic ranges as well as Fixed 256 px: FDAT M and DAT's
  faults only showed above 256 px.

## Records

Local, git-ignored records (SP1-R6). Spec: `docs/superpowers/specs/2026-10-01-chainner-c-program-design.md`. Plans and their execution
rulings: `docs/superpowers/plans/` (`2026-10-01-sp0-foundation.md`, `2026-10-02-sp1-single-source.md`, `2026-10-02-sp2-cpu-budget.md`, `2026-10-02-sp3-cross-item-pipelining.md`, `2026-10-03-sp4a-measurement-dispatch.md`, `2026-10-03-sp4b-kernels.md`, `2026-10-05-sp3b-decode-off-the-producer.md`, `2026-10-05-sp4c-output-allocation.md`).
History: `git show chainner-c-checkpoint-20261001:chainner_c_handoff.md` (2026-09-26 handoff: paths, invariants,
results, 161-node catalog); the retired `native/*.md` reports, overlays and transplant tools are at the same tag.
