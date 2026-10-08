# DUAL specialized ONNX and TensorRT integration for chaiNNer

Updated: 2026-10-08. Canonical implementation: `traiNNer/archs/dual_arch.py` in this repository.

Windows source path:

```text
\\wsl.localhost\Ubuntu\home\rift\workspace\traiNNer-redux\traiNNer\archs\dual_arch.py
```

Linux source path:

```text
/home/rift/workspace/traiNNer-redux/traiNNer/archs/dual_arch.py
```

Reviewed source SHA-256:

```text
a7c579c5cc7179f835b3dcc40f115d64a4faec2adb810bd066160348d1ff0abf
```

This is the final **DUAL3X-derived standalone DUAL**: pre-mixed dynamic group convolutions, ordinary FFN/attention pointwise projections, static EDBB reparameterization, RFB spatial attention, regional channel attention (RCA), and XR/XG cross-scale retrieval. It is **not** the earlier DUAL4 consolidation, deleted DUAL5, or the rolled-back training-speed optimization.

## 1. Delivery and validation status — read before integration

There are two distinct implementations/evidence sets:

1. **Historical XS specialization, measured on Linux:** native C128, selected 16-block schedule, B1 RGB 512×512 → 2048×2048, TensorRT 11.2.1.2, RTX 5090. This used the historical C128 plugins and seeded synthetic nonzero experts. Median CUDA-Graph GPU-only latency was **57.230846 ms**, 303 samples. It was not a trained-checkpoint quality evaluation.
2. **New family-aware integration source in this repository:** all six canonical factories, net scales 1/2/4, static batch 1, explicitly supplied height/width. It generalizes the exporter, RCA/projection specialization and plugin build tooling, and supplies Windows/Linux build paths. **New AOT kernels, DLL/SO binaries and TensorRT engines have not been compiled or runtime-validated in this change. No new latency or quality result is claimed.**

CPU checks cover the 18 factory/scale contracts from the actual source, FP32 graph-lowering comparisons, polyphase/border behavior, nonzero dynamic branches and ONNX structural checking. The CPU plugin interpreter uses the canonical RCA math; it does not execute or validate the CUDA plugin implementation. A successful ONNX checker does not establish TensorRT custom-plugin correctness.

The tools deliberately record `exported_unvalidated`, `compiled_unvalidated` or `built_unvalidated`. Do not present one of these states as a verified engine in chaiNNer. Native TensorRT parsing, plugin output parity and fresh-process runtime validation remain required on **each target operating system** before a production release. The chaiNNer application itself is not modified by this work.

## 2. Canonical factory matrix

Geometry is read from the model returned by the canonical factory; the exporter does not substitute a DUAL2/3/4 architecture or maintain a separate executable factory table.

| Factory | C | Heads | Groups / selected depths | Input transform | Dynamic sites | Internal scales for net 1/2/4 |
|---|---:|---:|---|---|---:|---|
| `dual_light` | 128 | 4 | 4 / `[4,3,3,6]` | RGB pixel-unshuffle 2 | 4 | 2 / 4 / 8 |
| `dual_xs` | 128 | 4 | 4 / `[4,3,3,6]` | Native | 4 | 1 / 2 / 4 |
| `dual_s` | 160 | 5 | 6 × 6 | Native | 6 | 1 / 2 / 4 |
| `dual_m` | 192 | 6 | 6 × 6 | Native | 6 | 1 / 2 / 4 |
| `dual_l` | 224 | 7 | 10 × 6 | Native | 10 | 1 / 2 / 4 |
| `dual_xl` | 256 | 8 | 12 × 6 | Native | 12 | 1 / 2 / 4 |

Shared default contract: head width 32, reconstruction width 64, K=4 experts, G16 local windows, C32/C64 context windows, dot-product attention, RFB rank 16, anchored windows. The raw `DUAL()` constructor's C96 default is **not** one of these factory exports.

Light and XS preserve these **original layer indices**, not just four depth counts:

```text
group 0: [0,1,2,3]
group 1: [3,4,5]
group 2: [0,1,2]
group 3: [0,1,2,3,4,5]
```

Original full-group schedule: local G16, shifted G16, context32+RCA, local G16, shifted G16, context64+RCA. All groups retain their dynamic write-back. The first two local layers are shared with the coarse branch. XR remains after group two.

### Static shape contract

- Input/output are **FP32 NCHW RGB**, batch 1; use the model's normal RGB value convention, ordinarily `[0,1]`. No BGR conversion, extra mean subtraction, clamping or gamma conversion belongs inside this exporter.
- Supply fixed `height` and `width`, each **at least 4 and divisible by 4**. Rectangular shapes are represented independently. Very large configurations exceeding the kernel indexing bound are rejected; this is not a VRAM-fit guarantee.
- The PyTorch architecture supports more boundary cases, but this export API deliberately rejects odd/tiny/non-multiple-of-four inputs. It never silently pads an engine input or claims dynamic-shape support.
- For Light, trunk shape is `(H/2,W/2)`; for all other factories it is `(H,W)`. The coarse branch is half the trunk shape.
- Output shape is `(1,3,H*net_scale,W*net_scale)`.
- Each factory/scale/shape/checkpoint combination gets its own ONNX and engine. Plugins may be shared when their exact width/trunk-shape/SM/ABI specialization is identical.
- Net ×1 still contains XR. HR readout exists when internal scale ≥2; XG exists when internal scale ≥4. In particular, Light ×1 has HR; Light ×2 has XG; Light ×4 has an internal ×8 tail and a 12-channel XG readout followed by an extra PixelShuffle2.

Context windows and some architecture choices are not recoverable from state-dict tensor shapes. The caller must confirm that the checkpoint was trained with the default geometry above. Do not infer compatibility solely from a successful strict state load. Non-default windows, cosine QK, RIB, expert counts or custom schedules need a separately reviewed contract, not a silent fallback.

## 3. Files and entry points

All paths in this table are relative to this repository root.

| Path | Purpose |
|---|---|
| `scripts/dual_tensorrt/spec.py` | Source identity, specialization metadata, shape/indexing checks; no torch/GPU import |
| `scripts/dual_tensorrt/export.py` | Strict checkpoint loading and CPU-only ONNX export |
| `scripts/dual_tensorrt/graph.py` | Explicit NHWC/token-major graph lowering, runtime kernel mixing, PU2, tails, plugins |
| `scripts/dual_tensorrt/kernels.py` | Generalized fixed-owner RCA and projection Triton kernels; no launch on import |
| `scripts/dual_tensorrt/aot.py` | Explicit-target offline cubin/header generation; no GPU query or kernel execution |
| `scripts/dual_tensorrt/build.py` | Native CMake/plugin and `trtexec` command generation; runs only with `--execute` |
| `tensorrt_plugins/dual_tensorrt/` | Parameterized C++/CUDA IPluginV3 implementations and Windows/Linux CMake project |
| `scripts/dual_tensorrt/test_contract.py` | Small CPU-only regression suite |
| `scripts/dual_tensorrt/cpu_graph_oracle.py` | Test-only FP32 interpreter; never an export/runtime fallback |

The new exporter imports **no legacy architecture or historical benchmark script**. Its private source loader executes the canonical file unchanged with only its explicitly optional Triton/direct-attention and trainer-registry imports disabled. This avoids importing/scanning all traiNNer architectures or registering factories inside chaiNNer. It does not mutate global import behavior. Run it in an isolated worker process with the canonical file packaged at the documented relative path.

A source-hash mismatch fails closed. When `dual_arch.py` changes, review the affected math, update the lowering and regression coverage, then update the pinned hash. Do not expose a user-facing “ignore hash” switch.

## 4. Model → deployment form → ONNX

### 4.1 Load actual weights and fold only the static branches

The sequence is:

```python
arch, model = make_model(factory, scale)
model.load_state_dict(state_dict, strict=True)
model.eval().requires_grad_(False)
model.prepare_for_export()
```

`prepare_for_export()` is destructive and terminal for the copy: it folds static EDBB factors, switches spatial attention to SDPA, and guards repeated folding. Never call it on an actively training model. The tooling constructs a private CPU model and does not change the original checkpoint or source.

For an already folded checkpoint, select `--folded`: build the deployment structure first, then strict-load that state. No automatic key stripping, architecture guessing, relaxed load, fake checkpoint, random-weight fallback, or automatic EMA preference is allowed. Direct `.safetensors` and direct tensor state dictionaries are accepted. Wrapped `.pth` dictionaries require an explicit `--state-key`, for example `params_ema` **only if that is actually the key present**. PyTorch loading uses `weights_only=True`.

### 4.2 Keep the dynamic convolution truly dynamic

For each group's feature input `x`, retain:

```text
FP32 GAP(x) → LayerNorm → Linear → GELU → Linear → Softmax − 1/K
                                                ↓
                         coefficients @ flattened expert kernels
                                                ↓
                              add folded static base kernel
                                                ↓
                            cast effective kernel to BF16
                                                ↓
                       one Conv(x, effective kernel, base bias)
```

Equation:

```text
c_k(x) = softmax(router(x))_k − 1/K
W_eff(x) = W_base + Σ_k c_k(x) E_k
y = Conv(x, W_eff(x), b_base)
```

The router and kernel combination are FP32; the convolution body is BF16. There are no expert-specific biases in this model. Every input recomputes routing. Do not constant-fold the router based on one sample, prune zero experts as a general trained-model assumption, replace the mixture by its mean, or substitute stacked expert-output convolutions. Batch >1 would require distinct generated kernels/grouped handling and is intentionally outside this deployment API.

### 4.3 Explicit graph layout and arithmetic

- NHWC/token-major trunk; appropriate NCHW conversions at convolutions.
- 1×1 projections become MatMul+bias where appropriate.
- Spatial attention uses recognizable QK MatMul → optional bias → Softmax → AV MatMul patterns, with exactly one intended attention scaling.
- RFB factors are appended to Q/K; V is zero-padded and the result sliced back to head width 32. This retains the low-rank additive positional term without constructing its dense positional matrix.
- Anchored/shifted window gathers and restores come from the canonical source's index helpers.
- Cross-scale downsampling uses the canonical taps and edge padding; selected BF16/FP32 boundaries follow the historical deployment formulation.
- FFN/gating depthwise expressions use FP32 shifted sums and epilogues before BF16 conversion.
- Tail and XG polyphase rewrites move suitable post-shuffle 3×3 convolutions before the shuffle and expand their phase channels. The XG head/slot/value packing order is distinct from ordinary CRD PixelShuffle and is handled explicitly.
- PU2 RGB packing uses explicit reshape/transpose ordering rather than assuming ONNX SpaceToDepth has PyTorch PixelUnshuffle channel order.
- The final RGB skip and mean restoration are FP32.

The ONNX uses standard opset 20 plus custom `trt` opset 1. It is **TensorRT-specific ONNX**, not a promise of ONNX Runtime/DirectML compatibility. We construct the graph with ONNX helpers; this path is not generic `torch.onnx.export()` and does not export the bundled training/direct Triton attention.

Floating-point reassociation and BF16 boundaries can differ from PyTorch. Algebraic equivalence and the CPU FP32 oracle do not establish bitwise BF16 equality or trained-image quality. Preserve the numerical acceptance step below.

## 5. Plugin specialization

For external input `(H,W)`, let `(Ht,Wt)` be the trunk shape. The plugin key is:

```text
c<C>_h<Ht>_w<Wt>
```

At external 512×512, Light uses `c128_h256_w256`; XS uses `c128_h512_w512`; S/M/L/XL use the corresponding C160/192/224/256 at 512×512. Scale changes the tail/graph but not this trunk specialization key.

| ONNX plugin name | Function |
|---|---|
| `DualNorm_<key>_TRT` | Per-token centered two-pass FP32 normalization; BF16 features and FP32 affine parameters |
| `DualCore_<key>_TRT` | Depthwise Q/K/V, 16×16 cell statistics, four shifted regional channel attentions, BF16 AV rounding, canonical FP32 hat blending |
| `DualProject_<key>_TRT` | Output projection and residual add, preserving BF16 projection rounding before addition |
| `DualAttentionBarrier_TRT` | Shape-preserving BF16 bitwise-copy boundary; **not an attention implementation** |

All creators use version `1`, namespace `""`. Core/project enforce static B1/trunk shape/channel counts. Norm can serve both fine and coarse spatial shapes at the specialized width. Current kernels require head width 32. Non-power-of-two widths 160/192/224 use masked power-of-two projection column tiles; C must not be silently rounded up as a model dimension.

RCA scratch sizing:

```text
heads = C/32
cells = ceil(Ht/16) * ceil(Wt/16)
regions = Σ_(sy,sx in {0,16}²) ceil((Ht+sy)/32) * ceil((Wt+sx)/32)
workspace bytes = cells*heads*1088*4 + regions*heads*1024*2
```

The generalized core retains the prior cell-statistics decomposition and fixed ownership, with no floating-point atomics. These are inference kernels, not the rolled-back deterministic training-speed kernels. They preserve intended BF16 boundaries but change FP32 reduction association relative to the dense reference. Correctness still requires device validation.

### Why the attention barrier exists

XR and XG reuse Q/K but consume sequentially dependent values. When their temperatures match, TensorRT can common their logits/softmax, materialize a long-lived quadratic tensor, and lose independent fused MHA execution. An opaque bitwise-copy boundary helps preserve separate fusion opportunities without changing logits, temperatures or weights. The exporter emits this boundary for the retrieval calls; the attention itself remains a TensorRT-native graph.

Do not remove it as an “identity cleanup” without inspecting the built engine. The historic XS engine contained native `_gemm_mha_v2` tactics, but no fusion/performance result is asserted for the new family engines until inspected.

## 6. Windows and WSL/Linux are separate deployment targets

**An engine or plugin library built on Linux is not a Windows artifact.**

| Artifact | Windows | WSL/Linux |
|---|---|---|
| ONNX + external weights | Same graph can be copied intact | Same graph can be copied intact |
| Plugin library | MSVC/CUDA-built `.dll` | Native Linux `.so` |
| TensorRT engine | Build and validate on Windows | Build and validate on Linux |
| AOT CUDA cubin/header | Can consume reviewed same-SM generated headers | Can generate headers with the existing Triton toolchain |
| Timing cache | Separate hardware/software identity | Separate hardware/software identity |

The build helper rejects a plugin manifest with the wrong OS/machine/specialization or modified library hash. It does not translate WSL paths into Windows paths automatically. Transfer the ONNX **and its `weights.bin`** together; copy the AOT directory as a unit when building native Windows plugins.

The AOT generation path is designed for a supported Triton host, normally WSL/Linux. Native Windows runtime and plugin builds do **not** require importing Triton: generated cubins are embedded in C++ headers. This does not imply a validated cross-OS binary ABI; native wrappers and runtime must still be checked. No dependency installation is performed by these scripts.

Required toolchains:

- CPU export worker: compatible PyTorch, NumPy, ONNX, and safetensors when that format is selected.
- AOT worker: existing compatible Triton/CUDA compilation toolchain. Supply SM explicitly, e.g. `120` for the historical RTX 5090 target; there is no GPU auto-detection or kernel launch in `aot.py`.
- Native plugin build: CMake ≥3.25, CUDA Toolkit, TensorRT SDK with IPluginV3, native C++17 toolchain. Windows helper selects Visual Studio 2022 x64; Linux uses the existing native toolchain.
- Engine build/runtime: native compatible TensorRT and NVIDIA driver/GPU. Historical reference is TensorRT 11.2.1.2; other versions are not silently certified.

The AOT generator checks for zero global/profile scratch and the expected tensor-pointer-plus-two-scratch-pointer PTX ABI. If that changes, it fails rather than launching with guessed arguments. It neither installs another compiler nor retries on failure.

## 7. Commands and chaiNNer worker API

Run commands from this repository root. These are integration instructions, **not commands executed as part of this document change**. Export is CPU work; AOT/plugin commands compile code; `build engine --execute` uses the target GPU. None of these commands performs a latency benchmark or starts training.

### Step A: export a real checkpoint on CPU

WSL/Linux example:

```bash
cd /home/rift/workspace/traiNNer-redux
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  ./venv/bin/python -m scripts.dual_tensorrt.export \
  --factory dual_light --scale 4 --height 512 --width 512 \
  --checkpoint /absolute/path/to/your_dual_light_x4.safetensors \
  --output /absolute/path/to/new_export_directory
```

Windows worker uses the same module arguments with its existing Python and Windows paths:

```powershell
Set-Location 'C:/path/to/traiNNer-redux'
$env:CUDA_VISIBLE_DEVICES = ''
& 'C:/path/to/existing/python.exe' -m scripts.dual_tensorrt.export `
  --factory dual_xl --scale 2 --height 512 --width 768 `
  --checkpoint 'D:/models/dual_xl_x2.safetensors' `
  --output 'D:/dual-builds/xl_x2_512x768/export'
```

Keep the CPU-only environment scoped to that worker. Do not propagate an empty `CUDA_VISIBLE_DEVICES` to the TensorRT build worker. Do not overwrite the user's global environment.

Results: `model.onnx`, external `weights.bin` when present, and `export.json`. The manifest records checkpoint/source hashes, explicit factory/scale/geometry, dynamic-site count, precision and plugin names. Output directories must not already exist.

Callable API for a dedicated export process:

```python
from scripts.dual_tensorrt.export import export_checkpoint

manifest = export_checkpoint(
    factory="dual_m", scale=4, height=512, width=512,
    checkpoint="/absolute/path/to/model.safetensors",
    output="/absolute/path/to/new_export",
    state_key=None, folded=False,
)
```

Use `--state-key` only for explicitly selected wrapped checkpoint content. A scale-specific checkpoint is required; this tool does not transplant a ×4 tail into ×1/×2.

### Step B: generate explicit-SM AOT headers

```bash
./venv/bin/python -m scripts.dual_tensorrt.aot \
  --export-dir /absolute/path/to/new_export_directory \
  --output /absolute/path/to/new_aot_directory --sm 120
```

This explicitly invokes compilation. It is not a “preview” command. Outputs: `Core.h`, `Project.h`, `aot.json`. The manifest pins kernel source, specialization, SM, Triton version and header hashes. Cache these by those identities, not just by factory name. Move this directory intact to Windows when using WSL for AOT generation.

### Step C: compile native plugins

```bash
./venv/bin/python -m scripts.dual_tensorrt.build plugins \
  --export-dir /absolute/path/to/new_export_directory \
  --aot-dir /absolute/path/to/new_aot_directory \
  --trt-root /absolute/path/to/native/TensorRT \
  --output /absolute/path/to/new_linux_plugin_build
```

Without `--execute`, this validates the input manifests and returns CMake argv only. Add `--execute` when intentionally building. Windows uses the same module with Windows paths, the native TensorRT SDK, Visual Studio 2022 and CUDA Toolkit; it produces DLLs under the Release directory. Linux produces SOs. The helper writes `plugins.json` with platform/SM/specialization/library hashes and exact absolute library paths.

Do not pass a Linux SDK, `.so`, or `/home/...` path into a native Windows build. If the default native CUDA compiler is not configured, select the existing intended toolkit in the caller's process or review the printed CMake command; do not auto-install or switch toolchains.

### Step D: build the native TensorRT engine

```bash
./venv/bin/python -m scripts.dual_tensorrt.build engine \
  --export-dir /absolute/path/to/new_export_directory \
  --plugins-file /absolute/path/to/new_linux_plugin_build/plugins.json \
  --output /absolute/path/to/new_engine_directory \
  --trtexec /usr/bin/trtexec
```

Again, add `--execute` only to perform the build. Native Windows uses the same arguments with native paths and `trtexec.exe`.

The generated command retains the historical policy:

```text
--onnx=<model.onnx>
--saveEngine=<model.engine>
--builderOptimizationLevel=3
--stronglyTyped
--noTF32
--maxAuxStreams=0
--memPoolSize=workspace:8G
--profilingVerbosity=detailed
--skipInference
--timingCacheFile=<new-build timing.cache>
--dynamicPlugins=<native library>          (for each of four libraries)
--setPluginsToSerialize=<same library>     (for each of four libraries)
```

Precision is encoded by ONNX tensor types/casts; this is not `--fp16` blanket conversion. No FP8/INT8 calibration, structural sparsity, expert pruning or training compile cache is used. Eight GiB is a **builder workspace allowance**, not a total GPU memory limit. Optimization level 3 is the reproduced policy, not a claim that it is best on every factory/GPU. This implementation uses a fresh per-build timing-cache path; it does not silently reuse unrelated caches.

Build workers are foreground and bounded, retain command/log/exit records and do not retry. Cancellation/timeout targets only their owned process group/tree. chaiNNer should allow cancellation, surface the retained log and not automatically retry until something passes. A build exit code and engine file alone are not runtime validation.

## 8. chaiNNer integration contract

Recommended distinct stages/node responsibilities:

```text
Load checkpoint + explicit factory/scale/training-geometry confirmation
    → specialized CPU ONNX export + manifest
    → select matching native plugin bundle (or explicitly build it)
    → native TensorRT engine build + receipt
    → fresh-process numerical validation
    → load verified engine for inference
```

### Inputs and UI

- Factory dropdown with exactly the six names, separate scale dropdown 1/2/4, static H/W, and target worker OS/GPU-SM/toolchain identity.
- Explicit checkpoint selection and raw-vs-EMA/state key selection where needed; do not silently replace selected weights.
- Show PU2/internal scale and required trunk plugin key. Do not label Light as native C128 XS.
- Show that the ONNX contains custom TensorRT nodes and requires its matching plugin bundle.
- Show `Unvalidated` until independent engine validation succeeds. Do not attach the old 57 ms measurement to new factories or checkpoints.
- Export, compilation, engine build, runtime checking, and optional benchmarking must be separate operations. No benchmark or user training is started by these tools.
- Leave image preprocessing/postprocessing and node data types compatible with chaiNNer's existing contracts. Do not change image color order, range or rounding to make a parity check pass.

### Required artifact/cache identity

Include architecture SHA, exporter/kernel/plugin source revisions, factory, selected indices, scale, PU2 flag, training geometry, external input and trunk shapes, checkpoint SHA/state key/folded flag, ONNX+external-data hashes, precision, plugin creator/version/namespace/binary hashes, SM, OS/architecture, TensorRT/CUDA/driver versions, builder flags and timing-cache provenance. Builder/version/device identities not directly collected by the current helper must be supplied/recorded by the integration worker from its actual environment.

The helper manifests are provenance inputs, not signatures and not a complete global cache implementation. Never accept an arbitrary manifest's claimed validation status as a trust decision. A plugin/engine can contain executable native code; load only trusted locally built or explicitly trusted binaries, after hash verification. Use an isolated worker for user-supplied native artifacts.

### Engine loading and execution

The historical runtime allowed embedded engine host code, deserialized the serialized plugin libraries with the engine, created an execution context, set the FP32 `input`/`output` tensor addresses, and called `execute_async_v3` on the caller's CUDA stream. Embedded plugins do not eliminate TensorRT/CUDA/native dependency compatibility requirements.

If the integration enables `runtime.engine_host_code_allowed = True`, make the trust decision explicit; do not enable it indiscriminately for downloaded engines. Avoid separately preloading the same serialized plugin libraries into an already populated creator registry. The historical XS build process reported duplicate-plugin deserialization errors; fresh-process deserialization and output checks subsequently succeeded. Preserve and surface such diagnostics, do not suppress them as blanket harmless warnings.

For multiple models, reuse a single matching plugin bundle where possible. Do not load different binaries with identical creator name/version/namespace into the same process (including differently compiled SM targets). Isolate incompatible plugin-bundle identities in separate native workers. The generic attention barrier intentionally shares one creator identity; do not repeatedly register distinct copies of it for each factory. Concurrent multi-engine registry coexistence has not been validated here.

CUDA Graph capture is optional runtime orchestration, not a property guaranteed by the ONNX. Warm the execution context, stabilize shapes/addresses, capture only engine execution, and replay on the intended stream. Do not capture allocations, data transfers, plugin first-load initialization or changing shapes. Record whether latency includes H2D/D2H, synchronization and Python overhead.

### Padding and tiling caveat

The engine does not accept arbitrary image sizes. A caller may pad/crop or tile outside it, but DUAL's global router pooling and cross-scale retrieval can change with tile extent, borders and donor support. Tiled inference is **not generally identical** to whole-image inference, even with overlap. An external padding policy also changes the router input. Treat seam/quality behavior as a separate real-image integration check, not as a guaranteed consequence of static ONNX parity.

## 9. Acceptance gates before calling a new specialization verified

1. Check factory/scale/source/checkpoint identities, all dynamic sites and default geometry. Validate ONNX structure and required plugin creators.
2. Check generated kernels and plugin outputs against canonical references for the new width/trunk shape. Include border regions, non-power-of-two widths, nonzero theta, all channels and masked projection columns. A C128 result is not a C224 result.
3. Build with the intended native toolchain and preserve the full log. Inspect the resulting graph for intended native MHA fusion and unexpected full attention-map materialization; no universal layer count is prescribed.
4. Deserialize in a **fresh process** on each target OS, without relying on a prior process's plugin registration. Verify exact static input/output shapes and FP32 I/O.
5. Compare the actual checkpoint's outputs against canonical folded PyTorch: random plus structured black/gray/checker/edge patterns, border errors, finite outputs and repeatability. Include nonzero expert/router response so a frozen or omitted dynamic branch cannot hide at zero initialization. For a real trained checkpoint also inspect representative image outputs; synthetic parity is not image quality evidence.
6. Establish a tolerance against that checkpoint's native BF16-vs-FP32 error floor. Historical gate: TensorRT relative-L2 ≤ `max(2 * native_BF16_floor, 1e-6)`, max absolute error <0.02, all finite, repeated engine outputs equal. This is an historical starting policy, not a universal safety/quality guarantee.
7. Only after those gates should the integration mark the engine verified. Benchmark only when separately requested, with the timing contract and full sample record. Never infer latency from MACs, parameter count or the old XS result.

No new GPU/device validation was run for this delivery. No user dataset, checkpoint, training log or TensorBoard inventory was accessed. CPU tests use synthetic weights and temporary test data only.

## 10. Historical record and primary references

The final historical XS wrapper is `C:/Users/PC/research/dual3x_consolidation_20261007/benchmark_xs.py`; its helper was `C:/Users/PC/research/fdat_dual3_trt_20261004/bench.py`. The graph builder was `scripts/dual2_plus/export.py`, with C128 plugins under `tensorrt_plugins/dual2_plus` and the barrier under `tensorrt_plugins/graft_xg`.

Exact historical artifacts:

```text
/home/rift/workspace/dual3x-consolidation_20261007/xs_d16/optimized/model.onnx
/home/rift/workspace/dual3x-consolidation_20261007/xs_d16/optimized/model.engine
/home/rift/workspace/dual3x-consolidation_20261007/xs_d16/optimized/build_command.json
/home/rift/workspace/dual3x-consolidation_20261007/xs_d16/optimized/export.json
/home/rift/workspace/dual3x-consolidation_20261007/xs_d16/optimized/runtime.json
C:/Users/PC/research/dual3x_consolidation_20261007/HANDOFF.txt
C:/Users/PC/research/dual3x_consolidation_20261007/FINAL_AUDIT.json
```

Historic engine SHA-256: `65966653cd948e1db593a855070908cab106a0ddeb3fb73c29566e8152aaf4a8`.

Its three block medians were 57.241055 / 57.353695 / 57.198078 ms. GPU-only CUDA Graph timing excluded compilation, transfers and Python. The source executable model AST matched the final canonical model except documentation/annotations. Those measurements do not certify new plugin binaries, PU2 Light, other widths/scales, Windows runtime or trained-checkpoint quality.

Primary technical references (consulted 2026-10-08; recheck against the toolchain being integrated):

- [NVIDIA TensorRT support matrix: platform/engine portability](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html).
- [NVIDIA trtexec: plugin loading and serialization](https://github.com/NVIDIA/TensorRT/blob/main/samples/trtexec/README.md).
- [NVIDIA TensorRT performance/benchmarking guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/benchmarking.html).
- [NVIDIA custom layer/plugin documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/extending-custom-layers.html).
- [Triton explicit-target AOT compile tool](https://github.com/triton-lang/triton/blob/main/python/triton/tools/compile.py).

These sources explain API behavior and portability; they do not substitute for local binary/runtime validation.

## chaiNNer-C additions (not traiNNer's)

- `scripts/dual_tensorrt/dynamic.py`: the dynamic-shape exporter (one ONNX for every input size; plugin
  nodes keyed by width only, `DualCore_c128_TRT`). It overrides only graph.py's size-dependent parts; see its
  docstring and chaiNNer-C's `nodes/impl/tensorrt/dual_aot.py` (`register_dynamic`) for the plugins.
- `graph.py`: only the RGB readout moves ahead of its pixel shuffle (`c0de1572`).
