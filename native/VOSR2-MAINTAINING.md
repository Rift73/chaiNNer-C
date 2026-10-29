# VOSR 2.0: maintainer notes

User guide: [`VOSR2.md`](VOSR2.md). Follow-ups: [`STATUS.md`](STATUS.md) Pending. The research journals stay on the
`experimental` branch (for example `git show 99dd0f1a:native/VOSR2-FP8.md`).

## Modules

Node, in `backend/src/packages/chaiNNer_pytorch/pytorch/`: `processing/upscale_image_vosr.py` (checks before any
work, then `ensure_model`; routes Low to PyTorch and Load Video loops to the temporal step). One fixed model, the
official upload: `vosr2_official.py` pins its revision, files, sizes and SHA-256; `vosr2_download.py` (`ensure_model`
into `<storage>/vosr2-model/<revision>/`, hashes remembered by `stored_hash`; the resumable, hashed `download` shared
with FP8). In `backend/src/nodes/impl/pytorch/`:

- `vosr2.py`: bundle paths, pinned configuration (`read_configs`), weights, enums, `MemoryPlan`, input preparation,
  colour correction. `vosr2_cache.py`: weights kept between runs. `vosr2_memory.py`: `vram_budget`.
- `vosr2_context.py` (PyTorch scene focus), `vosr2_scene.py` (input-only routing and window layout, both backends:
  `window_positions` places each axis by dynamic programming, same count as the regular grid, steps 40-56 latents;
  `face_windows` adds a window on each face (YuNet, `find_faces`) whose middle (central half of the box) the layout
  splits, centred on the box, only where it can hold that middle alone; both backends lay it over the grid's
  normalized blend with `face_feather` (1 out to 25 latents from its centre, a raised cosine to 0 by 31), so its edge
  hands over more gently than two grid windows do; `WindowPlan` keeps a video's layout per shot,
  re-planned at a cut; 2026-10-14, after detail-, difference- and amount-based seam triggers failed), `vosr2_streaming.py`
  (Low's whole-image VAE in bounded bands), `vosr2_texture.py` (grain pass, both backends: synthesized grain hashed
  from seed, frame and pixel position; `GRAIN_GAIN` 1.16 is the choice of 2026-10-13, 1.4 times the bicubic
  source grain on Castorice 2x; in a video the node keeps the last input, and areas whose fine texture changed get
  that frame's grain while still areas keep the seed's, 2026-10-14; stateless, so a changed area returns to the
  seed's grain one still frame later; temporal restorations get no grain, the stabilized output does).
- `vosr2_trt_models.py` (the ten `STAGES`, profiles), `vosr2_trt_rewrite.py`, `vosr2_export.py`,
  `vosr2_trt_compile.py` (builder settings), `vosr2_trt_build.py`, `vosr2_fp8_build.py` (compile workers in an owned
  job, `../owned_job.py` and `vosr2_process.py`, 20-minute limit).
- `vosr2_engine_cache.py`, `vosr2_stage_cache.py`, `vosr2_fp8.py` (engine caches, FP8 download);
  `vosr2_trt_pipeline.py` (`FullSession`), `vosr2_trt_execution.py` (CUDA graphs; outputs are borrowed: copy one before
  calling that stage again), `vosr2_trt_restore.py` (outer tiles above 3840 × 2176), `vosr2_trt_plugin.py`.
- `vosr2_temporal.py` (flow, Three Frames), `vosr2_temporal_state.py` (Persistent Detail), `vosr2_temporal_step.py`.
- AMD, in `backend/src/nodes/impl/migraphx/`: `vosr2_migx_compile.py` (the four programs and their fixed shapes, the
  plugin node turned into the op, `load_ops`), `vosr2_migx_build.py` (compile worker, one hour: export, FP8 calibration
  inputs from four synthetic images, compile, check), `vosr2_migx.py` (`ProgramStages`, `MigraphxSession`: `FullSession` with the programs and the VAE stages as
  PyTorch BF16 modules, program cache, `check_usable`).

Policy: BF16 DiT and DINO, FP32 VAE, FP32 on the CPU. Default caps PyTorch at the device total minus other use minus
800 MB; Low at 7 GB total, streaming DiT blocks. A BF16 bundle reads the DiT from its copy
(`vosr2_download.bf16_dit`: `checkpoints/ema_model.bf16.safetensors` and the sidecar `ema_model.bf16.json` with the
format, the source's SHA-256 and file identity, the copy's SHA-256 and size), checked at each cold load (its SHA-256
remembered by file identity, as the official files' are: `stored_hash`) and made again on any difference, under the
model folder's `download.lock`; bump `BF16_FORMAT` when the conversion changes.
`BundlePaths.dit_copy` carries it, so `VosrBundle.__init__` and `load_weights`, whose source keys the BF16 engines, stay
as they are.

## Engine caches

In the backend storage folder a set is `<root>/<key>/` (key: SHA-256 of the identity JSON), published by writing
`manifest.json` after every stage passed its check; a corrupt set moves to `quarantine-*`, `compile.lock` serializes
builds across processes, and a build needs 5.5 GB of free GPU memory. Once the current set is valid, still under the
lock, `remove_unused_sets` touches the set's `last-used` file (never the manifest or plans, whose file identities key
the sessions) and deletes what the cache made and no longer needs: sets whose manifest identity (keying their folder)
is another version's and whose `last-used` (or, without one, newest file) is over 14 days old, since builds sharing one
storage folder (an installed release, a source checkout, the CLI) each keep theirs; `quarantine-*` and `building-*`
folders; and the `migrating-*` folders of killed migrations (named and filled as `migrate_stage_cache` stages them:
`migration_leftover`). Each goes to `quarantine-*` first, so a set with an open file stays whole (Windows refuses the
move) and a deletion stopped halfway is finished by a later check; a folder it cannot read stays. Another GPU's set
(identity differing only in `gpu`, `capability` or `gfx`), timing caches and anything it does not recognize stay.

- **BF16**, `vosr2-tensorrt-bf16`: the five model files' pinned SHA-256 (not their folder), `vosr2.py`, `vosr2_export.py`, `vendor/vosr2`,
  `vosr2_trt_{models,build,compile,rewrite,plugin}.py` and the plugin DLL; torch, TensorRT, CUDA, driver, GPU,
  profiles, `runtime_contract`. Check: each stage within relative L2 0.03 of PyTorch BF16 (DINO 0.05 of FP32), repeats
  bit-identical.
- **FP8**, `vosr2-tensorrt-fp8` (graphs in `vosr2-fp8-graphs/<folder>`): manifest SHA-256, TensorRT, GPU and capability,
  plugin DLL SHA-256, `vosr2_trt_compile.py`. Check: finite and repeatable on seeded inputs.

- **MIGraphX**, `vosr2-migraphx-bf16`, `-fp8` and `-fp16`: as BF16 with MIGraphX's version, GPU name and gfx, the
  op library's SHA-256 and the program shapes instead of TensorRT's; plain programs (MIGraphX loads a file). Checks:
  BF16 and FP16 (`fp16_types`: BF16 tensors FP16, FP32 kept, the op BF16) as BF16 engines; FP8 finite and repeatable,
  its relative L2 from PyTorch BF16 kept in the manifest. FP8: MIGraphX's `quantize_fp8` quantizes FP32 products only,
  so `fp8_products` gives FP32 inputs to exactly the hosted set's Q/DQ products (checked against its graphs: 74, 216,
  108) and turns DINO's FP32 patch convolution BF16; calibration inputs from four synthetic images. `vosr2_trt_build.py`'s export is repeated in `vosr2_migx_build.py`, since moving it would change the BF16
  engines' identity: change both together.

Runtime-only sources stay outside the identity: `migrate_stage_cache` hardlinks a compatible set's plans to the new
key (never edit a plan in place); a compatible set with a missing or changed plan moves to `quarantine-*` and the next
one (or a fresh build) takes its place, while two compatible sets with different plans stop the run naming both
folders. Bump `RUNTIME_CONTRACT` when stage inputs, outputs or preprocessing change.

## TensorRT plugin

`VosrQkvPack` v1 (`native/src/vosr2`): one warp per 64-channel head does RMS normalization, BF16 rounding, adjacent-pair
RoPE and the Q/K/V layout in the original rounding order; batch 1, 24 heads, 1 to 1152 tokens. Optional native target
(README Build; MAINTAINING section 4): SASS sm_80, 86, 89, 120 and compute_120 PTX (no sm_90 or sm_100), hash
`14dfaec0…`; its sm_80 and sm_120 code equals the DLL validated on 2026-10-12. Packages ship it when it was built.

On AMD the same kernel is `native/src/vosr2_hip` (HIP, every ROCm 7.14 target through `cmake/rocm.cmake`): 32-lane
groups whose shuffles have width 32, so wave64 GPUs run two heads per wavefront in the same order; contraction off. It
is the MIGraphX op `chainner_vosr2_qkv` in `vosr2_migx_ops`: one output (custom ops have one), Q, K and V packed
(3, 1, 24, tokens, 64), split back in the graph. Built through HIP's NVIDIA path with `--fmad=false` it matched the CUDA
kernel bit for bit on the RTX 5090 (2026-10-29: 1 to 1152 tokens, three input scales).

## FP8 hosted set

`https://steropes.feralhosting.com/rift/vosr2-fp8-f9e142df5846/` (`vosr2-fp8-` plus 12 hex digits of the manifest's
SHA-256, pinned as `vosr2_fp8.MANIFEST_SHA256`), uploaded 2026-10-13 with server and public hashes checked: 16 files,
3,450,737,806 bytes: three FP8 graphs (`dit_kv`, `dit_cached`, `dino`, with `.onnx.data`), seven BF16/FP32 graphs
(this code's export), `manifest.json` (recipe, source and file SHA-256), `LICENSE` (Apache-2.0),
`NOTICE` (authors, DINOv2, Qwen-Image VAE, changes). `ensure_fp8_engine` checks GPU, manifest,
TensorRT, plugin and disk before it downloads (resumable, hashed) and compiles. No ModelOpt at run time. AMD does
not use the set: MIGraphX 2.16.1's ONNX parser reads FNUZ FP8 only (not the set's OCP E4M3), so MIGraphX FP8 is
MIGraphX's own quantization (ruling 2026-10-29); converting the set to FNUZ (scales doubled) parsed on gfx942 and was
dropped with it (commit 55639ec4 on the AMD branch).

A new set (notes `99dd0f1a:native/VOSR2-FP8.md`; scripts local, git-ignored, `native/local/vosr2/fp8*`):

1. Capture real stage activations from images and video frames (DiT widths 64 and 72 separately).
2. Calibrate with NVIDIA ModelOpt 0.47.0 (max, per node) in its own environment; rescale to amax/448 by (127/448)².
3. Put the Q/DQ pairs on the original graph, with DiT attention, input and output projections and the VAE in BF16/FP32.
4. Compile and compare with BF16 on real images and video (contours, motion error).
5. Assemble the 16 files, upload to the new folder (approved), check hashes, change `MANIFEST_SHA256`.

## Temporal design

A `TemporalStep` lives in `context.iteration.state` for one Load Video loop run (dropped at its end, failure or Stop).
The plain path runs for strength 0, outside a loop, or when `iteration.iterated_by(0)` is not Load Video. At the first
item `lookahead(0)` decides once: at least 1 (Three Frames) or 5 (Persistent Detail's fifth bootstrap observation needs
frame i + 5) selects peek mode, else past frames only. `peek(0, k)` waits for item i + k; None (no such frame, or it
failed) ends the sequence, and a gap in item indices starts a new one. Each frame is restored once; TensorRT restores
the next frame on a one-thread pool, settled before the item returns. Peek-mode outputs equal the removed Temporal
Sequence node's (oracle: that generator, kept in `test_vosr2_temporal_step.py`; 30 GPU frames bit-identical on PyTorch).
Executor side: ARCHITECTURE section 4.

## Vendored code and tests

`backend/src/vendor/vosr2/`: from ComfyUI-VOSR2 `6a21810b` and VOSR `516f292b` (Apache-2.0), `pos_embed.py` from EVA-02
(MIT); licences and changes in its `NOTICE.md`. Edits rebuild BF16 engines. CPU tests, `native/tests/test_vosr2*`:
`.py`, `_fp8` (nodes); `_fp8_fetch` (model and FP8 downloads); `_context`, `_streaming`, `_texture`, `_outer_tiling`; `_trt_rewrite`, `_bf16_pipeline`,
`_acceleration`, `_stage_cache` (engines); `_trt_plugin` with `test_package_recovery_integration.py`;
`_temporal`, `_temporal_state`, `_temporal_step`; plus `test_iteration_view.py`, `test_item_window.py`,
`tests/common/condition.test.ts`; `_migraphx` (the AMD route on the CPU). GPU checks are manual on NVIDIA;
`_migraphx_gpu` (amd_gpu) is the AMD kit's (`CHAINNER_VOSR2_MODEL`, `CHAINNER_VOSR2_FP8=1`).
