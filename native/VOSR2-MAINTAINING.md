# VOSR 2.0: maintainer notes

User guide: [`VOSR2.md`](VOSR2.md). Follow-ups: [`STATUS.md`](STATUS.md) Pending. The research journals stay on the
`experimental` branch (for example `git show 99dd0f1a:native/VOSR2-FP8.md`).

## Modules

Nodes, in `backend/src/packages/chaiNNer_pytorch/pytorch/`: `io/load_vosr_model.py`, `processing/upscale_image_vosr.py`
(checks before any work, routes Low to PyTorch and Load Video loops to the temporal step). Type `VosrModel`
(`nodes/properties/*/vosr_*.py`, `src/common/types/chainner-scope.ts`), value `BundlePaths`. In
`backend/src/nodes/impl/pytorch/`:

- `vosr2.py`: bundle paths, pinned configuration (`read_configs`), weights, enums, `MemoryPlan`, input preparation,
  colour correction. `vosr2_cache.py`: weights kept between runs. `vosr2_memory.py`: `vram_budget`.
- `vosr2_context.py` (PyTorch scene focus), `vosr2_scene.py` (input-only routing, both backends), `vosr2_streaming.py`
  (Low's whole-image VAE in bounded bands), `vosr2_texture.py` (grain pass, both backends).
- `vosr2_trt_models.py` (the ten `STAGES`, profiles), `vosr2_trt_rewrite.py`, `vosr2_export.py`,
  `vosr2_trt_compile.py` (builder settings), `vosr2_trt_build.py`, `vosr2_fp8_build.py` (compile workers in an owned
  job, `vosr2_process.py`, 20-minute limit).
- `vosr2_engine_cache.py`, `vosr2_stage_cache.py`, `vosr2_fp8.py` (engine caches, FP8 download);
  `vosr2_trt_pipeline.py` (`FullSession`), `vosr2_trt_execution.py` (CUDA graphs; outputs are borrowed: copy one before
  calling that stage again), `vosr2_trt_restore.py` (outer tiles above 3840 × 2176), `vosr2_trt_plugin.py`.
- `vosr2_temporal.py` (flow, Three Frames), `vosr2_temporal_state.py` (Persistent Detail), `vosr2_temporal_step.py`.

Policy: BF16 DiT and DINO, FP32 VAE, FP32 on the CPU. Default caps PyTorch at the device total minus other use minus
800 MB; Low at 7 GB total, streaming DiT blocks.

## Engine caches

In the backend storage folder a set is `<root>/<key>/` (key: SHA-256 of the identity JSON), published by writing
`manifest.json` after every stage passed its check; a corrupt set moves to `quarantine-*`, `compile.lock` serializes
builds across processes, and a build needs 5.5 GB of free GPU memory.

- **BF16**, `vosr2-tensorrt-bf16`: hashes of the five bundle files, `vosr2.py`, `vosr2_export.py`, `vendor/vosr2`,
  `vosr2_trt_{models,build,compile,rewrite,plugin}.py` and the plugin DLL; torch, TensorRT, CUDA, driver, GPU,
  profiles, `runtime_contract`. Check: each stage within relative L2 0.03 of PyTorch BF16 (DINO 0.05 of FP32), repeats
  bit-identical.
- **FP8**, `vosr2-tensorrt-fp8` (graphs in `vosr2-fp8-graphs/<folder>`): manifest SHA-256, TensorRT, GPU and capability,
  plugin DLL SHA-256, `vosr2_trt_compile.py`. Check: finite and repeatable on seeded inputs.

Runtime-only sources stay outside the identity: `migrate_stage_cache` hardlinks a compatible set's plans to the new
key (never edit a plan in place). Bump `RUNTIME_CONTRACT` when stage inputs, outputs or preprocessing change.

## TensorRT plugin

`VosrQkvPack` v1 (`native/src/vosr2`): one warp per 64-channel head does RMS normalization, BF16 rounding, adjacent-pair
RoPE and the Q/K/V layout in the original rounding order; batch 1, 24 heads, 1 to 1152 tokens. Optional native target
(README Build; MAINTAINING section 4): SASS sm_80, 86, 89, 120 and compute_120 PTX (no sm_90 or sm_100), hash
`14dfaec0…`; its sm_80 and sm_120 code equals the DLL validated on 2026-10-12. Packages ship it when it was built.

## FP8 hosted set

`https://steropes.feralhosting.com/rift/vosr2-fp8-f9e142df5846/` (`vosr2-fp8-` plus 12 hex digits of the manifest's
SHA-256, pinned as `vosr2_fp8.MANIFEST_SHA256`), uploaded 2026-10-13 with server and public hashes checked: 16 files,
3,450,737,806 bytes: three FP8 graphs (`dit_kv`, `dit_cached`, `dino`, with `.onnx.data`), seven BF16/FP32 graphs
(this code's export), `manifest.json` (recipe, source and file SHA-256), `LICENSE` (Apache-2.0),
`NOTICE` (authors, DINOv2, Qwen-Image VAE, changes). `ensure_fp8_engine` checks GPU, manifest, official bundle,
TensorRT, plugin and disk before it downloads (resumable, hashed) and compiles. No ModelOpt at run time.

A new set (notes `99dd0f1a:native/VOSR2-FP8.md`; scripts local, git-ignored, `native/local/vosr2/fp8*`):

1. Capture real stage activations from images and video frames (DiT widths 64 and 72 separately).
2. Calibrate with NVIDIA ModelOpt 0.47.0 (max, per node) in its own environment; rescale to amax/448 by (127/448)².
3. Put the Q/DQ pairs on the original graph, with DiT attention, input and output projections and the VAE in BF16/FP32.
4. Compile and compare with BF16 on real images and video (contours, motion error).
5. Assemble the 16 files, upload to the new folder (owner-approved), check hashes, change `MANIFEST_SHA256`.

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
`.py`, `_fp8` (nodes); `_context`, `_streaming`, `_texture`, `_outer_tiling`; `_trt_rewrite`, `_bf16_pipeline`,
`_acceleration`, `_stage_cache` (engines); `_fp8_fetch`; `_trt_plugin` with `test_package_recovery_integration.py`;
`_temporal`, `_temporal_state`, `_temporal_step`; plus `test_iteration_view.py`, `test_item_window.py`,
`tests/common/condition.test.ts`. GPU checks are manual.
