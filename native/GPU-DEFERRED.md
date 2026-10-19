# GPU-deferred work

The GPU was busy as of 2026-10-01, so every item that needs it is parked here for later. GPU work follows the owner's
rules of 2026-10-07:
- ESRGAN-architecture models only, never the heavy DAT/DAT2/HAT models;
- benchmarks of about 50 timed iterations per side, averaged (not 1000);
- only publishing needs the owner.

| # | Item | Why it needs the GPU | Origin |
| --- | --- | --- | --- |
| 2 | Retired 2026-10-05: the nightly's GPU lease was removed on the owner's decision, so there is nothing left to check | — | SP1 |
| 3 | Model-inference throughput (Upscale Image) under the new scheduler and pipelining. In progress 2026-10-07: four ESRGAN cases (PyTorch 1x/4x, ONNX CUDA, NCNN Vulkan), one paired run of 48 images per side each, exactness port = oracle and port vs installed (lane `u4/gpu`, report `u4gpu-report.md`) | Inference workloads are excluded from CPU benchmarks | SP2, SP3 |
| 4 | Retired 2026-10-07 (Consult 17): PGO training with real models. The owner's build decision is ThinLTO with no PGO | — | SP5 |
| 5 | Quality check of the chain-scoped Align Image to Reference / Upscale Face helper leases with trained models. On hold: their models (RIFE, GFPGAN) are outside the owner's ESRGAN-only rule | Only mocks and CPU contract tests exist (`ARCHITECTURE.md` section 6; history: `chainner_c_handoff.md` §6 at tag `chainner-c-checkpoint-20261001`) | Pre-program |
| 6 | Upscale results interleaved once per backend: timing. Measure: wall time per 4K image, one tile and tiled, PyTorch, NCNN and ONNX with an ESRGAN model, 48 images, one paired run | Real CUDA, Vulkan and TensorRT-provider device-to-host paths | instant-exec hunt-upscale |
| 7 | Native TensorRT tile blend: timing. Measure: TensorRT Upscale Image tiled to 4K, 48 images, one paired run | A real TensorRT engine | instant-exec hunt-upscale |
| 8 | Narrow the TensorRT session lock (`chaiNNer_tensorrt/tensorrt/processing/upscale_image.py:76,89`) to the tiled upscale calls. Measure: a folder of 48 with run-ahead, wall time and GPU idle gaps | Pays only when GPU work overlaps the CPU tail | instant-exec hunt-upscale |
| 9 | Depth-2 pipelined tile loop with pinned buffers (PyTorch, ONNX, NCNN). Measure: tiled 4K wall time per image | Hides CPU work behind the device | instant-exec hunt-upscale |
| 10 | Remember the working tile size per model, device, dtype and size. Measure: first-tile retries and time over a folder | Only with GPU out-of-memory retries (overlaps upstream #3318 in the owner queue) | instant-exec hunt-upscale |
| 11 | NCNN: clear the Vulkan allocators per image, not per tile. Measure: tiled NCNN wall time and VRAM per image | Vulkan only | instant-exec hunt-upscale |
| 12 | Transparency mode: reuse the black-background pass for fully opaque tiles. Measure: RGBA folder wall time | Saves model passes | instant-exec hunt-upscale |
| 13 | GPU node overlap in the native engine (two independent ESRGAN sinks). Measure: ms per item and VRAM, two sinks, 48 images | No GPU overlap in this run (Fable ruling B): nothing on the GPU could be measured | instant-exec engine |
| 14 | Engine look-ahead through GPU and context nodes feeding VOSR or Temporal Fix (CPU-tested only), with each run's CUDA state on its driver thread. Measure: VOSR and Temporal Fix chain outputs equal to `CHAINNER_C_EXECUTOR=python`'s, VRAM per run | Needs CUDA | instant-exec engine |
| 15 | The UI cache's key with Electron's GPU information on NVIDIA, and the TensorRT node category in the cached list. Measure: first and second launch with the GPU visible: cache hit, TensorRT nodes present | A real NVIDIA driver | instant-exec startup |
| 16 | Save and reload a real TensorRT engine through the atomic writer. Measure: Build Engine -> Save Engine -> Load Engine -> Upscale Image on one image | A real engine | instant-exec codecs |
| 17 | Build Engine cache: real build, memory hit, restart, disk hit (ONNX and DUAL). Measure: build, rerun, restart with Engine Cache on; the cached engine's 512 px output equals the fresh engine's | Real TensorRT | instant-exec hunt-models |
| 18 | ONNX sessions on CUDA, DirectML and TensorRT rebuilt on a GPU, FP16 or cache-path change; NCNN Vulkan sessions rebuilt on a GPU change. Measure: change each setting between two runs; the session is rebuilt and the output equals a fresh load's | Real providers | instant-exec hunt-models |
| 19 | Convert To ONNX on a CUDA device: the loaded model's device and dtype unchanged afterwards; an FP32 export of an FP16-loaded model correct. Measure: convert, then upscale with the same loaded model; compare the exports | CUDA | instant-exec hunt-models |
| 20 | PyTorch Load Models on the GPU: CPU read plus move timing and VRAM during read-ahead. Measure: a folder of 5 ESRGAN models, wall time and peak VRAM | CUDA | instant-exec hunt-models |
| 21 | Memo for model-inference nodes (Upscale Image, Inpaint, Upscale Face, Convert To ONNX/NCNN): key = model file identity, device, precision and every input and setting; admitted by node time >= 1 s whatever the budget's eighth; the previous run's bits accepted in principle; prerequisite: row 22. Measure: rerun after a downstream-only change (upscale skipped), RAM and VRAM retention | Its value is on GPU paths | instant-exec gate (Fable R2) |
| 22 | Load Model cross-run cache (PyTorch, ONNX, NCNN) on the `SharedCache` pattern; prerequisites: no in-place moves of shared models (`interpolate_models.py:87-89`, auto_split), at most 4 models, Load Models items not cached, VRAM retention measured. Measure: reopened chain, connected path, two nodes on one file: load time and VRAM | VRAM | instant-exec gate (Fable R3) |

Add a row when a task hits a GPU dependency. Delete the row once the item is verified, and note the result in `STATUS.md`.
