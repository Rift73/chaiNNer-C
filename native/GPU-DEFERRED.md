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

Add a row when a task hits a GPU dependency. Delete the row once the item is verified, and note the result in `STATUS.md`.
