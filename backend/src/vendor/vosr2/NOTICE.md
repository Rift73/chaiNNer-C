# VOSR 2.0 inference architecture sources

Adapted from **ComfyUI-VOSR2**, copyright 2026 ComfyUI-VOSR2 contributors,
Apache-2.0, revision `6a21810b3f4f1a0ffc477c55330db0ac03c671f0`:
https://github.com/ylchen333/ComfyUI-VOSR2/tree/6a21810b3f4f1a0ffc477c55330db0ac03c671f0/models

Underlying VOSR reference: https://github.com/cswry/VOSR at
`516f292b99cf23c76fdc33351e86dc4f97711fe8` (Apache-2.0).
LightningDiT is credited to Jingfeng Yao / HUST-VL, built on DiT and SiT;
Qwen-Image 2D VAE to the Qwen-Image, Wan and Hugging Face teams;
DINOv2 and SwiGLU to Meta Platforms, Inc. and affiliates (Apache-2.0).
See `LICENSE-APACHE-2.0.txt`.

`pos_embed.py` derives from EVA-02 by Yuxin Fang, copyright (c) 2023 Beijing
Academy of Artificial Intelligence (BAAI), under MIT, itself adapted from
lucidrains/rotary-embedding-torch. See `LICENSE-EVA-MIT.txt`.

chaiNNer-C modifications: remove all Comfy imports and operation wrappers, use
plain torch.nn with meta-device construction in the loader, remove the unused
VAE filesystem loader and compile decorator, and add DiT cancellation callbacks.
The whole-image research extension adds rectangular RoPE/unpatchification,
chunked attention queries and token operations, and optional DiT block offload.
These optional execution paths do not change the checkpoint parameter names.
`rmsnorm.py` is a local implementation of the RMS-normalization equation, not
the upstream Llama 2 source file. Model weights are separate downloads and are
not distributed here. No Comfy loader, model patcher or execution server is used.
