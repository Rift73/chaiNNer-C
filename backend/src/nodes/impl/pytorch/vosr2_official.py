"""The official VOSR 2.0 bundle: its pinned CSWRY/VOSR revision and files.

Load VOSR Model checks the sizes; native/tools/download_vosr2.py fetches and verifies
the files. Importing this module loads nothing else.
"""

from __future__ import annotations

REVISION = "f24b3061b7f350b81e1907bdcfc27f71fb4ff3f3"
# Remote name, local name, size, SHA-256 (official CSWRY/VOSR revision above).
FILES = (
    (
        "VOSR2/args.json",
        "args.json",
        816,
        "63b2f423eafae1f684ca860eb8fd0e81aa5fac858397f783d8f337871241c281",
    ),
    (
        "VOSR2/checkpoints/ema_model.safetensors",
        "checkpoints/ema_model.safetensors",
        5576385424,
        "bdcaa81e4c675b6074de643e27daafe73eb8125d4d0264d1bf30a004e9644b71",
    ),
    (
        "Qwen-Image-vae-2d/config.json",
        "Qwen-Image-vae-2d/config.json",
        811,
        "1e294faf15188e06fcb72ce09652e8fb7368bd77b8dd794b5b39fe77d610681b",
    ),
    (
        "Qwen-Image-vae-2d/diffusion_pytorch_model.safetensors",
        "Qwen-Image-vae-2d/diffusion_pytorch_model.safetensors",
        178418892,
        "1dd2c67f3f29734d48709513ed181d4f351966d76ba95f77a5cff8e1daf547fa",
    ),
    (
        "torch_cache/checkpoints/dinov2_vitl14_pretrain.pth",
        "dinov2_vitl14_pretrain.pth",
        1217586395,
        "d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428",
    ),
)
