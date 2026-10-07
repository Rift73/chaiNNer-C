from __future__ import annotations

import zipfile
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
from torch.nn import functional

from api import NodeContext
from nodes.impl.pytorch.resource_cache import (
    allow_model_reuse,
    file_identity,
    lease_resource,
)
from nodes.impl.pytorch.rife.IFNet_HDv3_v4_14_align import IFNet
from nodes.impl.pytorch.utils import np2tensor, safe_cuda_cache_empty, tensor2np
from nodes.impl.pytorch.xfeat.xfeat_align import XFeat, gen_grid
from nodes.properties.inputs import BoolInput, EnumInput, ImageInput
from nodes.properties.outputs import ImageOutput

from ...settings import PyTorchSettings, get_settings
from .. import processing_group


class PrecisionMode(Enum):
    FIFTY_PERCENT = 1
    ONE_HUNDRED_PERCENT = 2
    TWO_HUNDRED_PERCENT = 3
    FOUR_HUNDRED_PERCENT = 4


def calculate_padding(height: int, width: int, modulus: int) -> tuple[int, int]:
    pad_height = (modulus - height % modulus) % modulus
    pad_width = (modulus - width % modulus) % modulus
    return (pad_height, pad_width)


def download_model(
    download_url: str,
    model_dir: Path,
    model_file: str,
    zip_inner_path: str | None = None,
) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / model_file
    if model_path.exists():
        return model_path
    try:
        response = requests.get(download_url)
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(
            f"Failed to download the file from {download_url}. Error: {e}"
        ) from e
    if zip_inner_path is not None:
        zip_path = model_dir / "model_temp.zip"
        with open(zip_path, "wb") as f:
            f.write(response.content)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zipped_file_path = f"{zip_inner_path}/{model_file}"
            zip_ref.extract(zipped_file_path, model_dir)
        extracted_file_path = model_dir / zipped_file_path
        if extracted_file_path != model_path:
            extracted_file_path.rename(model_path)
        zip_path.unlink(missing_ok=True)
        subfolder = model_dir / zip_inner_path
        try:
            subfolder.rmdir()
        except OSError:
            pass
    else:
        with open(model_path, "wb") as f:
            f.write(response.content)
    return model_path


def align_images(
    context: NodeContext,
    clip: np.ndarray,
    ref: np.ndarray,
    mask: np.ndarray | None,
    precision_mode: PrecisionMode,
    lq_input: bool,
    wide_search: bool,
    exec_options: PyTorchSettings,
) -> np.ndarray:
    device = exec_options.device
    paths = [context.storage_dir / "rife_v4.14/weights/flownet.pkl"]
    if wide_search:
        paths.append(context.storage_dir / "xfeat/weights/xfeat.pt")

    def key():
        return (
            str(device),
            exec_options.use_fp16,
            wide_search,
            tuple(file_identity(p) for p in paths),
        )

    with lease_resource(
        context,
        key,
        lambda: _load_alignment_models(context, wide_search, exec_options),
        reuse=allow_model_reuse(
            device.type, exec_options.budget_limit, exec_options.force_cache_wipe
        ),
    ) as models:
        return _align_images(
            clip, ref, mask, precision_mode, lq_input, wide_search, exec_options, models
        )


@processing_group.register(
    schema_id="chainner:pytorch:image_align_rife",
    name="Align Image to Reference",
    description=[
        "Aligns and removes distortions by warping an Image towards a Reference Image. Tips for best results:",
        "- Always crop black borders or letterboxes on both images.",
        "- If Image's brightness or colors are vastly different, make Reference Image roughly match.",
    ],
    icon="BsRulers",
    inputs=[
        ImageInput("Image", channels=3),
        ImageInput("Reference Image", channels=3),
        ImageInput("Mask")
        .make_optional()
        .with_docs(
            "Optionally use a black & white mask where white excludes areas from warping, like a watermark or text that is only on one image. Masked areas will instead be warped like the surroundings.",
            "The position of masked areas are based on the Reference Image.",
            hint=True,
        ),
        EnumInput(
            PrecisionMode,
            label="Precision",
            default=PrecisionMode.TWO_HUNDRED_PERCENT,
            option_labels={
                PrecisionMode.FIFTY_PERCENT: "50%",
                PrecisionMode.ONE_HUNDRED_PERCENT: "100%",
                PrecisionMode.TWO_HUNDRED_PERCENT: "200%",
                PrecisionMode.FOUR_HUNDRED_PERCENT: "400%",
            },
        ).with_docs(
            "Speed/Quality tradeoff with higher meaning finer, more exact alignment up to a subpixel level. Higher is slower and requires more VRAM.",
            "100% or 200% works great in most cases. 400% is only needed if both input images are very high quality.",
            hint=True,
        ),
        BoolInput("Wide Search", default=False).with_docs(
            "Enables a large search radius. When turned on completely different crops, shearing, and rotations up to 45° can be aligned.",
            "Recommended if the misalignment is larger than about 20 pixel.",
            hint=True,
        ),
        BoolInput("Low Quality Input", default=False).with_docs(
            "Enables better handling for low-quality input images. When turned on general shapes are prioritized over high-frequency details like noise, grain, or compression artifacts by averaging the warping across a small area.",
            "Also fixes an issue sometimes noticeable in Anime images, where lines can get slightly thicker/thinner due to the warping.",
            hint=True,
        ),
    ],
    outputs=[ImageOutput(shape_as=1).with_docs("Returns the aligned image.")],
    node_context=True,
)
def align_image_to_reference_node(
    context: NodeContext,
    clip: np.ndarray,
    ref: np.ndarray,
    mask: np.ndarray | None,
    precision_mode: PrecisionMode,
    wide_search: bool,
    lq_input: bool,
) -> np.ndarray:
    exec_options = get_settings(context)
    context.add_cleanup(
        safe_cuda_cache_empty,
        after="node" if exec_options.force_cache_wipe else "chain",
    )
    return align_images(
        context, clip, ref, mask, precision_mode, lq_input, wide_search, exec_options
    )


def _load_alignment_models(
    context: NodeContext, wide_search: bool, exec_options: PyTorchSettings
):
    device = exec_options.device
    fp16 = exec_options.use_fp16
    rife_model_path = download_model(
        download_url="https://drive.usercontent.google.com/download?id=1BjuEY7CHZv1wzmwXSQP9ZTj0mLWu_4xy&export=download&authuser=0",
        model_dir=context.storage_dir / "rife_v4.14/weights",
        model_file="flownet.pkl",
        zip_inner_path="train_log",
    )
    state_dict = torch.load(rife_model_path, map_location=device, weights_only=True)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    rife_align = IFNet().to(device)
    loaded = rife_align.load_state_dict(state_dict, strict=False)
    rife_align.eval()
    if fp16:
        rife_align.half()
    descriptor = matcher = None
    if wide_search:
        xfeat_model_path = download_model(
            download_url="https://raw.githubusercontent.com/verlab/accelerated_features/e92685f57f8318b18725c5c8c0bd28c7fe188d9a/weights/xfeat.pt",
            model_dir=context.storage_dir / "xfeat/weights",
            model_file="xfeat.pt",
        )
        state_dict = torch.load(
            xfeat_model_path, map_location=device, weights_only=True
        )
        descriptor = XFeat(
            top_k=3000, weights=state_dict, height=480, width=704, device=device
        )
        matcher = XFeat(device=device)
        if fp16:
            descriptor.half()
            matcher.half()
    return ((rife_align, descriptor, matcher), not loaded.missing_keys)


def _align_images(
    clip: np.ndarray,
    ref: np.ndarray,
    mask: np.ndarray | None,
    precision_mode: PrecisionMode,
    lq_input: bool,
    wide_search: bool,
    exec_options: PyTorchSettings,
    models: tuple[IFNet, XFeat | None, XFeat | None],
) -> np.ndarray:
    device = exec_options.device
    fp16 = exec_options.use_fp16
    precision = precision_mode.value
    blur = 2 if lq_input else 0
    smooth = 11 if lq_input else 0
    s1 = (16, 8, 4, 2)
    s2 = (8, 4, 2, 1)
    s3 = (4, 2, 1, 0.5)
    s4 = (2, 1, 0.5, 0.25)
    rife_align, descriptor, matcher = models
    fclip = np2tensor(clip, change_range=True).to(device)
    fref = np2tensor(ref, change_range=True).to(device)
    if mask is not None:
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        fmask = np2tensor(mask, change_range=True).to(device)
    else:
        fmask = None
    if fp16:
        fclip = fclip.half()
        fref = fref.half()
    with torch.inference_mode():
        if wide_search:
            assert descriptor is not None and matcher is not None
            fref = functional.pad(fref, (4, 4, 4, 4), mode="replicate")
            _, _, fref_h, fref_w = fref.shape
            _, _, fclip_h, fclip_w = fclip.shape
            fclip_points = descriptor.detectAndCompute(fclip)[0]
            fref_points = descriptor.detectAndCompute(fref)[0]
            kpts1, descs1 = (fclip_points["keypoints"], fclip_points["descriptors"])
            kpts2, descs2 = (fref_points["keypoints"], fref_points["descriptors"])
            points_found = len(kpts1) > 100 and len(kpts2) > 100
            if points_found:
                idx0, idx1 = matcher.match(descs1, descs2, 0.82)
                match_found = len(idx0) > 50
                if match_found:
                    pts1 = kpts1[idx0].cpu().numpy()
                    pts2 = kpts2[idx1].cpu().numpy()
                    homography, _ = cv2.findHomography(
                        pts1,
                        pts2,
                        cv2.USAC_MAGSAC,
                        10.0,
                        maxIters=1000,
                        confidence=0.995,
                    )
                    homography_t = torch.tensor(
                        homography, dtype=torch.float32, device=device
                    )
                    homography_t = gen_grid(
                        homography_t, fclip_h, fclip_w, fref_h, fref_w, device
                    )
                    if fp16:
                        homography_t = homography_t.half()
                    fclip = functional.grid_sample(
                        fclip,
                        homography_t,
                        mode="bicubic",
                        padding_mode="border",
                        align_corners=True,
                    )
                    corners_src = np.float32(
                        [  # type: ignore
                            [0, 0],
                            [fclip_w - 1, 0],
                            [fclip_w - 1, fclip_h - 1],
                            [0, fclip_h - 1],
                        ]
                    ).reshape(-1, 1, 2)
                    corners_dst = cv2.perspectiveTransform(corners_src, homography)
                    homography_mask = np.ones((fref_h, fref_w), dtype=np.float32)
                    cv2.fillConvexPoly(  # type: ignore
                        homography_mask,
                        np.int32(corners_dst),  # type: ignore
                        0.0,  # type: ignore
                    )  # fill inside corners with black
                    corners = corners_dst.reshape(-1, 2)
                    min_xy = np.floor(corners.min(axis=0)).astype(int)
                    max_xy = np.ceil(corners.max(axis=0)).astype(int)
                    min_x, min_y = np.maximum(min_xy, 0)
                    max_x, max_y = np.minimum(max_xy, [fref_w, fref_h])
                    if max_x - min_x < 16:
                        min_x, max_x = (0, 16)
                    if max_y - min_y < 16:
                        min_y, max_y = (0, 16)
                    if fmask is not None:
                        homography_mask = torch.from_numpy(homography_mask)[
                            None, None
                        ].to(device)
                        if fmask.shape[2] != fref_h or fmask.shape[3] != fref_w:
                            fmask = functional.interpolate(
                                fmask, size=(fref_h - 8, fref_w - 8), mode="nearest"
                            )
                            fmask = functional.pad(
                                fmask, (4, 4, 4, 4), mode="replicate"
                            )
                        fmask = fmask + homography_mask
                    else:
                        fmask = torch.from_numpy(homography_mask)[None, None].to(device)
                    fclip = fclip[:, :, min_y:max_y, min_x:max_x]
                    fref = fref[:, :, min_y:max_y, min_x:max_x]
                    if fmask is not None:  # type: ignore
                        fmask = fmask[:, :, min_y:max_y, min_x:max_x]
        _, _, fref_h_new, fref_w_new = fref.shape
        if precision in (1, 2):
            p1 = calculate_padding(fref_h_new, fref_w_new, 64)
        if precision > 1:
            p2 = calculate_padding(fref_h_new, fref_w_new, 32)
        if precision > 2:
            p3 = calculate_padding(fref_h_new, fref_w_new, 16)
        if precision > 3:
            p4 = calculate_padding(fref_h_new, fref_w_new, 8)
        with torch.amp.autocast(device.type, enabled=fp16):
            if precision == 1:
                fclip = rife_align(
                    fclip,
                    fref,
                    fmask if fmask is not None else None,
                    s1,
                    p1,
                    blur=blur,
                    smooth=smooth * 3 if smooth > 0 else 11,
                    compensate=True,
                    device=device,
                    fp16=fp16,
                )
            elif precision == 2:
                fclip = rife_align(
                    fclip,
                    fref,
                    fmask if fmask is not None else None,
                    s1,
                    p1,
                    blur=9,
                    smooth=91,
                    compensate=False,
                    device=device,
                    fp16=fp16,
                )
                fclip = rife_align(
                    fclip,
                    fref,
                    fmask if fmask is not None else None,
                    s2,
                    p2,
                    blur=blur,
                    smooth=smooth,
                    compensate=True,
                    device=device,
                    fp16=fp16,
                )
            elif precision == 3:
                fclip = rife_align(
                    fclip,
                    fref,
                    fmask if fmask is not None else None,
                    s2,
                    p2,
                    blur=9,
                    smooth=15,
                    compensate=False,
                    device=device,
                    fp16=fp16,
                )
                fclip = rife_align(
                    fclip,
                    fref,
                    fmask if fmask is not None else None,
                    s3,
                    p3,
                    blur=blur,
                    smooth=smooth,
                    compensate=True,
                    device=device,
                    fp16=fp16,
                )
            elif precision == 4:
                fclip = rife_align(
                    fclip,
                    fref,
                    fmask if fmask is not None else None,
                    s2,
                    p2,
                    blur=9,
                    smooth=15,
                    compensate=False,
                    device=device,
                    fp16=fp16,
                )
                fclip = rife_align(
                    fclip,
                    fref,
                    fmask if fmask is not None else None,
                    s3,
                    p3,
                    blur=2,
                    smooth=7,
                    compensate=False,
                    device=device,
                    fp16=fp16,
                )
                fclip = rife_align(
                    fclip,
                    fref,
                    fmask if fmask is not None else None,
                    s4,
                    p4,
                    blur=blur,
                    smooth=smooth,
                    compensate=True,
                    device=device,
                    fp16=fp16,
                )
        if wide_search:
            if points_found and match_found:
                fclip = functional.pad(
                    fclip,
                    (min_x, fref_w - max_x, min_y, fref_h - max_y),  # type: ignore
                    mode="replicate",
                )
            fclip = fclip[:, :, 4:-4, 4:-4]
    return tensor2np(fclip.squeeze(0).cpu(), change_range=False, imtype=np.float32)
