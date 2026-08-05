"""NLF (Neural Localizer Fields) feed-forward SMPL-X estimator.

Wraps the NLF TorchScript model. Given an RGB image, returns SMPL-X
betas, pose, and translation, plus the reconstructed mesh vertices in
metres using the smplx library. Y is flipped so the mesh lives in our
canonical Y-up frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchvision


DEFAULT_NLF_MODEL_PATH = "models/nlf/nlf_l_multi.torchscript"
NLF_MAX_EDGE_PX = 1280


@dataclass(frozen=True)
class NlfPrediction:
    pose_rotvecs: np.ndarray
    betas: np.ndarray
    translation: np.ndarray
    vertices_m: np.ndarray
    faces: np.ndarray


_NLF_MODELS: dict[str, torch.jit.ScriptModule] = {}
_SMPLX_BODY_MODELS: dict[str, torch.nn.Module] = {}


def predict_smplx_from_rgb(
    rgb_image: np.ndarray,
    smplx_model_path: str,
    nlf_model_path: str,
) -> NlfPrediction:
    _validate_rgb_image(rgb_image)
    _validate_path(nlf_model_path, "missing_nlf_model")
    _validate_path(smplx_model_path, "missing_smplx_model")
    image_tensor = _to_image_tensor(rgb_image)
    image_batch = _downscale_to_max_edge(image_tensor, NLF_MAX_EDGE_PX).unsqueeze(0)
    nlf_model = _load_nlf_model(nlf_model_path)
    with torch.inference_mode():
        prediction = nlf_model.detect_smpl_batched(image_batch, model_name="smplx")
    if len(prediction["pose"][0]) == 0:
        raise ValueError("nlf_no_person: NLF found no person in the RGB image")
    pose = prediction["pose"][0][0].reshape(1, 165)
    betas = prediction["betas"][0][0].reshape(1, 10)
    translation = prediction["trans"][0][0].reshape(1, 3)
    smplx_body = _load_smplx_body_model(smplx_model_path)
    with torch.no_grad():
        smplx_output = smplx_body(
            global_orient=pose[:, :3],
            body_pose=pose[:, 3:22 * 3],
            jaw_pose=pose[:, 22 * 3:23 * 3],
            leye_pose=pose[:, 23 * 3:24 * 3],
            reye_pose=pose[:, 24 * 3:25 * 3],
            left_hand_pose=pose[:, 25 * 3:40 * 3],
            right_hand_pose=pose[:, 40 * 3:55 * 3],
            betas=betas,
            transl=translation,
        )
    vertices_m = smplx_output.vertices[0].detach().cpu().numpy().astype(np.float64)
    # NLF returns mesh in camera frame with Y axis pointing down (computer-vision
    # convention). Our render code expects Y up (world convention), so flip.
    vertices_m[:, 1] *= -1.0
    faces = smplx_body.faces.astype(np.int32) if not isinstance(smplx_body.faces, np.ndarray) else smplx_body.faces.astype(np.int32)
    return NlfPrediction(
        pose_rotvecs=pose[0].detach().cpu().numpy().astype(np.float64),
        betas=betas[0].detach().cpu().numpy().astype(np.float64),
        translation=translation[0].detach().cpu().numpy().astype(np.float64),
        vertices_m=vertices_m,
        faces=faces,
    )


def _validate_rgb_image(rgb_image: np.ndarray) -> None:
    if rgb_image.size == 0:
        raise ValueError("rgb_image is empty")
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError(f"rgb_image must be (H, W, 3), got {rgb_image.shape!r}")


def _validate_path(path: str, error_code: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"{error_code}: required file not found at {path}")


def _to_image_tensor(rgb_image: np.ndarray) -> torch.Tensor:
    if rgb_image.dtype != np.uint8:
        if rgb_image.max() <= 1.0:
            rgb_image = (rgb_image * 255.0).astype(np.uint8)
        else:
            rgb_image = rgb_image.astype(np.uint8)
    return torch.from_numpy(rgb_image).permute(2, 0, 1).contiguous()


def _downscale_to_max_edge(image_tensor: torch.Tensor, max_edge: int) -> torch.Tensor:
    height, width = int(image_tensor.shape[1]), int(image_tensor.shape[2])
    long_edge = max(height, width)
    if long_edge <= max_edge:
        return image_tensor
    scale = float(max_edge) / float(long_edge)
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))
    return torchvision.transforms.functional.resize(
        image_tensor, [new_height, new_width], antialias=True
    )


def _load_nlf_model(nlf_model_path: str) -> torch.jit.ScriptModule:
    if nlf_model_path not in _NLF_MODELS:
        _NLF_MODELS[nlf_model_path] = torch.jit.load(nlf_model_path, map_location="cpu").eval()
    return _NLF_MODELS[nlf_model_path]


def _load_smplx_body_model(smplx_model_path: str) -> torch.nn.Module:
    if smplx_model_path not in _SMPLX_BODY_MODELS:
        try:
            import smplx
        except ImportError as exc:
            raise RuntimeError(
                "missing_smplx: install smplx[all] into the project .venv before NLF inference"
            ) from exc
        _SMPLX_BODY_MODELS[smplx_model_path] = smplx.SMPLX(smplx_model_path, use_pca=False).eval()
    return _SMPLX_BODY_MODELS[smplx_model_path]
