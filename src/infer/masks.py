"""Mask I/O: load → resize → binarize → tensor.

Matches the training-time loader in ``src/train/data.py``: ``INTER_AREA`` for
downscale, ``INTER_NEAREST`` for upscale, threshold 127 to binarize.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import torch


def load_mask_binary(path: str, target_hw: Tuple[int, int]) -> np.ndarray:
    """Read a grayscale mask, resize to ``target_hw`` (H, W), threshold to {0, 255}.

    Raises ``FileNotFoundError`` if the path does not resolve to a readable image.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"could not read mask: {path}")
    h, w = int(target_hw[0]), int(target_hw[1])
    if img.shape != (h, w):
        interp = cv2.INTER_AREA if (img.shape[0] > h or img.shape[1] > w) else cv2.INTER_NEAREST
        img = cv2.resize(img, (w, h), interpolation=interp)
    return ((img > 127).astype(np.uint8)) * 255


def mask_to_tensor(mask: np.ndarray, device: str) -> torch.Tensor:
    """Convert a uint8 mask of shape (H, W) to a (1, 1, H, W) float tensor in [0, 1]."""
    if mask.ndim != 2:
        raise ValueError(f"expected 2-D mask, got shape {mask.shape!r}")
    t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0) / 255.0
    return t.to(device)
