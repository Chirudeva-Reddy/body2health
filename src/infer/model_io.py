"""Checkpoint loading + DualViewContrastive construction."""
from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import torch

from src.model.contrastive_dualview import DualViewContrastive

logger = logging.getLogger(__name__)


def load_dualview_checkpoint(ckpt_path: str, device: str) -> Tuple[DualViewContrastive, Dict[str, Any]]:
    """Load a DualViewContrastive checkpoint and return (model_in_eval_mode, raw_ckpt_dict).

    The model is moved to ``device`` and put in eval mode. The raw checkpoint
    dict is returned so callers can read ``meas_mean`` / ``meas_std`` / etc.
    if they need denormalization (the training pipeline does not normalize, so
    by default callers should not denormalize either).

    Required fields on the checkpoint: ``model``, ``meas_mean``, ``input_size``.
    Missing fields raise; this is intentional per project convention.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    for key in ("model", "meas_mean", "input_size"):
        if key not in ckpt:
            raise KeyError(f"checkpoint {ckpt_path!r} is missing required field {key!r}")

    convit_hw = tuple(ckpt["input_size"])
    model = DualViewContrastive(
        out_meas=ckpt["meas_mean"].shape[0],
        proj_dim=128,
        use_large=ckpt.get("use_large", False),
        base_dim=ckpt.get("base_dim", 80),
        use_bbox_features=ckpt.get("use_bbox_features", False),
        encoder=ckpt.get("encoder", "cnn"),
        convit_patch_size=ckpt.get("convit_patch_size", 16),
        convit_dim=ckpt.get("convit_dim", 256),
        convit_depth=ckpt.get("convit_depth", 6),
        convit_heads=ckpt.get("convit_heads", 4),
        convit_mlp_dim=ckpt.get("convit_mlp_dim", 512),
        convit_drop=ckpt.get("convit_drop", 0.0),
        convit_pool=ckpt.get("convit_pool", "mean"),
        convit_gpsa_layers=ckpt.get("convit_gpsa_layers", 2),
        convit_shared=ckpt.get("convit_shared", True),
        convit_img_hw=(int(convit_hw[0]), int(convit_hw[1])),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    logger.debug("loaded checkpoint %s on %s (input_size=%s)", ckpt_path, device, convit_hw)
    return model, ckpt
