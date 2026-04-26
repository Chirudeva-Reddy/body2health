"""Pure prediction + sanity clipping. No file I/O, no prints."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import torch

from src.model.contrastive_dualview import DualViewContrastive

# Physical-plausibility clips. These guard against degenerate model outputs;
# they are not calibration. The training pipeline does not normalize targets,
# so the model's BMI head emits values in raw BMI units and the BF head emits
# a sigmoid in [0, 1] that we render as a percentage in [0, 100].
BMI_CLIP_LOW = 10.0
BMI_CLIP_HIGH = 80.0
BF_PCT_CLIP_LOW = 0.0
BF_PCT_CLIP_HIGH = 100.0


@dataclass
class PredictionResult:
    bmi: float
    bf_pct: float
    measurements: List[float]
    raw: Dict[str, np.ndarray] = field(default_factory=dict)


def predict_from_pair(
    model: DualViewContrastive,
    ckpt: Dict[str, Any],
    front_t: torch.Tensor,
    side_t: torch.Tensor,
) -> PredictionResult:
    """Run a single forward pass and clip outputs to plausible ranges.

    ``ckpt`` is the raw checkpoint dict returned by ``load_dualview_checkpoint``.
    It is accepted (and validated for ``meas_mean``) so callers cannot acquire
    a model without also having a checkpoint dict in scope; future per-target
    metadata can hang off it without changing the call signature.
    """
    if "meas_mean" not in ckpt:
        raise KeyError("checkpoint dict is missing 'meas_mean'; load via load_dualview_checkpoint")

    with torch.no_grad():
        out = model(front_t, side_t)

    meas = out["meas"][0].detach().cpu().numpy().astype(np.float64)
    bf_frac = float(out["bf"][0, 0].item())
    bmi = float(np.clip(meas[0], BMI_CLIP_LOW, BMI_CLIP_HIGH))
    bf_pct = float(np.clip(bf_frac * 100.0, BF_PCT_CLIP_LOW, BF_PCT_CLIP_HIGH))

    return PredictionResult(
        bmi=bmi,
        bf_pct=bf_pct,
        measurements=meas.tolist(),
        raw={
            "meas": out["meas"].detach().cpu().numpy(),
            "bf": out["bf"].detach().cpu().numpy(),
            "f_z": out["f_z"].detach().cpu().numpy(),
            "s_z": out["s_z"].detach().cpu().numpy(),
        },
    )
