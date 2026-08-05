"""Pure prediction + sanity clipping. No file I/O, no prints."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
import torch

from src.metrics.body_indices import derive_indices, derive_risk_categories
from src.metrics.health_risk import assess_health_risk
from src.model.contrastive_dualview import DualViewContrastive


@dataclass
class PredictionResult:
    measurements: Dict[str, float]
    indices: Dict[str, float]
    risks: Dict[str, str]
    health_summary: Dict[str, object]
    invalid_indices: Dict[str, str] = field(default_factory=dict)
    raw: Dict[str, np.ndarray] = field(default_factory=dict)


def predict_from_pair(
    model: DualViewContrastive,
    ckpt: Dict[str, Any],
    front_t: torch.Tensor,
    side_t: torch.Tensor,
    height_cm: float | None,
    sex: str | None,
) -> PredictionResult:
    """Run a single forward pass and return named dimensions plus indices."""
    if "meas_mean" not in ckpt:
        raise KeyError("checkpoint dict is missing 'meas_mean'; load via load_dualview_checkpoint")
    if "measurement_cols" not in ckpt:
        raise KeyError("checkpoint dict is missing 'measurement_cols'; retrain with the dimension-first trainer")

    with torch.no_grad():
        out = model(front_t, side_t)

    meas = out["meas"][0].detach().cpu().numpy().astype(np.float64)
    measurement_cols = [str(col) for col in ckpt["measurement_cols"]]
    if len(measurement_cols) != len(meas):
        raise ValueError(
            f"checkpoint measurement_cols length {len(measurement_cols)} "
            f"does not match model output length {len(meas)}"
        )
    measurements = {
        col: float(meas[idx])
        for idx, col in enumerate(measurement_cols)
    }

    try:
        indices = derive_indices(measurements, height_cm)
        risks = derive_risk_categories(measurements, indices, sex)
        health_summary = assess_health_risk(measurements, indices, risks, sex, True).to_dict()
        invalid_indices: Dict[str, str] = {}
    except ValueError as exc:
        indices = {}
        risks = {}
        health_summary = assess_health_risk(measurements, indices, risks, sex, False).to_dict()
        invalid_indices = {"all": str(exc)}

    return PredictionResult(
        measurements=measurements,
        indices=indices,
        risks=risks,
        health_summary=health_summary,
        invalid_indices=invalid_indices,
        raw={
            "meas": out["meas"].detach().cpu().numpy(),
            "f_z": out["f_z"].detach().cpu().numpy(),
            "s_z": out["s_z"].detach().cpu().numpy(),
        },
    )
