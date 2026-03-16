import torch
import torch.nn as nn
from typing import Dict, Tuple


def _bbox_features(mask: torch.Tensor) -> torch.Tensor:
    """
    mask: (B,1,H,W) float/bool
    Returns (B,4): normalized height, width, area ratio, aspect (w/h).
    """
    m = (mask > 0).float().squeeze(1)
    B, H, W = m.shape
    ys = torch.arange(H, device=m.device).view(1, H, 1)
    xs = torch.arange(W, device=m.device).view(1, 1, W)

    y_any = (m.sum(dim=2) > 0).float()
    x_any = (m.sum(dim=1) > 0).float()

    def _extent(v_any, axis_len):
        idx = torch.arange(axis_len, device=m.device).view(1, -1)
        has = v_any > 0
        if has.sum(dim=1).min() == 0:
            # empty mask fallback
            return torch.zeros(has.shape[0], device=m.device)
        min_idx = (idx * has).masked_fill(~has, axis_len).min(dim=1).values
        max_idx = (idx * has).max(dim=1).values
        return (max_idx - min_idx + 1).float()

    h = _extent(y_any, H) / float(H)
    w = _extent(x_any, W) / float(W)
    area = m.sum(dim=(1, 2)) / float(H * W)
    aspect = w / (h + 1e-6)
    return torch.stack([h, w, area, aspect], dim=1)


class WidthHeightBaseline(nn.Module):
    """
    Simple baseline using coarse shape features from front/side masks.
    """
    def __init__(self, out_meas: int = 14):
        super().__init__()
        self.fc_meas = nn.Linear(8, out_meas)
        self.fc_bf = nn.Sequential(nn.Linear(8, 1), nn.Sigmoid())

    def forward(self, front_mask: torch.Tensor, side_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        f_feat = _bbox_features(front_mask)  # (B,4)
        s_feat = _bbox_features(side_mask)   # (B,4)
        feat = torch.cat([f_feat, s_feat], dim=1)
        meas = self.fc_meas(feat)
        bf = self.fc_bf(feat)
        return {"meas": meas, "bf": bf}


def baseline_features(front_mask: torch.Tensor, side_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Expose raw feature vectors for analysis or external regressors."""
    return _bbox_features(front_mask), _bbox_features(side_mask)
