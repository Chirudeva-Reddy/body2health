import torch
import torch.nn.functional as F
from typing import Dict


def info_nce_loss(f_z: torch.Tensor, s_z: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """
    Symmetric InfoNCE between front and side embeddings.
    f_z, s_z: (B, D) L2-normalized.
    """
    B = f_z.size(0)
    logits_fs = (f_z @ s_z.t()) / tau  # (B,B)
    logits_sf = (s_z @ f_z.t()) / tau
    targets = torch.arange(B, device=f_z.device)
    loss_fs = F.cross_entropy(logits_fs, targets)
    loss_sf = F.cross_entropy(logits_sf, targets)
    return 0.5 * (loss_fs + loss_sf)


def regression_losses(out: Dict[str, torch.Tensor],
                      y_meas: torch.Tensor,
                      y_bf: torch.Tensor,
                      meas_weight: float = 1.0,
                      bf_weight: float = 1.0) -> torch.Tensor:
    l_meas = F.l1_loss(out["meas"], y_meas)
    l_bf = F.l1_loss(out["bf"], y_bf)
    return meas_weight * l_meas + bf_weight * l_bf


def total_loss(out: Dict[str, torch.Tensor],
               y_meas: torch.Tensor,
               y_bf: torch.Tensor,
               lambda_reg: float = 0.1,
               tau: float = 0.1,
               use_contrastive: bool = True) -> Dict[str, torch.Tensor]:
    """
    Returns dict with contrastive, reg, total.
    """
    l_con = info_nce_loss(out["f_z"], out["s_z"], tau=tau) if use_contrastive else torch.zeros((), device=y_meas.device)
    l_reg = regression_losses(out, y_meas, y_bf)
    return {"contrastive": l_con, "regression": l_reg, "total": l_con + lambda_reg * l_reg}
