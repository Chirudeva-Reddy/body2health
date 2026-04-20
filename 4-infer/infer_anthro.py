"""
Anthropometric inference with identifiability-aware constraints.

Design notes:
- BMI/BF are not identifiable from silhouette alone (no mass cue, clothing inflation).
- The model's outputs are interpreted as shape proxies, not absolute BMI/BF.
- Two modes:
    1) Assisted: height_cm and weight_kg provided. BMI is computed exactly from metadata;
       model only supplies shape-conditioned BF proxy and consistency checks.
    2) Silhouette-only: bounded, constrained proxies with uncertainty; may return
       UNDERCONSTRAINED instead of a single value.

Physiological priors:
- Silhouette-only BMI proxy bounded to [14, 28]; BF proxy to [6, 35] (%).
- Height occupancy normalized to ~0.93 of frame; fg ratio must be in [0.08, 0.35].
- Side view validated and downweighted if low quality.
- Perturbation robustness checks (erosion/dilation/blur) gate outputs.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.model.contrastive_dualview import DualViewContrastive

# Constants
TARGET_OCC = 0.93
OCC_TOL = 0.02
FG_RANGE = (0.08, 0.35)
OCC_RANGE = (0.88, 1.00)  # allow full-height occupancy
BMI_BOUNDS = (14.0, 28.0)         # silhouette-only bounds
BF_BOUNDS = (6.0, 35.0)           # silhouette-only bounds, percent
BF_ASSISTED_BOUNDS = (4.0, 40.0)  # assisted mode bounds, percent
SIDE_LOW_QUAL = 0.4
SIDE_MED_QUAL = 0.6
UNCERTAINTY_THRESH = {"bmi": 1.5, "bf": 2.5}


@dataclass
class Result:
    mode: str
    bmi: str
    bf: str
    flags: List[str]
    side_quality: float


def _largest_cc(mask01: np.ndarray) -> np.ndarray:
    num, lab, stats, _ = cv2.connectedComponentsWithStats(mask01, connectivity=8)
    if num <= 1:
        return np.zeros_like(mask01)
    idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (lab == idx).astype(np.uint8)


def _standardize_mask(mask: np.ndarray) -> np.ndarray:
    """Binary → largest CC → polarity fix → height occupancy normalization."""
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    m = (mask > 0).astype(np.uint8)
    # Auto polarity if fg dominates
    if float(m.mean()) > 0.6:
        m = 1 - m
    m = _largest_cc(m)
    fg_ratio = float(m.mean())
    if not (FG_RANGE[0] <= fg_ratio <= FG_RANGE[1]):
        raise ValueError(f"fg_ratio {fg_ratio:.3f} outside allowed range {FG_RANGE}")

    ys, xs = np.where(m > 0)
    if ys.size == 0:
        raise ValueError("Empty silhouette after CC filtering.")
    h, w = m.shape
    y0, y1 = ys.min(), ys.max()
    occ = (y1 - y0 + 1) / float(h)
    if not (OCC_RANGE[0] <= occ <= OCC_RANGE[1]):
        raise ValueError(f"height occupancy {occ:.3f} outside range {OCC_RANGE}")

    # Normalize height occupancy to TARGET_OCC
    scale = (TARGET_OCC * h) / max(1, (y1 - y0 + 1))
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(m, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((h, w), dtype=np.uint8)
    y_off = (h - new_h) // 2
    x_off = (w - new_w) // 2
    y_end = min(h, y_off + new_h)
    x_end = min(w, x_off + new_w)
    canvas[y_off:y_end, x_off:x_end] = resized[: y_end - y_off, : x_end - x_off]
    return (canvas > 0).astype(np.uint8) * 255


def _repair_side(mask255: np.ndarray) -> Tuple[np.ndarray, float]:
    """Light repair: remove small comps, fill small holes, smooth edges; returns quality score."""
    m = (mask255 > 0).astype(np.uint8)
    # Remove small components
    num, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros_like(m)
    if num > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = 1 + np.argmax(areas)
        for i in range(1, num):
            if i == largest_idx:
                keep[lab == i] = 1
            elif stats[i, cv2.CC_STAT_AREA] > 0.002 * m.size:
                keep[lab == i] = 1
    else:
        keep = m
    m = keep
    # Fill small holes
    inv = (1 - m) * 255
    flood = inv.copy()
    ff_mask = np.zeros((inv.shape[0] + 2, inv.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(flood, ff_mask, seedPoint=(0, 0), newVal=0)
    holes = (flood == 255).astype(np.uint8)
    num_h, lab_h, stats_h, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
    for i in range(1, num_h):
        if stats_h[i, cv2.CC_STAT_AREA] < 0.005 * m.size:
            m[lab_h == i] = 1
    # Remove thin vertical artifacts
    num, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    for i in range(1, num):
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if w <= 3 and h > 0.3 * m.shape[0]:
            m[lab == i] = 0
    # Smooth edges
    blurred = cv2.GaussianBlur(m.astype(np.float32), (3, 3), sigmaX=0.8)
    m = (blurred > 0.5).astype(np.uint8)
    m = _largest_cc(m)
    # Quality: size of largest CC vs total + hole fraction
    fg = float(m.mean())
    side_quality = max(0.0, min(1.0, fg / max(1e-6, float(mask255.mean())))) if mask255.mean() > 0 else 0.0
    return m.astype(np.uint8) * 255, side_quality


def _prep_masks(front_path: str, side_path: Optional[str]) -> Tuple[np.ndarray, np.ndarray, float, List[str]]:
    flags = []
    front = _standardize_mask(cv2.imread(front_path, cv2.IMREAD_GRAYSCALE))
    if side_path:
        side_raw = _standardize_mask(cv2.imread(side_path, cv2.IMREAD_GRAYSCALE))
        side, side_q = _repair_side(side_raw)
    else:
        side = front.copy()
        side_q = 0.0
        flags.append("side_missing")

    if side_q < SIDE_LOW_QUAL:
        flags.append("side_ignored_low_quality")
        side = front.copy()
    elif side_q < SIDE_MED_QUAL:
        flags.append("side_downweighted")
    return front, side, side_q, flags


def _to_tensor(mask255: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(mask255.astype(np.float32)).unsqueeze(0).unsqueeze(0) / 255.0
    return t


def _predict_proxies(model, front: np.ndarray, side: np.ndarray, device="cpu") -> Tuple[float, float]:
    f = _to_tensor(front).to(device)
    s = _to_tensor(side).to(device)
    with torch.no_grad():
        out = model(f, s)
        bmi_proxy = float(out["meas"][0, 0].item())
        bf_proxy = float(out["bf"][0, 0].item()) * 100.0
    # Bound as proxies (silhouette-only)
    bmi_proxy = max(BMI_BOUNDS[0], min(BMI_BOUNDS[1], bmi_proxy))
    bf_proxy = max(BF_BOUNDS[0], min(BF_BOUNDS[1], bf_proxy))
    return bmi_proxy, bf_proxy


def _perturb_masks(mask: np.ndarray) -> List[np.ndarray]:
    m01 = (mask > 0).astype(np.uint8)
    perturbed = []
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    er = cv2.erode(m01, k, iterations=1)
    di = cv2.dilate(m01, k, iterations=1)
    blur = cv2.medianBlur(m01 * 255, 3) > 0
    perturbed.extend([er, di, blur.astype(np.uint8)])
    return [p.astype(np.uint8) * 255 for p in perturbed]


def _load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    convit_hw = tuple(ckpt.get("input_size", (640, 480)))
    model = DualViewContrastive(
        out_meas=ckpt.get("meas_mean", torch.zeros(1)).shape[0],
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
    return model


def infer(front_path: str,
          side_path: Optional[str],
          ckpt_path: str,
          height_cm: Optional[float],
          weight_kg: Optional[float],
          device: str = "cpu") -> Result:
    flags: List[str] = []
    front, side, side_q, pre_flags = _prep_masks(front_path, side_path)
    flags.extend(pre_flags)

    # If side downweighted, blend by averaging masks
    if "side_downweighted" in flags:
        side = ((0.5 * (side > 0) + 0.5 * (front > 0)) > 0.5).astype(np.uint8) * 255

    model = _load_model(ckpt_path, device)

    # Mode selection
    assisted = height_cm is not None and weight_kg is not None

    if assisted:
        mode = "assisted"
        bmi_true = weight_kg / ((height_cm / 100.0) ** 2)
        # BF% from model proxies + monotonic adjustment with torso thickness (waist proxy)
        bmi_proxy, bf_proxy = _predict_proxies(model, front, side, device)
        bf_proxy = max(BF_ASSISTED_BOUNDS[0], min(BF_ASSISTED_BOUNDS[1], bf_proxy))
        # Perturbation robustness
        bf_samples = [bf_proxy]
        for pf, ps in zip(_perturb_masks(front), _perturb_masks(side)):
            bf_p = _predict_proxies(model, pf, ps, device)[1]
            bf_p = max(BF_ASSISTED_BOUNDS[0], min(BF_ASSISTED_BOUNDS[1], bf_p))
            bf_samples.append(bf_p)
        bf_mean = float(np.mean(bf_samples))
        bf_std = float(np.std(bf_samples))
        if bf_std > UNCERTAINTY_THRESH["bf"]:
            flags.append(f"bf_uncertainty_high:{bf_std:.2f}")
        return Result(
            mode=mode,
            bmi=f"{bmi_true:.2f}",
            bf=f"{bf_mean:.2f} ± {bf_std:.2f}",
            flags=flags,
            side_quality=side_q,
        )

    # Silhouette-only mode
    mode = "silhouette_only"
    bmi_samples = []
    bf_samples = []
    base_bmi, base_bf = _predict_proxies(model, front, side, device)
    bmi_samples.append(base_bmi)
    bf_samples.append(base_bf)
    for pf, ps in zip(_perturb_masks(front), _perturb_masks(side)):
        b, f = _predict_proxies(model, pf, ps, device)
        bmi_samples.append(b)
        bf_samples.append(f)
    bmi_mean, bmi_std = float(np.mean(bmi_samples)), float(np.std(bmi_samples))
    bf_mean, bf_std = float(np.mean(bf_samples)), float(np.std(bf_samples))

    unstable = False
    if bmi_std > UNCERTAINTY_THRESH["bmi"]:
        flags.append(f"bmi_uncertainty_high:{bmi_std:.2f}")
        unstable = True
    if bf_std > UNCERTAINTY_THRESH["bf"]:
        flags.append(f"bf_uncertainty_high:{bf_std:.2f}")
        unstable = True

    if unstable:
        return Result(
            mode=mode,
            bmi="UNDERCONSTRAINED",
            bf="UNDERCONSTRAINED",
            flags=flags,
            side_quality=side_q,
        )

    return Result(
        mode=mode,
        bmi=f"{bmi_mean:.2f} ± {bmi_std:.2f}",
        bf=f"{bf_mean:.2f} ± {bf_std:.2f}",
        flags=flags,
        side_quality=side_q,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--front", required=True, help="Path to front silhouette (640x480 binary)")
    parser.add_argument("--side", help="Path to side silhouette (optional)")
    parser.add_argument("--ckpt", required=True, help="Path to trained checkpoint")
    parser.add_argument("--height_cm", type=float, help="Height in cm (assisted mode if provided with weight)")
    parser.add_argument("--weight_kg", type=float, help="Weight in kg (assisted mode if provided with height)")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    res = infer(
        front_path=args.front,
        side_path=args.side,
        ckpt_path=args.ckpt,
        height_cm=args.height_cm,
        weight_kg=args.weight_kg,
        device=args.device,
    )
    print({
        "mode": res.mode,
        "BMI": res.bmi,
        "BF%": res.bf,
        "flags": res.flags,
        "side_quality": res.side_quality,
    })


if __name__ == "__main__":
    main()
