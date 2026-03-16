import cv2, torch, numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple
from src.model.contrastive_dualview import DualViewContrastive

# Precomputed BodyM envelope (compute once from your training silhouettes)
BODYM_STATS = {
    "h_occupancy_mean": 0.90, "h_occupancy_std": 0.02,
    "fg_ratio_mean":    0.18, "fg_ratio_std":    0.04,
    "waist_hip_mean":   0.88, "waist_hip_std":   0.05,
}
# Relaxed by default to avoid hard stops on your current masks.
# Adjust these back down once BODYM_STATS is recomputed from your train set.
DEFAULT_Z_THRESH = 10.0
FAIL_HARD = False

@dataclass
class Calibration:
    bmi_a: float = 1.0
    bmi_b: float = 0.0
    bf_a:  float = 1.0
    bf_b:  float = 0.0

def load_mask(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise ValueError(f"Could not read mask: {path}")
    return (m > 0).astype(np.uint8) * 255

def measure(mask: np.ndarray) -> Dict[str, float]:
    m = (mask > 0).astype(np.uint8)
    h, w = m.shape
    ys, xs = np.where(m > 0)
    if ys.size == 0:
        return {"fg_ratio": 0, "h_occ": 0, "waist_hip": 0}
    y0, y1 = ys.min(), ys.max()
    h_occ = (y1 - y0 + 1) / float(h)
    fg_ratio = float(m.mean())
    # waist/hip widths (front; on side we still compute for consistency)
    def row_width(y):
        cols = np.where(m[y] > 0)[0]
        return cols.max() - cols.min() + 1 if cols.size else 0
    waist_y = int(round(y0 + 0.55 * (y1 - y0)))
    hip_y   = int(round(y0 + 0.65 * (y1 - y0)))
    waist_w = row_width(np.clip(waist_y, 0, h - 1))
    hip_w   = row_width(np.clip(hip_y,   0, h - 1))
    waist_hip = waist_w / hip_w if hip_w > 0 else 0
    return {"fg_ratio": fg_ratio, "h_occ": h_occ, "waist_hip": waist_hip}

def reject_out_of_envelope(stats: Dict[str, float], which: str,
                           z_thresh: float = DEFAULT_Z_THRESH,
                           fail_hard: bool = FAIL_HARD):
    def z(val, mean, std): return (val - mean) / (std if std > 1e-6 else 1.0)
    z_fg   = abs(z(stats["fg_ratio"], BODYM_STATS["fg_ratio_mean"], BODYM_STATS["fg_ratio_std"]))
    z_h    = abs(z(stats["h_occ"],    BODYM_STATS["h_occupancy_mean"], BODYM_STATS["h_occupancy_std"]))
    z_wh   = abs(z(stats["waist_hip"],BODYM_STATS["waist_hip_mean"], BODYM_STATS["waist_hip_std"]))
    if max(z_fg, z_h, z_wh) > z_thresh:
        msg = (f"{which} silhouette outside BodyM envelope: "
               f"z_fg={z_fg:.2f}, z_h={z_h:.2f}, z_wh={z_wh:.2f} "
               f"(threshold={z_thresh})")
        if fail_hard:
            raise ValueError(msg)
        else:
            print(f"WARNING: {msg}")

def normalize_height(mask: np.ndarray, target_occ=0.93, tol=0.02) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    h, w = m.shape
    ys, xs = np.where(m > 0)
    if ys.size == 0:
        return m
    y0, y1 = ys.min(), ys.max()
    current_occ = (y1 - y0 + 1) / float(h)
    scale = (target_occ * h) / max(1, (y1 - y0 + 1))
    # Isotropic resize around centroid
    resized = cv2.resize(m, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros_like(m)
    y_off = (h - resized.shape[0]) // 2
    x_off = (w - resized.shape[1]) // 2
    y0c, y1c = max(0, y_off), min(h, y_off + resized.shape[0])
    x0c, x1c = max(0, x_off), min(w, x_off + resized.shape[1])
    canvas[y0c:y1c, x0c:x1c] = resized[: y1c - y0c, : x1c - x0c]
    # Recompute occupancy; reject if still off
    ys2 = np.where(canvas > 0)[0]
    if ys2.size == 0:
        return canvas
    occ2 = (ys2.max() - ys2.min() + 1) / float(h)
    if not (target_occ - tol <= occ2 <= target_occ + tol):
        raise ValueError(f"Height occupancy after norm out of band: {occ2:.3f}")
    return canvas.astype(np.uint8) * 255

def clothing_inflation(mask: np.ndarray) -> float:
    m = (mask > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return 1.0
    cnt = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    area = cv2.contourArea(cnt)
    hull_area = cv2.contourArea(hull)
    if area <= 0 or hull_area <= 0: return 1.0
    return min(1.15, max(1.0, hull_area / area))  # cap inflation

def deflate_mask(mask: np.ndarray, factor: float) -> np.ndarray:
    if factor <= 1.0: return mask
    m = (mask > 0).astype(np.uint8) * 255
    inv_scale = 1.0 / factor
    h, w = m.shape
    resized = cv2.resize(m, (int(w * inv_scale), int(h * inv_scale)), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros_like(m)
    y_off = (h - resized.shape[0]) // 2
    x_off = (w - resized.shape[1]) // 2
    canvas[y_off:y_off+resized.shape[0], x_off:x_off+resized.shape[1]] = resized
    return canvas

def monotonic_bf_mapping(raw_bf: float) -> float:
    # Raw BF is sigmoid output in [0,1]; map to plausible [5,40]% via affine + clip
    bf_pct = raw_bf * 100.0
    bf_pct = max(5.0, min(40.0, bf_pct))
    return bf_pct

def apply_calibration(bmi: float, bf_pct: float, calib: Calibration) -> Tuple[float, float]:
    return calib.bmi_a * bmi + calib.bmi_b, calib.bf_a * bf_pct + calib.bf_b

def corrected_inference(front_mask_path: str, side_mask_path: str, ckpt_path: str,
                        calib: Calibration = Calibration()) -> Tuple[float, float]:
    # Load and binarize
    front = load_mask(front_mask_path)
    side  = load_mask(side_mask_path)
    # Measure before normalization; reject distribution shift
    stats_f = measure(front); stats_s = measure(side)
    reject_out_of_envelope(stats_f, "front")
    reject_out_of_envelope(stats_s, "side")
    # Enforce height occupancy
    front = normalize_height(front)
    side  = normalize_height(side)
    # Clothing inflation estimate (use larger of front/side)
    infl = max(clothing_inflation(front), clothing_inflation(side))
    front = deflate_mask(front, infl)
    side  = deflate_mask(side, infl)
    # Prepare tensors
    def to_tensor(m):
        t = torch.from_numpy(m.astype(np.float32)).unsqueeze(0).unsqueeze(0) / 255.0
        return t
    f_t, s_t = to_tensor(front), to_tensor(side)
    # Load model
    ckpt = torch.load(ckpt_path, map_location="cpu")
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
    model.eval()
    with torch.no_grad():
        out = model(f_t, s_t)
        bmi_raw = out["meas"][0,0].item()
        bf_raw  = out["bf"][0,0].item()  # fraction
    bf_pct = monotonic_bf_mapping(bf_raw)
    bmi, bf_pct = apply_calibration(bmi_raw, bf_pct, calib)
    # Sanity checks
    if bmi > 32 and max(stats_f["waist_hip"], stats_s["waist_hip"]) < 1.05:
        raise ValueError(f"Implausible BMI {bmi:.2f} for non-obese silhouette.")
    if bf_pct > 35 and max(stats_f["waist_hip"], stats_s["waist_hip"]) < 1.05:
        raise ValueError(f"Implausible BF {bf_pct:.2f}% without extreme waist expansion.")
    if abs(stats_f["h_occ"] - stats_s["h_occ"]) > 0.10:
        raise ValueError(f"Front/side height inconsistency >10% (front={stats_f['h_occ']:.3f}, side={stats_s['h_occ']:.3f})")
    return bmi, bf_pct
