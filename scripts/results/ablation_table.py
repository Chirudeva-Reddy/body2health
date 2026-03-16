"""
Build a compact ablation table similar to common silhouette/anthro papers.

This script evaluates a trained checkpoint under different inference settings:
  - Single-View: use front only (front duplicated as side)
  - Multi-View: use front + side
  - Multi-View + Height: fit a linear correction using height_cm
  - Multi-View + Weight: fit a linear correction using weight_kg
  - Multi-View + Height + Weight: fit a linear correction using both

Important: the "+ Height/Weight" rows are POST-HOC regressions fitted on the
train split of the same CSV for fast ablations; they do not change the neural
model architecture/training.

Outputs:
  - Prints a Markdown table to stdout.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.append(str(ROOT))

from src.model.contrastive_dualview import DualViewContrastive
from src.train.data import SilhouetteDataset


@dataclass
class Metrics:
    tp: Dict[float, float]
    mae: Dict[str, float]


def _parse_list(s: str) -> List[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def _parse_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in (s or "").split(",") if x.strip()]


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _predict_dataset(
    ckpt_path: str,
    csv_path: str,
    measurement_cols: List[str],
    mode: str,
    batch_size: int,
    device: str,
    max_rows: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      df_meta: dataframe with height_cm/weight_kg and derived gt columns
      y_true: (N, D) measurement targets in native units (BMI computed from hw if requested)
      y_pred: (N, D) model predictions for those measurements
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_out_meas = int(ckpt.get("meas_mean", torch.zeros(1)).shape[0])
    if ckpt_out_meas != len(measurement_cols):
        raise SystemExit(
            f"Checkpoint expects out_meas={ckpt_out_meas} but --measurement_cols has {len(measurement_cols)}.\n"
            f"Retrain with: PYTHONPATH=. python scripts/train_contrastive_640x480.py --csv \"{csv_path}\" "
            f"--measurement_cols \"{','.join(measurement_cols)}\" ..."
        )
    model = DualViewContrastive(
        out_meas=len(measurement_cols),
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
        convit_img_hw=tuple(ckpt.get("input_size", (640, 480))),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = SilhouetteDataset(csv_path, target_hw=(640, 480), measurement_cols=measurement_cols, augment=False)
    if max_rows is not None:
        n = min(len(ds), int(max_rows))
        ds = Subset(ds, list(range(n)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Load CSV metadata (height/weight) without pandas dependency
    heights: List[float] = []
    weights: List[float] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows is not None and i >= int(max_rows):
                break
            try:
                heights.append(float(row.get("height_cm", 0.0) or 0.0))
            except Exception:
                heights.append(0.0)
            try:
                weights.append(float(row.get("weight_kg", 0.0) or 0.0))
            except Exception:
                weights.append(0.0)
    height_cm = np.asarray(heights, dtype=np.float32)
    weight_kg = np.asarray(weights, dtype=np.float32)

    preds: List[np.ndarray] = []
    gts: List[np.ndarray] = []
    bf_preds: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            front = batch["front"].to(device)
            side = batch["side"].to(device)
            if mode == "single":
                side = front
            out = model(front, side)
            preds.append(_to_numpy(out["meas"]))
            gts.append(_to_numpy(batch["y_meas"]))
            bf_preds.append(_to_numpy(out["bf"]))  # fraction in [0,1]

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(gts, axis=0)
    bf_pred = np.concatenate(bf_preds, axis=0).reshape(-1)
    if y_true.shape[0] != height_cm.shape[0] or y_true.shape[0] != weight_kg.shape[0]:
        raise SystemExit(f"CSV rows mismatch: dataset={y_true.shape[0]} csv_height={height_cm.shape[0]} csv_weight={weight_kg.shape[0]}")
    return height_cm, weight_kg, y_true, y_pred, bf_pred


def _tp_rates(y_true: np.ndarray, y_pred: np.ndarray, thresholds: List[float]) -> Dict[float, float]:
    err = np.abs(y_true - y_pred)
    tp = {}
    for t in thresholds:
        tp[t] = float((err <= t).mean() * 100.0)
    return tp


def _mae_cols(y_true: np.ndarray, y_pred: np.ndarray, cols: List[str]) -> Dict[str, float]:
    err = np.abs(y_true - y_pred)
    out: Dict[str, float] = {}
    for i, c in enumerate(cols):
        out[c] = float(err[:, i].mean())
    return out


def _fit_linear(train_X: np.ndarray, train_y: np.ndarray) -> np.ndarray:
    """
    OLS with intercept: returns beta of shape (F+1,)
    """
    X = np.concatenate([np.ones((train_X.shape[0], 1), dtype=np.float64), train_X.astype(np.float64)], axis=1)
    y = train_y.astype(np.float64)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def _apply_linear(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    X1 = np.concatenate([np.ones((X.shape[0], 1), dtype=np.float64), X.astype(np.float64)], axis=1)
    return (X1 @ beta).astype(np.float32)


def _metrics_for_setting(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cols: List[str],
    tp_thresholds: List[float],
    tp_on_idx: int,
) -> Metrics:
    tp = _tp_rates(y_true[:, tp_on_idx], y_pred[:, tp_on_idx], tp_thresholds)
    mae = _mae_cols(y_true, y_pred, cols)
    return Metrics(tp=tp, mae=mae)


def _format_md_table(rows: List[Tuple[str, Metrics]], tp_thresholds: List[float], meas_cols: List[str]) -> str:
    thr_sorted = list(tp_thresholds)
    # Keep user's order (often TP90, TP75, TP50)
    headers = ["Setting"] + [f"TP{int(t)}" for t in thr_sorted] + [f"{c} MAE" for c in meas_cols]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for name, m in rows:
        parts = [name]
        for t in thr_sorted:
            parts.append(f"{m.tp[t]:.2f}")
        for c in meas_cols:
            parts.append(f"{m.mae[c]:.3f}")
        lines.append("| " + " | ".join(parts) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Processed training CSV (front_path/side_path/height_cm/weight_kg/...).")
    ap.add_argument("--ckpt", required=True, help="Model checkpoint path.")
    ap.add_argument("--measurement_cols", default="bmi", help="Comma-separated measurement columns (e.g., bmi,hip_cm).")
    # Note: BF% ablations were removed from the default table output; this script focuses on `meas` targets.
    ap.add_argument("--tp_thresholds", default="90,75,50", help="Comma-separated absolute-error thresholds for TP (percent).")
    ap.add_argument("--tp_on", default=None, help="Which measurement column to compute TP on (defaults to first).")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--max_rows", type=int, default=None, help="Limit evaluation to first N rows (for quick ablations).")
    args = ap.parse_args()

    meas_cols = _parse_list(args.measurement_cols)
    if not meas_cols:
        raise SystemExit("Provide --measurement_cols (e.g., bmi)")
    tp_thresholds = _parse_floats(args.tp_thresholds)
    if not tp_thresholds:
        raise SystemExit("Provide --tp_thresholds (e.g., 90,75,50)")

    tp_on = (args.tp_on or meas_cols[0]).strip()
    if tp_on not in meas_cols:
        raise SystemExit(f"--tp_on must be one of {meas_cols}, got {tp_on!r}")
    tp_on_idx = meas_cols.index(tp_on)

    # Base predictions
    height_cm, weight_kg, y_true_single, y_pred_single, bf_pred_single = _predict_dataset(
        args.ckpt, args.csv, meas_cols, mode="single", batch_size=args.batch_size, device=args.device, max_rows=args.max_rows
    )
    height_cm2, weight_kg2, y_true_multi, y_pred_multi, bf_pred_multi = _predict_dataset(
        args.ckpt, args.csv, meas_cols, mode="multi", batch_size=args.batch_size, device=args.device, max_rows=args.max_rows
    )
    if not np.allclose(y_true_single, y_true_multi):
        raise SystemExit("Internal error: targets mismatch between runs.")
    if not (np.allclose(height_cm, height_cm2) and np.allclose(weight_kg, weight_kg2)):
        raise SystemExit("Internal error: metadata mismatch between runs.")

    N = y_true_multi.shape[0]
    rng = np.random.default_rng(args.seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_train = max(1, int(round(args.train_ratio * N)))
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]
    if test_idx.size == 0:
        raise SystemExit("train_ratio too high; no test samples left.")

    def eval_row(name: str, y_pred: np.ndarray) -> Tuple[str, Metrics]:
        m = _metrics_for_setting(y_true_multi[test_idx], y_pred[test_idx], meas_cols, tp_thresholds, tp_on_idx)
        return name, m

    rows: List[Tuple[str, Metrics]] = []

    def eval_row_with_bf(name: str, y_pred: np.ndarray, bf_pred_frac: np.ndarray) -> Tuple[str, Metrics]:
        m = _metrics_for_setting(y_true_multi[test_idx], y_pred[test_idx], meas_cols, tp_thresholds, tp_on_idx)
        return name, m

    rows.append(eval_row_with_bf("Single-View", y_pred_single, bf_pred_single))
    rows.append(eval_row_with_bf("Multi-View", y_pred_multi, bf_pred_multi))

    # Post-hoc linear corrections per measurement dim (fit on train split, evaluate on test split)
    height = height_cm.reshape(-1, 1)
    weight = weight_kg.reshape(-1, 1)

    def corrected(pred_base: np.ndarray, extra: np.ndarray) -> np.ndarray:
        out = np.zeros_like(pred_base, dtype=np.float32)
        for d in range(pred_base.shape[1]):
            X_train = np.concatenate([pred_base[train_idx, d : d + 1], extra[train_idx]], axis=1)
            y_train = y_true_multi[train_idx, d]
            beta = _fit_linear(X_train, y_train)
            X_all = np.concatenate([pred_base[:, d : d + 1], extra], axis=1)
            out[:, d] = _apply_linear(beta, X_all)
        return out

    rows.append(eval_row_with_bf("Multi-View + Height", corrected(y_pred_multi, height), bf_pred_multi))
    rows.append(eval_row_with_bf("Multi-View + Weight", corrected(y_pred_multi, weight), bf_pred_multi))
    rows.append(eval_row_with_bf("Multi-View + Height + Weight", corrected(y_pred_multi, np.concatenate([height, weight], axis=1)), bf_pred_multi))

    print(_format_md_table(rows, tp_thresholds, list(meas_cols)))


if __name__ == "__main__":
    main()
