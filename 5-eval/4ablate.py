"""
Build the dimension-first ablation table.

Rows:
  - Silhouette-only: front duplicated as side.
  - Dual-view: front + side silhouettes.
  - Dual-view + Height: post-hoc correction with height_cm.
  - Dual-view + Weight: post-hoc correction with weight_kg.
  - Dual-view + Height + Weight: post-hoc correction with both metadata fields.

The metadata rows are diagnostic leakage checks, not the canonical model.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.metrics.body_indices import bri, whr, whtr
from src.model.contrastive_dualview import DualViewContrastive
from src.train.data import SilhouetteDataset


@dataclass
class Metrics:
    tp: Dict[float, float]
    mae: Dict[str, float]
    rmse: Dict[str, float]
    index_mae: Dict[str, float]


def _parse_list(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _parse_floats(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _load_model(ckpt_path: str, measurement_cols: List[str], device: str) -> DualViewContrastive:
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_cols = [str(col) for col in ckpt.get("measurement_cols", [])]
    if ckpt_cols and ckpt_cols != measurement_cols:
        raise ValueError(f"checkpoint targets {ckpt_cols} do not match requested targets {measurement_cols}")
    ckpt_out_meas = int(ckpt.get("meas_mean", torch.zeros(len(measurement_cols))).shape[0])
    if ckpt_out_meas != len(measurement_cols):
        raise ValueError(f"checkpoint output width {ckpt_out_meas} does not match {len(measurement_cols)} targets")
    convit_hw = tuple(ckpt.get("input_size", (640, 480)))
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
        convit_img_hw=(int(convit_hw[0]), int(convit_hw[1])),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def _load_metadata(csv_path: str, max_rows: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    heights: List[float] = []
    weights: List[float] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            if max_rows is not None and row_idx >= max_rows:
                break
            heights.append(float(row["height_cm"]))
            weights.append(float(row["weight_kg"]))
    return np.asarray(heights, dtype=np.float32), np.asarray(weights, dtype=np.float32)


def _predict_dataset(
    ckpt_path: str,
    csv_path: str,
    measurement_cols: List[str],
    mode: str,
    batch_size: int,
    device: str,
    max_rows: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    model = _load_model(ckpt_path, measurement_cols, device)
    ds = SilhouetteDataset(csv_path, target_hw=(640, 480), measurement_cols=measurement_cols, augment=False)
    if max_rows is not None:
        ds = Subset(ds, list(range(min(len(ds), max_rows))))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"ablate-{mode}", leave=False, dynamic_ncols=True):
            front = batch["front"].to(device)
            side = batch["side"].to(device)
            if mode == "single":
                side = front
            out = model(front, side)
            preds.append(_to_numpy(out["meas"]))
            targets.append(_to_numpy(batch["y_meas"]))
    return np.concatenate(targets, axis=0), np.concatenate(preds, axis=0)


def _fit_linear(train_x: np.ndarray, train_y: np.ndarray) -> np.ndarray:
    x = np.concatenate([np.ones((train_x.shape[0], 1), dtype=np.float64), train_x.astype(np.float64)], axis=1)
    beta, *_ = np.linalg.lstsq(x, train_y.astype(np.float64), rcond=None)
    return beta


def _apply_linear(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    x1 = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x.astype(np.float64)], axis=1)
    return (x1 @ beta).astype(np.float32)


def _corrected(pred_base: np.ndarray, y_true: np.ndarray, extra: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    corrected = np.zeros_like(pred_base, dtype=np.float32)
    for dim_idx in range(pred_base.shape[1]):
        x_train = np.concatenate([pred_base[train_idx, dim_idx : dim_idx + 1], extra[train_idx]], axis=1)
        beta = _fit_linear(x_train, y_true[train_idx, dim_idx])
        x_all = np.concatenate([pred_base[:, dim_idx : dim_idx + 1], extra], axis=1)
        corrected[:, dim_idx] = _apply_linear(beta, x_all)
    return corrected


def _dimension_metrics(y_true: np.ndarray, y_pred: np.ndarray, cols: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    err = y_true - y_pred
    mae = {col: float(np.mean(np.abs(err[:, idx]))) for idx, col in enumerate(cols)}
    rmse = {col: float(np.sqrt(np.mean(err[:, idx] ** 2))) for idx, col in enumerate(cols)}
    return mae, rmse


def _index_values(values: np.ndarray, cols: List[str], heights: np.ndarray) -> Dict[str, np.ndarray]:
    col_idx = {col: idx for idx, col in enumerate(cols)}
    out: Dict[str, np.ndarray] = {}
    if "waist_cm" in col_idx and "hip_cm" in col_idx:
        out["WHR"] = np.asarray([
            whr(max(1e-6, float(row[col_idx["waist_cm"]])), max(1e-6, float(row[col_idx["hip_cm"]])))
            for row in values
        ])
    if "waist_cm" in col_idx:
        out["WHtR"] = np.asarray([
            whtr(max(1e-6, float(row[col_idx["waist_cm"]])), float(heights[idx]))
            for idx, row in enumerate(values)
        ])
        out["BRI"] = np.asarray([
            bri(max(1e-6, float(row[col_idx["waist_cm"]])), float(heights[idx]))
            for idx, row in enumerate(values)
        ])
    return out


def _metrics_for_setting(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cols: List[str],
    heights: np.ndarray,
    thresholds: List[float],
    tp_on_idx: int,
) -> Metrics:
    mae, rmse = _dimension_metrics(y_true, y_pred, cols)
    tp = {
        threshold: float((np.abs(y_true[:, tp_on_idx] - y_pred[:, tp_on_idx]) <= threshold).mean() * 100.0)
        for threshold in thresholds
    }
    true_indices = _index_values(y_true, cols, heights)
    pred_indices = _index_values(y_pred, cols, heights)
    index_mae = {
        name: float(np.mean(np.abs(true_indices[name] - pred_indices[name])))
        for name in sorted(true_indices.keys() & pred_indices.keys())
    }
    return Metrics(tp=tp, mae=mae, rmse=rmse, index_mae=index_mae)


def _format_table(rows: List[Tuple[str, Metrics]], thresholds: List[float], cols: List[str]) -> str:
    index_names = sorted({name for _, metric in rows for name in metric.index_mae.keys()})
    headers = ["Setting"] + [f"TP<= {threshold:g}cm" for threshold in thresholds]
    headers += [f"{col} MAE" for col in cols] + [f"{col} RMSE" for col in cols]
    headers += [f"{name} MAE" for name in index_names]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for name, metric in rows:
        parts = [name]
        parts += [f"{metric.tp[threshold]:.2f}" for threshold in thresholds]
        parts += [f"{metric.mae[col]:.3f}" for col in cols]
        parts += [f"{metric.rmse[col]:.3f}" for col in cols]
        parts += [f"{metric.index_mae[index_name]:.4f}" for index_name in index_names]
        lines.append("| " + " | ".join(parts) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/bodym/pairs_dimensions.csv")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--measurement_cols", default="waist_cm,hip_cm")
    parser.add_argument("--tp_thresholds", default="2,5,10")
    parser.add_argument("--tp_on", default="waist_cm")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--max_rows", type=int, default=None)
    args = parser.parse_args()

    cols = _parse_list(args.measurement_cols)
    if not cols:
        raise ValueError("--measurement_cols must include at least one target")
    if args.tp_on not in cols:
        raise ValueError(f"--tp_on must be one of {cols}, got {args.tp_on!r}")

    thresholds = _parse_floats(args.tp_thresholds)
    heights, weights = _load_metadata(args.csv, args.max_rows)
    y_true_single, y_pred_single = _predict_dataset(args.ckpt, args.csv, cols, "single", args.batch_size, args.device, args.max_rows)
    y_true_multi, y_pred_multi = _predict_dataset(args.ckpt, args.csv, cols, "multi", args.batch_size, args.device, args.max_rows)
    if not np.allclose(y_true_single, y_true_multi):
        raise ValueError("single-view and dual-view target arrays do not match")

    row_count = y_true_multi.shape[0]
    rng = np.random.default_rng(args.seed)
    indices = np.arange(row_count)
    rng.shuffle(indices)
    train_count = max(1, int(round(args.train_ratio * row_count)))
    train_idx = indices[:train_count]
    test_idx = indices[train_count:]
    if test_idx.size == 0:
        raise ValueError("train_ratio leaves no rows for evaluation")

    height_feature = heights.reshape(-1, 1)
    weight_feature = weights.reshape(-1, 1)
    tp_idx = cols.index(args.tp_on)

    settings = [
        ("Silhouette-only", y_pred_single),
        ("Dual-view", y_pred_multi),
        ("Dual-view + Height", _corrected(y_pred_multi, y_true_multi, height_feature, train_idx)),
        ("Dual-view + Weight", _corrected(y_pred_multi, y_true_multi, weight_feature, train_idx)),
        ("Dual-view + Height + Weight", _corrected(y_pred_multi, y_true_multi, np.concatenate([height_feature, weight_feature], axis=1), train_idx)),
    ]
    rows = [
        (
            name,
            _metrics_for_setting(
                y_true_multi[test_idx],
                pred[test_idx],
                cols,
                heights[test_idx],
                thresholds,
                tp_idx,
            ),
        )
        for name, pred in settings
    ]
    print(_format_table(rows, thresholds, cols))


if __name__ == "__main__":
    main()
