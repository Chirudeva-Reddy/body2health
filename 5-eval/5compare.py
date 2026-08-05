"""
Compare two dimension-first checkpoints on waist/hip and derived indices.
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
class EvalOut:
    meas_mae: Dict[str, float]
    meas_rmse: Dict[str, float]
    meas_r2: Dict[str, float]
    meas_pearson_r: Dict[str, float]
    index_mae: Dict[str, float]


def _parse_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def _pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return 0.0
    true_std = float(np.std(y_true))
    pred_std = float(np.std(y_pred))
    if true_std <= 1e-12 or pred_std <= 1e-12:
        return 0.0
    return float(np.corrcoef(y_true.reshape(-1), y_pred.reshape(-1))[0, 1])


def _load_model(ckpt_path: str, measurement_cols: List[str], device: str) -> DualViewContrastive:
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_cols = [str(col) for col in ckpt.get("measurement_cols", [])]
    if ckpt_cols and ckpt_cols != measurement_cols:
        raise ValueError(f"checkpoint {ckpt_path!r} targets {ckpt_cols} do not match {measurement_cols}")
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


def _load_heights(csv_path: str, max_rows: Optional[int]) -> np.ndarray:
    heights: List[float] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            if max_rows is not None and row_idx >= max_rows:
                break
            heights.append(float(row["height_cm"]))
    return np.asarray(heights, dtype=np.float32)


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


def _eval_checkpoint(
    csv_path: str,
    ckpt_path: str,
    measurement_cols: List[str],
    mode: str,
    max_rows: Optional[int],
    batch_size: int,
    device: str,
    test_idx: List[int],
) -> EvalOut:
    full_ds = SilhouetteDataset(csv_path, target_hw=(640, 480), measurement_cols=measurement_cols, augment=False)
    if max_rows is not None:
        full_ds = Subset(full_ds, list(range(min(len(full_ds), max_rows))))
    test_ds = Subset(full_ds, test_idx)
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model = _load_model(ckpt_path, measurement_cols, device)

    true_batches: List[np.ndarray] = []
    pred_batches: List[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="compare", leave=False, dynamic_ncols=True):
            front = batch["front"].to(device)
            side = batch["side"].to(device)
            if mode == "single":
                side = front
            out = model(front, side)
            true_batches.append(_to_numpy(batch["y_meas"]))
            pred_batches.append(_to_numpy(out["meas"]))

    y_true = np.concatenate(true_batches, axis=0)
    y_pred = np.concatenate(pred_batches, axis=0)
    heights = _load_heights(csv_path, max_rows)[test_idx]
    true_indices = _index_values(y_true, measurement_cols, heights)
    pred_indices = _index_values(y_pred, measurement_cols, heights)

    return EvalOut(
        meas_mae={col: float(np.mean(np.abs(y_true[:, idx] - y_pred[:, idx]))) for idx, col in enumerate(measurement_cols)},
        meas_rmse={col: _rmse(y_true[:, idx], y_pred[:, idx]) for idx, col in enumerate(measurement_cols)},
        meas_r2={col: _r2(y_true[:, idx], y_pred[:, idx]) for idx, col in enumerate(measurement_cols)},
        meas_pearson_r={col: _pearson_r(y_true[:, idx], y_pred[:, idx]) for idx, col in enumerate(measurement_cols)},
        index_mae={
            name: float(np.mean(np.abs(true_indices[name] - pred_indices[name])))
            for name in sorted(true_indices.keys() & pred_indices.keys())
        },
    )


def _format_table(cols: List[str], rows: List[Tuple[str, EvalOut]]) -> str:
    index_names = sorted({name for _, row in rows for name in row.index_mae.keys()})
    headers = ["Setting"]
    for col in cols:
        headers += [f"{col} MAE", f"{col} RMSE", f"{col} R2", f"{col} r"]
    headers += [f"{name} MAE" for name in index_names]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for name, out in rows:
        parts = [name]
        for col in cols:
            parts += [
                f"{out.meas_mae[col]:.3f}",
                f"{out.meas_rmse[col]:.3f}",
                f"{out.meas_r2[col]:.3f}",
                f"{out.meas_pearson_r[col]:.3f}",
            ]
        parts += [f"{out.index_mae[index_name]:.4f}" for index_name in index_names]
        lines.append("| " + " | ".join(parts) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/bodym/pairs_dimensions.csv")
    parser.add_argument("--ckpt_contrastive", required=True)
    parser.add_argument("--ckpt_regonly", required=True)
    parser.add_argument("--measurement_cols", default="waist_cm,hip_cm")
    parser.add_argument("--mode", default="multi", choices=["multi", "single"])
    parser.add_argument("--max_rows", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    args = parser.parse_args()

    cols = _parse_list(args.measurement_cols)
    if not cols:
        raise ValueError("--measurement_cols must include at least one target")

    row_count = len(SilhouetteDataset(args.csv, target_hw=(640, 480), measurement_cols=cols, augment=False))
    if args.max_rows is not None:
        row_count = min(row_count, args.max_rows)
    rng = np.random.default_rng(args.seed)
    indices = np.arange(row_count)
    rng.shuffle(indices)
    test_count = max(1, int(round(args.test_ratio * row_count)))
    test_idx = indices[:test_count].tolist()

    rows = [
        ("Contrastive", _eval_checkpoint(args.csv, args.ckpt_contrastive, cols, args.mode, args.max_rows, args.batch_size, args.device, test_idx)),
        ("No-Contrastive", _eval_checkpoint(args.csv, args.ckpt_regonly, cols, args.mode, args.max_rows, args.batch_size, args.device, test_idx)),
    ]
    print(_format_table(cols, rows))


if __name__ == "__main__":
    main()
