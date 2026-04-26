"""
Quick, low-compute comparison table: contrastive vs regression-only checkpoints.

Outputs a Markdown table with:
  - Standard regression metrics for BMI (or other measurement_cols) and BF%
    * MAE, RMSE, R^2, Pearson r

Designed to run on CPU with --max_rows for minimal stress.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.append(str(ROOT))

from src.model.contrastive_dualview import DualViewContrastive
from src.train.data import SilhouetteDataset


def _parse_list(s: str) -> List[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def _parse_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in (s or "").split(",") if x.strip()]


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


@dataclass
class EvalOut:
    meas_mae: List[float]
    meas_rmse: List[float]
    meas_r2: List[float]
    meas_pearson_r: List[float]
    bf_mae_pct: float
    bf_rmse_pct: float
    bf_r2: float
    bf_pearson_r: float


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def _pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.size < 2:
        return 0.0
    t_std = float(np.std(y_true))
    p_std = float(np.std(y_pred))
    if t_std <= 1e-12 or p_std <= 1e-12:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _load_model(ckpt_path: str, out_meas: int, device: str) -> DualViewContrastive:
    ckpt = torch.load(ckpt_path, map_location=device)
    convit_hw = tuple(ckpt.get("input_size", (640, 480)))
    model = DualViewContrastive(
        out_meas=out_meas,
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


def _format_md_table(
    measurement_cols: List[str],
    rows: List[Tuple[str, EvalOut]],
) -> str:
    headers = ["Setting"]
    for c in measurement_cols:
        headers += [f"{c} MAE↓", f"{c} RMSE↓", f"{c} R²↑", f"{c} r↑"]
    headers += ["BF% MAE↓", "BF% RMSE↓", "BF% R²↑", "BF% r↑"]

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for name, out in rows:
        parts: List[str] = [name]
        for i in range(len(measurement_cols)):
            parts += [
                f"{out.meas_mae[i]:.3f}",
                f"{out.meas_rmse[i]:.3f}",
                f"{out.meas_r2[i]:.3f}",
                f"{out.meas_pearson_r[i]:.3f}",
            ]
        parts += [f"{out.bf_mae_pct:.3f}", f"{out.bf_rmse_pct:.3f}", f"{out.bf_r2:.3f}", f"{out.bf_pearson_r:.3f}"]
        lines.append("| " + " | ".join(parts) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ckpt_contrastive", required=True)
    ap.add_argument("--ckpt_regonly", required=True)
    ap.add_argument("--measurement_cols", default="bmi", help="Comma-separated (e.g., bmi,bai,hip_cm).")
    ap.add_argument("--mode", default="multi", choices=["multi", "single"], help="Evaluate multi-view or single-view.")
    ap.add_argument("--max_rows", type=int, default=150, help="Limit rows for cheap eval.")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42, help="Seed for deterministic test split.")
    ap.add_argument("--test_ratio", type=float, default=0.2, help="Fraction of rows used for test metrics.")
    args = ap.parse_args()

    meas_cols = _parse_list(args.measurement_cols)
    if not meas_cols:
        raise SystemExit("Provide --measurement_cols (e.g., bmi)")

    # Deterministic split evaluated on test subset only
    full_ds = SilhouetteDataset(args.csv, target_hw=(640, 480), measurement_cols=meas_cols, augment=False)
    if args.max_rows is not None:
        n = min(len(full_ds), int(args.max_rows))
        full_ds = Subset(full_ds, list(range(n)))
    n_total = len(full_ds)
    n_test = max(1, int(round(float(args.test_ratio) * n_total)))
    rng = np.random.default_rng(int(args.seed))
    idx = np.arange(n_total)
    rng.shuffle(idx)
    test_idx = idx[:n_test].tolist()
    test_ds = Subset(full_ds, test_idx)

    def eval_on(ds: Subset, ckpt: str) -> EvalOut:
        # Reuse _eval_ckpt by writing through a temporary "csv_path" is awkward; instead
        # evaluate directly with a lightweight loader over the provided subset.
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        model = _load_model(ckpt, out_meas=len(meas_cols), device=args.device)

        y_meas_true_all: List[np.ndarray] = []
        y_meas_pred_all: List[np.ndarray] = []
        y_bf_true_all: List[np.ndarray] = []
        y_bf_pred_all: List[np.ndarray] = []

        for batch in loader:
            front = batch["front"].to(args.device)
            side = batch["side"].to(args.device)
            if args.mode == "single":
                side = front
            out = model(front, side)
            y_meas_true_all.append(_to_numpy(batch["y_meas"]))
            y_meas_pred_all.append(_to_numpy(out["meas"]))
            y_bf_true_all.append(_to_numpy(batch["y_bf"]).reshape(-1))
            y_bf_pred_all.append(_to_numpy(out["bf"]).reshape(-1))

        y_meas_true = np.concatenate(y_meas_true_all, axis=0)
        y_meas_pred = np.concatenate(y_meas_pred_all, axis=0)
        y_bf_true = np.concatenate(y_bf_true_all, axis=0)
        y_bf_pred = np.concatenate(y_bf_pred_all, axis=0)

        meas_mae = np.abs(y_meas_true - y_meas_pred).mean(axis=0).astype(np.float64).tolist()
        meas_rmse = [float(_rmse(y_meas_true[:, i], y_meas_pred[:, i])) for i in range(y_meas_true.shape[1])]
        meas_r2 = [float(_r2(y_meas_true[:, i], y_meas_pred[:, i])) for i in range(y_meas_true.shape[1])]
        meas_r = [float(_pearson_r(y_meas_true[:, i], y_meas_pred[:, i])) for i in range(y_meas_true.shape[1])]

        # BF: compute metrics in percentage points for MAE/RMSE; R² and r are invariant to scaling
        y_bf_true_pct = y_bf_true * 100.0
        y_bf_pred_pct = y_bf_pred * 100.0
        bf_mae_pct = float(np.abs(y_bf_true_pct - y_bf_pred_pct).mean())
        bf_rmse_pct = float(_rmse(y_bf_true_pct, y_bf_pred_pct))
        bf_r2 = float(_r2(y_bf_true_pct, y_bf_pred_pct))
        bf_r = float(_pearson_r(y_bf_true_pct, y_bf_pred_pct))

        return EvalOut(
            meas_mae=meas_mae,
            meas_rmse=meas_rmse,
            meas_r2=meas_r2,
            meas_pearson_r=meas_r,
            bf_mae_pct=bf_mae_pct,
            bf_rmse_pct=bf_rmse_pct,
            bf_r2=bf_r2,
            bf_pearson_r=bf_r,
        )

    con = eval_on(test_ds, args.ckpt_contrastive)
    reg = eval_on(test_ds, args.ckpt_regonly)

    rows = [("Contrastive", con), ("No-Contrastive", reg)]
    print(_format_md_table(meas_cols, rows))


if __name__ == "__main__":
    main()
