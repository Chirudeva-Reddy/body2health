#!/usr/bin/env python3
"""
Generate publication-quality figures for a single-subject, multi-sample silhouette study.

Constraints honored:
- Single subject only; plots show optimization and intra-subject stability, not population generalization.
- IoU is a preprocessing metric only and is kept separate from BMI/BF results.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

# Input paths
TRAIN_LOG = Path("logs/training_log.csv")
IOU_LOG = Path("logs/silhouette_iou.csv")

# Output paths
OUT_DIR = Path("outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MA_WINDOW = 3  # moving average window for visualization only
RAW_LOG_FALLBACK = Path("out/train_640x480.log")  # used to recover batch losses if CSV lacks decomposition


def read_csv_numeric(path: Path, required_cols: List[str]) -> Dict[str, np.ndarray]:
    """Read required columns from CSV and return as float arrays; fail fast on missing columns."""
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in required_cols if c not in reader.fieldnames]
        if missing:
            raise SystemExit(f"Missing required columns in {path}: {missing}")
        cols = {c: [] for c in required_cols}
        for row in reader:
            for c in required_cols:
                try:
                    cols[c].append(float(row[c]))
                except Exception:
                    cols[c].append(math.nan)
    return {k: np.asarray(v, dtype=float) for k, v in cols.items()}


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """Simple moving average with NaN padding to preserve length."""
    if w <= 1 or x.size == 0:
        return x
    filt = np.convolve(x, np.ones(w, dtype=float) / float(w), mode="valid")
    pad = np.full((w - 1,), np.nan, dtype=float)
    return np.concatenate([pad, filt])


def recover_batch_losses_from_raw_log(raw_log: Path) -> Dict[int, Dict[str, float]]:
    """
    Parse batch-level lines from the raw training log to recover per-epoch
    mean contrastive and regression losses when the CSV lacks them.
    """
    if not raw_log.exists():
        return {}
    import re

    pat = re.compile(
        r"Epoch (\d+)/(\d+) \| Batch (\d+)/(\d+) \| Loss ([\d.]+) \| Contrastive ([\d.]+) \| Regression ([\d.]+)"
    )
    per_epoch = {}
    with raw_log.open() as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            epoch = int(m.group(1))
            # total_loss = float(m.group(5))  # not used here
            con = float(m.group(6))
            reg = float(m.group(7))
            acc = per_epoch.setdefault(epoch, {"con": [], "reg": []})
            acc["con"].append(con)
            acc["reg"].append(reg)
    out = {}
    for e, acc in per_epoch.items():
        if acc["con"] and acc["reg"]:
            out[e] = {
                "con": float(np.nanmean(acc["con"])),
                "reg": float(np.nanmean(acc["reg"])),
            }
    return out


def fill_missing_decomposition(train: Dict[str, np.ndarray], epoch: np.ndarray):
    """Fill train_contrastive/train_regression if missing, using raw log batch means."""
    con = train["train_contrastive"]
    reg = train["train_regression"]
    if not (np.all(np.isnan(con)) or np.all(np.isnan(reg))):
        return  # nothing to fill
    recovered = recover_batch_losses_from_raw_log(RAW_LOG_FALLBACK)
    if not recovered:
        return
    con_filled = []
    reg_filled = []
    for e in epoch.astype(int):
        if e in recovered:
            con_filled.append(recovered[e]["con"])
            reg_filled.append(recovered[e]["reg"])
        else:
            con_filled.append(np.nan)
            reg_filled.append(np.nan)
    train["train_contrastive"] = np.asarray(con_filled, dtype=float)
    train["train_regression"] = np.asarray(reg_filled, dtype=float)


def style_ax(ax, title: str, xlabel: str, ylabel: str):
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)


def fig1_train_val_loss(epoch: np.ndarray, train: np.ndarray, val: np.ndarray):
    fig, ax = plt.subplots(figsize=(6, 4))
    # Raw curves faint: optimizer trace is valid even for single-subject optimization.
    ax.plot(epoch, train, color="tab:blue", alpha=0.25, linewidth=1, label="Train total (raw)")
    ax.plot(epoch, val, color="tab:orange", alpha=0.25, linewidth=1, label="Val total (raw)")
    ax.plot(epoch, moving_average(train, MA_WINDOW), color="tab:blue", linewidth=2, label=f"Train total (MA{MA_WINDOW})")
    ax.plot(epoch, moving_average(val, MA_WINDOW), color="tab:orange", linewidth=2, label=f"Val total (MA{MA_WINDOW})")
    style_ax(ax, "Training and Validation Loss (Single-Subject Study)", "Epoch", "Total Loss")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_train_val_loss.png", dpi=300)
    plt.close(fig)


def fig2_loss_decomposition(epoch: np.ndarray, con: np.ndarray, reg: np.ndarray):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epoch, con, color="tab:green", linewidth=2, label="Contrastive loss")
    ax.plot(epoch, reg, color="tab:red", linewidth=2, label="Regression loss")
    style_ax(ax, "Loss Decomposition During Training", "Epoch", "Loss")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_loss_decomposition.png", dpi=300)
    plt.close(fig)


def fig3_val_mae_bmi(epoch: np.ndarray, bmi_mae: np.ndarray):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epoch, bmi_mae, color="tab:blue", linewidth=2, label="Val MAE BMI")
    # Single-subject validity: MAE measures intra-subject consistency across multiple samples of the same individual.
    style_ax(ax, "Validation MAE for BMI (Single-Subject Consistency)", "Epoch", "MAE (kg/m²)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_val_mae_bmi.png", dpi=300)
    plt.close(fig)


def fig4_val_mae_bf(epoch: np.ndarray, bf_mae: np.ndarray):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epoch, bf_mae, color="tab:purple", linewidth=2, label="Val MAE BF")
    # Single-subject validity: MAE measures intra-subject consistency across multiple samples of the same individual.
    style_ax(ax, "Validation MAE for BF% (Single-Subject Consistency)", "Epoch", "MAE (%)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_val_mae_bf.png", dpi=300)
    plt.close(fig)


def fig8a_iou_hist(iou_vals: np.ndarray):
    clean = iou_vals[np.isfinite(iou_vals)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(clean, bins=20, color="tab:gray", alpha=0.85, edgecolor="white")
    mean_iou = float(np.nanmean(clean)) if clean.size else float("nan")
    ax.axvline(mean_iou, color="tab:blue", linestyle="--", linewidth=2, label=f"Mean IoU = {mean_iou:.3f}")
    ax.text(mean_iou, ax.get_ylim()[1] * 0.9, f"{mean_iou:.3f}", color="tab:blue", ha="center", va="top", fontsize=9)
    # IoU is a preprocessing-only metric; it is not related to BMI/BF prediction performance.
    style_ax(ax, "Silhouette Extraction Quality (IoU Distribution)", "Intersection-over-Union (IoU)", "Number of Samples")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig8a_iou_distribution.png", dpi=300)
    plt.close(fig)


def fig8b_iou_vs_epoch(iou_epochs: np.ndarray, iou_vals: np.ndarray):
    mask = np.isfinite(iou_epochs) & np.isfinite(iou_vals)
    iou_epochs = iou_epochs[mask]
    iou_vals = iou_vals[mask]
    if iou_vals.size == 0:
        return
    unique_epochs = np.unique(iou_epochs)
    mean_per_epoch = []
    for e in unique_epochs:
        epoch_mask = iou_epochs == e
        mean_per_epoch.append(np.nanmean(iou_vals[epoch_mask]))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(unique_epochs, mean_per_epoch, color="tab:green", linewidth=2, marker="o", label="Mean IoU")
    # IoU trend shows preprocessing stability over epochs; it should not be interpreted as BMI/BF accuracy.
    style_ax(ax, "Mean Silhouette IoU Across Training Epochs", "Epoch", "Mean IoU")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig8b_iou_vs_epoch.png", dpi=300)
    plt.close(fig)


def main():
    # Load training log
    train_cols = [
        "epoch",
        "train_total",
        "train_contrastive",
        "train_regression",
        "val_total",
        "val_mae_bmi",
        "val_mae_bf",
    ]
    train = read_csv_numeric(TRAIN_LOG, train_cols)
    epoch = train["epoch"]
    fill_missing_decomposition(train, epoch)

    fig1_train_val_loss(epoch, train["train_total"], train["val_total"])
    fig2_loss_decomposition(epoch, train["train_contrastive"], train["train_regression"])
    fig3_val_mae_bmi(epoch, train["val_mae_bmi"])
    fig4_val_mae_bf(epoch, train["val_mae_bf"])

    # Load IoU log if available
    if IOU_LOG.exists():
        iou_cols = ["sample_id", "epoch", "iou"]
        iou = read_csv_numeric(IOU_LOG, iou_cols)
        fig8a_iou_hist(iou["iou"])
        fig8b_iou_vs_epoch(iou["epoch"], iou["iou"])


if __name__ == "__main__":
    main()
