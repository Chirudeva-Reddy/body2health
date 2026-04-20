#!/usr/bin/env python3
"""
Plot training curves from `3-train/train_contrastive_640x480.py` logs.

Parses the log text directly (same patterns as `1-data/extract_training_metrics.py`)
and writes PNGs to an output directory.

Usage:
  PYTHONPATH=. python 5-eval/plot_training_curves.py --log out/train_640x480.log --out_dir out/plots
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _parse_log(log_path: str) -> Tuple[List[Dict], List[Dict]]:
    batch_pattern = re.compile(
        r"Epoch (\d+)/(\d+) \| Batch (\d+)/(\d+) \| Loss ([\d.]+) \| "
        r"Contrastive ([\d.]+) \| Regression ([\d.]+)"
    )
    epoch_pattern = re.compile(
        r"Epoch (\d+)/(\d+) \| Train Loss ([\d.]+) \| Val Loss ([\d.]+) \| "
        r"Val MAE BMI ([\d.]+) \| Val MAE BF ([\d.]+)%"
    )

    batches: List[Dict] = []
    epochs: List[Dict] = []

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            m = batch_pattern.search(line)
            if m:
                epoch, total_epochs, batch, total_batches = map(int, m.groups()[:4])
                total_loss, contrastive_loss, regression_loss = map(float, m.groups()[4:])
                # Convert to a continuous step axis for plotting
                step = (epoch - 1) * total_batches + batch
                batches.append(
                    {
                        "epoch": epoch,
                        "batch": batch,
                        "total_batches": total_batches,
                        "step": step,
                        "total_loss": total_loss,
                        "contrastive_loss": contrastive_loss,
                        "regression_loss": regression_loss,
                    }
                )
                continue

            m = epoch_pattern.search(line)
            if m:
                epoch, total_epochs = map(int, m.groups()[:2])
                train_loss, val_loss, bmi_mae, bf_mae = map(float, m.groups()[2:])
                epochs.append(
                    {
                        "epoch": epoch,
                        "total_epochs": total_epochs,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "bmi_mae": bmi_mae,
                        "bf_mae_pct": bf_mae,
                    }
                )

    if not epochs and not batches:
        raise SystemExit(f"No metrics matched expected patterns in {log_path}")
    return epochs, batches


def _moving_average(y: List[float], window: int) -> List[float]:
    if window <= 1:
        return y[:]
    out: List[float] = []
    s = 0.0
    q: List[float] = []
    for v in y:
        q.append(v)
        s += v
        if len(q) > window:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Training log path (e.g. out/train_640x480.log).")
    ap.add_argument("--out_dir", default="out/plots", help="Where to write PNGs.")
    ap.add_argument("--ma_window", type=int, default=25, help="Moving average window for batch curves.")
    args = ap.parse_args()

    log_path = str(args.log)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import cv2  # type: ignore
    except Exception as e:
        raise SystemExit(f"opencv-python is required for plotting fallback but is not available: {e}")

    epochs, batches = _parse_log(log_path)

    def _plot_series(
        out_path: Path,
        x: List[float],
        series: List[Tuple[str, List[float], Tuple[int, int, int]]],
        title: str,
        x_label: str,
        y_label: str,
        width: int = 1200,
        height: int = 650,
    ) -> None:
        # Simple OpenCV plotting (no matplotlib dependency).
        # BGR colors.
        bg = np.full((height, width, 3), 255, dtype=np.uint8)
        margin_l, margin_r, margin_t, margin_b = 80, 20, 60, 80

        if not x:
            return
        x_min, x_max = float(min(x)), float(max(x))
        if x_max <= x_min:
            x_max = x_min + 1.0

        # Determine y-range across all series
        y_vals = [v for _, ys, _ in series for v in ys if np.isfinite(v)]
        if not y_vals:
            return
        y_min, y_max = float(min(y_vals)), float(max(y_vals))
        if y_max <= y_min:
            y_max = y_min + 1.0
        # Add a little padding
        pad = 0.05 * (y_max - y_min)
        y_min -= pad
        y_max += pad

        def xy_to_px(xv: float, yv: float) -> Tuple[int, int]:
            px = margin_l + int(round((xv - x_min) / (x_max - x_min) * (width - margin_l - margin_r)))
            py = margin_t + int(round((1.0 - (yv - y_min) / (y_max - y_min)) * (height - margin_t - margin_b)))
            return px, py

        # Axes
        cv2.line(bg, (margin_l, margin_t), (margin_l, height - margin_b), (0, 0, 0), 2)
        cv2.line(bg, (margin_l, height - margin_b), (width - margin_r, height - margin_b), (0, 0, 0), 2)

        # Grid
        for i in range(1, 6):
            yv = y_min + (y_max - y_min) * i / 6.0
            _, py = xy_to_px(x_min, yv)
            cv2.line(bg, (margin_l, py), (width - margin_r, py), (220, 220, 220), 1)

        # Plot each series
        for name, ys, color in series:
            pts = []
            for xv, yv in zip(x, ys):
                if not np.isfinite(yv):
                    continue
                pts.append(xy_to_px(float(xv), float(yv)))
            for p0, p1 in zip(pts, pts[1:]):
                cv2.line(bg, p0, p1, color, 2, lineType=cv2.LINE_AA)

        # Title + labels
        cv2.putText(bg, title, (margin_l, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(bg, x_label, (width // 2 - 40, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(bg, y_label, (10, margin_t + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        # Legend
        lx, ly = margin_l + 10, margin_t + 10
        for name, _, color in series:
            cv2.rectangle(bg, (lx, ly - 12), (lx + 20, ly + 2), color, -1)
            cv2.putText(bg, name, (lx + 28, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            ly += 24

        cv2.imwrite(str(out_path), bg)

    # 1) Batch losses (total/contrastive/regression) vs step
    if batches:
        steps = [b["step"] for b in batches]
        total = [b["total_loss"] for b in batches]
        con = [b["contrastive_loss"] for b in batches]
        reg = [b["regression_loss"] for b in batches]
        out_path = out_dir / "batch_loss_components.png"
        _plot_series(
            out_path,
            steps,
            [
                (f"total (MA{args.ma_window})", _moving_average(total, args.ma_window), (0, 0, 0)),
                (f"contrastive (MA{args.ma_window})", _moving_average(con, args.ma_window), (0, 130, 255)),
                (f"regression (MA{args.ma_window})", _moving_average(reg, args.ma_window), (0, 170, 0)),
            ],
            title="Batch Loss Components (moving average)",
            x_label="optimizer step",
            y_label="loss",
        )

    # 2) Epoch train/val loss vs epoch
    if epochs:
        e = [x["epoch"] for x in epochs]
        train_loss = [x["train_loss"] for x in epochs]
        val_loss = [x["val_loss"] for x in epochs]
        bmi_mae = [x["bmi_mae"] for x in epochs]
        bf_mae = [x["bf_mae_pct"] for x in epochs]
        _plot_series(
            out_dir / "epoch_loss.png",
            e,
            [("train loss", train_loss, (0, 0, 0)), ("val loss", val_loss, (0, 0, 255))],
            title="Epoch Loss",
            x_label="epoch",
            y_label="loss",
        )
        # MAE plot (same y-axis; BF in percent, BMI in units; keep separate images for clarity)
        _plot_series(
            out_dir / "epoch_bmi_mae.png",
            e,
            [("BMI MAE", bmi_mae, (255, 0, 0))],
            title="Validation BMI MAE",
            x_label="epoch",
            y_label="BMI MAE",
        )
        _plot_series(
            out_dir / "epoch_bf_mae.png",
            e,
            [("BF MAE (%)", bf_mae, (0, 130, 255))],
            title="Validation BF MAE (%)",
            x_label="epoch",
            y_label="BF MAE (%)",
        )

    print(f"Wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
