#!/usr/bin/env python3
"""
Compute IoU for paired silhouettes and write logs/silhouette_iou.csv.

Expected input CSV columns:
  - sample_id
  - pred_path: path to predicted silhouette (binary mask; white=255 fg, black=0 bg)
  - gt_path: path to ground-truth silhouette (same format)
Optional:
  - epoch: if present, propagated to output; otherwise use --epoch to set a constant value.

Output:
  - logs/silhouette_iou.csv with columns: sample_id, epoch, iou

Notes:
  - This measures silhouette accuracy only; it is not related to BMI/BF prediction performance.
  - IoU is valid as a preprocessing metric even in single-subject settings.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def load_mask(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return (m > 127).astype(np.uint8)


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_csv", required=True, help="CSV with columns: sample_id,pred_path,gt_path[,epoch]")
    ap.add_argument("--epoch", type=int, default=None, help="Fallback epoch value if not present in CSV")
    ap.add_argument("--out_csv", default="logs/silhouette_iou.csv", help="Output CSV path")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    pairs_path = Path(args.pairs_csv)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows_out = []
    with pairs_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = ["sample_id", "pred_path", "gt_path"]
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise SystemExit(f"Missing required columns in {pairs_path}: {missing}")
        has_epoch = "epoch" in reader.fieldnames

        for row in reader:
            sample_id = row["sample_id"]
            pred_path = Path(row["pred_path"])
            gt_path = Path(row["gt_path"])
            epoch_val = int(row["epoch"]) if has_epoch and row.get("epoch") not in (None, "",) else args.epoch
            if epoch_val is None:
                raise SystemExit("No epoch provided; add an 'epoch' column or pass --epoch")

            pred = load_mask(pred_path)
            gt = load_mask(gt_path)
            iou_val = iou(pred, gt)
            rows_out.append(
                {
                    "sample_id": sample_id,
                    "epoch": epoch_val,
                    "iou": f"{iou_val:.6f}",
                }
            )

    with out_path.open("w", newline="") as f:
        fieldnames = ["sample_id", "epoch", "iou"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)


if __name__ == "__main__":
    main()
