"""Evaluate the SMPLX geometry reliability gate on BodyM rows."""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.infer.masks import load_mask_binary, mask_to_tensor
from src.infer.model_io import load_dualview_checkpoint
from src.infer.predict import predict_from_pair
from src.metrics.body_indices import derive_indices, derive_risk_categories
from src.smpl.gate import default_gate_thresholds, evaluate_smplx_gate


@dataclass(frozen=True)
class SampleResult:
    score: float
    accepted_default: bool
    dimension_mae: float
    max_dimension_error: float
    whr_error: float
    whtr_error: float
    bri_error: float
    waist_risk_match: bool
    whr_risk_match: bool


def _parse_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_rows(csv_path: str, max_rows: int | None) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if max_rows is not None and idx >= max_rows:
                break
            rows.append(row)
    if not rows:
        raise ValueError(f"no rows loaded from {csv_path!r}")
    return rows


def _true_measurements(row: Dict[str, str], cols: List[str]) -> Dict[str, float]:
    return {col: float(row[col]) for col in cols}


def _dimension_errors(true_values: Dict[str, float], pred_values: Dict[str, float], cols: List[str]) -> Tuple[float, float]:
    errors = np.asarray([abs(true_values[col] - pred_values[col]) for col in cols], dtype=np.float64)
    return float(np.mean(errors)), float(np.max(errors))


def _risk_matches(
    true_values: Dict[str, float],
    pred_values: Dict[str, float],
    true_indices: Dict[str, float],
    pred_indices: Dict[str, float],
    sex: str | None,
) -> Tuple[bool, bool]:
    true_risks = derive_risk_categories(true_values, true_indices, sex)
    pred_risks = derive_risk_categories(pred_values, pred_indices, sex)
    waist_match = true_risks.get("waist_circumference") == pred_risks.get("waist_circumference")
    whr_match = true_risks.get("WHR") == pred_risks.get("WHR")
    return waist_match, whr_match


def _safe_index_error(true_indices: Dict[str, float], pred_indices: Dict[str, float], key: str) -> float:
    if key not in true_indices or key not in pred_indices:
        return float("nan")
    return float(abs(true_indices[key] - pred_indices[key]))


def _evaluate_rows(args: argparse.Namespace, rows: List[Dict[str, str]], cols: List[str]) -> List[SampleResult]:
    model, ckpt = load_dualview_checkpoint(args.ckpt, args.device)
    target_hw = tuple(int(x) for x in ckpt["input_size"])
    thresholds = default_gate_thresholds()
    results: List[SampleResult] = []
    for row in tqdm(rows, desc="smplx-gate", dynamic_ncols=True):
        height_cm = float(row["height_cm"])
        sex = row.get("gender")
        front_mask = load_mask_binary(row["front_path"], target_hw)
        side_mask = load_mask_binary(row["side_path"], target_hw)
        front_t = mask_to_tensor(front_mask, args.device)
        side_t = mask_to_tensor(side_mask, args.device)
        pred = predict_from_pair(model, ckpt, front_t, side_t, height_cm, sex)
        true_measurements = _true_measurements(row, cols)
        true_indices = derive_indices(true_measurements, height_cm)
        dim_mae, max_dim_error = _dimension_errors(true_measurements, pred.measurements, cols)
        waist_match, whr_match = _risk_matches(
            true_measurements,
            pred.measurements,
            true_indices,
            pred.indices,
            sex,
        )
        gate = evaluate_smplx_gate(
            front_mask,
            side_mask,
            pred.measurements,
            height_cm,
            sex,
            args.smpl_model_dir,
            thresholds,
        )
        results.append(
            SampleResult(
                score=gate.score,
                accepted_default=gate.accepted,
                dimension_mae=dim_mae,
                max_dimension_error=max_dim_error,
                whr_error=_safe_index_error(true_indices, pred.indices, "WHR"),
                whtr_error=_safe_index_error(true_indices, pred.indices, "WHtR"),
                bri_error=_safe_index_error(true_indices, pred.indices, "BRI"),
                waist_risk_match=waist_match,
                whr_risk_match=whr_match,
            )
        )
    return results


def _mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _coverage_rows(results: List[SampleResult], coverages: List[float]) -> str:
    scores = np.asarray([r.score for r in results], dtype=np.float64)
    dim_mae = np.asarray([r.dimension_mae for r in results], dtype=np.float64)
    max_dim = np.asarray([r.max_dimension_error for r in results], dtype=np.float64)
    whr_err = np.asarray([r.whr_error for r in results], dtype=np.float64)
    whtr_err = np.asarray([r.whtr_error for r in results], dtype=np.float64)
    bri_err = np.asarray([r.bri_error for r in results], dtype=np.float64)
    waist_match = np.asarray([r.waist_risk_match for r in results], dtype=bool)
    whr_match = np.asarray([r.whr_risk_match for r in results], dtype=bool)

    lines = [
        "| Coverage | Score threshold | Accepted | Dimension MAE | Max dimension error | WHR MAE | WHtR MAE | BRI MAE | Waist risk agreement | WHR risk agreement |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for coverage in coverages:
        threshold = float(np.quantile(scores, coverage))
        accepted = scores <= threshold
        accepted_count = int(accepted.sum())
        if accepted_count == 0:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{coverage * 100.0:.0f}%",
                    f"{threshold:.4f}",
                    str(accepted_count),
                    f"{_mean(dim_mae[accepted]):.3f}",
                    f"{_mean(max_dim[accepted]):.3f}",
                    f"{_mean(whr_err[accepted]):.4f}",
                    f"{_mean(whtr_err[accepted]):.4f}",
                    f"{_mean(bri_err[accepted]):.4f}",
                    f"{float(waist_match[accepted].mean() * 100.0):.2f}%",
                    f"{float(whr_match[accepted].mean() * 100.0):.2f}%",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _default_acceptance(results: List[SampleResult]) -> str:
    accepted = np.asarray([r.accepted_default for r in results], dtype=bool)
    if int(accepted.sum()) == 0:
        return "Default gate accepted 0 samples."
    dim_mae = np.asarray([r.dimension_mae for r in results], dtype=np.float64)
    whr_err = np.asarray([r.whr_error for r in results], dtype=np.float64)
    return (
        f"Default gate accepted {int(accepted.sum())}/{len(results)} samples "
        f"({float(accepted.mean() * 100.0):.2f}%). "
        f"Accepted dimension MAE={_mean(dim_mae[accepted]):.3f} cm, "
        f"accepted WHR MAE={_mean(whr_err[accepted]):.4f}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/bodym/pairs_dimensions.csv")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--measurement_cols", default="waist_cm,hip_cm,chest_cm")
    parser.add_argument("--smpl_model_dir", default="models/smplx")
    parser.add_argument("--max_rows", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--coverages", default="1.0,0.9,0.8,0.7")
    args = parser.parse_args()

    cols = _parse_list(args.measurement_cols)
    coverages = [float(item) for item in _parse_list(args.coverages)]
    rows = _load_rows(args.csv, args.max_rows)
    results = _evaluate_rows(args, rows, cols)
    print(_coverage_rows(results, coverages))
    print()
    print(_default_acceptance(results))


if __name__ == "__main__":
    main()

