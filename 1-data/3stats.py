"""
Compute dataset statistics for labels and pairs CSVs.

Usage:
python 1-data/3stats.py --labels "data copy/labels.csv" --pairs "data copy/pairs.csv" --out docs/dataset_stats.md
"""
import argparse
import csv
import math
import os
from statistics import mean, median, pstdev
from typing import Dict, List, Tuple


NumericList = List[float]


def _safe_float(val: str):
    try:
        return float(val)
    except Exception:
        return None


def _summary(vals: NumericList) -> Dict[str, float]:
    if not vals:
        return {"count": 0, "min": math.nan, "max": math.nan, "mean": math.nan, "median": math.nan, "std": math.nan}
    return {
        "count": len(vals),
        "min": min(vals),
        "max": max(vals),
        "mean": mean(vals),
        "median": median(vals),
        "std": pstdev(vals) if len(vals) > 1 else 0.0,
    }


def _read_numeric_column(path: str, col: str) -> NumericList:
    vals: NumericList = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            v = _safe_float(row.get(col, ""))
            if v is not None:
                vals.append(v)
    return vals


def _summaries(path: str, cols: List[str]) -> Dict[str, Dict[str, float]]:
    return {c: _summary(_read_numeric_column(path, c)) for c in cols}


def _format_summary(name: str, stats: Dict[str, float]) -> str:
    if stats["count"] == 0:
        return f"- {name}: no data"
    return (
        f"- {name}: n={stats['count']}, "
        f"min={stats['min']:.2f}, max={stats['max']:.2f}, "
        f"mean={stats['mean']:.2f}, median={stats['median']:.2f}, "
        f"std={stats['std']:.2f}"
    )


def _write_markdown(out_path: str,
                    label_stats: Dict[str, Dict[str, float]],
                    pair_stats: Dict[str, Dict[str, float]],
                    label_path: str,
                    pair_path: str) -> None:
    lines: List[str] = []
    lines.append("# Dataset Statistics\n")
    lines.append(f"- Labels CSV: `{label_path}`")
    lines.append(f"- Pairs  CSV: `{pair_path}`\n")

    lines.append("## labels.csv\n")
    for name, stats in label_stats.items():
        lines.append(_format_summary(name, stats))
    lines.append("")

    lines.append("## pairs.csv\n")
    for name, stats in pair_stats.items():
        lines.append(_format_summary(name, stats))
    lines.append("")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=str, default="data copy/labels.csv")
    parser.add_argument("--pairs", type=str, default="data copy/pairs.csv")
    parser.add_argument("--out", type=str, default="docs/dataset_stats.md")
    args = parser.parse_args()

    label_cols = ["height_cm", "hip_cm", "bai"]
    pair_cols = ["height_cm", "weight_kg", "bf_pseudo"]

    label_stats = _summaries(args.labels, label_cols)
    pair_stats = _summaries(args.pairs, pair_cols)

    print("labels.csv stats:")
    for name, stats in label_stats.items():
        print(_format_summary(name, stats))

    print("\npairs.csv stats:")
    for name, stats in pair_stats.items():
        print(_format_summary(name, stats))

    _write_markdown(args.out, label_stats, pair_stats, args.labels, args.pairs)
    print(f"\nWrote markdown to {args.out}")


if __name__ == "__main__":
    main()
