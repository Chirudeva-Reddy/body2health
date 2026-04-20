"""
Resize existing paired masks to 640x480 and optionally create a 10% subset CSV.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def _resolve_loose_data_path(path: str) -> str:
    p = Path(path)
    if p.exists():
        return str(p)
    alt = path.replace("data/", "data /") if "data/" in path else path.replace("data /", "data/")
    p2 = Path(alt)
    if p2.exists():
        return str(p2)
    return path


def prepare_640x480_dataset(csv_path: str, output_dir: str) -> Path:
    src = Path(_resolve_loose_data_path(csv_path))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src)
    rows = []

    for idx, row in df.iterrows():
        front_img = cv2.imread(row["front_path"], cv2.IMREAD_GRAYSCALE)
        side_img = cv2.imread(row["side_path"], cv2.IMREAD_GRAYSCALE)
        if front_img is None or side_img is None:
            print(f"Skipping idx {idx}: missing mask files")
            continue

        front_resized = cv2.resize(front_img, (480, 640), interpolation=cv2.INTER_AREA)
        side_resized = cv2.resize(side_img, (480, 640), interpolation=cv2.INTER_AREA)

        front_out = out_dir / f"front_{idx:06d}.png"
        side_out = out_dir / f"side_{idx:06d}.png"
        cv2.imwrite(str(front_out), front_resized)
        cv2.imwrite(str(side_out), side_resized)

        rows.append(
            {
                "front_path": str(front_out),
                "side_path": str(side_out),
                "height_cm": row.get("height_cm", 0.0),
                "weight_kg": row.get("weight_kg", 0.0),
                "bf_pseudo": row.get("bf_pseudo", row.get("bf", 0.0)),
            }
        )

    processed = pd.DataFrame(rows)
    processed_csv = out_dir / "pairs_640x480.csv"
    processed.to_csv(processed_csv, index=False)
    print(f"Processed dataset saved: {processed_csv} (rows: {len(processed)})")
    return processed_csv


def create_subset(input_csv: Path, ratio: float, seed: int) -> Path:
    df = pd.read_csv(input_csv)
    subset_size = max(1, int(len(df) * ratio))
    np.random.seed(seed)
    subset_idx = np.random.choice(len(df), subset_size, replace=False)
    subset_df = df.iloc[subset_idx].reset_index(drop=True)
    subset_csv = input_csv.parent / f"pairs_{int(ratio*100)}percent.csv"
    subset_df.to_csv(subset_csv, index=False)
    print(f"Subset saved: {subset_csv} (rows: {len(subset_df)})")
    return subset_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/640x480_processed")
    parser.add_argument("--create_subset", action="store_true")
    parser.add_argument("--subset_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    processed_csv = prepare_640x480_dataset(args.input_csv, args.output_dir)
    if args.create_subset:
        create_subset(processed_csv, args.subset_ratio, args.seed)


if __name__ == "__main__":
    main()
