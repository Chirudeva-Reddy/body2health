#!/usr/bin/env python3
"""Build clean dimension-focused CSVs from the renamed BodyM dataset."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable


MEASUREMENT_COLUMN_MAP = {
    "ankle": "ankle_cm",
    "arm-length": "arm_length_cm",
    "bicep": "bicep_cm",
    "calf": "calf_cm",
    "chest": "chest_cm",
    "forearm": "forearm_cm",
    "height": "measurement_height_cm",
    "hip": "hip_cm",
    "leg-length": "leg_length_cm",
    "shoulder-breadth": "shoulder_breadth_cm",
    "shoulder-to-crotch": "shoulder_to_crotch_cm",
    "thigh": "thigh_cm",
    "waist": "waist_cm",
    "wrist": "wrist_cm",
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def write_csv_rows(path: Path, fieldnames: list[str], rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def renamed_measurement_rows(
    measurements_rows: list[dict[str, str]],
    subject_key_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    subject_key_by_original = {row["original_subject_id"]: row["subject_key"] for row in subject_key_rows}
    rows: list[dict[str, str]] = []
    for row in measurements_rows:
        subject_id = row["subject_id"]
        if subject_id not in subject_key_by_original:
            raise KeyError(f"measurement subject_id missing from subject_key_map: {subject_id}")
        out = {
            "subject_key": subject_key_by_original[subject_id],
            "original_subject_id": subject_id,
        }
        for old_name, new_name in MEASUREMENT_COLUMN_MAP.items():
            out[new_name] = row[old_name]
        rows.append(out)
    return sorted(rows, key=lambda item: item["subject_key"])


def dimension_pair_rows(
    pairs_rows: list[dict[str, str]],
    measurement_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    measurements_by_subject = {row["original_subject_id"]: row for row in measurement_rows}
    rows: list[dict[str, str]] = []
    for row in pairs_rows:
        subject_id = row["original_subject_id"]
        if subject_id not in measurements_by_subject:
            raise KeyError(f"pair subject missing from measurements: {subject_id}")
        measurement = measurements_by_subject[subject_id]
        out = {
            "subject_key": row["subject_key"],
            "capture_key": row["capture_key"],
            "original_subject_id": subject_id,
            "original_photo_id": row["original_photo_id"],
            "front_path": row["front_path"],
            "side_path": row["side_path"],
            "gender": row["gender"],
            "height_cm": row["height_cm"],
            "weight_kg": row["weight_kg"],
        }
        for measurement_column in MEASUREMENT_COLUMN_MAP.values():
            out[measurement_column] = measurement[measurement_column]
        rows.append(out)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renamed-dir", required=True, type=Path)
    parser.add_argument("--measurements-csv", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs_path = args.renamed_dir / "pairs_staging.csv"
    subject_key_map_path = args.renamed_dir / "subject_key_map.csv"
    measurement_rows_raw = read_csv_rows(args.measurements_csv)
    subject_key_rows = read_csv_rows(subject_key_map_path)
    pairs_rows = read_csv_rows(pairs_path)

    measurement_rows = renamed_measurement_rows(measurement_rows_raw, subject_key_rows)
    write_csv_rows(
        args.renamed_dir / "metadata" / "measurements_renamed.csv",
        ["subject_key", "original_subject_id", *MEASUREMENT_COLUMN_MAP.values()],
        measurement_rows,
    )

    pair_rows = dimension_pair_rows(pairs_rows, measurement_rows)
    write_csv_rows(
        args.renamed_dir / "pairs_dimensions.csv",
        [
            "subject_key",
            "capture_key",
            "original_subject_id",
            "original_photo_id",
            "front_path",
            "side_path",
            "gender",
            "height_cm",
            "weight_kg",
            *MEASUREMENT_COLUMN_MAP.values(),
        ],
        pair_rows,
    )
    print(
        {
            "subjects": len(measurement_rows),
            "captures": len(pair_rows),
            "pairs_dimensions": str(args.renamed_dir / "pairs_dimensions.csv"),
            "measurements_renamed": str(args.renamed_dir / "metadata" / "measurements_renamed.csv"),
        }
    )


if __name__ == "__main__":
    main()
