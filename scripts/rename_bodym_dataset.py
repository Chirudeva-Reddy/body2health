#!/usr/bin/env python3
"""Create a readable, non-destructive mirror of the BodyM mask dataset."""
from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class CaptureRecord:
    subject_key: str
    capture_key: str
    original_subject_id: str
    original_photo_id: str
    old_front_path: Path
    old_side_path: Path
    new_front_path: Path
    new_side_path: Path


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def write_csv_rows(path: Path, fieldnames: list[str], rows: Iterable[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_subject_keys(map_rows: list[dict[str, str]]) -> dict[str, str]:
    subject_keys: dict[str, str] = {}
    for row in map_rows:
        subject_id = row["subject_id"]
        if subject_id not in subject_keys:
            subject_keys[subject_id] = f"sub_{len(subject_keys) + 1:04d}"
    return subject_keys


def build_capture_records(raw_dir: Path, renamed_dir: Path, map_rows: list[dict[str, str]]) -> list[CaptureRecord]:
    subject_keys = build_subject_keys(map_rows)
    capture_counts: dict[str, int] = {}
    records: list[CaptureRecord] = []
    for row in map_rows:
        subject_id = row["subject_id"]
        photo_id = row["photo_id"]
        subject_key = subject_keys[subject_id]
        capture_index = capture_counts.get(subject_id, 0) + 1
        capture_counts[subject_id] = capture_index
        capture_key = f"cap_{capture_index:03d}"
        old_front_path = raw_dir / "mask" / f"{photo_id}.png"
        old_side_path = raw_dir / "mask_left" / f"{photo_id}.png"
        subject_dir = renamed_dir / "raw_masks" / subject_key
        records.append(
            CaptureRecord(
                subject_key=subject_key,
                capture_key=capture_key,
                original_subject_id=subject_id,
                original_photo_id=photo_id,
                old_front_path=old_front_path,
                old_side_path=old_side_path,
                new_front_path=subject_dir / f"{capture_key}_front.png",
                new_side_path=subject_dir / f"{capture_key}_side.png",
            )
        )
    return records


def validate_capture_records(records: list[CaptureRecord]) -> None:
    missing_paths: list[str] = []
    for record in records:
        if not record.old_front_path.exists():
            missing_paths.append(str(record.old_front_path))
        if not record.old_side_path.exists():
            missing_paths.append(str(record.old_side_path))
    if missing_paths:
        preview = "\n".join(missing_paths[:20])
        raise FileNotFoundError(f"Missing source mask files. First missing paths:\n{preview}")


def copy_capture_records(records: list[CaptureRecord]) -> None:
    for record in records:
        record.new_front_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record.old_front_path, record.new_front_path)
        shutil.copy2(record.old_side_path, record.new_side_path)


def make_manifest_rows(records: list[CaptureRecord]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for record in records:
        rows.append(
            {
                "subject_key": record.subject_key,
                "original_subject_id": record.original_subject_id,
                "capture_key": record.capture_key,
                "original_photo_id": record.original_photo_id,
                "view": "front",
                "old_path": str(record.old_front_path),
                "new_path": str(record.new_front_path),
            }
        )
        rows.append(
            {
                "subject_key": record.subject_key,
                "original_subject_id": record.original_subject_id,
                "capture_key": record.capture_key,
                "original_photo_id": record.original_photo_id,
                "view": "side",
                "old_path": str(record.old_side_path),
                "new_path": str(record.new_side_path),
            }
        )
    return rows


def make_subject_map_rows(records: list[CaptureRecord]) -> list[dict[str, str]]:
    seen: dict[str, str] = {}
    for record in records:
        seen[record.original_subject_id] = record.subject_key
    return [
        {"subject_key": subject_key, "original_subject_id": subject_id}
        for subject_id, subject_key in sorted(seen.items(), key=lambda item: item[1])
    ]


def make_pairs_rows(pairs_rows: list[dict[str, str]], records: list[CaptureRecord]) -> tuple[list[str], list[dict[str, str]]]:
    by_photo_id = {record.original_photo_id: record for record in records}
    renamed_rows: list[dict[str, str]] = []
    original_fieldnames = list(pairs_rows[0].keys()) if pairs_rows else []
    fieldnames = [
        "subject_key",
        "capture_key",
        "original_subject_id",
        "original_photo_id",
        "front_path",
        "side_path",
        "old_front_path",
        "old_side_path",
    ]
    for fieldname in original_fieldnames:
        if fieldname not in {"subject_id", "photo_id", "front_path", "side_path"}:
            fieldnames.append(fieldname)
    for row in pairs_rows:
        record = by_photo_id[row["photo_id"]]
        renamed_row = {
            "subject_key": record.subject_key,
            "capture_key": record.capture_key,
            "original_subject_id": record.original_subject_id,
            "original_photo_id": record.original_photo_id,
            "front_path": str(record.new_front_path),
            "side_path": str(record.new_side_path),
            "old_front_path": str(record.old_front_path),
            "old_side_path": str(record.old_side_path),
        }
        for fieldname in original_fieldnames:
            if fieldname not in {"subject_id", "photo_id", "front_path", "side_path"}:
                renamed_row[fieldname] = row[fieldname]
        renamed_rows.append(renamed_row)
    return fieldnames, renamed_rows


def write_readme(renamed_dir: Path) -> None:
    readme_path = renamed_dir / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                "# Renamed BodyM Mask Mirror",
                "",
                "This folder is generated by `scripts/rename_bodym_dataset.py`.",
                "The source hash-named dataset passed via `--raw-dir` is not modified.",
                "",
                "Naming rule:",
                "",
                "```text",
                "raw_masks/sub_0001/cap_001_front.png",
                "raw_masks/sub_0001/cap_001_side.png",
                "```",
                "",
                "- `sub_XXXX` is a stable anonymized subject index.",
                "- `cap_XXX` is the repeated photo/capture index for that subject.",
                "- `front` comes from `raw/mask/`.",
                "- `side` comes from `raw/mask_left/`.",
                "- `manifest.csv` maps every renamed file back to the original subject and photo IDs.",
                "- `pairs_staging.csv` is an intermediate paired CSV using renamed mask paths.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def ensure_clean_output_dir(output_dir: Path, overwrite: bool) -> None:
    if not output_dir.exists():
        return
    if not overwrite:
        raise FileExistsError(f"Output directory already exists: {output_dir}. Pass --overwrite to rebuild it.")
    shutil.rmtree(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--subject-map", required=True, type=Path)
    parser.add_argument("--pairs-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_clean_output_dir(args.output_dir, bool(args.overwrite))
    map_rows = read_csv_rows(args.subject_map)
    pairs_rows = read_csv_rows(args.pairs_csv)
    records = build_capture_records(args.raw_dir, args.output_dir, map_rows)
    validate_capture_records(records)
    photo_ids = {record.original_photo_id for record in records}
    missing_pair_ids = [row["photo_id"] for row in pairs_rows if row["photo_id"] not in photo_ids]
    if missing_pair_ids:
        preview = "\n".join(missing_pair_ids[:20])
        raise KeyError(f"Pairs CSV contains photo IDs missing from subject map. First IDs:\n{preview}")

    copy_capture_records(records)
    write_csv_rows(
        args.output_dir / "manifest.csv",
        ["subject_key", "original_subject_id", "capture_key", "original_photo_id", "view", "old_path", "new_path"],
        make_manifest_rows(records),
    )
    write_csv_rows(
        args.output_dir / "subject_key_map.csv",
        ["subject_key", "original_subject_id"],
        make_subject_map_rows(records),
    )
    pairs_fieldnames, pairs_staging_rows = make_pairs_rows(pairs_rows, records)
    write_csv_rows(args.output_dir / "pairs_staging.csv", pairs_fieldnames, pairs_staging_rows)
    write_readme(args.output_dir)
    print(
        {
            "subjects": len({record.subject_key for record in records}),
            "captures": len(records),
            "files": len(records) * 2,
            "output_dir": str(args.output_dir),
        }
    )


if __name__ == "__main__":
    main()
