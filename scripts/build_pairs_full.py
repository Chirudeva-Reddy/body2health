"""
Build an expanded pairs CSV by merging subject metadata with available front/side masks.

Sources:
- data /raw/metadata/hwg_metadata.csv (subject_id, gender, height_cm, weight_kg)
- data /raw/metadata/measurements.csv (subject_id, hip, etc.)
- data /raw/metadata/subject_to_photo_map.csv (subject_id -> photo_id)
- data /raw/mask/<photo_id>.png and data /raw/mask_left/<photo_id>.png

Output:
- data /pairs_full.csv with columns:
  photo_id, subject_id, gender, height_cm, weight_kg, front_path, side_path,
  bf_pseudo (BAI approximation), hip_cm, bmi, bai
"""
import csv
from pathlib import Path
from typing import Dict, Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data "
RAW = DATA / "raw"


def load_hwg(path: Path) -> Dict[str, Dict[str, Any]]:
    out = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[row["subject_id"]] = {
                "gender": row.get("gender", ""),
                "height_cm": float(row.get("height_cm", "0") or 0.0),
                "weight_kg": float(row.get("weight_kg", "0") or 0.0),
            }
    return out


def load_measurements(path: Path) -> Dict[str, Dict[str, Any]]:
    out = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[row["subject_id"]] = row
    return out


def bmi_from_hw(weight_kg: float, height_cm: float) -> float:
    h_m = max(1e-6, height_cm / 100.0)
    return weight_kg / (h_m * h_m)


def bai_from_hip_height(hip_cm: float, height_cm: float) -> float:
    h_m = max(1e-6, height_cm / 100.0)
    return (hip_cm / (h_m ** 1.5)) - 18.0


def main():
    hwg = load_hwg(RAW / "metadata" / "hwg_metadata.csv")
    meas = load_measurements(RAW / "metadata" / "measurements.csv")

    photo_map = []
    with (RAW / "metadata" / "subject_to_photo_map.csv").open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            photo_map.append(row)

    mask_dir = RAW / "mask"
    mask_left_dir = RAW / "mask_left"

    rows_out = []
    for row in photo_map:
        sid = row["subject_id"]
        pid = row["photo_id"]
        front_path = mask_dir / f"{pid}.png"
        side_path = mask_left_dir / f"{pid}.png"
        if not (front_path.exists() and side_path.exists()):
            continue
        if sid not in hwg:
            continue
        meta = hwg[sid]
        height_cm = meta.get("height_cm", 0.0)
        weight_kg = meta.get("weight_kg", 0.0)
        if height_cm <= 0 or weight_kg <= 0:
            continue
        hip_cm = None
        if sid in meas:
            try:
                hip_cm = float(meas[sid].get("hip", "nan"))
            except Exception:
                hip_cm = None
        bmi = bmi_from_hw(weight_kg, height_cm)
        bai = bai_from_hip_height(hip_cm, height_cm) if hip_cm else ""
        bf_pseudo = bai if hip_cm else ""
        rows_out.append({
            "photo_id": pid,
            "subject_id": sid,
            "gender": meta.get("gender", ""),
            "height_cm": f"{height_cm:.2f}",
            "weight_kg": f"{weight_kg:.2f}",
            "front_path": str(front_path),
            "side_path": str(side_path),
            "bf_pseudo": bf_pseudo,
            "hip_cm": f"{hip_cm:.2f}" if hip_cm else "",
            "bmi": f"{bmi:.4f}",
            "bai": f"{bai:.4f}" if hip_cm else "",
        })

    out_path = DATA / "pairs_full.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        fieldnames = ["photo_id", "subject_id", "gender", "height_cm", "weight_kg",
                      "front_path", "side_path", "bf_pseudo", "hip_cm", "bmi", "bai"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote {len(rows_out)} rows to {out_path}")


if __name__ == "__main__":
    main()
