"""Create silhouettes from local test photos with strict YOLO+SAM2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from pipeline.iphone_pipeline import process_iphone_image_with_metadata
from src.infer.silhouette_checks import envelope_check


TESTPHOTO_SAMPLES: Tuple[Tuple[str, str], ...] = (
    ("IMG_4373.jpeg", "front"),
    ("IMG_4577.jpg", "side"),
    ("deva_front.png", "front"),
    ("deva_side.png", "side"),
    ("image copy 2.png", "front"),
    ("image copy 3.png", "front"),
    ("image copy 4.png", "front"),
    ("image copy 5.png", "front"),
    ("image copy 6.png", "front"),
    ("image copy.png", "front"),
    ("image.png", "front"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="TestPhoto", help="Directory containing the sample RGB photos.")
    parser.add_argument("--debug_dir", help="Output directory for silhouettes.")
    parser.add_argument("--json", help="Optional JSON report path.")
    parser.add_argument("--sam_model_path", help="SAM2 checkpoint path.")
    parser.add_argument("--sam_config_path", help="SAM2 config path.")
    parser.add_argument("--yolo_model_path", help="YOLO checkpoint path.")
    parser.add_argument("--yolo_confidence", type=float, default=0.35)
    return parser.parse_args()


def read_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def sample_prefix(filename: str) -> str:
    return Path(filename).stem.replace(" ", "_")


def mask_metrics(mask: np.ndarray, view: str) -> Dict[str, Any]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        raise ValueError("cannot compute metrics for empty mask")
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    width = max(1, bbox[2] - bbox[0] + 1)
    height = max(1, bbox[3] - bbox[1] + 1)
    report = envelope_check(mask, view)
    return {
        "shape": list(mask.shape),
        "foreground_ratio": float((mask > 0).mean()),
        "bbox": list(bbox),
        "aspect_ratio": float(height) / float(width),
        "envelope_ok": report.ok,
        "envelope_failed": report.failed,
    }


def process_sample(
    root: Path,
    filename: str,
    view: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    image_path = root / filename
    rgb = read_rgb(image_path)
    processed = process_iphone_image_with_metadata(
        img_rgb=rgb,
        sam_model_path=args.sam_model_path,
        sam_config_path=args.sam_config_path,
        yolo_model_path=args.yolo_model_path,
        view=view,
        debug_dir=args.debug_dir,
        debug_prefix=sample_prefix(filename),
        yolo_confidence=args.yolo_confidence,
    )
    bbox = processed.segmentation.bbox
    return {
        "file": filename,
        "prefix": sample_prefix(filename),
        "view": view,
        "ok": True,
        "yolo_box": {
            "x0": bbox.x0,
            "y0": bbox.y0,
            "x1": bbox.x1,
            "y1": bbox.y1,
            "confidence": bbox.confidence,
        },
        "sam_score": processed.segmentation.sam_score,
        "raw_foreground_ratio": processed.segmentation.foreground_ratio,
        "standardization": {
            "source_bbox": list(processed.standardization.source_bbox),
            "crop_shape": list(processed.standardization.crop_shape),
            "scale": processed.standardization.scale,
            "resized_shape": list(processed.standardization.resized_shape),
            "offset_xy": list(processed.standardization.offset_xy),
            "target_shape": list(processed.standardization.target_shape),
        },
        "mask": mask_metrics(processed.mask, view),
    }


def run(args: argparse.Namespace) -> List[Dict[str, Any]]:
    root = Path(args.root)
    rows: List[Dict[str, Any]] = []
    for filename, view in TESTPHOTO_SAMPLES:
        try:
            row = process_sample(root, filename, view, args)
            print(
                "OK {file} view={view} fg={fg:.3f} aspect={aspect:.3f} sam={sam:.3f}".format(
                    file=filename,
                    view=view,
                    fg=row["mask"]["foreground_ratio"],
                    aspect=row["mask"]["aspect_ratio"],
                    sam=row["sam_score"],
                )
            )
        except Exception as exc:
            row = {
                "file": filename,
                "view": view,
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            print(f"FAIL {filename} view={view} {type(exc).__name__}: {exc}")
        rows.append(row)
    return rows


def write_contact_sheet(rows: List[Dict[str, Any]], debug_dir: str) -> str:
    root = Path(debug_dir)
    masks: List[Tuple[str, np.ndarray]] = []
    for row in rows:
        if not bool(row.get("ok")):
            continue
        prefix = str(row["prefix"])
        mask_path = root / f"{prefix}_final_silhouette.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"could not read generated mask for contact sheet: {mask_path}")
        masks.append((prefix, mask))
    if len(masks) == 0:
        raise ValueError("cannot create contact sheet because no segmentation masks succeeded")

    tile_h = 240
    tile_w = 180
    label_h = 30
    cols = min(4, len(masks))
    rows_count = int(np.ceil(float(len(masks)) / float(cols)))
    canvas = np.zeros((rows_count * (tile_h + label_h), cols * tile_w, 3), dtype=np.uint8)
    for index, (label, mask) in enumerate(masks):
        row = index // cols
        col = index % cols
        resized = cv2.resize(mask, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        y0 = row * (tile_h + label_h)
        x0 = col * tile_w
        canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        cv2.putText(
            canvas,
            label[:22],
            (x0 + 6, y0 + tile_h + 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
    output_path = root / "contact_sheet.png"
    ok = cv2.imwrite(str(output_path), canvas)
    if not ok:
        raise OSError(f"failed to write contact sheet: {output_path}")
    return str(output_path)


def main() -> None:
    args = parse_args()
    rows = run(args)
    if args.debug_dir:
        contact_sheet_path = write_contact_sheet(rows, args.debug_dir)
        print(f"Contact sheet: {contact_sheet_path}")
    if args.json:
        output_path = Path(args.json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(rows, indent=2, sort_keys=True))
        print(f"JSON: {output_path}")


if __name__ == "__main__":
    main()
