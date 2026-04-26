"""Evaluate a trained 640x480 model on an iPhone image or a precomputed silhouette."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from pipeline.iphone_pipeline import process_iphone_image, validate_bodym_compatibility
from src.infer.masks import mask_to_tensor
from src.infer.model_io import load_dualview_checkpoint
from src.infer.predict import predict_from_pair


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--image_path", help="Path to raw iPhone image.")
    parser.add_argument("--silhouette_path", help="Path to preprocessed silhouette.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.image_path and not args.silhouette_path:
        raise SystemExit("Provide --image_path or --silhouette_path")

    model, ckpt = load_dualview_checkpoint(args.model_path, args.device)

    if args.image_path:
        img_bgr = cv2.imread(args.image_path)
        if img_bgr is None:
            raise SystemExit(f"Could not read image: {args.image_path}")
        sil = process_iphone_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    else:
        sil = cv2.imread(args.silhouette_path, cv2.IMREAD_GRAYSCALE)
        if sil is None:
            raise SystemExit(f"Could not read silhouette: {args.silhouette_path}")

    validation = validate_bodym_compatibility(sil)
    if not validation["compatible"]:
        print(f"Warning: silhouette validation failed: {validation['issues']}")

    tensor = mask_to_tensor(sil, args.device)
    result = predict_from_pair(model, ckpt, tensor, tensor)

    print("\nPrediction Results")
    print(f"BMI: {result.bmi:.2f}")
    print(f"Body Fat %: {result.bf_pct:.2f}")
    print(f"Measurements: {result.measurements}")


if __name__ == "__main__":
    main()
