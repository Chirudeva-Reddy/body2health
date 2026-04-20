"""
Evaluate a trained 640x480 contrastive model on an iPhone image or precomputed silhouette.
"""

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.model.contrastive_dualview import DualViewContrastive
from src.preprocess.iphone_pipeline import process_iphone_image, validate_bodym_compatibility


def load_model(model_path: str, device: str = "cuda"):
    checkpoint = torch.load(model_path, map_location=device)
    convit_hw = tuple(checkpoint.get("input_size", (640, 480)))
    model = DualViewContrastive(
        out_meas=checkpoint.get("meas_mean", torch.zeros(1)).shape[0],
        proj_dim=128,
        use_large=checkpoint.get("use_large", False),
        base_dim=checkpoint.get("base_dim", 80),
        use_bbox_features=checkpoint.get("use_bbox_features", False),
        encoder=checkpoint.get("encoder", "cnn"),
        convit_patch_size=checkpoint.get("convit_patch_size", 16),
        convit_dim=checkpoint.get("convit_dim", 256),
        convit_depth=checkpoint.get("convit_depth", 6),
        convit_heads=checkpoint.get("convit_heads", 4),
        convit_mlp_dim=checkpoint.get("convit_mlp_dim", 512),
        convit_drop=checkpoint.get("convit_drop", 0.0),
        convit_pool=checkpoint.get("convit_pool", "mean"),
        convit_gpsa_layers=checkpoint.get("convit_gpsa_layers", 2),
        convit_shared=checkpoint.get("convit_shared", True),
        convit_img_hw=(int(convit_hw[0]), int(convit_hw[1])),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, checkpoint


def predict(silhouette: np.ndarray, model, ckpt, device: str = "cuda"):
    tensor = torch.from_numpy(silhouette).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0

    with torch.no_grad():
        out = model(tensor, tensor)
        # Model was trained on raw BMI targets and BF fractions; no denorm needed.
        meas_pred = out["meas"]
        bf_pred = out["bf"]
    bmi = meas_pred[0, 0].item()
    bf_pct = bf_pred[0, 0].item() * 100.0
    return bmi, bf_pct, meas_pred[0].cpu().numpy().tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, help="Path to raw iPhone image")
    parser.add_argument("--silhouette_path", type=str, help="Path to preprocessed silhouette")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, ckpt = load_model(args.model_path, args.device)

    if args.image_path:
        img_bgr = cv2.imread(args.image_path)
        if img_bgr is None:
            raise SystemExit(f"Could not read image: {args.image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        sil = process_iphone_image(img_rgb)
    elif args.silhouette_path:
        sil = cv2.imread(args.silhouette_path, cv2.IMREAD_GRAYSCALE)
        if sil is None:
            raise SystemExit(f"Could not read silhouette: {args.silhouette_path}")
    else:
        raise SystemExit("Provide --image_path or --silhouette_path")

    validation = validate_bodym_compatibility(sil)
    if not validation["compatible"]:
        print(f"Warning: silhouette validation failed: {validation['issues']}")

    bmi, bf_pct, meas = predict(sil, model, ckpt, device=args.device)
    print("\nPrediction Results")
    print(f"BMI: {bmi:.2f}")
    print(f"Body Fat %: {bf_pct:.2f}")
    print(f"Measurements: {meas}")


if __name__ == "__main__":
    main()
