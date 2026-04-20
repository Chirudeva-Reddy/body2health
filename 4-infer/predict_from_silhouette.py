"""
Predict BMI and Body Fat % from a precomputed 640x480 silhouette mask.

Usage:
  # front-only (duplicates front as side)
  PYTHONPATH=. python 4-infer/predict_from_silhouette.py --mask "out/front_final.png"

  # two-view (front + side)
  PYTHONPATH=. python 4-infer/predict_from_silhouette.py --front "out/front_final.png" --side "out/IMG_4577_final_silhouette.png"

Optional:
  --ckpt checkpoints/best_640x480.pt
"""

import argparse
from pathlib import Path
import sys

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.model.contrastive_dualview import DualViewContrastive


def _load_mask(path: str, target_hw=(640, 480)) -> torch.Tensor:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise SystemExit(f"Could not read mask: {path}")
    h, w = int(target_hw[0]), int(target_hw[1])
    if m.shape != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    m = (m > 0).astype("uint8") * 255
    return torch.from_numpy(m).float().unsqueeze(0).unsqueeze(0) / 255.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/best_640x480.pt")
    parser.add_argument("--mask", type=str, help="Single silhouette path (used as both front and side).")
    parser.add_argument("--front", type=str, help="Front silhouette path.")
    parser.add_argument("--side", type=str, help="Side silhouette path.")
    args = parser.parse_args()

    if args.mask:
        front_path = args.mask
        side_path = args.mask
    else:
        if not args.front:
            raise SystemExit("Provide --mask or --front (and optionally --side).")
        front_path = args.front
        side_path = args.side or args.front

    ckpt = torch.load(args.ckpt, map_location="cpu")
    convit_hw = tuple(ckpt.get("input_size", (640, 480)))
    model = DualViewContrastive(
        out_meas=ckpt.get("meas_mean", torch.zeros(1)).shape[0],
        proj_dim=128,
        use_large=ckpt.get("use_large", False),
        base_dim=ckpt.get("base_dim", 80),
        use_bbox_features=ckpt.get("use_bbox_features", False),
        encoder=ckpt.get("encoder", "cnn"),
        convit_patch_size=ckpt.get("convit_patch_size", 16),
        convit_dim=ckpt.get("convit_dim", 256),
        convit_depth=ckpt.get("convit_depth", 6),
        convit_heads=ckpt.get("convit_heads", 4),
        convit_mlp_dim=ckpt.get("convit_mlp_dim", 512),
        convit_drop=ckpt.get("convit_drop", 0.0),
        convit_pool=ckpt.get("convit_pool", "mean"),
        convit_gpsa_layers=ckpt.get("convit_gpsa_layers", 2),
        convit_shared=ckpt.get("convit_shared", True),
        convit_img_hw=(int(convit_hw[0]), int(convit_hw[1])),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    front = _load_mask(front_path, target_hw=(int(convit_hw[0]), int(convit_hw[1])))
    side = _load_mask(side_path, target_hw=(int(convit_hw[0]), int(convit_hw[1])))

    with torch.no_grad():
        out = model(front, side)
        bmi = out["meas"][0, 0].item()
        bf_pct = out["bf"][0, 0].item() * 100.0

    print(f"Front: {Path(front_path)}")
    print(f"Side:  {Path(side_path)}")
    print(f"CKPT:  {Path(args.ckpt)}")
    print("")
    print(f"BMI: {bmi:.2f}")
    print(f"Body Fat %: {bf_pct:.2f}")


if __name__ == "__main__":
    main()
