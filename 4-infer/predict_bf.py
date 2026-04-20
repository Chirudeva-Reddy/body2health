import argparse
import pathlib
import sys

import torch
import torch.nn as nn
import numpy as np
import cv2

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.model.contrastive_dualview import DualViewContrastive
from src.train.data import SilhouetteDataset
from src.utils.io import read_image

def load_latest_ckpt(model: nn.Module, ckpt_dir: str):
    ckpt_path = pathlib.Path(ckpt_dir) / "latest.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    meas_mean = ckpt["meas_mean"]
    meas_std = ckpt["meas_std"]
    bf_mean = ckpt["bf_mean"]
    bf_std = ckpt["bf_std"]
    # Return model config flags for consistency
    use_large = ckpt.get("use_large", False)
    base_dim = ckpt.get("base_dim", 64)
    use_bbox_features = ckpt.get("use_bbox_features", False)
    encoder = ckpt.get("encoder", "cnn")
    convit_patch_size = ckpt.get("convit_patch_size", 16)
    convit_dim = ckpt.get("convit_dim", 256)
    convit_depth = ckpt.get("convit_depth", 6)
    convit_heads = ckpt.get("convit_heads", 4)
    convit_mlp_dim = ckpt.get("convit_mlp_dim", 512)
    convit_drop = ckpt.get("convit_drop", 0.0)
    convit_pool = ckpt.get("convit_pool", "mean")
    convit_shared = ckpt.get("convit_shared", True)
    convit_img_hw = tuple(ckpt.get("input_size", (640, 480)))
    return (
        meas_mean,
        meas_std,
        bf_mean,
        bf_std,
        use_large,
        base_dim,
        use_bbox_features,
        encoder,
        convit_patch_size,
        convit_dim,
        convit_depth,
        convit_heads,
        convit_mlp_dim,
        convit_drop,
        convit_pool,
        convit_shared,
        convit_img_hw,
    )

def preprocess_silhouette(img_path, target_hw=(256,128)):
    img = read_image(img_path)
    # Assume binary silhouette; ensure single channel and normalize
    if img.ndim == 3 and img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = target_hw
    resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    arr = (resized.astype(np.float32) / 255.0)[None, None, :, :]  # (1,1,H,W)
    return torch.from_numpy(arr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--front", type=str, required=True, help="Path to front silhouette PNG")
    parser.add_argument("--side", type=str, default=None, help="Optional side silhouette PNG")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load checkpoint to get model config
    ckpt_path = pathlib.Path(args.ckpt_dir) / "latest.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    use_large = ckpt.get("use_large", False)
    base_dim = ckpt.get("base_dim", 64)
    use_bbox_features = ckpt.get("use_bbox_features", False)
    convit_hw = tuple(ckpt.get("input_size", (640, 480)))
    # Instantiate model with loaded config
    model = DualViewContrastive(
        out_meas=1,
        proj_dim=128,
        use_large=use_large,
        base_dim=base_dim,
        use_bbox_features=use_bbox_features,
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
    ).to(args.device)
    meas_mean, meas_std, bf_mean, bf_std = ckpt["meas_mean"], ckpt["meas_std"], ckpt["bf_mean"], ckpt["bf_std"]

    target_hw = (int(convit_hw[0]), int(convit_hw[1]))
    front_tensor = preprocess_silhouette(args.front, target_hw=target_hw).to(args.device)
    if args.side:
        side_tensor = preprocess_silhouette(args.side, target_hw=target_hw).to(args.device)
    else:
        # Duplicate front as side placeholder (model expects both)
        side_tensor = front_tensor

    model.eval()
    with torch.no_grad():
        out = model(front_tensor, side_tensor)

    # De-normalize
    pred_meas = out["meas"] * meas_std + meas_mean
    pred_bf = out["bf"] * bf_std + bf_mean
    # Clip to plausible ranges
    pred_meas = torch.clamp(pred_meas, min=15.0, max=45.0)  # BMI range
    pred_bf = torch.clamp(pred_bf, min=8.0, max=30.0)      # BF% range
    bmi_val = float(pred_meas[0, 0].item())
    bf_val = float(pred_bf[0, 0].item())
    print(f"Predicted BMI: {bmi_val:.2f}")
    print(f"Predicted Body Fat %: {bf_val:.2f}")

if __name__ == "__main__":
    main()
