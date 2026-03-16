"""
Refine noisy binary silhouettes by fitting SMPL and re-rendering canonical masks.

Usage:
python scripts/refine_silhouettes_with_smpl.py --front_mask <path> --side_mask <path> --height_cm 175 --out_dir out/refined
"""
import argparse
import os
import pathlib
import sys
import cv2

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.smpl.fitter import fit_smpl_to_silhouettes
from src.render.canonical import render_canonical_silhouettes
from src.utils.io import save_png, load_yaml


def _read_mask(path: str):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return ((m > 127).astype("uint8")) * 255


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--front_mask", type=str, required=True)
    parser.add_argument("--side_mask", type=str, required=True)
    parser.add_argument("--height_cm", type=float, required=True)
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--out_dir", type=str, default=str(ROOT / "out" / "refined"))
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    render_cfg = cfg.get("render", {})
    fx = float(render_cfg.get("fx", 900))
    fy = float(render_cfg.get("fy", 900))
    cx = float(render_cfg.get("cx", 128))
    cy = float(render_cfg.get("cy", 128))
    out_hw = tuple(render_cfg.get("out_size", [256, 128]))

    f_mask = _read_mask(args.front_mask)
    s_mask = _read_mask(args.side_mask)

    fit_cfg = cfg.get("smpl_fit", {})
    out = fit_smpl_to_silhouettes(
        f_mask,
        s_mask,
        height_cm=float(args.height_cm),
        iters=int(fit_cfg.get("iters", 400)),
        lr=float(fit_cfg.get("lr", 0.01)),
    )

    sils = render_canonical_silhouettes(out["verts"], out["faces"], fx, fy, cx, cy, out_size=out_hw)

    os.makedirs(args.out_dir, exist_ok=True)
    front_out = os.path.join(args.out_dir, "front_refined.png")
    side_out = os.path.join(args.out_dir, "side_refined.png")
    save_png(front_out, sils["front"])
    save_png(side_out, sils["side"])
    print(f"Saved refined silhouettes:\n{front_out}\n{side_out}")


if __name__ == "__main__":
    main()
