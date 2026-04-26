"""Predict BMI and Body Fat % from one or two precomputed silhouette masks.

Usage:
  # Single mask used as both front and side:
  PYTHONPATH=. python 4-infer/1infer.py --mask out/front_final.png --ckpt checkpoints/latest.pt

  # Front + side:
  PYTHONPATH=. python 4-infer/1infer.py \\
      --front out/front_final.png --side out/IMG_4577_final_silhouette.png \\
      --ckpt checkpoints/latest.pt
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.infer.masks import load_mask_binary, mask_to_tensor
from src.infer.model_io import load_dualview_checkpoint
from src.infer.predict import predict_from_pair
from src.infer.silhouette_checks import envelope_check

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True, help="Path to the trained checkpoint (.pt).")
    parser.add_argument("--mask", help="Single silhouette path (used as both front and side).")
    parser.add_argument("--front", help="Front silhouette path.")
    parser.add_argument("--side", help="Side silhouette path. Defaults to the front mask if omitted.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--skip-envelope-check",
        action="store_true",
        help="Skip the silhouette envelope sanity check (still warns by default).",
    )
    return parser.parse_args()


def _resolve_pair(args: argparse.Namespace) -> tuple[str, str]:
    if args.mask:
        return args.mask, args.mask
    if not args.front:
        raise SystemExit("provide --mask or --front (and optionally --side)")
    return args.front, args.side or args.front


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    front_path, side_path = _resolve_pair(args)
    model, ckpt = load_dualview_checkpoint(args.ckpt, args.device)
    target_hw = tuple(int(x) for x in ckpt["input_size"])

    front_np = load_mask_binary(front_path, target_hw)
    side_np = load_mask_binary(side_path, target_hw)

    if not args.skip_envelope_check:
        for label, mask in (("front", front_np), ("side", side_np)):
            report = envelope_check(mask)
            if not report.ok:
                logger.warning("%s silhouette failed envelope checks: %s", label, ", ".join(report.failed))

    front_t = mask_to_tensor(front_np, args.device)
    side_t = mask_to_tensor(side_np, args.device)

    result = predict_from_pair(model, ckpt, front_t, side_t)

    print(f"Front: {Path(front_path)}")
    print(f"Side:  {Path(side_path)}")
    print(f"CKPT:  {Path(args.ckpt)}")
    print()
    print(f"BMI: {result.bmi:.2f}")
    print(f"Body Fat %: {result.bf_pct:.2f}")


if __name__ == "__main__":
    main()
