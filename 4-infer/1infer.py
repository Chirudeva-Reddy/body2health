"""Predict body dimensions and derived indices from RGB photos or silhouette masks.

Usage:
  # Front + side RGB photos:
  PYTHONPATH=. python 4-infer/1infer.py \\
      --front_rgb TestPhoto/deva_front.png --side_rgb TestPhoto/deva_side.png \\
      --ckpt checkpoints/best_640x480_v4_resnet.pt --height_cm 175 \\
      --sex male --smplx_fit --save_silhouettes outputs/final/deva \\
      --save_smplx outputs/final/deva/smplx --json outputs/final/deva/result.json

  # Single front RGB photo, diagnostic mode:
  PYTHONPATH=. python 4-infer/1infer.py \\
      --front_rgb TestPhoto/deva_front.png --single_front \\
      --ckpt checkpoints/best_640x480_v4_resnet.pt --height_cm 175 \\
      --sex male --smplx_fit --save_silhouettes outputs/final/deva_single \\
      --save_smplx outputs/final/deva_single/smplx --json outputs/final/deva_single/result.json

  # Front + side masks:
  PYTHONPATH=. python 4-infer/1infer.py \\
      --front out/front_final.png --side out/IMG_4577_final_silhouette.png \\
      --ckpt checkpoints/best_640x480_v4_resnet.pt --height_cm 175

  # Cached masks + front RGB through NLF SMPL-X reliability:
  PYTHONPATH=. python 4-infer/1infer.py \\
      --front out/deva_front_silhouette.png --side out/deva_side_silhouette.png \\
      --front_rgb TestPhoto/deva_front.png \\
      --ckpt checkpoints/best_640x480_v4_resnet.pt --height_cm 175 \\
      --sex male --smplx_fit --save_smplx outputs/final/deva_cached/smplx \\
      --json outputs/final/deva_cached/result.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from pipeline.iphone_pipeline import process_iphone_image
from src.infer.masks import load_mask_binary, mask_to_tensor
from src.infer.model_io import load_dualview_checkpoint
from src.infer.predict import predict_from_pair
from src.infer.silhouette_checks import envelope_check
from src.infer.silhouette_regions import save_silhouette_region_artifacts
from src.metrics.health_risk import assess_health_risk
from src.smpl.gate import default_gate_thresholds, evaluate_smplx_gate, export_measurement_proxy_obj

logger = logging.getLogger(__name__)
DEFAULT_NLF_MODEL_PATH = "models/nlf/nlf_l_multi.torchscript"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True, help="Path to the trained checkpoint (.pt).")
    parser.add_argument("--mask", help="Single silhouette path (used as both front and side).")
    parser.add_argument("--front", help="Front silhouette path.")
    parser.add_argument("--side", help="Side silhouette path. Defaults to the front mask if omitted.")
    parser.add_argument("--front_rgb", help="Front RGB image path. Segmented before inference.")
    parser.add_argument("--side_rgb", help="Side RGB image path. Segmented before inference.")
    parser.add_argument(
        "--single_front",
        action="store_true",
        help="Run a front-only diagnostic demo by duplicating the front silhouette for the side branch.",
    )
    parser.add_argument("--sam_model_path", help="SAM2 checkpoint path for RGB segmentation.")
    parser.add_argument("--sam_config_path", help="SAM2 config path for RGB segmentation.")
    parser.add_argument("--yolo_model_path", help="YOLO person detector checkpoint path for RGB segmentation.")
    parser.add_argument("--save_silhouettes", help="Directory where generated RGB silhouettes should be saved.")
    parser.add_argument("--height_cm", type=float, help="Height in cm, required for WHtR and BRI.")
    parser.add_argument("--sex", help="Sex used only for waist/WHR risk thresholds: male or female.")
    parser.add_argument("--json", help="Write the prediction result to this JSON path.")
    parser.add_argument("--smpl_gate", action="store_true", help="Run the legacy lightweight SMPLX proxy gate.")
    parser.add_argument("--smpl_model_dir", default="models/smplx", help="Directory containing SMPLX_NEUTRAL.npz.")
    parser.add_argument("--gate_threshold", type=float, help="Override the default maximum reliability score.")
    parser.add_argument("--save_gate_render", help="Directory where gate render masks and overlays should be saved.")
    parser.add_argument("--smplx_fit", action="store_true", help="Run NLF SMPL-X fit and render-back reliability gate.")
    parser.add_argument("--save_smplx", help="Directory where SMPL-X mesh, renders, and overlays should be saved.")
    parser.add_argument("--nlf_model_path", default=DEFAULT_NLF_MODEL_PATH, help="NLF TorchScript model path.")
    parser.add_argument("--smplx_model_path", help="SMPL-X .npz model path. Defaults to sex-specific model when --sex is male/female, otherwise neutral.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--skip-envelope-check",
        action="store_true",
        help="Skip the silhouette envelope sanity check (still warns by default).",
    )
    return parser.parse_args()


def _resolve_mask_pair(args: argparse.Namespace) -> tuple[str, str]:
    if args.mask:
        return args.mask, args.mask
    if not args.front:
        raise SystemExit("provide --mask or --front (and optionally --side)")
    return args.front, args.side or args.front


def _read_rgb(path: str) -> np.ndarray:
    image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"could not read RGB image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _save_mask(mask: np.ndarray, output_dir: str, filename: str) -> str:
    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), mask)
    if not ok:
        raise OSError(f"failed to write silhouette: {path}")
    return str(path)


def _ensure_target_hw(mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    h, w = int(target_hw[0]), int(target_hw[1])
    if mask.shape == (h, w):
        return mask
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def _save_gate_visuals(
    front_mask: np.ndarray,
    side_mask: np.ndarray,
    rendered_front: np.ndarray,
    rendered_side: np.ndarray,
    output_dir: str,
) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "rendered_front": root / "gate_rendered_front.png",
        "rendered_side": root / "gate_rendered_side.png",
        "front_overlay": root / "gate_front_overlay.png",
        "side_overlay": root / "gate_side_overlay.png",
    }
    cv2.imwrite(str(paths["rendered_front"]), rendered_front)
    cv2.imwrite(str(paths["rendered_side"]), rendered_side)
    cv2.imwrite(str(paths["front_overlay"]), _overlay_masks(front_mask, rendered_front))
    cv2.imwrite(str(paths["side_overlay"]), _overlay_masks(side_mask, rendered_side))
    return {key: str(path) for key, path in paths.items()}


def _overlay_masks(input_mask: np.ndarray, rendered_mask: np.ndarray) -> np.ndarray:
    base = np.zeros((input_mask.shape[0], input_mask.shape[1], 3), dtype=np.uint8)
    input_bin = input_mask > 0
    render_bin = rendered_mask > 0
    base[input_bin, 1] = 180
    base[render_bin, 2] = 220
    base[np.logical_and(input_bin, render_bin)] = (230, 230, 230)
    return base


def _segment_rgb_pair(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, tuple[str, str]]:
    if not args.front_rgb:
        raise SystemExit("provide --front_rgb with optional --side_rgb, or use --front/--side masks")
    if args.single_front and args.side_rgb:
        raise SystemExit("--single_front cannot be combined with --side_rgb")
    if not args.single_front and not args.side_rgb:
        raise SystemExit("provide --side_rgb for dual-view RGB inference, or pass --single_front for diagnostic front-only mode")
    front_rgb = _read_rgb(args.front_rgb)
    front_mask = process_iphone_image(
        front_rgb,
        sam_model_path=args.sam_model_path,
        sam_config_path=args.sam_config_path,
        yolo_model_path=args.yolo_model_path,
        view="front",
        debug_dir=None,
        debug_prefix="front",
    )
    side_mask = process_iphone_image(
        _read_rgb(args.side_rgb),
        sam_model_path=args.sam_model_path,
        sam_config_path=args.sam_config_path,
        yolo_model_path=args.yolo_model_path,
        view="side",
        debug_dir=None,
        debug_prefix="side",
    )
    front_label = args.front_rgb
    side_label = args.side_rgb
    if args.save_silhouettes:
        front_label = _save_mask(front_mask, args.save_silhouettes, "front_silhouette.png")
        side_label = _save_mask(side_mask, args.save_silhouettes, "side_silhouette.png")
    _validate_rgb_mask_pair(front_mask, side_mask)
    return front_mask, side_mask, (front_label, side_label)


def _segment_single_front(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, tuple[str, str]]:
    if not args.front_rgb:
        raise SystemExit("--single_front requires --front_rgb")
    if args.side_rgb:
        raise SystemExit("--single_front cannot be combined with --side_rgb")
    front_rgb = _read_rgb(args.front_rgb)
    front_mask = process_iphone_image(
        front_rgb,
        sam_model_path=args.sam_model_path,
        sam_config_path=args.sam_config_path,
        yolo_model_path=args.yolo_model_path,
        view="front",
        debug_dir=None,
        debug_prefix="front",
    )
    side_mask = front_mask.copy()
    front_label = args.front_rgb
    side_label = args.front_rgb
    if args.save_silhouettes:
        front_label = _save_mask(front_mask, args.save_silhouettes, "front_silhouette.png")
        side_label = _save_mask(side_mask, args.save_silhouettes, "side_silhouette_from_front.png")
    return front_mask, side_mask, (front_label, side_label)


def _validate_rgb_mask_pair(front_mask: np.ndarray, side_mask: np.ndarray) -> None:
    front_ratio = float((front_mask > 0).mean())
    side_ratio = float((side_mask > 0).mean())
    if front_ratio > 0.42:
        raise ValueError(f"front_foreground_too_large: ratio={front_ratio:.4f}")
    if side_ratio > 0.36:
        raise ValueError(f"side_foreground_too_large: ratio={side_ratio:.4f}")
    if side_ratio <= 0.0 or front_ratio <= 0.0:
        raise ValueError(f"empty_rgb_silhouette_pair: front={front_ratio:.4f} side={side_ratio:.4f}")
    side_front_ratio = side_ratio / front_ratio
    if side_front_ratio < 0.25:
        raise ValueError(f"side_front_area_ratio_too_small: ratio={side_front_ratio:.4f}")
    if side_front_ratio > 1.05:
        raise ValueError(f"side_front_area_ratio_too_large: ratio={side_front_ratio:.4f}")


def _load_inputs(args: argparse.Namespace, target_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, tuple[str, str]]:
    uses_rgb = bool(args.front_rgb or args.side_rgb)
    uses_mask = bool(args.mask or args.front or args.side)
    if uses_rgb and uses_mask:
        if args.smplx_fit and args.front_rgb and not args.side_rgb:
            front_path, side_path = _resolve_mask_pair(args)
            return load_mask_binary(front_path, target_hw), load_mask_binary(side_path, target_hw), (front_path, side_path)
        raise SystemExit(
            "choose either RGB inputs (--front_rgb/--side_rgb) or mask inputs (--mask or --front/--side), "
            "except --smplx_fit may combine --front_rgb with cached masks for NLF"
        )
    if uses_rgb:
        if args.single_front:
            front_mask, side_mask, labels = _segment_single_front(args)
            return _ensure_target_hw(front_mask, target_hw), _ensure_target_hw(side_mask, target_hw), labels
        front_mask, side_mask, labels = _segment_rgb_pair(args)
        return _ensure_target_hw(front_mask, target_hw), _ensure_target_hw(side_mask, target_hw), labels
    if args.single_front:
        raise SystemExit("--single_front is only valid with --front_rgb")
    front_path, side_path = _resolve_mask_pair(args)
    return load_mask_binary(front_path, target_hw), load_mask_binary(side_path, target_hw), (front_path, side_path)


def _reportable_status(
    gate_payload: dict[str, object] | None,
    smplx_payload: dict[str, object] | None,
) -> bool | None:
    if smplx_payload is not None:
        return bool(smplx_payload["accepted"])
    if gate_payload is not None:
        return bool(gate_payload["accepted"])
    return None


def _resolve_smplx_model_path(explicit_path: str | None, sex: str | None) -> str:
    if explicit_path:
        return explicit_path
    normalized = "" if sex is None else sex.strip().lower()
    if normalized == "male":
        return "models/smplx/SMPLX_MALE.npz"
    if normalized == "female":
        return "models/smplx/SMPLX_FEMALE.npz"
    return "models/smplx/SMPLX_NEUTRAL.npz"


def _capture_mode(args: argparse.Namespace) -> str:
    if args.single_front:
        return "single_front_rgb"
    if args.front_rgb and args.side_rgb:
        return "front_side_rgb"
    if args.front_rgb and (args.front or args.mask):
        return "cached_masks_with_front_rgb"
    return "mask_pair"


def _estimate_mode(args: argparse.Namespace) -> str:
    if args.single_front:
        return "single_front_view"
    if args.front_rgb and args.side_rgb:
        return "dual_view"
    if args.front_rgb and (args.front or args.mask):
        return "cached_mask_pair_with_front_rgb"
    return "mask_pair"


def _source_image_paths(args: argparse.Namespace) -> dict[str, str]:
    paths = {}
    if args.front_rgb:
        paths["front"] = args.front_rgb
    if args.side_rgb:
        paths["side"] = args.side_rgb
    return paths


def _output_dir(args: argparse.Namespace) -> str | None:
    if args.json:
        return str(Path(args.json).parent)
    if args.save_silhouettes:
        return args.save_silhouettes
    if args.save_smplx:
        return str(Path(args.save_smplx).parent)
    return None


def _save_region_artifacts(
    args: argparse.Namespace,
    front_mask: np.ndarray,
    side_mask: np.ndarray,
) -> dict[str, dict[str, str]]:
    if not args.save_silhouettes:
        return {}
    return {
        "front": save_silhouette_region_artifacts(front_mask, args.save_silhouettes, "front"),
        "side": save_silhouette_region_artifacts(side_mask, args.save_silhouettes, "side"),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    model, ckpt = load_dualview_checkpoint(args.ckpt, args.device)
    target_hw = tuple(int(x) for x in ckpt["input_size"])
    front_np, side_np, labels = _load_inputs(args, target_hw)
    front_label, side_label = labels
    region_artifacts = _save_region_artifacts(args, front_np, side_np)

    if not args.skip_envelope_check:
        for label, view, mask in (("front", "front", front_np), ("side", "side", side_np)):
            if args.single_front and label == "side":
                continue
            report = envelope_check(mask, view)
            if not report.ok:
                logger.warning("%s silhouette failed envelope checks: %s", label, ", ".join(report.failed))

    front_t = mask_to_tensor(front_np, args.device)
    side_t = mask_to_tensor(side_np, args.device)

    result = predict_from_pair(model, ckpt, front_t, side_t, args.height_cm, args.sex)

    gate_payload = None
    if args.smpl_gate:
        if args.height_cm is None:
            raise SystemExit("--height_cm is required when --smpl_gate is enabled")
        thresholds = default_gate_thresholds()
        if args.gate_threshold is not None:
            thresholds = type(thresholds)(
                max_score=args.gate_threshold,
                max_band_error=thresholds.max_band_error,
                min_render_iou=thresholds.min_render_iou,
                max_chamfer=thresholds.max_chamfer,
                max_front_foreground_ratio=thresholds.max_front_foreground_ratio,
                max_side_foreground_ratio=thresholds.max_side_foreground_ratio,
                min_side_front_area_ratio=thresholds.min_side_front_area_ratio,
                max_side_front_area_ratio=thresholds.max_side_front_area_ratio,
            )
        gate = evaluate_smplx_gate(
            front_np,
            side_np,
            result.measurements,
            args.height_cm,
            args.sex,
            args.smpl_model_dir,
            thresholds,
        )
        visual_paths = {}
        if args.save_gate_render:
            visual_paths = _save_gate_visuals(
                front_np,
                side_np,
                gate.rendered_front,
                gate.rendered_side,
                args.save_gate_render,
            )
            visual_paths["body_proxy_obj"] = export_measurement_proxy_obj(
                str(Path(args.save_gate_render) / "gate_body_proxy.obj"),
                result.measurements,
                args.height_cm,
                args.sex,
                radial_segments=32,
            )
        gate_payload = {
            "accepted": gate.accepted,
            "score": gate.score,
            "reasons": gate.reasons,
            "metrics": gate.metrics,
            "asset_info": {
                "model_dir": gate.asset_info.model_dir,
                "neutral_model_path": gate.asset_info.neutral_model_path,
                "vertex_count": gate.asset_info.vertex_count,
                "face_count": gate.asset_info.face_count,
                "template_height_units": gate.asset_info.template_height_units,
            },
            "visuals": visual_paths,
        }

    smplx_payload = None
    if args.smplx_fit:
        if not args.front_rgb:
            raise SystemExit("--smplx_fit requires --front_rgb (NLF needs the RGB image)")
        from src.smplx_fit import default_smplx_fit_config, fit_smplx_to_rgb, save_smplx_fit_outputs

        output_dir = args.save_smplx or args.save_gate_render or "outputs/smplx_fit"
        config = default_smplx_fit_config(
            smplx_model_path=_resolve_smplx_model_path(args.smplx_model_path, args.sex),
            nlf_model_path=args.nlf_model_path,
        )
        front_rgb_for_nlf = _read_rgb(args.front_rgb)
        fit = fit_smplx_to_rgb(
            front_rgb=front_rgb_for_nlf,
            front_mask=front_np,
            config=config,
        )
        fit_visuals = save_smplx_fit_outputs(
            result=fit,
            front_mask=front_np,
            output_dir=output_dir,
        )
        smplx_payload = {
            "accepted": fit.accepted,
            "score": fit.score,
            "reasons": fit.reasons,
            "metrics": fit.metrics,
            "smplx_model_path": fit.smplx_model_path,
            "betas": fit.betas.tolist(),
            "global_orient": fit.global_orient.tolist(),
            "translation": fit.translation.tolist(),
            "body_pose": fit.body_pose.tolist(),
            "alignment": fit.alignment,
            "visuals": fit_visuals,
        }

    print(f"Front: {front_label}")
    print(f"Side:  {side_label}")
    print(f"CKPT:  {Path(args.ckpt)}")
    print()
    print("Dimensions")
    for name, value in result.measurements.items():
        print(f"{name}: {value:.2f} cm")
    if result.indices:
        print()
        print("Derived indices")
        for name, value in result.indices.items():
            print(f"{name}: {value:.4f}")
    if result.risks:
        print()
        print("Risk categories")
        for name, value in result.risks.items():
            print(f"{name}: {value}")
    if result.invalid_indices:
        print()
        print("Skipped indices")
        for name, reason in result.invalid_indices.items():
            print(f"{name}: {reason}")
    if gate_payload is not None:
        print()
        print("Legacy SMPLX proxy reliability")
        print(f"status: {'ACCEPTED' if gate_payload['accepted'] else 'REJECTED'}")
        print(f"reportable: {'yes' if gate_payload['accepted'] else 'no - recapture recommended'}")
        print(f"score: {gate_payload['score']:.4f}")
        if gate_payload["reasons"]:
            print(f"reasons: {', '.join(gate_payload['reasons'])}")
        for name, value in gate_payload["metrics"].items():
            if name == "score":
                continue
            print(f"{name}: {value:.4f}")
    if smplx_payload is not None:
        print()
        print("SMPL-X fit reliability")
        print(f"status: {'ACCEPTED' if smplx_payload['accepted'] else 'REJECTED'}")
        print(f"reportable: {'yes' if smplx_payload['accepted'] else 'no - recapture recommended'}")
        print(f"score: {smplx_payload['score']:.4f}")
        if smplx_payload["reasons"]:
            print(f"reasons: {', '.join(smplx_payload['reasons'])}")
        for name, value in smplx_payload["metrics"].items():
            if name == "score":
                continue
            print(f"{name}: {value:.4f}")

    gate_reportable = _reportable_status(gate_payload, smplx_payload)
    reportable = bool(gate_reportable) if gate_reportable is not None else True
    health_summary = assess_health_risk(
        result.measurements,
        result.indices,
        result.risks,
        args.sex,
        reportable,
    ).to_dict()
    print()
    print("Central adiposity risk summary")
    print(f"overall_risk: {health_summary['overall_risk']}")
    print(f"primary_driver: {health_summary['primary_driver']}")
    print(f"reportable: {health_summary['reportable']}")
    print(f"message: {health_summary['message']}")

    if args.json:
        output = {
            "capture_mode": _capture_mode(args),
            "estimate_mode": _estimate_mode(args),
            "front": front_label,
            "side": side_label,
            "checkpoint": str(Path(args.ckpt)),
            "height_cm": args.height_cm,
            "sex": args.sex,
            "measurements": result.measurements,
            "indices": result.indices,
            "risks": result.risks,
            "health_summary": health_summary,
            "invalid_indices": result.invalid_indices,
            "segmentation_paths": {
                "front": front_label,
                "side": side_label,
            },
            "silhouette_region_artifacts": region_artifacts,
            "source_image_paths": _source_image_paths(args),
            "output_dir": _output_dir(args),
            "reportable": reportable,
            "smpl_gate": gate_payload,
            "smplx_fit": smplx_payload,
        }
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(output, indent=2, sort_keys=True))
        print()
        print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
