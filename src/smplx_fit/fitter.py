"""Front-view SMPL-X estimation via NLF + render-back reliability gate.

The fit itself is feed-forward: NLF predicts SMPL-X pose+betas+translation
from one RGB image. The mesh is then rendered into 2D and IoU-compared
against the segmented silhouette as an abstention signal. Tape-measure
extraction from the mesh lives elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import trimesh

from src.smplx_fit.alignment import align_projection_to_mask, projection_params_to_dict
from src.smplx_fit.metrics import fit_metrics, largest_component, mask_band, mask_bbox
from src.smplx_fit.nlf_fit import DEFAULT_NLF_MODEL_PATH, predict_smplx_from_rgb
from src.smplx_fit.render import initial_projection_params, render_silhouette


@dataclass(frozen=True)
class SmplxFitConfig:
    smplx_model_path: str
    nlf_model_path: str
    min_iou: float
    max_chamfer: float
    min_shoulder_iou: float
    max_shoulder_chamfer: float
    min_torso_iou: float
    min_hip_iou: float
    min_lower_body_iou: float
    max_bbox_center_distance_px: float
    max_shoulder_row_abs_error_px: float
    min_bbox_width_ratio: float
    max_bbox_width_ratio: float
    min_bbox_height_ratio: float
    max_bbox_height_ratio: float
    max_shoulder_residual_area: float
    render_backend: str
    align_camera: bool


@dataclass(frozen=True)
class SmplxFitResult:
    accepted: bool
    reasons: List[str]
    score: float
    metrics: Dict[str, float]
    vertices_m: np.ndarray
    faces: np.ndarray
    rendered_front: np.ndarray
    betas: np.ndarray
    global_orient: np.ndarray
    translation: np.ndarray
    body_pose: np.ndarray
    smplx_model_path: str
    alignment: Dict[str, float]


def default_smplx_fit_config(
    smplx_model_path: str,
    nlf_model_path: str,
) -> SmplxFitConfig:
    return SmplxFitConfig(
        smplx_model_path=smplx_model_path,
        nlf_model_path=nlf_model_path,
        min_iou=0.55,
        max_chamfer=0.05,
        min_shoulder_iou=0.40,
        max_shoulder_chamfer=0.08,
        min_torso_iou=0.78,
        min_hip_iou=0.68,
        min_lower_body_iou=0.70,
        max_bbox_center_distance_px=10.0,
        max_shoulder_row_abs_error_px=18.0,
        min_bbox_width_ratio=0.90,
        max_bbox_width_ratio=1.10,
        min_bbox_height_ratio=0.95,
        max_bbox_height_ratio=1.05,
        max_shoulder_residual_area=0.14,
        render_backend="opencv",
        align_camera=True,
    )


def fit_smplx_to_rgb(
    front_rgb: np.ndarray,
    front_mask: np.ndarray,
    config: SmplxFitConfig,
) -> SmplxFitResult:
    _validate_rgb_image(front_rgb)
    _validate_mask(front_mask)
    prediction = predict_smplx_from_rgb(
        rgb_image=front_rgb,
        smplx_model_path=config.smplx_model_path,
        nlf_model_path=config.nlf_model_path,
    )
    front = largest_component(front_mask)
    initial_params = initial_projection_params(prediction.vertices_m, mask_bbox(front))
    aligned = align_projection_to_mask(
        vertices_m=prediction.vertices_m,
        faces=prediction.faces,
        target_mask=front,
        initial_params=initial_params,
    )
    render_params = aligned.params if config.align_camera else initial_params
    rendered_front = render_silhouette(
        prediction.vertices_m,
        prediction.faces,
        front.shape,
        render_params,
        config.render_backend,
    )
    metrics = fit_metrics(front, rendered_front)
    reasons = _failure_reasons(metrics, config)
    return SmplxFitResult(
        accepted=len(reasons) == 0,
        reasons=reasons,
        score=metrics["score"],
        metrics=metrics,
        vertices_m=prediction.vertices_m,
        faces=prediction.faces,
        rendered_front=rendered_front,
        betas=prediction.betas,
        global_orient=prediction.pose_rotvecs[:3],
        translation=prediction.translation,
        body_pose=prediction.pose_rotvecs[3:22 * 3],
        smplx_model_path=config.smplx_model_path,
        alignment=projection_params_to_dict(render_params),
    )


def save_smplx_fit_outputs(
    result: SmplxFitResult,
    front_mask: np.ndarray,
    output_dir: str,
) -> Dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "obj": root / "smplx_fit.obj",
        "rendered_front": root / "smplx_rendered_front.png",
        "front_overlay": root / "smplx_front_overlay.png",
        "front_overlay_bboxes": root / "smplx_front_overlay_bboxes.png",
        "front_shoulder_overlay": root / "smplx_front_shoulder_overlay.png",
        "alignment_diagnostics": root / "smplx_alignment_diagnostics.json",
    }
    mesh = trimesh.Trimesh(vertices=result.vertices_m, faces=result.faces, process=False)
    mesh.export(paths["obj"])
    _write_image(paths["rendered_front"], result.rendered_front)
    _write_image(paths["front_overlay"], _overlay(front_mask, result.rendered_front))
    _write_image(paths["front_overlay_bboxes"], _overlay_with_bboxes(front_mask, result.rendered_front))
    _write_image(
        paths["front_shoulder_overlay"],
        _shoulder_overlay(front_mask, result.rendered_front),
    )
    paths["alignment_diagnostics"].write_text(
        json.dumps(_alignment_diagnostics(result), indent=2, sort_keys=True)
    )
    return {name: str(path) for name, path in paths.items()}


def _validate_rgb_image(rgb_image: np.ndarray) -> None:
    if rgb_image.size == 0:
        raise ValueError("front_rgb is empty")
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError(f"front_rgb must be (H, W, 3), got {rgb_image.shape!r}")


def _validate_mask(mask: np.ndarray) -> None:
    if mask.size == 0:
        raise ValueError("front_mask is empty")
    if mask.ndim != 2:
        raise ValueError(f"front_mask must be (H, W), got {mask.shape!r}")


def _failure_reasons(metrics: Dict[str, float], config: SmplxFitConfig) -> List[str]:
    reasons: List[str] = []
    if metrics["front_iou"] < config.min_iou:
        reasons.append("front_render_mismatch")
    if metrics["front_chamfer"] > config.max_chamfer:
        reasons.append("front_contour_mismatch")
    if metrics["shoulder_iou"] < config.min_shoulder_iou:
        reasons.append("shoulder_render_mismatch")
    if metrics["shoulder_chamfer"] > config.max_shoulder_chamfer:
        reasons.append("shoulder_contour_mismatch")
    if metrics["torso_iou"] < config.min_torso_iou:
        reasons.append("torso_render_mismatch")
    if metrics["hip_iou"] < config.min_hip_iou:
        reasons.append("hip_render_mismatch")
    if metrics["lower_body_iou"] < config.min_lower_body_iou:
        reasons.append("lower_body_render_mismatch")
    if metrics["bbox_center_distance_px"] > config.max_bbox_center_distance_px:
        reasons.append("bbox_center_mismatch")
    if metrics["shoulder_row_abs_error_px"] > config.max_shoulder_row_abs_error_px:
        reasons.append("shoulder_height_mismatch")
    if (
        metrics["bbox_width_ratio"] < config.min_bbox_width_ratio
        or metrics["bbox_width_ratio"] > config.max_bbox_width_ratio
    ):
        reasons.append("bbox_width_mismatch")
    input_touches_vertical_border = bool(metrics["input_touches_top"] or metrics["input_touches_bottom"])
    if not input_touches_vertical_border and (
        metrics["bbox_height_ratio"] < config.min_bbox_height_ratio
        or metrics["bbox_height_ratio"] > config.max_bbox_height_ratio
    ):
        reasons.append("bbox_height_mismatch")
    max_residual = max(metrics["left_shoulder_residual_area"], metrics["right_shoulder_residual_area"])
    if max_residual > config.max_shoulder_residual_area:
        reasons.append("local_pose_mismatch")
    return reasons


def _overlay(input_mask: np.ndarray, rendered_mask: np.ndarray) -> np.ndarray:
    base = np.zeros((input_mask.shape[0], input_mask.shape[1], 3), dtype=np.uint8)
    input_binary = input_mask > 0
    render_binary = rendered_mask > 0
    base[input_binary, 1] = 180
    base[render_binary, 2] = 220
    base[np.logical_and(input_binary, render_binary)] = (230, 230, 230)
    return base


def _overlay_with_bboxes(input_mask: np.ndarray, rendered_mask: np.ndarray) -> np.ndarray:
    overlay = _overlay(input_mask, rendered_mask)
    input_bbox = mask_bbox(input_mask)
    rendered_bbox = mask_bbox(rendered_mask)
    _draw_bbox(overlay, input_bbox, (0, 220, 0))
    _draw_bbox(overlay, rendered_bbox, (0, 0, 220))
    _draw_center(overlay, input_bbox, (0, 220, 0))
    _draw_center(overlay, rendered_bbox, (0, 0, 220))
    cv2.putText(
        overlay,
        "green=input mask  red=SMPL-X render  white=overlap",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return overlay


def _shoulder_overlay(input_mask: np.ndarray, rendered_mask: np.ndarray) -> np.ndarray:
    band = mask_band(input_mask, 0.14, 0.34)
    return _overlay(input_mask, rendered_mask)[band]


def _draw_bbox(image: np.ndarray, bbox: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)


def _draw_center(image: np.ndarray, bbox: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    center_x = int(round(float(bbox[0] + bbox[2]) / 2.0))
    center_y = int(round(float(bbox[1] + bbox[3]) / 2.0))
    cv2.drawMarker(
        image,
        (center_x, center_y),
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=14,
        thickness=2,
    )


def _alignment_diagnostics(result: SmplxFitResult) -> Dict[str, object]:
    return {
        "accepted": result.accepted,
        "reasons": result.reasons,
        "score": result.score,
        "alignment": result.alignment,
        "metrics": result.metrics,
        "legend": {
            "green": "SAM2/input silhouette only",
            "red": "SMPL-X rendered silhouette only",
            "white": "overlap between input and render",
        },
    }


def _write_image(path: Path, image: np.ndarray) -> None:
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise OSError(f"failed to write image: {path}")
