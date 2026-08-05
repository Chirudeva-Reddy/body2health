"""Deterministic camera alignment search for NLF-initialized SMPL-X meshes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from src.smplx_fit.metrics import fit_metrics, mask_bbox
from src.smplx_fit.render import ProjectionParams, project_vertices_to_pixels, render_silhouette


BBox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class AlignmentResult:
    params: ProjectionParams
    search_metrics: Dict[str, float]


def align_projection_to_mask(
    vertices_m: np.ndarray,
    faces: np.ndarray,
    target_mask: np.ndarray,
    initial_params: ProjectionParams,
) -> AlignmentResult:
    target_bbox = mask_bbox(target_mask)
    best_params = _center_on_target(vertices_m, initial_params, target_bbox)
    best_metrics = _evaluate(vertices_m, faces, target_mask, best_params)

    for params in _coarse_candidates(vertices_m, initial_params, target_bbox):
        metrics = _evaluate(vertices_m, faces, target_mask, params)
        if metrics["score"] < best_metrics["score"]:
            best_params = params
            best_metrics = metrics

    for params in _refined_candidates(vertices_m, best_params, target_bbox):
        metrics = _evaluate(vertices_m, faces, target_mask, params)
        if metrics["score"] < best_metrics["score"]:
            best_params = params
            best_metrics = metrics

    return AlignmentResult(params=best_params, search_metrics=best_metrics)


def projection_params_to_dict(params: ProjectionParams) -> Dict[str, float]:
    return {
        "scale_px_per_m": params.scale_px_per_m,
        "x_scale": params.x_scale,
        "center_x_px": params.center_x_px,
        "center_y_px": params.center_y_px,
        "yaw_deg": params.yaw_deg,
        "roll_deg": params.roll_deg,
    }


def _coarse_candidates(
    vertices_m: np.ndarray,
    initial_params: ProjectionParams,
    target_bbox: BBox,
) -> Tuple[ProjectionParams, ...]:
    candidates: list[ProjectionParams] = []
    for scale_multiplier in (0.96, 1.0, 1.04, 1.08):
        for x_scale in (0.95, 1.0, 1.08, 1.16, 1.24):
            for yaw_deg in (-4.0, 0.0, 4.0):
                params = ProjectionParams(
                    scale_px_per_m=initial_params.scale_px_per_m * scale_multiplier,
                    x_scale=x_scale,
                    center_x_px=initial_params.center_x_px,
                    center_y_px=initial_params.center_y_px,
                    yaw_deg=yaw_deg,
                    roll_deg=0.0,
                )
                candidates.append(_center_on_target(vertices_m, params, target_bbox))
    return tuple(candidates)


def _refined_candidates(
    vertices_m: np.ndarray,
    best_params: ProjectionParams,
    target_bbox: BBox,
) -> Tuple[ProjectionParams, ...]:
    candidates: list[ProjectionParams] = []
    for scale_multiplier in (1.0,):
        for yaw_delta in (-1.0, 0.0, 1.0):
            for roll_delta in (0.0,):
                centered = ProjectionParams(
                    scale_px_per_m=best_params.scale_px_per_m * scale_multiplier,
                    x_scale=best_params.x_scale,
                    center_x_px=best_params.center_x_px,
                    center_y_px=best_params.center_y_px,
                    yaw_deg=best_params.yaw_deg + yaw_delta,
                    roll_deg=best_params.roll_deg + roll_delta,
                )
                recentered = _center_on_target(vertices_m, centered, target_bbox)
                for x_offset in (-6.0, 0.0, 6.0):
                    for y_offset in (-6.0, 0.0, 6.0):
                        candidates.append(
                            ProjectionParams(
                                scale_px_per_m=recentered.scale_px_per_m,
                                x_scale=recentered.x_scale,
                                center_x_px=recentered.center_x_px + x_offset,
                                center_y_px=recentered.center_y_px + y_offset,
                                yaw_deg=recentered.yaw_deg,
                                roll_deg=recentered.roll_deg,
                            )
                        )
    return tuple(candidates)


def _center_on_target(
    vertices_m: np.ndarray,
    params: ProjectionParams,
    target_bbox: BBox,
) -> ProjectionParams:
    projected, _ = project_vertices_to_pixels(vertices_m, params)
    target_center_x = float(target_bbox[0] + target_bbox[2]) / 2.0
    target_center_y = float(target_bbox[1] + target_bbox[3]) / 2.0
    projected_center_x = float(np.min(projected[:, 0]) + np.max(projected[:, 0])) / 2.0
    projected_center_y = float(np.min(projected[:, 1]) + np.max(projected[:, 1])) / 2.0
    return ProjectionParams(
        scale_px_per_m=params.scale_px_per_m,
        x_scale=params.x_scale,
        center_x_px=params.center_x_px + (target_center_x - projected_center_x),
        center_y_px=params.center_y_px + (target_center_y - projected_center_y),
        yaw_deg=params.yaw_deg,
        roll_deg=params.roll_deg,
    )


def _evaluate(
    vertices_m: np.ndarray,
    faces: np.ndarray,
    target_mask: np.ndarray,
    params: ProjectionParams,
) -> Dict[str, float]:
    rendered = render_silhouette(vertices_m, faces, target_mask.shape, params, "opencv")
    return fit_metrics(target_mask, rendered)
