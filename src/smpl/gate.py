"""Geometry-based reliability gate for front/side anthropometry.

This module intentionally avoids a heavyweight SMPLX runtime dependency. It
validates that SMPLX assets are present, then uses a deterministic
measurement-constrained body geometry proxy to render front/side silhouettes
and score whether the predicted girths are consistent with the observed masks.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from src.infer.silhouette_checks import envelope_check


BandSpec = Tuple[str, float]


@dataclass(frozen=True)
class SmplxAssetInfo:
    model_dir: str
    neutral_model_path: str
    vertex_count: int
    face_count: int
    template_height_units: float


@dataclass(frozen=True)
class GateThresholds:
    max_score: float
    max_band_error: float
    min_render_iou: float
    max_chamfer: float
    max_front_foreground_ratio: float
    max_side_foreground_ratio: float
    min_side_front_area_ratio: float
    max_side_front_area_ratio: float


@dataclass(frozen=True)
class ReliabilityGateResult:
    accepted: bool
    score: float
    reasons: List[str]
    metrics: Dict[str, float]
    rendered_front: np.ndarray
    rendered_side: np.ndarray
    asset_info: SmplxAssetInfo


def default_gate_thresholds() -> GateThresholds:
    return GateThresholds(
        max_score=1.05,
        max_band_error=1.20,
        min_render_iou=0.15,
        max_chamfer=0.15,
        max_front_foreground_ratio=0.30,
        max_side_foreground_ratio=0.25,
        min_side_front_area_ratio=0.25,
        max_side_front_area_ratio=1.20,
    )


@lru_cache(maxsize=8)
def validate_smplx_assets(model_dir: str) -> SmplxAssetInfo:
    root = Path(model_dir)
    neutral_path = root / "SMPLX_NEUTRAL.npz"
    if not neutral_path.exists():
        raise FileNotFoundError(
            f"SMPLX neutral model missing at {neutral_path}. "
            "Place SMPLX_NEUTRAL.npz under models/smplx/."
        )

    data = np.load(neutral_path, allow_pickle=True)
    for key in ("v_template", "f"):
        if key not in data:
            raise KeyError(f"{neutral_path} is missing required SMPLX array {key!r}")
    vertices = np.asarray(data["v_template"], dtype=np.float64)
    faces = np.asarray(data["f"])
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"SMPLX v_template must have shape (N, 3), got {vertices.shape!r}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"SMPLX faces must have shape (F, 3), got {faces.shape!r}")
    template_height = float(np.max(vertices[:, 1]) - np.min(vertices[:, 1]))
    if template_height <= 0.0:
        raise ValueError(f"SMPLX template has non-positive height: {template_height}")
    return SmplxAssetInfo(
        model_dir=str(root),
        neutral_model_path=str(neutral_path),
        vertex_count=int(vertices.shape[0]),
        face_count=int(faces.shape[0]),
        template_height_units=template_height,
    )


def evaluate_smplx_gate(
    front_mask: np.ndarray,
    side_mask: np.ndarray,
    measurements: Dict[str, float],
    height_cm: float,
    sex: str | None,
    model_dir: str,
    thresholds: GateThresholds,
) -> ReliabilityGateResult:
    if height_cm <= 0.0:
        raise ValueError(f"height_cm must be positive for geometry gate, got {height_cm}")
    asset_info = validate_smplx_assets(model_dir)
    front = _largest_component(front_mask)
    side = _largest_component(side_mask)
    front_bbox = _body_bbox(front)
    side_bbox = _body_bbox(side)
    front_px_per_cm = _px_per_cm(front_bbox, height_cm)
    side_px_per_cm = _px_per_cm(side_bbox, height_cm)
    band_errors = _band_errors(
        front,
        side,
        front_bbox,
        side_bbox,
        front_px_per_cm,
        side_px_per_cm,
        measurements,
    )
    rendered_front, rendered_side = _render_pair_from_measurements(
        front.shape,
        side.shape,
        front_bbox,
        side_bbox,
        front_px_per_cm,
        side_px_per_cm,
        measurements,
        sex,
    )
    front_iou = _iou(front, rendered_front)
    side_iou = _iou(side, rendered_side)
    front_chamfer = _chamfer_distance(front, rendered_front)
    side_chamfer = _chamfer_distance(side, rendered_side)
    max_band_error = max(band_errors.values()) if band_errors else 1.0
    front_foreground_ratio = float((front > 0).mean())
    side_foreground_ratio = float((side > 0).mean())
    side_front_area_ratio = side_foreground_ratio / max(1e-6, front_foreground_ratio)
    mean_iou_penalty = 1.0 - ((front_iou + side_iou) / 2.0)
    mean_chamfer = (front_chamfer + side_chamfer) / 2.0
    score = float((0.55 * max_band_error) + (0.30 * mean_iou_penalty) + (0.15 * mean_chamfer))

    metrics: Dict[str, float] = {
        "score": score,
        "front_iou": front_iou,
        "side_iou": side_iou,
        "front_chamfer": front_chamfer,
        "side_chamfer": side_chamfer,
        "max_band_error": max_band_error,
        "front_foreground_ratio": front_foreground_ratio,
        "side_foreground_ratio": side_foreground_ratio,
        "side_front_area_ratio": side_front_area_ratio,
    }
    for name, value in band_errors.items():
        metrics[f"{name}_band_error"] = value

    reasons = _failure_reasons(front, side, metrics, thresholds)
    accepted = len(reasons) == 0
    return ReliabilityGateResult(
        accepted=accepted,
        score=score,
        reasons=reasons,
        metrics=metrics,
        rendered_front=rendered_front,
        rendered_side=rendered_side,
        asset_info=asset_info,
    )


def export_measurement_proxy_obj(
    output_path: str,
    measurements: Dict[str, float],
    height_cm: float,
    sex: str | None,
    radial_segments: int,
) -> str:
    if height_cm <= 0.0:
        raise ValueError(f"height_cm must be positive for OBJ export, got {height_cm}")
    if radial_segments < 8:
        raise ValueError(f"radial_segments must be >= 8, got {radial_segments}")
    vertices, faces = _proxy_mesh(measurements, height_cm, sex, radial_segments)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Measurement-constrained anthropometry proxy mesh\n")
        for vertex in vertices:
            f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")
    return str(path)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError(f"expected 2-D mask, got shape {mask.shape!r}")
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        raise ValueError("mask has no foreground component")
    largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return ((labels == largest_idx).astype(np.uint8)) * 255


def _body_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        raise ValueError("cannot compute body bbox from empty mask")
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _px_per_cm(bbox: Tuple[int, int, int, int], height_cm: float) -> float:
    _, y0, _, y1 = bbox
    body_height_px = float(y1 - y0 + 1)
    if body_height_px <= 0.0:
        raise ValueError(f"invalid bbox height from {bbox!r}")
    return body_height_px / height_cm


def _band_specs(measurements: Dict[str, float]) -> List[BandSpec]:
    specs: List[BandSpec] = []
    if "chest_cm" in measurements:
        specs.append(("chest_cm", 0.34))
    if "waist_cm" in measurements:
        specs.append(("waist_cm", 0.49))
    if "hip_cm" in measurements:
        specs.append(("hip_cm", 0.61))
    return specs


def _band_width_cm(
    mask: np.ndarray,
    bbox: Tuple[int, int, int, int],
    px_per_cm: float,
    body_fraction: float,
) -> float:
    _, y0, _, y1 = bbox
    body_height = y1 - y0 + 1
    center_y = int(round(y0 + (body_fraction * body_height)))
    radius = max(3, int(round(0.025 * body_height)))
    start_y = max(y0, center_y - radius)
    end_y = min(y1 + 1, center_y + radius + 1)
    widths: List[float] = []
    for y in range(start_y, end_y):
        xs = np.where(mask[y, :] > 0)[0]
        if xs.size > 0:
            widths.append(float(xs.max() - xs.min() + 1))
    if not widths:
        raise ValueError(f"empty band at fraction {body_fraction}")
    return float(np.median(np.asarray(widths, dtype=np.float64)) / px_per_cm)


def _ellipse_circumference(width_cm: float, depth_cm: float) -> float:
    if width_cm <= 0.0 or depth_cm <= 0.0:
        raise ValueError(f"ellipse diameters must be positive, got {width_cm}, {depth_cm}")
    a = width_cm / 2.0
    b = depth_cm / 2.0
    return float(np.pi * (3.0 * (a + b) - np.sqrt((3.0 * a + b) * (a + 3.0 * b))))


def _band_errors(
    front_mask: np.ndarray,
    side_mask: np.ndarray,
    front_bbox: Tuple[int, int, int, int],
    side_bbox: Tuple[int, int, int, int],
    front_px_per_cm: float,
    side_px_per_cm: float,
    measurements: Dict[str, float],
) -> Dict[str, float]:
    errors: Dict[str, float] = {}
    for name, fraction in _band_specs(measurements):
        target = float(measurements[name])
        if target <= 0.0:
            raise ValueError(f"measurement {name} must be positive, got {target}")
        front_width = _band_width_cm(front_mask, front_bbox, front_px_per_cm, fraction)
        side_width = _band_width_cm(side_mask, side_bbox, side_px_per_cm, fraction)
        implied = _ellipse_circumference(front_width, side_width)
        errors[name] = float(abs(implied - target) / target)
    return errors


def _diameters_for_circumference(circumference_cm: float, ratio: float) -> Tuple[float, float]:
    safe_ratio = float(np.clip(ratio, 0.35, 3.50))
    base_front = np.sqrt(safe_ratio)
    base_side = 1.0 / np.sqrt(safe_ratio)
    base_circ = _ellipse_circumference(base_front, base_side)
    scale = circumference_cm / base_circ
    return float(base_front * scale), float(base_side * scale)


def _observed_ratio(
    front_mask: np.ndarray,
    side_mask: np.ndarray,
    front_bbox: Tuple[int, int, int, int],
    side_bbox: Tuple[int, int, int, int],
    front_px_per_cm: float,
    side_px_per_cm: float,
    fraction: float,
) -> float:
    front_width = _band_width_cm(front_mask, front_bbox, front_px_per_cm, fraction)
    side_width = _band_width_cm(side_mask, side_bbox, side_px_per_cm, fraction)
    return front_width / max(1e-6, side_width)


def _render_pair_from_measurements(
    front_shape: Tuple[int, int],
    side_shape: Tuple[int, int],
    front_bbox: Tuple[int, int, int, int],
    side_bbox: Tuple[int, int, int, int],
    front_px_per_cm: float,
    side_px_per_cm: float,
    measurements: Dict[str, float],
    sex: str | None,
) -> Tuple[np.ndarray, np.ndarray]:
    front_bands, side_bands = _render_band_profiles(
        measurements,
        front_px_per_cm,
        side_px_per_cm,
        front_bbox,
        side_bbox,
        sex,
    )
    front = _render_mask_from_profile(front_shape, front_bbox, front_bands)
    side = _render_mask_from_profile(side_shape, side_bbox, side_bands)
    return front, side


def _render_band_profiles(
    measurements: Dict[str, float],
    front_px_per_cm: float,
    side_px_per_cm: float,
    front_bbox: Tuple[int, int, int, int],
    side_bbox: Tuple[int, int, int, int],
    sex: str | None,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    waist = float(measurements.get("waist_cm", 90.0))
    hip = float(measurements.get("hip_cm", max(waist, 100.0)))
    chest = float(measurements.get("chest_cm", max(waist * 1.05, hip * 0.95)))
    sex_norm = (sex or "").strip().lower()
    shoulder_scale = 1.18 if sex_norm in {"male", "m"} else 1.10
    front_depths: Dict[str, Tuple[float, float]] = {}
    for name, circumference, fraction in (
        ("chest", chest, 0.34),
        ("waist", waist, 0.49),
        ("hip", hip, 0.61),
    ):
        ratio = _proxy_ratio(front_bbox, side_bbox, fraction)
        front_cm, side_cm = _diameters_for_circumference(circumference, ratio)
        front_depths[name] = (front_cm, side_cm)
    shoulder_front = front_depths["chest"][0] * shoulder_scale
    shoulder_side = front_depths["chest"][1] * 1.04
    front_profile_cm = [
        (0.00, 0.0),
        (0.06, waist * 0.20),
        (0.14, waist * 0.26),
        (0.24, shoulder_front),
        (0.34, front_depths["chest"][0]),
        (0.49, front_depths["waist"][0]),
        (0.61, front_depths["hip"][0]),
        (0.77, front_depths["hip"][0] * 0.48),
        (0.92, waist * 0.22),
        (1.00, 0.0),
    ]
    side_profile_cm = [
        (0.00, 0.0),
        (0.06, waist * 0.18),
        (0.14, waist * 0.22),
        (0.24, shoulder_side),
        (0.34, front_depths["chest"][1]),
        (0.49, front_depths["waist"][1]),
        (0.61, front_depths["hip"][1]),
        (0.77, front_depths["hip"][1] * 0.58),
        (0.92, waist * 0.18),
        (1.00, 0.0),
    ]
    return (
        [(frac, width_cm * front_px_per_cm) for frac, width_cm in front_profile_cm],
        [(frac, width_cm * side_px_per_cm) for frac, width_cm in side_profile_cm],
    )


def _proxy_mesh(
    measurements: Dict[str, float],
    height_cm: float,
    sex: str | None,
    radial_segments: int,
) -> Tuple[np.ndarray, np.ndarray]:
    front_profile, side_profile = _proxy_profiles_cm(measurements, sex)
    fractions = np.asarray([item[0] for item in front_profile], dtype=np.float64)
    front_widths = np.asarray([item[1] for item in front_profile], dtype=np.float64)
    side_widths = np.asarray([item[1] for item in side_profile], dtype=np.float64)
    angles = np.linspace(0.0, 2.0 * np.pi, radial_segments, endpoint=False)
    vertices: List[Tuple[float, float, float]] = []
    for row_idx, fraction in enumerate(fractions):
        y = (1.0 - fraction) * height_cm
        rx = front_widths[row_idx] / 2.0
        rz = side_widths[row_idx] / 2.0
        for angle in angles:
            vertices.append((float(rx * np.cos(angle)), float(y), float(rz * np.sin(angle))))
    faces: List[Tuple[int, int, int]] = []
    for row_idx in range(len(fractions) - 1):
        row_start = row_idx * radial_segments
        next_start = (row_idx + 1) * radial_segments
        for seg_idx in range(radial_segments):
            a = row_start + seg_idx + 1
            b = row_start + ((seg_idx + 1) % radial_segments) + 1
            c = next_start + seg_idx + 1
            d = next_start + ((seg_idx + 1) % radial_segments) + 1
            faces.append((a, c, b))
            faces.append((b, c, d))
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _proxy_profiles_cm(
    measurements: Dict[str, float],
    sex: str | None,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    waist = float(measurements.get("waist_cm", 90.0))
    hip = float(measurements.get("hip_cm", max(waist, 100.0)))
    chest = float(measurements.get("chest_cm", max(waist * 1.05, hip * 0.95)))
    sex_norm = (sex or "").strip().lower()
    base_ratio = 1.35 if sex_norm in {"male", "m"} else 1.25
    chest_front, chest_side = _diameters_for_circumference(chest, base_ratio)
    waist_front, waist_side = _diameters_for_circumference(waist, base_ratio * 0.92)
    hip_front, hip_side = _diameters_for_circumference(hip, base_ratio)
    shoulder_scale = 1.18 if sex_norm in {"male", "m"} else 1.10
    front_profile = [
        (0.00, 0.0),
        (0.06, waist * 0.20),
        (0.14, waist * 0.26),
        (0.24, chest_front * shoulder_scale),
        (0.34, chest_front),
        (0.49, waist_front),
        (0.61, hip_front),
        (0.77, hip_front * 0.48),
        (0.92, waist * 0.22),
        (1.00, 0.0),
    ]
    side_profile = [
        (0.00, 0.0),
        (0.06, waist * 0.18),
        (0.14, waist * 0.22),
        (0.24, chest_side * 1.04),
        (0.34, chest_side),
        (0.49, waist_side),
        (0.61, hip_side),
        (0.77, hip_side * 0.58),
        (0.92, waist * 0.18),
        (1.00, 0.0),
    ]
    return front_profile, side_profile


def _proxy_ratio(
    front_bbox: Tuple[int, int, int, int],
    side_bbox: Tuple[int, int, int, int],
    fraction: float,
) -> float:
    front_width = float(front_bbox[2] - front_bbox[0] + 1)
    side_width = float(side_bbox[2] - side_bbox[0] + 1)
    base = front_width / max(1.0, side_width)
    waist_adjusted = base * (0.92 if fraction < 0.55 else 1.0)
    return float(np.clip(waist_adjusted, 0.55, 2.75))


def _render_mask_from_profile(
    shape: Tuple[int, int],
    bbox: Tuple[int, int, int, int],
    profile: List[Tuple[float, float]],
) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    x0, y0, x1, y1 = bbox
    body_height = y1 - y0 + 1
    center_x = int(round((x0 + x1) / 2.0))
    fractions = np.asarray([p[0] for p in profile], dtype=np.float64)
    widths = np.asarray([p[1] for p in profile], dtype=np.float64)
    mask = np.zeros((height, width), dtype=np.uint8)
    for y in range(y0, y1 + 1):
        fraction = (y - y0) / float(max(1, body_height - 1))
        row_width = float(np.interp(fraction, fractions, widths))
        half = int(round(max(0.0, row_width) / 2.0))
        left = max(0, center_x - half)
        right = min(width - 1, center_x + half)
        if right >= left:
            mask[y, left : right + 1] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)


def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


def _chamfer_distance(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    contour_a = _contour_mask(mask_a)
    contour_b = _contour_mask(mask_b)
    if int(contour_a.sum()) == 0 or int(contour_b.sum()) == 0:
        return 1.0
    dist_to_b = cv2.distanceTransform((255 - contour_b).astype(np.uint8), cv2.DIST_L2, 3)
    dist_to_a = cv2.distanceTransform((255 - contour_a).astype(np.uint8), cv2.DIST_L2, 3)
    a_to_b = float(np.mean(dist_to_b[contour_a > 0]))
    b_to_a = float(np.mean(dist_to_a[contour_b > 0]))
    bbox = _body_bbox(mask_a)
    norm = float(max(1, bbox[3] - bbox[1] + 1))
    return float(((a_to_b + b_to_a) / 2.0) / norm)


def _contour_mask(mask: np.ndarray) -> np.ndarray:
    binary = ((mask > 0).astype(np.uint8)) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(binary)
    cv2.drawContours(out, contours, contourIdx=-1, color=255, thickness=1)
    return out


def _failure_reasons(
    front_mask: np.ndarray,
    side_mask: np.ndarray,
    metrics: Dict[str, float],
    thresholds: GateThresholds,
) -> List[str]:
    reasons: List[str] = []
    front_report = envelope_check(front_mask, "front")
    side_report = envelope_check(side_mask, "side")
    severe_front = {"empty_input", "non_2d_input", "zero_size", "no_foreground", "no_components"}
    severe_side = {"empty_input", "non_2d_input", "zero_size", "no_foreground", "no_components"}
    reasons.extend([f"front_{name}" for name in front_report.failed if name in severe_front])
    reasons.extend([f"side_{name}" for name in side_report.failed if name in severe_side])
    if metrics["score"] > thresholds.max_score:
        reasons.append("geometry_score")
    if metrics["max_band_error"] > thresholds.max_band_error:
        reasons.append("band_circumference_mismatch")
    if metrics["front_iou"] < thresholds.min_render_iou:
        reasons.append("front_render_mismatch")
    if metrics["side_iou"] < thresholds.min_render_iou:
        reasons.append("side_render_mismatch")
    if max(metrics["front_chamfer"], metrics["side_chamfer"]) > thresholds.max_chamfer:
        reasons.append("contour_mismatch")
    if metrics["front_foreground_ratio"] > thresholds.max_front_foreground_ratio:
        reasons.append("front_foreground_too_large")
    if metrics["side_foreground_ratio"] > thresholds.max_side_foreground_ratio:
        reasons.append("side_foreground_too_large")
    if metrics["side_front_area_ratio"] < thresholds.min_side_front_area_ratio:
        reasons.append("side_front_area_ratio_too_small")
    if metrics["side_front_area_ratio"] > thresholds.max_side_front_area_ratio:
        reasons.append("side_front_area_ratio_too_large")
    return sorted(set(reasons))
