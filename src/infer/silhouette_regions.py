from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


RegionMetrics = Dict[str, float | int | Tuple[int, int, int, int]]
RegionArtifacts = Dict[str, str]


REGION_FRACTIONS: Dict[str, Tuple[float, float]] = {
    "head_neck": (0.00, 0.18),
    "shoulders": (0.14, 0.28),
    "torso": (0.24, 0.52),
    "waist": (0.46, 0.53),
    "hip": (0.58, 0.68),
    "legs": (0.68, 1.00),
}


REGION_COLORS: Dict[str, Tuple[int, int, int]] = {
    "head_neck": (255, 180, 80),
    "shoulders": (80, 180, 255),
    "torso": (110, 220, 110),
    "waist": (80, 80, 255),
    "hip": (220, 110, 220),
    "legs": (255, 220, 80),
}


def save_silhouette_region_artifacts(mask: np.ndarray, output_dir: str, prefix: str) -> RegionArtifacts:
    binary = _validate_mask(mask)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    overlay_path = root / f"{prefix}_regions_overlay.png"
    metrics_path = root / f"{prefix}_regions.json"
    overlay = make_region_overlay(binary)
    metrics = region_metrics(binary)
    _write_image(overlay_path, overlay)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    return {"overlay": str(overlay_path), "metrics": str(metrics_path)}


def make_region_overlay(mask: np.ndarray) -> np.ndarray:
    binary = _validate_mask(mask)
    bbox = _bbox(binary)
    overlay = _base_overlay(binary)
    _draw_contour(overlay, binary)
    _draw_region_bands(overlay, binary, bbox)
    _draw_arm_torso_boundaries(overlay, binary, bbox)
    _draw_research_lines(overlay, binary, bbox)
    _draw_legend(overlay)
    return overlay


def region_metrics(mask: np.ndarray) -> RegionMetrics:
    binary = _validate_mask(mask)
    x0, y0, x1, y1 = _bbox(binary)
    height = max(1, y1 - y0 + 1)
    width = max(1, x1 - x0 + 1)
    metrics: RegionMetrics = {
        "bbox": (x0, y0, x1, y1),
        "bbox_width_px": width,
        "bbox_height_px": height,
        "foreground_ratio": float(binary.mean()),
        "body_area_px": int(binary.sum()),
    }
    for name, band in REGION_FRACTIONS.items():
        y_start, y_end = _band_rows(y0, y1, band)
        metrics[f"{name}_area_px"] = int(binary[y_start : y_end + 1].sum())
        metrics[f"{name}_mean_width_px"] = _mean_row_width(binary, y_start, y_end)
        metrics[f"{name}_max_width_px"] = _max_row_width(binary, y_start, y_end)
    metrics["shoulder_to_waist_width_ratio"] = _safe_ratio(
        float(metrics["shoulders_max_width_px"]),
        float(metrics["waist_mean_width_px"]),
    )
    metrics["hip_to_waist_width_ratio"] = _safe_ratio(
        float(metrics["hip_mean_width_px"]),
        float(metrics["waist_mean_width_px"]),
    )
    metrics["left_right_torso_asymmetry"] = _torso_asymmetry(binary, x0, x1, y0, y1)
    arm_torso = _arm_torso_metrics(binary, x0, x1, y0, y1)
    metrics.update(arm_torso)
    return metrics


def _validate_mask(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0:
        raise ValueError("silhouette mask is empty")
    if mask.ndim != 2:
        raise ValueError(f"silhouette mask must be 2D, got {mask.shape!r}")
    return (mask > 0).astype(np.uint8)


def _bbox(binary: np.ndarray) -> Tuple[int, int, int, int]:
    y_values, x_values = np.where(binary > 0)
    if y_values.size == 0 or x_values.size == 0:
        raise ValueError("silhouette mask has no foreground")
    return int(x_values.min()), int(y_values.min()), int(x_values.max()), int(y_values.max())


def _base_overlay(binary: np.ndarray) -> np.ndarray:
    base = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)
    base[binary > 0] = (235, 235, 235)
    return base


def _draw_contour(overlay: np.ndarray, binary: np.ndarray) -> None:
    contours, _ = cv2.findContours((binary * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)


def _draw_region_bands(
    overlay: np.ndarray,
    binary: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> None:
    x0, y0, x1, y1 = bbox
    for name, band in REGION_FRACTIONS.items():
        color = REGION_COLORS[name]
        y_start, y_end = _band_rows(y0, y1, band)
        region = binary[y_start : y_end + 1, :] > 0
        colored = np.zeros_like(overlay[y_start : y_end + 1, :])
        colored[region] = color
        overlay[y_start : y_end + 1, :] = cv2.addWeighted(
            overlay[y_start : y_end + 1, :],
            0.68,
            colored,
            0.32,
            0.0,
        )
        cv2.line(overlay, (x0, y_start), (x1, y_start), color, 1, cv2.LINE_AA)
        cv2.line(overlay, (x0, y_end), (x1, y_end), color, 1, cv2.LINE_AA)


def _draw_research_lines(
    overlay: np.ndarray,
    binary: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> None:
    x0, y0, x1, y1 = bbox
    center_x = int(round((x0 + x1) / 2.0))
    cv2.line(overlay, (center_x, y0), (center_x, y1), (255, 255, 255), 1, cv2.LINE_AA)
    for name in ("shoulders", "waist", "hip"):
        y_start, y_end = _band_rows(y0, y1, REGION_FRACTIONS[name])
        y_mid = int(round((y_start + y_end) / 2.0))
        row = binary[y_mid, :]
        x_values = np.where(row > 0)[0]
        if x_values.size == 0:
            continue
        left = int(x_values.min())
        right = int(x_values.max())
        cv2.line(overlay, (left, y_mid), (right, y_mid), REGION_COLORS[name], 3, cv2.LINE_AA)
        cv2.circle(overlay, (left, y_mid), 4, REGION_COLORS[name], -1)
        cv2.circle(overlay, (right, y_mid), 4, REGION_COLORS[name], -1)


def _draw_arm_torso_boundaries(
    overlay: np.ndarray,
    binary: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> None:
    x0, y0, x1, y1 = bbox
    left_points: list[Tuple[int, int]] = []
    right_points: list[Tuple[int, int]] = []
    y_start, y_end = _band_rows(y0, y1, (0.24, 0.58))
    center_x = int(round((x0 + x1) / 2.0))
    for row_idx in range(y_start, y_end + 1):
        spans = _foreground_spans(binary[row_idx, :])
        if len(spans) < 3:
            continue
        torso_index = _torso_span_index(spans, center_x)
        if torso_index <= 0 or torso_index >= len(spans) - 1:
            continue
        left_arm = spans[torso_index - 1]
        torso = spans[torso_index]
        right_arm = spans[torso_index + 1]
        left_points.append((int(round((left_arm[1] + torso[0]) / 2.0)), row_idx))
        right_points.append((int(round((torso[1] + right_arm[0]) / 2.0)), row_idx))
    _draw_polyline(overlay, left_points, (255, 80, 180))
    _draw_polyline(overlay, right_points, (255, 80, 180))


def _draw_polyline(
    overlay: np.ndarray,
    points: list[Tuple[int, int]],
    color: Tuple[int, int, int],
) -> None:
    if len(points) < 2:
        return
    point_array = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(overlay, [point_array], False, color, 2, cv2.LINE_AA)


def _draw_legend(overlay: np.ndarray) -> None:
    entries = [
        ("shoulder", REGION_COLORS["shoulders"]),
        ("torso", REGION_COLORS["torso"]),
        ("waist", REGION_COLORS["waist"]),
        ("hip", REGION_COLORS["hip"]),
        ("arm split", (255, 80, 180)),
    ]
    x = 8
    y = 18
    for label, color in entries:
        cv2.rectangle(overlay, (x, y - 9), (x + 10, y + 1), color, -1)
        cv2.putText(
            overlay,
            label,
            (x + 14, y + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 16


def _band_rows(y0: int, y1: int, band: Tuple[float, float]) -> Tuple[int, int]:
    height = max(1, y1 - y0 + 1)
    start = y0 + int(round(height * band[0]))
    end = y0 + int(round(height * band[1]))
    return max(y0, start), min(y1, end)


def _row_widths(binary: np.ndarray, y_start: int, y_end: int) -> np.ndarray:
    widths = []
    for row_idx in range(y_start, y_end + 1):
        x_values = np.where(binary[row_idx, :] > 0)[0]
        if x_values.size > 0:
            widths.append(int(x_values.max() - x_values.min() + 1))
    return np.asarray(widths, dtype=np.float64)


def _mean_row_width(binary: np.ndarray, y_start: int, y_end: int) -> float:
    widths = _row_widths(binary, y_start, y_end)
    if widths.size == 0:
        return 0.0
    return float(widths.mean())


def _max_row_width(binary: np.ndarray, y_start: int, y_end: int) -> float:
    widths = _row_widths(binary, y_start, y_end)
    if widths.size == 0:
        return 0.0
    return float(widths.max())


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _torso_asymmetry(binary: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> float:
    center_x = int(round((x0 + x1) / 2.0))
    y_start, y_end = _band_rows(y0, y1, REGION_FRACTIONS["torso"])
    torso = binary[y_start : y_end + 1, :]
    left_area = float(torso[:, :center_x].sum())
    right_area = float(torso[:, center_x:].sum())
    total = left_area + right_area
    if total <= 0.0:
        return 0.0
    return abs(left_area - right_area) / total


def _arm_torso_metrics(binary: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> RegionMetrics:
    center_x = int(round((x0 + x1) / 2.0))
    y_start, y_end = _band_rows(y0, y1, (0.24, 0.58))
    left_arm_area = 0
    torso_area = 0
    right_arm_area = 0
    separated_rows = 0
    for row_idx in range(y_start, y_end + 1):
        spans = _foreground_spans(binary[row_idx, :])
        if len(spans) < 3:
            continue
        torso_index = _torso_span_index(spans, center_x)
        if torso_index <= 0 or torso_index >= len(spans) - 1:
            continue
        left_arm = spans[torso_index - 1]
        torso = spans[torso_index]
        right_arm = spans[torso_index + 1]
        left_arm_area += left_arm[1] - left_arm[0] + 1
        torso_area += torso[1] - torso[0] + 1
        right_arm_area += right_arm[1] - right_arm[0] + 1
        separated_rows += 1
    return {
        "arm_torso_separation_rows": separated_rows,
        "left_arm_area_px": left_arm_area,
        "torso_core_area_px": torso_area,
        "right_arm_area_px": right_arm_area,
    }


def _foreground_spans(row: np.ndarray) -> list[Tuple[int, int]]:
    x_values = np.where(row > 0)[0]
    if x_values.size == 0:
        return []
    breaks = np.where(np.diff(x_values) > 1)[0]
    spans: list[Tuple[int, int]] = []
    start_idx = 0
    for break_idx in breaks:
        spans.append((int(x_values[start_idx]), int(x_values[break_idx])))
        start_idx = int(break_idx + 1)
    spans.append((int(x_values[start_idx]), int(x_values[-1])))
    return spans


def _torso_span_index(spans: list[Tuple[int, int]], center_x: int) -> int:
    containing = [
        index
        for index, span in enumerate(spans)
        if span[0] <= center_x <= span[1]
    ]
    if containing:
        return containing[0]
    distances = [
        abs(((span[0] + span[1]) / 2.0) - float(center_x))
        for span in spans
    ]
    return int(np.argmin(np.asarray(distances, dtype=np.float64)))


def _write_image(path: Path, image: np.ndarray) -> None:
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise OSError(f"failed to write silhouette region artifact: {path}")
