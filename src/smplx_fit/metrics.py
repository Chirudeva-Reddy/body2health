"""Render-back reliability metrics for SMPL-X silhouettes."""
from __future__ import annotations

from typing import Dict, Tuple, cast

import cv2
import numpy as np


BBox = Tuple[int, int, int, int]


def fit_metrics(input_mask: np.ndarray, rendered_mask: np.ndarray) -> Dict[str, float]:
    input_clean = largest_component(input_mask)
    rendered_clean = largest_component(rendered_mask)
    input_bbox = mask_bbox(input_clean)
    rendered_bbox = mask_bbox(rendered_clean)
    shoulder_crop = mask_band(input_clean, 0.14, 0.34)
    torso_crop = mask_band(input_clean, 0.28, 0.62)
    hip_crop = mask_band(input_clean, 0.50, 0.72)
    lower_body_crop = mask_band(input_clean, 0.62, 0.95)
    shoulder_iou = iou(input_clean[shoulder_crop], rendered_clean[shoulder_crop])
    shoulder_chamfer = chamfer(input_clean[shoulder_crop], rendered_clean[shoulder_crop])
    torso_iou = iou(input_clean[torso_crop], rendered_clean[torso_crop])
    hip_iou = iou(input_clean[hip_crop], rendered_clean[hip_crop])
    lower_body_iou = iou(input_clean[lower_body_crop], rendered_clean[lower_body_crop])
    left_residual = side_residual_area(input_clean, rendered_clean, shoulder_crop, "left")
    right_residual = side_residual_area(input_clean, rendered_clean, shoulder_crop, "right")
    front_iou = iou(input_clean, rendered_clean)
    front_chamfer = chamfer(input_clean, rendered_clean)
    bbox_metrics = bbox_alignment_metrics(input_bbox, rendered_bbox, input_clean.shape)
    shoulder_metrics = shoulder_alignment_metrics(input_clean, rendered_clean)
    score = float(
        (0.30 * (1.0 - front_iou))
        + (0.20 * front_chamfer)
        + (0.17 * (1.0 - shoulder_iou))
        + (0.14 * (1.0 - torso_iou))
        + (0.10 * (1.0 - hip_iou))
        + (0.07 * (1.0 - lower_body_iou))
        + (0.02 * min(1.0, bbox_metrics["bbox_center_distance_px"] / 20.0))
        + (0.03 * min(1.0, abs(bbox_metrics["bbox_height_ratio"] - 1.0) / 0.08))
        + (0.08 * min(1.0, shoulder_metrics["shoulder_row_abs_error_px"] / 25.0))
    )
    metrics = {
        "score": score,
        "front_iou": front_iou,
        "front_chamfer": front_chamfer,
        "shoulder_iou": shoulder_iou,
        "shoulder_chamfer": shoulder_chamfer,
        "torso_iou": torso_iou,
        "hip_iou": hip_iou,
        "lower_body_iou": lower_body_iou,
        "left_shoulder_residual_area": left_residual,
        "right_shoulder_residual_area": right_residual,
        "front_foreground_ratio": float((input_clean > 0).mean()),
        "rendered_foreground_ratio": float((rendered_clean > 0).mean()),
    }
    metrics.update(bbox_metrics)
    metrics.update(shoulder_metrics)
    return metrics


def shoulder_alignment_metrics(input_mask: np.ndarray, rendered_mask: np.ndarray) -> Dict[str, float]:
    input_row = shoulder_row(input_mask)
    rendered_row = shoulder_row(rendered_mask)
    return {
        "input_shoulder_row_px": float(input_row),
        "rendered_shoulder_row_px": float(rendered_row),
        "shoulder_row_error_px": float(rendered_row - input_row),
        "shoulder_row_abs_error_px": float(abs(rendered_row - input_row)),
    }


def shoulder_row(mask: np.ndarray) -> int:
    bbox = mask_bbox(mask)
    x0, y0, x1, y1 = bbox
    widths = _row_widths(mask)
    body_width = max(1, x1 - x0 + 1)
    threshold = int(round(float(body_width) * 0.55))
    y_start = max(0, y0)
    y_stop = min(mask.shape[0], y1 + 1)
    candidates = np.where(widths[y_start:y_stop] >= threshold)[0]
    if candidates.size == 0:
        return y_start
    return int(y_start + candidates[0])


def _row_widths(mask: np.ndarray) -> np.ndarray:
    binary = mask > 0
    widths = np.zeros(mask.shape[0], dtype=np.int32)
    for y_index in range(mask.shape[0]):
        x_values = np.where(binary[y_index])[0]
        if x_values.size > 0:
            widths[y_index] = int(x_values.max() - x_values.min() + 1)
    return widths


def bbox_alignment_metrics(
    input_bbox: BBox,
    rendered_bbox: BBox,
    shape: Tuple[int, int],
) -> Dict[str, float]:
    input_width, input_height = _bbox_size(input_bbox)
    rendered_width, rendered_height = _bbox_size(rendered_bbox)
    input_center_x, input_center_y = _bbox_center(input_bbox)
    rendered_center_x, rendered_center_y = _bbox_center(rendered_bbox)
    center_x_error = rendered_center_x - input_center_x
    center_y_error = rendered_center_y - input_center_y
    center_distance = float(np.hypot(center_x_error, center_y_error))
    height, width = int(shape[0]), int(shape[1])
    return {
        "input_bbox_x0": float(input_bbox[0]),
        "input_bbox_y0": float(input_bbox[1]),
        "input_bbox_x1": float(input_bbox[2]),
        "input_bbox_y1": float(input_bbox[3]),
        "rendered_bbox_x0": float(rendered_bbox[0]),
        "rendered_bbox_y0": float(rendered_bbox[1]),
        "rendered_bbox_x1": float(rendered_bbox[2]),
        "rendered_bbox_y1": float(rendered_bbox[3]),
        "bbox_center_x_error_px": float(center_x_error),
        "bbox_center_y_error_px": float(center_y_error),
        "bbox_center_distance_px": center_distance,
        "bbox_width_ratio": float(rendered_width) / float(max(1, input_width)),
        "bbox_height_ratio": float(rendered_height) / float(max(1, input_height)),
        "input_touches_left": float(input_bbox[0] <= 0),
        "input_touches_top": float(input_bbox[1] <= 0),
        "input_touches_right": float(input_bbox[2] >= width - 1),
        "input_touches_bottom": float(input_bbox[3] >= height - 1),
        "input_touches_border": float(
            input_bbox[0] <= 0
            or input_bbox[1] <= 0
            or input_bbox[2] >= width - 1
            or input_bbox[3] >= height - 1
        ),
    }


def mask_bbox(mask: np.ndarray) -> BBox:
    y_values, x_values = np.where(mask > 0)
    if y_values.size == 0:
        raise ValueError("cannot compute bbox from empty mask")
    return int(x_values.min()), int(y_values.min()), int(x_values.max()), int(y_values.max())


def _bbox_size(bbox: BBox) -> Tuple[int, int]:
    return max(1, bbox[2] - bbox[0] + 1), max(1, bbox[3] - bbox[1] + 1)


def _bbox_center(bbox: BBox) -> Tuple[float, float]:
    return float(bbox[0] + bbox[2]) / 2.0, float(bbox[1] + bbox[3]) / 2.0


def mask_band(mask: np.ndarray, top_fraction: float, bottom_fraction: float) -> Tuple[slice, slice]:
    x0, y0, x1, y1 = mask_bbox(mask)
    height = max(1, y1 - y0 + 1)
    top = int(round(float(y0) + (float(height) * top_fraction)))
    bottom = int(round(float(y0) + (float(height) * bottom_fraction)))
    y_start = max(0, min(mask.shape[0] - 1, top))
    y_stop = max(y_start + 1, min(mask.shape[0], bottom))
    return slice(y_start, y_stop), slice(max(0, x0), min(mask.shape[1], x1 + 1))


def side_residual_area(
    input_mask: np.ndarray,
    rendered_mask: np.ndarray,
    band: Tuple[slice, slice],
    side: str,
) -> float:
    cropped_input = input_mask[band] > 0
    cropped_render = rendered_mask[band] > 0
    width = cropped_input.shape[1]
    if side == "left":
        side_slice = slice(0, max(1, width // 2))
    elif side == "right":
        side_slice = slice(max(0, width // 2), width)
    else:
        raise ValueError(f"unsupported_side: {side}")
    input_side = cropped_input[:, side_slice]
    render_side = cropped_render[:, side_slice]
    denominator = max(1, int(input_side.sum()))
    return float(np.logical_xor(input_side, render_side).sum()) / float(denominator)


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum()) / float(union)


def chamfer(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    edge_a = edge(mask_a)
    edge_b = edge(mask_b)
    if int(edge_a.sum()) == 0 or int(edge_b.sum()) == 0:
        return 1.0
    dist_a = cv2.distanceTransform((1 - edge_a).astype(np.uint8), cv2.DIST_L2, 3)
    dist_b = cv2.distanceTransform((1 - edge_b).astype(np.uint8), cv2.DIST_L2, 3)
    a_to_b = float(dist_b[edge_a > 0].mean())
    b_to_a = float(dist_a[edge_b > 0].mean())
    diagonal = float(np.hypot(mask_a.shape[0], mask_a.shape[1]))
    if diagonal <= 0.0:
        return 1.0
    return (a_to_b + b_to_a) / (2.0 * diagonal)


def edge(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=1)
    return cast(np.ndarray, binary - eroded)


def largest_component(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if component_count <= 1:
        raise ValueError("mask has no foreground component")
    largest_index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return cast(np.ndarray, (labels == largest_index).astype(np.uint8) * 255)
