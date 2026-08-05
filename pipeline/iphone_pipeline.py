"""Strict BodyM-compatible RGB-to-silhouette preprocessing."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, cast

import cv2
import numpy as np

from pipeline.sam_seg import SegmentationAssets, StrictSegmentationResult, segment_person_strict
from src.infer.silhouette_checks import MIN_ROW_WIDTH_PX as MIN_ROW_WIDTH
from src.infer.silhouette_checks import envelope_check


TARGET_HW: Tuple[int, int] = (640, 480)
DEFAULT_SAM_MODEL_PATH = "models/segmentation/sam2.1_hiera_large.pt"
DEFAULT_SAM_CONFIG_PATH = "configs/segmentation/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_YOLO_MODEL_PATH = "models/segmentation/yolo11m.pt"


@dataclass(frozen=True)
class ProcessedSilhouette:
    mask: np.ndarray
    segmentation: StrictSegmentationResult
    standardization: "MaskStandardization"


@dataclass(frozen=True)
class MaskStandardization:
    source_bbox: Tuple[int, int, int, int]
    crop_shape: Tuple[int, int]
    scale: float
    resized_shape: Tuple[int, int]
    offset_xy: Tuple[int, int]
    target_shape: Tuple[int, int]


@dataclass(frozen=True)
class StandardizedMask:
    mask: np.ndarray
    standardization: MaskStandardization


def is_valid_silhouette(mask: np.ndarray) -> bool:
    return envelope_check(mask, "front").ok


def process_iphone_image(
    img_rgb: np.ndarray,
    sam_model_path: str | None = None,
    sam_config_path: str | None = None,
    yolo_model_path: str | None = None,
    view: str = "front",
    debug_dir: str | None = None,
    debug_prefix: str = "silhouette",
    yolo_confidence: float = 0.35,
) -> np.ndarray:
    processed = process_iphone_image_with_metadata(
        img_rgb=img_rgb,
        sam_model_path=sam_model_path,
        sam_config_path=sam_config_path,
        yolo_model_path=yolo_model_path,
        view=view,
        debug_dir=debug_dir,
        debug_prefix=debug_prefix,
        yolo_confidence=yolo_confidence,
    )
    return processed.mask


def process_iphone_image_with_metadata(
    img_rgb: np.ndarray,
    sam_model_path: str | None,
    sam_config_path: str | None,
    yolo_model_path: str | None,
    view: str,
    debug_dir: str | None,
    debug_prefix: str,
    yolo_confidence: float,
) -> ProcessedSilhouette:
    _validate_rgb_image(img_rgb)
    assets = _resolve_assets(
        sam_model_path=sam_model_path,
        sam_config_path=sam_config_path,
        yolo_model_path=yolo_model_path,
    )
    segmentation = segment_person_strict(
        image_rgb=img_rgb,
        assets=assets,
        view=view,
        yolo_confidence=yolo_confidence,
    )
    standardized = _standardize_mask(segmentation.mask, TARGET_HW, view)
    mask = standardized.mask
    _validate_bodym_mask(mask, view)
    if debug_dir is not None:
        _write_debug_mask(mask, debug_dir, f"{debug_prefix}_final_silhouette.png")
    return ProcessedSilhouette(
        mask=mask,
        segmentation=segmentation,
        standardization=standardized.standardization,
    )


def validate_bodym_compatibility(mask: np.ndarray) -> Dict[str, Any]:
    issues: list[str] = []
    checks: Dict[str, Any] = {}
    checks["shape"] = mask.shape == TARGET_HW
    checks["dtype"] = mask.dtype == np.uint8
    checks["binary_values"] = set(np.unique(mask).tolist()).issubset({0, 255})
    checks["anatomy_valid"] = is_valid_silhouette(mask)
    foreground_ratio = float((mask > 0).mean())
    checks["foreground_ratio"] = 0.01 <= foreground_ratio <= 0.80
    if not checks["shape"]:
        issues.append(f"shape {mask.shape!r} != {TARGET_HW!r}")
    if not checks["dtype"]:
        issues.append(f"dtype {mask.dtype!r} != uint8")
    if not checks["binary_values"]:
        issues.append(f"values {set(np.unique(mask).tolist())!r} are not binary")
    if not checks["anatomy_valid"]:
        issues.append("failed anatomical silhouette validation")
    if not checks["foreground_ratio"]:
        issues.append(f"foreground ratio {foreground_ratio:.3f} outside [0.01, 0.80]")
    return {"compatible": len(issues) == 0, "issues": issues, "checks": checks}


class BodyMPipeline:
    def __init__(
        self,
        sam_model_path: str | None = None,
        sam_config_path: str | None = None,
        yolo_model_path: str | None = None,
    ) -> None:
        self.sam_model_path = sam_model_path
        self.sam_config_path = sam_config_path
        self.yolo_model_path = yolo_model_path

    def process_iphone_image(
        self,
        img_rgb: np.ndarray,
        view: str = "front",
        debug_dir: str | None = None,
        debug_prefix: str = "silhouette",
    ) -> np.ndarray:
        return process_iphone_image(
            img_rgb=img_rgb,
            sam_model_path=self.sam_model_path,
            sam_config_path=self.sam_config_path,
            yolo_model_path=self.yolo_model_path,
            view=view,
            debug_dir=debug_dir,
            debug_prefix=debug_prefix,
            yolo_confidence=0.35,
        )


def _validate_rgb_image(img_rgb: np.ndarray) -> None:
    if img_rgb.size == 0:
        raise ValueError("Invalid input image: empty RGB array")
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError(f"Input must be RGB with shape (H, W, 3), got {img_rgb.shape!r}")


def _resolve_assets(
    sam_model_path: str | None,
    sam_config_path: str | None,
    yolo_model_path: str | None,
) -> SegmentationAssets:
    assets = SegmentationAssets(
        sam_checkpoint_path=sam_model_path or DEFAULT_SAM_MODEL_PATH,
        sam_config_path=sam_config_path or DEFAULT_SAM_CONFIG_PATH,
        yolo_model_path=yolo_model_path or DEFAULT_YOLO_MODEL_PATH,
    )
    missing = [
        path
        for path in (assets.sam_checkpoint_path, assets.sam_config_path, assets.yolo_model_path)
        if not Path(path).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "missing_segmentation_asset: expected strict YOLO+SAM2 assets at "
            + ", ".join(missing)
        )
    return assets


def _standardize_mask(mask: np.ndarray, target_hw: Tuple[int, int], view: str) -> StandardizedMask:
    binary = _clean_person_mask((mask > 0).astype(np.uint8) * 255, view)
    y_values, x_values = np.where(binary > 0)
    if y_values.size == 0 or x_values.size == 0:
        raise ValueError("sam2_empty_person_mask: no foreground after connected-component filtering")
    y0 = int(y_values.min())
    y1 = int(y_values.max())
    x0 = int(x_values.min())
    x1 = int(x_values.max())
    cropped = binary[y0 : y1 + 1, x0 : x1 + 1]
    target_h, target_w = target_hw
    crop_h, crop_w = cropped.shape
    scale = min(float(target_h) / float(crop_h), float(target_w) / float(crop_w))
    resized_w = max(1, int(round(float(crop_w) * scale)))
    resized_h = max(1, int(round(float(crop_h) * scale)))
    resized = cv2.resize(cropped, (resized_w, resized_h), interpolation=cv2.INTER_NEAREST)
    output = np.zeros((target_h, target_w), dtype=np.uint8)
    start_y = (target_h - resized_h) // 2
    start_x = (target_w - resized_w) // 2
    output[start_y : start_y + resized_h, start_x : start_x + resized_w] = resized
    cleaned = _clean_person_mask(output, view)
    standardization = MaskStandardization(
        source_bbox=(x0, y0, x1, y1),
        crop_shape=(crop_h, crop_w),
        scale=scale,
        resized_shape=(resized_h, resized_w),
        offset_xy=(start_x, start_y),
        target_shape=target_hw,
    )
    return StandardizedMask(mask=cleaned, standardization=standardization)


def _clean_person_mask(mask: np.ndarray, view: str) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8) * 255
    normalized_view = view.strip().lower()
    kernel_size = 9 if normalized_view in {"side", "profile"} else 5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    filled = _fill_internal_holes(closed)
    smoothed = _smooth_binary_edges(filled)
    return _largest_component(smoothed)


def _smooth_binary_edges(mask: np.ndarray) -> np.ndarray:
    blurred = cv2.medianBlur((mask > 0).astype(np.uint8) * 255, 5)
    return (blurred > 127).astype(np.uint8) * 255


def _fill_internal_holes(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8) * 255
    padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    h, w = flood.shape
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    flood = flood[1:-1, 1:-1]
    holes = cv2.bitwise_and(cv2.bitwise_not(flood), cv2.bitwise_not(binary))
    return cv2.bitwise_or(binary, holes)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if component_count <= 1:
        raise ValueError("mask has no foreground component")
    largest_index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return cast(np.ndarray, (labels == largest_index).astype(np.uint8) * 255)


def _validate_bodym_mask(mask: np.ndarray, view: str) -> None:
    if mask.shape != TARGET_HW:
        raise ValueError(f"BodyM mask shape must be {TARGET_HW!r}, got {mask.shape!r}")
    if mask.dtype != np.uint8:
        raise ValueError(f"BodyM mask dtype must be uint8, got {mask.dtype!r}")
    values = set(np.unique(mask).tolist())
    if not values.issubset({0, 255}):
        raise ValueError(f"BodyM mask must be binary 0/255, got {values!r}")
    foreground_ratio = float((mask > 0).mean())
    normalized_view = view.strip().lower()
    if normalized_view in {"side", "profile"}:
        if foreground_ratio < 0.05 or foreground_ratio > 0.36:
            raise ValueError(f"side_foreground_ratio_invalid: {foreground_ratio:.3f}")
    elif foreground_ratio < 0.08 or foreground_ratio > 0.42:
        raise ValueError(f"front_foreground_ratio_invalid: {foreground_ratio:.3f}")
    report = envelope_check(mask, view)
    if not report.ok:
        raise ValueError("silhouette_envelope_invalid: " + ", ".join(report.failed))


def _write_debug_mask(mask: np.ndarray, debug_dir: str, filename: str) -> None:
    root = Path(debug_dir)
    root.mkdir(parents=True, exist_ok=True)
    output_path = root / filename
    ok = cv2.imwrite(str(output_path), mask)
    if not ok:
        raise OSError(f"failed to write debug mask: {output_path}")


if __name__ == "__main__":
    raise SystemExit("Import process_iphone_image from pipeline.iphone_pipeline.")
