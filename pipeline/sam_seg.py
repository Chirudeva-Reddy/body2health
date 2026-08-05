"""Strict YOLO + SAM2 person segmentation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple, cast

import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class SegmentationAssets:
    sam_checkpoint_path: str
    sam_config_path: str
    yolo_model_path: str


@dataclass(frozen=True)
class PersonBox:
    x0: int
    y0: int
    x1: int
    y1: int
    confidence: float


@dataclass(frozen=True)
class StrictSegmentationResult:
    mask: np.ndarray
    bbox: PersonBox
    sam_score: float
    foreground_ratio: float


_SAM2_PREDICTOR: Any | None = None
_YOLO_MODEL: Any | None = None


def segment_person_strict(
    image_rgb: np.ndarray,
    assets: SegmentationAssets,
    view: str,
    yolo_confidence: float,
) -> StrictSegmentationResult:
    _validate_rgb_image(image_rgb)
    _validate_assets(assets)
    bbox = detect_person_box(image_rgb, assets.yolo_model_path, yolo_confidence)
    predictor = _get_sam2_predictor(assets.sam_checkpoint_path, assets.sam_config_path)
    masks, scores = _predict_sam2_masks(predictor, image_rgb, bbox, view)
    mask, score = _select_valid_mask(masks, scores, view)
    foreground_ratio = float((mask > 0).mean())
    return StrictSegmentationResult(
        mask=mask,
        bbox=bbox,
        sam_score=score,
        foreground_ratio=foreground_ratio,
    )


def detect_person_box(
    image_rgb: np.ndarray,
    yolo_model_path: str,
    yolo_confidence: float,
) -> PersonBox:
    if yolo_confidence <= 0.0 or yolo_confidence >= 1.0:
        raise ValueError(f"yolo_confidence must be in (0, 1), got {yolo_confidence}")
    yolo_model = _get_yolo(yolo_model_path)
    result = yolo_model.predict(source=image_rgb, classes=[0], conf=yolo_confidence, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        raise ValueError(
            "yolo_no_person: YOLO found no COCO person box. "
            f"model_path={yolo_model_path} confidence={yolo_confidence}"
        )
    best_index = int(result.boxes.conf.argmax().item())
    confidence = float(result.boxes.conf[best_index].item())
    x0, y0, x1, y1 = result.boxes.xyxy[best_index].cpu().numpy().astype(int).tolist()
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"yolo_invalid_box: box={(x0, y0, x1, y1)}")
    return PersonBox(x0=x0, y0=y0, x1=x1, y1=y1, confidence=confidence)


def _validate_rgb_image(image_rgb: np.ndarray) -> None:
    if image_rgb.size == 0:
        raise ValueError("image_rgb is empty")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"image_rgb must have shape (H, W, 3), got {image_rgb.shape!r}")


def _validate_assets(assets: SegmentationAssets) -> None:
    missing_paths = [
        path
        for path in (assets.sam_checkpoint_path, assets.sam_config_path, assets.yolo_model_path)
        if not Path(path).exists()
    ]
    if missing_paths:
        raise FileNotFoundError(
            "missing_segmentation_asset: required YOLO/SAM2 file is missing: "
            + ", ".join(missing_paths)
        )


def _get_sam2_predictor(checkpoint_path: str, config_path: str) -> Any:
    global _SAM2_PREDICTOR
    if _SAM2_PREDICTOR is None:
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from hydra import compose, initialize_config_dir
            from hydra.core.global_hydra import GlobalHydra
            from omegaconf import OmegaConf
            from hydra.utils import instantiate
            from sam2.build_sam import _load_checkpoint
        except ImportError as exc:
            raise RuntimeError(
                "missing_sam2: install SAM2 into the project .venv before RGB inference"
            ) from exc
        device = "cuda" if torch.cuda.is_available() else "cpu"
        config_file = Path(config_path)
        if config_file.exists():
            GlobalHydra.instance().clear()
            with initialize_config_dir(version_base=None, config_dir=str(config_file.parent.resolve())):
                cfg = compose(config_name=config_file.name, overrides=[])
                OmegaConf.resolve(cfg)
                sam2_model = instantiate(cfg.model, _recursive_=True)
                _load_checkpoint(sam2_model, checkpoint_path)
                sam2_model = sam2_model.to(device)
                sam2_model.eval()
        else:
            sam2_model = build_sam2(config_path, checkpoint_path, device=device)
        _SAM2_PREDICTOR = SAM2ImagePredictor(sam2_model)
    return _SAM2_PREDICTOR


def _get_yolo(model_path: str) -> Any:
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "missing_ultralytics: install ultralytics into the project .venv before RGB inference"
            ) from exc
        _YOLO_MODEL = YOLO(model_path)
    return _YOLO_MODEL


def _predict_sam2_masks(
    predictor: Any,
    image_rgb: np.ndarray,
    bbox: PersonBox,
    view: str,
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = image_rgb.shape[:2]
    box = _expanded_box(bbox, width, height, view)
    predictor.set_image(image_rgb)
    masks, scores, _ = predictor.predict(
        box=box[None, :],
        multimask_output=True,
    )
    masks_array = np.asarray(masks)
    scores_array = np.asarray(scores, dtype=np.float64)
    if masks_array.ndim != 3 or masks_array.shape[0] == 0:
        raise RuntimeError("sam2_invalid_output: SAM2 returned no masks")
    return masks_array, scores_array


def _expanded_box(
    bbox: PersonBox,
    image_width: int,
    image_height: int,
    view: str,
) -> np.ndarray:
    box_width = max(1, bbox.x1 - bbox.x0)
    box_height = max(1, bbox.y1 - bbox.y0)
    normalized_view = view.strip().lower()
    horizontal_pad = 0.10 if normalized_view not in {"side", "profile"} else 0.14
    vertical_pad = 0.03
    pad_x = int(round(float(box_width) * horizontal_pad))
    pad_y = int(round(float(box_height) * vertical_pad))
    return np.array(
        [
            max(0, bbox.x0 - pad_x),
            max(0, bbox.y0 - pad_y),
            min(image_width - 1, bbox.x1 + pad_x),
            min(image_height - 1, bbox.y1 + pad_y),
        ],
        dtype=np.float32,
    )


def _select_valid_mask(
    masks: np.ndarray,
    scores: np.ndarray,
    view: str,
) -> Tuple[np.ndarray, float]:
    candidates: list[tuple[float, np.ndarray, float]] = []
    rejected: list[str] = []
    for index in range(masks.shape[0]):
        mask = _largest_component((masks[index] > 0).astype(np.uint8) * 255)
        if mask is None:
            rejected.append(f"idx={index}:empty_component")
            continue
        foreground_ratio = float((mask > 0).mean())
        vertical_extent = _vertical_extent(mask)
        sam_score = float(scores[index]) if index < len(scores) else 0.0
        reason = ""
        if vertical_extent < 0.55:
            reason = "person_too_small_or_cropped"
        elif not _foreground_ratio_ok(foreground_ratio, view):
            reason = "foreground_ratio"
        rejected.append(
            "idx={idx}:fg={fg:.3f},vext={vext:.3f},sam={sam:.3f},reason={reason}".format(
                idx=index,
                fg=foreground_ratio,
                vext=vertical_extent,
                sam=sam_score,
                reason=reason or "accepted",
            )
        )
        if reason:
            continue
        candidates.append((_selection_score(sam_score, foreground_ratio, vertical_extent), mask, sam_score))
    if not candidates:
        raise ValueError(
            "sam2_invalid_mask: SAM2 masks failed strict selection for "
            f"view={view!r}; candidates=" + " | ".join(rejected)
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    best = candidates[0]
    return best[1], best[2]


def _selection_score(sam_score: float, foreground_ratio: float, vertical_extent: float) -> float:
    return sam_score + foreground_ratio + (0.25 * vertical_extent)


def _foreground_ratio_ok(foreground_ratio: float, view: str) -> bool:
    normalized_view = view.strip().lower()
    if normalized_view in {"side", "profile"}:
        return 0.06 <= foreground_ratio <= 0.36
    return 0.10 <= foreground_ratio <= 0.42


def _largest_component(mask: np.ndarray) -> np.ndarray | None:
    binary = _close_small_gaps((mask > 0).astype(np.uint8))
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if component_count <= 1:
        return None
    largest_index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return cast(np.ndarray, ((labels == largest_index).astype(np.uint8) * 255))


def _close_small_gaps(binary: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cast(np.ndarray, closed)


def _vertical_extent(mask: np.ndarray) -> float:
    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    _, _, _, height = cv2.boundingRect(contour)
    return float(height) / float(mask.shape[0])


segment_person_sam2 = segment_person_strict
