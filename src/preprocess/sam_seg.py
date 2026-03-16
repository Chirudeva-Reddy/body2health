import cv2
import numpy as np
from typing import Optional, Tuple, Any

try:
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    _HAS_SAM2 = True
except Exception:
    _HAS_SAM2 = False
    SAM2ImagePredictor = Any  # type: ignore

try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except Exception:
    _HAS_YOLO = False
    YOLO = Any  # type: ignore

_SAM2_PREDICTOR: Optional[SAM2ImagePredictor] = None
_YOLO_MODEL: Optional[YOLO] = None


def _get_sam2_predictor(
    checkpoint_path: str = "models/sam2.1_hiera_large.pt",
    config_path: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
) -> SAM2ImagePredictor:
    """Load SAM2 predictor lazily.

    The user must download the checkpoint and place it at ``checkpoint_path``.
    """
    global _SAM2_PREDICTOR
    if not _HAS_SAM2:
        raise RuntimeError("sam2 is not installed; install it to use SAM2.")
    if _SAM2_PREDICTOR is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam2_model = build_sam2(config_path, checkpoint_path, device=device)
        _SAM2_PREDICTOR = SAM2ImagePredictor(sam2_model)
    return _SAM2_PREDICTOR


def _get_yolo(model_path: str = "models/yolo11m.pt") -> YOLO:
    # Loads YOLO once and reuses it.
    global _YOLO_MODEL
    if not _HAS_YOLO:
        raise RuntimeError("ultralytics is not installed; install it to use YOLO bbox prompts.")
    if _YOLO_MODEL is None:
        _YOLO_MODEL = YOLO(model_path)
    return _YOLO_MODEL


def _bbox_yolo(img: np.ndarray, model_path: str = "models/yolo11m.pt", conf: float = 0.35) -> Tuple[int, int, int, int]:
    # Runs person detection (COCO class 0) and picks highest-confidence box.
    m = _get_yolo(model_path)
    r = m.predict(source=img, classes=[0], conf=conf, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        raise ValueError("YOLO found no person in the image.")
    k = int(r.boxes.conf.argmax().item())
    x1, y1, x2, y2 = r.boxes.xyxy[k].cpu().numpy().astype(int).tolist()
    return x1, y1, x2, y2


def segment_person_sam(img: np.ndarray,
                       bbox: Optional[Tuple[int, int, int, int]] = None,
                       checkpoint_path: str = "models/sam2.1_hiera_large.pt",
                       config_path: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
                       yolo_model_path: str = "models/yolo11m.pt",
                       yolo_conf: float = 0.35) -> np.ndarray:
    """Segment person using SAM2 with a box prompt.

    If ``bbox`` is None, bbox is obtained from YOLO person detection.
    Returns a binary mask (uint8, 0/255) for the largest connected component.
    """
    if not _HAS_SAM2:
        raise RuntimeError("sam2 not available; install it to use SAM2.")

    # Uses caller bbox if provided; otherwise gets one from YOLO.
    if bbox is None:
        bbox = _bbox_yolo(img, model_path=yolo_model_path, conf=yolo_conf)
    x0, y0, x1, y1 = bbox
    predictor = _get_sam2_predictor(checkpoint_path=checkpoint_path, config_path=config_path)
    predictor.set_image(img)
    box = np.array([x0, y0, x1, y1], dtype=np.float32)
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box[None, :],
        multimask_output=True,
    )
    best = None
    best_key = None

    # Picks best mask by compactness first, then area, then SAM score.
    for i in range(masks.shape[0]):
        m = (masks[i] > 0).astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if num <= 1:
            continue
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        c = (labels == k).astype(np.uint8)
        a = int(c.sum())
        r = float(a) / float(max(1, m.sum()))
        s = float(scores[i]) if scores is not None else 0.0
        key = (r, a, s)
        if best is None or key > best_key:
            best = c
            best_key = key

    if best is None:
        raise RuntimeError("SAM2 returned no valid mask.")
    return (best * 255).astype(np.uint8)


segment_person_sam2 = segment_person_sam
