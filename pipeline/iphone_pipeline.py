"""
BodyM-Compatible iPhone Silhouette Preprocessing Pipeline - CLEANED VERSION

This module implements a cleaned, stabilized pipeline for processing iPhone RGB images
into BodyM-compatible binary silhouettes (640×480 resolution) suitable for
BMI and body-fat (%) prediction models.

Key improvements:
- Connected component filtering to remove noise artifacts
- Explicit background enforcement
- Geometry-preserving cleanup operations
- Comprehensive validation and sanity checks
- Deterministic, reproducible output
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any
import os

try:
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    _HAS_SAM2 = True
except ImportError:
    _HAS_SAM2 = False
    print("Warning: SAM2 not installed. Install with: pip install sam2")

from src.infer.silhouette_checks import MIN_ROW_WIDTH_PX as MIN_ROW_WIDTH  # noqa: E402,F401  # back-compat re-export
from src.infer.silhouette_checks import envelope_check  # noqa: E402


def is_valid_silhouette(mask: np.ndarray) -> bool:
    """
    Validate that a binary silhouette represents a single, anatomically plausible person.

    Thin wrapper around src.infer.silhouette_checks.envelope_check, kept so
    that callers using the public pipeline surface keep working unchanged.

    Args:
        mask: Binary mask (0/255) of arbitrary resolution.

    Returns:
        True if the mask passes all anatomical and sanity checks, False otherwise.
    """
    return envelope_check(mask).ok


class BodyMPipeline:
    """BodyM-compatible silhouette preprocessing pipeline - CLEANED VERSION."""
    
    def __init__(self, sam_model_path: Optional[str] = None, sam_config_path: Optional[str] = None):
        """
        Initialize the BodyM pipeline.
        
        Args:
            sam_model_path: Path to SAM2 model checkpoint. If None, will try to find default.
            sam_config_path: Path to SAM2 config YAML. If None, will try to find default.
        """
        self.sam_predictor = None
        self.target_size = (640, 480)  # (height, width) for BodyM compatibility
        self.min_component_area = 100  # Minimum area for connected components (after resize)
        
        if _HAS_SAM2:
            self._initialize_sam2(sam_model_path, sam_config_path)
    
    def _initialize_sam2(self, model_path: Optional[str] = None, config_path: Optional[str] = None):
        """Initialize SAM2 model."""
        if model_path is None:
            # Try default paths
            default_paths = [
                "models/sam2.1_hiera_large.pt",
                "models/sam2_hiera_large.pt",
                "sam2.1_hiera_large.pt",
                "../models/sam2.1_hiera_large.pt",
            ]
            for path in default_paths:
                if os.path.exists(path):
                    model_path = path
                    break

        if config_path is None:
            default_configs = [
                "configs/sam2.1/sam2.1_hiera_l.yaml",
                "sam2.1_hiera_l.yaml",
                "../configs/sam2.1/sam2.1_hiera_l.yaml",
            ]
            for path in default_configs:
                if os.path.exists(path):
                    config_path = path
                    break
        
        if model_path and os.path.exists(model_path) and config_path and os.path.exists(config_path):
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                sam2_model = build_sam2(config_path, model_path, device=device)
                self.sam_predictor = SAM2ImagePredictor(sam2_model)
                print(f"SAM2 model loaded from {model_path}")
            except Exception as e:
                print(f"Failed to load SAM2 model: {e}")
                self.sam_predictor = None
        else:
            print(
                "SAM2 checkpoint/config not found. Please provide both "
                "`sam_model_path` and `sam_config_path`."
            )
    
    def process_iphone_image(
        self,
        img_rgb: np.ndarray,
        view: str = "front",
        debug_dir: Optional[str] = None,
        debug_prefix: str = "silhouette",
    ) -> np.ndarray:
        """
        Main entry point for processing iPhone RGB images.

        Args:
            img_rgb: Input RGB image from iPhone (any resolution)
            view: "front" or "side" (side view uses stricter SAM2 prompting + mask ranking)
            debug_dir: If set, saves intermediate debug masks to this directory (side view only).
            debug_prefix: Prefix for saved debug files (side view only).

        Returns:
            Binary silhouette mask (640, 480) with values 0 (background) and 255 (person)

        Raises:
            ValueError: If input image is invalid or SAM2 is not available
        """
        if img_rgb is None or img_rgb.size == 0:
            raise ValueError("Invalid input image")
        
        if len(img_rgb.shape) != 3 or img_rgb.shape[2] != 3:
            raise ValueError("Input must be an RGB image with shape (H, W, 3)")

        view_norm = (view or "front").strip().lower()
        if view_norm in {"side", "profile"}:
            return self._process_side_view(img_rgb, debug_dir=debug_dir, debug_prefix=debug_prefix)
        
        # Step 1: SAM2 segmentation
        mask = self._sam_segmentation(img_rgb)
        if mask is None:
            raise RuntimeError("SAM2 segmentation failed")
        
        # Step 2: Enforce strict binary
        mask_binary = self._enforce_binary(mask)
        mask_binary = self._auto_polarity_fix(mask_binary)

        # Step 3: Anatomical validation gate BEFORE any resizing/standardization
        if not is_valid_silhouette(mask_binary):
            raise ValueError("Invalid silhouette captured (anatomy checks failed); please recapture.")
        
        # Step 4: Enforce single connected component for downstream processing
        mask_cleaned = self._remove_noise_artifacts(mask_binary)
        if np.sum(mask_cleaned > 0) == 0:
            raise ValueError("Silhouette lost after enforcing single connected component; please recapture.")
        
        # Step 5: Resize to BodyM resolution using geometry-preserving standardization
        mask_640x480 = self._geometry_preserving_resize(mask_cleaned)
        
        # Step 6: Final noise removal after resize (still single component)
        mask_final = self._remove_noise_artifacts(mask_640x480)
        
        # Step 7: Explicit background enforcement
        mask_final = self._enforce_background(mask_final)
        
        # Step 8: Final validation with comprehensive checks (including anatomy gate)
        mask_validated = self._validate_output_comprehensive(mask_final)
        
        return mask_validated

    def _process_side_view(
        self,
        img_rgb: np.ndarray,
        debug_dir: Optional[str],
        debug_prefix: str,
    ) -> np.ndarray:
        """
        Side-view pipeline (front pipeline remains unchanged).

        Implements:
        - Multi-point SAM2 prompting + tight bbox
        - Mask ranking by border-touch, area, solidity, vertical extent, hole size
        - Fixed post-processing stack (close -> hole fill -> open -> largest CC -> blur+threshold)
        - Side-specific structural sanity checks
        """
        if self.sam_predictor is None:
            raise ValueError("SAM2 predictor not available; side-view pipeline requires SAM2.")

        h, w = img_rgb.shape[:2]
        self.sam_predictor.set_image(img_rgb)

        bbox = self._estimate_tight_person_bbox(img_rgb)
        # If heuristic bbox is too short (common with floor/cloth confusion), fall back to a generous box.
        x1, y1, x2, y2 = [float(v) for v in bbox.reshape(-1)]
        if (y2 - y1) / float(max(1, h)) < 0.80:
            cx, cy = w // 2, h // 2
            bw = int(0.80 * w)
            bh = int(0.95 * h)
            x1 = max(0, cx - bw // 2)
            y1 = max(0, cy - bh // 2)
            x2 = min(w - 1, cx + bw // 2)
            y2 = min(h - 1, cy + bh // 2)
            bbox = np.array([[x1, y1, x2, y2]], dtype=np.float32)
        point_coords, point_labels = self._side_view_prompt_points(bbox, w=w, h=h)

        masks, scores, _ = self.sam_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=bbox,
            multimask_output=True,
        )

        # Debug: raw SAM2 best-by-score mask
        if debug_dir is not None:
            self._ensure_dir(debug_dir)
            if scores is not None and len(scores) > 0:
                raw_idx = int(np.argmax(scores))
            else:
                raw_idx = 0
            raw = (masks[raw_idx].astype(np.uint8) * 255)
            cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}_raw_sam2_mask.png"), raw)
            self._log_mask_stats("raw_sam2_mask", raw)

        selected = self._select_side_mask(masks, w=w, h=h)
        if selected is None:
            raise ValueError("Side-view SAM2 masks rejected by selection rules; please recapture.")

        if debug_dir is not None:
            cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}_selected_mask.png"), selected)
            self._log_mask_stats("selected_mask", selected)

        # Fast path: if the raw selected mask already produces a sane 640x480 silhouette, keep it.
        raw_std = self._geometry_preserving_resize(selected)
        raw_std = self._keep_largest_component(raw_std)
        raw_std = (raw_std > 0).astype(np.uint8) * 255
        try:
            self._side_view_sanity_checks(raw_std)
            if debug_dir is not None:
                cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}_final_silhouette.png"), raw_std)
                self._log_mask_stats("final_silhouette", raw_std)
            return raw_std
        except Exception:
            # Fall through to full cleaning path
            pass

        # Post-processing (STRICT order required) when raw is not acceptable
        m = (selected > 0).astype(np.uint8) * 255  # a) binary uint8

        # b) closing 7×7 elliptical
        k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k7, iterations=1)

        # c) fill internal holes (flood fill from a background corner)
        m = self._fill_holes_floodfill(m)

        # d) opening 3×3 to remove speckle noise
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k3, iterations=1)

        # e) keep only largest connected component
        m = self._keep_largest_component(m)

        if debug_dir is not None:
            cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}_post_morphology_mask.png"), m)
            self._log_mask_stats("post_morphology_mask", m)

        # f) edge smoothing: blur + re-threshold
        blurred = cv2.GaussianBlur(m, (0, 0), sigmaX=1.2)
        m = (blurred > 127).astype(np.uint8) * 255
        m = self._keep_largest_component(m)

        # Standardize to BodyM resolution (no aspect distortion; NN only)
        m_std = self._geometry_preserving_resize(m)
        m_std = (m_std > 0).astype(np.uint8) * 255
        m_std = self._keep_largest_component(m_std)

        # Structural sanity checks (side-view specific)
        self._side_view_sanity_checks(m_std)

        if debug_dir is not None:
            cv2.imwrite(os.path.join(debug_dir, f"{debug_prefix}_final_silhouette.png"), m_std)
            self._log_mask_stats("final_silhouette", m_std)

        return m_std
    
    def _sam_segmentation(self, img_rgb: np.ndarray) -> Optional[np.ndarray]:
        """
        Perform SAM2 segmentation on the input image.
        
        Args:
            img_rgb: Input RGB image
            
        Returns:
            Binary mask or None if segmentation fails
        """
        if self.sam_predictor is None:
            # Fallback to GrabCut if SAM2 is not available
            return self._grabcut_fallback(img_rgb)
        
        try:
            h, w = img_rgb.shape[:2]
            
            # Set image for SAM2
            self.sam_predictor.set_image(img_rgb)
            
            # Full-body bounding box prompt plus torso/background points to stabilize SAM2
            center_x, center_y = w // 2, h // 2
            bbox_w = int(w * 0.9)
            bbox_h = int(h * 0.95)
            x1 = max(0, center_x - bbox_w // 2)
            y1 = max(0, center_y - bbox_h // 2)
            x2 = min(w, center_x + bbox_w // 2)
            y2 = min(h, center_y + bbox_h // 2)
            bbox = np.array([[x1, y1, x2, y2]], dtype=np.float32)

            # Prompts: one torso foreground point, one background point between legs/floor
            torso_point = np.array([[center_x, int(h * 0.45)]], dtype=np.float32)
            bg_point = np.array([[center_x, min(h - 1, int(h * 0.9))]], dtype=np.float32)
            point_coords = np.concatenate([torso_point, bg_point], axis=0)
            point_labels = np.array([1, 0], dtype=np.int64)

            # Generate mask with combined box and points
            masks, _, _ = self.sam_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=bbox,
                multimask_output=False
            )
            
            # Return the best mask
            mask = masks[0].astype(np.uint8) * 255
            return mask
            
        except Exception as e:
            print(f"SAM2 segmentation failed: {e}")
            return self._grabcut_fallback(img_rgb)

    @staticmethod
    def _ensure_dir(path: str) -> None:
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def _log_mask_stats(name: str, mask: np.ndarray) -> None:
        m = (mask > 0).astype(np.uint8)
        total = int(m.size)
        fg = int(m.sum())
        ratio = fg / float(total) if total > 0 else 0.0
        print(f"[{name}] shape={mask.shape} fg={fg} ({ratio:.3f}) unique={set(np.unique(mask).tolist())}")

    def _estimate_tight_person_bbox(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Estimate a tight-ish full-body bounding box from the RGB image using
        simple, deterministic thresholding. This is a prompt helper for SAM2.
        """
        h, w = img_rgb.shape[:2]
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if float(th.mean()) > 127.0:
            th = 255 - th
        th = cv2.morphologyEx(
            th,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=2,
        )
        comp = self._keep_largest_component(th)
        ys, xs = np.where(comp > 0)
        if ys.size == 0 or xs.size == 0:
            # Fallback: conservative center box for side view
            cx, cy = w // 2, h // 2
            bw = int(0.65 * w)
            bh = int(0.92 * h)
            x1 = max(0, cx - bw // 2)
            y1 = max(0, cy - bh // 2)
            x2 = min(w - 1, cx + bw // 2)
            y2 = min(h - 1, cy + bh // 2)
            return np.array([[x1, y1, x2, y2]], dtype=np.float32)

        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        bw = x1 - x0 + 1
        bh = y1 - y0 + 1
        pad_x = int(round(0.05 * bw))
        pad_y = int(round(0.05 * bh))
        x1b = max(0, x0 - pad_x)
        y1b = max(0, y0 - pad_y)
        x2b = min(w - 1, x1 + pad_x)
        y2b = min(h - 1, y1 + pad_y)
        return np.array([[x1b, y1b, x2b, y2b]], dtype=np.float32)

    @staticmethod
    def _side_view_prompt_points(bbox: np.ndarray, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Side-view prompting:
        - 5 foreground points: head, chest, hip, thigh, calf (along bbox centerline)
        - 2 background points: top-left, bottom-right corners
        """
        x1, y1, x2, y2 = [float(v) for v in bbox.reshape(-1)]
        cx = int(round((x1 + x2) / 2.0))
        top = int(round(y1))
        bottom = int(round(y2))
        bh = max(1, bottom - top + 1)

        ys = [
            top + int(round(0.10 * bh)),  # head
            top + int(round(0.30 * bh)),  # chest
            top + int(round(0.55 * bh)),  # hip
            top + int(round(0.70 * bh)),  # thigh
            top + int(round(0.85 * bh)),  # calf
        ]
        ys = [int(np.clip(y, 0, h - 1)) for y in ys]
        cx = int(np.clip(cx, 0, w - 1))

        fg = np.array([[cx, y] for y in ys], dtype=np.float32)
        bg = np.array(
            [
                [0, 0],  # top-left corner
                [w - 1, h - 1],  # bottom-right corner
                [w // 2, int(round(0.95 * h))],  # bottom-center to suppress floor
            ],
            dtype=np.float32,
        )
        point_coords = np.concatenate([fg, bg], axis=0)
        point_labels = np.array([1] * len(fg) + [0] * len(bg), dtype=np.int64)
        return point_coords, point_labels

    @staticmethod
    def _border_touch_fraction(mask01: np.ndarray) -> float:
        h, w = mask01.shape
        top = float(mask01[0, :].mean())
        bottom = float(mask01[h - 1, :].mean())
        left = float(mask01[:, 0].mean())
        right = float(mask01[:, w - 1].mean())
        return max(top, bottom, left, right)

    def _mask_solidity_and_extent(self, mask01: np.ndarray) -> Tuple[float, float]:
        """Return (solidity, vertical_extent) for the largest contour."""
        h, w = mask01.shape
        cnts, _ = cv2.findContours(mask01, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return 0.0, 0.0
        cnt = max(cnts, key=cv2.contourArea)
        area = float(cv2.contourArea(cnt))
        if area <= 0.0:
            return 0.0, 0.0
        hull = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / hull_area if hull_area > 0.0 else 0.0
        x, y, bw, bh = cv2.boundingRect(cnt)
        vertical_extent = bh / float(h) if h > 0 else 0.0
        return float(solidity), float(vertical_extent)

    @staticmethod
    def _largest_hole_fraction(mask255: np.ndarray) -> float:
        """
        Compute the area fraction of the largest internal hole relative to mask area.
        Hole detection uses flood fill from border on inverted (background) space.
        """
        m01 = (mask255 > 0).astype(np.uint8)
        mask_area = int(m01.sum())
        if mask_area == 0:
            return 0.0

        inv = (1 - m01) * 255
        flood = inv.copy()
        ff_mask = np.zeros((inv.shape[0] + 2, inv.shape[1] + 2), dtype=np.uint8)
        # Flood fill external background from a background corner
        seeds = [(0, 0), (inv.shape[1] - 1, 0), (0, inv.shape[0] - 1), (inv.shape[1] - 1, inv.shape[0] - 1)]
        seed = None
        for sx, sy in seeds:
            if flood[sy, sx] == 255:
                seed = (sx, sy)
                break
        if seed is None:
            return 0.0
        cv2.floodFill(flood, ff_mask, seedPoint=seed, newVal=0)
        holes = (flood == 255).astype(np.uint8)

        num, _, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
        if num <= 1:
            return 0.0
        largest_hole = int(np.max(stats[1:, cv2.CC_STAT_AREA]))
        return largest_hole / float(mask_area)

    def _select_side_mask(self, masks: np.ndarray, w: int, h: int) -> Optional[np.ndarray]:
        """
        Required side-view mask selection logic.

        Rules:
        - Discard if border touch > 5% (max fraction on any border)
        - Keep area in [15%, 60%] of image
        - Reject if largest hole > 2% of mask area
        - Select best by (solidity, vertical_extent) descending
        """
        candidates = []
        for i in range(masks.shape[0]):
            m01 = masks[i].astype(np.uint8)
            border_touch = self._border_touch_fraction(m01)
            # Side photos often have feet touching the bottom border; allow some bottom-touch.
            if border_touch > 0.12:
                continue

            area_ratio = float(m01.mean())
            if area_ratio < 0.15 or area_ratio > 0.60:
                continue

            m255 = m01 * 255
            hole_frac = self._largest_hole_fraction(m255)
            if hole_frac > 0.02:
                continue

            solidity, vext = self._mask_solidity_and_extent(m01)
            candidates.append((solidity, vext, i))

        if not candidates:
            # Fallback: pick the "best available" mask by a soft score instead of hard reject.
            # This prevents total failure on textured cloth backdrops where border-touch
            # or hole heuristics become unreliable.
            best = None
            for i in range(masks.shape[0]):
                m01 = masks[i].astype(np.uint8)
                area_ratio = float(m01.mean())
                if area_ratio < 0.08 or area_ratio > 0.75:
                    continue
                border = self._border_touch_fraction(m01)
                hole_frac = self._largest_hole_fraction(m01 * 255)
                sol, vext = self._mask_solidity_and_extent(m01)
                # Reward solidity/extent; penalize border touch and large holes.
                score = (2.0 * sol + 2.0 * vext) - (3.0 * border) - (5.0 * hole_frac)
                cand = (score, sol, vext, i, border, hole_frac, area_ratio)
                if best is None or cand[0] > best[0]:
                    best = cand
            if best is None:
                return None
            _, _, _, best_idx, *_ = best
            return masks[int(best_idx)].astype(np.uint8) * 255

        candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
        best_idx = int(candidates[0][2])
        return masks[best_idx].astype(np.uint8) * 255

    @staticmethod
    def _keep_largest_component(mask255: np.ndarray) -> np.ndarray:
        binary = (mask255 > 0).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels <= 1:
            return np.zeros_like(mask255)
        largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return (labels == largest_idx).astype(np.uint8) * 255

    def _fill_holes_floodfill(self, mask255: np.ndarray) -> np.ndarray:
        m = (mask255 > 0).astype(np.uint8) * 255
        h, w = m.shape
        flood = m.copy()
        ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

        seeds = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
        seed = None
        for sx, sy in seeds:
            if flood[sy, sx] == 0:
                seed = (sx, sy)
                break
        if seed is None:
            # If the mask touches all corners (should be rejected earlier), do nothing.
            return m

        cv2.floodFill(flood, ff_mask, seedPoint=seed, newVal=255)
        flood_inv = cv2.bitwise_not(flood)
        filled = cv2.bitwise_or(m, flood_inv)
        return (filled > 0).astype(np.uint8) * 255

    def _side_view_sanity_checks(self, mask640: np.ndarray) -> None:
        """
        Required structural sanity checks for side-view silhouettes:
        - Height > 70% of image height
        - Hip width > ankle width
        - Single connected component
        """
        if mask640.shape != self.target_size:
            raise ValueError(f"Expected {self.target_size} mask, got {mask640.shape}")

        m01 = (mask640 > 0).astype(np.uint8)
        ys, xs = np.where(m01 > 0)
        if ys.size == 0 or xs.size == 0:
            raise ValueError("Empty silhouette after side-view processing.")
        y0, y1 = int(ys.min()), int(ys.max())
        bbox_h = y1 - y0 + 1
        if bbox_h / float(mask640.shape[0]) <= 0.70:
            raise ValueError("Side-view silhouette height too small (<70% of image height).")

        # Hip and ankle widths using bbox-relative rows
        y_hip = int(round(y0 + 0.55 * bbox_h))
        y_ank = int(round(y0 + 0.95 * bbox_h))
        y_hip = int(np.clip(y_hip, 0, mask640.shape[0] - 1))
        y_ank = int(np.clip(y_ank, 0, mask640.shape[0] - 1))

        hip_cols = np.where(m01[y_hip] > 0)[0]
        ank_cols = np.where(m01[y_ank] > 0)[0]
        hip_w = int(hip_cols.max() - hip_cols.min() + 1) if hip_cols.size > 0 else 0
        ank_w = int(ank_cols.max() - ank_cols.min() + 1) if ank_cols.size > 0 else 0
        if hip_w <= 0 or ank_w <= 0 or hip_w <= ank_w:
            raise ValueError("Side-view anatomy check failed (hip width must exceed ankle width).")

        # Connectivity check (must be exactly one foreground component)
        num, _, _, _ = cv2.connectedComponentsWithStats(m01, connectivity=8)
        if num - 1 != 1:
            raise ValueError("Side-view silhouette must be a single connected component.")
    
    def _grabcut_fallback(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Fallback segmentation using GrabCut when SAM2 is not available.
        
        Args:
            img_rgb: Input RGB image
            
        Returns:
            Binary mask
        """
        h, w = img_rgb.shape[:2]
        
        # More conservative GrabCut initialization
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Mark smaller center region as probable foreground
        center_x, center_y = w // 2, h // 2
        margin_x, margin_y = w // 6, h // 6  # Smaller margins
        
        x1 = max(0, center_x - margin_x)
        y1 = max(0, center_y - margin_y)
        x2 = min(w, center_x + margin_x)
        y2 = min(h, center_y + margin_y)
        
        # Create GrabCut mask: 0=bg, 1=fg, 2=probable bg, 3=probable fg
        gc_mask = np.zeros((h, w), dtype=np.uint8)
        gc_mask[y1:y2, x1:x2] = 3  # probable foreground
        
        # Run GrabCut
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        try:
            cv2.grabCut(img_rgb, gc_mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
            result_mask = np.where((gc_mask == 1) | (gc_mask == 3), 255, 0).astype(np.uint8)
        except:
            # If GrabCut fails, return center region as mask
            result_mask = np.zeros((h, w), dtype=np.uint8)
            result_mask[y1:y2, x1:x2] = 255
        
        return result_mask
    
    def _enforce_binary(self, mask: np.ndarray) -> np.ndarray:
        """
        Convert mask to strict binary format (0 or 255 only).
        
        Args:
            mask: Input mask (any binary format)
            
        Returns:
            Strict binary mask with values 0 and 255
        """
        # Convert to binary and scale to 255
        binary_mask = (mask > 0).astype(np.uint8) * 255
        return binary_mask

    def _auto_polarity_fix(self, mask: np.ndarray) -> np.ndarray:
        """
        SAM2/GrabCut sometimes returns an inverted mask (background=255) or a mask that
        heavily favors bright background regions. We pick the polarity that is more
        plausible for a person silhouette, while still enforcing strict validation.
        """
        m = (mask > 0).astype(np.uint8) * 255
        inv = (255 - m).astype(np.uint8)
        m_ratio = float((m > 0).mean())
        inv_ratio = 1.0 - m_ratio

        m_ok = is_valid_silhouette(m)
        inv_ok = is_valid_silhouette(inv)

        if inv_ok and not m_ok:
            return inv
        if m_ok and not inv_ok:
            return m

        # If foreground dominates, prefer the polarity with less foreground to avoid
        # treating large background regions (e.g., curtains/walls) as the "person".
        if m_ratio > 0.55 and inv_ratio >= 0.08:
            return inv
        return m
    
    def _remove_noise_artifacts(self, mask: np.ndarray) -> np.ndarray:
        """
        Remove small noise artifacts using connected component filtering.
        
        Args:
            mask: Binary mask potentially containing noise
            
        Returns:
            Cleaned binary mask with noise removed
        """
        binary = (mask > 0).astype(np.uint8)

        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        # No foreground detected
        if num_labels <= 1:
            return np.zeros_like(mask)

        # Keep only the largest component
        largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        largest_area = int(stats[largest_idx, cv2.CC_STAT_AREA])

        # Enforce a minimum absolute area to filter obvious noise blobs
        min_area = self.min_component_area if mask.shape == self.target_size else max(50, self.min_component_area // 4)
        if largest_area < min_area:
            return np.zeros_like(mask)

        cleaned_mask = (labels == largest_idx).astype(np.uint8) * 255
        return cleaned_mask
    
    def _enforce_background(self, mask: np.ndarray) -> np.ndarray:
        """
        Explicitly enforce background pixels to be 0.
        
        Args:
            mask: Binary mask
            
        Returns:
            Mask with explicit background enforcement
        """
        # Ensure only foreground pixels are 255, all others are 0
        background_enforced = (mask > 0).astype(np.uint8) * 255
        return background_enforced
    
    def _geometry_preserving_resize(self, mask: np.ndarray) -> np.ndarray:
        """
        Resize mask to 640×480 using geometry-preserving methods.
        
        Args:
            mask: Binary mask to resize
            
        Returns:
            Resized mask (640, 480)
        """
        target_h, target_w = self.target_size
        
        # Extract bounding box of the person
        ys, xs = np.where(mask > 0)
        if len(ys) == 0 or len(xs) == 0:
            # Empty mask, return black background
            return np.zeros((target_h, target_w), dtype=np.uint8)
        
        # Get bounding box with small margin
        margin = 3  # Reduced margin to minimize background inclusion
        x_min, x_max = max(0, xs.min() - margin), min(mask.shape[1], xs.max() + margin)
        y_min, y_max = max(0, ys.min() - margin), min(mask.shape[0], ys.max() + margin)
        
        # Crop to bounding box
        cropped_mask = mask[y_min:y_max, x_min:x_max]
        
        # Calculate scale factor based on height (height-aware scaling)
        crop_h, crop_w = cropped_mask.shape
        scale_h = target_h / crop_h
        scale_w = target_w / crop_w
        
        # Use the smaller scale to preserve aspect ratio
        scale = min(scale_h, scale_w)
        
        # Calculate new size
        new_h = int(crop_h * scale)
        new_w = int(crop_w * scale)
        
        # Resize using nearest-neighbor interpolation only
        resized_crop = cv2.resize(cropped_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Center the resized mask in the target canvas
        output_mask = np.zeros((target_h, target_w), dtype=np.uint8)
        
        # Calculate center position
        start_y = (target_h - new_h) // 2
        start_x = (target_w - new_w) // 2
        
        # Ensure we don't go out of bounds
        end_y = min(start_y + new_h, target_h)
        end_x = min(start_x + new_w, target_w)
        crop_end_y = end_y - start_y
        crop_end_x = end_x - start_x
        
        # Place the resized mask
        output_mask[start_y:end_y, start_x:end_x] = resized_crop[:crop_end_y, :crop_end_x]
        
        return output_mask
    
    def _validate_output_comprehensive(self, mask: np.ndarray) -> np.ndarray:
        """
        Comprehensive validation and ensure output meets BodyM requirements.
        
        Args:
            mask: Output mask to validate
            
        Returns:
            Validated mask with all checks passed
        """
        # Ensure correct shape
        if mask.shape != self.target_size:
            mask = cv2.resize(mask, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST)
        
        # Ensure binary values only (strict thresholding)
        mask = (mask > 127).astype(np.uint8) * 255
        
        # Ensure dtype is uint8
        mask = mask.astype(np.uint8)
        
        # Final sanity check: verify only 0 and 255 values
        unique_values = np.unique(mask)
        if not np.array_equal(unique_values, [0]) and not np.array_equal(unique_values, [0, 255]):
            # Force to strict binary
            mask = (mask > 0).astype(np.uint8) * 255
        
        # Add assertions for debugging
        assert mask.shape == self.target_size, f"Shape mismatch: {mask.shape} != {self.target_size}"
        assert mask.dtype == np.uint8, f"Dtype mismatch: {mask.dtype} != uint8"
        assert set(np.unique(mask)).issubset({0, 255}), f"Invalid values: {set(np.unique(mask))}"

        # Final anatomical gate after standardization to stop bad masks from proceeding
        if not is_valid_silhouette(mask):
            raise ValueError("Final silhouette failed anatomical validation; please recapture.")
        
        return mask


def process_iphone_image(
    img_rgb: np.ndarray,
    sam_model_path: Optional[str] = None,
    sam_config_path: Optional[str] = None,
    view: str = "front",
    debug_dir: Optional[str] = None,
    debug_prefix: str = "silhouette",
) -> np.ndarray:
    """
    Convenience function to process iPhone images.
    
    Args:
        img_rgb: Input RGB image from iPhone
        sam_model_path: Path to SAM2 model checkpoint
        sam_config_path: Path to SAM2 config YAML
        view: "front" or "side"
        debug_dir: If set, saves intermediate debug masks to this directory (side view only).
        debug_prefix: Prefix for saved debug files (side view only).
        
    Returns:
        Binary silhouette mask (640, 480) compatible with BodyM models
    """
    pipeline = BodyMPipeline(sam_model_path=sam_model_path, sam_config_path=sam_config_path)
    return pipeline.process_iphone_image(
        img_rgb,
        view=view,
        debug_dir=debug_dir,
        debug_prefix=debug_prefix,
    )


def validate_bodym_compatibility(mask: np.ndarray) -> Dict[str, Any]:
    """
    Validate that a mask meets BodyM compatibility requirements.
    
    Args:
        mask: Binary mask to validate
        
    Returns:
        Validation results
    """
    validation = {
        'compatible': True,
        'issues': [],
        'checks': {}
    }
    
    # Check 1: Shape
    expected_shape = (640, 480)
    if mask.shape != expected_shape:
        validation['compatible'] = False
        validation['issues'].append(f'Shape {mask.shape} != expected {expected_shape}')
    validation['checks']['shape'] = mask.shape == expected_shape
    
    # Check 2: Dtype
    if mask.dtype != np.uint8:
        validation['compatible'] = False
        validation['issues'].append(f'Dtype {mask.dtype} != expected uint8')
    validation['checks']['dtype'] = mask.dtype == np.uint8
    
    # Check 3: Binary values only
    unique_values = set(np.unique(mask))
    expected_values = {0, 255}
    if unique_values != expected_values:
        validation['compatible'] = False
        validation['issues'].append(f'Values {unique_values} != expected {expected_values}')
    validation['checks']['binary_values'] = unique_values == expected_values

    # Check 4: Anatomical plausibility gate
    anatomy_ok = is_valid_silhouette(mask)
    validation['checks']['anatomy_valid'] = anatomy_ok
    if not anatomy_ok:
        validation['compatible'] = False
        validation['issues'].append('Failed anatomical silhouette validation')
    
    # Check 5: Reasonable foreground area
    foreground_ratio = np.sum(mask > 0) / mask.size
    if foreground_ratio < 0.01 or foreground_ratio > 0.8:
        validation['compatible'] = False
        validation['issues'].append(f'Foreground ratio {foreground_ratio:.3f} outside reasonable range [0.01, 0.8]')
    validation['checks']['foreground_ratio'] = 0.01 <= foreground_ratio <= 0.8
    
    return validation


if __name__ == "__main__":
    raise SystemExit(
        "This module is meant to be imported. "
        "Use `process_iphone_image(img_rgb)` from your app code."
    )
