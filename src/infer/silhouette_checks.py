"""Silhouette envelope checks shared by the iPhone preprocessing pipeline and
inference entrypoints.

Replaces the body of ``is_valid_silhouette`` in ``pipeline/iphone_pipeline.py``
and the ad-hoc envelope checks scattered through the old 4-infer scripts. The
returned dict's shape is intentionally similar to ``validate_bodym_compatibility``
in the iPhone pipeline so callers can be ported one-for-one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

# Anatomical / segmentation rule thresholds. Values mirror the long-standing
# is_valid_silhouette implementation; tightening them is a behavior change and
# belongs in a separate review.
MIN_FOREGROUND_RATIO = 0.08
MAX_FOREGROUND_RATIO = 0.70
MIN_LARGEST_CC_SHARE = 0.25       # largest CC must be >=25% of total foreground
MIN_LARGEST_CC_OF_FRAME = 0.08    # ...and >=8% of the frame
MIN_ASPECT_RATIO = 1.2
MAX_ASPECT_RATIO = 4.0
MIN_TORSO_RATIO = 0.10            # central band 20%-70% of height
MIN_ROW_WIDTH_PX = 10             # rows with width below this don't count as occupied
MIN_OCCUPIED_ROW_FRACTION = 0.55


@dataclass
class EnvelopeReport:
    ok: bool
    failed: List[str]


def envelope_check(mask: np.ndarray) -> EnvelopeReport:
    """Validate that ``mask`` looks like a single, anatomically plausible person.

    Returns an ``EnvelopeReport`` with ``ok=True`` and an empty ``failed`` list
    when every rule passes; otherwise ``ok=False`` and ``failed`` lists the
    short identifiers of the rules that failed (in the order they were checked).
    """
    failed: List[str] = []

    if mask is None or mask.size == 0:
        return EnvelopeReport(ok=False, failed=["empty_input"])
    if mask.ndim != 2:
        return EnvelopeReport(ok=False, failed=["non_2d_input"])

    m = (mask > 0).astype(np.uint8)
    h, w = m.shape
    total_pixels = h * w
    if total_pixels == 0:
        return EnvelopeReport(ok=False, failed=["zero_size"])

    foreground_pixels = int(m.sum())
    if foreground_pixels == 0:
        return EnvelopeReport(ok=False, failed=["no_foreground"])
    fg_ratio = foreground_pixels / float(total_pixels)
    if fg_ratio < MIN_FOREGROUND_RATIO or fg_ratio > MAX_FOREGROUND_RATIO:
        failed.append("foreground_ratio")

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num_labels <= 1:
        return EnvelopeReport(ok=False, failed=failed + ["no_components"])
    largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    largest_area = int(stats[largest_idx, cv2.CC_STAT_AREA])
    if largest_area / float(foreground_pixels) < MIN_LARGEST_CC_SHARE:
        failed.append("dominant_component_share")
    if largest_area / float(total_pixels) < MIN_LARGEST_CC_OF_FRAME:
        failed.append("dominant_component_size")
    largest_mask = (labels == largest_idx).astype(np.uint8)

    ys, xs = np.where(largest_mask > 0)
    if ys.size == 0 or xs.size == 0:
        return EnvelopeReport(ok=False, failed=failed + ["empty_dominant_component"])
    bbox_h = int(ys.max() - ys.min() + 1)
    bbox_w = int(xs.max() - xs.min() + 1)
    if bbox_w == 0:
        return EnvelopeReport(ok=False, failed=failed + ["zero_width_bbox"])
    aspect_ratio = bbox_h / float(bbox_w)
    if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
        failed.append("aspect_ratio")

    torso_start = int(0.2 * h)
    torso_end = min(h, torso_start + int(0.5 * h))
    if torso_end <= torso_start:
        return EnvelopeReport(ok=False, failed=failed + ["torso_band_empty"])
    torso_region = largest_mask[torso_start:torso_end, :]
    torso_ratio = torso_region.sum() / float(torso_region.size)
    if torso_ratio < MIN_TORSO_RATIO:
        failed.append("torso_presence")

    widths_ok = 0
    for row in range(h):
        cols = np.where(largest_mask[row] > 0)[0]
        if cols.size > 0:
            row_width = int(cols.max() - cols.min() + 1)
            if row_width > MIN_ROW_WIDTH_PX:
                widths_ok += 1
    if widths_ok / float(h) < MIN_OCCUPIED_ROW_FRACTION:
        failed.append("row_width_continuity")

    return EnvelopeReport(ok=(len(failed) == 0), failed=failed)
