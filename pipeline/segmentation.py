"""
DEPRECATED MODULE

The historical segmentation + standardization implementation was archived to:
`archives/src/preprocess/segmentation.py`.

The maintained iPhone preprocessing entry point is:
`2-pipeline/pipeline/iphone_pipeline.py:process_iphone_image`.

This stub intentionally avoids importing heavy/optional dependencies (MediaPipe,
scipy, rembg, etc.) and only provides a compatibility shim.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def process_iphone_image(
    img_rgb: np.ndarray,
    target_hw: Tuple[int, int] = (640, 480),
    *,
    sam_model_path: Optional[str] = None,
    view: str = "front",
    debug_dir: Optional[str] = None,
    debug_prefix: str = "silhouette",
) -> np.ndarray:
    """
    Compatibility shim for older imports.

    For the actual implementation, see `2-pipeline/pipeline/iphone_pipeline.py`.
    """
    from .iphone_pipeline import process_iphone_image as _process

    # Canonical pipeline currently enforces BodyM-compatible output at (640, 480).
    _ = target_hw
    return _process(
        img_rgb,
        sam_model_path=sam_model_path,
        view=view,
        debug_dir=debug_dir,
        debug_prefix=debug_prefix,
    )
