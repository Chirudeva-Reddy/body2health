import random
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _resolve_path(path_str: str, root: Path) -> str:
    p = Path(path_str)
    if p.exists():
        return str(p)
    repo_relative = root / path_str
    if repo_relative.exists():
        return str(repo_relative)
    return path_str


def _load_mask(path: str, target_hw: Tuple[int, int], augment: bool = False, root: Optional[Path] = None) -> torch.Tensor:
    real_path = _resolve_path(path, root or Path(__file__).resolve().parents[2])
    img = cv2.imread(real_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(real_path)
    h_t, w_t = int(target_hw[0]), int(target_hw[1])
    if img.shape != (h_t, w_t):
        interp = cv2.INTER_AREA if (img.shape[0] > h_t or img.shape[1] > w_t) else cv2.INTER_NEAREST
        img = cv2.resize(img, (w_t, h_t), interpolation=interp)
    if augment:
        # Light jitter: random erode/dilate or small shift
        if random.random() < 0.5:
            k = random.choice([3, 3, 5])
            op = cv2.MORPH_ERODE if random.random() < 0.5 else cv2.MORPH_DILATE
            img = cv2.morphologyEx(img, op, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        if random.random() < 0.5:
            tx = random.randint(-2, 2)
            ty = random.randint(-2, 2)
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    mask = (img > 127).astype(np.float32)
    t = torch.from_numpy(mask).unsqueeze(0)  # (1,H,W)
    return t


class SilhouetteDataset(Dataset):
    """
    Reads paired silhouettes and dimension targets from CSV.

    Required CSV columns: front_path, side_path, subject_key, capture_key, and
    the requested measurement columns.
    """

    def __init__(self,
                 csv_path: str,
                 target_hw: Tuple[int, int] = (640, 480),
                 measurement_cols: Optional[List[str]] = None,
                 augment: bool = False):
        self.rows = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.rows.append(row)
        self.target_hw = target_hw
        self.measurement_cols = measurement_cols or ["waist_cm", "hip_cm"]
        self.augment = augment
        self._parsed: List[Dict] = []
        for row in self.rows:
            meas_vals: List[float] = []
            for c in self.measurement_cols:
                if c not in row:
                    raise KeyError(f"measurement column {c!r} is missing from {csv_path!r}")
                try:
                    meas_vals.append(float(row[c]))
                except ValueError as exc:
                    raise ValueError(
                        f"measurement column {c!r} has non-numeric value {row[c]!r} "
                        f"for subject_key={row.get('subject_key', '<missing>')}"
                    ) from exc
            self._parsed.append({
                "front_path": row["front_path"],
                "side_path": row["side_path"],
                "meas": meas_vals,
                "subject_key": row.get("subject_key", ""),
                "capture_key": row.get("capture_key", ""),
            })

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        root = Path(__file__).resolve().parents[2]
        parsed = self._parsed[idx]
        front = _load_mask(parsed["front_path"], self.target_hw, augment=self.augment, root=root)
        side = _load_mask(parsed["side_path"], self.target_hw, augment=self.augment, root=root)

        y_meas = torch.tensor(parsed["meas"], dtype=torch.float32)

        return {
            "front": front,
            "side": side,
            "y_meas": y_meas,
            "subject_key": parsed["subject_key"],
            "capture_key": parsed["capture_key"],
        }
