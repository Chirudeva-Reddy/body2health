import csv
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Dict
import random
from pathlib import Path


def _bmi(height_cm: float, weight_kg: float) -> float:
    h_m = max(1e-6, height_cm / 100.0)
    return float(weight_kg) / (h_m * h_m)


def _resolve_path(path_str: str, root: Path) -> str:
    """
    Fix absolute paths that still point to the old DesignProject location by remapping
    to this repo's data copy/raw folder.
    """
    p = Path(path_str)
    if p.exists():
        return str(p)
    for token in ["data/raw/", "data /raw/", "data copy/raw/"]:
        if token in path_str:
            suffix = path_str.split(token, 1)[1]  # e.g., mask/filename.png
            for base in ["data copy", "data", "data "]:
                candidate = root / base / "raw" / suffix
                if candidate.exists():
                    return str(candidate)
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
    Reads paired silhouettes and targets from CSV.
    Expects columns: front_path, side_path, height_cm, weight_kg, bf_pseudo (or bf), plus any extra measurement columns.
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
        self.measurement_cols = measurement_cols or ["bmi"]
        self.augment = augment
        # Precompute labels as float to avoid repeated parsing
        self._parsed: List[Dict] = []
        for row in self.rows:
            height_cm = float(row.get("height_cm", 0.0))
            weight_kg = float(row.get("weight_kg", 0.0))
            bf_raw = float(row.get("bf_pseudo", row.get("bf", 0.0)))
            # Store body-fat as fraction [0,1] to match model sigmoid output
            bf = max(0.0, min(1.0, bf_raw / 100.0))
            meas_vals: List[float] = []
            for c in self.measurement_cols:
                if c == "bmi":
                    meas_vals.append(_bmi(height_cm, weight_kg))
                else:
                    meas_vals.append(float(row.get(c, 0.0)))
            self._parsed.append({
                "front_path": row["front_path"],
                "side_path": row["side_path"],
                "meas": meas_vals,
                "bf": bf,
            })

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        root = Path(__file__).resolve().parents[2]
        parsed = self._parsed[idx]
        front = _load_mask(parsed["front_path"], self.target_hw, augment=self.augment, root=root)
        side = _load_mask(parsed["side_path"], self.target_hw, augment=self.augment, root=root)

        y_meas = torch.tensor(parsed["meas"], dtype=torch.float32)
        y_bf = torch.tensor([parsed["bf"]], dtype=torch.float32)

        return {
            "front": front,
            "side": side,
            "y_meas": y_meas,
            "y_bf": y_bf,
        }
