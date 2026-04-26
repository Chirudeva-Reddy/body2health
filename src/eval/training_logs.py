"""Training-log parsing shared by 5-eval/1metrics.py and 5-eval/6plots.py."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

# Format produced by 3-train/1train.py at the end of every epoch.
_EPOCH_RE = re.compile(
    r"Epoch (\d+)/(\d+) \| Train Loss ([\d.]+) \| Val Loss ([\d.]+) \| "
    r"Val MAE BMI ([\d.]+) \| Val MAE BF ([\d.]+)%"
)

# Per-batch progress lines. Optional for callers that only need epoch summaries.
_BATCH_RE = re.compile(
    r"Epoch (\d+)/(\d+) \| Batch (\d+)/(\d+) \| Loss ([\d.]+) \| "
    r"Contrastive ([\d.]+) \| Regression ([\d.]+)"
)


@dataclass
class TrainingLogRow:
    epoch: int
    train_loss: float
    val_loss: float
    val_mae_bmi: float
    val_mae_bf: float


@dataclass
class BatchLogRow:
    epoch: int
    batch: int
    total_batches: int
    loss: float
    contrastive: float
    regression: float


def parse_training_log(path: str) -> List[TrainingLogRow]:
    """Read ``path`` and return one ``TrainingLogRow`` per epoch-summary line.

    Lines that do not match either the epoch or batch pattern are ignored
    (training logs interleave batch progress with epoch summaries; everything
    else is informational).
    """
    rows: List[TrainingLogRow] = []
    with open(path, "r") as f:
        for line in f:
            m = _EPOCH_RE.search(line)
            if m is None:
                continue
            epoch = int(m.group(1))
            train_loss = float(m.group(3))
            val_loss = float(m.group(4))
            val_mae_bmi = float(m.group(5))
            val_mae_bf = float(m.group(6))
            rows.append(
                TrainingLogRow(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_mae_bmi=val_mae_bmi,
                    val_mae_bf=val_mae_bf,
                )
            )
    return rows


def parse_batch_log(path: str) -> List[BatchLogRow]:
    """Read ``path`` and return one ``BatchLogRow`` per per-batch progress line."""
    rows: List[BatchLogRow] = []
    with open(path, "r") as f:
        for line in f:
            m = _BATCH_RE.search(line)
            if m is None:
                continue
            rows.append(
                BatchLogRow(
                    epoch=int(m.group(1)),
                    batch=int(m.group(3)),
                    total_batches=int(m.group(4)),
                    loss=float(m.group(5)),
                    contrastive=float(m.group(6)),
                    regression=float(m.group(7)),
                )
            )
    return rows
