# Pipeline step 5: PyTorch Dataset for gray PNG crops (no dependency on other src steps).
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def knee_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, str]],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.stack([b[1] for b in batch], dim=0)
    stems = [b[2] for b in batch]
    return xs, ys, stems


class KneeKLGrayscaleDataset(Dataset):
    """Single-channel PNG crops; returns float tensor [1,H,W] in [0,1]."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        image_size: int = 224,
        augment: bool = False,
    ) -> None:
        self.records = records
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        r = self.records[idx]
        path = Path(r["gray_png"])
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Cannot read {path}")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        x = img.astype(np.float32) / 255.0
        if self.augment:
            if np.random.random() < 0.5:
                x = np.fliplr(x).copy()
            if np.random.random() < 0.2:
                noise = np.random.normal(0, 0.02, x.shape).astype(np.float32)
                x = np.clip(x + noise, 0.0, 1.0)
        t = torch.from_numpy(x).unsqueeze(0)
        y = torch.tensor(int(r["kl"]), dtype=torch.long)
        return t, y, str(r["stem"])
