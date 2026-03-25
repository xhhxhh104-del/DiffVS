from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class MarkerwiseNpyDataset(Dataset):
    """Simple dataset layout:

    data_root/
      sample_000/
        he.npy
        marker_CD3.npy
        marker_CD20.npy
      sample_001/
        ...
    """

    def __init__(self, data_root: Path, marker_names: Sequence[str]):
        self.data_root = Path(data_root)
        self.marker_names = list(marker_names)
        self.samples = sorted([p for p in self.data_root.iterdir() if p.is_dir()])
        if not self.samples:
            raise ValueError(f"No sample directories found under {self.data_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_npy(self, path: Path) -> torch.Tensor:
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample_dir = self.samples[idx]
        he = self._load_npy(sample_dir / "he.npy")

        markers = []
        for name in self.marker_names:
            markers.append(self._load_npy(sample_dir / f"marker_{name}.npy"))

        marker_tensor = torch.cat(markers, dim=0)
        return {
            "he": he,
            "markers": marker_tensor,
        }
