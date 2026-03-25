from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class HEMITSample:
    he_path: Path
    mihc_path: Path
    marker_name: str


def _read_tif(path: Path) -> np.ndarray:
    arr = tifffile.imread(path)
    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            arr = arr[..., :3].mean(axis=-1)
        else:
            arr = arr.mean(axis=0)
    return arr.astype(np.float32)


def _resize(arr: np.ndarray, image_size: int) -> np.ndarray:
    img = Image.fromarray(arr)
    img = img.resize((image_size, image_size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


def _normalize(arr: np.ndarray) -> np.ndarray:
    vmax = np.percentile(arr, 99.5)
    vmin = np.percentile(arr, 0.5)
    arr = np.clip((arr - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    return arr


class HEMITMarkerwiseDataset(Dataset):
    def __init__(
        self,
        samples: list[HEMITSample],
        marker_to_idx: dict[str, int],
        image_size: int,
    ):
        self.samples = samples
        self.marker_to_idx = marker_to_idx
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.samples[idx]
        he = _normalize(_resize(_read_tif(s.he_path), self.image_size))
        mihc = _normalize(_resize(_read_tif(s.mihc_path), self.image_size))

        he_t = torch.from_numpy(he)[None, ...]
        mihc_t = torch.from_numpy(mihc)[None, ...]
        marker_idx = torch.tensor(self.marker_to_idx[s.marker_name], dtype=torch.long)
        return {
            "he": he_t,
            "target": mihc_t,
            "marker_idx": marker_idx,
            "marker_name": s.marker_name,
            "he_path": str(s.he_path),
            "mihc_path": str(s.mihc_path),
        }


def discover_hemit_samples(
    root: Path,
    he_glob: str,
    mihc_glob: str,
    sep: str,
    marker_idx: int,
) -> tuple[list[HEMITSample], list[str]]:
    he_files = sorted(root.glob(he_glob))
    mihc_files = sorted(root.glob(mihc_glob))
    if not he_files or not mihc_files:
        raise ValueError("No matching HE or mIHC tif files found. Check globs in config.")

    he_by_parent = {p.parent: p for p in he_files}
    samples: list[HEMITSample] = []
    markers: set[str] = set()

    for mp in mihc_files:
        he_path = he_by_parent.get(mp.parent)
        if he_path is None:
            continue
        stem_parts = mp.stem.split(sep)
        marker_name = stem_parts[marker_idx] if stem_parts else mp.stem
        markers.add(marker_name)
        samples.append(HEMITSample(he_path=he_path, mihc_path=mp, marker_name=marker_name))

    if not samples:
        raise ValueError("No paired HE/mIHC files found under same parent directory.")

    return samples, sorted(markers)
