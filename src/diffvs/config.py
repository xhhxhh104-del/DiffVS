from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    data_root: Path
    marker_names: list[str]
    image_size: int = 256
    batch_size: int = 4
    epochs: int = 20
    lr: float = 1e-4
    num_workers: int = 2
    save_dir: Path = Path("checkpoints")
    seed: int = 42


@dataclass
class InferenceConfig:
    checkpoint: Path
    marker_names: list[str]
    image_size: int = 256
    steps: int = 1
    device: str = "cpu"
