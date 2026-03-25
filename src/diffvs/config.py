from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    root: Path
    he_glob: str = "**/*he*.tif"
    mihc_glob: str = "**/*mihc*.tif"
    marker_from_name_sep: str = "_"
    marker_from_name_index: int = -1
    image_size: int = 256
    train_split: float = 0.9
    seed: int = 42


@dataclass
class ModelConfig:
    latent_channels: int = 4
    unet_base_channels: int = 128
    marker_embed_dim: int = 64
    time_embed_dim: int = 128


@dataclass
class TrainConfig:
    batch_size: int = 8
    num_workers: int = 4
    epochs_diffusion: int = 50
    epochs_onestep: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-2
    save_every: int = 1
    output_dir: Path = Path("outputs")
    device: str = "cuda"


@dataclass
class LossConfig:
    lambda_noise: float = 1.0
    lambda_x0: float = 0.5
    lambda_l1: float = 1.0
    lambda_mse: float = 0.5


@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2


@dataclass
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)



def _to_path_if_needed(key: str, value):
    if key in {"root", "output_dir"}:
        return Path(value)
    return value


def load_config(path: Path) -> ExperimentConfig:
    raw = yaml.safe_load(path.read_text())

    data = DataConfig(**{k: _to_path_if_needed(k, v) for k, v in raw["data"].items()})
    model = ModelConfig(**raw.get("model", {}))
    train = TrainConfig(**{k: _to_path_if_needed(k, v) for k, v in raw.get("train", {}).items()})
    loss = LossConfig(**raw.get("loss", {}))
    diffusion = DiffusionConfig(**raw.get("diffusion", {}))
    return ExperimentConfig(data=data, model=model, train=train, loss=loss, diffusion=diffusion)
