from __future__ import annotations

import argparse
from pathlib import Path

from diffvs.config import load_config
from diffvs.engine.trainer import DiffVSTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DiffVS according to marker-conditioned latent diffusion pipeline")
    p.add_argument("--config", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    trainer = DiffVSTrainer(cfg)
    trainer.fit()


if __name__ == "__main__":
    main()
