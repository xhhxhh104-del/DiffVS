from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .model import MarkerConditionedDenoiser


def load_he(path: Path) -> torch.Tensor:
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    return torch.from_numpy(arr)[None, ...]


def infer(checkpoint: Path, he_path: Path, output_dir: Path, steps: int = 1) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint, map_location=device)
    marker_names = ckpt["marker_names"]

    model = MarkerConditionedDenoiser(marker_count=len(marker_names)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    he = load_he(he_path).to(device)
    b, _, h, w = he.shape

    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for marker_idx, marker_name in enumerate(marker_names):
            x = torch.randn((b, 1, h, w), device=device)
            idx = torch.tensor([marker_idx], device=device)

            for _ in range(steps):
                pred_noise = model(he, x, idx)
                x = x - pred_noise

            out = x.squeeze(0).squeeze(0).cpu().numpy()
            np.save(output_dir / f"pred_{marker_name}.npy", out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Infer marker channels from H&E")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--he-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    infer(args.checkpoint, args.he_path, args.output_dir, steps=args.steps)


if __name__ == "__main__":
    main()
