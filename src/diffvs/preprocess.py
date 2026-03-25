from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def image_to_npy(input_path: Path, output_path: Path) -> None:
    img = Image.open(input_path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    np.save(output_path, arr)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert image to normalized grayscale npy")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image_to_npy(args.input, args.output)


if __name__ == "__main__":
    main()
