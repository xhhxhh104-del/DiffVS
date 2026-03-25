from __future__ import annotations

import argparse
from pathlib import Path

from diffvs.config import load_config
from diffvs.engine.inferencer import DiffVSInferencer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Infer all markers from a single HE tif")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--he-tif", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    inferencer = DiffVSInferencer(cfg, args.checkpoint)
    inferencer.predict_all_markers(args.he_tif, args.output_dir)


if __name__ == "__main__":
    main()
