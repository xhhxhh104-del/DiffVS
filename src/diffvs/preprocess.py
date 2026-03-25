from __future__ import annotations

import argparse
import json
from pathlib import Path

from diffvs.config import load_config
from diffvs.data import discover_hemit_samples


def build_manifest(config_path: Path, output_path: Path) -> None:
    cfg = load_config(config_path)
    samples, markers = discover_hemit_samples(
        cfg.data.root,
        cfg.data.he_glob,
        cfg.data.mihc_glob,
        cfg.data.marker_from_name_sep,
        cfg.data.marker_from_name_index,
    )

    payload = {
        "markers": markers,
        "count": len(samples),
        "samples": [
            {
                "he": str(s.he_path),
                "mihc": str(s.mihc_path),
                "marker": s.marker_name,
            }
            for s in samples
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Manifest written: {output_path} (samples={len(samples)}, markers={len(markers)})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate/scan raw HEMIT tif dataset and export manifest")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("outputs/manifest.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build_manifest(args.config, args.output)


if __name__ == "__main__":
    main()
