from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .data import MarkerwiseNpyDataset
from .model import MarkerConditionedDenoiser


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(
    data_root: Path,
    marker_names: list[str],
    epochs: int,
    batch_size: int,
    lr: float,
    num_workers: int,
    save_dir: Path,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = MarkerwiseNpyDataset(data_root=data_root, marker_names=marker_names)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = MarkerConditionedDenoiser(marker_count=len(marker_names)).to(device)
    optim = AdamW(model.parameters(), lr=lr)

    save_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    for epoch in range(1, epochs + 1):
        running = 0.0
        for batch in dl:
            he = batch["he"].to(device)
            markers = batch["markers"].to(device)

            b, m, h, w = markers.shape
            marker_idx = torch.randint(0, m, (b,), device=device)

            clean = markers[torch.arange(b, device=device), marker_idx][:, None, :, :]
            noise = torch.randn_like(clean)
            noisy = clean + 0.1 * noise

            pred = model(he, noisy, marker_idx)
            loss = F.mse_loss(pred, noise)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            running += float(loss.item())

        avg = running / max(len(dl), 1)
        print(f"[epoch {epoch:03d}] loss={avg:.6f}")

        ckpt = {
            "model": model.state_dict(),
            "marker_names": marker_names,
        }
        torch.save(ckpt, save_dir / f"epoch_{epoch:03d}.pt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DiffVS starter model")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--markers", nargs="+", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--save-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    train(
        data_root=args.data_root,
        marker_names=args.markers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
