"""Entry point for training the thin-layer elasticity PINN.

Usage:
    python train.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from src.geometry.domain import DomainSpec, LoadPatch
from src.models.mlp import DisplacementMLP
from src.training.loop import TrainingSchedule, TrainingWeights, build_adam, train_epoch
from src.utils.logging import format_losses


def load_config(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def main(config_path: str) -> None:
    cfg = load_config(Path(config_path))

    domain = DomainSpec(**cfg["domain"])
    patch = LoadPatch(**cfg["load_patch"])
    material = cfg["material"]

    device = torch.device(cfg.get("device", "cpu"))
    model = DisplacementMLP(**cfg["model"]).to(device)

    weights = TrainingWeights(**cfg["loss_weights"])
    schedule = TrainingSchedule(**cfg["schedule"])

    optimizer = build_adam(model, lr=schedule.adam_lr)

    for epoch in range(schedule.adam_steps):
        losses = train_epoch(
            model=model,
            domain=domain,
            patch=patch,
            E=material["E"],
            nu=material["nu"],
            device=device,
            n_interior=cfg["sampling"]["interior"],
            n_dirichlet=cfg["sampling"]["dirichlet_per_side"],
            n_load=cfg["sampling"]["load_patch"],
            n_free=cfg["sampling"]["free_top"],
            weights=weights,
            optimizer=optimizer,
            epoch=epoch,
        )

        if (epoch + 1) % cfg.get("log_every", 100) == 0:
            print(f"Epoch {epoch+1}: {format_losses(losses)}")

    # Placeholder for optional L-BFGS fine-tuning
    if schedule.lbfgs:
        print("L-BFGS phase not implemented in skeleton; extend as needed.")

    output_dir = Path(cfg.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pth")
    print(f"Saved model to {output_dir / 'model.pth'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
