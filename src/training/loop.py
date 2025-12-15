"""High-level training loop scaffolding for the PINN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import torch
from torch import optim

from src.geometry.domain import DomainSpec, LoadPatch
from src.geometry.sampling import sample_clamped_sides, sample_free_top, sample_interior, sample_load_patch
from src.training.losses import dirichlet_loss, free_surface_loss, interior_loss, traction_loss, weighted_total


@dataclass
class TrainingWeights:
    interior: float = 1.0
    dirichlet: float = 10.0
    load: float = 10.0
    free: float = 5.0


@dataclass
class TrainingSchedule:
    adam_steps: int = 5000
    adam_lr: float = 1e-3
    lbfgs: bool = False


def train_epoch(
    model: torch.nn.Module,
    domain: DomainSpec,
    patch: LoadPatch,
    E: float,
    nu: float,
    device: torch.device,
    n_interior: int,
    n_dirichlet: int,
    n_load: int,
    n_free: int,
    weights: TrainingWeights,
    optimizer: optim.Optimizer,
    refresh_sampler: Callable[[int], bool] | None = None,
    epoch: int | None = None,
) -> Dict[str, float]:
    """One epoch of PINN training with resampled collocation points."""

    refresh = True if refresh_sampler is None else refresh_sampler(epoch or 0)
    device = torch.device(device)

    if refresh:
        int_pts = sample_interior(domain, n_interior, device=device)
        dir_pts = sample_clamped_sides(domain, n_dirichlet, device=device)
        load_pts = sample_load_patch(domain, patch, n_load, device=device)
        free_pts = sample_free_top(domain, patch, n_free, device=device)
    else:
        raise ValueError("Sampling refresh strategy must resample at least once per epoch.")

    target_load = torch.tensor([[0.0, 0.0, -patch.pressure]], device=device).repeat(load_pts.shape[0], 1)
    normal_top = torch.tensor([0.0, 0.0, 1.0], device=device)

    optimizer.zero_grad()
    losses = {
        "interior": interior_loss(model, int_pts, E=E, nu=nu),
        "dirichlet": dirichlet_loss(model, dir_pts),
        "load": traction_loss(model, load_pts, target_load, E=E, nu=nu, normal=normal_top),
        "free": free_surface_loss(model, free_pts, E=E, nu=nu, normal=normal_top),
    }

    loss = weighted_total(
        losses,
        weights={
            "interior": weights.interior,
            "dirichlet": weights.dirichlet,
            "load": weights.load,
            "free": weights.free,
        },
    )
    loss.backward()
    optimizer.step()

    return {name: val.detach().item() for name, val in losses.items()} | {"total": loss.detach().item()}


def build_adam(model: torch.nn.Module, lr: float) -> optim.Optimizer:
    return optim.Adam(model.parameters(), lr=lr)
