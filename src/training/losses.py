"""Loss functions for the thin-layer elasticity PINN."""

from __future__ import annotations

import torch

from src.physics.elasticity import pde_residual, compute_strain, compute_stress


def interior_loss(model, coords: torch.Tensor, E: float, nu: float) -> torch.Tensor:
    coords.requires_grad_(True)
    u = model(coords)
    residual = pde_residual(u, coords, E=E, nu=nu)
    return (residual.pow(2).sum(dim=1)).mean()


def dirichlet_loss(model, coords: torch.Tensor) -> torch.Tensor:
    coords.requires_grad_(True)
    u = model(coords)
    return (u.pow(2).sum(dim=1)).mean()


def traction_loss(model, coords: torch.Tensor, target: torch.Tensor, E: float, nu: float, normal: torch.Tensor) -> torch.Tensor:
    coords.requires_grad_(True)
    u = model(coords)
    strain = compute_strain(u, coords)
    stress = compute_stress(strain, E=E, nu=nu)
    traction = torch.einsum("bij,j->bi", stress, normal)
    diff = traction - target
    return (diff.pow(2).sum(dim=1)).mean()


def free_surface_loss(model, coords: torch.Tensor, E: float, nu: float, normal: torch.Tensor) -> torch.Tensor:
    zero_target = torch.zeros((coords.shape[0], 3), device=coords.device, dtype=coords.dtype)
    return traction_loss(model, coords, zero_target, E=E, nu=nu, normal=normal)


def weighted_total(
    losses: dict[str, torch.Tensor],
    weights: dict[str, float],
) -> torch.Tensor:
    return sum(weights.get(name, 1.0) * loss for name, loss in losses.items())
