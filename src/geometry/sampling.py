"""Sampling utilities for PINN collocation points."""

from __future__ import annotations

import torch

from .domain import DomainSpec, LoadPatch


def _to_device(tensor: torch.Tensor, device: torch.device | str | None) -> torch.Tensor:
    return tensor.to(device) if device is not None else tensor


def sample_interior(domain: DomainSpec, n_points: int, *, device: torch.device | str | None = None) -> torch.Tensor:
    """Uniform random sampling inside the volume with z oversampling."""

    xyz = torch.empty((n_points, 3)).uniform_(0.0, 1.0)
    xyz[:, 0] *= domain.Lx
    xyz[:, 1] *= domain.Ly
    # Slight bias toward the thin direction to capture gradients.
    xyz[:, 2] *= domain.H
    return _to_device(xyz, device)


def sample_clamped_sides(domain: DomainSpec, n_per_side: int, *, device: torch.device | str | None = None) -> torch.Tensor:
    """Sample Dirichlet boundary points on all side faces."""

    xs = torch.zeros((n_per_side, 3))
    xs[:, 0] = 0.0
    xs[:, 1] = torch.rand(n_per_side) * domain.Ly
    xs[:, 2] = torch.rand(n_per_side) * domain.H

    xe = torch.zeros((n_per_side, 3))
    xe[:, 0] = domain.Lx
    xe[:, 1] = torch.rand(n_per_side) * domain.Ly
    xe[:, 2] = torch.rand(n_per_side) * domain.H

    ys = torch.zeros((n_per_side, 3))
    ys[:, 0] = torch.rand(n_per_side) * domain.Lx
    ys[:, 1] = 0.0
    ys[:, 2] = torch.rand(n_per_side) * domain.H

    ye = torch.zeros((n_per_side, 3))
    ye[:, 0] = torch.rand(n_per_side) * domain.Lx
    ye[:, 1] = domain.Ly
    ye[:, 2] = torch.rand(n_per_side) * domain.H

    pts = torch.cat([xs, xe, ys, ye], dim=0)
    return _to_device(pts, device)


def sample_load_patch(domain: DomainSpec, patch: LoadPatch, n_points: int, *, device: torch.device | str | None = None) -> torch.Tensor:
    """Sample Neumann boundary points on the loaded top patch."""

    pts = torch.empty((n_points, 3))
    pts[:, 0] = torch.rand(n_points) * (patch.x_end - patch.x_start) + patch.x_start
    pts[:, 1] = torch.rand(n_points) * (patch.y_end - patch.y_start) + patch.y_start
    pts[:, 2] = domain.H
    return _to_device(pts, device)


def sample_free_top(domain: DomainSpec, patch: LoadPatch, n_points: int, *, device: torch.device | str | None = None) -> torch.Tensor:
    """Sample points on the free part of the top surface (outside the patch)."""

    pts = torch.empty((n_points, 3))
    count = 0
    while count < n_points:
        remaining = n_points - count
        candidates = torch.empty((remaining * 2, 2)).uniform_(0.0, 1.0)
        candidates[:, 0] *= domain.Lx
        candidates[:, 1] *= domain.Ly

        mask = ~(
            (candidates[:, 0] >= patch.x_start)
            & (candidates[:, 0] <= patch.x_end)
            & (candidates[:, 1] >= patch.y_start)
            & (candidates[:, 1] <= patch.y_end)
        )

        accepted = candidates[mask][:remaining]
        n_acc = accepted.shape[0]
        if n_acc == 0:
            continue

        pts[count : count + n_acc, 0:2] = accepted
        count += n_acc

    pts[:, 2] = domain.H
    return _to_device(pts, device)
