"""Elasticity operators using autograd for PINNs."""

from __future__ import annotations

import torch


def compute_strain(u: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Symmetric gradient of displacement.

    u: (batch, 3) displacement
    coords: (batch, 3) coordinates with gradients enabled
    returns: (batch, 3, 3) strain tensor
    """

    grads = torch.autograd.grad(
        u,
        coords,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True,
    )[0]
    grad_u = grads.view(-1, 3, 3)
    return 0.5 * (grad_u + grad_u.transpose(1, 2))


def compute_stress(strain: torch.Tensor, E: float, nu: float) -> torch.Tensor:
    """Hooke's law for isotropic linear elasticity in 3D."""

    lam = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    trace = torch.einsum("bii->b", strain)
    identity = torch.eye(3, device=strain.device, dtype=strain.dtype).expand_as(strain)
    return lam * trace[:, None, None] * identity + 2 * mu * strain


def divergence_of_stress(stress: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Compute divergence of stress tensor for equilibrium equations."""

    outputs = []
    for i in range(3):
        component = stress[:, :, i]
        grad = torch.autograd.grad(
            component,
            coords,
            grad_outputs=torch.ones_like(component),
            retain_graph=True,
            create_graph=True,
        )[0]
        outputs.append(grad.sum(dim=1))
    return torch.stack(outputs, dim=1)


def pde_residual(u: torch.Tensor, coords: torch.Tensor, E: float, nu: float) -> torch.Tensor:
    """Strong-form residual: -div(sigma(u))."""

    strain = compute_strain(u, coords)
    stress = compute_stress(strain, E=E, nu=nu)
    div_sigma = divergence_of_stress(stress, coords)
    return -div_sigma
