# 3D Thin-Layer Linear Elasticity PINN (strong form)

Project scaffold that follows the requested workflow: geometry/BC specs, sampling, PINN model, physics operators, and training loop. Uses PyTorch with autograd for strains, stresses, and PDE residuals.

## Layout

- `configs/default.yaml` — dimensions, material props, sampling counts, loss weights, training schedule.
- `src/geometry/` — domain definitions and collocation sampling (interior, clamped sides, load patch, free top).
- `src/models/` — displacement MLP with optional Fourier features.
- `src/physics/` — strain, stress, and `-div(sigma)` residual operators.
- `src/training/` — losses and epoch routine wiring sampling → losses → optimizer step.
- `src/utils/` — logging helpers.
- `train.py` — entry point; loads config, builds model, runs Adam epochs, saves weights.
- `data/`, `notebooks/`, `outputs/`, `tests/` — drop FEM references, exploratory work, generated plots/weights, and checks.

## Quickstart

```bash
pip install torch pyyaml
python train.py --config configs/default.yaml
```

Extend `train.py` with L-BFGS and richer logging/validation as needed.
