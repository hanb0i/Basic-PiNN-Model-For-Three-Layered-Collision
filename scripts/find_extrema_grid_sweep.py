#!/usr/bin/env python3
"""
Find true best/worst cases via gradient descent in parameter space.

Instead of random sampling (which misses extrema in high-D spaces), this script
uses gradient-based optimization to find parameter combinations that minimize
or maximize the PINN-vs-FEM MAE%.

Usage:
    cd /path/to/repo
    python scripts/find_extrema_grid_sweep.py

Environment variables:
    PINN_DEVICE        – torch device (default: auto-detect)
    PINN_N_STARTS      – number of random starting points (default: 20)
    PINN_OPTIM_STEPS   – optimization steps per start (default: 100)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPHS_DIR = REPO_ROOT / "graphs" / "figures"
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)


def _select_device():
    if os.getenv("PINN_FORCE_CPU", "0") == "1":
        return torch.device("cpu")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ═════════════════════════════════════════════════════════════════════════════
# THREE-LAYER PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

THREE_LAYER_DIR = REPO_ROOT / "three-layer-workflow"
PINN_WORKFLOW_DIR = REPO_ROOT / "pinn-workflow"
FEA_DIR = REPO_ROOT / "fea-workflow" / "solver"


def _tl_load_pinn(device):
    for mod_name in list(sys.modules.keys()):
        if mod_name in ("model", "pinn_config"):
            del sys.modules[mod_name]
    sys.path.insert(0, str(THREE_LAYER_DIR))
    sys.path.insert(0, str(FEA_DIR))
    import model as tl_model
    import pinn_config as tl_config

    model_path = Path(os.getenv("PINN_THREE_LAYER_MODEL") or PINN_WORKFLOW_DIR / "pinn_model.pth")
    pinn = tl_model.MultiLayerPINN().to(device)
    sd = torch.load(str(model_path), map_location=device, weights_only=True)
    sd = tl_model.adapt_legacy_state_dict(sd, pinn.state_dict())
    pinn.load_state_dict(sd, strict=False)
    pinn.eval()
    return pinn, tl_config


def _tl_predict_raw(pinn, cfg, device, xf, yf, zf, e1, e2, e3, t1, t2, t3):
    """Raw prediction without compliance scaling — for differentiability."""
    r = float(getattr(cfg, "RESTITUTION_REF", 0.5))
    mu = float(getattr(cfg, "FRICTION_REF", 0.3))
    v0 = float(getattr(cfg, "IMPACT_VELOCITY_REF", 1.0))
    pts = np.stack([xf, yf, zf] + [np.full_like(xf, v) for v in [e1, t1, e2, t2, e3, t3, r, mu, v0]], axis=1)
    with torch.no_grad():
        v = pinn(torch.tensor(pts, dtype=torch.float32, device=device)).cpu().numpy()
    return v


def _tl_apply_compliance(v, pts, cfg):
    """Apply compliance scaling to raw network output."""
    es = (pts[:, 3:4] + pts[:, 5:6] + pts[:, 7:8]) / 3
    ts = pts[:, 4:5] + pts[:, 6:7] + pts[:, 8:9]
    ep = float(getattr(cfg, "E_COMPLIANCE_POWER", 1.0))
    al = float(getattr(cfg, "THICKNESS_COMPLIANCE_ALPHA", 0.0))
    sc = float(getattr(cfg, "DISPLACEMENT_COMPLIANCE_SCALE", 1.0))
    hr = float(getattr(cfg, "H", 0.1))
    return sc * v / (es**ep) * (hr / np.clip(ts, 1e-8, None))**al


def _tl_run_fem(e1, e2, e3, t1, t2, t3):
    import fem_solver
    cfg = {
        "geometry": {"Lx": 1.0, "Ly": 1.0, "H": t1 + t2 + t3, "ne_x": 16, "ne_y": 16, "ne_z": 8},
        "material": {"E_layers": [float(e1), float(e2), float(e3)], "t_layers": [float(t1), float(t2), float(t3)], "nu": 0.3},
        "load_patch": {"pressure": 1.0, "x_start": 1 / 3, "x_end": 2 / 3, "y_start": 1 / 3, "y_end": 2 / 3},
    }
    return fem_solver.solve_three_layer_fem(cfg)


def _tl_compute_mae(pinn, cfg, device, e1, e2, e3, t1, t2, t3):
    """Compute MAE% for given parameters."""
    xn, yn, zn, u_fea = _tl_run_fem(e1, e2, e3, t1, t2, t3)
    xn, yn = [np.array(a, dtype=float) for a in [xn, yn]]
    u_fea = np.array(u_fea, dtype=float)
    H = t1 + t2 + t3

    xg, yg = np.meshgrid(xn, yn, indexing="ij")
    v = _tl_predict_raw(pinn, cfg, device, xg.ravel(), yg.ravel(), np.full(xg.size, H), e1, e2, e3, t1, t2, t3)

    # Build pts array for compliance scaling
    r = float(getattr(cfg, "RESTITUTION_REF", 0.5))
    mu = float(getattr(cfg, "FRICTION_REF", 0.3))
    v0 = float(getattr(cfg, "IMPACT_VELOCITY_REF", 1.0))
    pts = np.stack([xg.ravel(), yg.ravel(), np.full(xg.size, H)] + [np.full_like(xg.ravel(), val) for val in [e1, t1, e2, t2, e3, t3, r, mu, v0]], axis=1)
    u = _tl_apply_compliance(v, pts, cfg)
    u_pinn = u.reshape(len(xn), len(yn), 3)

    uz_fea = u_fea[:, :, -1, 2]
    uz_pinn = u_pinn[:, :, 2]
    denom = float(np.max(np.abs(uz_fea)))
    mae = 100.0 * float(np.mean(np.abs(uz_pinn - uz_fea))) / denom if denom > 0 else 0.0
    return mae


# ═════════════════════════════════════════════════════════════════════════════
# Gradient-based optimization (finite differences since FEM is not diffable)
# ═════════════════════════════════════════════════════════════════════════════

def optimize_mae(pinn, cfg, device, initial_params, steps=100, lr=0.5, maximize=False, bounds=None):
    """
    Use finite-difference gradient descent to find extrema of MAE%.
    
    Since FEM solver is not differentiable, we use numerical gradients.
    """
    params = np.array(initial_params, dtype=float).copy()
    if bounds is None:
        bounds = [(1.0, 10.0), (1.0, 10.0), (1.0, 10.0), (0.02, 0.10), (0.02, 0.10), (0.02, 0.10)]
    
    history = []
    best_params = params.copy()
    best_mae = _tl_compute_mae(pinn, cfg, device, *params)
    history.append((params.copy(), best_mae))
    
    eps = 0.1  # finite difference step
    
    for step in range(steps):
        # Compute gradient via finite differences
        grad = np.zeros(6)
        for i in range(6):
            params_plus = params.copy()
            params_plus[i] = min(params_plus[i] + eps, bounds[i][1])
            mae_plus = _tl_compute_mae(pinn, cfg, device, *params_plus)
            
            params_minus = params.copy()
            params_minus[i] = max(params_minus[i] - eps, bounds[i][0])
            mae_minus = _tl_compute_mae(pinn, cfg, device, *params_minus)
            
            grad[i] = (mae_plus - mae_minus) / (params_plus[i] - params_minus[i] + 1e-10)
        
        # Update parameters
        if maximize:
            params = params + lr * grad
        else:
            params = params - lr * grad
        
        # Clip to bounds
        for i in range(6):
            params[i] = np.clip(params[i], bounds[i][0], bounds[i][1])
        
        mae = _tl_compute_mae(pinn, cfg, device, *params)
        history.append((params.copy(), mae))
        
        if maximize:
            if mae > best_mae:
                best_mae = mae
                best_params = params.copy()
        else:
            if mae < best_mae:
                best_mae = mae
                best_params = params.copy()
        
        if step % 10 == 0:
            print(f"  Step {step}: MAE={mae:.2f}%, params=[{params[0]:.2f},{params[1]:.2f},{params[2]:.2f},{params[3]:.3f},{params[4]:.3f},{params[5]:.3f}]")
    
    return best_params, best_mae, history


# ═════════════════════════════════════════════════════════════════════════════
# Main: Multi-start optimization
# ═════════════════════════════════════════════════════════════════════════════

def main():
    n_starts = int(os.getenv("PINN_N_STARTS", "20"))
    n_steps = int(os.getenv("PINN_OPTIM_STEPS", "100"))
    
    device = _select_device()
    print(f"Using device: {device}")
    print(f"Multi-start optimization: {n_starts} random starts × {n_steps} steps each")
    
    pinn, cfg = _tl_load_pinn(device)
    
    # ── Find MINIMUM (best case) ───────────────────────────────────────────
    print("\n=== Finding MINIMUM MAE (best case) ===")
    best_overall_mae = float('inf')
    best_overall_params = None
    
    for start_idx in range(n_starts):
        # Random starting point
        np.random.seed(1000 + start_idx)
        init_params = np.random.uniform([1.0, 1.0, 1.0, 0.02, 0.02, 0.02], [10.0, 10.0, 10.0, 0.10, 0.10, 0.10])
        init_mae = _tl_compute_mae(pinn, cfg, device, *init_params)
        print(f"\nStart {start_idx+1}/{n_starts}: init MAE={init_mae:.2f}% at E=[{init_params[0]:.1f},{init_params[1]:.1f},{init_params[2]:.1f}] t=[{init_params[3]:.2f},{init_params[4]:.2f},{init_params[5]:.2f}]")
        
        opt_params, opt_mae, history = optimize_mae(pinn, cfg, device, init_params, steps=n_steps, lr=0.3, maximize=False)
        print(f"  → Optimized: MAE={opt_mae:.2f}% at E=[{opt_params[0]:.2f},{opt_params[1]:.2f},{opt_params[2]:.2f}] t=[{opt_params[3]:.3f},{opt_params[4]:.3f},{opt_params[5]:.3f}]")
        
        if opt_mae < best_overall_mae:
            best_overall_mae = opt_mae
            best_overall_params = opt_params.copy()
    
    print(f"\n*** BEST CASE FOUND ***")
    print(f"  MAE = {best_overall_mae:.2f}%")
    print(f"  E = [{best_overall_params[0]:.3f}, {best_overall_params[1]:.3f}, {best_overall_params[2]:.3f}]")
    print(f"  t = [{best_overall_params[3]:.4f}, {best_overall_params[4]:.4f}, {best_overall_params[5]:.4f}]")
    
    # ── Find MAXIMUM (worst case) ──────────────────────────────────────────
    print("\n=== Finding MAXIMUM MAE (worst case) ===")
    worst_overall_mae = 0.0
    worst_overall_params = None
    
    for start_idx in range(n_starts):
        np.random.seed(2000 + start_idx)
        init_params = np.random.uniform([1.0, 1.0, 1.0, 0.02, 0.02, 0.02], [10.0, 10.0, 10.0, 0.10, 0.10, 0.10])
        init_mae = _tl_compute_mae(pinn, cfg, device, *init_params)
        print(f"\nStart {start_idx+1}/{n_starts}: init MAE={init_mae:.2f}% at E=[{init_params[0]:.1f},{init_params[1]:.1f},{init_params[2]:.1f}] t=[{init_params[3]:.2f},{init_params[4]:.2f},{init_params[5]:.2f}]")
        
        opt_params, opt_mae, history = optimize_mae(pinn, cfg, device, init_params, steps=n_steps, lr=0.3, maximize=True)
        print(f"  → Optimized: MAE={opt_mae:.2f}% at E=[{opt_params[0]:.2f},{opt_params[1]:.2f},{opt_params[2]:.2f}] t=[{opt_params[3]:.3f},{opt_params[4]:.3f},{opt_params[5]:.3f}]")
        
        if opt_mae > worst_overall_mae:
            worst_overall_mae = opt_mae
            worst_overall_params = opt_params.copy()
    
    print(f"\n*** WORST CASE FOUND ***")
    print(f"  MAE = {worst_overall_mae:.2f}%")
    print(f"  E = [{worst_overall_params[0]:.3f}, {worst_overall_params[1]:.3f}, {worst_overall_params[2]:.3f}]")
    print(f"  t = [{worst_overall_params[3]:.4f}, {worst_overall_params[4]:.4f}, {worst_overall_params[5]:.4f}]")
    
    # Save results
    summary = {
        "method": "gradient_descent_finite_difference",
        "n_starts": n_starts,
        "n_steps": n_steps,
        "best": {
            "e1": float(best_overall_params[0]),
            "e2": float(best_overall_params[1]),
            "e3": float(best_overall_params[2]),
            "t1": float(best_overall_params[3]),
            "t2": float(best_overall_params[4]),
            "t3": float(best_overall_params[5]),
            "top_uz_mae_pct": float(best_overall_mae),
        },
        "worst": {
            "e1": float(worst_overall_params[0]),
            "e2": float(worst_overall_params[1]),
            "e3": float(worst_overall_params[2]),
            "t1": float(worst_overall_params[3]),
            "t2": float(worst_overall_params[4]),
            "t3": float(worst_overall_params[5]),
            "top_uz_mae_pct": float(worst_overall_mae),
        },
    }
    out_path = REPO_ROOT / "graphs" / "data" / "extrema_optimization_results.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
