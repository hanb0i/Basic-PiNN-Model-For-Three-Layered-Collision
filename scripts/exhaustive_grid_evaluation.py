#!/usr/bin/env python3
"""
Exhaustive grid evaluation with fine resolution to find true best/worst cases.

Instead of random sampling, this evaluates on a regular grid with N points per
parameter. For a 6D space with N=5, this is 5^6 = 15,625 cases — manageable.

Usage:
    cd /path/to/repo
    python scripts/exhaustive_grid_evaluation.py

Environment variables:
    PINN_GRID_N        – points per parameter (default: 5, giving 5^6 = 15625 cases)
    PINN_DEVICE        – torch device (default: auto-detect)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

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


def _tl_predict(pinn, cfg, device, xf, yf, zf, e1, e2, e3, t1, t2, t3):
    r = float(getattr(cfg, "RESTITUTION_REF", 0.5))
    mu = float(getattr(cfg, "FRICTION_REF", 0.3))
    v0 = float(getattr(cfg, "IMPACT_VELOCITY_REF", 1.0))
    pts = np.stack([xf, yf, zf] + [np.full_like(xf, v) for v in [e1, t1, e2, t2, e3, t3, r, mu, v0]], axis=1)
    with torch.no_grad():
        v = pinn(torch.tensor(pts, dtype=torch.float32, device=device)).cpu().numpy()
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
    xn, yn, zn, u_fea = _tl_run_fem(e1, e2, e3, t1, t2, t3)
    xn, yn = [np.array(a, dtype=float) for a in [xn, yn]]
    u_fea = np.array(u_fea, dtype=float)
    H = t1 + t2 + t3

    xg, yg = np.meshgrid(xn, yn, indexing="ij")
    u_pinn = _tl_predict(pinn, cfg, device, xg.ravel(), yg.ravel(), np.full(xg.size, H), e1, e2, e3, t1, t2, t3).reshape(len(xn), len(yn), 3)

    uz_fea = u_fea[:, :, -1, 2]
    uz_pinn = u_pinn[:, :, 2]
    denom = float(np.max(np.abs(uz_fea)))
    mae = 100.0 * float(np.mean(np.abs(uz_pinn - uz_fea))) / denom if denom > 0 else 0.0
    return mae


# ═════════════════════════════════════════════════════════════════════════════
# Exhaustive grid evaluation
# ═════════════════════════════════════════════════════════════════════════════

def main():
    grid_n = int(os.getenv("PINN_GRID_N", "5"))
    
    device = _select_device()
    print(f"Using device: {device}")
    
    pinn, cfg = _tl_load_pinn(device)
    
    # Create grid points
    e_vals = np.linspace(1.0, 10.0, grid_n)
    t_vals = np.linspace(0.02, 0.10, grid_n)
    
    n_total = grid_n ** 6
    print(f"Grid resolution: {grid_n} points per parameter")
    print(f"E values: {e_vals}")
    print(f"t values: {t_vals}")
    print(f"Total cases: {grid_n}^6 = {n_total:,}")
    
    all_results = []
    best_mae = float('inf')
    worst_mae = 0.0
    best_params = None
    worst_params = None
    
    count = 0
    for e1 in e_vals:
        for e2 in e_vals:
            for e3 in e_vals:
                for t1 in t_vals:
                    for t2 in t_vals:
                        for t3 in t_vals:
                            mae = _tl_compute_mae(pinn, cfg, device, e1, e2, e3, t1, t2, t3)
                            all_results.append({
                                "e1": float(e1), "e2": float(e2), "e3": float(e3),
                                "t1": float(t1), "t2": float(t2), "t3": float(t3),
                                "mae": float(mae),
                            })
                            
                            if mae < best_mae:
                                best_mae = mae
                                best_params = (e1, e2, e3, t1, t2, t3)
                            if mae > worst_mae:
                                worst_mae = mae
                                worst_params = (e1, e2, e3, t1, t2, t3)
                            
                            count += 1
                            if count % 1000 == 0:
                                print(f"  Progress: {count}/{n_total} ({100*count/n_total:.1f}%) | best={best_mae:.2f}% worst={worst_mae:.2f}%")
    
    print(f"\n=== EXHAUSTIVE GRID RESULTS ({grid_n}^6 = {n_total:,} cases) ===")
    print(f"\nBEST CASE:")
    print(f"  MAE = {best_mae:.2f}%")
    print(f"  E = [{best_params[0]:.3f}, {best_params[1]:.3f}, {best_params[2]:.3f}]")
    print(f"  t = [{best_params[3]:.4f}, {best_params[4]:.4f}, {best_params[5]:.4f}]")
    
    print(f"\nWORST CASE:")
    print(f"  MAE = {worst_mae:.2f}%")
    print(f"  E = [{worst_params[0]:.3f}, {worst_params[1]:.3f}, {worst_params[2]:.3f}]")
    print(f"  t = [{worst_params[3]:.4f}, {worst_params[4]:.4f}, {worst_params[5]:.4f}]")
    
    # Statistics
    maes = [r["mae"] for r in all_results]
    print(f"\nSTATISTICS:")
    print(f"  Mean: {np.mean(maes):.2f}%")
    print(f"  Std:  {np.std(maes):.2f}%")
    print(f"  Median: {np.median(maes):.2f}%")
    print(f"  95th percentile: {np.percentile(maes, 95):.2f}%")
    
    # Save
    summary = {
        "method": f"exhaustive_grid_{grid_n}pts",
        "grid_n": grid_n,
        "n_cases": n_total,
        "e_values": e_vals.tolist(),
        "t_values": t_vals.tolist(),
        "best": {
            "e1": float(best_params[0]), "e2": float(best_params[1]), "e3": float(best_params[2]),
            "t1": float(best_params[3]), "t2": float(best_params[4]), "t3": float(best_params[5]),
            "mae": float(best_mae),
        },
        "worst": {
            "e1": float(worst_params[0]), "e2": float(worst_params[1]), "e3": float(worst_params[2]),
            "t1": float(worst_params[3]), "t2": float(worst_params[4]), "t3": float(worst_params[5]),
            "mae": float(worst_mae),
        },
        "statistics": {
            "mean": float(np.mean(maes)),
            "std": float(np.std(maes)),
            "median": float(np.median(maes)),
            "p95": float(np.percentile(maes, 95)),
            "p5": float(np.percentile(maes, 5)),
        },
    }
    out_path = REPO_ROOT / "graphs" / "data" / f"exhaustive_grid_{grid_n}pts_results.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
