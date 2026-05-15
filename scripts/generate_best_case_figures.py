#!/usr/bin/env python3
"""
Generate FEA | PINN | Abs Error figures for the best-performing cases
from the 100-case random sweep (one-layer and three-layer).

Outputs (PNG + PDF) in graphs/figures/:
  best_1l_top.{png,pdf}       -- 1L top-surface
  best_1l_cross.{png,pdf}     -- 1L xz cross-section
  best_3l_top.{png,pdf}       -- 3L top-surface
  best_3l_cross.{png,pdf}     -- 3L xz cross-section
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "fea-workflow" / "solver"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import fem_solver

OUT = REPO / "graphs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

NE = (8, 8, 4)
nu = 0.3
PATCH = {"pressure": 1.0, "x_start": 1/3, "x_end": 2/3, "y_start": 1/3, "y_end": 2/3}

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "dejavuserif",
})
CMAP_F, CMAP_E = "jet", "magma"


# ── helpers ───────────────────────────────────────────────────────────────────

def mae_pct(pred, ref):
    d = float(np.max(np.abs(ref)))
    return 100 * float(np.mean(np.abs(pred - ref))) / d if d > 0 else 0.0


def save_fig(fig, stem):
    for ext in (".png", ".pdf"):
        p = OUT / f"{stem}{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"  Saved: {p}")


def plot_top(xg, yg, uz_fea, uz_pinn, title, param_str, stem):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    vmin = min(uz_fea.min(), uz_pinn.min())
    vmax = max(uz_fea.max(), uz_pinn.max())
    levels = np.linspace(vmin, vmax, 51)

    for ax, data, ttl in [
        (axes[0], uz_fea, f"{title} FEA\n{param_str}"),
        (axes[1], uz_pinn, f"{title} PINN"),
    ]:
        c = ax.contourf(xg, yg, data, levels=levels, cmap=CMAP_F)
        plt.colorbar(c, ax=ax)
        ax.set_title(ttl); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")

    err = np.abs(uz_pinn - uz_fea)
    m = mae_pct(uz_pinn, uz_fea)
    c = axes[2].contourf(xg, yg, err, levels=50, cmap=CMAP_E)
    plt.colorbar(c, ax=axes[2])
    axes[2].set_title(f"Abs Error\nMAE={m:.2f}%")
    axes[2].set_xlabel("x"); axes[2].set_ylabel("y"); axes[2].set_aspect("equal")

    fig.tight_layout()
    save_fig(fig, stem)
    plt.close(fig)
    print(f"  Top-surface MAE = {m:.2f}%")


def plot_cross(xc, zc, uz_fea, uz_pinn, title, interfaces, stem):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    vmin = min(uz_fea.min(), uz_pinn.min())
    vmax = max(uz_fea.max(), uz_pinn.max())
    levels = np.linspace(vmin, vmax, 51)

    for ax, data, ttl in [
        (axes[0], uz_fea, f"{title} FEA Cross-Section"),
        (axes[1], uz_pinn, f"{title} PINN Cross-Section"),
    ]:
        c = ax.contourf(xc, zc, data, levels=levels, cmap=CMAP_F)
        plt.colorbar(c, ax=ax)
        ax.set_title(ttl); ax.set_xlabel("x"); ax.set_ylabel("z")
        for z in (interfaces or []):
            ax.axhline(z, color="white", linewidth=3)

    err = np.abs(uz_pinn - uz_fea)
    m_abs = float(np.mean(err))
    c = axes[2].contourf(xc, zc, err, levels=50, cmap=CMAP_E)
    plt.colorbar(c, ax=axes[2])
    axes[2].set_title(f"Abs Error Cross-Section\nMAE={m_abs:.5f} m")
    axes[2].set_xlabel("x"); axes[2].set_ylabel("z")
    for z in (interfaces or []):
        axes[2].axhline(z, color="white", linewidth=3)

    fig.tight_layout()
    save_fig(fig, stem)
    plt.close(fig)
    print(f"  Cross-section MAE = {m_abs:.5f} m")


# ── Three-layer best ──────────────────────────────────────────────────────────

def run_3l():
    e1, e2, e3 = 1.312892, 7.527073, 6.895858
    t1, t2, t3 = 0.028637, 0.030723, 0.099804
    H = t1 + t2 + t3
    interfaces = [t1, t1 + t2]
    param_str = f"E=[{e1:.2f},{e2:.2f},{e3:.2f}], t=[{t1:.3f},{t2:.3f},{t3:.3f}]"

    # Clear sys.modules to avoid model/config conflicts
    for m in ["model", "pinn_config", "three_layer_experiment_utils", "one_layer_experiment_utils"]:
        sys.modules.pop(m, None)

    # Set calibration path for 3L
    os.environ["PINN_CALIBRATION_JSON"] = str(REPO / "graphs" / "data" / "three_layer_compliance_calibration.json")

    from three_layer_experiment_utils import (
        ThreeLayerCase, load_pinn as load_3l_pinn,
        make_points as make_points_3l, 
        predict_displacement as predict_displacement_3l, 
        solve_fem_case,
    )

    print(f"\n── 3L Best: {param_str} ──")
    device = torch.device("cpu")
    pinn3, _ = load_3l_pinn(device)
    case = ThreeLayerCase("best", e1, e2, e3, t1, t2, t3)

    print("  Running 3L FEM...")
    xn, yn, zn, u_fem, _ = solve_fem_case(case, *NE)
    xn, yn, zn = [np.array(a, dtype=float) for a in [xn, yn, zn]]
    u_fem = np.array(u_fem, dtype=float)

    xg, yg = np.meshgrid(xn, yn, indexing="ij")

    # Top surface
    uz_fea_top = u_fem[:, :, -1, 2]
    pts_top = make_points_3l(xg.ravel(), yg.ravel(), np.full(xg.size, H), case)
    uz_pinn_top = predict_displacement_3l(pinn3, device, pts_top).reshape(len(xn), len(yn), 3)[:, :, 2]

    plot_top(xg, yg, uz_fea_top, uz_pinn_top, "Three-Layer", param_str, "best_3l_top")

    # Cross-section at y=0.5
    mi = len(yn) // 2
    xc, zc = np.meshgrid(xn, zn, indexing="ij")
    uz_fea_cs = u_fem[:, mi, :, 2]
    pts_cs = make_points_3l(xc.ravel(), np.full(xc.size, yn[mi]), zc.ravel(), case)
    uz_pinn_cs = predict_displacement_3l(pinn3, device, pts_cs).reshape(len(xn), len(zn), 3)[:, :, 2]

    plot_cross(xc, zc, uz_fea_cs, uz_pinn_cs, "Three-Layer", interfaces, "best_3l_cross")


# ── One-layer best ────────────────────────────────────────────────────────────

def run_1l():
    E, t = 1.3644, 0.0832
    param_str = f"E={E:.2f}, t={t:.3f}"

    # Clear sys.modules to avoid model/config conflicts
    for m in ["model", "pinn_config", "three_layer_experiment_utils", "one_layer_experiment_utils"]:
        sys.modules.pop(m, None)

    # Set calibration path for 1L
    os.environ["PINN_CALIBRATION_JSON"] = str(REPO / "graphs" / "data" / "one_layer_compliance_calibration.json")

    from one_layer_experiment_utils import (
        OneLayerCase, load_pinn as load_1l_pinn,
        make_points as make_points_1l,
        predict_displacement as predict_displacement_1l,
    )

    print(f"\n── 1L Best: {param_str} ──")
    device = torch.device("cpu")
    
    # Load 1L model correctly using utils
    lbfgs_path = REPO / "one-layer-workflow" / "lbfgs_finetuned" / "pinn_model.pth"
    pinn1, _ = load_1l_pinn(device, model_path=lbfgs_path)
    case = OneLayerCase("best", E, t)

    # FEM
    print("  Running 1L FEM...")
    fem_cfg = {
        "geometry": {"Lx": 1.0, "Ly": 1.0, "H": t, "ne_x": NE[0], "ne_y": NE[1], "ne_z": NE[2]},
        "material": {"E": E, "nu": nu}, "load_patch": PATCH,
    }
    xn, yn, zn, u_fem = fem_solver.solve_fem(fem_cfg)
    xn, yn, zn = [np.array(a, dtype=float) for a in [xn, yn, zn]]
    u_fem = np.array(u_fem, dtype=float)

    xg, yg = np.meshgrid(xn, yn, indexing="ij")

    # Top surface
    uz_fea_top = u_fem[:, :, -1, 2]
    pts_top = make_points_1l(xg.ravel(), yg.ravel(), np.full(xg.size, t), case)
    uz_pinn_top = predict_displacement_1l(pinn1, device, pts_top).reshape(len(xn), len(yn), 3)[:, :, 2]
    plot_top(xg, yg, uz_fea_top, uz_pinn_top, "One-Layer", param_str, "best_1l_top")

    # Cross-section
    mi = len(yn) // 2
    xc, zc = np.meshgrid(xn, zn, indexing="ij")
    uz_fea_cs = u_fem[:, mi, :, 2]
    pts_cs = make_points_1l(xc.ravel(), np.full(xc.size, yn[mi]), zc.ravel(), case)
    uz_pinn_cs = predict_displacement_1l(pinn1, device, pts_cs).reshape(len(xn), len(zn), 3)[:, :, 2]
    plot_cross(xc, zc, uz_fea_cs, uz_pinn_cs, "One-Layer", None, "best_1l_cross")


if __name__ == "__main__":
    run_3l()
    run_1l()
    print("\nAll done.")
