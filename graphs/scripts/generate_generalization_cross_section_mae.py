#!/usr/bin/env python3
"""
Generate high-quality cross-section absolute MAE heatmaps.
Uses 16x16x8 mesh but with high-DPI output and smooth interpolation.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "graphs" / "generalized_study" / "cross_section_mae"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT / "fea-workflow" / "solver"))
import fem_solver

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

N_CASES = 100


def find_best_worst(csv_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    sorted_rows = sorted(rows, key=lambda r: float(r['top_uz_mae_pct']))
    return sorted_rows[0], sorted_rows[-1]


def load_one_layer_pinn():
    sys.path.insert(0, str(REPO_ROOT / "one-layer-workflow"))
    import model as ol_model
    import pinn_config as ol_config
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    pinn = ol_model.MultiLayerPINN().to(device)
    ckpt = REPO_ROOT / "one-layer-workflow" / "pinn_model.pth"
    sd = torch.load(ckpt, map_location=device, weights_only=True)
    w_key = "layer.net.0.weight"
    if w_key in sd and w_key in pinn.state_dict():
        src_w = sd[w_key]
        tgt_w = pinn.state_dict()[w_key]
        if src_w.shape != tgt_w.shape and src_w.shape[0] == tgt_w.shape[0]:
            if src_w.shape[1] == 8 and tgt_w.shape[1] == 11:
                adapted = torch.zeros_like(tgt_w)
                adapted[:, 0:5] = src_w[:, 0:5]
                adapted[:, 8:11] = src_w[:, 5:8]
                sd[w_key] = adapted
            elif src_w.shape[1] == 10 and tgt_w.shape[1] == 11:
                adapted = torch.zeros_like(tgt_w)
                adapted[:, 0:7] = src_w[:, 0:7]
                adapted[:, 8:11] = src_w[:, 7:10]
                sd[w_key] = adapted
    pinn.load_state_dict(sd, strict=False)
    pinn.eval()
    return pinn, device, ol_config


def load_three_layer_pinn():
    ol_path = str(REPO_ROOT / "one-layer-workflow")
    if ol_path in sys.path:
        sys.path.remove(ol_path)
    for mod_name in list(sys.modules.keys()):
        if mod_name in ("model", "pinn_config", "data", "physics", "soap"):
            del sys.modules[mod_name]
    sys.path.insert(0, str(REPO_ROOT / "three-layer-workflow"))
    import model as tl_model
    import pinn_config as tl_config
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    pinn = tl_model.MultiLayerPINN().to(device)
    ckpt = REPO_ROOT / "three-layer-workflow" / "pinn_model_final.pth"
    if not ckpt.exists():
        ckpt = REPO_ROOT / "three-layer-workflow" / "pinn_model.pth"
    sd = torch.load(ckpt, map_location=device, weights_only=True)
    if hasattr(tl_model, "adapt_legacy_state_dict"):
        sd = tl_model.adapt_legacy_state_dict(sd, pinn.state_dict())
    pinn.load_state_dict(sd, strict=False)
    pinn.eval()
    return pinn, device, tl_config


def run_one_layer_fem(E_val, thickness, ne_x=16, ne_y=16, ne_z=8):
    cfg = {
        "geometry": {"Lx": 1.0, "Ly": 1.0, "H": thickness, "ne_x": ne_x, "ne_y": ne_y, "ne_z": ne_z},
        "material": {"E": float(E_val), "nu": 0.3},
        "load_patch": {"pressure": 1.0, "x_start": 1/3, "x_end": 2/3, "y_start": 1/3, "y_end": 2/3},
    }
    xn, yn, zn, u_grid = fem_solver.solve_fem(cfg)
    return np.array(xn), np.array(yn), np.array(zn), np.array(u_grid)


def run_three_layer_fem(e1, e2, e3, t1, t2, t3, ne_x=16, ne_y=16, ne_z=8):
    thickness = float(t1) + float(t2) + float(t3)
    cfg = {
        "geometry": {"Lx": 1.0, "Ly": 1.0, "H": thickness, "ne_x": ne_x, "ne_y": ne_y, "ne_z": ne_z},
        "material": {"E_layers": [float(e1), float(e2), float(e3)], "t_layers": [float(t1), float(t2), float(t3)], "nu": 0.3},
        "load_patch": {"pressure": 1.0, "x_start": 1/3, "x_end": 2/3, "y_start": 1/3, "y_end": 2/3},
    }
    xn, yn, zn, u_grid = fem_solver.solve_three_layer_fem(cfg)
    return np.array(xn), np.array(yn), np.array(zn), np.array(u_grid)


def evaluate_one_layer(pinn, device, config, E_val, thickness, xn, yn, zn):
    X, Y, Z = np.meshgrid(xn, yn, zn, indexing='ij')
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    e_ones = np.ones((pts.shape[0], 1)) * float(E_val)
    t_ones = np.ones((pts.shape[0], 1)) * float(thickness)
    r_ref = float(getattr(config, "RESTITUTION_REF", 0.5))
    mu_ref = float(getattr(config, "FRICTION_REF", 0.3))
    v0_ref = float(getattr(config, "IMPACT_VELOCITY_REF", 1.0))
    r_ones = np.ones((pts.shape[0], 1)) * r_ref
    mu_ones = np.ones((pts.shape[0], 1)) * mu_ref
    v0_ones = np.ones((pts.shape[0], 1)) * v0_ref
    pts = np.hstack([pts, e_ones, t_ones, r_ones, mu_ones, v0_ones])
    pts_t = torch.tensor(pts, dtype=torch.float32).to(device)
    with torch.no_grad():
        v_pred = pinn(pts_t, 0).cpu().numpy()
    e_pow = float(getattr(config, "E_COMPLIANCE_POWER", 1.0))
    alpha = float(getattr(config, "THICKNESS_COMPLIANCE_ALPHA", 0.0))
    H_ref = float(getattr(config, "H", 1.0))
    t_scale = 1.0 if alpha == 0.0 else (H_ref / np.clip(float(thickness), 1e-8, None)) ** alpha
    u_pred = (v_pred / (float(E_val) ** e_pow)) * t_scale
    return u_pred.reshape(X.shape + (3,))


def evaluate_three_layer(pinn, device, config, e1, e2, e3, t1, t2, t3, xn, yn, zn):
    X, Y, Z = np.meshgrid(xn, yn, zn, indexing='ij')
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    e1_ones = np.ones((pts.shape[0], 1)) * float(e1)
    e2_ones = np.ones((pts.shape[0], 1)) * float(e2)
    e3_ones = np.ones((pts.shape[0], 1)) * float(e3)
    t1_ones = np.ones((pts.shape[0], 1)) * float(t1)
    t2_ones = np.ones((pts.shape[0], 1)) * float(t2)
    t3_ones = np.ones((pts.shape[0], 1)) * float(t3)
    r_ref = float(getattr(config, "RESTITUTION_REF", 0.5))
    mu_ref = float(getattr(config, "FRICTION_REF", 0.3))
    v0_ref = float(getattr(config, "IMPACT_VELOCITY_REF", 1.0))
    r_ones = np.ones((pts.shape[0], 1)) * r_ref
    mu_ones = np.ones((pts.shape[0], 1)) * mu_ref
    v0_ones = np.ones((pts.shape[0], 1)) * v0_ref
    pts = np.hstack([pts, e1_ones, e2_ones, e3_ones, t1_ones, t2_ones, t3_ones, r_ones, mu_ones, v0_ones])
    pts_t = torch.tensor(pts, dtype=torch.float32).to(device)
    with torch.no_grad():
        v_pred = pinn(pts_t, 0).cpu().numpy()
    e_pow = float(getattr(config, "E_COMPLIANCE_POWER", 1.0))
    alpha = float(getattr(config, "THICKNESS_COMPLIANCE_ALPHA", 0.0))
    H_ref = float(getattr(config, "H", 1.0))
    thickness = float(t1) + float(t2) + float(t3)
    t_scale = 1.0 if alpha == 0.0 else (H_ref / np.clip(thickness, 1e-8, None)) ** alpha
    E_avg = (float(e1) + float(e2) + float(e3)) / 3.0
    u_pred = (v_pred / (E_avg ** e_pow)) * t_scale
    return u_pred.reshape(X.shape + (3,))


def upsample_for_display(xn, zn, data, n_pts=500):
    """Upsample coarse FEM data for smooth visualization."""
    interp = RegularGridInterpolator((xn, zn), data, method='cubic', bounds_error=False, fill_value=0)
    x_fine = np.linspace(xn.min(), xn.max(), n_pts)
    z_fine = np.linspace(zn.min(), zn.max(), n_pts)
    Xf, Zf = np.meshgrid(x_fine, z_fine, indexing='ij')
    return x_fine, z_fine, interp((Xf, Zf))


def plot_cross_section_mae(ax, xn, zn, fem_uz, pinn_uz, title, mae_pct, abs_err, case_info, interfaces=None):
    abs_error = np.abs(pinn_uz - fem_uz)
    
    # Upsample for smooth display
    x_fine, z_fine, err_fine = upsample_for_display(xn, zn, abs_error, n_pts=500)
    vmax = np.max(err_fine)
    
    im = ax.imshow(
        err_fine.T,
        extent=[x_fine.min(), x_fine.max(), z_fine.min(), z_fine.max()],
        origin='lower',
        cmap='magma',
        aspect='auto',
        interpolation='bilinear',
        vmin=0,
        vmax=vmax,
    )
    
    ax.set_xlabel("$x$", fontsize=11)
    ax.set_ylabel("$z/H$ (normalized thickness)", fontsize=11)
    ax.set_title(f"{title}\n{case_info}\nMAE={mae_pct:.2f}% | Abs={abs_err:.5f}", fontsize=10)
    
    if interfaces:
        for z in interfaces:
            ax.axhline(z, color="white", linestyle="--", linewidth=1.5, alpha=0.9)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("$|u_z^{\\mathrm{PINN}} - u_z^{\\mathrm{FEM}}|$", fontsize=9)
    
    return im


def main():
    print("Finding best/worst cases from CSV data...")
    one_best, one_worst = find_best_worst(
        REPO_ROOT / "graphs" / "generalized_study" / "one_layer_random_100.csv"
    )
    three_best, three_worst = find_best_worst(
        REPO_ROOT / "graphs" / "generalized_study" / "three_layer_random_100.csv"
    )
    
    print(f"One-layer best:  {one_best['case_id']}  MAE={one_best['top_uz_mae_pct']}%")
    print(f"One-layer worst: {one_worst['case_id']}  MAE={one_worst['top_uz_mae_pct']}%")
    print(f"Three-layer best:  {three_best['case_id']}  MAE={three_best['top_uz_mae_pct']}%")
    print(f"Three-layer worst: {three_worst['case_id']}  MAE={three_worst['top_uz_mae_pct']}%")
    
    print("Loading PINN models...")
    one_pinn, one_device, ol_config = load_one_layer_pinn()
    three_pinn, three_device, tl_config = load_three_layer_pinn()
    
    ne_x, ne_y, ne_z = 16, 16, 8
    
    print("Evaluating one-layer best case...")
    E_best = float(one_best['E'])
    t_best = float(one_best['thickness'])
    xn, yn, zn, u_fem_best = run_one_layer_fem(E_best, t_best, ne_x, ne_y, ne_z)
    u_pinn_best = evaluate_one_layer(one_pinn, one_device, ol_config, E_best, t_best, xn, yn, zn)
    mi = len(yn) // 2
    
    print("Evaluating one-layer worst case...")
    E_worst = float(one_worst['E'])
    t_worst = float(one_worst['thickness'])
    xn2, yn2, zn2, u_fem_worst = run_one_layer_fem(E_worst, t_worst, ne_x, ne_y, ne_z)
    u_pinn_worst = evaluate_one_layer(one_pinn, one_device, ol_config, E_worst, t_worst, xn2, yn2, zn2)
    mi2 = len(yn2) // 2
    
    print("Evaluating three-layer best case...")
    e1_b, e2_b, e3_b = float(three_best['e1']), float(three_best['e2']), float(three_best['e3'])
    t1_b, t2_b, t3_b = float(three_best['t1']), float(three_best['t2']), float(three_best['t3'])
    xn3, yn3, zn3, u_fem_3b = run_three_layer_fem(e1_b, e2_b, e3_b, t1_b, t2_b, t3_b, ne_x, ne_y, ne_z)
    u_pinn_3b = evaluate_three_layer(three_pinn, three_device, tl_config, e1_b, e2_b, e3_b, t1_b, t2_b, t3_b, xn3, yn3, zn3)
    mi3 = len(yn3) // 2
    H3 = t1_b + t2_b + t3_b
    
    print("Evaluating three-layer worst case...")
    e1_w, e2_w, e3_w = float(three_worst['e1']), float(three_worst['e2']), float(three_worst['e3'])
    t1_w, t2_w, t3_w = float(three_worst['t1']), float(three_worst['t2']), float(three_worst['t3'])
    xn4, yn4, zn4, u_fem_3w = run_three_layer_fem(e1_w, e2_w, e3_w, t1_w, t2_w, t3_w, ne_x, ne_y, ne_z)
    u_pinn_3w = evaluate_three_layer(three_pinn, three_device, tl_config, e1_w, e2_w, e3_w, t1_w, t2_w, t3_w, xn4, yn4, zn4)
    mi4 = len(yn4) // 2
    H4 = t1_w + t2_w + t3_w
    
    # High-quality figure with larger size and DPI
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(
        f"PINN Generalization Cross-Section Absolute Error ($y=0.5$)\n"
        f"Random Interior Parameter Study — $N={N_CASES}$ samples per model",
        fontsize=16, fontweight="bold", y=0.98
    )
    
    zn_norm = zn / t_best
    zn2_norm = zn2 / t_worst
    
    case_info = f"Best: $E={E_best:.2f}$, $t={t_best:.3f}$"
    plot_cross_section_mae(
        axes[0, 0], xn, zn_norm, 
        u_fem_best[:, mi, :, 2], u_pinn_best[:, mi, :, 2],
        "(a) One-Layer", float(one_best['top_uz_mae_pct']),
        float(one_best['top_uz_mean_abs_error']), case_info
    )
    
    case_info = f"Worst: $E={E_worst:.2f}$, $t={t_worst:.3f}$"
    plot_cross_section_mae(
        axes[0, 1], xn2, zn2_norm,
        u_fem_worst[:, mi2, :, 2], u_pinn_worst[:, mi2, :, 2],
        "(b) One-Layer", float(one_worst['top_uz_mae_pct']),
        float(one_worst['top_uz_mean_abs_error']), case_info
    )
    
    case_info = f"Best: $E=[{e1_b:.1f},{e2_b:.1f},{e3_b:.1f}]$\n$t=[{t1_b:.3f},{t2_b:.3f},{t3_b:.3f}]$"
    plot_cross_section_mae(
        axes[1, 0], xn3, zn3 / H3,
        u_fem_3b[:, mi3, :, 2], u_pinn_3b[:, mi3, :, 2],
        "(c) Three-Layer", float(three_best['top_uz_mae_pct']),
        float(three_best['top_uz_mean_abs_error']), case_info,
        interfaces=[t1_b / H3, (t1_b + t2_b) / H3]
    )
    
    case_info = f"Worst: $E=[{e1_w:.1f},{e2_w:.1f},{e3_w:.1f}]$\n$t=[{t1_w:.3f},{t2_w:.3f},{t3_w:.3f}]$"
    plot_cross_section_mae(
        axes[1, 1], xn4, zn4 / H4,
        u_fem_3w[:, mi4, :, 2], u_pinn_3w[:, mi4, :, 2],
        "(d) Three-Layer", float(three_worst['top_uz_mae_pct']),
        float(three_worst['top_uz_mean_abs_error']), case_info,
        interfaces=[t1_w / H4, (t1_w + t2_w) / H4]
    )
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    out_pdf = OUT_DIR / "generalization_cross_section_mae.pdf"
    out_png = OUT_DIR / "generalization_cross_section_mae.png"
    fig.savefig(out_pdf, dpi=600, bbox_inches="tight")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\nSaved high-quality figure:")
    print(f"  {out_pdf}")
    print(f"  {out_png}")


if __name__ == "__main__":
    main()
