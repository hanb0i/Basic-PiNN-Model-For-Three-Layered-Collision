#!/usr/bin/env python3
"""
Generate FEM validation figures for IEEE single-column format.

Changes from original:
1. Two separate PDFs (one-layer and three-layer) instead of one 4-panel figure
2. Each PDF has two subplots stacked vertically (peak displacement on top, L2 convergence on bottom)
3. L2 is rendered with superscript 2 (not subscript)
4. Font sizes increased for IEEE paper readability
5. Figure size optimized for single IEEE column (~3.5 inches wide)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

REPO_ROOT = Path("/Users/hanbosong/GitHub/Basic-PiNN-Model-For-Three-Layered-Collision")
OUT_DIR = REPO_ROOT / "graphs" / "generalized_study" / "fem_convergence"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEA_DIR = REPO_ROOT / "fea-workflow" / "solver"
sys.path.insert(0, str(FEA_DIR))

import fem_solver

# Publication font setup — increased for IEEE readability
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


def run_one_layer_fem(E_val, thickness, ne_x, ne_y, ne_z):
    cfg = {
        "geometry": {"Lx": 1.0, "Ly": 1.0, "H": thickness, "ne_x": ne_x, "ne_y": ne_y, "ne_z": ne_z},
        "material": {"E": E_val, "nu": 0.3},
        "load_patch": {"pressure": 1.0, "x_start": 1/3, "x_end": 2/3, "y_start": 1/3, "y_end": 2/3},
    }
    xn, yn, zn, u_grid = fem_solver.solve_fem(cfg)
    xn = np.array(xn, dtype=float)
    yn = np.array(yn, dtype=float)
    zn = np.array(zn, dtype=float)
    u_grid = np.array(u_grid, dtype=float)
    return xn, yn, zn, u_grid


def run_three_layer_fem(e1, e2, e3, t1, t2, t3, ne_x, ne_y, ne_z):
    thickness = t1 + t2 + t3
    cfg = {
        "geometry": {"Lx": 1.0, "Ly": 1.0, "H": thickness, "ne_x": ne_x, "ne_y": ne_y, "ne_z": ne_z},
        "material": {"E_layers": [float(e1), float(e2), float(e3)], "t_layers": [float(t1), float(t2), float(t3)], "nu": 0.3},
        "load_patch": {"pressure": 1.0, "x_start": 1/3, "x_end": 2/3, "y_start": 1/3, "y_end": 2/3},
    }
    xn, yn, zn, u_grid = fem_solver.solve_three_layer_fem(cfg)
    xn = np.array(xn, dtype=float)
    yn = np.array(yn, dtype=float)
    zn = np.array(zn, dtype=float)
    u_grid = np.array(u_grid, dtype=float)
    return xn, yn, zn, u_grid


def compute_l2_error_vs_reference(u_test, x_test, y_test, z_test, u_ref, x_ref, y_ref, z_ref):
    """Relative L2 error of test solution vs reference (finest) solution."""
    interpolator = RegularGridInterpolator(
        (x_test, y_test, z_test), u_test,
        method='linear', bounds_error=False, fill_value=0
    )
    
    X_ref, Y_ref, Z_ref = np.meshgrid(x_ref, y_ref, z_ref, indexing='ij')
    pts = np.stack([X_ref.ravel(), Y_ref.ravel(), Z_ref.ravel()], axis=-1)
    u_test_on_ref = interpolator(pts).reshape(u_ref.shape)
    
    dx = x_ref[1] - x_ref[0] if len(x_ref) > 1 else 1.0
    dy = y_ref[1] - y_ref[0] if len(y_ref) > 1 else 1.0
    dz = z_ref[1] - z_ref[0] if len(z_ref) > 1 else 1.0
    dV = dx * dy * dz
    
    diff = u_test_on_ref - u_ref
    l2_error = np.sqrt(np.sum(diff**2) * dV)
    l2_ref = np.sqrt(np.sum(u_ref**2) * dV)
    
    rel_l2 = 100.0 * l2_error / l2_ref if l2_ref > 0 else 0.0
    return rel_l2


def make_single_model_figure(
    h_plot, h_benchmark, h_finest,
    peak_rel_err, l2_err,
    color, title_prefix, param_text,
    out_pdf, out_png
):
    """
    Create a 2x1 figure for a single model (one-layer or three-layer).
    Top: peak displacement convergence
    Bottom: L2 convergence rate
    """
    # IEEE single column width is ~3.5 inches; height scales to fit two subplots
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 6.5))
    # No suptitle for IEEE single-column figure; title goes in LaTeX caption
    
    # --- Top panel: Peak Displacement Convergence ---
    ax = axes[0]
    ax.loglog(h_plot, peak_rel_err, "o-", color=color, linewidth=2, markersize=8,
              label="Peak $|u_z|$ convergence", zorder=3)
    # Connecting line to reference star
    ax.plot([h_benchmark, h_finest], [peak_rel_err[-1], 0.1],
            color=color, linewidth=2, linestyle="-", zorder=3)
    ax.plot(h_finest, 0.1, "*", color="gold", markersize=10, markeredgecolor="black",
            markeredgewidth=1.5, label="Reference: 32$\\times$32$\\times$16", zorder=5)
    
    # O(h^2) reference line
    C_peak = np.max(np.array(peak_rel_err) / h_plot**2)
    h_ref = np.array([h_plot[0], h_finest])
    ax.loglog(h_ref, C_peak * h_ref**2, "k--", alpha=0.5, label="O($h^2$) reference", zorder=2)
    
    ax.axvline(x=h_benchmark, color="#d62728", linestyle="--", linewidth=1.5, zorder=2)
    ax.text(h_benchmark, ax.get_ylim()[1] * 0.5, "  Benchmark\n  16$\\times$16$\\times$8",
            color="#d62728", fontsize=8, ha="left", va="top", fontweight="bold")
    ax.set_xlabel("Mesh size $h = 1/n_{e,x}$", fontsize=10)
    ax.set_ylabel("Rel. error in peak $|u_z|$ (%)", fontsize=10)
    ax.set_title(f"(a) Peak Displacement Convergence\n{param_text}", fontsize=10)
    ax.invert_xaxis()
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3, which="both")
    
    # --- Bottom panel: L2 Convergence Rate ---
    ax = axes[1]
    ax.loglog(h_plot, l2_err, "o-", color=color, linewidth=2, markersize=8,
              label="FEM $L^2$ convergence", zorder=3)
    # Connecting line to reference star
    ax.plot([h_benchmark, h_finest], [l2_err[-1], 0.1],
            color=color, linewidth=2, linestyle="-", zorder=3)
    ax.plot(h_finest, 0.1, "*", color="gold", markersize=10, markeredgecolor="black",
            markeredgewidth=1.5, label="Reference: 32$\\times$32$\\times$16", zorder=5)
    
    # O(h^2) reference line
    C_l2 = np.max(np.array(l2_err) / h_plot**2)
    ax.loglog(h_ref, C_l2 * h_ref**2, "k--", alpha=0.5, label="O($h^2$) reference", zorder=2)
    
    ax.axvline(x=h_benchmark, color="#d62728", linestyle="--", linewidth=1.5, zorder=2)
    ax.text(h_benchmark, ax.get_ylim()[1] * 0.5, "  Benchmark\n  16$\\times$16$\\times$8",
            color="#d62728", fontsize=8, ha="left", va="top", fontweight="bold")
    ax.set_xlabel("Mesh size $h$", fontsize=10)
    ax.set_ylabel("Rel. $L^2$ error vs 32$\\times$32$\\times$16 (%)", fontsize=10)
    ax.set_title(f"(b) $L^2$ Convergence Rate\n{param_text}", fontsize=10)
    ax.invert_xaxis()
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3, which="both")
    
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved:\n  {out_pdf}\n  {out_png}")


def main():
    meshes = [(4, 4, 2), (8, 8, 4), (16, 16, 8), (32, 32, 16)]
    h_values = np.array([1.0 / nx for nx, _, _ in meshes])
    h_benchmark = 1.0 / 16  # 0.0625 for 16x16x8
    h_finest = 1.0 / 32     # 0.03125 for 32x32x16
    
    print("=== Computing FEM solutions ===")
    
    # One-layer
    one_fem_solutions = []
    for ne_x, ne_y, ne_z in meshes:
        print(f"  One-layer {ne_x}x{ne_y}x{ne_z} ...", end=" ", flush=True)
        xn, yn, zn, u = run_one_layer_fem(E_val=1.0, thickness=0.15, ne_x=ne_x, ne_y=ne_y, ne_z=ne_z)
        one_fem_solutions.append((xn, yn, zn, u[:, :, :, 2]))
        print(f"peak = {np.min(u[:,:,:,2]):.6f}")
    
    # Three-layer
    three_fem_solutions = []
    for ne_x, ne_y, ne_z in meshes:
        print(f"  Three-layer {ne_x}x{ne_y}x{ne_z} ...", end=" ", flush=True)
        xn, yn, zn, u = run_three_layer_fem(10.0, 10.0, 10.0, 0.02, 0.10, 0.02, ne_x, ne_y, ne_z)
        three_fem_solutions.append((xn, yn, zn, u[:, :, :, 2]))
        print(f"peak = {np.min(u[:,:,:,2]):.6f}")
    
    # Reference = finest mesh (32x32x16) = index 3
    x_ref_1l, y_ref_1l, z_ref_1l, u_ref_1l = one_fem_solutions[3]
    x_ref_3l, y_ref_3l, z_ref_3l, u_ref_3l = three_fem_solutions[3]
    
    print("\n=== FEM L2 Convergence vs 32x32x16 Reference ===")
    
    one_l2_fem = []
    for i, (x_c, y_c, z_c, u_c) in enumerate(one_fem_solutions[:3]):
        rel_l2 = compute_l2_error_vs_reference(u_c, x_c, y_c, z_c, u_ref_1l, x_ref_1l, y_ref_1l, z_ref_1l)
        one_l2_fem.append(rel_l2)
        print(f"  One-layer {meshes[i][0]}x{meshes[i][1]}x{meshes[i][2]} vs 32x32x16: L2 = {rel_l2:.4f}%")
    
    three_l2_fem = []
    for i, (x_c, y_c, z_c, u_c) in enumerate(three_fem_solutions[:3]):
        rel_l2 = compute_l2_error_vs_reference(u_c, x_c, y_c, z_c, u_ref_3l, x_ref_3l, y_ref_3l, z_ref_3l)
        three_l2_fem.append(rel_l2)
        print(f"  Three-layer {meshes[i][0]}x{meshes[i][1]}x{meshes[i][2]} vs 32x32x16: L2 = {rel_l2:.4f}%")
    
    print("\n=== Peak Displacement Convergence ===")
    
    one_peak = []
    for i, (x_c, y_c, z_c, u_c) in enumerate(one_fem_solutions):
        peak = np.min(u_c)
        one_peak.append(abs(peak))
        print(f"  One-layer {meshes[i][0]}x{meshes[i][1]}x{meshes[i][2]}: peak |u_z| = {abs(peak):.6f}")
    
    three_peak = []
    for i, (x_c, y_c, z_c, u_c) in enumerate(three_fem_solutions):
        peak = np.min(u_c)
        three_peak.append(abs(peak))
        print(f"  Three-layer {meshes[i][0]}x{meshes[i][1]}x{meshes[i][2]}: peak |u_z| = {abs(peak):.6f}")
    
    one_peak_rel_err = []
    for i in range(3):
        err = 100.0 * abs(one_peak[i] - one_peak[3]) / one_peak[3]
        one_peak_rel_err.append(err)
        print(f"  One-layer {meshes[i][0]}x{meshes[i][1]}x{meshes[i][2]} peak error vs 32x32x16: {err:.4f}%")
    
    three_peak_rel_err = []
    for i in range(3):
        err = 100.0 * abs(three_peak[i] - three_peak[3]) / three_peak[3]
        three_peak_rel_err.append(err)
        print(f"  Three-layer {meshes[i][0]}x{meshes[i][1]}x{meshes[i][2]} peak error vs 32x32x16: {err:.4f}%")
    
    h_plot = h_values[:3]
    
    # ============================================================
    # Generate One-Layer Figure (2x1 vertical layout)
    # ============================================================
    print("\n=== Generating One-Layer Figure ===")
    make_single_model_figure(
        h_plot, h_benchmark, h_finest,
        one_peak_rel_err, one_l2_fem,
        color="#1f77b4",
        title_prefix="One-Layer",
        param_text="$E=1$, $t=0.15$",
        out_pdf=OUT_DIR / "fem_validation_one_layer.pdf",
        out_png=OUT_DIR / "fem_validation_one_layer.png",
    )
    
    # ============================================================
    # Generate Three-Layer Figure (2x1 vertical layout)
    # ============================================================
    print("\n=== Generating Three-Layer Figure ===")
    make_single_model_figure(
        h_plot, h_benchmark, h_finest,
        three_peak_rel_err, three_l2_fem,
        color="#2ca02c",
        title_prefix="Three-Layer",
        param_text="$E=[10,10,10]$, $t=[0.02,0.10,0.02]$",
        out_pdf=OUT_DIR / "fem_validation_three_layer.pdf",
        out_png=OUT_DIR / "fem_validation_three_layer.png",
    )
    
    # Save data
    with open(OUT_DIR / "l2_error_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "mesh", "h", "peak_rel_err", "l2_error_vs_32"])
        for i, (nx, ny, nz) in enumerate(meshes[:3]):
            h = 1.0 / nx
            writer.writerow(["one_layer", f"{nx}x{ny}x{nz}", h, one_peak_rel_err[i], one_l2_fem[i]])
            writer.writerow(["three_layer", f"{nx}x{ny}x{nz}", h, three_peak_rel_err[i], three_l2_fem[i]])
    print(f"\nSaved data: {OUT_DIR / 'l2_error_data.csv'}")


if __name__ == "__main__":
    main()
