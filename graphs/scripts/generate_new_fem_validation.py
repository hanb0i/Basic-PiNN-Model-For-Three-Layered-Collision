#!/usr/bin/env python3
"""
Generate new FEM validation figure with:
1. Left panels: Relative L2 error between mesh sizes for max displacement
2. Right panels: L2 error convergence (FEM vs finest 32x32x16)

ALL comparisons use 32x32x16 as the reference/baseline.
Finest mesh (32x32x16) is marked with a gold star.
Connecting line from 16x16x8 data point to reference star for continuity.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "graphs" / "generalized_study" / "fem_convergence"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEA_DIR = REPO_ROOT / "fea-workflow" / "solver"
sys.path.insert(0, str(FEA_DIR))

import fem_solver

# Publication font setup
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
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
    """
    Relative L2 error of test solution vs reference (finest) solution.
    """
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


def main():
    meshes = [(4, 4, 2), (8, 8, 4), (16, 16, 8), (32, 32, 16)]
    h_values = np.array([1.0 / nx for nx, _, _ in meshes])
    h_benchmark = 1.0 / 16  # 0.0625 for 16x16x8
    h_finest = 1.0 / 32     # 0.03125 for 32x32x16
    
    # ============================================================
    # PART 1: Compute FEM solutions for all meshes
    # ============================================================
    
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
    
    # ============================================================
    # PART 2: FEM L2 convergence (coarse vs 32x32x16 reference)
    # ============================================================
    
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
    
    # ============================================================
    # PART 3: Compute peak displacement convergence
    # ============================================================
    
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
    
    # Compute relative error in peak displacement vs finest mesh
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
    
    # ============================================================
    # PART 4: Generate Figure
    # ============================================================
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "FEM Solver Validation: Mesh Convergence Study \nReference: 32 x 32 x 16 mesh",
        fontsize=13, fontweight="bold", y=0.98
    )
    
    h_plot = h_values[:3]  # Exclude finest mesh
    
    # --- Panel (a): One-Layer Peak Displacement Convergence ---
    ax = axes[0, 0]
    ax.loglog(h_plot, one_peak_rel_err, "o-", color="#1f77b4", linewidth=2, markersize=10, 
              label="Peak $|u_z|$ convergence", zorder=3)
    # Add connecting line from 16x16x8 to reference star
    ax.plot([h_benchmark, h_finest], [one_peak_rel_err[-1], 0.1], 
            color="#1f77b4", linewidth=2, linestyle="-", zorder=3)
    # Add finest mesh point with a prominent star at y=0.1 (bottom of plot)
    ax.plot(h_finest, 0.1, "*", color="gold", markersize=10, markeredgecolor="black", 
            markeredgewidth=1.5, label="Reference: 32$\\times$32$\\times$16", zorder=5)
    
    # O(h^2) reference line for peak displacement
    C_1l_peak = np.max(np.array(one_peak_rel_err) / h_plot**2)
    h_ref = np.array([h_plot[0], h_finest])
    ax.loglog(h_ref, C_1l_peak * h_ref**2, "k--", alpha=0.5, label="O($h^2$) reference", zorder=2)
    
    ax.axvline(x=h_benchmark, color="#d62728", linestyle="--", linewidth=1.5, zorder=2)
    ax.text(h_benchmark, ax.get_ylim()[1] * 0.5, "  Benchmark\n  16$\\times$16$\\times$8", 
            color="#d62728", fontsize=9, ha="left", va="top", fontweight="bold")
    ax.set_xlabel("Mesh size $h = 1/n_{e,x}$", fontsize=11)
    ax.set_ylabel("Relative error in peak $|u_z|$ vs 32$\\times$32$\\times$16 (%)", fontsize=11)
    ax.set_title("(a) One-Layer: Peak Displacement Convergence\n$E=1$, $t=0.15$", fontsize=11)
    ax.invert_xaxis()
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, which="both")
    
    # --- Panel (b): One-Layer L2 Convergence Rate ---
    ax = axes[0, 1]
    ax.loglog(h_plot, one_l2_fem, "o-", color="#1f77b4", linewidth=2, markersize=10, 
              label="FEM $L_2$ convergence", zorder=3)
    # Add connecting line from 16x16x8 to reference star
    ax.plot([h_benchmark, h_finest], [one_l2_fem[-1], 0.1], 
            color="#1f77b4", linewidth=2, linestyle="-", zorder=3)
    # Add finest mesh point with a prominent star at y=0.1 (bottom of plot)
    ax.plot(h_finest, 0.1, "*", color="gold", markersize=10, markeredgecolor="black", 
            markeredgewidth=1.5, label="Reference: 32$\\times$32$\\times$16", zorder=5)
    
    # O(h^2) reference line
    C_1l = np.max(np.array(one_l2_fem) / h_plot**2)
    ax.loglog(h_ref, C_1l * h_ref**2, "k--", alpha=0.5, label="O($h^2$) reference", zorder=2)
    
    ax.axvline(x=h_benchmark, color="#d62728", linestyle="--", linewidth=1.5, zorder=2)
    ax.text(h_benchmark, ax.get_ylim()[1] * 0.5, "  Benchmark\n  16$\\times$16$\\times$8", 
            color="#d62728", fontsize=9, ha="left", va="top", fontweight="bold")
    ax.set_xlabel("Mesh size $h$", fontsize=11)
    ax.set_ylabel("Relative $L_2$ error vs 32$\\times$32$\\times$16 (%)", fontsize=11)
    ax.set_title("(b) One-Layer: $L_2$ Convergence Rate\n$E=1$, $t=0.15$", fontsize=11)
    ax.invert_xaxis()
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, which="both")
    
    # --- Panel (c): Three-Layer Peak Displacement Convergence ---
    ax = axes[1, 0]
    ax.loglog(h_plot, three_peak_rel_err, "o-", color="#2ca02c", linewidth=2, markersize=10,
            label="Peak $|u_z|$ convergence", zorder=3)
    # Add connecting line from 16x16x8 to reference star
    ax.plot([h_benchmark, h_finest], [three_peak_rel_err[-1], 0.1], 
            color="#2ca02c", linewidth=2, linestyle="-", zorder=3)
    # Add finest mesh point with a prominent star at y=0.1 (bottom of plot)
    ax.plot(h_finest, 0.1, "*", color="gold", markersize=10, markeredgecolor="black", 
            markeredgewidth=1.5, label="Reference: 32$\\times$32$\\times$16", zorder=5)
    
    # O(h^2) reference line for peak displacement
    C_3l_peak = np.max(np.array(three_peak_rel_err) / h_plot**2)
    ax.loglog(h_ref, C_3l_peak * h_ref**2, "k--", alpha=0.5, label="O($h^2$) reference", zorder=2)
    
    ax.axvline(x=h_benchmark, color="#d62728", linestyle="--", linewidth=1.5, zorder=2)
    ax.text(h_benchmark, ax.get_ylim()[1] * 0.5, "  Benchmark\n  16$\\times$16$\\times$8", 
            color="#d62728", fontsize=9, ha="left", va="top", fontweight="bold")
    ax.set_xlabel("Mesh size $h = 1/n_{e,x}$", fontsize=11)
    ax.set_ylabel("Relative error in peak $|u_z|$ vs 32$\\times$32$\\times$16 (%)", fontsize=11)
    ax.set_title("(c) Three-Layer: Peak Displacement Convergence\n$E=[10,10,10]$, $t=[0.02,0.10,0.02]$", fontsize=11)
    ax.invert_xaxis()
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, which="both")
    
    # --- Panel (d): Three-Layer L2 Convergence Rate ---
    ax = axes[1, 1]
    ax.loglog(h_plot, three_l2_fem, "o-", color="#2ca02c", linewidth=2, markersize=10,
              label="FEM $L_2$ convergence", zorder=3)
    # Add connecting line from 16x16x8 to reference star
    ax.plot([h_benchmark, h_finest], [three_l2_fem[-1], 0.1], 
            color="#2ca02c", linewidth=2, linestyle="-", zorder=3)
    # Add finest mesh point with a prominent star at y=0.1 (bottom of plot)
    ax.plot(h_finest, 0.1, "*", color="gold", markersize=10, markeredgecolor="black", 
            markeredgewidth=1.5, label="Reference: 32$\\times$32$\\times$16", zorder=5)
    
    # O(h^2) reference line
    C_3l = np.max(np.array(three_l2_fem) / h_plot**2)
    ax.loglog(h_ref, C_3l * h_ref**2, "k--", alpha=0.5, label="O($h^2$) reference", zorder=2)
    
    ax.axvline(x=h_benchmark, color="#d62728", linestyle="--", linewidth=1.5, zorder=2)
    ax.text(h_benchmark, ax.get_ylim()[1] * 0.5, "  Benchmark\n  16$\\times$16$\\times$8", 
            color="#d62728", fontsize=9, ha="left", va="top", fontweight="bold")
    ax.set_xlabel("Mesh size $h$", fontsize=11)
    ax.set_ylabel("Relative $L_2$ error vs 32$\\times$32$\\times$16 (%)", fontsize=11)
    ax.set_title("(d) Three-Layer: $L_2$ Convergence Rate\n$E=[10,10,10]$, $t=[0.02,0.10,0.02]$", fontsize=11)
    ax.invert_xaxis()
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, which="both")
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    out_pdf = OUT_DIR / "fem_validation_l2_combined.pdf"
    out_png = OUT_DIR / "fem_validation_l2_combined.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\nSaved combined figure:")
    print(f"  {out_pdf}")
    print(f"  {out_png}")
    
    # Save data
    with open(OUT_DIR / "l2_error_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "mesh", "h", "peak_rel_err", "l2_error_vs_32"])
        for i, (nx, ny, nz) in enumerate(meshes[:3]):
            h = 1.0 / nx
            writer.writerow(["one_layer", f"{nx}x{ny}x{nz}", h, one_peak_rel_err[i], one_l2_fem[i]])
            writer.writerow(["three_layer", f"{nx}x{ny}x{nz}", h, three_peak_rel_err[i], three_l2_fem[i]])
    print(f"  {OUT_DIR / 'l2_error_data.csv'}")


if __name__ == "__main__":
    main()
