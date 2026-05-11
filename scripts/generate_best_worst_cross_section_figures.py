#!/usr/bin/env python3
"""
Generate cross-section error heatmaps for best and worst random interior cases.

For both one-layer and three-layer models, this script:
1. Evaluates N random interior cases on 16×16×8 mesh
2. Identifies the best (lowest MAE%) and worst (highest MAE%) cases
3. Generates side-by-side cross-section error heatmaps

Usage:
    cd /path/to/repo
    python scripts/generate_best_worst_cross_section_figures.py

Environment variables:
    PINN_N_CASES       – number of random cases to evaluate (default: 16)
    PINN_SEED          – random seed (default: 42)
    PINN_DEVICE        – torch device (default: auto-detect)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPHS_DIR = REPO_ROOT / "graphs" / "figures"
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

# ── Shared plotting style ───────────────────────────────────────────────────
CMAP_ERR = "magma"
IF_LW, IF_COL = 2.5, "white"
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "dejavuserif",
})


def draw_interfaces(ax, interfaces):
    if not interfaces:
        return
    for z in interfaces:
        ax.axhline(z, color=IF_COL, linestyle="--", linewidth=IF_LW, alpha=0.9)


def mae_pct(pred, ref):
    d = float(np.max(np.abs(ref)))
    return 100 * float(np.mean(np.abs(pred - ref))) / d if d > 0 else 0


# ═════════════════════════════════════════════════════════════════════════════
# ONE-LAYER PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

ONE_LAYER_DIR = REPO_ROOT / "one-layer-workflow"
FEA_DIR = REPO_ROOT / "fea-workflow" / "solver"


def _ol_load_pinn(device):
    sys.path.insert(0, str(ONE_LAYER_DIR))
    sys.path.insert(0, str(FEA_DIR))
    import model as ol_model
    import pinn_config as ol_config

    model_path = Path(os.getenv("PINN_ONE_LAYER_MODEL") or ONE_LAYER_DIR / "pinn_model.pth")
    pinn = ol_model.MultiLayerPINN().to(device)
    sd = torch.load(str(model_path), map_location=device, weights_only=True)
    # Adapt state dict if needed
    w_key = "layer.net.0.weight"
    if w_key in sd:
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
    return pinn, ol_config


def _ol_predict(pinn, cfg, device, xf, yf, zf, E, t):
    r = float(getattr(cfg, "RESTITUTION_REF", 0.5))
    mu = float(getattr(cfg, "FRICTION_REF", 0.3))
    v0 = float(getattr(cfg, "IMPACT_VELOCITY_REF", 1.0))
    pts = np.stack([xf, yf, zf,
                    np.full_like(xf, E),
                    np.full_like(xf, t),
                    np.full_like(xf, r),
                    np.full_like(xf, mu),
                    np.full_like(xf, v0)], axis=1)
    with torch.no_grad():
        v = pinn(torch.tensor(pts, dtype=torch.float32, device=device)).cpu().numpy()
    ep = float(getattr(cfg, "E_COMPLIANCE_POWER", 1.0))
    al = float(getattr(cfg, "THICKNESS_COMPLIANCE_ALPHA", 0.0))
    sc = float(getattr(cfg, "DISPLACEMENT_COMPLIANCE_SCALE", 1.0))
    hr = float(getattr(cfg, "H", 0.1))
    return sc * v / (pts[:, 3:4]**ep) * (hr / np.clip(pts[:, 4:5], 1e-8, None))**al


def _ol_run_fem(E, t):
    import fem_solver
    cfg = {
        "geometry": {"Lx": 1.0, "Ly": 1.0, "H": t, "ne_x": 16, "ne_y": 16, "ne_z": 8},
        "material": {"E": E, "nu": 0.3},
        "load_patch": {"pressure": 1.0, "x_start": 1/3, "x_end": 2/3, "y_start": 1/3, "y_end": 2/3},
    }
    return fem_solver.solve_fem(cfg)


def _ol_evaluate_case(pinn, cfg, device, E, t):
    xn, yn, zn, u_fea = _ol_run_fem(E, t)
    xn, yn, zn = [np.array(a, dtype=float) for a in [xn, yn, zn]]
    u_fea = np.array(u_fea, dtype=float)

    # Top surface
    xg, yg = np.meshgrid(xn, yn, indexing="ij")
    u_pinn_top = _ol_predict(pinn, cfg, device, xg.ravel(), yg.ravel(),
                             np.full(xg.size, t), E, t).reshape(len(xn), len(yn), 3)
    uz_fea_top = u_fea[:, :, -1, 2]
    uz_pinn_top = u_pinn_top[:, :, 2]
    top_mae = mae_pct(uz_pinn_top, uz_fea_top)

    # Cross-section at y=0.5
    mi = len(yn) // 2
    xc, zc = np.meshgrid(xn, zn, indexing="ij")
    u_pinn_cs = _ol_predict(pinn, cfg, device, xc.ravel(), np.full(xc.size, yn[mi]),
                            zc.ravel(), E, t).reshape(len(xn), len(zn), 3)
    uz_fea_cs = u_fea[:, mi, :, 2]
    uz_pinn_cs = u_pinn_cs[:, :, 2]

    return {
        "E": float(E), "thickness": float(t),
        "top_uz_mae_pct": float(top_mae),
        "x_nodes": xn, "y_nodes": yn, "z_nodes": zn,
        "uz_fea_top": uz_fea_top, "uz_pinn_top": uz_pinn_top,
        "uz_fea_cs": uz_fea_cs, "uz_pinn_cs": uz_pinn_cs,
    }


# ═════════════════════════════════════════════════════════════════════════════
# THREE-LAYER PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

THREE_LAYER_DIR = REPO_ROOT / "three-layer-workflow"
PINN_WORKFLOW_DIR = REPO_ROOT / "pinn-workflow"


def _tl_load_pinn(device):
    code_dir = THREE_LAYER_DIR
    # Remove previously loaded model modules to avoid conflicts
    for mod_name in list(sys.modules.keys()):
        if mod_name in ("model", "pinn_config"):
            del sys.modules[mod_name]
    sys.path.insert(0, str(code_dir))
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
        "geometry": {"Lx": 1.0, "Ly": 1.0, "H": t1+t2+t3, "ne_x": 16, "ne_y": 16, "ne_z": 8},
        "material": {"E_layers": [e1, e2, e3], "t_layers": [t1, t2, t3], "nu": 0.3},
        "load_patch": {"pressure": 1.0, "x_start": 1/3, "x_end": 2/3, "y_start": 1/3, "y_end": 2/3},
    }
    return fem_solver.solve_three_layer_fem(cfg)


def _tl_evaluate_case(pinn, cfg, device, e1, e2, e3, t1, t2, t3):
    xn, yn, zn, u_fea = _tl_run_fem(e1, e2, e3, t1, t2, t3)
    xn, yn, zn = [np.array(a, dtype=float) for a in [xn, yn, zn]]
    u_fea = np.array(u_fea, dtype=float)
    H = t1 + t2 + t3

    # Top surface
    xg, yg = np.meshgrid(xn, yn, indexing="ij")
    u_pinn_top = _tl_predict(pinn, cfg, device, xg.ravel(), yg.ravel(),
                             np.full(xg.size, H), e1, e2, e3, t1, t2, t3).reshape(len(xn), len(yn), 3)
    uz_fea_top = u_fea[:, :, -1, 2]
    uz_pinn_top = u_pinn_top[:, :, 2]
    top_mae = mae_pct(uz_pinn_top, uz_fea_top)

    # Cross-section at y=0.5
    mi = len(yn) // 2
    xc, zc = np.meshgrid(xn, zn, indexing="ij")
    u_pinn_cs = _tl_predict(pinn, cfg, device, xc.ravel(), np.full(xc.size, yn[mi]),
                            zc.ravel(), e1, e2, e3, t1, t2, t3).reshape(len(xn), len(zn), 3)
    uz_fea_cs = u_fea[:, mi, :, 2]
    uz_pinn_cs = u_pinn_cs[:, :, 2]

    return {
        "e1": float(e1), "e2": float(e2), "e3": float(e3),
        "t1": float(t1), "t2": float(t2), "t3": float(t3),
        "top_uz_mae_pct": float(top_mae),
        "x_nodes": xn, "y_nodes": yn, "z_nodes": zn,
        "uz_fea_top": uz_fea_top, "uz_pinn_top": uz_pinn_top,
        "uz_fea_cs": uz_fea_cs, "uz_pinn_cs": uz_pinn_cs,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Plotting
# ═════════════════════════════════════════════════════════════════════════════

def plot_case_comparison(ax_fea, ax_pinn, ax_err, x, z, uz_fea, uz_pinn, title, interfaces=None):
    """Plot FEA, PINN, and error on three axes."""
    vmin = min(float(np.min(uz_fea)), float(np.min(uz_pinn)))
    vmax = max(float(np.max(uz_fea)), float(np.max(uz_pinn)))

    c0 = ax_fea.contourf(x, z, uz_fea, levels=50, cmap="jet", vmin=vmin, vmax=vmax)
    plt.colorbar(c0, ax=ax_fea, fraction=0.046, pad=0.04)
    ax_fea.set_title(f"{title}\nFEA")
    ax_fea.set_xlabel("x")
    ax_fea.set_ylabel("z")
    draw_interfaces(ax_fea, interfaces)

    c1 = ax_pinn.contourf(x, z, uz_pinn, levels=50, cmap="jet", vmin=vmin, vmax=vmax)
    plt.colorbar(c1, ax=ax_pinn, fraction=0.046, pad=0.04)
    ax_pinn.set_title(f"{title}\nPINN")
    ax_pinn.set_xlabel("x")
    ax_pinn.set_ylabel("z")
    draw_interfaces(ax_pinn, interfaces)

    err = np.abs(uz_pinn - uz_fea)
    c2 = ax_err.contourf(x, z, err, levels=50, cmap=CMAP_ERR)
    plt.colorbar(c2, ax=ax_err, fraction=0.046, pad=0.04)
    mae = float(np.mean(err))
    ax_err.set_title(f"{title}\nAbs Error | MAE={mae:.5f}")
    ax_err.set_xlabel("x")
    ax_err.set_ylabel("z")
    draw_interfaces(ax_err, interfaces)


def plot_best_worst_panel(fig, cases, title_prefix, interfaces_fn, param_str_fn):
    """Create a 2×3 panel: best case (FEA/PINN/Error) and worst case (FEA/PINN/Error)."""
    # Sort by MAE
    sorted_cases = sorted(cases, key=lambda c: c["top_uz_mae_pct"])
    best = sorted_cases[0]
    worst = sorted_cases[-1]

    # Create 2×3 grid: row 0 = best, row 1 = worst
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    for row, case, label in [(0, best, "Best"), (1, worst, "Worst")]:
        xn, zn = case["x_nodes"], case["z_nodes"]
        xc, zc = np.meshgrid(xn, zn, indexing="ij")
        interfaces = interfaces_fn(case) if interfaces_fn else None
        param_str = param_str_fn(case)

        ax_fea = fig.add_subplot(gs[row, 0])
        ax_pinn = fig.add_subplot(gs[row, 1])
        ax_err = fig.add_subplot(gs[row, 2])

        plot_case_comparison(
            ax_fea, ax_pinn, ax_err,
            xc, zc, case["uz_fea_cs"], case["uz_pinn_cs"],
            f"{label}: {param_str}",
            interfaces
        )

    return best, worst


# ═════════════════════════════════════════════════════════════════════════════
# Random case generation
# ═════════════════════════════════════════════════════════════════════════════

def random_one_layer_cases(n, seed):
    rng = np.random.default_rng(seed)
    e_min, e_max = 1.0, 10.0
    t_min, t_max = 0.05, 0.15
    cases = []
    for i in range(n):
        E = float(rng.uniform(e_min, e_max))
        t = float(rng.uniform(t_min, t_max))
        cases.append((E, t))
    return cases


def random_three_layer_cases(n, seed):
    rng = np.random.default_rng(seed)
    e_min, e_max = 1.0, 10.0
    t_min, t_max = 0.02, 0.10
    cases = []
    for i in range(n):
        e1 = float(rng.uniform(e_min, e_max))
        e2 = float(rng.uniform(e_min, e_max))
        e3 = float(rng.uniform(e_min, e_max))
        t1 = float(rng.uniform(t_min, t_max))
        t2 = float(rng.uniform(t_min, t_max))
        t3 = float(rng.uniform(t_min, t_max))
        cases.append((e1, e2, e3, t1, t2, t3))
    return cases


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    n_cases = int(os.getenv("PINN_N_CASES", "16"))
    seed = int(os.getenv("PINN_SEED", "42"))

    # Device
    if os.getenv("PINN_FORCE_CPU", "0") == "1":
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    print(f"Evaluating {n_cases} random cases with seed={seed}")

    # ── One-layer ──────────────────────────────────────────────────────────
    print("\n=== One-Layer ===")
    ol_pinn, ol_cfg = _ol_load_pinn(device)
    ol_cases = []
    for E, t in random_one_layer_cases(n_cases, seed):
        result = _ol_evaluate_case(ol_pinn, ol_cfg, device, E, t)
        ol_cases.append(result)
        print(f"  E={E:.2f} t={t:.3f}: top MAE={result['top_uz_mae_pct']:.2f}%")

    ol_sorted = sorted(ol_cases, key=lambda c: c["top_uz_mae_pct"])
    ol_best, ol_worst = ol_sorted[0], ol_sorted[-1]
    print(f"  Best:  E={ol_best['E']:.2f} t={ol_best['thickness']:.3f} -> {ol_best['top_uz_mae_pct']:.2f}%")
    print(f"  Worst: E={ol_worst['E']:.2f} t={ol_worst['thickness']:.3f} -> {ol_worst['top_uz_mae_pct']:.2f}%")

    # ── Three-layer ────────────────────────────────────────────────────────
    print("\n=== Three-Layer ===")
    tl_pinn, tl_cfg = _tl_load_pinn(device)
    tl_cases = []
    for e1, e2, e3, t1, t2, t3 in random_three_layer_cases(n_cases, seed):
        result = _tl_evaluate_case(tl_pinn, tl_cfg, device, e1, e2, e3, t1, t2, t3)
        tl_cases.append(result)
        print(f"  E=[{e1:.1f},{e2:.1f},{e3:.1f}] t=[{t1:.2f},{t2:.2f},{t3:.2f}]: top MAE={result['top_uz_mae_pct']:.2f}%")

    tl_sorted = sorted(tl_cases, key=lambda c: c["top_uz_mae_pct"])
    tl_best, tl_worst = tl_sorted[0], tl_sorted[-1]
    print(f"  Best:  E=[{tl_best['e1']:.1f},{tl_best['e2']:.1f},{tl_best['e3']:.1f}] -> {tl_best['top_uz_mae_pct']:.2f}%")
    print(f"  Worst: E=[{tl_worst['e1']:.1f},{tl_worst['e2']:.1f},{tl_worst['e3']:.1f}] -> {tl_worst['top_uz_mae_pct']:.2f}%")

    # ── Generate figures ───────────────────────────────────────────────────
    print("\nGenerating figures...")

    # One-layer best/worst cross-section
    fig1 = plt.figure(figsize=(18, 10))
    fig1.suptitle("One-Layer: Best vs Worst Cross-Section Error", fontsize=14, fontweight="bold")
    plot_best_worst_panel(
        fig1, ol_cases,
        "One-Layer",
        interfaces_fn=lambda c: None,
        param_str_fn=lambda c: f"E={c['E']:.2f}, t={c['thickness']:.3f}, MAE={c['top_uz_mae_pct']:.2f}%"
    )
    fig1.savefig(GRAPHS_DIR / "one_layer_best_worst_cross_section.png", dpi=200, bbox_inches="tight")
    fig1.savefig(GRAPHS_DIR / "one_layer_best_worst_cross_section.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"  Saved: one_layer_best_worst_cross_section.png")

    # Three-layer best/worst cross-section
    fig2 = plt.figure(figsize=(18, 10))
    fig2.suptitle("Three-Layer: Best vs Worst Cross-Section Error", fontsize=14, fontweight="bold")
    plot_best_worst_panel(
        fig2, tl_cases,
        "Three-Layer",
        interfaces_fn=lambda c: [c["t1"], c["t1"] + c["t2"]],
        param_str_fn=lambda c: f"E=[{c['e1']:.1f},{c['e2']:.1f},{c['e3']:.1f}], MAE={c['top_uz_mae_pct']:.2f}%"
    )
    fig2.savefig(GRAPHS_DIR / "three_layer_best_worst_cross_section.png", dpi=200, bbox_inches="tight")
    fig2.savefig(GRAPHS_DIR / "three_layer_best_worst_cross_section.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: three_layer_best_worst_cross_section.png")

    # Combined figure: one-layer best + three-layer best (side by side)
    fig3 = plt.figure(figsize=(18, 10))
    fig3.suptitle("Best Cases: One-Layer vs Three-Layer Cross-Section Error", fontsize=14, fontweight="bold")
    gs = fig3.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # Row 0: One-layer best
    xn, zn = ol_best["x_nodes"], ol_best["z_nodes"]
    xc, zc = np.meshgrid(xn, zn, indexing="ij")
    plot_case_comparison(
        fig3.add_subplot(gs[0, 0]), fig3.add_subplot(gs[0, 1]), fig3.add_subplot(gs[0, 2]),
        xc, zc, ol_best["uz_fea_cs"], ol_best["uz_pinn_cs"],
        f"1L Best: E={ol_best['E']:.2f}, t={ol_best['thickness']:.3f}"
    )

    # Row 1: Three-layer best
    xn, zn = tl_best["x_nodes"], tl_best["z_nodes"]
    xc, zc = np.meshgrid(xn, zn, indexing="ij")
    plot_case_comparison(
        fig3.add_subplot(gs[1, 0]), fig3.add_subplot(gs[1, 1]), fig3.add_subplot(gs[1, 2]),
        xc, zc, tl_best["uz_fea_cs"], tl_best["uz_pinn_cs"],
        f"3L Best: E=[{tl_best['e1']:.1f},{tl_best['e2']:.1f},{tl_best['e3']:.1f}]",
        interfaces=[tl_best["t1"], tl_best["t1"] + tl_best["t2"]]
    )

    fig3.savefig(GRAPHS_DIR / "best_cases_comparison.png", dpi=200, bbox_inches="tight")
    fig3.savefig(GRAPHS_DIR / "best_cases_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Saved: best_cases_comparison.png")

    # Combined figure: one-layer worst + three-layer worst (side by side)
    fig4 = plt.figure(figsize=(18, 10))
    fig4.suptitle("Worst Cases: One-Layer vs Three-Layer Cross-Section Error", fontsize=14, fontweight="bold")
    gs = fig4.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # Row 0: One-layer worst
    xn, zn = ol_worst["x_nodes"], ol_worst["z_nodes"]
    xc, zc = np.meshgrid(xn, zn, indexing="ij")
    plot_case_comparison(
        fig4.add_subplot(gs[0, 0]), fig4.add_subplot(gs[0, 1]), fig4.add_subplot(gs[0, 2]),
        xc, zc, ol_worst["uz_fea_cs"], ol_worst["uz_pinn_cs"],
        f"1L Worst: E={ol_worst['E']:.2f}, t={ol_worst['thickness']:.3f}"
    )

    # Row 1: Three-layer worst
    xn, zn = tl_worst["x_nodes"], tl_worst["z_nodes"]
    xc, zc = np.meshgrid(xn, zn, indexing="ij")
    plot_case_comparison(
        fig4.add_subplot(gs[1, 0]), fig4.add_subplot(gs[1, 1]), fig4.add_subplot(gs[1, 2]),
        xc, zc, tl_worst["uz_fea_cs"], tl_worst["uz_pinn_cs"],
        f"3L Worst: E=[{tl_worst['e1']:.1f},{tl_worst['e2']:.1f},{tl_worst['e3']:.1f}]",
        interfaces=[tl_worst["t1"], tl_worst["t1"] + tl_worst["t2"]]
    )

    fig4.savefig(GRAPHS_DIR / "worst_cases_comparison.png", dpi=200, bbox_inches="tight")
    fig4.savefig(GRAPHS_DIR / "worst_cases_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig4)
    print(f"  Saved: worst_cases_comparison.png")

    # Save summary JSON
    summary = {
        "seed": seed,
        "n_cases": n_cases,
        "mesh": {"ne_x": 16, "ne_y": 16, "ne_z": 8},
        "one_layer": {
            "best": {"E": ol_best["E"], "thickness": ol_best["thickness"], "top_uz_mae_pct": ol_best["top_uz_mae_pct"]},
            "worst": {"E": ol_worst["E"], "thickness": ol_worst["thickness"], "top_uz_mae_pct": ol_worst["top_uz_mae_pct"]},
            "mean_mae": float(np.mean([c["top_uz_mae_pct"] for c in ol_cases])),
        },
        "three_layer": {
            "best": {"e1": tl_best["e1"], "e2": tl_best["e2"], "e3": tl_best["e3"],
                     "t1": tl_best["t1"], "t2": tl_best["t2"], "t3": tl_best["t3"],
                     "top_uz_mae_pct": tl_best["top_uz_mae_pct"]},
            "worst": {"e1": tl_worst["e1"], "e2": tl_worst["e2"], "e3": tl_worst["e3"],
                      "t1": tl_worst["t1"], "t2": tl_worst["t2"], "t3": tl_worst["t3"],
                      "top_uz_mae_pct": tl_worst["top_uz_mae_pct"]},
            "mean_mae": float(np.mean([c["top_uz_mae_pct"] for c in tl_cases])),
        },
    }
    summary_path = REPO_ROOT / "graphs" / "data" / "best_worst_cross_section_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved to: {summary_path}")
    print("Done!")


if __name__ == "__main__":
    main()
