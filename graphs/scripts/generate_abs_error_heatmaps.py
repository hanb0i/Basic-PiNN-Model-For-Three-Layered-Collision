#!/usr/bin/env python3
"""
Generate 4 absolute-error cross-section heatmaps (y=0.5 slice) for best/worst
one-layer and three-layer PINN cases. Shared colorbar scale 0–0.3 m.
Saves each case as both PNG (600 DPI) and PDF.

Output dir: graphs/generalized_study/
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "graphs" / "generalized_study"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT / "fea-workflow" / "solver"))
import fem_solver  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 12,
    "axes.titlesize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

NE_X, NE_Y, NE_Z = 16, 16, 8
CMAP = "magma"
DPI = 600
FIGSIZE = (10, 6)
FIXED_VMAX = 0.3  # shared colorbar max across all 4 figures (m)


# ── CSV helpers ───────────────────────────────────────────────────────────────

def find_best_worst(csv_path: Path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: float(r["top_uz_mae_pct"]))
    return rows[0], rows[-1]


# ── Model loaders ─────────────────────────────────────────────────────────────

def load_one_layer_pinn():
    ol_dir = REPO_ROOT / "one-layer-workflow"
    sys.path.insert(0, str(ol_dir))
    import model as m
    import pinn_config as cfg
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pinn = m.MultiLayerPINN().to(device)
    sd = torch.load(ol_dir / "pinn_model.pth", map_location=device, weights_only=True)
    # Adapt legacy 8- or 10-feature checkpoints → 11-feature model
    wk = "layer.net.0.weight"
    if wk in sd and wk in pinn.state_dict():
        sw, tw = sd[wk], pinn.state_dict()[wk]
        if sw.shape != tw.shape and sw.shape[0] == tw.shape[0]:
            adapted = torch.zeros_like(tw)
            if sw.shape[1] == 8:
                adapted[:, 0:5] = sw[:, 0:5]; adapted[:, 8:11] = sw[:, 5:8]
            elif sw.shape[1] == 10:
                adapted[:, 0:7] = sw[:, 0:7]; adapted[:, 8:11] = sw[:, 7:10]
            sd[wk] = adapted
    pinn.load_state_dict(sd, strict=False)
    pinn.eval()
    return pinn, device, cfg


def load_three_layer_pinn():
    # Remove one-layer from path to avoid module collisions
    ol_str = str(REPO_ROOT / "one-layer-workflow")
    if ol_str in sys.path:
        sys.path.remove(ol_str)
    for mod in ["model", "pinn_config", "data", "physics", "soap"]:
        sys.modules.pop(mod, None)
    tl_dir = REPO_ROOT / "three-layer-workflow"
    sys.path.insert(0, str(tl_dir))
    import model as m
    import pinn_config as cfg
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pinn = m.MultiLayerPINN().to(device)
    ckpt = tl_dir / "pinn_model_final.pth"
    if not ckpt.exists():
        ckpt = tl_dir / "pinn_model.pth"
    sd = torch.load(ckpt, map_location=device, weights_only=True)
    if hasattr(m, "adapt_legacy_state_dict"):
        sd = m.adapt_legacy_state_dict(sd, pinn.state_dict())
    pinn.load_state_dict(sd, strict=False)
    pinn.eval()
    return pinn, device, cfg


# ── FEM runners ───────────────────────────────────────────────────────────────

def run_fem_1l(E, thickness, ne_x=NE_X, ne_y=NE_Y, ne_z=NE_Z):
    cfg = {
        "geometry": {"Lx": 1.0, "Ly": 1.0, "H": float(thickness),
                     "ne_x": ne_x, "ne_y": ne_y, "ne_z": ne_z},
        "material": {"E": float(E), "nu": 0.3},
        "load_patch": {"pressure": 1.0, "x_start": 1/3, "x_end": 2/3,
                       "y_start": 1/3, "y_end": 2/3},
    }
    xn, yn, zn, u = fem_solver.solve_fem(cfg)
    return np.array(xn), np.array(yn), np.array(zn), np.array(u)


def run_fem_3l(e1, e2, e3, t1, t2, t3, ne_x=NE_X, ne_y=NE_Y, ne_z=NE_Z):
    thickness = float(t1) + float(t2) + float(t3)
    cfg = {
        "geometry": {"Lx": 1.0, "Ly": 1.0, "H": thickness,
                     "ne_x": ne_x, "ne_y": ne_y, "ne_z": ne_z},
        "material": {"E_layers": [float(e1), float(e2), float(e3)],
                     "t_layers": [float(t1), float(t2), float(t3)], "nu": 0.3},
        "load_patch": {"pressure": 1.0, "x_start": 1/3, "x_end": 2/3,
                       "y_start": 1/3, "y_end": 2/3},
    }
    xn, yn, zn, u = fem_solver.solve_three_layer_fem(cfg)
    return np.array(xn), np.array(yn), np.array(zn), np.array(u)


# ── PINN evaluators ───────────────────────────────────────────────────────────

def eval_1l(pinn, device, cfg, E, thickness, xn, yn, zn):
    X, Y, Z = np.meshgrid(xn, yn, zn, indexing="ij")
    N = X.size
    r = float(getattr(cfg, "RESTITUTION_REF", 0.5))
    mu = float(getattr(cfg, "FRICTION_REF", 0.3))
    v0 = float(getattr(cfg, "IMPACT_VELOCITY_REF", 1.0))
    pts = np.column_stack([
        X.ravel(), Y.ravel(), Z.ravel(),
        np.full(N, E), np.full(N, thickness),
        np.full(N, r), np.full(N, mu), np.full(N, v0),
    ])
    with torch.no_grad():
        v = pinn(torch.tensor(pts, dtype=torch.float32, device=device), 0).cpu().numpy()
    e_pow = float(getattr(cfg, "E_COMPLIANCE_POWER", 1.0))
    alpha = float(getattr(cfg, "THICKNESS_COMPLIANCE_ALPHA", 0.0))
    H_ref = float(getattr(cfg, "H", 1.0))
    t_sc = 1.0 if alpha == 0.0 else (H_ref / max(float(thickness), 1e-8)) ** alpha
    u = (v / (float(E) ** e_pow)) * t_sc
    return u.reshape(X.shape + (3,))


def eval_3l(pinn, device, cfg, e1, e2, e3, t1, t2, t3, xn, yn, zn):
    X, Y, Z = np.meshgrid(xn, yn, zn, indexing="ij")
    N = X.size
    r = float(getattr(cfg, "RESTITUTION_REF", 0.5))
    mu = float(getattr(cfg, "FRICTION_REF", 0.3))
    v0 = float(getattr(cfg, "IMPACT_VELOCITY_REF", 1.0))
    pts = np.column_stack([
        X.ravel(), Y.ravel(), Z.ravel(),
        np.full(N, e1), np.full(N, e2), np.full(N, e3),
        np.full(N, t1), np.full(N, t2), np.full(N, t3),
        np.full(N, r), np.full(N, mu), np.full(N, v0),
    ])
    with torch.no_grad():
        v = pinn(torch.tensor(pts, dtype=torch.float32, device=device), 0).cpu().numpy()
    e_pow = float(getattr(cfg, "E_COMPLIANCE_POWER", 1.0))
    alpha = float(getattr(cfg, "THICKNESS_COMPLIANCE_ALPHA", 0.0))
    H_ref = float(getattr(cfg, "H", 1.0))
    thickness = float(t1) + float(t2) + float(t3)
    t_sc = 1.0 if alpha == 0.0 else (H_ref / max(thickness, 1e-8)) ** alpha
    E_avg = (float(e1) + float(e2) + float(e3)) / 3.0
    u = (v / (E_avg ** e_pow)) * t_sc
    return u.reshape(X.shape + (3,))


# ── Upsampling ────────────────────────────────────────────────────────────────

def upsample(xn, zn, data, n=500):
    interp = RegularGridInterpolator(
        (xn, zn), data, method="cubic", bounds_error=False, fill_value=0.0
    )
    xf = np.linspace(xn.min(), xn.max(), n)
    zf = np.linspace(zn.min(), zn.max(), n)
    Xf, Zf = np.meshgrid(xf, zf, indexing="ij")
    return xf, zf, interp((Xf, Zf))


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_heatmap(xn, zn, abs_err, title, vmax, interfaces, out_path):
    xf, zf, err_fine = upsample(xn, zn, abs_err)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    im = ax.imshow(
        err_fine.T,
        extent=[xf.min(), xf.max(), zf.min(), zf.max()],
        origin="lower",
        cmap=CMAP,
        aspect="auto",
        interpolation="bilinear",
        vmin=0.0,
        vmax=vmax,
    )
    ax.set_xlabel("x (m)", fontsize=12)
    ax.set_ylabel("z (m)", fontsize=12)
    ax.set_title(title, fontsize=10)

    if interfaces:
        for z_if in interfaces:
            ax.axhline(z_if, color="white", linestyle="--", linewidth=2.0, alpha=0.9)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Absolute Error (m)", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close(fig)


def make_title(label, n_cases, cs_mae, cs_max, top_mae_pct):
    return (
        f"{label} (N={n_cases}): "
        f"CS MAE={cs_mae:.2e} m | CS Max={cs_max:.2e} m | "
        f"Top MAE={top_mae_pct:.2f}%"
    )


def process_case(label, top_mae_pct, xn, zn, u_fem_cs, u_pinn_cs, interfaces, n_cases):
    abs_err = np.abs(u_pinn_cs - u_fem_cs)
    cs_mae = float(np.mean(abs_err))
    cs_max = float(np.max(abs_err))
    title = make_title(label, n_cases, cs_mae, cs_max, top_mae_pct)
    return abs_err, cs_mae, cs_max, title


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    csv_1l = REPO_ROOT / "graphs" / "generalized_study" / "one_layer_random_100.csv"
    csv_3l = REPO_ROOT / "graphs" / "generalized_study" / "three_layer_random_100.csv"

    print("Reading CSVs...")
    one_best, one_worst = find_best_worst(csv_1l)
    three_best, three_worst = find_best_worst(csv_3l)

    n1 = sum(1 for _ in open(csv_1l)) - 1
    n3 = sum(1 for _ in open(csv_3l)) - 1

    print(f"1L best:  {one_best['case_id']}  MAE={float(one_best['top_uz_mae_pct']):.3f}%")
    print(f"1L worst: {one_worst['case_id']} MAE={float(one_worst['top_uz_mae_pct']):.3f}%")
    print(f"3L best:  {three_best['case_id']} MAE={float(three_best['top_uz_mae_pct']):.3f}%")
    print(f"3L worst: {three_worst['case_id']} MAE={float(three_worst['top_uz_mae_pct']):.3f}%")

    print("\nLoading models...")
    pinn1, dev1, cfg1 = load_one_layer_pinn()
    pinn3, dev3, cfg3 = load_three_layer_pinn()

    cases = []

    # ── 1L BEST ──────────────────────────────────────────────────────────────
    print("\n[1L Best] Running FEM + PINN...")
    E_b = float(one_best["E"]); t_b = float(one_best["thickness"])
    xn, yn, zn, u_fem = run_fem_1l(E_b, t_b)
    u_pinn = eval_1l(pinn1, dev1, cfg1, E_b, t_b, xn, yn, zn)
    mi = len(yn) // 2
    abs_err, cs_mae, cs_max, title = process_case(
        f"1L Best | E={E_b:.2f}, t={t_b:.3f} m",
        float(one_best["top_uz_mae_pct"]),
        xn, zn, u_fem[:, mi, :, 2], u_pinn[:, mi, :, 2],
        interfaces=None, n_cases=n1,
    )
    cases.append(("one_layer_best", title, xn, zn, abs_err, None))

    # ── 1L WORST ─────────────────────────────────────────────────────────────
    print("[1L Worst] Running FEM + PINN...")
    E_w = float(one_worst["E"]); t_w = float(one_worst["thickness"])
    xn2, yn2, zn2, u_fem2 = run_fem_1l(E_w, t_w)
    u_pinn2 = eval_1l(pinn1, dev1, cfg1, E_w, t_w, xn2, yn2, zn2)
    mi2 = len(yn2) // 2
    abs_err2, cs_mae2, cs_max2, title2 = process_case(
        f"1L Worst | E={E_w:.2f}, t={t_w:.3f} m",
        float(one_worst["top_uz_mae_pct"]),
        xn2, zn2, u_fem2[:, mi2, :, 2], u_pinn2[:, mi2, :, 2],
        interfaces=None, n_cases=n1,
    )
    cases.append(("one_layer_worst", title2, xn2, zn2, abs_err2, None))

    # ── 3L BEST ──────────────────────────────────────────────────────────────
    print("[3L Best] Running FEM + PINN...")
    e1b, e2b, e3b = float(three_best["e1"]), float(three_best["e2"]), float(three_best["e3"])
    t1b, t2b, t3b = float(three_best["t1"]), float(three_best["t2"]), float(three_best["t3"])
    H3b = t1b + t2b + t3b
    xn3, yn3, zn3, u_fem3 = run_fem_3l(e1b, e2b, e3b, t1b, t2b, t3b)
    u_pinn3 = eval_3l(pinn3, dev3, cfg3, e1b, e2b, e3b, t1b, t2b, t3b, xn3, yn3, zn3)
    mi3 = len(yn3) // 2
    abs_err3, cs_mae3, cs_max3, title3 = process_case(
        f"3L Best | E=[{e1b:.1f},{e2b:.1f},{e3b:.1f}], t=[{t1b:.3f},{t2b:.3f},{t3b:.3f}] m",
        float(three_best["top_uz_mae_pct"]),
        xn3, zn3, u_fem3[:, mi3, :, 2], u_pinn3[:, mi3, :, 2],
        interfaces=[t1b, t1b + t2b], n_cases=n3,
    )
    cases.append(("three_layer_best", title3, xn3, zn3, abs_err3, [t1b, t1b + t2b]))

    # ── 3L WORST ─────────────────────────────────────────────────────────────
    print("[3L Worst] Running FEM + PINN...")
    e1w, e2w, e3w = float(three_worst["e1"]), float(three_worst["e2"]), float(three_worst["e3"])
    t1w, t2w, t3w = float(three_worst["t1"]), float(three_worst["t2"]), float(three_worst["t3"])
    xn4, yn4, zn4, u_fem4 = run_fem_3l(e1w, e2w, e3w, t1w, t2w, t3w)
    u_pinn4 = eval_3l(pinn3, dev3, cfg3, e1w, e2w, e3w, t1w, t2w, t3w, xn4, yn4, zn4)
    mi4 = len(yn4) // 2
    abs_err4, cs_mae4, cs_max4, title4 = process_case(
        f"3L Worst | E=[{e1w:.1f},{e2w:.1f},{e3w:.1f}], t=[{t1w:.3f},{t2w:.3f},{t3w:.3f}] m",
        float(three_worst["top_uz_mae_pct"]),
        xn4, zn4, u_fem4[:, mi4, :, 2], u_pinn4[:, mi4, :, 2],
        interfaces=[t1w, t1w + t2w], n_cases=n3,
    )
    cases.append(("three_layer_worst", title4, xn4, zn4, abs_err4, [t1w, t1w + t2w]))

    # ── Save figures (fixed scale 0–0.3 m, PNG + PDF) ────────────────────────
    print(f"\nFixed colorbar scale: 0 – {FIXED_VMAX} m")
    print("Saving figures...")
    for stem, title, xn, zn, abs_err, interfaces in cases:
        for ext in (".png", ".pdf"):
            plot_heatmap(xn, zn, abs_err, title, FIXED_VMAX, interfaces,
                         OUT_DIR / f"{stem}_abs_error_heatmap{ext}")

    print("\nDone! 8 files (4 PNG + 4 PDF) written to", OUT_DIR)


if __name__ == "__main__":
    main()
