#!/usr/bin/env python3
"""Generate paper-quality PINN vs FEM comparison figures (6-page PDF).

Uses the existing compare_three_layer_pinn_fem.py pipeline for three-layer
and a parallel one-layer pipeline, including calibration where available.
"""
import csv, os, sys, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
FEA_DIR = REPO / "fea-workflow" / "solver"
sys.path.insert(0, str(FEA_DIR))
sys.path.insert(0, str(REPO / "scripts"))
import fem_solver
from three_layer_experiment_utils import (
    ThreeLayerCase,
    load_pinn as _load_pinn_utils,
    make_points,
    predict_displacement,
    solve_fem_case,
)

Lx, Ly, p0, nu = 1.0, 1.0, 1.0, 0.3
NE = (16, 16, 8)
PATCH = {"pressure": p0, "x_start": 1/3, "x_end": 2/3, "y_start": 1/3, "y_end": 2/3}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 28,
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",
    "axes.titlesize": 30,
    "axes.labelsize": 28,
    "xtick.labelsize": 23,
    "ytick.labelsize": 23,
    "axes.linewidth": 1.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
CMAP_F, CMAP_E = "jet", "magma"
IF_LW, IF_COL = 5.0, "white"

# ── Load three-layer PINN ────────────────────────────────────────────────────
def load_3l(model_path=None):
    """Load the 3-layer PINN via the canonical utils pipeline (includes calibration)."""
    device = torch.device("cpu")
    path = Path(model_path) if model_path else REPO / "three-layer-workflow" / "pinn_model_final.pth"
    pinn, _ = _load_pinn_utils(device, path)
    return pinn, device

def predict_3l(pinn, device, xf, yf, zf, e1, e2, e3, t1, t2, t3):
    """Predict displacements using the canonical pipeline (compliance + calibration)."""
    case = ThreeLayerCase("fig", e1, e2, e3, t1, t2, t3)
    pts = make_points(xf, yf, zf, case)
    return predict_displacement(pinn, device, pts)

# ── Load one-layer PINN + config ────────────────────────────────────────────
def load_1l():
    d = REPO / "one-layer-workflow"
    for m in ["pinn_config","model"]:
        if m in sys.modules: del sys.modules[m]
    sys.path.insert(0, str(d))
    import importlib
    cfg = importlib.import_module("pinn_config"); mdl = importlib.import_module("model")
    p = mdl.MultiLayerPINN()
    sd = torch.load(d/"pinn_model.pth", map_location="cpu", weights_only=True)
    p.load_state_dict(sd, strict=False); p.eval()
    return p, cfg

def predict_1l(pinn, cfg, xf, yf, zf, E, t):
    r = float(getattr(cfg,"RESTITUTION_REF",0.5))
    mu = float(getattr(cfg,"FRICTION_REF",0.3))
    v0 = float(getattr(cfg,"IMPACT_VELOCITY_REF",1.0))
    pts = np.stack([xf,yf,zf]+[np.full_like(xf,v) for v in [E,t,r,mu,v0]], axis=1)
    with torch.no_grad():
        v = pinn(torch.tensor(pts, dtype=torch.float32)).cpu().numpy()
    ep = float(getattr(cfg,"E_COMPLIANCE_POWER",1.0))
    al = float(getattr(cfg,"THICKNESS_COMPLIANCE_ALPHA",0.0))
    sc = float(getattr(cfg,"DISPLACEMENT_COMPLIANCE_SCALE",1.0))
    hr = float(getattr(cfg,"H",0.1))
    return sc * v / (pts[:,3:4]**ep) * (hr/np.clip(pts[:,4:5],1e-8,None))**al

# ── FEM solvers ──────────────────────────────────────────────────────────────
def fem_3l(e1,e2,e3,t1,t2,t3):
    cfg = {"geometry":{"Lx":Lx,"Ly":Ly,"H":t1+t2+t3,"ne_x":NE[0],"ne_y":NE[1],"ne_z":NE[2]},
           "material":{"E_layers":[e1,e2,e3],"t_layers":[t1,t2,t3],"nu":nu},"load_patch":PATCH}
    return fem_solver.solve_three_layer_fem(cfg)

def fem_1l(E,t):
    cfg = {"geometry":{"Lx":Lx,"Ly":Ly,"H":t,"ne_x":NE[0],"ne_y":NE[1],"ne_z":NE[2]},
           "material":{"E":E,"nu":nu},"load_patch":PATCH}
    return fem_solver.solve_fem(cfg)

def mae_pct(pred, ref):
    d = float(np.max(np.abs(ref)))
    return 100*float(np.mean(np.abs(pred-ref)))/d if d>0 else 0

def worst_volume_case(csv_path):
    with Path(csv_path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    return max(rows, key=lambda row: float(row["volume_mae_pct"]))

# ── Plotting ─────────────────────────────────────────────────────────────────
def draw_interfaces(ax, interfaces):
    if not interfaces: return
    for z in interfaces:
        ax.axhline(z, color=IF_COL, linestyle="-", linewidth=IF_LW, alpha=1.0)

def add_colorbar(fig, contour, ax):
    colorbar = fig.colorbar(contour, ax=ax, pad=0.025)
    colorbar.ax.tick_params(labelsize=21, width=1.2, length=5)
    colorbar.ax.yaxis.get_offset_text().set_size(21)
    return colorbar

def page_top(pdf, xg, yg, uz_fea, uz_pinn, volume_mae, title, param, pdf_path=None, png_prefix=""):
    fig, axes = plt.subplots(1,3,figsize=(18,5.7))
    vmin = min(float(np.min(uz_fea)), float(np.min(uz_pinn))); vmax = max(float(np.max(uz_fea)), float(np.max(uz_pinn)))
    levels_field = np.linspace(vmin, vmax, 51)
    for ax, data, ttl in [(axes[0],uz_fea,f"{title} FEA\n{param}"), (axes[1],uz_pinn,f"{title} PINN")]:
        c = ax.contourf(xg,yg,data,levels=levels_field,cmap=CMAP_F)
        add_colorbar(fig,c,ax); ax.set_title(ttl, pad=12); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
    err = np.abs(uz_pinn-uz_fea)
    c = axes[2].contourf(xg,yg,err,levels=50,cmap=CMAP_E)
    add_colorbar(fig,c,axes[2]); axes[2].set_title(f"Abs Error\nVolume MAE={volume_mae:.2f}%", pad=12)
    axes[2].set_xlabel("x"); axes[2].set_ylabel("y"); axes[2].set_aspect("equal")
    fig.tight_layout(pad=0.8, w_pad=1.0); pdf.savefig(fig,dpi=600,bbox_inches="tight")
    if png_prefix:
        fig.savefig(pdf_path.parent / f"fig_{png_prefix}_top.png", dpi=600, bbox_inches="tight")
        fig.savefig(pdf_path.parent / f"fig_{png_prefix}_top.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig); return err

def page_cross(pdf, xc, zc, uz_fea, uz_pinn, volume_mae, title, interfaces, pdf_path=None, png_prefix=""):
    fig, axes = plt.subplots(1,3,figsize=(18,5.7))
    vmin = min(float(np.min(uz_fea)), float(np.min(uz_pinn))); vmax = max(float(np.max(uz_fea)), float(np.max(uz_pinn)))
    levels_field = np.linspace(vmin, vmax, 51)
    for ax, data, ttl in [(axes[0],uz_fea,f"{title} FEA Cross-Section"),(axes[1],uz_pinn,f"{title} PINN Cross-Section")]:
        c = ax.contourf(xc,zc,data,levels=levels_field,cmap=CMAP_F)
        add_colorbar(fig,c,ax); ax.set_title(ttl, pad=12); ax.set_xlabel("x"); ax.set_ylabel("z")
        draw_interfaces(ax, interfaces)
    err = np.abs(uz_pinn-uz_fea)
    c = axes[2].contourf(xc,zc,err,levels=50,cmap=CMAP_E)
    add_colorbar(fig,c,axes[2]); axes[2].set_title(f"Abs Error Cross-Section\nVolume MAE={volume_mae:.2f}%", pad=12)
    axes[2].set_xlabel("x"); axes[2].set_ylabel("z"); draw_interfaces(axes[2], interfaces)
    fig.tight_layout(pad=0.8, w_pad=1.0); pdf.savefig(fig,dpi=600,bbox_inches="tight")
    if pdf_path and png_prefix:
        fig.savefig(pdf_path.parent / f"fig_{png_prefix}_cross.png", dpi=600, bbox_inches="tight")
        fig.savefig(pdf_path.parent / f"fig_{png_prefix}_cross.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig); return err

def page_error_maps(pdf, xg, yg, err_top, xc, zc, err_cross, volume_mae, title, interfaces, pdf_path=None, png_prefix=""):
    fig, axes = plt.subplots(1,2,figsize=(14,5.7))
    c0 = axes[0].contourf(xg,yg,err_top,levels=50,cmap=CMAP_E)
    add_colorbar(fig,c0,axes[0]); axes[0].set_title(f"Abs Error\nVolume MAE={volume_mae:.2f}%", pad=12)
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y"); axes[0].set_aspect("equal")
    c1 = axes[1].contourf(xc,zc,err_cross,levels=50,cmap=CMAP_E)
    add_colorbar(fig,c1,axes[1]); axes[1].set_title(f"Abs Error Cross-Section\nVolume MAE={volume_mae:.2f}%", pad=12)
    axes[1].set_xlabel("x"); axes[1].set_ylabel("z"); draw_interfaces(axes[1], interfaces)
    fig.suptitle(title, fontsize=32, fontweight="bold", y=1.04)
    fig.tight_layout(pad=0.8, w_pad=1.0); pdf.savefig(fig,dpi=600,bbox_inches="tight")
    if pdf_path and png_prefix:
        fig.savefig(pdf_path.parent / f"fig_{png_prefix}_errors.png", dpi=600, bbox_inches="tight")
        fig.savefig(pdf_path.parent / f"fig_{png_prefix}_errors.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    out = REPO/"graphs"/"figures"; out.mkdir(parents=True, exist_ok=True)
    pdf_path = out/"pinn_fem_comparison_figures.pdf"

    # Use the worst-volume-MAE cases from the paper's fixed-seed (20260430)
    # 100-case evaluations so the plotted labels match the reported results.
    row3 = worst_volume_case(REPO/"graphs"/"data"/"three_layer_final_supervised_random100.csv")
    e1,e2,e3 = (float(row3[key]) for key in ("e1","e2","e3"))
    t1,t2,t3 = (float(row3[key]) for key in ("t1","t2","t3"))
    H3 = t1+t2+t3; ifs3 = [t1, t1+t2]
    row1 = worst_volume_case(REPO/"graphs"/"data"/"one_layer_random_generalization.csv")
    E1l, H1l = float(row1["E"]), float(row1["thickness"])

    # Three-layer — use canonical pipeline (compliance + calibration)
    pinn3, device3 = load_3l()
    case3 = ThreeLayerCase("fig3l", e1, e2, e3, t1, t2, t3)
    print("Running 3L FEM...")
    xn3, yn3, zn3, u3, _ = solve_fem_case(case3, *NE)
    xn3,yn3,zn3,u3 = [np.array(a,dtype=float) for a in [xn3,yn3,zn3,u3]]
    xg3,yg3 = np.meshgrid(xn3,yn3,indexing="ij")
    uz_fea_top3 = u3[:,:,-1,2]
    uz_pinn_top3 = predict_3l(pinn3,device3,xg3.ravel(),yg3.ravel(),
        np.full(xg3.size,H3),e1,e2,e3,t1,t2,t3).reshape(len(xn3),len(yn3),3)[:,:,2]
    mi = len(yn3)//2
    xc3,zc3 = np.meshgrid(xn3,zn3,indexing="ij")
    uz_fea_cs3 = u3[:,mi,:,2]
    uz_pinn_cs3 = predict_3l(pinn3,device3,xc3.ravel(),np.full(xc3.size,yn3[mi]),
        zc3.ravel(),e1,e2,e3,t1,t2,t3).reshape(len(xn3),len(zn3),3)[:,:,2]
    x3,y3,z3 = np.meshgrid(xn3,yn3,zn3,indexing="ij")
    u3_pinn = predict_3l(
        pinn3,device3,x3.ravel(),y3.ravel(),z3.ravel(),e1,e2,e3,t1,t2,t3
    ).reshape(u3.shape)
    volume_mae3 = mae_pct(u3_pinn,u3)

    # One-layer
    for m in ["pinn_config","model"]:
        if m in sys.modules: del sys.modules[m]
    pinn1, cfg1 = load_1l()
    print("Running 1L FEM..."); xn1,yn1,zn1,u1 = fem_1l(E1l,H1l)
    xn1,yn1,zn1,u1 = [np.array(a,dtype=float) for a in [xn1,yn1,zn1,u1]]
    xg1,yg1 = np.meshgrid(xn1,yn1,indexing="ij")
    uz_fea_top1 = u1[:,:,-1,2]
    uz_pinn_top1 = predict_1l(pinn1,cfg1,xg1.ravel(),yg1.ravel(),
        np.full(xg1.size,H1l),E1l,H1l).reshape(len(xn1),len(yn1),3)[:,:,2]
    mi1 = len(yn1)//2
    xc1,zc1 = np.meshgrid(xn1,zn1,indexing="ij")
    uz_fea_cs1 = u1[:,mi1,:,2]
    uz_pinn_cs1 = predict_1l(pinn1,cfg1,xc1.ravel(),np.full(xc1.size,yn1[mi1]),
        zc1.ravel(),E1l,H1l).reshape(len(xn1),len(zn1),3)[:,:,2]
    x1,y1,z1 = np.meshgrid(xn1,yn1,zn1,indexing="ij")
    u1_pinn = predict_1l(
        pinn1,cfg1,x1.ravel(),y1.ravel(),z1.ravel(),E1l,H1l
    ).reshape(u1.shape)
    volume_mae1 = mae_pct(u1_pinn,u1)

    # Generate PDF
    print(f"Writing PDF: {pdf_path}")
    with PdfPages(str(pdf_path)) as pdf:
        ps3 = f"E=[{e1:.3f},{e2:.3f},{e3:.3f}], t=[{t1:.3f},{t2:.3f},{t3:.3f}]"
        e1t = page_top(pdf,xg3,yg3,uz_fea_top3,uz_pinn_top3,volume_mae3,"Three-Layer",ps3,pdf_path,"3l")
        print(f"  P1: 3L volume MAE={volume_mae3:.6f}%")
        e2c = page_cross(pdf,xc3,zc3,uz_fea_cs3,uz_pinn_cs3,volume_mae3,"Three-Layer",ifs3,pdf_path,"3l")
        print(f"  P2: 3L volume MAE={volume_mae3:.6f}%")
        page_error_maps(pdf,xg3,yg3,e1t,xc3,zc3,e2c,volume_mae3,f"Three-Layer Error Maps — {ps3}",ifs3,pdf_path,"3l")
        print(f"  P3: 3L error maps")
        ps1 = f"E={E1l}, t={H1l:.3f}"
        e4t = page_top(pdf,xg1,yg1,uz_fea_top1,uz_pinn_top1,volume_mae1,"One-Layer",ps1,pdf_path,"1l")
        print(f"  P4: 1L volume MAE={volume_mae1:.6f}%")
        e5c = page_cross(pdf,xc1,zc1,uz_fea_cs1,uz_pinn_cs1,volume_mae1,"One-Layer",None,pdf_path,"1l")
        print(f"  P5: 1L volume MAE={volume_mae1:.6f}%")
        page_error_maps(pdf,xg1,yg1,e4t,xc1,zc1,e5c,volume_mae1,f"One-Layer Error Maps — {ps1}",None,pdf_path,"1l")
        print(f"  P6: 1L error maps")
    print(f"Done! {pdf_path}")

if __name__ == "__main__":
    main()
