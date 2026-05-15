"""
Generate 3-panel cross-section visualization: FEA Cross-Section | PINN Cross-Section | Abs Error Cross-Section
Matches the exact style of the reference figure.
Uses CSV best case parameters and correct compliance scaling.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
THREE_LAYER_DIR = REPO_ROOT / "three-layer-workflow"
FEA_DIR = REPO_ROOT / "fea-workflow" / "solver"
OUTPUT_DIR = REPO_ROOT / "graphs" / "generalized_study"
CSV_THREE = OUTPUT_DIR / "three_layer_random_100.csv"

for _path in (FEA_DIR,):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.interpolate
import torch
import pandas as pd

import fem_solver

DPI = 600


def load_three_layer_model():
    original_modules = dict(sys.modules)
    original_path = sys.path.copy()
    sys.path.insert(0, str(THREE_LAYER_DIR))
    import pinn_config as tl_config
    sys.modules['pinn_config'] = tl_config
    from model import MultiLayerPINN
    device = torch.device('cpu')
    model = MultiLayerPINN().to(device)
    model.load_state_dict(torch.load(
        THREE_LAYER_DIR / 'pinn_model_final.pth', map_location=device))
    model.eval()
    # Restore
    sys.path = original_path
    for key in list(sys.modules.keys()):
        if key not in original_modules:
            del sys.modules[key]
    sys.modules.update(original_modules)
    return model, tl_config


def main():
    model, tl_config = load_three_layer_model()
    
    # Get best case from CSV
    df = pd.read_csv(CSV_THREE)
    best = df.loc[df['top_uz_mae_pct'].idxmin()]
    
    E1, E2, E3 = best['e1'], best['e2'], best['e3']
    t1, t2, t3 = best['t1'], best['t2'], best['t3']
    H = t1 + t2 + t3
    nu = 0.3
    load = 1.0
    
    print(f"Using case: {best['case_id']}")
    print(f"E=[{E1:.2f}, {E2:.2f}, {E3:.2f}], t=[{t1:.4f}, {t2:.4f}, {t3:.4f}], H={H:.4f}")
    
    # FEM reference on fine mesh
    cfg = {
        "geometry": {"Lx": 1.0, "Ly": 1.0, "H": H, "ne_x": 32, "ne_y": 32, "ne_z": 16},
        "material": {"E_layers": [E1, E2, E3], "t_layers": [t1, t2, t3], "nu": nu},
        "load_patch": {"pressure": load, "x_start": 1/3, "x_end": 2/3, "y_start": 1/3, "y_end": 2/3},
    }
    x, y, z, u_fem = fem_solver.solve_three_layer_fem(cfg)
    
    # Create meshgrid for proper masking
    xg, yg, zg = np.meshgrid(x, y, z, indexing='ij')
    
    # Extract cross-section at y = 0.5
    y_target = 0.5
    tol = 1e-3
    mask = np.abs(yg - y_target) < tol
    x_cs = xg[mask]
    z_cs = zg[mask]
    u_fem_cs = u_fem[:,:,:,2][mask]
    
    # PINN prediction on ALL grid points (then extract cross-section)
    X = xg.ravel()
    Y = yg.ravel()
    Z = zg.ravel()
    
    r_ref = float(getattr(tl_config, 'RESTITUTION_REF', 0.5))
    mu_ref = float(getattr(tl_config, 'FRICTION_REF', 0.3))
    v0_ref = float(getattr(tl_config, 'IMPACT_VELOCITY_REF', 1.0))
    
    pts = np.stack([
        X, Y, Z,
        np.full_like(X, E1, dtype=float),
        np.full_like(X, t1, dtype=float),
        np.full_like(X, E2, dtype=float),
        np.full_like(X, t2, dtype=float),
        np.full_like(X, E3, dtype=float),
        np.full_like(X, t3, dtype=float),
        np.full_like(X, r_ref, dtype=float),
        np.full_like(X, mu_ref, dtype=float),
        np.full_like(X, v0_ref, dtype=float),
    ], axis=1)
    
    with torch.no_grad():
        v = model(torch.tensor(pts, dtype=torch.float32)).numpy()
    
    # Apply compliance scaling (same as training)
    e_scale = (pts[:, 3:4] + pts[:, 5:6] + pts[:, 7:8]) / 3.0
    t_scale = pts[:, 4:5] + pts[:, 6:7] + pts[:, 8:9]
    e_pow = float(getattr(tl_config, 'E_COMPLIANCE_POWER', 1.0))
    alpha = float(getattr(tl_config, 'THICKNESS_COMPLIANCE_ALPHA', 0.0))
    scale = float(getattr(tl_config, 'DISPLACEMENT_COMPLIANCE_SCALE', 1.0))
    h_ref = float(getattr(tl_config, 'H', 1.0))
    
    u_pred = scale * v / (e_scale ** e_pow) * (h_ref / np.clip(t_scale, 1e-8, None)) ** alpha
    u_pred_reshaped = u_pred.reshape(u_fem.shape)
    
    u_pinn_cs = u_pred_reshaped[:,:,:,2][mask]
    
    # Absolute error
    abs_err = np.abs(u_pinn_cs - u_fem_cs)
    mae = np.mean(abs_err)
    
    # Interpolate to regular grid for smooth plotting
    xi = np.linspace(0, 1, 500)
    zi = np.linspace(0, H, 500)
    Xi, Zi = np.meshgrid(xi, zi)
    
    points = np.column_stack([x_cs, z_cs])
    U_fem = scipy.interpolate.griddata(points, u_fem_cs, (Xi, Zi), method='cubic')
    U_pinn = scipy.interpolate.griddata(points, u_pinn_cs, (Xi, Zi), method='cubic')
    U_err = scipy.interpolate.griddata(points, abs_err, (Xi, Zi), method='cubic')
    
    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Common color scale for FEA and PINN
    vmin = min(np.nanmin(U_fem), np.nanmin(U_pinn))
    vmax = max(np.nanmax(U_fem), np.nanmax(U_pinn))
    
    # Interface lines
    interface1 = t1
    interface2 = t1 + t2
    
    # Panel 1: FEA Cross-Section
    im1 = axes[0].imshow(U_fem, extent=[0, 1, 0, H], origin='lower',
                         aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
    axes[0].axhline(y=interface1, color='white', linewidth=3)
    axes[0].axhline(y=interface2, color='white', linewidth=3)
    axes[0].set_title('Three-Layer FEA Cross-Section', fontsize=12)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('z')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Panel 2: PINN Cross-Section
    im2 = axes[1].imshow(U_pinn, extent=[0, 1, 0, H], origin='lower',
                         aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
    axes[1].axhline(y=interface1, color='white', linewidth=3)
    axes[1].axhline(y=interface2, color='white', linewidth=3)
    axes[1].set_title('Three-Layer PINN Cross-Section', fontsize=12)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('z')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Panel 3: Absolute Error Cross-Section
    im3 = axes[2].imshow(U_err, extent=[0, 1, 0, H], origin='lower',
                         aspect='auto', cmap='magma', vmin=0)
    axes[2].axhline(y=interface1, color='white', linewidth=3)
    axes[2].axhline(y=interface2, color='white', linewidth=3)
    axes[2].set_title(f'Abs Error Cross-Section\nMAE={mae:.5f}', fontsize=12)
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('z')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'three_layer_cross_section_comparison.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == '__main__':
    main()
