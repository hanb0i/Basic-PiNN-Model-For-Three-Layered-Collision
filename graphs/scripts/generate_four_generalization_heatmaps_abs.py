"""
Generate 4 generalization heatmaps with absolute error (no normalization).

This script:
1. Loads one-layer and three-layer PINN models (with proper config isolation)
2. Identifies best/worst cases from CSV files
3. Evaluates FEM reference and PINN prediction for each case
4. Generates cross-section (y=0.5) absolute error heatmaps
5. Produces both individual-scale and global-scale versions

Usage:
    cd /path/to/repo && python3 graphs/scripts/generate_four_generalization_heatmaps_abs.py

Output:
    graphs/generalized_study/{one,three}_layer_{best,worst}_abs_error_heatmap.png
    graphs/generalized_study/{one,three}_layer_{best,worst}_abs_error_heatmap_global_scale.png
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ONE_LAYER_DIR = REPO_ROOT / "one-layer-workflow"
THREE_LAYER_DIR = REPO_ROOT / "three-layer-workflow"
FEA_DIR = REPO_ROOT / "fea-workflow" / "solver"
GRAPH_DIR = REPO_ROOT / "graphs"
OUTPUT_DIR = GRAPH_DIR / "generalized_study"
CSV_ONE = OUTPUT_DIR / "one_layer_random_100.csv"
CSV_THREE = OUTPUT_DIR / "three_layer_random_100.csv"

for _path in (FEA_DIR,):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.interpolate
import torch

import fem_solver

# Config
DPI = 600
FIG_SIZE = (10, 6)
INTERP_GRID = (500, 500)
METHOD = "cubic"


def load_one_layer_model():
    """Load one-layer PINN with proper config isolation."""
    original_modules = dict(sys.modules)
    original_path = sys.path.copy()
    
    sys.path.insert(0, str(ONE_LAYER_DIR))
    import pinn_config as ol_config
    sys.modules['pinn_config'] = ol_config
    from model import MultiLayerPINN
    
    device = torch.device('cpu')
    model = MultiLayerPINN().to(device)
    model.load_state_dict(torch.load(
        ONE_LAYER_DIR / 'pinn_model.pth', map_location=device))
    model.eval()
    
    # Restore
    sys.path = original_path
    for key in list(sys.modules.keys()):
        if key not in original_modules:
            del sys.modules[key]
    sys.modules.update(original_modules)
    
    return model


def load_three_layer_model():
    """Load three-layer PINN with proper config isolation."""
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
    
    return model


def get_params_from_csv(csv_path, case_id):
    """Extract parameters from CSV row."""
    df = pd.read_csv(csv_path)
    row = df[df['case_id'] == case_id].iloc[0]
    
    # One-layer CSV has 'E' and 'thickness'; three-layer has 'e1','e2','e3','t1','t2','t3'
    if 'E' in row:
        # One-layer format
        E = row['E']
        thickness = row['thickness']
        return {
            'E1': E,
            'E2': E,
            'E3': E,
            'thickness': thickness,
            'nu': 0.3,  # Default from config
            'load': 1.0,  # Default from config
        }
    else:
        # Three-layer format
        return {
            'E1': row['e1'],
            'E2': row['e2'],
            'E3': row['e3'],
            't1': row['t1'],
            't2': row['t2'],
            't3': row['t3'],
            'nu': 0.3,  # Default from config
            'load': 1.0,  # Default from config
        }


def evaluate_case(model, params, is_one_layer=True):
    """Run FEM and PINN for a given parameter case."""
    E1, E2, E3 = params['E1'], params['E2'], params['E3']
    nu = params['nu']
    load = params['load']
    
    if is_one_layer:
        thickness = params['thickness']
        cfg = {
            "geometry": {"Lx": 1.0, "Ly": 1.0, "H": thickness, "ne_x": 16, "ne_y": 16, "ne_z": 8},
            "material": {"E": E1, "nu": nu},
            "load_patch": {"pressure": load, "x_start": 0.4, "x_end": 0.6, "y_start": 0.4, "y_end": 0.6},
        }
        benchmark = fem_solver.solve_fem(cfg)
        # Use same mesh for reference to match grid sizes
        reference = benchmark
    else:
        t1, t2, t3 = params['t1'], params['t2'], params['t3']
        thickness = t1 + t2 + t3
        cfg = {
            "geometry": {"Lx": 1.0, "Ly": 1.0, "H": thickness, "ne_x": 16, "ne_y": 16, "ne_z": 8},
            "material": {"E_layers": [E1, E2, E3], "t_layers": [t1, t2, t3], "nu": nu},
            "load_patch": {"pressure": load, "x_start": 0.4, "x_end": 0.6, "y_start": 0.4, "y_end": 0.6},
        }
        benchmark = fem_solver.solve_three_layer_fem(cfg)
        # Use same mesh for reference to match grid sizes
        reference = benchmark
    
    x_b, y_b, z_b, u_b = benchmark
    x_r, y_r, z_r, u_r = reference
    
    # Use benchmark grid for PINN evaluation
    xg, yg, zg = np.meshgrid(x_b, y_b, z_b, indexing='ij')
    X = xg.flatten()
    Y = yg.flatten()
    Z = zg.flatten()
    
    if is_one_layer:
        E_eff = (E1 + E2 + E3) / 3.0
        p = np.full_like(X, E_eff)
        q = np.full_like(X, nu)
        r = np.full_like(X, load)
        inputs = np.stack([X, Y, Z, p, q, r, p, q, r, p, q], axis=1)
    else:
        p1 = np.full_like(X, E1)
        q1 = np.full_like(X, nu)
        p2 = np.full_like(X, E2)
        q2 = np.full_like(X, nu)
        p3 = np.full_like(X, E3)
        q3 = np.full_like(X, nu)
        r = np.full_like(X, load)
        inputs = np.stack([X, Y, Z, p1, q1, p2, q2, p3, q3, r,
                           p1, q1, p2, q2, p3, q3, r,
                           p1, q1, p2, q2], axis=1)
    
    with torch.no_grad():
        pred = model(torch.tensor(inputs, dtype=torch.float32)).numpy()
    
    # Both benchmark and reference have shape (nx, ny, nz, 3)
    # Use uz component (index 2) and flatten
    u_fem = u_r[:,:,:,2].flatten()
    u_pinn = pred[:,2] if pred.ndim > 1 else pred.flatten()
    
    nodes = np.column_stack([X, Y, Z])
    abs_err = np.abs(u_pinn - u_fem)
    
    return nodes, abs_err, u_fem, u_pinn


def plot_heatmap(nodes, abs_err, title, output_path, vmax=None):
    """Plot cross-section absolute error heatmap."""
    y_val = 0.5
    tol = 1e-3
    mask = np.abs(nodes[:, 1] - y_val) < tol
    
    x_cs = nodes[mask, 0]
    z_cs = nodes[mask, 2]
    err_cs = abs_err[mask]
    
    xi = np.linspace(x_cs.min(), x_cs.max(), INTERP_GRID[0])
    zi = np.linspace(z_cs.min(), z_cs.max(), INTERP_GRID[1])
    Xi, Zi = np.meshgrid(xi, zi)
    
    points = np.column_stack([x_cs, z_cs])
    Ei = scipy.interpolate.griddata(points, err_cs, (Xi, Zi), method=METHOD)
    
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    if vmax is None:
        vmax = np.nanmax(Ei)
    
    im = ax.imshow(Ei, extent=[x_cs.min(), x_cs.max(), z_cs.min(), z_cs.max()],
                   origin='lower', aspect='auto', cmap='YlOrRd', vmin=0, vmax=vmax)
    
    cbar = plt.colorbar(im, ax=ax, label='Absolute Error (m)')
    
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('z (m)', fontsize=12)
    ax.set_title(title, fontsize=13)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Main entry point."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading one-layer model...")
    one_layer = load_one_layer_model()
    print("Loading three-layer model...")
    three_layer = load_three_layer_model()
    
    df_one = pd.read_csv(CSV_ONE)
    df_three = pd.read_csv(CSV_THREE)
    
    one_best = df_one.loc[df_one['top_uz_mae_pct'].idxmin(), 'case_id']
    one_worst = df_one.loc[df_one['top_uz_mae_pct'].idxmax(), 'case_id']
    three_best = df_three.loc[df_three['top_uz_mae_pct'].idxmin(), 'case_id']
    three_worst = df_three.loc[df_three['top_uz_mae_pct'].idxmax(), 'case_id']
    
    print(f"One-layer best: {one_best}")
    print(f"One-layer worst: {one_worst}")
    print(f"Three-layer best: {three_best}")
    print(f"Three-layer worst: {three_worst}")
    
    cases = [
        (one_best, one_layer, True, "1L Best", "one_layer_best"),
        (one_worst, one_layer, True, "1L Worst", "one_layer_worst"),
        (three_best, three_layer, False, "3L Best", "three_layer_best"),
        (three_worst, three_layer, False, "3L Worst", "three_layer_worst"),
    ]
    
    # First pass: compute all errors to find global vmax
    all_errs = []
    for case_id, model, is_one, label, fname in cases:
        print(f"Evaluating {label}...")
        params = get_params_from_csv(CSV_ONE if is_one else CSV_THREE, case_id)
        nodes, abs_err, u_fem, u_pinn = evaluate_case(model, params, is_one)
        
        y_val = 0.5
        tol = 1e-3
        mask = np.abs(nodes[:, 1] - y_val) < tol
        all_errs.append(abs_err[mask])
    
    global_vmax = max([e.max() for e in all_errs])
    print(f"Global max absolute error: {global_vmax:.6e}")
    
    # Second pass: plot each with individual and global scales
    for case_id, model, is_one, label, fname in cases:
        params = get_params_from_csv(CSV_ONE if is_one else CSV_THREE, case_id)
        nodes, abs_err, u_fem, u_pinn = evaluate_case(model, params, is_one)
        
        y_val = 0.5
        tol = 1e-3
        mask = np.abs(nodes[:, 1] - y_val) < tol
        local_vmax = abs_err[mask].max()
        
        df = df_one if is_one else df_three
        row = df[df['case_id'] == case_id].iloc[0]
        top_mae = row['top_uz_mae_pct']
        
        cs_mae = abs_err[mask].mean()
        cs_max = abs_err[mask].max()
        
        # Individual scale
        title = f"{label} (N=100): CS MAE={cs_mae:.2e} m | CS Max={cs_max:.2e} m | Top MAE={top_mae:.2f}%"
        plot_heatmap(nodes, abs_err, title,
                     OUTPUT_DIR / f"{fname}_abs_error_heatmap.png",
                     vmax=local_vmax)
        
        # Global scale
        title_global = f"{label} (N=100): CS MAE={cs_mae:.2e} m | CS Max={cs_max:.2e} m | Top MAE={top_mae:.2f}% [Global Scale]"
        plot_heatmap(nodes, abs_err, title_global,
                     OUTPUT_DIR / f"{fname}_abs_error_heatmap_global_scale.png",
                     vmax=global_vmax)
    
    print("Done!")


if __name__ == '__main__':
    main()
