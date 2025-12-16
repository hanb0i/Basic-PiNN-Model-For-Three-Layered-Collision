
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'pinn-workflow'))
import config
import model
from scipy.interpolate import RegularGridInterpolator

def compare():
    print("Loading FEA Solution...")
    data = np.load("fea_solution.npy", allow_pickle=True).item()
    X_fea = data['x'] # (nx, ny, nz)
    Y_fea = data['y']
    Z_fea = data['z']
    U_fea = data['u'] # (nx, ny, nz, 3)
    
    # Grid axes
    x_axis = X_fea[:, 0, 0]
    y_axis = Y_fea[0, :, 0]
    z_axis = Z_fea[0, 0, :]
    
    print(f"FEA Grid: {len(x_axis)}x{len(y_axis)}x{len(z_axis)}")
    
    # 1. Generate PINN Predictions on same Grid
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    pinn = model.MultiLayerPINN().to(device)
    model_path = "pinn_model.pth"
    if not os.path.exists(model_path):
        # Check in pinn-workflow directory
        potential_path = os.path.join(os.path.dirname(__file__), 'pinn-workflow', 'pinn_model.pth')
        if os.path.exists(potential_path):
            model_path = potential_path

    try:
        pinn.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Loaded PINN model from {model_path}")
    except Exception as e:
        print(f"PINN model not found or error loading: {e}")
        return
        
    pinn.eval()
    
    pts = np.stack([X_fea.ravel(), Y_fea.ravel(), Z_fea.ravel()], axis=1) # (N, 3)
    
    # We need to query layer-wise because PINN takes layer_idx
    # Layer interfaces: 0, 0.033, 0.066, 0.1
    # Z coordinate determines layer
    z_flat = pts[:, 2]
    
    # Masks
    eps = 1e-5
    mask1 = (z_flat >= config.Layer_Interfaces[0] - eps) & (z_flat <= config.Layer_Interfaces[1] + eps)
    mask2 = (z_flat >= config.Layer_Interfaces[1] - eps) & (z_flat <= config.Layer_Interfaces[2] + eps)
    mask3 = (z_flat >= config.Layer_Interfaces[2] - eps) & (z_flat <= config.Layer_Interfaces[3] + eps)
    
    # Prioritize higher layers for overlaps (standard practice or arbitrary)
    # Actually, interfaces match, so just pick one.
    mask2 = mask2 & (~mask1) # simple exclusivity cleanup if needed, but overlap is fine if continuous
    # Better: strict intervals
    m1 = z_flat <= config.Layer_Interfaces[1]
    m2 = (z_flat > config.Layer_Interfaces[1]) & (z_flat <= config.Layer_Interfaces[2])
    m3 = z_flat > config.Layer_Interfaces[2]
    
    U_pinn_flat = np.zeros_like(pts)
    
    with torch.no_grad():
        # Layer 1
        p1 = torch.tensor(pts[m1], dtype=torch.float32).to(device)
        if len(p1) > 0:
            U_pinn_flat[m1] = pinn(p1, 0).cpu().numpy()
            
        # Layer 2
        p2 = torch.tensor(pts[m2], dtype=torch.float32).to(device)
        if len(p2) > 0:
            U_pinn_flat[m2] = pinn(p2, 1).cpu().numpy()
            
        # Layer 3
        p3 = torch.tensor(pts[m3], dtype=torch.float32).to(device)
        if len(p3) > 0:
            U_pinn_flat[m3] = pinn(p3, 2).cpu().numpy()
            
    U_pinn = U_pinn_flat.reshape(U_fea.shape)
    
    # 2. Compute Metrics
    # U_z at top surface
    u_z_fea_top = U_fea[:, :, -1, 2]
    u_z_pinn_top = U_pinn[:, :, -1, 2]
    
    abs_diff = np.abs(u_z_fea_top - u_z_pinn_top)
    mae = np.mean(abs_diff)
    max_err = np.max(abs_diff)
    
    print(f"Comparison Results (Top Surface u_z):")
    print(f"MAE: {mae:.6f}")
    print(f"Max Error: {max_err:.6f}")
    print(f"Peak Deflection FEA: {u_z_fea_top.min():.6f}")
    print(f"Peak Deflection PINN: {u_z_pinn_top.min():.6f}")
    
    # 3. Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Contours
    # FEA
    c1 = axes[0].contourf(X_fea[:,:,0], Y_fea[:,:,0], u_z_fea_top, levels=50, cmap='jet')
    axes[0].set_title("FEA Displacement u_z (Top)")
    plt.colorbar(c1, ax=axes[0])
    
    # PINN
    c2 = axes[1].contourf(X_fea[:,:,0], Y_fea[:,:,0], u_z_pinn_top, levels=50, cmap='jet')
    axes[1].set_title("PINN Displacement u_z (Top)")
    plt.colorbar(c2, ax=axes[1])
    
    # Error
    c3 = axes[2].contourf(X_fea[:,:,0], Y_fea[:,:,0], abs_diff, levels=50, cmap='magma')
    axes[2].set_title("Absolute Error |FEA - PINN|")
    plt.colorbar(c3, ax=axes[2])
    
    plt.savefig("comparison_top.png")
    print("Saved comparison_top.png")
    
    # Cross section
    # y index middle
    mid_y = U_fea.shape[1] // 2
    
    xz_X = X_fea[:, mid_y, :]
    xz_Z = Z_fea[:, mid_y, :]
    xz_Uz_fea = U_fea[:, mid_y, :, 2]
    xz_Uz_pinn = U_pinn[:, mid_y, :, 2]
    
    fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6))
    
    c4 = axes2[0].contourf(xz_X, xz_Z, xz_Uz_fea, levels=50, cmap='jet')
    axes2[0].set_title("FEA Cross Section u_z")
    plt.colorbar(c4, ax=axes2[0])
    
    c5 = axes2[1].contourf(xz_X, xz_Z, xz_Uz_pinn, levels=50, cmap='jet')
    axes2[1].set_title("PINN Cross Section u_z")
    plt.colorbar(c5, ax=axes2[1])
    
    plt.savefig("comparison_cross_section.png")
    print("Saved comparison_cross_section.png")

if __name__ == "__main__":
    compare()
