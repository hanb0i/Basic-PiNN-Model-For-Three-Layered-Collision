import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import os

PINN_DIR = os.path.join(os.path.dirname(__file__), "pinn-workflow")
if PINN_DIR not in sys.path:
    sys.path.insert(0, PINN_DIR)

import pinn_config as config
import model
import physics

def _load_model(device):
    pinn = model.MultiLayerPINN().to(device)
    model_path = "pinn_model.pth"
    if not os.path.exists(model_path):
        potential_path = os.path.join(os.path.dirname(__file__), "pinn-workflow", "pinn_model.pth")
        if os.path.exists(potential_path):
            model_path = potential_path
    try:
        pinn.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Loaded PINN model from {model_path}")
    except Exception as e:
        print(f"PINN model not found or error loading: {e}")
        return None
    pinn.eval()
    return pinn

def _predict_points(pinn, pts, device):
    z_flat = pts[:, 2]
    m1 = z_flat <= config.Layer_Interfaces[1]
    m2 = (z_flat > config.Layer_Interfaces[1]) & (z_flat <= config.Layer_Interfaces[2])
    m3 = z_flat > config.Layer_Interfaces[2]

    u_out = np.zeros_like(pts)
    with torch.no_grad():
        p1 = torch.tensor(pts[m1], dtype=torch.float32, device=device)
        if len(p1) > 0:
            u_out[m1] = pinn(p1, 0).cpu().numpy()
        p2 = torch.tensor(pts[m2], dtype=torch.float32, device=device)
        if len(p2) > 0:
            u_out[m2] = pinn(p2, 1).cpu().numpy()
        p3 = torch.tensor(pts[m3], dtype=torch.float32, device=device)
        if len(p3) > 0:
            u_out[m3] = pinn(p3, 2).cpu().numpy()
    return u_out

def run():
    print("Running diagnostics...")
    data = np.load("pinn-workflow/fea_solution.npy", allow_pickle=True).item()
    X_fea = data["x"]
    Y_fea = data["y"]
    Z_fea = data["z"]
    U_fea = data["u"]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    pinn = _load_model(device)
    if pinn is None:
        return

    x_axis = X_fea[:, 0, 0]
    y_axis = Y_fea[0, :, 0]
    z_axis = Z_fea[0, 0, :]

    top_k = len(z_axis) - 1
    mid_k = len(z_axis) // 2

    sample_ix = np.linspace(0, len(x_axis) - 1, 5, dtype=int)
    sample_iy = np.linspace(0, len(y_axis) - 1, 5, dtype=int)

    def gather_points(k_idx):
        pts = []
        fea_vals = []
        for ix in sample_ix:
            for iy in sample_iy:
                pts.append([x_axis[ix], y_axis[iy], z_axis[k_idx]])
                fea_vals.append(U_fea[ix, iy, k_idx, 2])
        return np.array(pts), np.array(fea_vals)

    for label, k_idx in [("top", top_k), ("mid", mid_k)]:
        pts, fea_uz = gather_points(k_idx)
        pinn_u = _predict_points(pinn, pts, device)[:, 2]
        abs_err = np.abs(fea_uz - pinn_u)
        print(f"{label} plane u_z diagnostics:")
        print(f"  mean |err|: {abs_err.mean():.6f}")
        print(f"  max  |err|: {abs_err.max():.6f}")
        print(f"  fea min/max: {fea_uz.min():.6f} / {fea_uz.max():.6f}")
        print(f"  pinn min/max: {pinn_u.min():.6f} / {pinn_u.max():.6f}")

    # Traction on top patch (Layer 3)
    n_plot = 40
    x = np.linspace(config.Lx / 3, 2 * config.Lx / 3, n_plot)
    y = np.linspace(config.Ly / 3, 2 * config.Ly / 3, n_plot)
    Xp, Yp = np.meshgrid(x, y)
    Zp = np.ones_like(Xp) * config.H
    pts_patch = np.stack([Xp.ravel(), Yp.ravel(), Zp.ravel()], axis=1)

    pts_t = torch.tensor(pts_patch, dtype=torch.float32, device=device, requires_grad=True)
    u_top = pinn(pts_t, 2)
    grad_u_top = physics.gradient(u_top, pts_t)
    lm3, mu3 = config.Lame_Params[2]
    sig_top = physics.stress(physics.strain(grad_u_top), lm3, mu3)
    traction = sig_top[:, :, 2]
    t_z = traction[:, 2].detach().cpu().numpy().reshape(Xp.shape)

    target = -config.p0
    print("Top patch traction diagnostics:")
    print(f"  target Tz: {target:.6f}")
    print(f"  mean  Tz: {t_z.mean():.6f}")
    print(f"  max   Tz: {t_z.max():.6f}")
    print(f"  min   Tz: {t_z.min():.6f}")
    print(f"  mean |Tz-target|: {np.mean(np.abs(t_z - target)):.6f}")

    plt.figure(figsize=(7, 5))
    c = plt.contourf(Xp, Yp, t_z, levels=50, cmap="coolwarm")
    plt.colorbar(c, label="Traction Tz")
    plt.title("PINN Top Patch Traction Tz")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("traction_top_patch.png")
    print("Saved traction_top_patch.png")

if __name__ == "__main__":
    run()
