
import sys
import os
import torch
import torch.optim as optim
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from geometry.sampling import Sampler
from solver.models import MultiLayerPINN
from solver.pinn_physics import compute_loss
from solver.fem_solver import solve_fem
from postprocessing.visualization import plot_pinn_results, plot_comparison

def get_config():
    # Hardcoded to avoid PyYAML dependency
    return {
        'geometry': {'Lx': 1.0, 'Ly': 1.0, 'H': 0.1},
        'load_patch': {
            'x_start': 0.33, 'x_end': 0.67,
            'y_start': 0.33, 'y_end': 0.67,
            'pressure': 1.0
        },
        'material': {'E': 1.0, 'nu': 0.3},
        'pinn': {
            'training': {'epochs_adam': 200, 'epochs_lbfgs': 50, 'learning_rate': 1e-3},
            'weights': {'pde': 1.0, 'bc': 100.0, 'load': 100.0, 'interface': 100.0},
            'sampling': {'n_interior': 1000, 'n_boundary': 200},
            'layers': {'hidden_depth': 6, 'hidden_width': 64}
        }
    }

def train_pinn(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training PINN on {device}...")
    
    sampler = Sampler(cfg)
    data = sampler.get_data()
    
    model = MultiLayerPINN(cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(cfg['pinn']['training']['learning_rate']))
    
    epochs = cfg['pinn']['training']['epochs_adam']
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss, losses = compute_loss(model, data, device, cfg)
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.5f}")
            
    if cfg['pinn']['training']['epochs_lbfgs'] > 0:
        print("L-BFGS...")
        opt_lbfgs = optim.LBFGS(model.parameters(), max_iter=20, line_search_fn="strong_wolfe")
        def closure():
            opt_lbfgs.zero_grad()
            loss, _ = compute_loss(model, data, device, cfg)
            loss.backward()
            return loss
        
        for i in range(cfg['pinn']['training']['epochs_lbfgs'] // 20):
            try:
                l = opt_lbfgs.step(closure)
                print(f"LBFGS Step {i}: {l.item():.5f}")
            except:
                pass
            
    return model

def main():
    cfg = get_config()
    pinn_model = train_pinn(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Running FEA Benchmark...")
    x, y, z, u_fea = solve_fem(cfg)
    
    print("Plotting results...")
    plot_pinn_results(pinn_model, cfg, device, save_path=os.path.dirname(__file__))
    plot_comparison(u_fea, (x, y, z), pinn_model, cfg, device, save_path=os.path.dirname(__file__))
    print("Done.")

if __name__ == "__main__":
    main()
