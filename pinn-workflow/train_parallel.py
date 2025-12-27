
import torch
import torch.optim as optim
import numpy as np
import time
import os

# IMPORT PARALLEL CONFIG
import pinn_config_parallel as config
import data
import model
import physics
import matplotlib.pyplot as plt

def train():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Initialize Model
    # Note: Model internally uses 'pinn_config', so we need to validly patch it or ensure model.py is flexible.
    # Hack: Inject parallel config into model.config if necessary, but model.py imports pinn_config.
    # Better: We will rely on model.config (which imports local pinn_config).
    # THIS IS A PROBLEM. model.py imports 'pinn_config'.
    # To run strictly parallel, we need to overwrite the module or use 'pinn_config' file content.
    
    # FIX: We will rely on the fact that running this script assumes 'pinn_config_parallel.py' is what we want,
    # BUT model.py imports 'pinn_config'.
    # We must patch sys.modules or modify model.py.
    # Easiest way for a separate device: Rename 'pinn_config_parallel.py' to 'pinn_config.py' on that device.
    
    print("WARNING: This script imports 'pinn_config_parallel' for training parameters,")
    print("BUT 'model.py' and 'physics.py' still import 'pinn_config'.")
    print("For a truly independent run on another device, rename 'pinn_config_parallel.py' to 'pinn_config.py'.")
    
    # However, for this runtime, let's try to patch it dynamically.
    import sys
    sys.modules['pinn_config'] = config
    
    pinn = model.MultiLayerPINN().to(device)
    
    # Initialize Optimizers
    optimizer_adam = optim.Adam(pinn.parameters(), lr=config.LEARNING_RATE)
    
    # Data Container
    training_data = data.get_data()
    
    # History
    loss_history = []
    
    print("Starting Adam Training (Parallel Config)...")
    start_time = time.time()
    last_time = start_time
    
    try:
        for epoch in range(config.EPOCHS_ADAM):
            optimizer_adam.zero_grad()
            
            if epoch % 1000 == 0 and epoch > 0:
                training_data = data.get_data()
                
            loss_val, losses = physics.compute_loss(pinn, training_data, device)
            loss_val.backward()
            optimizer_adam.step()
            
            loss_history.append(loss_val.item())
            
            if epoch % 100 == 0:
                current_time = time.time()
                step_duration = current_time - last_time
                last_time = current_time
                print(f"Epoch {epoch}: Total Loss: {loss_val.item():.6f} | "
                      f"PDE: {losses['pde']:.6f} | BC: {losses['bc_sides']:.6f} | "
                      f"Load: {losses['load']:.6f} | Interface U: {losses['interface']:.6f} | Stress: {losses['interface_stress']:.6f} | "
                      f"Time: {step_duration:.4f}s")
                
        print(f"Adam Training Complete. Total Time: {time.time() - start_time:.2f}s")
        
        # L-BFGS Training
        print("Starting L-BFGS Training...")
        optimizer_lbfgs = optim.LBFGS(pinn.parameters(), 
                                      max_iter=200, 
                                      history_size=50, 
                                      line_search_fn="strong_wolfe")
        
        def closure():
            optimizer_lbfgs.zero_grad()
            loss_val, _ = physics.compute_loss(pinn, training_data, device)
            loss_val.backward()
            return loss_val
            
        num_lbfgs_steps = config.EPOCHS_LBFGS // 20
        print(f"Running {num_lbfgs_steps} L-BFGS outer steps.")
        
        for i in range(num_lbfgs_steps): 
            step_start = time.time()
            loss_val = optimizer_lbfgs.step(closure)
            step_end = time.time()
            loss_history.append(loss_val.item())
            
            print(f"L-BFGS Step {i}: Loss: {loss_val.item():.6f} | Time: {step_end - step_start:.4f}s")
            torch.save(pinn.state_dict(), "pinn_model_parallel.pth")
            np.save("loss_history_parallel.npy", np.array(loss_history))
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current state...")
        
    # Save Model
    torch.save(pinn.state_dict(), "pinn_model_parallel.pth")
    print(f"Model saved: pinn_model_parallel.pth")
    np.save("loss_history_parallel.npy", np.array(loss_history))
    print("Model saved.")
    return pinn

if __name__ == "__main__":
    train()
