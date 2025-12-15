
import torch
import torch.optim as optim
import numpy as np
import time
import os

import config
import data
import model
import physics
import matplotlib.pyplot as plt

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize Model
    pinn = model.MultiLayerPINN().to(device)
    
    # Initialize Optimizers
    optimizer_adam = optim.Adam(pinn.parameters(), lr=config.LEARNING_RATE)
    
    # Data Container
    training_data = data.get_data()
    
    # History
    loss_history = []
    
    print("Starting Adam Training...")
    start_time = time.time()
    
    for epoch in range(config.EPOCHS_ADAM):
        optimizer_adam.zero_grad()
        
        # Periodic data refresh (optional, computationally expensive to re-sample every Step)
        if epoch % 1000 == 0 and epoch > 0:
            training_data = data.get_data()
            
        loss_val, losses = physics.compute_loss(pinn, training_data, device)
        loss_val.backward()
        optimizer_adam.step()
        
        loss_history.append(loss_val.item())
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Total Loss: {loss_val.item():.6f} | "
                  f"PDE: {losses['pde']:.6f} | BC: {losses['bc_sides']:.6f} | "
                  f"Load: {losses['load']:.6f} | Interface: {losses['interface']:.6f}")
            
    print(f"Adam Training Complete. Time: {time.time() - start_time:.2f}s")
    
    # L-BFGS Training
    print("Starting L-BFGS Training...")
    optimizer_lbfgs = optim.LBFGS(pinn.parameters(), 
                                  max_iter=20, 
                                  history_size=50, 
                                  line_search_fn="strong_wolfe")
    
    def closure():
        optimizer_lbfgs.zero_grad()
        loss_val, _ = physics.compute_loss(pinn, training_data, device)
        loss_val.backward()
        return loss_val
        
    for i in range(config.EPOCHS_LBFGS // 20): # LBFGS step takes multiple evals
        loss_val = optimizer_lbfgs.step(closure)
        loss_history.append(loss_val.item())
        if i % 10 == 0:
            print(f"L-BFGS Step {i}: Loss: {loss_val.item():.6f}")
            
    # Save Model
    torch.save(pinn.state_dict(), "pinn_model.pth")
    np.save("loss_history.npy", np.array(loss_history))
    print("Model saved.")

if __name__ == "__main__":
    train()
