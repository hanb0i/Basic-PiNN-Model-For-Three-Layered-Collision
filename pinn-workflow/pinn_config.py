
import torch
import numpy as np

# --- Geometry Dimensions ---
Lx = 1.0
Ly = 1.0
H = 0.1  # Total height
# Layer interfaces (assuming equal thickness for simplicity unless specified)
# z goes from 0 to H.
# Layer 1: 0 to H/3
# Layer 2: H/3 to 2H/3
# Layer 3: 2H/3 to H
Layer_Interfaces = [0.0, H/3, 2*H/3, H]

# --- Material Properties ---
# Young's Modulus (E) and Poisson's Ratio (nu)
# Can be different per layer
E_vals = [1.0, 1.0, 1.0] # Match FEA material
nu_vals = [0.3, 0.3, 0.3]

def get_lame_params(E, nu):
    lm = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lm, mu

Lame_Params = [get_lame_params(e, n) for e, n in zip(E_vals, nu_vals)]

# --- Loading ---
p0 = 0.1 # Load magnitude (Matched to FEA: 0.1)

# --- Training Hyperparameters ---
LEARNING_RATE = 1e-3 # Increased LR for larger load/network
EPOCHS_ADAM = 1000 # Optimized based on convergence analysis
EPOCHS_LBFGS = 2000 
#Plot Physical Residuals Every N Epochs every 100 epochs. 
WEIGHTS = {
    'pde': 1.0,     # Normalized
    'bc': 10.0,    # Reduced to allow deformation
    'load': 500.0,   # Boosted to force deformation (vs zero solution)
    'interface_u': 100.0,
    'interface_stress': 10.0 
}
# Sampling
N_INTERIOR = 8000 # Increased sampling
N_BOUNDARY = 2000

# Model Architecture
HIDDEN_LAYERS = 5
HIDDEN_UNITS = 128

# Fourier Features
FOURIER_DIM = 64 
FOURIER_SCALE = 2.0 
OUTPUT_SCALE = 10.0 # Restored scaling to enable bending mode capture
