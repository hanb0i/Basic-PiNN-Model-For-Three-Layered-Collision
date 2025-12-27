
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
<<<<<<< HEAD
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

# Fourier Features
FOURIER_DIM = 64 
FOURIER_SCALE = 2.0 
OUTPUT_SCALE = 1.0 # Normalized scale sufficient for p0=0.1 (disp ~0.6)
=======
p0 = 1.0 # Match FEA load magnitude

# --- Training Hyperparameters ---
LEARNING_RATE = 1e-3
EPOCHS_ADAM = 500 # Match FEA workflow defaults
EPOCHS_LBFGS = 100 # Match FEA workflow defaults
#Plot Physical Residuals Every N Epochs every 100 epochs. 
WEIGHTS = {
    'pde': 0.1,
    'bc': 100.0,
    'load': 1000.0,
    'interface_u': 100.0
}
# Sampling
N_INTERIOR = 4000 # Per layer
N_BOUNDARY = 2000  # Per face type

# Model size
HIDDEN_LAYERS = 6
HIDDEN_UNITS = 64

# Fourier Features
FOURIER_DIM = 0 # Disable Fourier features for smoother fields
FOURIER_SCALE = 2.0 # Standard deviation for frequency sampling
>>>>>>> 45a9ac0bcd22395799ffb9e1c4bbdc8060943c4d
