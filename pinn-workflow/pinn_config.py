
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
E_vals = [1.0, 1.0, 1.0] # Normalized
nu_vals = [0.3, 0.3, 0.3]

def get_lame_params(E, nu):
    lm = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lm, mu

Lame_Params = [get_lame_params(e, n) for e, n in zip(E_vals, nu_vals)]

# --- Loading ---
p0 = 1.0 # Load magnitude

# --- Training Hyperparameters ---
LEARNING_RATE = 1e-3
EPOCHS_ADAM = 1000 # Longer Adam to improve load matching
EPOCHS_LBFGS = 2000 # Increased from 500. Resampling here. Should help convergence. 
#Plot Physical Residuals Every N Epochs every 100 epochs. 
WEIGHTS = {
    'pde': 20.0,    # Stronger physics consistency
    'bc': 2.0,      # Tighten clamped boundaries
    'load': 2000.0, # Increase load emphasis for deeper deflection
    'interface_u': 150.0 
}
# Sampling
N_INTERIOR = 4000 # Per layer
N_BOUNDARY = 1000  # Per face type

# Fourier Features
FOURIER_DIM = 64 # Number of Fourier frequencies
FOURIER_SCALE = 1.0 # Standard deviation for frequency sampling
