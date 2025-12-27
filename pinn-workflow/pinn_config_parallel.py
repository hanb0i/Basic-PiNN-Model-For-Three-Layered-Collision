
import torch
import numpy as np

# --- Geometry Dimensions ---
Lx = 1.0
Ly = 1.0
H = 0.1  # Total height
Layer_Interfaces = [0.0, H/3, 2*H/3, H]

# --- Material Properties ---
E_vals = [1.0, 1.0, 1.0] 
nu_vals = [0.3, 0.3, 0.3]

def get_lame_params(E, nu):
    lm = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lm, mu

Lame_Params = [get_lame_params(e, n) for e, n in zip(E_vals, nu_vals)]

# --- Loading ---
p0 = 0.1 # Load magnitude

# --- Training Hyperparameters ---
LEARNING_RATE = 1e-3 
EPOCHS_ADAM = 2000 # Increased for parallel run to ensure convergence
EPOCHS_LBFGS = 2000 

WEIGHTS = {
    'pde': 1.0,     
    'bc': 10.0,     # Stronger BC than current run (10.0 vs 1.0)
    'load': 500.0,  # Much stronger load forcing (500.0 vs 100.0)
    'interface_u': 100.0,
    'interface_stress': 10.0 
}

# Sampling
N_INTERIOR = 8000 
N_BOUNDARY = 2000

# Model Architecture
HIDDEN_LAYERS = 5
HIDDEN_UNITS = 128

# Fourier Features
FOURIER_DIM = 64 
FOURIER_SCALE = 2.0 
OUTPUT_SCALE = 1.0 # Standard Scaling (vs 10.0 in main run). Testing if Weights can solve it without scaling hack.
