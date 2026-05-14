#!/usr/bin/env python3
"""Re-evaluate the 5 Table IX cases for both 1L and 3L models with the current checkpoints."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "fea-workflow" / "solver"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# ── One-layer ─────────────────────────────────────────────────────────────────
import one_layer_experiment_utils as u1
from one_layer_experiment_utils import OneLayerCase

device1 = u1.select_device()
pinn1, _ = u1.load_pinn(device1)

ONE_LAYER_CASES = [
    OneLayerCase("I",   4.37, 0.096),
    OneLayerCase("II",  7.59, 0.068),
    OneLayerCase("III", 2.40, 0.032),
    OneLayerCase("IV",  1.52, 0.089),
    OneLayerCase("V",   6.41, 0.077),
]

print("=" * 55)
print("(a) One-Layer Model")
print(f"{'Trial':<6} {'E':>6} {'t':>7}  {'Vol MAE (%)':>12}")
print("-" * 35)
for case in ONE_LAYER_CASES:
    res = u1.evaluate_case_grid(pinn1, device1, case, ne_x=16, ne_y=16, ne_z=8)
    print(f"  {case.case_id:<4} {case.E:>6.2f} {case.thickness:>7.3f}  {res['volume_mae_pct']:>12.2f}")

# ── Three-layer ───────────────────────────────────────────────────────────────
# Clean module namespace before importing 3L utils
for mod in ["model", "pinn_config", "data", "physics", "soap"]:
    sys.modules.pop(mod, None)
ol_str = str(REPO_ROOT / "one-layer-workflow")
if ol_str in sys.path:
    sys.path.remove(ol_str)

import three_layer_experiment_utils as u3
from three_layer_experiment_utils import ThreeLayerCase

device3 = u3.select_device()
pinn3, _ = u3.load_pinn(device3)

THREE_LAYER_CASES = [
    ThreeLayerCase("I",   4.37, 9.56, 7.59, 0.068, 0.032, 0.032),
    ThreeLayerCase("II",  1.52, 8.80, 6.41, 0.077, 0.022, 0.098),
    ThreeLayerCase("III", 8.49, 2.91, 2.64, 0.035, 0.044, 0.062),
    ThreeLayerCase("IV",  4.89, 3.62, 6.51, 0.031, 0.043, 0.049),
    ThreeLayerCase("V",   5.10, 8.07, 2.80, 0.061, 0.067, 0.024),
]

print()
print("=" * 70)
print("(b) Three-Layer Model")
print(f"{'Trial':<6} {'E':>25} {'t':>20}  {'Vol MAE (%)':>12}")
print("-" * 68)
for case in THREE_LAYER_CASES:
    res = u3.evaluate_case_grid(pinn3, device3, case, ne_x=16, ne_y=16, ne_z=8)
    e_str = f"[{case.e1:.2f},{case.e2:.2f},{case.e3:.2f}]"
    t_str = f"[{case.t1:.3f},{case.t2:.3f},{case.t3:.3f}]"
    print(f"  {case.case_id:<4} {e_str:>25} {t_str:>20}  {res['volume_mae_pct']:>12.2f}")
