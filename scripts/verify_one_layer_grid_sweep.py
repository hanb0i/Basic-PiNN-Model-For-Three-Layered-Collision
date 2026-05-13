#!/usr/bin/env python3
"""
Reproducible random verification for the one-layer PINN.

This script evaluates the one-layer PINN against FEM on random parameter
combinations sampled from the configured E and thickness bounds. It reports
top-surface z-displacement MAE% for every case and summarizes mean / worst /
best.

Usage:
    cd /path/to/repo
    python scripts/verify_one_layer_grid_sweep.py

Environment variables (all optional):
    PINN_MODEL_PATH      – path to the one-layer PINN checkpoint
                           (default: one-layer-workflow/pinn_model.pth)
    PINN_DEVICE          – torch device string, e.g. "cpu", "cuda", "mps"
                           (default: auto-detect)
    PINN_EVAL_E_VALUES   – comma-separated E values used as random sampling bounds
                           (default: from one-layer-workflow/pinn_config.py E_RANGE)
    PINN_EVAL_T_VALUES   – comma-separated thickness values used as random sampling bounds
                           (default: from one-layer-workflow/pinn_config.py DATA_THICKNESS_VALUES)
    PINN_EVAL_N_CASES    – number of random cases to evaluate
                           (default: old grid size from configured value counts)
    PINN_EVAL_SEED       – random seed for reproducible verification cases
                           (default: 20260428)
    PINN_CALIBRATION_JSON – optional compliance calibration JSON. If present,
                           tuned_params override the config scaling.
    PINN_EVAL_NE_X       – FEM mesh elements in x (default: 16)
    PINN_EVAL_NE_Y       – FEM mesh elements in y (default: 16)
    PINN_EVAL_NE_Z       – FEM mesh elements in z (default: 8)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
ONE_LAYER_DIR = REPO_ROOT / "one-layer-workflow"
FEA_DIR = REPO_ROOT / "fea-workflow" / "solver"

for _path in (ONE_LAYER_DIR, FEA_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import fem_solver  # noqa: E402
import model  # noqa: E402
import pinn_config as config  # noqa: E402

_CALIBRATION_CACHE: dict[str, dict | None] = {}

# ---------------------------------------------------------------------------
# Helpers (mirrored from one_layer_experiment_utils.py to keep script standalone)
# ---------------------------------------------------------------------------


def _select_device() -> torch.device:
    requested = os.getenv("PINN_DEVICE")
    if requested:
        return torch.device(requested)
    if os.getenv("PINN_FORCE_CPU", "0") == "1":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _adapt_state_dict(state: dict, target: dict) -> dict:
    state = dict(state)
    w_key = "layer.net.0.weight"
    if w_key not in state or w_key not in target:
        return state
    src_w = state[w_key]
    tgt_w = target[w_key]
    if src_w.shape == tgt_w.shape or src_w.shape[0] != tgt_w.shape[0]:
        return state
    if src_w.shape[1] == 8 and tgt_w.shape[1] == 11:
        adapted = torch.zeros_like(tgt_w)
        adapted[:, 0:5] = src_w[:, 0:5]
        adapted[:, 8:11] = src_w[:, 5:8]
        state[w_key] = adapted
    elif src_w.shape[1] == 10 and tgt_w.shape[1] == 11:
        adapted = torch.zeros_like(tgt_w)
        adapted[:, 0:7] = src_w[:, 0:7]
        adapted[:, 8:11] = src_w[:, 7:10]
        state[w_key] = adapted
    return state


def _load_pinn(device: torch.device) -> torch.nn.Module:
    model_path = Path(os.getenv("PINN_MODEL_PATH") or ONE_LAYER_DIR / "pinn_model.pth")
    if not model_path.exists():
        raise FileNotFoundError(f"PINN checkpoint not found: {model_path}")
    pinn = model.MultiLayerPINN().to(device)
    state = torch.load(str(model_path), map_location=device, weights_only=True)
    state = _adapt_state_dict(state, pinn.state_dict())
    pinn.load_state_dict(state, strict=False)
    pinn.eval()
    print(f"Loaded PINN checkpoint: {model_path}")
    return pinn


def _load_calibration() -> dict | None:
    path = os.getenv("PINN_CALIBRATION_JSON")
    if not path:
        return None
    if path not in _CALIBRATION_CACHE:
        cal_path = Path(path)
        if not cal_path.is_absolute():
            cal_path = REPO_ROOT / cal_path
        _CALIBRATION_CACHE[path] = json.loads(cal_path.read_text()) if cal_path.exists() else None
    return _CALIBRATION_CACHE[path]


def _compliance_params() -> tuple[float, float, float]:
    scale = float(getattr(config, "DISPLACEMENT_COMPLIANCE_SCALE", 1.0))
    e_pow = float(getattr(config, "E_COMPLIANCE_POWER", 1.0))
    alpha = float(getattr(config, "THICKNESS_COMPLIANCE_ALPHA", 0.0))
    cal = _load_calibration()
    tuned = cal.get("tuned_params") if cal else None
    if tuned:
        scale = float(tuned.get("PINN_DISPLACEMENT_COMPLIANCE_SCALE", scale))
        e_pow = float(tuned.get("PINN_E_COMPLIANCE_POWER", e_pow))
        alpha = float(tuned.get("PINN_THICKNESS_COMPLIANCE_ALPHA", alpha))
    return scale, e_pow, alpha


def _u_from_v(v: np.ndarray, pts: np.ndarray) -> np.ndarray:
    e_vals = pts[:, 3:4]
    t_vals = pts[:, 4:5]
    scale, e_pow, alpha = _compliance_params()
    h_ref = float(getattr(config, "H", 1.0))
    return scale * v / (e_vals**e_pow) * (h_ref / np.clip(t_vals, 1e-8, None)) ** alpha


def _make_points(x: np.ndarray, y: np.ndarray, z: np.ndarray, E: float, thickness: float) -> np.ndarray:
    r_ref = float(getattr(config, "RESTITUTION_REF", 0.5))
    mu_ref = float(getattr(config, "FRICTION_REF", 0.3))
    v0_ref = float(getattr(config, "IMPACT_VELOCITY_REF", 1.0))
    return np.stack(
        [
            x,
            y,
            z,
            np.full_like(x, float(E), dtype=float),
            np.full_like(x, float(thickness), dtype=float),
            np.full_like(x, float(r_ref), dtype=float),
            np.full_like(x, float(mu_ref), dtype=float),
            np.full_like(x, float(v0_ref), dtype=float),
        ],
        axis=1,
    )


def _predict_displacement(
    pinn: torch.nn.Module, device: torch.device, pts: np.ndarray, batch_size: int = 32768
) -> np.ndarray:
    out = []
    with torch.no_grad():
        for start in range(0, len(pts), batch_size):
            batch_pts = pts[start : start + batch_size]
            v = pinn(torch.tensor(batch_pts, dtype=torch.float32, device=device)).detach().cpu().numpy()
            out.append(_u_from_v(v, batch_pts))
    return np.concatenate(out, axis=0)


def _fem_cfg(E: float, thickness: float, ne_x: int, ne_y: int, ne_z: int) -> dict:
    return {
        "geometry": {
            "Lx": float(config.Lx),
            "Ly": float(config.Ly),
            "H": float(thickness),
            "ne_x": int(ne_x),
            "ne_y": int(ne_y),
            "ne_z": int(ne_z),
        },
        "material": {"E": float(E), "nu": float(config.nu_vals[0])},
        "load_patch": {
            "pressure": float(config.p0),
            "x_start": float(config.LOAD_PATCH_X[0]) / float(config.Lx),
            "x_end": float(config.LOAD_PATCH_X[1]) / float(config.Lx),
            "y_start": float(config.LOAD_PATCH_Y[0]) / float(config.Ly),
            "y_end": float(config.LOAD_PATCH_Y[1]) / float(config.Ly),
        },
    }


def _mae_pct(pred: np.ndarray, ref: np.ndarray) -> float:
    denom = float(np.max(np.abs(ref)))
    return 100.0 * float(np.mean(np.abs(pred - ref))) / denom if denom > 0 else 0.0


def _relative_l2_pct(pred: np.ndarray, ref: np.ndarray) -> float:
    denom = float(np.sqrt(np.sum(np.asarray(ref, dtype=float) ** 2)))
    return 100.0 * float(np.sqrt(np.sum((np.asarray(pred) - np.asarray(ref)) ** 2))) / denom if denom > 0 else 0.0


def _bias_pct(pred: np.ndarray, ref: np.ndarray) -> float:
    denom = float(np.max(np.abs(ref)))
    return 100.0 * float(np.mean(pred - ref)) / denom if denom > 0 else 0.0


def _peak_uz_metrics(pred: np.ndarray, ref: np.ndarray) -> dict:
    peak_ref = float(np.min(ref))
    peak_pred = float(np.min(pred))
    denom = abs(peak_ref)
    return {
        "peak_fem_uz": peak_ref,
        "peak_pinn_uz": peak_pred,
        "peak_uz_error_pct": 100.0 * abs(peak_pred - peak_ref) / denom if denom > 0 else 0.0,
    }


def _float_list_env(name: str) -> list[float] | None:
    value = os.getenv(name)
    if not value:
        return None
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def _bounds(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("Cannot derive bounds from an empty value list")
    return float(min(values)), float(max(values))


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------


def main() -> None:
    device = _select_device()
    print(f"Using device: {device}")

    # Mesh resolution
    ne_x = int(os.getenv("PINN_EVAL_NE_X", "16"))
    ne_y = int(os.getenv("PINN_EVAL_NE_Y", "16"))
    ne_z = int(os.getenv("PINN_EVAL_NE_Z", "8"))
    print(f"FEM mesh: {ne_x} × {ne_y} × {ne_z}")

    # Random parameter cases. The configured/evaluated values are used as bounds
    # instead of fixed threshold grid points.
    e_values_for_bounds = _float_list_env("PINN_EVAL_E_VALUES")
    if e_values_for_bounds is None:
        e_values_for_bounds = [float(v) for v in getattr(config, "E_RANGE", getattr(config, "DATA_E_VALUES", [1.0, 10.0]))]

    t_values_for_bounds = _float_list_env("PINN_EVAL_T_VALUES")
    if t_values_for_bounds is None:
        t_values_for_bounds = [float(v) for v in getattr(config, "DATA_THICKNESS_VALUES", [0.05, 0.10, 0.15])]

    e_bounds = _bounds(e_values_for_bounds)
    t_bounds = _bounds(t_values_for_bounds)
    default_n_cases = max(1, len(e_values_for_bounds) * len(t_values_for_bounds))
    n_cases = int(os.getenv("PINN_EVAL_N_CASES", str(default_n_cases)))
    seed = int(os.getenv("PINN_EVAL_SEED", "20260428"))
    rng = np.random.default_rng(seed)

    random_cases = [
        (
            float(rng.uniform(e_bounds[0], e_bounds[1])),
            float(rng.uniform(t_bounds[0], t_bounds[1])),
        )
        for _ in range(n_cases)
    ]

    print(f"E bounds: {e_bounds}")
    print(f"Thickness bounds: {t_bounds}")
    print(f"Random cases: {n_cases} (seed={seed})")

    pinn = _load_pinn(device)
    scale, e_pow, alpha = _compliance_params()
    calibration_json = os.getenv("PINN_CALIBRATION_JSON")
    if calibration_json:
        print(f"Calibration JSON: {calibration_json}")
    print(f"Compliance scaling: scale={scale:.6g}, E power={e_pow:.6g}, thickness alpha={alpha:.6g}")

    results = []
    for idx, (E, thickness) in enumerate(random_cases):
        # FEM solve
        x_nodes, y_nodes, z_nodes, u_fem = fem_solver.solve_fem(_fem_cfg(E, thickness, ne_x, ne_y, ne_z))
        x_nodes = np.asarray(x_nodes)
        y_nodes = np.asarray(y_nodes)
        z_nodes = np.asarray(z_nodes)
        u_fem = np.asarray(u_fem)

        # PINN predict (full volume)
        xg, yg, zg = np.meshgrid(x_nodes, y_nodes, z_nodes, indexing="ij")
        pts = _make_points(xg.ravel(), yg.ravel(), zg.ravel(), E, thickness)
        u_pinn = _predict_displacement(pinn, device, pts).reshape(u_fem.shape)

        # Top-surface z-displacement MAE%
        top_pred = u_pinn[:, :, -1, 2]
        top_ref = u_fem[:, :, -1, 2]
        mae = _mae_pct(top_pred, top_ref)
        volume_mae = _mae_pct(u_pinn, u_fem)

        case_id = f"random_{idx:03d}"
        results.append({
            "case_id": case_id,
            "E": float(E),
            "thickness": float(thickness),
            "top_uz_mae_pct": float(mae),
            "top_uz_relative_l2_pct": float(_relative_l2_pct(top_pred, top_ref)),
            "top_uz_bias_pct": float(_bias_pct(top_pred, top_ref)),
            "volume_mae_pct": float(volume_mae),
            "volume_relative_l2_pct": float(_relative_l2_pct(u_pinn, u_fem)),
            **_peak_uz_metrics(top_pred, top_ref),
        })
        print(f"  {case_id}: E={E:.6g}, t={thickness:.6g}, top MAE = {mae:.2f}%")

    maes = [r["top_uz_mae_pct"] for r in results]
    l2s = [r["top_uz_relative_l2_pct"] for r in results]
    biases = [r["top_uz_bias_pct"] for r in results]
    volume_maes = [r["volume_mae_pct"] for r in results]
    peak_errors = [r["peak_uz_error_pct"] for r in results]
    mean_mae = float(np.mean(maes))
    worst_mae = float(np.max(maes))
    best_mae = float(np.min(maes))
    worst_idx = int(np.argmax(maes))
    worst_case = results[worst_idx]

    summary = {
        "model": "one-layer",
        "sampling_protocol": "random_parameter_verification",
        "seed": seed,
        "calibration_json": calibration_json,
        "compliance_params": {
            "PINN_DISPLACEMENT_COMPLIANCE_SCALE": scale,
            "PINN_E_COMPLIANCE_POWER": e_pow,
            "PINN_THICKNESS_COMPLIANCE_ALPHA": alpha,
        },
        "model_path": str(os.getenv("PINN_MODEL_PATH") or ONE_LAYER_DIR / "pinn_model.pth"),
        "device": str(device),
        "mesh": {"ne_x": ne_x, "ne_y": ne_y, "ne_z": ne_z},
        "e_bounds": list(e_bounds),
        "t_bounds": list(t_bounds),
        "n_cases": len(results),
        "mean_top_uz_mae_pct": mean_mae,
        "worst_top_uz_mae_pct": worst_mae,
        "best_top_uz_mae_pct": best_mae,
        "mean_top_uz_relative_l2_pct": float(np.mean(l2s)),
        "worst_top_uz_relative_l2_pct": float(np.max(l2s)),
        "mean_top_uz_bias_pct": float(np.mean(biases)),
        "mean_abs_top_uz_bias_pct": float(np.mean(np.abs(biases))),
        "mean_volume_mae_pct": float(np.mean(volume_maes)),
        "worst_volume_mae_pct": float(np.max(volume_maes)),
        "mean_peak_uz_error_pct": float(np.mean(peak_errors)),
        "worst_peak_uz_error_pct": float(np.max(peak_errors)),
        "worst_case": worst_case,
        "cases": results,
    }

    out_dir = REPO_ROOT / "graphs" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "one_layer_grid_sweep_verification.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print()
    print("=" * 60)
    print("ONE-LAYER RANDOM VERIFICATION")
    print("=" * 60)
    print(f"  Mean top MAE: {mean_mae:.2f}%")
    print(f"  Worst top MAE: {worst_mae:.2f}%")
    print(f"  Best top MAE: {best_mae:.2f}%")
    print(f"  Mean top relative L2: {summary['mean_top_uz_relative_l2_pct']:.2f}%")
    print(f"  Mean top bias: {summary['mean_top_uz_bias_pct']:.2f}%")
    print(f"  Mean volume MAE: {summary['mean_volume_mae_pct']:.2f}%")
    print(f"  Mean peak u_z error: {summary['mean_peak_uz_error_pct']:.2f}%")
    print(f"  Results saved to: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
