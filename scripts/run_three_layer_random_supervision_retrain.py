"""Retrain the three-layer PINN with random FEM supervision cases.

This runner targets the random-interior generalization protocol directly by
generating FEM supervision on random parameter cases, oversampling the top
surface, fine-tuning from the current checkpoint, and evaluating on the same
random generalization script used for reporting.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "graphs" / "data"
RUNS_DIR = DATA_DIR / "random_supervision_runs"
DEFAULT_WARM_START = REPO_ROOT / "pinn-workflow" / "pinn_model.pth"


def _python() -> str:
    return sys.executable or "python3"


def _run(cmd: list[str], env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running: {' '.join(cmd)}")
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env={**os.environ, **env},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit {return_code}. See {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="three_layer_random_supervision")
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--lbfgs-steps", type=int, default=100)
    parser.add_argument("--device", default=None)
    parser.add_argument("--warm-start", default=str(DEFAULT_WARM_START))
    parser.add_argument("--n-data-points", type=int, default=36000)
    parser.add_argument("--n-supervision-cases", type=int, default=64)
    parser.add_argument("--supervision-seed", type=int, default=20260429)
    parser.add_argument("--eval-seed", type=int, default=20260428)
    parser.add_argument("--eval-cases", type=int, default=8)
    parser.add_argument("--top-fraction", type=float, default=0.60)
    parser.add_argument("--interface-fraction", type=float, default=0.30)
    parser.add_argument("--interface-points", type=int, default=16000)
    parser.add_argument("--data-weight", type=float, default=400.0)
    parser.add_argument("--pde-weight", type=float, default=10.0)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    run_dir = RUNS_DIR / args.run_name
    log_dir = run_dir / "logs"
    ckpt = run_dir / "pinn_model.pth"
    eval_csv = run_dir / "random_interior_generalization.csv"
    eval_summary = run_dir / "random_interior_generalization_summary.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    env = {
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(REPO_ROOT / ".mplconfig"),
        "PYTHONPYCACHEPREFIX": str(REPO_ROOT / ".pycache"),
        "PYTHONUNBUFFERED": "1",
        "PINN_OUT_DIR": str(run_dir),
        "PINN_MODEL_PATH": str(ckpt),
        "PINN_WARM_START": "1",
        "PINN_WARM_START_PATH": str(Path(args.warm_start).resolve()),
        "PINN_SUPERVISION_CACHE": "1",
        "PINN_REGEN_SUPERVISION": "0",
        "PINN_RANDOM_SUPERVISION_CASES": str(args.n_supervision_cases),
        "PINN_RANDOM_SUPERVISION_SEED": str(args.supervision_seed),
        "PINN_SUPERVISION_TOP_FRACTION": str(args.top_fraction),
        "PINN_SUPERVISION_INTERFACE_FRACTION": str(args.interface_fraction),
        "PINN_N_DATA_POINTS": str(args.n_data_points),
        "PINN_USE_SUPERVISION_DATA": "1",
        "PINN_W_DATA": str(args.data_weight),
        "PINN_W_PDE": str(args.pde_weight),
        "PINN_W_BC": "0.7",
        "PINN_W_LOAD": "5.0",
        "PINN_E_COMPLIANCE_POWER": "0.95",
        "PINN_THICKNESS_COMPLIANCE_ALPHA": "3",
        "PINN_DISPLACEMENT_COMPLIANCE_SCALE": "1",
        "PINN_PDE_DECOMPOSE_BY_LAYER": "1",
        "PINN_W_INTERFACE_U": "300",
        "PINN_N_INTERFACE": str(args.interface_points),
        "PINN_INTERFACE_SAMPLE_FRACTION": "0.75",
        "PINN_SUPERVISION_THICKNESS_POWER": "3.0",
        "PINN_ADAPTIVE_RESAMPLE_EVERY": "0",
        "PINN_FEM_NE_X": "16",
        "PINN_FEM_NE_Y": "16",
        "PINN_FEM_NE_Z": "8",
        "PINN_EPOCHS_ADAM": str(args.epochs),
        "PINN_EPOCHS_SOAP": str(args.epochs),
        "PINN_EPOCHS_LBFGS": str(args.lbfgs_steps),
    }
    if args.device:
        env["PINN_DEVICE"] = args.device

    manifest = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt),
        "warm_start": env["PINN_WARM_START_PATH"],
        "reason": "Fine-tune three-layer PINN with random FEM supervision and top-surface oversampling.",
        "env": env,
        "evaluation": {
            "seed": args.eval_seed,
            "n_cases": args.eval_cases,
            "mesh": {"ne_x": 16, "ne_y": 16, "ne_z": 8},
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if args.skip_train:
        if not ckpt.exists():
            raise FileNotFoundError(f"--skip-train requested but checkpoint is missing: {ckpt}")
        print(f"Skipping training; using existing checkpoint {ckpt}")
    else:
        _run([_python(), "three-layer-workflow/train.py"], env, log_dir / "train.log")

    eval_env = dict(env)
    eval_env["PINN_REGEN_SUPERVISION"] = "0"
    _run(
        [
            _python(),
            "scripts/run_random_interior_generalization.py",
            "--model-path",
            str(ckpt),
            "--n-cases",
            str(args.eval_cases),
            "--seed",
            str(args.eval_seed),
            "--ne-x",
            "16",
            "--ne-y",
            "16",
            "--ne-z",
            "8",
            "--out-csv",
            str(eval_csv),
            "--out-summary",
            str(eval_summary),
        ],
        eval_env,
        log_dir / "eval.log",
    )
    print(f"Wrote {eval_summary}")


if __name__ == "__main__":
    main()
